"""Fail-closed recovery gate over independently rendered motion truth."""

from __future__ import annotations

import json
import math
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from aoa_editing.analysis.motion import MotionAnalyzer
from aoa_editing.domain.models import (
    CheckResult,
    MotionGroundTruthFixture,
    MotionRecoveryGateReport,
    MotionRecoveryReport,
    Provenance,
)
from aoa_editing.evals.motion_fixtures import create_ground_truth_corpus
from aoa_editing.infrastructure.media import sha256_file

REQUIRED_CORPUS_TAGS = {
    "linear",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "cubic_bezier",
    "hermite",
    "overshoot",
    "settle",
    "scale_rotation",
    "curved_translation",
    "moving_pivot",
    "coupled_scale_rotation",
    "affine",
    "homography",
    "local_warp",
    "compression",
    "blur",
    "texture_loss",
    "partial_offscreen",
    "different_aspect_ratios",
    "alpha_source",
    "low_resolution",
}


class MotionRecoveryGateError(RuntimeError):
    """The generic recovery gate cannot safely create or join its evidence."""


def compare_recovery_to_truth(
    truth: MotionGroundTruthFixture,
    report: MotionRecoveryReport,
) -> dict[str, Any]:
    """Compare observable matrices and derivatives without assuming a pivot gauge."""

    expected = {item.frame: item for item in truth.frames}
    pairs = [(expected[item.frame], item) for item in report.frames if item.frame in expected]
    if len(pairs) != report.analyzed_frame_count:
        raise MotionRecoveryGateError("recovery frames do not align with ground truth")
    center_errors = [
        math.hypot(
            expected_frame.center_x - actual.center_x,
            expected_frame.center_y - actual.center_y,
        )
        for expected_frame, actual in pairs
        if actual.center_x is not None and actual.center_y is not None
    ]
    scale_errors = [
        expected_frame.scale_relative_to_contain - actual.scale_relative_to_contain
        for expected_frame, actual in pairs
        if actual.scale_relative_to_contain is not None
    ]
    rotation_errors = [
        _angle_delta(expected_frame.rotation_degrees, actual.rotation_degrees)
        for expected_frame, actual in pairs
        if actual.rotation_degrees is not None
    ]
    corner_errors = [
        _corner_error(
            np.asarray(expected_frame.matrix_3x3, dtype=np.float64),
            np.asarray(actual.matrix_3x3, dtype=np.float64),
            source_width=truth.source_width,
            source_height=truth.source_height,
            output_width=truth.output_width,
            output_height=truth.output_height,
        )
        for expected_frame, actual in pairs
    ]
    derivative_metrics: dict[str, float] = {}
    for order in ("velocity", "acceleration", "jerk"):
        for channel in ("center_x", "center_y", "scale", "rotation"):
            differences = [
                float(getattr(expected_frame, order)[channel])
                - float(getattr(actual, order)[channel])
                for expected_frame, actual in pairs
                if channel in getattr(actual, order)
            ]
            derivative_metrics[f"{order}_{channel}_rmse"] = _rmse(differences)
    return {
        "frame_count": len(pairs),
        "expected_model": truth.expected_model,
        "actual_model": report.selected_model,
        "model_match": truth.expected_model == report.selected_model,
        "expected_curve": truth.expected_curve,
        "actual_curve": report.curve_hint,
        "curve_match": truth.expected_curve == report.curve_hint,
        "center_rmse_normalized": _rmse(center_errors),
        "scale_rmse": _rmse(scale_errors),
        "rotation_rmse_degrees": _rmse(rotation_errors),
        "corner_rmse_normalized": _rmse(corner_errors),
        "mean_confidence": report.mean_confidence,
        "model_mismatch": report.model_mismatch,
        "outlier_frames": report.outlier_frames,
        "derivatives": derivative_metrics,
        "phase_boundaries": report.phase_boundaries,
        "pivot_identifiability": sorted(
            {item.pivot_identifiability for item in report.frames}
        ),
    }


def run_motion_recovery_gate(
    output_root: Path,
    *,
    fixture_ids: set[str] | None = None,
    require_clean_revision: bool = False,
) -> MotionRecoveryGateReport:
    """Render, recover, compare, and persist a generic motion corpus."""

    if output_root.exists():
        raise MotionRecoveryGateError(f"motion recovery root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    git_status = _git(repo, ["status", "--porcelain"]).splitlines()
    corpus, corpus_path = create_ground_truth_corpus(
        output_root / "corpus",
        fixture_ids=fixture_ids,
    )
    analyzer = MotionAnalyzer()
    checks: list[CheckResult] = []
    metrics: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, str] = {}

    if require_clean_revision:
        checks.append(
            _check(
                "repository-state",
                not git_status,
                "motion-recovery receipt is bound to a clean Git revision",
                {"git_revision": revision, "status": git_status},
            )
        )
    coverage = set(corpus.requirement_coverage)
    expected_coverage = (
        REQUIRED_CORPUS_TAGS
        if fixture_ids is None
        else {
            tag
            for fixture in corpus.fixtures
            for tag in fixture.requirement_tags
        }
    )
    checks.append(
        _check(
            "ground-truth-corpus-coverage",
            expected_coverage <= coverage,
            "synthetic corpus covers every selected motion and degradation family",
            {
                "required": sorted(expected_coverage),
                "actual": sorted(coverage),
                "sealed_reference_used": False,
            },
        )
    )

    for truth in corpus.fixtures:
        report_path = output_root / "recoveries" / truth.id / "motion-recovery.json"
        try:
            recovery_report = analyzer.analyze_video(
                Path(truth.source_path),
                Path(truth.video_path),
                output_path=report_path,
            )
            measured = compare_recovery_to_truth(truth, recovery_report)
            metrics[truth.id] = measured
            report_paths[truth.id] = str(report_path)
            _fixture_checks(checks, truth, recovery_report, measured)
        except Exception as error:
            metrics[truth.id] = {"error": f"{type(error).__name__}: {error}"}
            checks.append(
                _check(
                    f"{truth.id}:recovery",
                    False,
                    "fixture recovery raised instead of preserving a result",
                    metrics[truth.id],
                )
            )

    overall: Literal["pass", "fail"] = (
        "pass" if all(item.status == "pass" for item in checks) else "fail"
    )
    gate_report = MotionRecoveryGateReport(
        git_revision=revision,
        git_clean=not git_status,
        corpus_path=str(corpus_path),
        corpus_sha256=sha256_file(corpus_path),
        fixture_reports=report_paths,
        metrics=metrics,
        checks=checks,
        mandatory_skips=sum(item.status == "skip" for item in checks),
        provenance=Provenance(
            tool="aoa-editing-motion-recovery-gate",
            tool_version=version("aoa-editing"),
            command=[],
            parameters={
                "fixture_selection": (
                    sorted(fixture_ids) if fixture_ids is not None else "complete"
                ),
                "sealed_reference_used": False,
                "all_frames": True,
                "threshold_revision": "motion-recovery-v1",
                "clean_revision_required": require_clean_revision,
            },
            deterministic=True,
        ),
        overall=overall,
    )
    report_path = output_root / "motion-recovery-gate.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(gate_report.model_dump_json(indent=2))
        stream.write("\n")
    return gate_report


def _fixture_checks(
    checks: list[CheckResult],
    truth: MotionGroundTruthFixture,
    report: MotionRecoveryReport,
    measured: dict[str, Any],
) -> None:
    degraded = bool(truth.degradations) or "low_resolution" in truth.requirement_tags
    geometric_threshold = 0.032 if degraded else 0.018
    scale_threshold = 0.055 if degraded else 0.035
    rotation_threshold = 0.85 if degraded else 0.55
    confidence_threshold = 0.48 if degraded else 0.62
    checks.append(
        _check(
            f"{truth.id}:all-frames",
            report.analyzed_frame_count == truth.frame_count,
            "every rendered fixture frame has a recovery record",
            {
                "expected": truth.frame_count,
                "actual": report.analyzed_frame_count,
            },
        )
    )
    checks.append(
        _check(
            f"{truth.id}:model",
            bool(measured["model_match"]),
            "transform family is recovered or explicitly classified unknown",
            {
                "expected": measured["expected_model"],
                "actual": measured["actual_model"],
                "mismatch": measured["model_mismatch"],
            },
        )
    )
    if truth.expected_model == "unknown":
        checks.append(
            _check(
                f"{truth.id}:uncertainty",
                report.model_mismatch
                and any("global" in item.lower() for item in report.uncertainties),
                "unsupported local motion yields uncertainty instead of false precision",
                {"uncertainties": report.uncertainties},
            )
        )
    else:
        checks.extend(
            (
                _threshold(
                    f"{truth.id}:center",
                    float(measured["center_rmse_normalized"]),
                    geometric_threshold,
                    "normalized center RMSE",
                ),
                _threshold(
                    f"{truth.id}:scale",
                    float(measured["scale_rmse"]),
                    scale_threshold,
                    "relative scale RMSE",
                ),
                _threshold(
                    f"{truth.id}:rotation",
                    float(measured["rotation_rmse_degrees"]),
                    rotation_threshold,
                    "rotation RMSE in degrees",
                ),
                _threshold(
                    f"{truth.id}:corners",
                    float(measured["corner_rmse_normalized"]),
                    geometric_threshold,
                    "normalized transformed-corner RMSE",
                ),
                _threshold(
                    f"{truth.id}:confidence",
                    float(measured["mean_confidence"]),
                    confidence_threshold,
                    "minimum mean confidence",
                    lower_is_better=False,
                ),
            )
        )
    if any(
        tag
        in {
            "linear",
            "ease_in",
            "ease_out",
            "ease_in_out",
            "cubic_bezier",
            "hermite",
            "overshoot",
            "settle",
        }
        for tag in truth.requirement_tags
    ):
        checks.append(
            _check(
                f"{truth.id}:curve",
                bool(measured["curve_match"]) and report.curve_confidence >= 0.55,
                "temporal curve family is recovered with non-trivial confidence",
                {
                    "expected": measured["expected_curve"],
                    "actual": measured["actual_curve"],
                    "confidence": report.curve_confidence,
                },
            )
        )
    checks.append(
        _check(
            f"{truth.id}:pivot-boundary",
            measured["pivot_identifiability"] == ["unidentifiable"]
            and any("pivot" in item.lower() for item in report.uncertainties),
            "pivot/translation gauge ambiguity remains explicit",
            {
                "identifiability": measured["pivot_identifiability"],
                "uncertainties": report.uncertainties,
            },
        )
    )


def _corner_error(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> float:
    points = np.asarray(
        [
            [0.0, 0.0],
            [source_width, 0.0],
            [source_width, source_height],
            [0.0, source_height],
            [source_width / 2, source_height / 2],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    expected_points = cv2.perspectiveTransform(points, expected).reshape(-1, 2)
    actual_points = cv2.perspectiveTransform(points, actual).reshape(-1, 2)
    diagonal = math.hypot(output_width, output_height)
    return float(
        math.sqrt(float(np.mean(np.square(expected_points - actual_points)))) / diagonal
    )


def _angle_delta(expected: float, actual: float | None) -> float:
    if actual is None:
        raise MotionRecoveryGateError("recovered rotation is missing")
    return (expected - actual + 180.0) % 360.0 - 180.0


def _rmse(values: list[float]) -> float:
    if not values:
        return math.inf
    return float(math.sqrt(sum(value * value for value in values) / len(values)))


def _check(
    check_id: str,
    passed: bool,
    summary: str,
    measured: dict[str, Any],
) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if passed else "fail",
        summary=summary,
        measured=measured,
    )


def _threshold(
    check_id: str,
    actual: float,
    threshold: float,
    name: str,
    *,
    lower_is_better: bool = True,
) -> CheckResult:
    passed = actual <= threshold if lower_is_better else actual >= threshold
    relation = "maximum" if lower_is_better else "minimum"
    return _check(
        check_id,
        passed,
        f"{name} satisfies the frozen {relation}",
        {"actual": actual, relation: threshold},
    )


def load_motion_gate(path: Path) -> MotionRecoveryGateReport:
    """Load a persisted gate through its typed contract."""

    return MotionRecoveryGateReport.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _git(repo: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout
