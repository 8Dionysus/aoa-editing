"""All-frame Comparison Protocol v2 with a separate human-review boundary."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from aoa_editing.analysis.motion import MotionAnalyzer, characterize_similarity_coupling
from aoa_editing.domain.models import (
    CheckResult,
    ComparisonCriteriaV2,
    EditorialCriterionV2,
    EditorialReviewV2,
    MotionRecoveryFrame,
    MotionRecoveryReport,
    Provenance,
    ReferenceComparisonProtocolV2,
    ReferenceComparisonReportV2,
    ReferenceMotionFrameV2,
    ReferenceReconstructionSpecV2,
    new_id,
    utc_now,
)
from aoa_editing.infrastructure.media import ffprobe, sha256_file

VMAF_SCORE = re.compile(r"VMAF score:\s*([0-9.]+)")
SSIM_ALL = re.compile(r"All:([0-9.]+)")
PSNR_AVERAGE = re.compile(r"average:(inf|[0-9.]+)", re.IGNORECASE)
MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?(?:inf|[0-9.]+))\s*dB", re.IGNORECASE)

MANDATORY_AXES = [
    "technical",
    "geometric",
    "temporal",
    "perceptual",
    "editorial",
    "lineage",
]
MANDATORY_CHECKS = [
    "technical-resolution",
    "technical-aspect-ratio",
    "technical-frame-rate",
    "technical-frame-count",
    "technical-duration",
    "technical-codec-structure",
    "technical-audio-structure",
    "geometric-center",
    "geometric-scale",
    "geometric-rotation",
    "geometric-visible-boundary",
    "geometric-homography-corners",
    "geometric-motion-model",
    "geometric-pivot-identifiability",
    "geometric-transform-order",
    "temporal-position-velocity",
    "temporal-scale-velocity",
    "temporal-angular-velocity",
    "temporal-position-acceleration",
    "temporal-scale-acceleration",
    "temporal-angular-acceleration",
    "temporal-jerk",
    "temporal-phase-onset",
    "temporal-phase-duration",
    "temporal-phase-settle",
    "temporal-overshoot",
    "temporal-coupling",
    "temporal-flow",
    "perceptual-vmaf",
    "perceptual-ssim",
    "perceptual-black-frames",
    "perceptual-edge-exposure",
    "perceptual-unwanted-blur",
    "perceptual-temporal-consistency",
    "render-lineage",
    "editorial-human-review",
]
REQUIRED_ARTIFACTS = [
    "report",
    "candidate_motion",
    "frame_diagnostics",
    "side_by_side",
    "aligned_difference",
    "phase_contact_sheet",
    "motion_error_plot",
    "derivative_error_plot",
    "vmaf_log",
    "ssim_log",
    "psnr_log",
]
ADDITIONAL_HUMAN_AXES: list[EditorialCriterionV2] = [
    "reference_overlay_absence",
    "encoding_defect_separation",
]
ALGORITHMS: dict[str, Any] = {
    "motion": {
        "sampling": "every_frame",
        "models": ["similarity", "affine", "homography"],
        "correspondences": "SIFT ratio-test with RANSAC",
        "kinematics": "local polynomial degree 3",
        "pivot_policy": "do_not_score_unidentifiable_gauge_as_numeric_error",
        "transform_order_policy": "score_observable_source_to_output_matrix",
    },
    "temporal_flow": {
        "algorithm": "Farneback",
        "analysis_width": 270,
        "normalization": "per-frame reference flow p95 with 0.25 pixel floor",
    },
    "aligned_frames": {
        "analysis_width": 270,
        "color_space": "decoded grayscale",
        "black_luma_max": 1.0,
        "reference_nonblack_luma_min": 5.0,
        "unexpected_black_frames_max": 0,
        "blur_measure": "variance of Laplacian",
        "blur_ratio_p10_min": 0.70,
    },
    "perceptual": {
        "metric_width_max": 540,
        "metrics": ["VMAF", "SSIM", "PSNR"],
        "psnr_role": "diagnostic_only",
    },
    "artifacts": {
        "difference_gain": 3.0,
        "phase_frames": "opening, onset, twist start/peak/end, settle, final",
    },
}


class ComparisonV2Error(RuntimeError):
    """The protocol, inputs, or objective evidence violated a frozen contract."""


def freeze_comparison_protocol_v2(
    spec_path: Path,
    output_root: Path,
    *,
    implementation_revision: str,
    git_clean: bool,
) -> ReferenceComparisonProtocolV2:
    """Bind the frozen v2 spec to executable algorithms before a candidate exists."""

    selected_spec_path = spec_path.expanduser().resolve(strict=True)
    if output_root.exists():
        raise FileExistsError(f"comparison v2 protocol root already exists: {output_root}")
    if not git_clean:
        raise ComparisonV2Error("comparison v2 protocol must bind to a clean revision")
    spec = ReferenceReconstructionSpecV2.model_validate_json(
        selected_spec_path.read_text(encoding="utf-8")
    )
    criteria = ComparisonCriteriaV2.model_validate(spec.comparison_criteria)
    _validate_protocol_inputs(spec, criteria)
    output_root.mkdir(parents=True, exist_ok=False)
    human_rubric = cast(
        list[EditorialCriterionV2],
        [*criteria.editorial.rubric, *ADDITIONAL_HUMAN_AXES],
    )
    protocol = ReferenceComparisonProtocolV2(
        spec_id=spec.id,
        spec_path=str(selected_spec_path),
        spec_sha256=sha256_file(selected_spec_path),
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        implementation_revision=implementation_revision,
        git_clean=True,
        threshold_revision=criteria.threshold_revision,
        criteria=criteria,
        algorithm_revision="comparison-v2-all-frame-1",
        algorithms=ALGORITHMS,
        mandatory_axes=cast(Any, MANDATORY_AXES),
        mandatory_checks=MANDATORY_CHECKS,
        required_artifacts=REQUIRED_ARTIFACTS,
        human_rubric=human_rubric,
        provenance=Provenance(
            tool="aoa-editing-comparison-protocol-freezer-v2",
            tool_version=version("aoa-editing"),
            command=[],
            parameters={
                "spec_sha256": sha256_file(selected_spec_path),
                "threshold_revision": criteria.threshold_revision,
                "algorithm_revision": "comparison-v2-all-frame-1",
                "candidate_render_seen": False,
            },
            deterministic=True,
        ),
    )
    protocol_path = output_root / "comparison-protocol-v2.json"
    _write_new(protocol_path, protocol.model_dump_json(indent=2) + "\n")
    template = {
        "schema_version": "2.0.0",
        "protocol_id": protocol.id,
        "candidate_sha256": None,
        "reviewer": None,
        "reviewer_role": "human_operator",
        "verdict": None,
        "rubric": [
            {"criterion": criterion, "status": None, "notes": None}
            for criterion in protocol.human_rubric
        ],
        "reviewed_artifacts": [
            "side_by_side",
            "phase_contact_sheet",
            "aligned_difference",
        ],
        "reference_media_role": "comparison_only",
    }
    _write_new(
        output_root / "human-review-template-v2.json",
        json.dumps(template, indent=2, ensure_ascii=False) + "\n",
    )
    return protocol


def run_comparison_v2(
    protocol_path: Path,
    spec_path: Path,
    source_path: Path,
    candidate_path: Path,
    reference_path: Path,
    output_root: Path,
    *,
    lineage_path: Path,
    human_review_path: Path | None = None,
    allow_synthetic_identity_fixture: bool = False,
) -> ReferenceComparisonReportV2:
    """Execute every frozen objective axis and leave human judgment attributable."""

    selected_protocol_path = protocol_path.expanduser().resolve(strict=True)
    selected_spec_path = spec_path.expanduser().resolve(strict=True)
    selected_source = source_path.expanduser().resolve(strict=True)
    selected_candidate = candidate_path.expanduser().resolve(strict=True)
    selected_reference = reference_path.expanduser().resolve(strict=True)
    selected_lineage = lineage_path.expanduser().resolve(strict=True)
    if output_root.exists():
        raise FileExistsError(f"comparison v2 output root already exists: {output_root}")
    protocol = ReferenceComparisonProtocolV2.model_validate_json(
        selected_protocol_path.read_text(encoding="utf-8")
    )
    spec = ReferenceReconstructionSpecV2.model_validate_json(
        selected_spec_path.read_text(encoding="utf-8")
    )
    _validate_bound_inputs(
        protocol,
        spec,
        selected_protocol_path,
        selected_spec_path,
        selected_source,
        selected_candidate,
        selected_reference,
        allow_synthetic_identity_fixture=allow_synthetic_identity_fixture,
    )
    output_root.mkdir(parents=True, exist_ok=False)

    candidate_motion_path = output_root / "candidate-motion-recovery.json"
    recovery = MotionAnalyzer().analyze_video(
        selected_source,
        selected_candidate,
        output_path=candidate_motion_path,
    )
    if recovery.analyzed_frame_count != spec.duration_frames:
        raise ComparisonV2Error("candidate all-frame recovery did not cover the frozen duration")

    _, candidate_metadata, candidate_probe, candidate_probe_command = ffprobe(selected_candidate)
    _, reference_metadata, reference_probe, reference_probe_command = ffprobe(selected_reference)
    audio, audio_command = _audio_measurement(selected_candidate)
    technical, technical_checks = _technical(
        protocol,
        spec,
        recovery,
        candidate_metadata,
        reference_metadata,
        candidate_probe,
        reference_probe,
        audio,
    )
    geometric, geometric_checks, motion_plot_data = _geometric(
        protocol,
        spec,
        recovery,
        selected_source,
    )
    diagnostics_path = output_root / "frame-diagnostics.json"
    frame_diagnostics = _paired_frame_diagnostics(
        selected_reference,
        selected_candidate,
        diagnostics_path,
        protocol.algorithms,
        expected_frames=spec.duration_frames,
    )
    temporal, temporal_checks, derivative_plot_data = _temporal(
        protocol,
        spec,
        recovery,
        frame_diagnostics,
    )

    metric_width, metric_height = _metric_geometry(spec.width, spec.height)
    vmaf_path = output_root / "vmaf.json"
    ssim_path = output_root / "ssim.log"
    psnr_path = output_root / "psnr.log"
    vmaf, vmaf_command = _vmaf(
        selected_candidate,
        selected_reference,
        metric_width,
        metric_height,
        vmaf_path,
    )
    ssim, ssim_command = _ssim(
        selected_candidate,
        selected_reference,
        metric_width,
        metric_height,
        ssim_path,
    )
    psnr, psnr_command = _psnr(
        selected_candidate,
        selected_reference,
        metric_width,
        metric_height,
        psnr_path,
    )
    perceptual, perceptual_checks = _perceptual(
        protocol,
        geometric,
        temporal,
        frame_diagnostics,
        vmaf,
        ssim,
        psnr,
    )
    lineage = _lineage(
        protocol,
        selected_lineage,
        selected_candidate,
        selected_reference,
        allow_synthetic_identity_fixture=allow_synthetic_identity_fixture,
    )
    lineage_check = _check(
        "render-lineage",
        bool(lineage["valid"]),
        "render lineage contains only the permitted source and no reference material",
        lineage,
    )

    side_by_side_path = output_root / "side-by-side.mp4"
    difference_path = output_root / "aligned-difference.mp4"
    phase_sheet_path = output_root / "phase-contact-sheet.jpg"
    motion_plot_path = output_root / "motion-error-plot.png"
    derivative_plot_path = output_root / "derivative-error-plot.png"
    side_command = _side_by_side(
        selected_reference,
        selected_candidate,
        side_by_side_path,
        metric_width,
        metric_height,
    )
    difference_command = _difference_video(
        selected_reference,
        selected_candidate,
        difference_path,
        metric_width,
        metric_height,
    )
    _phase_contact_sheet(
        selected_reference,
        selected_candidate,
        phase_sheet_path,
        _phase_frames(spec),
    )
    _error_plot(
        motion_plot_path,
        motion_plot_data,
        title="All-frame geometric error / frozen maximum",
    )
    _error_plot(
        derivative_plot_path,
        derivative_plot_data,
        title="All-frame temporal derivative error / frozen maximum",
    )

    candidate_hash = sha256_file(selected_candidate)
    review = (
        EditorialReviewV2.model_validate_json(
            human_review_path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
        )
        if human_review_path is not None
        else None
    )
    editorial, editorial_check = _editorial(
        protocol,
        candidate_hash,
        review,
        {
            "side_by_side": side_by_side_path,
            "phase_contact_sheet": phase_sheet_path,
            "aligned_difference": difference_path,
        },
    )
    checks = [
        *technical_checks,
        *geometric_checks,
        *temporal_checks,
        *perceptual_checks,
        lineage_check,
        editorial_check,
    ]
    _validate_check_coverage(protocol, checks)
    objective_checks = [item for item in checks if item.id != "editorial-human-review"]
    objective_overall: Literal["pass", "fail"] = (
        "pass" if all(item.status == "pass" for item in objective_checks) else "fail"
    )
    overall = _combined_overall(objective_overall, editorial_check)
    report_path = output_root / "comparison-v2.json"
    artifacts = {
        "report": str(report_path),
        "candidate_motion": str(candidate_motion_path),
        "frame_diagnostics": str(diagnostics_path),
        "side_by_side": str(side_by_side_path),
        "aligned_difference": str(difference_path),
        "phase_contact_sheet": str(phase_sheet_path),
        "motion_error_plot": str(motion_plot_path),
        "derivative_error_plot": str(derivative_plot_path),
        "vmaf_log": str(vmaf_path),
        "ssim_log": str(ssim_path),
        "psnr_log": str(psnr_path),
    }
    if set(artifacts) != set(protocol.required_artifacts):
        raise ComparisonV2Error("comparison artifact coverage drifted from the frozen protocol")
    commands = [
        candidate_probe_command,
        reference_probe_command,
        audio_command,
        vmaf_command,
        ssim_command,
        psnr_command,
        side_command,
        difference_command,
    ]
    report = ReferenceComparisonReportV2(
        protocol_id=protocol.id,
        protocol_path=str(selected_protocol_path),
        protocol_sha256=sha256_file(selected_protocol_path),
        spec_id=spec.id,
        spec_sha256=sha256_file(selected_spec_path),
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        candidate_sha256=candidate_hash,
        candidate_path=str(selected_candidate),
        technical=technical,
        geometric=geometric,
        temporal=temporal,
        perceptual=perceptual,
        editorial=editorial,
        lineage=lineage,
        checks=checks,
        artifacts=artifacts,
        human_review=review,
        provenance=Provenance(
            tool="aoa-editing-reference-comparison-v2",
            tool_version=version("aoa-editing"),
            command=[item for command in commands for item in command],
            parameters={
                "algorithm_revision": protocol.algorithm_revision,
                "threshold_revision": protocol.threshold_revision,
                "all_frames": True,
                "metric_geometry": [metric_width, metric_height],
                "reference_role": "analysis_and_comparison_only",
                "synthetic_identity_fixture": allow_synthetic_identity_fixture,
            },
            deterministic=True,
        ),
        objective_overall=objective_overall,
        overall=overall,
    )
    _write_new(report_path, report.model_dump_json(indent=2) + "\n")
    return report


def record_editorial_review_v2(
    report_path: Path,
    protocol_path: Path,
    review_path: Path,
    output_path: Path,
) -> ReferenceComparisonReportV2:
    """Attach a human-authored verdict as a new immutable report revision."""

    selected_report = report_path.expanduser().resolve(strict=True)
    selected_protocol = protocol_path.expanduser().resolve(strict=True)
    selected_review = review_path.expanduser().resolve(strict=True)
    if output_path.exists():
        raise FileExistsError(f"reviewed comparison report already exists: {output_path}")
    report = ReferenceComparisonReportV2.model_validate_json(
        selected_report.read_text(encoding="utf-8")
    )
    protocol = ReferenceComparisonProtocolV2.model_validate_json(
        selected_protocol.read_text(encoding="utf-8")
    )
    review = EditorialReviewV2.model_validate_json(selected_review.read_text(encoding="utf-8"))
    if report.protocol_id != protocol.id:
        raise ComparisonV2Error("comparison report and protocol IDs differ")
    if report.protocol_sha256 != sha256_file(selected_protocol):
        raise ComparisonV2Error("comparison report protocol hash is stale")
    artifact_subset = {
        name: Path(report.artifacts[name])
        for name in ("side_by_side", "phase_contact_sheet", "aligned_difference")
    }
    editorial, editorial_check = _editorial(
        protocol,
        report.candidate_sha256,
        review,
        artifact_subset,
    )
    checks = [
        editorial_check if item.id == "editorial-human-review" else item for item in report.checks
    ]
    overall = _combined_overall(report.objective_overall, editorial_check)
    reviewed = report.model_copy(
        update={
            "id": new_id("comparison"),
            "generated_at": utc_now(),
            "supersedes_report_id": report.id,
            "editorial": editorial,
            "checks": checks,
            "human_review": review,
            "provenance": report.provenance.model_copy(
                update={
                    "parameters": {
                        **report.provenance.parameters,
                        "editorial_review_id": review.id,
                        "editorial_review_sha256": sha256_file(selected_review),
                    }
                }
            ),
            "overall": overall,
        }
    )
    _write_new(output_path, reviewed.model_dump_json(indent=2) + "\n")
    return reviewed


def _validate_protocol_inputs(
    spec: ReferenceReconstructionSpecV2,
    criteria: ComparisonCriteriaV2,
) -> None:
    if not spec.frozen_before_first_v2_render:
        raise ComparisonV2Error("reference spec is not frozen before v2 rendering")
    if criteria.lineage.only_visual_source_hash != spec.source_sha256:
        raise ComparisonV2Error("comparison lineage source differs from the frozen spec")
    if criteria.lineage.reference_hash_forbidden != spec.reference_sha256:
        raise ComparisonV2Error("comparison forbidden hash differs from the frozen spec")
    if criteria.technical.resolution != (spec.width, spec.height):
        raise ComparisonV2Error("comparison resolution differs from the frozen spec")
    if criteria.technical.frame_rate != spec.frame_rate:
        raise ComparisonV2Error("comparison frame rate differs from the frozen spec")


def _validate_bound_inputs(
    protocol: ReferenceComparisonProtocolV2,
    spec: ReferenceReconstructionSpecV2,
    protocol_path: Path,
    spec_path: Path,
    source_path: Path,
    candidate_path: Path,
    reference_path: Path,
    *,
    allow_synthetic_identity_fixture: bool,
) -> None:
    if protocol.spec_id != spec.id or protocol.spec_sha256 != sha256_file(spec_path):
        raise ComparisonV2Error("comparison protocol does not bind the supplied spec")
    if Path(protocol.spec_path) != spec_path:
        raise ComparisonV2Error("comparison protocol spec path differs from the supplied path")
    if sha256_file(source_path) != protocol.source_sha256:
        raise ComparisonV2Error("source hash does not match the comparison protocol")
    if sha256_file(reference_path) != protocol.reference_sha256:
        raise ComparisonV2Error("reference hash does not match the comparison protocol")
    if protocol_path.name != "comparison-protocol-v2.json":
        raise ComparisonV2Error("comparison protocol path is not the canonical artifact name")
    if (
        candidate_path == reference_path or sha256_file(candidate_path) == protocol.reference_sha256
    ) and not allow_synthetic_identity_fixture:
        raise ComparisonV2Error("production candidate must not be the reference media")


def _technical(
    protocol: ReferenceComparisonProtocolV2,
    spec: ReferenceReconstructionSpecV2,
    recovery: MotionRecoveryReport,
    candidate_metadata: Any,
    reference_metadata: Any,
    candidate_probe: dict[str, Any],
    reference_probe: dict[str, Any],
    audio: dict[str, Any],
) -> tuple[dict[str, Any], list[CheckResult]]:
    criteria = protocol.criteria.technical
    resolution = [recovery.width, recovery.height]
    expected_resolution = list(criteria.resolution)
    aspect_error = abs(
        recovery.width / recovery.height - criteria.resolution[0] / criteria.resolution[1]
    )
    fps_error = abs(recovery.frame_rate.fps - criteria.frame_rate.fps)
    frame_error = abs(recovery.frame_count - spec.duration_frames)
    actual_duration = candidate_metadata.duration_seconds
    expected_duration = spec.duration_frames / spec.frame_rate.fps
    duration_error = (
        abs(float(actual_duration) - expected_duration) if actual_duration is not None else math.inf
    )
    one_frame_seconds = 1.0 / spec.frame_rate.fps
    codec_ok = bool(candidate_metadata.video_codec and candidate_metadata.pixel_format)
    expected_audio = spec.audio
    audio_exact = (
        audio["has_audio_stream"] == expected_audio.has_audio_stream
        and (
            expected_audio.sample_rate is None or audio["sample_rate"] == expected_audio.sample_rate
        )
        and (expected_audio.channels is None or audio["channels"] == expected_audio.channels)
        and (not expected_audio.silent_for_full_duration or audio["silent_below_minus_60_db"])
    )
    measured = {
        "expected_resolution": expected_resolution,
        "actual_resolution": resolution,
        "expected_aspect_ratio": criteria.resolution[0] / criteria.resolution[1],
        "actual_aspect_ratio": recovery.width / recovery.height,
        "aspect_ratio_error": aspect_error,
        "expected_frame_rate": criteria.frame_rate.model_dump(mode="json"),
        "actual_frame_rate": recovery.frame_rate.model_dump(mode="json"),
        "frame_rate_error": fps_error,
        "expected_frame_count": spec.duration_frames,
        "actual_frame_count": recovery.frame_count,
        "frame_count_error": frame_error,
        "expected_duration_seconds": expected_duration,
        "actual_duration_seconds": actual_duration,
        "duration_error_seconds": duration_error,
        "candidate_codec": {
            "format": candidate_metadata.format_name,
            "video": candidate_metadata.video_codec,
            "pixel_format": candidate_metadata.pixel_format,
            "color_space": candidate_metadata.color_space,
        },
        "reference_codec": {
            "format": reference_metadata.format_name,
            "video": reference_metadata.video_codec,
            "pixel_format": reference_metadata.pixel_format,
            "color_space": reference_metadata.color_space,
        },
        "candidate_stream_count": len(candidate_probe.get("streams", [])),
        "reference_stream_count": len(reference_probe.get("streams", [])),
        "audio": audio,
    }
    checks = [
        _check(
            "technical-resolution",
            resolution == expected_resolution,
            "candidate resolution matches the frozen technical contract",
            measured,
        ),
        _check(
            "technical-aspect-ratio",
            aspect_error <= 1e-12,
            "candidate aspect ratio matches the frozen technical contract",
            {"error": aspect_error},
        ),
        _check(
            "technical-frame-rate",
            fps_error <= 1e-9,
            "candidate frame rate matches exactly",
            {"error": fps_error},
        ),
        _check(
            "technical-frame-count",
            frame_error <= criteria.frame_count_tolerance,
            "candidate frame count matches exactly",
            {"error": frame_error, "maximum": criteria.frame_count_tolerance},
        ),
        _check(
            "technical-duration",
            duration_error <= one_frame_seconds,
            "container duration is within one frame of the decoded contract",
            {"error_seconds": duration_error, "maximum_seconds": one_frame_seconds},
        ),
        _check(
            "technical-codec-structure",
            codec_ok,
            "candidate has a decodable video codec and pixel format",
            measured["candidate_codec"],
        ),
        _check(
            "technical-audio-structure",
            audio_exact,
            "candidate audio presence, layout, and silence match the frozen structure",
            {
                "expected": expected_audio.model_dump(mode="json"),
                "actual": audio,
            },
        ),
    ]
    return measured, checks


def _geometric(
    protocol: ReferenceComparisonProtocolV2,
    spec: ReferenceReconstructionSpecV2,
    recovery: MotionRecoveryReport,
    source_path: Path,
) -> tuple[dict[str, Any], list[CheckResult], dict[str, Any]]:
    source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise ComparisonV2Error(f"cannot decode comparison source: {source_path}")
    source_height, source_width = source.shape[:2]
    pairs = _frame_pairs(spec, recovery)
    center_errors: list[float] = []
    scale_errors: list[float] = []
    rotation_errors: list[float] = []
    boundary_errors: list[float] = []
    corner_errors: list[float] = []
    for expected, actual in pairs:
        center_errors.append(
            math.hypot(
                expected.center_x - _required(actual.center_x, "center_x"),
                expected.center_y - _required(actual.center_y, "center_y"),
            )
        )
        scale_errors.append(
            expected.scale_relative_to_contain
            - _required(actual.scale_relative_to_contain, "scale")
        )
        rotation_errors.append(
            expected.rotation_degrees - _required(actual.rotation_degrees, "rotation")
        )
        actual_boundary = _normalized_boundary(
            actual.matrix_3x3,
            source_width=source_width,
            source_height=source_height,
            output_width=spec.width,
            output_height=spec.height,
        )
        boundary_errors.append(_point_rmse(expected.visible_source_boundary, actual_boundary))
        corner_errors.append(
            _projected_grid_rmse(
                expected.matrix_3x3,
                actual.matrix_3x3,
                source_width=source_width,
                source_height=source_height,
                output_width=spec.width,
                output_height=spec.height,
            )
        )
    criteria = protocol.criteria.geometric
    center_rmse = _rmse(center_errors)
    scale_rmse = _rmse(scale_errors)
    rotation_rmse = _rmse(rotation_errors)
    boundary_rmse = _rmse(boundary_errors)
    corner_rmse = _rmse(corner_errors)
    expected_pivot = {item.pivot_identifiability for item, _ in pairs}
    actual_pivot = {item.pivot_identifiability for _, item in pairs}
    pivot_honest = expected_pivot == {"unidentifiable"} and actual_pivot == {"unidentifiable"}
    model_match = recovery.selected_model == spec.motion_model
    transform_order_honest = (
        spec.transform_order.get("conclusion")
        == "global_matrix_observable_but_authoring_order_not_unique"
        and corner_rmse <= criteria.homography_corner_rmse_normalized_max
    )
    measured = {
        "sample_count": len(pairs),
        "all_frames": [item.frame for item, _ in pairs],
        "center_rmse_normalized": center_rmse,
        "scale_rmse": scale_rmse,
        "rotation_rmse_degrees": rotation_rmse,
        "visible_boundary_rmse_normalized": boundary_rmse,
        "homography_corner_rmse_normalized": corner_rmse,
        "expected_motion_model": spec.motion_model,
        "actual_motion_model": recovery.selected_model,
        "actual_model_distribution": recovery.model_distribution,
        "pivot_error": None,
        "pivot_assessment": (
            "numeric pivot error is inapplicable because pivot and translation are "
            "gauge-equivalent in both evidence sets"
        ),
        "transform_order": {
            "expected_assessment": spec.transform_order,
            "scored_observable": "source_to_output_matrix_projected_grid",
            "unique_authoring_order_claimed": False,
        },
        "thresholds": criteria.model_dump(mode="json"),
    }
    checks = [
        _maximum_check(
            "geometric-center",
            center_rmse,
            criteria.all_frame_center_rmse_normalized_max,
            "all-frame center RMSE",
        ),
        _maximum_check(
            "geometric-scale",
            scale_rmse,
            criteria.all_frame_scale_rmse_max,
            "all-frame scale RMSE",
        ),
        _maximum_check(
            "geometric-rotation",
            rotation_rmse,
            criteria.all_frame_rotation_rmse_degrees_max,
            "all-frame rotation RMSE",
        ),
        _maximum_check(
            "geometric-visible-boundary",
            boundary_rmse,
            criteria.visible_boundary_rmse_normalized_max,
            "visible source-boundary RMSE",
        ),
        _maximum_check(
            "geometric-homography-corners",
            corner_rmse,
            criteria.homography_corner_rmse_normalized_max,
            "projected-grid homography RMSE",
        ),
        _check(
            "geometric-motion-model",
            model_match,
            "candidate global motion family matches the frozen evidence",
            {
                "expected": spec.motion_model,
                "actual": recovery.selected_model,
                "distribution": recovery.model_distribution,
            },
        ),
        _check(
            "geometric-pivot-identifiability",
            pivot_honest,
            "unidentifiable pivot is preserved without a fabricated numeric score",
            {
                "expected": sorted(expected_pivot),
                "actual": sorted(actual_pivot),
                "numeric_error": None,
            },
        ),
        _check(
            "geometric-transform-order",
            transform_order_honest,
            "observable matrices match without claiming a non-identifiable authoring order",
            cast(dict[str, Any], measured["transform_order"]),
        ),
    ]
    plot = {
        "center": _normalized_series(center_errors, criteria.all_frame_center_rmse_normalized_max),
        "scale": _normalized_series(scale_errors, criteria.all_frame_scale_rmse_max),
        "rotation": _normalized_series(
            rotation_errors,
            criteria.all_frame_rotation_rmse_degrees_max,
        ),
        "boundary": _normalized_series(
            boundary_errors,
            criteria.visible_boundary_rmse_normalized_max,
        ),
        "matrix-grid": _normalized_series(
            corner_errors,
            criteria.homography_corner_rmse_normalized_max,
        ),
    }
    return measured, checks, plot


def _paired_frame_diagnostics(
    reference_path: Path,
    candidate_path: Path,
    output_path: Path,
    algorithms: dict[str, Any],
    *,
    expected_frames: int,
) -> dict[str, Any]:
    width = int(algorithms["temporal_flow"]["analysis_width"])
    aligned = algorithms["aligned_frames"]
    reference_capture = cv2.VideoCapture(str(reference_path))
    candidate_capture = cv2.VideoCapture(str(candidate_path))
    if not reference_capture.isOpened() or not candidate_capture.isOpened():
        reference_capture.release()
        candidate_capture.release()
        raise ComparisonV2Error("cannot decode aligned comparison videos")
    records: list[dict[str, float | int | bool]] = []
    flow_errors: list[float] = []
    temporal_squared = 0.0
    temporal_count = 0
    aligned_squared = 0.0
    aligned_count = 0
    blur_ratios: list[float] = []
    unexpected_black = 0
    previous_reference: NDArray[np.uint8] | None = None
    previous_candidate: NDArray[np.uint8] | None = None
    decoded = 0
    try:
        while True:
            ref_ok, reference = reference_capture.read()
            candidate_ok, candidate = candidate_capture.read()
            if ref_ok != candidate_ok:
                raise ComparisonV2Error(
                    "reference and candidate decoders ended on different frames"
                )
            if not ref_ok:
                break
            reference_gray = _scaled_gray(cast(NDArray[np.uint8], reference), width)
            candidate_gray = _scaled_gray(cast(NDArray[np.uint8], candidate), width)
            if reference_gray.shape != candidate_gray.shape:
                raise ComparisonV2Error("aligned comparison frames have different shapes")
            difference = candidate_gray.astype(np.float32) - reference_gray.astype(np.float32)
            aligned_squared += float(np.sum(np.square(difference)))
            aligned_count += difference.size
            reference_luma = float(np.mean(reference_gray))
            candidate_luma = float(np.mean(candidate_gray))
            is_unexpected_black = candidate_luma <= float(
                aligned["black_luma_max"]
            ) and reference_luma >= float(aligned["reference_nonblack_luma_min"])
            unexpected_black += int(is_unexpected_black)
            reference_blur = float(cv2.Laplacian(reference_gray, cv2.CV_64F).var())
            candidate_blur = float(cv2.Laplacian(candidate_gray, cv2.CV_64F).var())
            blur_ratio = candidate_blur / max(reference_blur, 1e-9)
            blur_ratios.append(blur_ratio)
            flow_error = 0.0
            temporal_error = 0.0
            if previous_reference is not None and previous_candidate is not None:
                reference_flow = _flow(previous_reference, reference_gray)
                candidate_flow = _flow(previous_candidate, candidate_gray)
                reference_magnitude = np.linalg.norm(reference_flow, axis=2)
                denominator = max(
                    0.25,
                    float(np.percentile(reference_magnitude, 95)),
                )
                flow_error = (
                    math.sqrt(float(np.mean(np.square(candidate_flow - reference_flow))))
                    / denominator
                )
                flow_errors.append(flow_error)
                reference_delta = reference_gray.astype(np.float32) - previous_reference.astype(
                    np.float32
                )
                candidate_delta = candidate_gray.astype(np.float32) - previous_candidate.astype(
                    np.float32
                )
                delta_error = candidate_delta - reference_delta
                temporal_squared += float(np.sum(np.square(delta_error)))
                temporal_count += delta_error.size
                temporal_error = math.sqrt(float(np.mean(np.square(delta_error)))) / 255.0
            records.append(
                {
                    "frame": decoded,
                    "reference_luma_mean": reference_luma,
                    "candidate_luma_mean": candidate_luma,
                    "unexpected_black": is_unexpected_black,
                    "reference_laplacian_variance": reference_blur,
                    "candidate_laplacian_variance": candidate_blur,
                    "blur_ratio": blur_ratio,
                    "flow_error_normalized": flow_error,
                    "temporal_delta_error_normalized": temporal_error,
                    "aligned_luma_mae_normalized": float(np.mean(np.abs(difference)) / 255.0),
                }
            )
            previous_reference = reference_gray
            previous_candidate = candidate_gray
            decoded += 1
    finally:
        reference_capture.release()
        candidate_capture.release()
    if decoded != expected_frames:
        raise ComparisonV2Error(
            f"aligned diagnostic decoded {decoded} frames, expected {expected_frames}"
        )
    measured = {
        "frame_count": decoded,
        "temporal_flow_rmse": _rmse(flow_errors) if flow_errors else 0.0,
        "temporal_consistency_rmse": (
            math.sqrt(temporal_squared / temporal_count) / 255.0 if temporal_count else 0.0
        ),
        "aligned_luma_rmse_normalized": (
            math.sqrt(aligned_squared / aligned_count) / 255.0 if aligned_count else 0.0
        ),
        "unexpected_black_frames": unexpected_black,
        "blur_ratio_p10": float(np.percentile(blur_ratios, 10)),
        "frames": records,
    }
    _write_new(output_path, json.dumps(measured, indent=2, ensure_ascii=False) + "\n")
    return measured


def _temporal(
    protocol: ReferenceComparisonProtocolV2,
    spec: ReferenceReconstructionSpecV2,
    recovery: MotionRecoveryReport,
    diagnostics: dict[str, Any],
) -> tuple[dict[str, Any], list[CheckResult], dict[str, Any]]:
    pairs = _frame_pairs(spec, recovery)
    criteria = protocol.criteria.temporal
    derivative_errors: dict[str, list[float]] = {}
    for order in ("velocity", "acceleration", "jerk"):
        derivative_errors[f"{order}_position"] = [
            math.hypot(
                expected_value(order, expected, "center_x")
                - actual_value(order, actual, "center_x"),
                expected_value(order, expected, "center_y")
                - actual_value(order, actual, "center_y"),
            )
            for expected, actual in pairs
        ]
        derivative_errors[f"{order}_scale"] = [
            expected_value(order, expected, "scale") - actual_value(order, actual, "scale")
            for expected, actual in pairs
        ]
        derivative_errors[f"{order}_rotation"] = [
            expected_value(order, expected, "rotation") - actual_value(order, actual, "rotation")
            for expected, actual in pairs
        ]
    derivative_rmse = {name: _rmse(values) for name, values in derivative_errors.items()}
    expected_onset = int(spec.phase_model["onset_frame"])
    expected_settle = int(spec.phase_model["settle_frame"])
    actual_onset = int(recovery.phase_boundaries["onset"])
    actual_settle = int(recovery.phase_boundaries["settle"])
    onset_error = abs(actual_onset - expected_onset)
    settle_error = abs(actual_settle - expected_settle)
    duration_error = abs((actual_settle - actual_onset) - (expected_settle - expected_onset))
    expected_overshoot = bool(spec.phase_model["overshoot_detected"])
    actual_overshoot = _overshoot_detected(pairs, protocol)

    expected_coupling = dict(spec.phase_model["channel_coupling"])
    actual_coupling = asdict(
        characterize_similarity_coupling(
            center_x=[_required(item.center_x, "center_x") for _, item in pairs],
            center_y=[_required(item.center_y, "center_y") for _, item in pairs],
            scale=[_required(item.scale_relative_to_contain, "scale") for _, item in pairs],
            rotation=[_required(item.rotation_degrees, "rotation") for _, item in pairs],
            fps=spec.frame_rate.fps,
        )
    )
    coupling_errors = {
        "scale_rotation_progress_rmse": abs(
            float(actual_coupling["scale_rotation_progress_rmse"])
            - float(expected_coupling["scale_rotation_progress_rmse"])
        ),
        "maximum_phase_offset": abs(
            float(actual_coupling["maximum_phase_offset"])
            - float(expected_coupling["maximum_phase_offset"])
        ),
        "peak_velocity_lag_frames": abs(
            int(actual_coupling["peak_velocity_lag_frames"])
            - int(expected_coupling["peak_velocity_lag_frames"])
        ),
        "center_path_max_deviation": abs(
            float(actual_coupling["center_path_max_deviation"])
            - float(expected_coupling["center_path_max_deviation"])
        ),
    }
    coupling_pass = (
        actual_coupling["leading_explanation"] == expected_coupling["leading_explanation"]
        and actual_coupling["center_phase_relation"] == expected_coupling["center_phase_relation"]
        and coupling_errors["scale_rotation_progress_rmse"]
        <= protocol.criteria.geometric.all_frame_scale_rmse_max
        and coupling_errors["maximum_phase_offset"] <= criteria.position_velocity_rmse_max
        and coupling_errors["peak_velocity_lag_frames"] <= criteria.phase_onset_tolerance_frames
        and coupling_errors["center_path_max_deviation"]
        <= protocol.criteria.geometric.visible_boundary_rmse_normalized_max
    )
    combined_jerk = max(
        derivative_rmse["jerk_position"],
        derivative_rmse["jerk_scale"],
        derivative_rmse["jerk_rotation"],
    )
    flow_rmse = float(diagnostics["temporal_flow_rmse"])
    temporal_consistency = float(diagnostics["temporal_consistency_rmse"])
    measured = {
        "derivative_rmse": derivative_rmse,
        "combined_jerk_rmse": combined_jerk,
        "phase": {
            "expected_onset": expected_onset,
            "actual_onset": actual_onset,
            "onset_error_frames": onset_error,
            "expected_settle": expected_settle,
            "actual_settle": actual_settle,
            "settle_error_frames": settle_error,
            "duration_error_frames": duration_error,
            "expected_overshoot": expected_overshoot,
            "actual_overshoot": actual_overshoot,
        },
        "coupling": {
            "expected": expected_coupling,
            "actual": actual_coupling,
            "errors": coupling_errors,
        },
        "temporal_flow_rmse": flow_rmse,
        "temporal_consistency_rmse": temporal_consistency,
        "thresholds": criteria.model_dump(mode="json"),
    }
    checks = [
        _maximum_check(
            "temporal-position-velocity",
            derivative_rmse["velocity_position"],
            criteria.position_velocity_rmse_max,
            "position velocity RMSE",
        ),
        _maximum_check(
            "temporal-scale-velocity",
            derivative_rmse["velocity_scale"],
            criteria.scale_velocity_rmse_max,
            "scale velocity RMSE",
        ),
        _maximum_check(
            "temporal-angular-velocity",
            derivative_rmse["velocity_rotation"],
            criteria.angular_velocity_rmse_degrees_max,
            "angular velocity RMSE",
        ),
        _maximum_check(
            "temporal-position-acceleration",
            derivative_rmse["acceleration_position"],
            criteria.position_acceleration_rmse_max,
            "position acceleration RMSE",
        ),
        _maximum_check(
            "temporal-scale-acceleration",
            derivative_rmse["acceleration_scale"],
            criteria.scale_acceleration_rmse_max,
            "scale acceleration RMSE",
        ),
        _maximum_check(
            "temporal-angular-acceleration",
            derivative_rmse["acceleration_rotation"],
            criteria.angular_acceleration_rmse_degrees_max,
            "angular acceleration RMSE",
        ),
        _maximum_check(
            "temporal-jerk",
            combined_jerk,
            criteria.jerk_rmse_max,
            "maximum channel jerk RMSE",
        ),
        _maximum_check(
            "temporal-phase-onset",
            float(onset_error),
            float(criteria.phase_onset_tolerance_frames),
            "phase onset error",
        ),
        _maximum_check(
            "temporal-phase-duration",
            float(duration_error),
            float(criteria.phase_onset_tolerance_frames + criteria.phase_settle_tolerance_frames),
            "active phase duration error",
        ),
        _maximum_check(
            "temporal-phase-settle",
            float(settle_error),
            float(criteria.phase_settle_tolerance_frames),
            "phase settle error",
        ),
        _check(
            "temporal-overshoot",
            actual_overshoot == expected_overshoot,
            "candidate overshoot presence matches the frozen phase model",
            {
                "expected": expected_overshoot,
                "actual": actual_overshoot,
            },
        ),
        _check(
            "temporal-coupling",
            coupling_pass,
            "candidate preserves the frozen channel-lag and curved-path explanation",
            cast(dict[str, Any], measured["coupling"]),
        ),
        _maximum_check(
            "temporal-flow",
            flow_rmse,
            criteria.temporal_flow_rmse_max,
            "normalized temporal optical-flow RMSE",
        ),
    ]
    plot = {
        "position velocity": _normalized_series(
            derivative_errors["velocity_position"],
            criteria.position_velocity_rmse_max,
        ),
        "scale velocity": _normalized_series(
            derivative_errors["velocity_scale"],
            criteria.scale_velocity_rmse_max,
        ),
        "angular velocity": _normalized_series(
            derivative_errors["velocity_rotation"],
            criteria.angular_velocity_rmse_degrees_max,
        ),
        "position acceleration": _normalized_series(
            derivative_errors["acceleration_position"],
            criteria.position_acceleration_rmse_max,
        ),
        "scale acceleration": _normalized_series(
            derivative_errors["acceleration_scale"],
            criteria.scale_acceleration_rmse_max,
        ),
        "angular acceleration": _normalized_series(
            derivative_errors["acceleration_rotation"],
            criteria.angular_acceleration_rmse_degrees_max,
        ),
        "jerk max channels": _normalized_series(
            [
                max(
                    abs(derivative_errors["jerk_position"][index]),
                    abs(derivative_errors["jerk_scale"][index]),
                    abs(derivative_errors["jerk_rotation"][index]),
                )
                for index in range(len(pairs))
            ],
            criteria.jerk_rmse_max,
        ),
    }
    return measured, checks, plot


def _perceptual(
    protocol: ReferenceComparisonProtocolV2,
    geometric: dict[str, Any],
    temporal: dict[str, Any],
    diagnostics: dict[str, Any],
    vmaf: float,
    ssim: float,
    psnr: float,
) -> tuple[dict[str, Any], list[CheckResult]]:
    criteria = protocol.criteria.perceptual
    aligned = protocol.algorithms["aligned_frames"]
    unexpected_black = int(diagnostics["unexpected_black_frames"])
    blur_ratio = float(diagnostics["blur_ratio_p10"])
    edge_error = float(geometric["visible_boundary_rmse_normalized"])
    temporal_consistency = float(temporal["temporal_consistency_rmse"])
    measured = {
        "vmaf_mean": vmaf,
        "ssim_all": ssim,
        "psnr_average_db": psnr,
        "psnr_role": "diagnostic_only",
        "aligned_luma_rmse_normalized": diagnostics["aligned_luma_rmse_normalized"],
        "temporal_consistency_rmse": temporal_consistency,
        "unexpected_black_frames": unexpected_black,
        "blur_ratio_p10": blur_ratio,
        "visible_boundary_rmse_normalized": edge_error,
        "thresholds": {
            **criteria.model_dump(mode="json"),
            "unexpected_black_frames_max": aligned["unexpected_black_frames_max"],
            "blur_ratio_p10_min": aligned["blur_ratio_p10_min"],
            "temporal_consistency_rmse_max": (protocol.criteria.temporal.temporal_flow_rmse_max),
        },
    }
    checks = [
        _minimum_check(
            "perceptual-vmaf",
            vmaf,
            criteria.vmaf_mean_min,
            "mean VMAF",
        ),
        _minimum_check(
            "perceptual-ssim",
            ssim,
            criteria.ssim_all_min,
            "aggregate SSIM",
        ),
        _maximum_check(
            "perceptual-black-frames",
            float(unexpected_black),
            float(aligned["unexpected_black_frames_max"]),
            "unexpected black-frame count",
        ),
        _maximum_check(
            "perceptual-edge-exposure",
            edge_error,
            protocol.criteria.geometric.visible_boundary_rmse_normalized_max,
            "edge-exposure boundary error",
        ),
        _minimum_check(
            "perceptual-unwanted-blur",
            blur_ratio,
            float(aligned["blur_ratio_p10_min"]),
            "candidate/reference sharpness ratio p10",
        ),
        _maximum_check(
            "perceptual-temporal-consistency",
            temporal_consistency,
            protocol.criteria.temporal.temporal_flow_rmse_max,
            "temporal decoded-frame consistency RMSE",
        ),
    ]
    return measured, checks


def _lineage(
    protocol: ReferenceComparisonProtocolV2,
    lineage_path: Path,
    candidate_path: Path,
    reference_path: Path,
    *,
    allow_synthetic_identity_fixture: bool,
) -> dict[str, Any]:
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    inputs = payload.get("input_hashes")
    forbidden = payload.get("forbidden_hashes_present")
    serialized = json.dumps(payload, sort_keys=True)
    identity_is_synthetic = allow_synthetic_identity_fixture and sha256_file(
        candidate_path
    ) == sha256_file(reference_path)
    valid = (
        inputs == [protocol.source_sha256]
        and forbidden == []
        and protocol.reference_sha256 not in serialized
        and (candidate_path != reference_path or identity_is_synthetic)
    )
    return {
        "path": str(lineage_path),
        "input_hashes": inputs,
        "forbidden_hashes_present": forbidden,
        "expected_only_visual_source_hash": protocol.source_sha256,
        "forbidden_reference_hash": protocol.reference_sha256,
        "reference_hash_present_anywhere": protocol.reference_sha256 in serialized,
        "candidate_is_reference": sha256_file(candidate_path) == sha256_file(reference_path),
        "synthetic_identity_fixture": identity_is_synthetic,
        "valid": valid,
    }


def _editorial(
    protocol: ReferenceComparisonProtocolV2,
    candidate_sha256: str,
    review: EditorialReviewV2 | None,
    required_artifacts: dict[str, Path],
) -> tuple[dict[str, Any], CheckResult]:
    required = set(protocol.human_rubric)
    artifacts_exist = all(path.is_file() for path in required_artifacts.values())
    if review is None:
        measured = {
            "required": True,
            "required_verdict": protocol.criteria.editorial.required_verdict,
            "rubric": protocol.human_rubric,
            "review": None,
            "artifacts": {name: str(path) for name, path in required_artifacts.items()},
        }
        return measured, CheckResult(
            id="editorial-human-review",
            status="warn",
            summary="objective comparison is complete; human editorial verdict is pending",
            measured=measured,
        )
    actual = {item.criterion for item in review.rubric}
    reviewed_tokens = set(review.reviewed_artifacts)
    reviewed_paths = {
        str(Path(item).expanduser().resolve(strict=False))
        for item in review.reviewed_artifacts
    }
    reviewed_artifact_roles = {
        role
        for role, path in required_artifacts.items()
        if (
            role in reviewed_tokens
            or str(path) in reviewed_tokens
            or str(path.expanduser().resolve(strict=False)) in reviewed_paths
        )
    }
    passed = (
        review.protocol_id == protocol.id
        and review.candidate_sha256 == candidate_sha256
        and actual == required
        and len(review.rubric) == len(required)
        and all(item.status == "pass" for item in review.rubric)
        and review.verdict == protocol.criteria.editorial.required_verdict
        and artifacts_exist
        and set(required_artifacts) <= reviewed_artifact_roles
    )
    measured = {
        "required": True,
        "required_verdict": protocol.criteria.editorial.required_verdict,
        "rubric": [item.model_dump(mode="json") for item in review.rubric],
        "review_id": review.id,
        "reviewer": review.reviewer,
        "reviewer_role": review.reviewer_role,
        "verdict": review.verdict,
        "reviewed_artifacts": review.reviewed_artifacts,
        "reviewed_artifact_roles": sorted(reviewed_artifact_roles),
        "required_artifacts": {
            role: str(path) for role, path in required_artifacts.items()
        },
        "artifacts_exist": artifacts_exist,
    }
    return measured, _check(
        "editorial-human-review",
        passed,
        "human operator passed every frozen editorial axis"
        if passed
        else "human editorial review is incomplete or requires revision",
        measured,
    )


def _audio_measurement(candidate: Path) -> tuple[dict[str, Any], list[str]]:
    _, metadata, _, _ = ffprobe(candidate)
    if not metadata.has_audio:
        return {
            "has_audio_stream": False,
            "codec": None,
            "sample_rate": None,
            "channels": None,
            "mean_volume_db": None,
            "silent_below_minus_60_db": True,
        }, []
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(candidate),
        "-af",
        "volumedetect",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    result = _run(command)
    match = MEAN_VOLUME.search(result.stderr)
    mean_volume = _db(match.group(1)) if match else None
    return {
        "has_audio_stream": True,
        "codec": metadata.audio_codec,
        "sample_rate": metadata.sample_rate,
        "channels": metadata.channels,
        "mean_volume_db": mean_volume,
        "silent_below_minus_60_db": (mean_volume is not None and mean_volume <= -60.0),
    }, command


def _frame_pairs(
    spec: ReferenceReconstructionSpecV2,
    recovery: MotionRecoveryReport,
) -> list[tuple[ReferenceMotionFrameV2, MotionRecoveryFrame]]:
    expected = {item.frame: item for item in spec.motion_frames}
    actual = {item.frame: item for item in recovery.frames}
    indexes = list(range(spec.duration_frames))
    if set(expected) != set(indexes) or set(actual) != set(indexes):
        raise ComparisonV2Error("all-frame motion sets do not align with the frozen duration")
    return [(expected[index], actual[index]) for index in indexes]


def _overshoot_detected(
    pairs: list[tuple[ReferenceMotionFrameV2, MotionRecoveryFrame]],
    protocol: ReferenceComparisonProtocolV2,
) -> bool:
    actual_scale = [_required(item.scale_relative_to_contain, "scale") for _, item in pairs]
    actual_rotation = [_required(item.rotation_degrees, "rotation") for _, item in pairs]
    actual_center = np.asarray(
        [
            (
                _required(item.center_x, "center_x"),
                _required(item.center_y, "center_y"),
            )
            for _, item in pairs
        ],
        dtype=np.float64,
    )
    displacement = actual_center[-1] - actual_center[0]
    norm = float(np.dot(displacement, displacement))
    center_progress = (
        ((actual_center - actual_center[0]) @ displacement) / norm
        if norm > 1e-12
        else np.zeros(len(actual_center))
    )
    return (
        _outside_endpoints(
            actual_scale,
            protocol.criteria.geometric.all_frame_scale_rmse_max,
        )
        or _outside_endpoints(
            actual_rotation,
            protocol.criteria.geometric.all_frame_rotation_rmse_degrees_max,
        )
        or bool(
            np.any(
                center_progress < -protocol.criteria.geometric.all_frame_center_rmse_normalized_max
            )
            or np.any(
                center_progress
                > 1.0 + protocol.criteria.geometric.all_frame_center_rmse_normalized_max
            )
        )
    )


def _outside_endpoints(values: list[float], tolerance: float) -> bool:
    low = min(values[0], values[-1]) - tolerance
    high = max(values[0], values[-1]) + tolerance
    return min(values) < low or max(values) > high


def expected_value(
    order: str,
    frame: ReferenceMotionFrameV2,
    channel: str,
) -> float:
    return float(getattr(frame, order)[channel])


def actual_value(order: str, frame: MotionRecoveryFrame, channel: str) -> float:
    try:
        return float(getattr(frame, order)[channel])
    except KeyError as error:
        raise ComparisonV2Error(f"candidate {order} lacks required channel {channel}") from error


def _normalized_boundary(
    matrix: list[list[float]],
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> list[list[float]]:
    corners = np.asarray(
        [
            [0.0, 0.0],
            [float(source_width), 0.0],
            [float(source_width), float(source_height)],
            [0.0, float(source_height)],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(
        corners,
        np.asarray(matrix, dtype=np.float64),
    ).reshape(-1, 2)
    return [[float(item[0] / output_width), float(item[1] / output_height)] for item in projected]


def _projected_grid_rmse(
    expected_matrix: list[list[float]],
    actual_matrix: list[list[float]],
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> float:
    points = np.asarray(
        [
            [0.0, 0.0],
            [source_width / 2.0, 0.0],
            [float(source_width), 0.0],
            [0.0, source_height / 2.0],
            [source_width / 2.0, source_height / 2.0],
            [float(source_width), source_height / 2.0],
            [0.0, float(source_height)],
            [source_width / 2.0, float(source_height)],
            [float(source_width), float(source_height)],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    expected = cv2.perspectiveTransform(
        points,
        np.asarray(expected_matrix, dtype=np.float64),
    ).reshape(-1, 2)
    actual = cv2.perspectiveTransform(
        points,
        np.asarray(actual_matrix, dtype=np.float64),
    ).reshape(-1, 2)
    differences = (expected - actual) / np.asarray([output_width, output_height])
    return math.sqrt(float(np.mean(np.sum(np.square(differences), axis=1))))


def _point_rmse(expected: list[list[float]], actual: list[list[float]]) -> float:
    differences = np.asarray(expected, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return math.sqrt(float(np.mean(np.sum(np.square(differences), axis=1))))


def _scaled_gray(frame: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    height = max(2, round(frame.shape[0] * width / frame.shape[1]))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cast(NDArray[np.uint8], cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))


def _flow(
    previous: NDArray[np.uint8],
    current: NDArray[np.uint8],
) -> NDArray[np.float32]:
    return cast(
        NDArray[np.float32],
        cast(Any, cv2.calcOpticalFlowFarneback)(
            previous,
            current,
            None,
            0.5,
            3,
            21,
            4,
            7,
            1.5,
            0,
        ),
    )


def _metric_geometry(width: int, height: int) -> tuple[int, int]:
    bounded_width = min(width, int(ALGORITHMS["perceptual"]["metric_width_max"]))
    bounded_height = round(height * bounded_width / width)
    bounded_height -= bounded_height % 2
    return bounded_width, max(2, bounded_height)


def _vmaf(
    candidate: Path,
    reference: Path,
    width: int,
    height: int,
    output: Path,
) -> tuple[float, list[str]]:
    command = _metric_command(
        candidate,
        reference,
        width,
        height,
        "libvmaf=log_fmt=json:log_path=" + str(output),
    )
    result = _run(command)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        try:
            return float(payload["pooled_metrics"]["vmaf"]["mean"]), command
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonV2Error("libvmaf JSON has no pooled mean") from error
    match = VMAF_SCORE.search(result.stderr)
    if match is None:
        raise ComparisonV2Error("libvmaf produced no score")
    return float(match.group(1)), command


def _ssim(
    candidate: Path,
    reference: Path,
    width: int,
    height: int,
    output: Path,
) -> tuple[float, list[str]]:
    command = _metric_command(
        candidate,
        reference,
        width,
        height,
        "ssim=stats_file=" + str(output),
    )
    result = _run(command)
    match = SSIM_ALL.search(result.stderr)
    if match is None:
        raise ComparisonV2Error("SSIM produced no aggregate score")
    return float(match.group(1)), command


def _psnr(
    candidate: Path,
    reference: Path,
    width: int,
    height: int,
    output: Path,
) -> tuple[float, list[str]]:
    command = _metric_command(
        candidate,
        reference,
        width,
        height,
        "psnr=stats_file=" + str(output),
    )
    result = _run(command)
    match = PSNR_AVERAGE.search(result.stderr)
    if match is None:
        raise ComparisonV2Error("PSNR produced no aggregate score")
    value = match.group(1)
    return (999.0 if value.lower() == "inf" else float(value)), command


def _metric_command(
    candidate: Path,
    reference: Path,
    width: int,
    height: int,
    terminal: str,
) -> list[str]:
    graph = (
        f"[0:v]scale={width}:{height}:flags=lanczos,settb=AVTB,setpts=PTS-STARTPTS[d];"
        f"[1:v]scale={width}:{height}:flags=lanczos,settb=AVTB,setpts=PTS-STARTPTS[r];"
        f"[d][r]{terminal}"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(candidate),
        "-i",
        str(reference),
        "-lavfi",
        graph,
        "-an",
        "-f",
        "null",
        "-",
    ]


def _side_by_side(
    reference: Path,
    candidate: Path,
    output: Path,
    width: int,
    height: int,
) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(reference),
        "-i",
        str(candidate),
        "-filter_complex",
        (
            f"[0:v]scale={width}:{height}:flags=lanczos,setpts=PTS-STARTPTS[left];"
            f"[1:v]scale={width}:{height}:flags=lanczos,setpts=PTS-STARTPTS[right];"
            "[left][right]hstack=inputs=2[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(output),
    ]
    _run(command)
    return command


def _difference_video(
    reference: Path,
    candidate: Path,
    output: Path,
    width: int,
    height: int,
) -> list[str]:
    gain = float(ALGORITHMS["artifacts"]["difference_gain"])
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(reference),
        "-i",
        str(candidate),
        "-filter_complex",
        (
            f"[0:v]scale={width}:{height}:flags=lanczos,setpts=PTS-STARTPTS[r];"
            f"[1:v]scale={width}:{height}:flags=lanczos,setpts=PTS-STARTPTS[c];"
            f"[r][c]blend=all_mode=difference,eq=contrast={gain},format=yuv420p[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-shortest",
        str(output),
    ]
    _run(command)
    return command


def _phase_frames(spec: ReferenceReconstructionSpecV2) -> list[int]:
    coupling = spec.phase_model["channel_coupling"]
    selected = [
        0,
        int(spec.phase_model["onset_frame"]),
        int(coupling["start_frame"]),
        int(coupling["maximum_phase_offset_frame"]),
        int(coupling["end_frame"]),
        int(spec.phase_model["settle_frame"]),
        spec.duration_frames - 1,
    ]
    return list(dict.fromkeys(min(spec.duration_frames - 1, max(0, item)) for item in selected))


def _phase_contact_sheet(
    reference_path: Path,
    candidate_path: Path,
    output_path: Path,
    frames: list[int],
) -> None:
    panels: list[NDArray[np.uint8]] = []
    for frame_index in frames:
        reference = _read_frame(reference_path, frame_index)
        candidate = _read_frame(candidate_path, frame_index)
        width = 180
        height = max(2, round(reference.shape[0] * width / reference.shape[1]))
        reference = cast(
            NDArray[np.uint8],
            cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA),
        )
        candidate = cast(
            NDArray[np.uint8],
            cv2.resize(candidate, (width, height), interpolation=cv2.INTER_AREA),
        )
        difference = cast(NDArray[np.uint8], cv2.absdiff(reference, candidate))
        difference = cast(
            NDArray[np.uint8],
            np.asarray(
                np.clip(difference.astype(np.float32) * 3.0, 0, 255),
                dtype=np.uint8,
            ),
        )
        panel = cast(NDArray[np.uint8], np.hstack((reference, candidate, difference)))
        cv2.putText(
            panel,
            f"frame {frame_index}: reference | candidate | 3x difference",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    sheet = np.vstack(panels)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ComparisonV2Error(f"cannot write phase contact sheet: {output_path}")


def _read_frame(path: Path, frame_index: int) -> NDArray[np.uint8]:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        available, frame = capture.read()
    finally:
        capture.release()
    if not available:
        raise ComparisonV2Error(f"cannot decode frame {frame_index} from {path}")
    return cast(NDArray[np.uint8], frame)


def _error_plot(
    path: Path,
    series: dict[str, list[float]],
    *,
    title: str,
) -> None:
    width, height = 1200, 520
    margin = 70
    image = Image.new("RGB", (width, height), "#11151c")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 20), title, fill="#f1f5f9")
    plot_left, plot_top = margin, 55
    plot_right, plot_bottom = width - 25, height - 55
    draw.rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        outline="#4b5563",
        width=1,
    )
    maximum = max(2.0, max(max(values, default=0.0) for values in series.values()))

    def y(value: float) -> int:
        return round(
            plot_bottom - min(maximum, max(0.0, value)) / maximum * (plot_bottom - plot_top)
        )

    threshold_y = y(1.0)
    draw.line((plot_left, threshold_y, plot_right, threshold_y), fill="#f87171", width=2)
    draw.text((8, threshold_y - 7), "1.0", fill="#f87171")
    palette = ["#60a5fa", "#34d399", "#fbbf24", "#c084fc", "#fb7185", "#22d3ee", "#a3e635"]
    for index, (name, values) in enumerate(series.items()):
        color = palette[index % len(palette)]
        if len(values) == 1:
            points = [(plot_left, y(values[0]))]
        else:
            points = [
                (
                    round(plot_left + item_index / (len(values) - 1) * (plot_right - plot_left)),
                    y(value),
                )
                for item_index, value in enumerate(values)
            ]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        draw.text((margin + (index % 3) * 360, height - 40 + (index // 3) * 15), name, fill=color)
    image.save(path)


def _validate_check_coverage(
    protocol: ReferenceComparisonProtocolV2,
    checks: list[CheckResult],
) -> None:
    actual = [item.id for item in checks]
    if actual != protocol.mandatory_checks:
        missing = sorted(set(protocol.mandatory_checks) - set(actual))
        extra = sorted(set(actual) - set(protocol.mandatory_checks))
        raise ComparisonV2Error(
            f"comparison mandatory check coverage drifted: missing={missing}, extra={extra}"
        )


def _combined_overall(
    objective_overall: Literal["pass", "fail"],
    editorial_check: CheckResult,
) -> Literal["pass", "fail", "warn"]:
    if objective_overall == "fail" or editorial_check.status == "fail":
        return "fail"
    if editorial_check.status == "pass":
        return "pass"
    return "warn"


def _normalized_series(values: list[float], maximum: float) -> list[float]:
    return [abs(float(value)) / maximum for value in values]


def _required(value: float | None, name: str) -> float:
    if value is None:
        raise ComparisonV2Error(f"candidate motion lacks required {name}")
    return float(value)


def _rmse(values: list[float]) -> float:
    if not values:
        raise ComparisonV2Error("cannot calculate RMSE for an empty series")
    return math.sqrt(sum(value * value for value in values) / len(values))


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


def _maximum_check(
    check_id: str,
    value: float,
    maximum: float,
    label: str,
) -> CheckResult:
    return _check(
        check_id,
        value <= maximum,
        f"{label} is within the frozen maximum",
        {"actual": value, "maximum": maximum},
    )


def _minimum_check(
    check_id: str,
    value: float,
    minimum: float,
    label: str,
) -> CheckResult:
    return _check(
        check_id,
        value >= minimum,
        f"{label} meets the frozen minimum",
        {"actual": value, "minimum": minimum},
    )


def _db(value: str) -> float:
    return -999.0 if value.lower() == "-inf" else float(value)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise ComparisonV2Error(result.stderr[-5000:] or f"{command[0]} failed")
    return result


def _write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
