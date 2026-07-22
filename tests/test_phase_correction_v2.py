from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aoa_editing.analysis.motion import (
    _curve_driver,
    _phase_boundaries,
    characterize_similarity_coupling,
    local_polynomial_derivatives,
)
from aoa_editing.analysis.phase_correction import (
    PhaseCorrectionV2Error,
    derive_phase_correction_v2,
)
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    CheckResult,
    FrameRate,
    MotionRecoveryFrame,
    MotionRecoveryReport,
    Provenance,
    ReferenceAudioStructure,
    ReferenceComparisonReportV2,
    ReferenceMotionFrameV2,
    ReferenceReconstructionSpecV2,
)
from aoa_editing.evals.reconstruction_v2 import run_reconstruction_pass_v2
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.scenarios.reference_v2 import evaluate_time_warp_v2

SOURCE_HASH = "1" * 64
REFERENCE_HASH = "2" * 64
CANDIDATE_HASH = "3" * 64
REVISION = "a" * 40


def _provenance() -> Provenance:
    return Provenance(tool="synthetic-phase-fixture", tool_version="1")


def _matrix(
    center_x: float,
    center_y: float,
    scale: float,
    rotation: float,
) -> list[list[float]]:
    angle = math.radians(rotation)
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    return [
        [cosine, -sine, center_x * 90],
        [sine, cosine, center_y * 160],
        [0.0, 0.0, 1.0],
    ]


def _derivatives(values: np.ndarray) -> dict[str, np.ndarray]:
    derived = local_polynomial_derivatives(values, fps=25.0, window=15, degree=3)
    return {
        "velocity": np.asarray(derived.velocity),
        "acceleration": np.asarray(derived.acceleration),
        "jerk": np.asarray(derived.jerk),
    }


def _fixture() -> tuple[
    ReferenceReconstructionSpecV2,
    ReferenceComparisonReportV2,
    MotionRecoveryReport,
]:
    count = 81
    indices = np.arange(count, dtype=np.float64)
    similarity = np.clip((indices - 6.0) / 68.0, 0.0, 1.0)
    center_progress = np.clip((indices - 12.0) / 64.0, 0.0, 1.0)
    expected = {
        "scale": 2.4 - 1.1 * similarity,
        "rotation": -7.0 + 7.0 * similarity,
        "center_x": 0.30 + 0.20 * center_progress,
        "center_y": (
            0.62
            - 0.12 * center_progress
            + 0.008 * np.sin(np.pi * center_progress)
        ),
    }
    expected_derivatives = {
        name: _derivatives(values) for name, values in expected.items()
    }
    expected_phase = _phase_boundaries(
        _curve_driver(
            {
                "scale": expected["scale"],
                "rotation": expected["rotation"],
                "center_x": expected["center_x"],
                "center_y": expected["center_y"],
            }
        )
    )
    expected_coupling = asdict(
        characterize_similarity_coupling(
            center_x=expected["center_x"],
            center_y=expected["center_y"],
            scale=expected["scale"],
            rotation=expected["rotation"],
            fps=25.0,
        )
    )
    expected_coupling["start_frame"] = 20
    expected_coupling["end_frame"] = 60

    motion_frames = []
    for frame in range(count):
        motion_frames.append(
            ReferenceMotionFrameV2(
                frame=frame,
                time_seconds=frame / 25.0,
                selected_model="similarity",
                matrix_3x3=_matrix(
                    expected["center_x"][frame],
                    expected["center_y"][frame],
                    expected["scale"][frame],
                    expected["rotation"][frame],
                ),
                visible_source_boundary=[
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                center_x=expected["center_x"][frame],
                center_y=expected["center_y"][frame],
                scale_relative_to_contain=expected["scale"][frame],
                rotation_degrees=expected["rotation"][frame],
                confidence=0.99,
                confidence_interval={
                    "center_normalized_radius": 0.001,
                    "scale_relative_radius": 0.001,
                    "rotation_degrees_radius": 0.01,
                },
                match_count=100,
                inlier_count=99,
                inlier_ratio=0.99,
                reprojection_rmse=0.1,
                residual_flow_mean=0.1,
                residual_flow_p95=0.2,
                residual_valid_fraction=1.0,
                velocity={
                    "center_x": expected_derivatives["center_x"]["velocity"][frame],
                    "center_y": expected_derivatives["center_y"]["velocity"][frame],
                    "scale": expected_derivatives["scale"]["velocity"][frame],
                    "rotation": expected_derivatives["rotation"]["velocity"][frame],
                },
                acceleration={
                    "center_x": expected_derivatives["center_x"]["acceleration"][frame],
                    "center_y": expected_derivatives["center_y"]["acceleration"][frame],
                    "scale": expected_derivatives["scale"]["acceleration"][frame],
                    "rotation": expected_derivatives["rotation"]["acceleration"][frame],
                },
                jerk={
                    "center_x": expected_derivatives["center_x"]["jerk"][frame],
                    "center_y": expected_derivatives["center_y"]["jerk"][frame],
                    "scale": expected_derivatives["scale"]["jerk"][frame],
                    "rotation": expected_derivatives["rotation"]["jerk"][frame],
                },
                model_scores={"similarity": 0.1, "affine": 0.1, "homography": 0.1},
                outlier=False,
                uncertainty=["pivot_translation_gauge"],
            )
        )

    criteria = {
        "threshold_revision": "reference-comparison-v2-frozen-1",
        "frozen_before_first_v2_render": True,
        "threshold_basis": {"fixture": "localized phase bias"},
        "technical": {
            "resolution": [90, 160],
            "frame_rate": {"numerator": 25, "denominator": 1},
            "frame_count_tolerance": 0,
            "audio_structure_exact": True,
        },
        "geometric": {
            "all_frame_center_rmse_normalized_max": 0.02,
            "all_frame_scale_rmse_max": 0.08,
            "all_frame_rotation_rmse_degrees_max": 0.8,
            "visible_boundary_rmse_normalized_max": 0.03,
            "homography_corner_rmse_normalized_max": 0.03,
        },
        "temporal": {
            "position_velocity_rmse_max": 0.2,
            "scale_velocity_rmse_max": 1.0,
            "angular_velocity_rmse_degrees_max": 8.0,
            "position_acceleration_rmse_max": 2.0,
            "scale_acceleration_rmse_max": 8.0,
            "angular_acceleration_rmse_degrees_max": 60.0,
            "jerk_rmse_max": 500.0,
            "phase_onset_tolerance_frames": 2,
            "phase_settle_tolerance_frames": 2,
            "temporal_flow_rmse_max": 1.0,
        },
        "perceptual": {
            "vmaf_mean_min": 90.0,
            "ssim_all_min": 0.97,
            "psnr_diagnostic_only": True,
            "no_black_frames": True,
            "no_edge_exposure": True,
            "no_unwanted_blur": True,
        },
        "editorial": {
            "human_review_required": True,
            "rubric": [
                "smoothness",
                "weight",
                "natural_acceleration",
                "rotation_character",
                "twist_character",
                "settle_timing",
                "absence_of_mechanical_linearity",
                "compositional_match",
            ],
            "required_verdict": "perceptually_and_editorially_indistinguishable",
        },
        "lineage": {
            "only_visual_source_hash": SOURCE_HASH,
            "reference_hash_forbidden": REFERENCE_HASH,
        },
    }
    spec = ReferenceReconstructionSpecV2(
        evidence_id="referencemotion_" + "b" * 32,
        source_sha256=SOURCE_HASH,
        reference_sha256=REFERENCE_HASH,
        readiness_revision=REVISION,
        motion_gate_revision=REVISION,
        duration_frames=count,
        duration_seconds=count / 25.0,
        frame_rate=FrameRate(numerator=25),
        width=90,
        height=160,
        audio=ReferenceAudioStructure(
            has_audio_stream=True,
            silent_for_full_duration=True,
        ),
        motion_model="similarity",
        motion_frames=motion_frames,
        curve_representation={"kind": "dense_observed_matrix_curve"},
        phase_model={
            "onset_frame": expected_phase["onset"],
            "peak_velocity_frame": expected_phase["peak_velocity"],
            "settle_frame": expected_phase["settle"],
            "overshoot_detected": False,
            "channel_coupling": expected_coupling,
            "twist_candidate": {
                "start_frame": 20,
                "peak_frame": 40,
                "end_frame": 60,
            },
        },
        transform_order={"supported_model": "similarity"},
        uncertainties=["pivot_translation_gauge"],
        alternative_explanations=[],
        comparison_criteria=criteria,
        artifacts={},
        provenance=_provenance(),
    )

    biased_source_frames = np.interp(
        indices,
        [0.0, 10.0, 16.0, 65.0, 70.0, 80.0],
        [0.0, 5.0, 16.0, 65.0, 75.0, 80.0],
    )
    candidate = {
        name: np.interp(biased_source_frames, indices, values)
        for name, values in expected.items()
    }
    candidate_derivatives = {
        name: _derivatives(values) for name, values in candidate.items()
    }
    candidate_phase = _phase_boundaries(
        _curve_driver(
            {
                "scale": candidate["scale"],
                "rotation": candidate["rotation"],
                "center_x": candidate["center_x"],
                "center_y": candidate["center_y"],
            }
        )
    )
    recovery_frames = []
    for frame in range(count):
        recovery_frames.append(
            MotionRecoveryFrame(
                frame=frame,
                time_seconds=frame / 25.0,
                selected_model="similarity",
                matrix_3x3=_matrix(
                    candidate["center_x"][frame],
                    candidate["center_y"][frame],
                    candidate["scale"][frame],
                    candidate["rotation"][frame],
                ),
                scale_relative_to_contain=candidate["scale"][frame],
                rotation_degrees=candidate["rotation"][frame],
                center_x=candidate["center_x"][frame],
                center_y=candidate["center_y"][frame],
                affine_anisotropy=0.0,
                affine_shear=0.0,
                perspective_strength=0.0,
                match_count=100,
                inlier_count=99,
                inlier_ratio=0.99,
                reprojection_rmse=0.1,
                confidence=0.99,
                model_scores={"similarity": 0.1, "affine": 0.1, "homography": 0.1},
                velocity={
                    name: candidate_derivatives[key]["velocity"][frame]
                    for name, key in (
                        ("center_x", "center_x"),
                        ("center_y", "center_y"),
                        ("scale", "scale"),
                        ("rotation", "rotation"),
                    )
                },
                acceleration={
                    name: candidate_derivatives[key]["acceleration"][frame]
                    for name, key in (
                        ("center_x", "center_x"),
                        ("center_y", "center_y"),
                        ("scale", "scale"),
                        ("rotation", "rotation"),
                    )
                },
                jerk={
                    name: candidate_derivatives[key]["jerk"][frame]
                    for name, key in (
                        ("center_x", "center_x"),
                        ("center_y", "center_y"),
                        ("scale", "scale"),
                        ("rotation", "rotation"),
                    )
                },
            )
        )
    recovery = MotionRecoveryReport(
        source_sha256=SOURCE_HASH,
        video_sha256=CANDIDATE_HASH,
        frame_count=count,
        analyzed_frame_count=count,
        frame_rate=FrameRate(numerator=25),
        width=90,
        height=160,
        selected_model="similarity",
        model_distribution={"similarity": count},
        curve_hint="linear",
        curve_confidence=0.99,
        mean_confidence=0.99,
        model_mismatch=False,
        outlier_frames=[],
        phase_boundaries=candidate_phase,
        uncertainties=[],
        alternative_explanations=[],
        frames=recovery_frames,
        provenance=_provenance(),
    )
    comparison = ReferenceComparisonReportV2(
        protocol_id="comparisonprotocol_" + "c" * 32,
        protocol_path="comparison-protocol-v2.json",
        protocol_sha256="4" * 64,
        spec_id=spec.id,
        spec_sha256="5" * 64,
        source_sha256=SOURCE_HASH,
        reference_sha256=REFERENCE_HASH,
        candidate_sha256=CANDIDATE_HASH,
        candidate_path="candidate.mp4",
        technical={},
        geometric={},
        temporal={
            "phase": {
                "expected_onset": expected_phase["onset"],
                "actual_onset": candidate_phase["onset"],
                "expected_settle": expected_phase["settle"],
                "actual_settle": candidate_phase["settle"],
            }
        },
        perceptual={},
        editorial={},
        lineage={},
        checks=[
            CheckResult(
                id="geometric-all-frame-center",
                status="pass",
                summary="synthetic geometry passes",
            ),
            CheckResult(
                id="temporal-phase-onset",
                status="fail",
                summary="synthetic onset bias",
            ),
            CheckResult(
                id="temporal-phase-duration",
                status="fail",
                summary="synthetic duration bias",
            ),
            CheckResult(
                id="temporal-phase-settle",
                status="fail",
                summary="synthetic settle bias",
            ),
            CheckResult(
                id="editorial-human-review",
                status="warn",
                summary="human review remains separate",
            ),
        ],
        artifacts={},
        provenance=_provenance(),
        objective_overall="fail",
        overall="fail",
    )
    return spec, comparison, recovery


def test_phase_correction_search_is_localized_and_preserves_coupling() -> None:
    spec, comparison, recovery = _fixture()
    correction = derive_phase_correction_v2(
        spec,
        comparison,
        recovery,
        spec_path="reference-spec-v2.json",
        spec_sha256=comparison.spec_sha256,
        comparison_path="comparison-v2.json",
        comparison_sha256="6" * 64,
        candidate_motion_path="candidate-motion-recovery.json",
        candidate_motion_sha256="7" * 64,
    )

    tolerance = spec.comparison_criteria["temporal"]
    assert abs(
        correction.predicted_phase["onset"] - spec.phase_model["onset_frame"]
    ) <= tolerance["phase_onset_tolerance_frames"]
    assert abs(
        correction.predicted_phase["settle"] - spec.phase_model["settle_frame"]
    ) <= tolerance["phase_settle_tolerance_frames"]
    assert all(item.status == "pass" for item in correction.predicted_checks)
    protected = correction.protected_coupling_range
    for frame in range(protected.start, protected.end):
        assert evaluate_time_warp_v2(correction, frame) == pytest.approx(frame)
    assert any(
        evaluate_time_warp_v2(correction, frame) != pytest.approx(frame)
        for frame in range(spec.duration_frames)
    )
    assert correction.reference_media_used_as_render_input is False
    assert correction.candidate_media_used_as_render_input is False


def test_phase_correction_refuses_non_phase_regression() -> None:
    spec, comparison, recovery = _fixture()
    comparison = comparison.model_copy(
        update={
            "checks": [
                *comparison.checks,
                CheckResult(
                    id="perceptual-vmaf",
                    status="fail",
                    summary="unrelated regression",
                ),
            ]
        }
    )
    with pytest.raises(PhaseCorrectionV2Error, match="non-phase objective failure"):
        derive_phase_correction_v2(
            spec,
            comparison,
            recovery,
            spec_path="reference-spec-v2.json",
            spec_sha256=comparison.spec_sha256,
            comparison_path="comparison-v2.json",
            comparison_sha256="6" * 64,
            candidate_motion_path="candidate-motion-recovery.json",
            candidate_motion_sha256="7" * 64,
        )


def test_phase_compensated_runner_uses_normal_source_only_product_path(
    tmp_path,
) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGB", (64, 64), "#13233f")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 5, 58, 57), outline="#f6c446", width=3)
    draw.ellipse((13, 10, 51, 53), fill="#d24c55")
    draw.polygon(((31, 6), (8, 56), (57, 55)), fill="#45a278")
    image.save(source)
    source_sha256 = sha256_file(source)

    spec, comparison, recovery = _fixture()
    spec_payload = spec.model_dump(mode="json")
    spec_payload["source_sha256"] = source_sha256
    spec_payload["comparison_criteria"]["lineage"]["only_visual_source_hash"] = (
        source_sha256
    )
    spec = ReferenceReconstructionSpecV2.model_validate(spec_payload)
    spec_path = tmp_path / "reference-spec-v2.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    spec_sha256 = sha256_file(spec_path)

    recovery = recovery.model_copy(update={"source_sha256": source_sha256})
    recovery_path = tmp_path / "candidate-motion-recovery.json"
    recovery_path.write_text(
        recovery.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = comparison.model_copy(
        update={
            "spec_id": spec.id,
            "spec_sha256": spec_sha256,
            "source_sha256": source_sha256,
            "artifacts": {"candidate_motion": str(recovery_path)},
        }
    )
    comparison_path = tmp_path / "comparison-v2.json"
    comparison_path.write_text(
        comparison.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    receipt = run_reconstruction_pass_v2(
        spec_path,
        source,
        tmp_path / "corrected-pass",
        pass_kind="phase_compensated_matrix",
        comparison_path=comparison_path,
        settings=Settings.for_home(tmp_path / "editing-home"),
    )

    assert receipt.overall == "pass"
    assert receipt.sequence == "5r"
    assert receipt.phase_correction is not None
    assert receipt.phase_correction_path is not None
    assert receipt.phase_correction_evidence is not None
    assert Path(receipt.phase_correction_path).is_file()
    assert Path(receipt.phase_correction_evidence).is_file()
    assert receipt.profiles["final"]["lineage"]["input_hashes"] == [source_sha256]
    assert receipt.profiles["final"]["lineage"]["forbidden_hashes_present"] == []
