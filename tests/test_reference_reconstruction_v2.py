from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Asset,
    CheckResult,
    DecomposedTransformMotionV2,
    EvidenceRecord,
    FrameRange,
    FrameRate,
    MatrixTransformMotionV2,
    MediaKind,
    MediaMetadata,
    ProjectVersion,
    Provenance,
    ReferenceAudioStructure,
    ReferenceMotionFrameV2,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionSpecV2,
    Timeline,
    TimeWarpAnchorV2,
    TransformEffectV2,
)
from aoa_editing.domain.motion_v2 import evaluate_transform_matrix
from aoa_editing.evals.clean_rerun_v2 import run_clean_rerun_v2
from aoa_editing.evals.reconstruction_v2 import (
    ReconstructionV2Error,
    run_reconstruction_pass_v2,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.scenarios.reference_v2 import (
    classify_reconstruction_hypotheses_v2,
    reference_reconstruct_v2,
)

SOURCE_HASH = "1" * 64
REFERENCE_HASH = "2" * 64
REVISION = "a" * 40


def _provenance() -> Provenance:
    return Provenance(tool="synthetic", tool_version="1", deterministic=True)


def _matrix(
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    center_x: float,
    center_y: float,
    scale: float,
    rotation_degrees: float,
) -> list[list[float]]:
    fit = min(output_width / source_width, output_height / source_height)
    angle = math.radians(rotation_degrees)
    linear = fit * scale * np.asarray(
        (
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        ),
        dtype=np.float64,
    )
    pivot = np.asarray((source_width / 2, source_height / 2), dtype=np.float64)
    target = np.asarray((center_x * output_width, center_y * output_height))
    translation = target - linear @ pivot
    return [
        [float(linear[0, 0]), float(linear[0, 1]), float(translation[0])],
        [float(linear[1, 0]), float(linear[1, 1]), float(translation[1])],
        [0.0, 0.0, 1.0],
    ]


def _spec(source_hash: str = SOURCE_HASH, *, duration: int = 9) -> ReferenceReconstructionSpecV2:
    source_width = 64
    source_height = 64
    width = 90
    height = 160
    frames: list[ReferenceMotionFrameV2] = []
    for frame in range(duration):
        progress = frame / (duration - 1)
        smooth = progress * progress * (3 - 2 * progress)
        center_x = 0.30 + 0.20 * smooth + 0.008 * math.sin(math.pi * progress)
        center_y = 0.62 - 0.12 * smooth + 0.004 * math.sin(2 * math.pi * progress)
        scale = 1.8 - 0.8 * smooth
        rotation = -6.0 + 6.0 * smooth
        frames.append(
            ReferenceMotionFrameV2(
                frame=frame,
                time_seconds=frame / 25,
                selected_model="similarity",
                matrix_3x3=_matrix(
                    source_width=source_width,
                    source_height=source_height,
                    output_width=width,
                    output_height=height,
                    center_x=center_x,
                    center_y=center_y,
                    scale=scale,
                    rotation_degrees=rotation,
                ),
                visible_source_boundary=[
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                center_x=center_x,
                center_y=center_y,
                scale_relative_to_contain=scale,
                rotation_degrees=rotation,
                confidence=0.98,
                confidence_interval={
                    "center_normalized_radius": 0.0002,
                    "scale_relative_radius": 0.001,
                    "rotation_degrees_radius": 0.04,
                },
                match_count=100,
                inlier_count=98,
                inlier_ratio=0.98,
                reprojection_rmse=0.2,
                residual_flow_mean=0.1,
                residual_flow_p95=0.2,
                residual_valid_fraction=1.0,
                velocity={
                    "scale": -0.8 / ((duration - 1) / 25),
                    "rotation": 6.0 / ((duration - 1) / 25),
                    "center_x": 0.2 / ((duration - 1) / 25),
                    "center_y": -0.12 / ((duration - 1) / 25),
                },
                acceleration={
                    "scale": 0.0,
                    "rotation": 0.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                },
                jerk={
                    "scale": 0.0,
                    "rotation": 0.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                },
                model_scores={
                    "similarity": 0.2,
                    "affine": 0.199,
                    "homography": 0.198,
                },
                outlier=False,
                uncertainty=[
                    "pivot_translation_gauge",
                    "nested_global_models_close_without_material_gain",
                ],
            )
        )
    return ReferenceReconstructionSpecV2(
        evidence_id="referencemotion_" + "b" * 32,
        source_sha256=source_hash,
        reference_sha256=REFERENCE_HASH,
        readiness_revision=REVISION,
        motion_gate_revision=REVISION,
        duration_frames=duration,
        duration_seconds=duration / 25,
        frame_rate=FrameRate(numerator=25),
        width=width,
        height=height,
        audio=ReferenceAudioStructure(
            has_audio_stream=True,
            silent_for_full_duration=True,
            sample_rate=48000,
            channels=2,
        ),
        motion_model="similarity",
        motion_frames=frames,
        curve_representation={
            "kind": "dense_observed_matrix_curve",
            "sampling": "every_frame",
        },
        phase_model={
            "onset_frame": 1,
            "peak_velocity_frame": duration // 2,
            "settle_frame": duration - 1,
            "overshoot_detected": False,
            "channel_coupling": {
                "leading_explanation": (
                    "staggered_center_path_with_synchronized_scale_rotation"
                ),
                "scale_rotation_progress_rmse": 0.001,
                "center_phase_relation": "center_lags",
            },
            "twist_candidate": {
                "start_frame": 2,
                "peak_frame": duration // 2,
                "end_frame": duration - 2,
            },
        },
        transform_order={
            "observable": "per-frame source-to-output matrix",
            "conclusion": "global_matrix_observable_but_authoring_order_not_unique",
            "supported_model": "similarity",
            "pivot_translation_order_identifiable": False,
            "required_execution_rule": (
                "reproduce the frozen matrix curve; do not claim a unique authoring order"
            ),
        },
        uncertainties=[
            "Pivot is not identifiable from a single global transform.",
        ],
        alternative_explanations=[
            "Pivot and translation remain a gauge-equivalent decomposition.",
        ],
        comparison_criteria={},
        artifacts={},
        provenance=_provenance(),
    )


def _asset(source_hash: str = SOURCE_HASH) -> Asset:
    return Asset(
        sha256=source_hash,
        original_name="generic-source.png",
        media_kind=MediaKind.IMAGE,
        stored_path="assets/source.png",
        size_bytes=123,
        metadata=MediaMetadata(width=64, height=64, has_video=True),
    )


def _base(spec: ReferenceReconstructionSpecV2) -> ProjectVersion:
    return ProjectVersion(
        project_id="project_" + "c" * 32,
        message="Initial timeline",
        timeline=Timeline(
            width=spec.width,
            height=spec.height,
            frame_rate=spec.frame_rate,
            duration_frames=spec.duration_frames,
        ),
    )


def _evidence(
    spec: ReferenceReconstructionSpecV2,
    asset: Asset,
    base: ProjectVersion,
) -> EvidenceRecord:
    return EvidenceRecord(
        project_id=base.project_id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="reference.reconstruction_spec.v2",
        confidence=0.98,
        payload={"spec_id": spec.id},
        provenance=_provenance(),
    )


def _phase_correction(spec: ReferenceReconstructionSpecV2) -> ReferencePhaseCorrectionV2:
    last = spec.duration_frames - 1
    return ReferencePhaseCorrectionV2(
        spec_id=spec.id,
        spec_sha256="4" * 64,
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        comparison_report_id="comparison_" + "5" * 32,
        comparison_report_path="comparison-v2.json",
        comparison_report_sha256="6" * 64,
        candidate_motion_report_id="motionrecovery_" + "7" * 32,
        candidate_motion_path="candidate-motion-recovery.json",
        candidate_motion_sha256="8" * 64,
        prior_candidate_sha256="9" * 64,
        duration_frames=spec.duration_frames,
        anchors=[
            TimeWarpAnchorV2(output_frame=0, source_frame=0.0),
            TimeWarpAnchorV2(output_frame=1, source_frame=2.0),
            TimeWarpAnchorV2(output_frame=3, source_frame=3.0),
            TimeWarpAnchorV2(output_frame=max(3, last - 2), source_frame=float(max(3, last - 2))),
            TimeWarpAnchorV2(output_frame=last - 1, source_frame=float(last - 1.5)),
            TimeWarpAnchorV2(output_frame=last, source_frame=float(last)),
        ],
        identity_ranges=[FrameRange(start=3, duration=max(1, last - 4))],
        protected_coupling_range=FrameRange(start=3, duration=max(1, last - 4)),
        previous_phase={"onset": 3, "settle": last - 2},
        predicted_phase={"onset": 1, "settle": last - 1},
        predicted_metrics={"geometric": {}, "temporal": {}, "coupling": {}},
        predicted_checks=[
            CheckResult(
                id="predicted-phase",
                status="pass",
                summary="synthetic phase prediction passes",
            )
        ],
        failed_check_ids=[
            "temporal-phase-onset",
            "temporal-phase-duration",
            "temporal-phase-settle",
        ],
        selection={"candidate_count": 1, "selected_score": 0.1},
        provenance=_provenance(),
    )


def _phase_correction_evidence(
    correction: ReferencePhaseCorrectionV2,
    asset: Asset,
    base: ProjectVersion,
) -> EvidenceRecord:
    return EvidenceRecord(
        project_id=base.project_id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="reference.phase_correction.v2",
        confidence=1.0,
        payload={
            "correction_id": correction.id,
            "phase_correction": correction.model_dump(mode="json"),
        },
        provenance=_provenance(),
    )


def _effect(treatment: object) -> TransformEffectV2:
    tracks = treatment.patch.operations[-1].value  # type: ignore[attr-defined]
    return TransformEffectV2.model_validate(tracks[0]["clips"][0]["effects"][0])


def test_reference_v2_passes_add_only_evidence_supported_complexity() -> None:
    spec = _spec()
    asset = _asset()
    base = _base(spec)
    evidence = [_evidence(spec, asset, base)]

    timing = _effect(
        reference_reconstruct_v2(
            base.project_id,
            base,
            asset,
            evidence,
            spec,
            pass_kind="timing_easing",
        )
    )
    assert isinstance(timing.motion, DecomposedTransformMotionV2)
    assert len(timing.motion.position.keyframes) == 2
    assert timing.motion.position.keyframes[0].interpolation == "cubic_bezier"
    assert {item.value for item in timing.motion.pivot.keyframes} == {
        timing.motion.pivot.keyframes[0].value
    }

    tangents = _effect(
        reference_reconstruct_v2(
            base.project_id,
            base,
            asset,
            evidence,
            spec,
            pass_kind="continuous_tangents",
        )
    )
    assert isinstance(tangents.motion, DecomposedTransformMotionV2)
    assert tangents.motion.position.continuity == "c1"
    assert len(tangents.motion.position.keyframes) > 2
    assert all(
        item.interpolation == "cubic_hermite"
        for item in tangents.motion.position.keyframes[:-1]
    )

    pivot = _effect(
        reference_reconstruct_v2(
            base.project_id,
            base,
            asset,
            evidence,
            spec,
            pass_kind="moving_pivot_hypothesis",
        )
    )
    assert isinstance(pivot.motion, DecomposedTransformMotionV2)
    assert len({item.value for item in pivot.motion.pivot.keyframes}) > 1
    for frame in range(spec.duration_frames):
        expected = evaluate_transform_matrix(
            tangents,
            frame,
            source_width=64,
            source_height=64,
            output_width=spec.width,
            output_height=spec.height,
        )
        actual = evaluate_transform_matrix(
            pivot,
            frame,
            source_width=64,
            source_height=64,
            output_width=spec.width,
            output_height=spec.height,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    coupled = _effect(
        reference_reconstruct_v2(
            base.project_id,
            base,
            asset,
            evidence,
            spec,
            pass_kind="coupled_scale_rotation",
        )
    )
    assert isinstance(coupled.motion, MatrixTransformMotionV2)
    assert len(coupled.motion.matrix.keyframes) == spec.duration_frames
    assert coupled.couplings[0].driver_channel == "scale"
    assert coupled.couplings[0].follower_channels == ["rotation"]
    for expected_frame, actual_keyframe in zip(
        spec.motion_frames,
        coupled.motion.matrix.keyframes,
        strict=True,
    ):
        np.testing.assert_allclose(
            actual_keyframe.matrix_3x3,
            expected_frame.matrix_3x3,
            atol=1e-12,
        )

    correction = _phase_correction(spec)
    corrected = _effect(
        reference_reconstruct_v2(
            base.project_id,
            base,
            asset,
            [*evidence, _phase_correction_evidence(correction, asset, base)],
            spec,
            pass_kind="phase_compensated_matrix",
            phase_correction=correction,
        )
    )
    assert isinstance(corrected.motion, MatrixTransformMotionV2)
    np.testing.assert_allclose(
        corrected.motion.matrix.keyframes[1].matrix_3x3,
        spec.motion_frames[2].matrix_3x3,
        atol=1e-12,
    )
    for frame in range(3, spec.duration_frames - 2):
        np.testing.assert_allclose(
            corrected.motion.matrix.keyframes[frame].matrix_3x3,
            spec.motion_frames[frame].matrix_3x3,
            atol=1e-12,
        )

    hypotheses = classify_reconstruction_hypotheses_v2(spec)
    assert hypotheses["moving_pivot"].status == "experimental_gauge_only"
    assert hypotheses["affine_homography"].status == "rejected_by_evidence"
    assert hypotheses["local_warp"].status == "rejected_by_evidence"
    assert hypotheses["local_warp"].measurements["frames_over_material_p95"] == 0


def test_phase_correction_rejects_non_monotonic_or_unbounded_warp() -> None:
    spec = _spec()
    correction = _phase_correction(spec)
    payload = correction.model_dump(mode="json")
    payload["anchors"][2]["source_frame"] = payload["anchors"][1]["source_frame"]
    with pytest.raises(ValueError, match="source frames must be strictly increasing"):
        ReferencePhaseCorrectionV2.model_validate(payload)

    payload = correction.model_dump(mode="json")
    payload["anchors"][-1]["source_frame"] = spec.duration_frames
    with pytest.raises(ValueError, match="end at the final identity endpoint"):
        ReferencePhaseCorrectionV2.model_validate(payload)


def test_reference_v2_runner_uses_approved_source_only_application_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generic-source.png"
    image = Image.new("RGB", (64, 64), "#102040")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 59, 59), outline="#f0c040", width=4)
    draw.ellipse((13, 11, 52, 50), fill="#d04454")
    draw.polygon(((32, 7), (9, 56), (56, 55)), fill="#48a078")
    image.save(source)
    spec = _spec(sha256_file(source))
    spec_path = tmp_path / "reference-spec-v2.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    receipt = run_reconstruction_pass_v2(
        spec_path,
        source,
        tmp_path / "pass",
        pass_kind="coupled_scale_rotation",
        settings=Settings.for_home(tmp_path / "editing-home"),
    )

    assert receipt.overall == "pass"
    assert receipt.pass_kind == "coupled_scale_rotation"
    assert receipt.source_sha256 == sha256_file(source)
    assert receipt.reference_media_used_as_input is False
    assert Path(receipt.approved_edit_graph).is_file()
    assert Path(receipt.treatment_path).is_file()
    assert Path(receipt.profiles["final"]["path"]).is_file()
    assert receipt.profiles["final"]["lineage"]["input_hashes"] == [sha256_file(source)]
    assert receipt.profiles["final"]["lineage"]["forbidden_hashes_present"] == []
    assert receipt.editable_exports["kdenlive"]["validator"] is not None
    assert receipt.editable_exports["otio"]["validator"] is not None


def test_clean_rerun_v2_replays_approved_semantics_without_prior_media(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generic-source.png"
    image = Image.new("RGB", (64, 64), "#183450")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 59, 59), outline="#f2b941", width=3)
    draw.ellipse((12, 10, 53, 51), fill="#bd3c62")
    draw.line((5, 57, 58, 7), fill="#48b088", width=4)
    image.save(source)
    spec = _spec(sha256_file(source))
    spec_path = tmp_path / "reference-spec-v2.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    correction = _phase_correction(spec).model_copy(
        update={"spec_sha256": sha256_file(spec_path)}
    )
    baseline_root = tmp_path / "baseline"
    baseline = run_reconstruction_pass_v2(
        spec_path,
        source,
        baseline_root,
        pass_kind="phase_compensated_matrix",
        approved_phase_correction=correction,
        settings=Settings.for_home(tmp_path / "baseline-home"),
    )
    report = run_clean_rerun_v2(
        spec_path,
        source,
        baseline_root / "reconstruction-pass-v2.json",
        tmp_path / "clean-rerun-v2",
        readiness_receipt={
            "overall": "pass",
            "git_revision": REVISION,
            "checks": [{"id": "git-clean", "status": "pass"}],
        },
        runtime_settings=Settings.for_home(tmp_path / "normal-home"),
    )

    assert baseline.provenance.parameters["phase_correction_source"] == (
        "approved_semantics"
    )
    assert report.overall == "pass"
    assert report.mandatory_skips == 0
    assert len(report.checks) == 15
    assert all(item.status == "pass" for item in report.checks)
    assert report.baseline.project_id != report.replay.project_id
    assert report.baseline.asset_inode != report.replay.asset_inode
    assert (
        report.baseline.semantic_timeline_sha256
        == report.replay.semantic_timeline_sha256
    )
    assert (
        report.baseline.profiles["final"].sha256
        == report.replay.profiles["final"].sha256
    )
    assert report.replay.profiles["final"].input_hashes == [sha256_file(source)]
    assert report.reference_media_used_as_input is False
    assert Path(report.artifacts["replay_kdenlive"]).is_file()
    assert Path(report.artifacts["replay_otio"]).is_file()


def test_approved_phase_correction_must_match_the_frozen_spec(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generic-source.png"
    Image.new("RGB", (64, 64), "#203050").save(source)
    spec = _spec(sha256_file(source))
    spec_path = tmp_path / "reference-spec-v2.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ReconstructionV2Error, match="does not match"):
        run_reconstruction_pass_v2(
            spec_path,
            source,
            tmp_path / "invalid-pass",
            pass_kind="phase_compensated_matrix",
            approved_phase_correction=_phase_correction(spec),
            settings=Settings.for_home(tmp_path / "invalid-home"),
        )


def test_clean_rerun_v2_decision_is_accepted_and_discoverable() -> None:
    root = Path(__file__).parents[1]
    decision = (
        root
        / "docs"
        / "decisions"
        / "ADR-0015-approved-semantics-clean-replay-v2.md"
    )
    text = " ".join(decision.read_text(encoding="utf-8").split())
    index = (root / "docs" / "decisions" / "README.md").read_text(encoding="utf-8")

    assert "- Status: accepted" in text
    assert "phase_correction_source=approved_semantics" in text
    assert "only media input is the hash-locked source PNG" in text
    assert "not human editorial indistinguishability" in text
    assert f"({decision.name})" in index


def test_reference_v2_planner_rejects_wrong_source_and_unknown_pass() -> None:
    spec = _spec()
    asset = _asset("3" * 64)
    base = _base(spec)
    evidence = [_evidence(spec, asset, base)]
    with pytest.raises(ValueError, match="source hash"):
        reference_reconstruct_v2(
            base.project_id,
            base,
            asset,
            evidence,
            spec,
            pass_kind="timing_easing",
        )
