from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Asset,
    CameraFrameMeasurement,
    EvidenceRecord,
    FrameRate,
    MediaKind,
    MediaMetadata,
    ProjectVersion,
    Provenance,
    ReferenceAudioStructure,
    ReferenceCameraMotion,
    ReferenceComposition,
    ReferenceReconstructionSpec,
    Scenario,
    Timeline,
    TransformEffect,
)
from aoa_editing.evals.clean_rerun import run_clean_rerun
from aoa_editing.evals.comparison import run_comparison
from aoa_editing.evals.fixtures import create_transfer_fixture
from aoa_editing.evals.reconstruction import run_reconstruction
from aoa_editing.evals.transfer import run_technique_transfer
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.knowledge.techniques import extract_contain_reveal_technique
from aoa_editing.scenarios.planners import (
    PlanningError,
    apply_camera_technique,
    reference_reconstruct,
)

SOURCE_HASH = "1" * 64
REFERENCE_HASH = "2" * 64


def _provenance() -> Provenance:
    return Provenance(tool="synthetic", tool_version="1", deterministic=True)


def _spec(source_hash: str = SOURCE_HASH) -> ReferenceReconstructionSpec:
    measurements = [
        CameraFrameMeasurement(
            frame=0,
            time_seconds=0,
            scale_relative_to_contain=2.4,
            rotation_degrees=-5,
            center_x=0.32,
            center_y=0.61,
            match_count=100,
            inlier_count=98,
            inlier_ratio=0.98,
            reprojection_rmse=0.2,
        ),
        CameraFrameMeasurement(
            frame=24,
            time_seconds=0.96,
            scale_relative_to_contain=1,
            rotation_degrees=0,
            center_x=0.5,
            center_y=0.5,
            match_count=100,
            inlier_count=97,
            inlier_ratio=0.97,
            reprojection_rmse=0.2,
        ),
    ]
    metadata = MediaMetadata(
        duration_seconds=1,
        width=270,
        height=480,
        frame_rate=FrameRate(numerator=25),
        video_codec="h264",
        has_video=True,
        has_audio=True,
    )
    return ReferenceReconstructionSpec(
        readiness_revision="synthetic",
        readiness_receipt_path="/eval/readiness.json",
        source_sha256=source_hash,
        reference_sha256=REFERENCE_HASH,
        reference_metadata=metadata,
        duration_frames=25,
        duration_seconds=1,
        camera_motion=ReferenceCameraMotion(
            model="single_source_similarity_transform",
            frame_count=25,
            frame_rate=FrameRate(numerator=25),
            width=270,
            height=480,
            curve_hint="ease_in_out",
            measurements=measurements,
        ),
        composition=ReferenceComposition(
            model="single_source_similarity_transform",
            cut_count=0,
            independent_layer_motion_observed=False,
            masks_observed=False,
            background="black",
            notes=[],
        ),
        audio=ReferenceAudioStructure(
            has_audio_stream=True,
            silent_for_full_duration=True,
        ),
        observed_effects={},
        uncertainties=[],
        alternative_explanations=[],
        comparison_criteria={},
        provenance=_provenance(),
    )


def test_reference_reconstruction_is_a_normal_evidence_backed_treatment() -> None:
    asset = Asset(
        sha256=SOURCE_HASH,
        original_name="unrelated-square.png",
        media_kind=MediaKind.IMAGE,
        stored_path="assets/source.png",
        size_bytes=123,
        metadata=MediaMetadata(width=512, height=512, has_video=True),
    )
    base = ProjectVersion(
        project_id="project_" + "a" * 32,
        message="Initial timeline",
        timeline=Timeline(
            width=270,
            height=480,
            frame_rate=FrameRate(numerator=25),
            duration_frames=25,
        ),
    )
    evidence = EvidenceRecord(
        project_id=base.project_id,
        asset_id=asset.id,
        source_sha256=SOURCE_HASH,
        kind="reference.reconstruction_spec",
        confidence=0.97,
        payload={"spec_id": _spec().id},
        provenance=_provenance(),
    )

    treatment = reference_reconstruct(base.project_id, base, asset, [evidence], _spec())

    assert treatment.scenario is Scenario.REFERENCE_RECONSTRUCT
    tracks = treatment.patch.operations[-1].value
    effect = TransformEffect.model_validate(tracks[0]["clips"][0]["effects"][0])
    assert effect.fit_mode == "contain"
    assert effect.position_mode == "canvas_center"
    assert [item.value for item in effect.scale] == [2.4, 1.0]
    assert [(item.x, item.y) for item in effect.position] == [(0.32, 0.61), (0.5, 0.5)]

    mismatched = asset.model_copy(update={"sha256": "3" * 64})
    with pytest.raises(PlanningError, match="source hash"):
        reference_reconstruct(base.project_id, base, mismatched, [evidence], _spec())


def test_reconstruction_runner_uses_only_the_permitted_source_lineage(tmp_path: Path) -> None:
    source_path = tmp_path / "different-subject.png"
    random = np.random.default_rng(91)
    texture = random.integers(20, 230, size=(512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(texture, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 488, 488), outline="#efc94c", width=18)
    draw.ellipse((105, 85, 415, 395), fill="#d95b43")
    draw.polygon([(256, 70), (90, 430), (430, 430)], fill="#5b8c85")
    image.save(source_path)
    source_hash = sha256_file(source_path)
    settings = Settings.for_home(tmp_path / "editing-home")

    reconstruction_spec = _spec(source_hash)
    receipt = run_reconstruction(
        reconstruction_spec,
        source_path,
        tmp_path / "eval",
        settings=settings,
    )

    assert receipt["overall"] == "pass"
    assert receipt["source_sha256"] == source_hash
    assert receipt["reference_sha256"] == REFERENCE_HASH
    for profile in receipt["profiles"].values():
        assert profile["lineage"]["input_hashes"] == [source_hash]
        assert profile["lineage"]["forbidden_hashes_present"] == []
    assert receipt["editable_exports"]["kdenlive"]["validator"] is not None
    assert receipt["editable_exports"]["otio"]["validator"] is not None

    candidate = Path(receipt["profiles"]["final"]["path"])
    comparison_spec = _spec(source_hash).model_copy(
        update={"reference_sha256": sha256_file(candidate)}
    )
    comparison = run_comparison(
        comparison_spec,
        source_path,
        candidate,
        candidate,
        tmp_path / "comparison",
        lineage_path=Path(receipt["profiles"]["final"]["lineage_path"]),
        sample_step=24,
        visual_review="Synthetic identity comparison is visually identical by construction.",
    )
    assert comparison.overall == "pass"
    assert comparison.perceptual["vmaf_mean"] >= 99.7
    assert comparison.perceptual["ssim_all"] == pytest.approx(1, abs=0.0001)

    packet = extract_contain_reveal_technique(comparison_spec, comparison)
    serialized = packet.model_dump_json()
    assert comparison_spec.source_sha256 not in serialized
    assert comparison_spec.reference_sha256 not in serialized
    assert (
        packet.application_history[0].evidence_path
        == "runtime-eval:reference/comparison.json"
    )
    unrelated = Asset(
        sha256="9" * 64,
        original_name="unrelated.png",
        media_kind=MediaKind.IMAGE,
        stored_path="assets/unrelated.png",
        size_bytes=100,
        metadata=MediaMetadata(width=640, height=360, has_video=True),
    )
    base = ProjectVersion(
        project_id="project_" + "b" * 32,
        message="Independent base",
        timeline=Timeline(
            width=270,
            height=480,
            frame_rate=FrameRate(numerator=25),
            duration_frames=25,
        ),
    )
    transfer_evidence = [
        EvidenceRecord(
            project_id=base.project_id,
            asset_id=unrelated.id,
            source_sha256=unrelated.sha256,
            kind=kind,
            confidence=0.8,
            payload={},
            provenance=_provenance(),
        )
        for kind in packet.required_evidence
    ]
    transfer = apply_camera_technique(
        base.project_id,
        base,
        unrelated,
        transfer_evidence,
        packet,
        width=270,
        height=480,
    )
    assert transfer.scenario is Scenario.STILL_MOTION
    transfer_tracks = transfer.patch.operations[-1].value
    transfer_effect = TransformEffect.model_validate(
        transfer_tracks[0]["clips"][0]["effects"][0]
    )
    assert transfer_effect.position == packet.camera_motion.position

    clean = run_clean_rerun(
        reconstruction_spec,
        source_path,
        tmp_path / "eval" / "reconstruction.json",
        tmp_path / "clean-rerun",
        runtime_settings=settings,
    )
    assert clean.overall == "pass"
    assert clean.old_render_used_as_input is False
    assert clean.baseline["project_id"] != clean.replay["project_id"]

    transfer_source = create_transfer_fixture(tmp_path / "procedural-landscape.png")
    transfer_report = run_technique_transfer(
        packet,
        transfer_source,
        tmp_path / "transfer-eval",
        runtime_settings=settings,
        canvas_width=270,
        canvas_height=480,
        forbidden_hashes=[source_hash, comparison_spec.reference_sha256],
    )
    assert transfer_report.overall == "pass"
    assert transfer_report.target_specific_media_used is False
    assert transfer_report.source_sha256 not in {source_hash, comparison_spec.reference_sha256}
