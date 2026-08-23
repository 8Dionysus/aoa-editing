from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aoa_editing.analysis.video_motion import VideoMotionAnalysisService
from aoa_editing.analysis.video_pipeline import VideoAnatomyPipelineService
from aoa_editing.application.service import EditingService
from aoa_editing.application.video_jobs import VideoAnatomyJobService
from aoa_editing.application.video_proposals import (
    VideoProposalError,
    VideoProposalService,
    canonical_model_sha256,
)
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    AICapabilityDeclaration,
    CheckResult,
    FrameRange,
    FrameRate,
    Intent,
    JobReceipt,
    JobStatus,
    LocalProviderBinding,
    LocalProviderHealthBinding,
    Provenance,
    ProviderHealthEvidence,
    ReferenceAudioStructure,
    ReferenceMotionFrameV2,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionSpecV2,
    Scenario,
    TimeWarpAnchorV2,
    TransformEffectV2,
    VideoAnatomyProfile,
    VideoAnatomyStatus,
    VideoMotionClassification,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.service import (
    ProviderExecution,
    ProviderExecutor,
    ProviderService,
    ProviderTimeout,
)
from aoa_editing.render.service import RenderService

VISION_ALIAS = "local-ai://vision/describe/default"


@dataclass
class _VisionExecutor(ProviderExecutor):
    invocation_count: int = 0

    def health(
        self,
        _binding: LocalProviderBinding,
        _declaration: AICapabilityDeclaration,
    ) -> ProviderHealthEvidence:
        return ProviderHealthEvidence(
            status="healthy",
            checked_at=datetime.now(UTC),
            latency_milliseconds=1,
            source="injected-test",
            response_sha256="a" * 64,
            reported_status="ready",
            protocol_version="aoa-local-ai-v1",
            backend="fixture-vision",
            model_id="fixture-vision-model",
            model_revision="r1",
            summary="fixture provider is healthy",
        )

    def invoke(
        self,
        _binding: LocalProviderBinding,
        _payload: dict[str, Any],
    ) -> ProviderExecution:
        self.invocation_count += 1
        return ProviderExecution(
            output={
                "descriptions": [
                    {
                        "time_seconds": 0.0,
                        "text": "A red full-frame field.",
                        "confidence": 0.95,
                    },
                    {
                        "time_seconds": 2.0,
                        "text": "A blue full-frame field.",
                        "confidence": 0.95,
                    },
                ],
                "partial": False,
            },
            partial=False,
        )


@dataclass
class _FailingVisionExecutor(_VisionExecutor):
    failure: str = "timeout"

    def invoke(
        self,
        _binding: LocalProviderBinding,
        _payload: dict[str, Any],
    ) -> ProviderExecution:
        self.invocation_count += 1
        if self.failure == "timeout":
            raise ProviderTimeout("fixture timeout")
        return ProviderExecution(output={"unexpected": True}, partial=False)


def _vision_service(settings: Settings, executor: _VisionExecutor) -> ProviderService:
    binding = LocalProviderBinding(
        alias=VISION_ALIAS,
        state="enabled",
        resolved_owner="abyss-machine",
        adapter_kind="abyss-machine-cli",
        backend="fixture-vision",
        model_id="fixture-vision-model",
        model_revision="r1",
        model_hash_authority="fixture hash",
        license_authority="fixture license",
        metadata_evidence="fixture://vision",
        command=["fixture-vision", "{request_json}"],
        health=LocalProviderHealthBinding(command=["fixture-health"]),
        data_boundary="local-host",
    )
    return ProviderService(
        settings,
        bindings={VISION_ALIAS: binding},
        executor=executor,
    )


def _hard_cut_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=320x180:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=320x180:rate=30:duration=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _moving_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _project(store: ProjectStore, name: str) -> str:
    return store.create_project(
        name,
        Intent(text="Measure before proposing", scenario=Scenario.MEMORY_MONTAGE),
    ).id


def _precise_spec(
    *,
    source_sha256: str,
    reference_sha256: str,
    frame_rate: FrameRate,
    duration_frames: int,
) -> ReferenceReconstructionSpecV2:
    frames: list[ReferenceMotionFrameV2] = []
    for frame in range(duration_frames):
        progress = frame / (duration_frames - 1)
        frames.append(
            ReferenceMotionFrameV2(
                frame=frame,
                time_seconds=frame / frame_rate.fps,
                selected_model="similarity",
                matrix_3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                visible_source_boundary=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                center_x=0.5 + 0.05 * progress,
                center_y=0.5,
                scale_relative_to_contain=1.0 + 0.2 * progress,
                rotation_degrees=3.0 * progress,
                confidence=0.98,
                confidence_interval={},
                match_count=100,
                inlier_count=98,
                inlier_ratio=0.98,
                reprojection_rmse=0.2,
                residual_flow_mean=0.0,
                residual_flow_p95=0.0,
                residual_valid_fraction=1.0,
                velocity={},
                acceleration={},
                jerk={},
                model_scores={"similarity": 1.0},
                outlier=False,
                uncertainty=["synthetic normalized transform fixture"],
            )
        )
    return ReferenceReconstructionSpecV2(
        evidence_id="referencemotion_" + "a" * 32,
        source_sha256=source_sha256,
        reference_sha256=reference_sha256,
        readiness_revision="b" * 40,
        motion_gate_revision="b" * 40,
        duration_frames=duration_frames,
        duration_seconds=duration_frames / frame_rate.fps,
        frame_rate=frame_rate,
        width=320,
        height=180,
        audio=ReferenceAudioStructure(
            has_audio_stream=False,
            silent_for_full_duration=True,
        ),
        motion_model="similarity",
        motion_frames=frames,
        curve_representation={"kind": "dense-normalized", "sampling": "every_frame"},
        phase_model={},
        transform_order={"authoring_order": "not_claimed"},
        uncertainties=["synthetic normalized transform fixture"],
        alternative_explanations=["equivalent decompositions exist"],
        comparison_criteria={},
        artifacts={},
        provenance=Provenance(tool="synthetic", tool_version="1", deterministic=True),
    )


def _phase_correction(
    spec: ReferenceReconstructionSpecV2,
    *,
    spec_sha256: str,
) -> ReferencePhaseCorrectionV2:
    return ReferencePhaseCorrectionV2(
        spec_id=spec.id,
        spec_sha256=spec_sha256,
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        comparison_report_id="comparison_" + "a" * 32,
        comparison_report_path="comparison-v2.json",
        comparison_report_sha256="d" * 64,
        candidate_motion_report_id="motionrecovery_" + "b" * 32,
        candidate_motion_path="candidate-motion-recovery.json",
        candidate_motion_sha256="e" * 64,
        prior_candidate_sha256="f" * 64,
        duration_frames=spec.duration_frames,
        anchors=[
            TimeWarpAnchorV2(output_frame=0, source_frame=0),
            TimeWarpAnchorV2(output_frame=10, source_frame=8),
            TimeWarpAnchorV2(output_frame=20, source_frame=20),
            TimeWarpAnchorV2(output_frame=90, source_frame=90),
            TimeWarpAnchorV2(output_frame=110, source_frame=108),
            TimeWarpAnchorV2(
                output_frame=spec.duration_frames - 1,
                source_frame=spec.duration_frames - 1,
            ),
        ],
        identity_ranges=[FrameRange(start=20, duration=71)],
        protected_coupling_range=FrameRange(start=30, duration=50),
        previous_phase={"onset": 12, "peak_velocity": 60, "settle": 108},
        predicted_phase={"onset": 10, "peak_velocity": 60, "settle": 110},
        predicted_metrics={"fixture": True},
        predicted_checks=[
            CheckResult(
                id="predicted-phase",
                status="pass",
                summary="synthetic correction stays within frozen bounds",
            )
        ],
        failed_check_ids=["temporal-phase-onset", "temporal-phase-settle"],
        selection={"fixture": True},
        provenance=Provenance(
            tool="synthetic-phase-correction",
            tool_version="1",
            deterministic=True,
        ),
    )


def test_semantic_profile_normalizes_provider_output_and_hits_revision_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hard-cut.mp4"
    _hard_cut_video(source)
    settings = Settings.for_home(tmp_path / "editing-home")
    store = ProjectStore(settings)
    project_id = _project(store, "Semantic fixture")
    asset, _ = store.ingest_asset(project_id, source)
    executor = _VisionExecutor()
    providers = _vision_service(settings, executor)
    pipeline = VideoAnatomyPipelineService(store, providers=providers)

    first = pipeline.analyze(
        project_id,
        asset.id,
        profile=VideoAnatomyProfile.SEMANTIC,
    )
    second = pipeline.analyze(
        project_id,
        asset.id,
        profile=VideoAnatomyProfile.SEMANTIC,
    )

    assert first.anatomy.status is VideoAnatomyStatus.COMPLETE
    assert len(first.anatomy.visual_observations) == 2
    assert first.anatomy.coverage_matrix["visual"] == 1.0
    assert all(item.raw_response_sha256 for item in first.anatomy.visual_observations)
    assert executor.invocation_count == 1
    cache_record = next(
        item for item in second.evidence_records if item.kind == "video.visual-observations"
    )
    assert cache_record.payload["cache_hit"] is True
    assert store.load_project(project_id).current_version_id is None


def test_quick_profile_uses_bounded_sparse_keyframe_and_uniform_scan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "quick.mp4"
    _hard_cut_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Quick sparse fixture")
    asset, _ = store.ingest_asset(project_id, source)

    result = VideoAnatomyPipelineService(store).analyze(
        project_id,
        asset.id,
        profile=VideoAnatomyProfile.QUICK,
        global_frame_budget=8,
    )

    quick = next(
        item
        for item in result.evidence_records
        if item.kind == "video.structure.quick-candidates"
    )
    detector = next(
        item
        for item in result.evidence_records
        if item.kind == "video.structure.detectors"
    )
    assert "keyframe" in result.anatomy.plan.strategies
    assert quick.payload["decoded_frame_count"] < quick.payload["source_frame_count"]
    assert detector.payload["scan_mode"] == "sparse"
    assert detector.payload["decoded_ratio"] < 0.25
    assert detector.provenance.parameters["operation"] == (
        "sparse-keyframe-uniform-structural-scan"
    )
    assert all(
        transition.transition_type.value == "unknown"
        for transition in result.anatomy.structure.transitions
    )
    assert store.load_project(project_id).current_version_id is None


def test_high_resolution_samples_use_bounded_analysis_frames_and_estimate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "high-resolution.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=24:duration=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Bounded sample memory")
    asset, _ = store.ingest_asset(project_id, source)
    jobs = VideoAnatomyJobService(store)

    estimate = jobs.estimate(project_id, asset.id, profile=VideoAnatomyProfile.RECONSTRUCT)
    result = jobs.run(project_id, asset.id, profile=VideoAnatomyProfile.QUICK)

    assert estimate.provenance.parameters["analysis_pixel_bytes"] == 320 * 180 * 3
    assert 512 * 1024 * 1024 <= estimate.estimated_peak_temporary_bytes < 768 * 1024 * 1024
    assert all(
        item.width <= 320 and item.height <= 180
        for item in result.anatomy.frame_manifest.samples
    )
    assert all(
        item.extraction_parameters["full_resolution_source_retained_in_memory"] is False
        for item in result.anatomy.frame_manifest.samples
    )


@pytest.mark.parametrize(
    ("failure", "failure_code"),
    [("timeout", "provider_timeout"), ("malformed", "malformed_response")],
)
def test_semantic_profile_localizes_timeout_and_malformed_provider_output(
    tmp_path: Path,
    failure: str,
    failure_code: str,
) -> None:
    source = tmp_path / f"{failure}.mp4"
    _hard_cut_video(source)
    settings = Settings.for_home(tmp_path / f"editing-home-{failure}")
    store = ProjectStore(settings)
    project_id = _project(store, f"Semantic {failure}")
    asset, _ = store.ingest_asset(project_id, source)
    executor = _FailingVisionExecutor(failure=failure)

    result = VideoAnatomyPipelineService(
        store,
        providers=_vision_service(settings, executor),
    ).analyze(
        project_id,
        asset.id,
        profile=VideoAnatomyProfile.SEMANTIC,
    )

    visual_record = next(
        item
        for item in result.evidence_records
        if item.kind == "video.visual-observations"
    )
    assert result.anatomy.status is VideoAnatomyStatus.PARTIAL
    assert result.anatomy.structure.coverage.coverage_ratio == 1.0
    assert result.anatomy.visual_observations == []
    assert visual_record.payload["provider_failure_code"] == failure_code
    assert visual_record.payload["structural_profile_preserved"] is True
    assert executor.invocation_count == 1
    assert store.load_project(project_id).current_version_id is None


def test_motion_profile_uses_dense_all_frame_evidence(tmp_path: Path) -> None:
    source = tmp_path / "moving.mp4"
    _moving_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Motion fixture")
    asset, _ = store.ingest_asset(project_id, source)

    result = VideoAnatomyPipelineService(store).analyze(
        project_id,
        asset.id,
        profile=VideoAnatomyProfile.MOTION,
    )

    assert result.anatomy.motion_evidence
    assert result.anatomy.coverage_matrix["motion"] == 1.0
    assert all(
        item.analyzed_frame_count == item.source_frame_count
        for item in result.anatomy.motion_evidence
    )
    assert all(item.all_frame_evidence_ref for item in result.anatomy.motion_evidence)
    assert any(
        item.classification is not VideoMotionClassification.STATIC
        for item in result.anatomy.motion_evidence
    )
    assert store.load_project(project_id).current_version_id is None


def test_proposal_is_inert_until_review_then_uses_reversible_patch_path(
    tmp_path: Path,
) -> None:
    reference_like = tmp_path / "structure.mp4"
    target_path = tmp_path / "target.png"
    _hard_cut_video(reference_like)
    Image.new("RGB", (320, 180), "#cc7722").save(target_path)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Reviewed reconstruction")
    anatomy_asset, _ = store.ingest_asset(project_id, reference_like)
    target_asset, _ = store.ingest_asset(project_id, target_path)
    editing = EditingService(store)
    base = editing.initialize_timeline(project_id, duration_frames=120, width=320, height=180)
    anatomy = (
        VideoAnatomyPipelineService(store)
        .analyze(
            project_id,
            anatomy_asset.id,
            profile=VideoAnatomyProfile.STRUCTURAL,
        )
        .anatomy
    )
    proposals = VideoProposalService(store)

    editorial = proposals.create_editorial_structure(anatomy)
    reconstruction = proposals.create_reference_reconstruction(
        anatomy,
        target_asset_id=target_asset.id,
        base_version_id=base.id,
    )

    assert editorial.timeline_mutated is False
    assert reconstruction.timeline_mutated is False
    assert reconstruction.reference_media_embedded is False
    assert reconstruction.patch_preview is not None
    assert store.load_project(project_id).current_version_id == base.id

    review = proposals.review_reconstruction(
        project_id,
        reconstruction.id,
        decision="approved",
        reviewer="fixture-human",
        rationale="Boundary triplets and patch preview were reviewed.",
    )
    assert store.load_project(project_id).current_version_id == base.id

    acceptance = proposals.accept_reconstruction(project_id, reconstruction.id, review.id)
    accepted = store.load_version(project_id, acceptance.version_id)
    assert accepted.parent_version_id == base.id
    assert acceptance.inverse_operation_count > 0
    reverted = editing.revert_version(project_id, accepted.id)
    assert reverted.timeline == base.timeline


def test_reconstruct_job_emits_inert_source_neutral_proposals_once(tmp_path: Path) -> None:
    reference_like = tmp_path / "structure.mp4"
    _hard_cut_video(reference_like)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Reconstruct output contract")
    asset, _ = store.ingest_asset(project_id, reference_like)
    jobs = VideoAnatomyJobService(store)

    first = jobs.run(project_id, asset.id, profile=VideoAnatomyProfile.RECONSTRUCT)
    cached = jobs.run(project_id, asset.id, profile=VideoAnatomyProfile.RECONSTRUCT)

    assert first.editorial_proposal is not None
    assert first.reconstruction_proposal is not None
    assert first.editorial_proposal.timeline_mutated is False
    assert first.reconstruction_proposal.source_neutral is True
    assert first.reconstruction_proposal.patch_preview is None
    assert store.load_project(project_id).current_version_id is None
    assert cached.editorial_proposal == first.editorial_proposal
    assert cached.reconstruction_proposal == first.reconstruction_proposal


def test_precise_focused_spec_compiles_dense_normalized_motion_without_reference_media(
    tmp_path: Path,
) -> None:
    reference_like = tmp_path / "reference-like.mp4"
    target_path = tmp_path / "target.png"
    _hard_cut_video(reference_like)
    Image.new("RGB", (320, 180), "#cc7722").save(target_path)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Precise normalized reconstruction")
    reference_asset, _ = store.ingest_asset(project_id, reference_like)
    target_asset, _ = store.ingest_asset(project_id, target_path)
    assert reference_asset.metadata.frame_rate is not None
    editing = EditingService(store)
    base = editing.initialize_timeline(
        project_id,
        duration_frames=120,
        width=320,
        height=180,
        frame_rate=reference_asset.metadata.frame_rate,
    )
    anatomy = (
        VideoAnatomyPipelineService(store)
        .analyze(
            project_id,
            reference_asset.id,
            profile=VideoAnatomyProfile.RECONSTRUCT,
        )
        .anatomy
    )
    spec = _precise_spec(
        source_sha256=target_asset.sha256,
        reference_sha256=reference_asset.sha256,
        frame_rate=reference_asset.metadata.frame_rate,
        duration_frames=120,
    )

    proposals = VideoProposalService(store)
    proposal = proposals.create_reference_reconstruction(
        anatomy,
        target_asset_id=target_asset.id,
        base_version_id=base.id,
        precise_motion_spec=spec,
        precise_motion_spec_ref=f"reference-spec-v2:{spec.id}",
    )

    assert proposal.patch_preview is not None
    tracks = proposal.patch_preview.patch.operations[-1].value
    assert isinstance(tracks, list)
    effects = [
        clip["effects"][0]
        for track in tracks
        for clip in track["clips"]
    ]
    assert all(TransformEffectV2.model_validate(effect) for effect in effects)
    assert sum(
        len(effect["motion"]["position"]["keyframes"])
        for effect in effects
    ) == spec.duration_frames
    assert proposal.source_neutral is True
    assert proposal.reference_media_embedded is False
    assert reference_asset.id not in json.dumps(
        proposal.patch_preview.patch.model_dump(mode="json")
    )
    assert store.load_project(project_id).current_version_id == base.id

    review = proposals.review_reconstruction(
        project_id,
        proposal.id,
        decision="approved",
        reviewer="fixture-human",
        rationale="Dense normalized curves and source isolation were reviewed.",
    )
    acceptance = proposals.accept_reconstruction(project_id, proposal.id, review.id)
    render = RenderService(store).render(
        project_id,
        acceptance.version_id,
        profile="preview",
    )
    lineage_path = store.project_path(project_id) / render.output_paths[0]
    lineage = json.loads((lineage_path.parent / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["input_hashes"] == [target_asset.sha256]
    assert reference_asset.sha256 not in lineage["input_hashes"]


def test_comparison_bound_phase_correction_creates_new_inert_hash_bound_preview(
    tmp_path: Path,
) -> None:
    reference_like = tmp_path / "reference-like.mp4"
    target_path = tmp_path / "target.png"
    _hard_cut_video(reference_like)
    Image.new("RGB", (320, 180), "#cc7722").save(target_path)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Comparison-bound refinement")
    reference_asset, _ = store.ingest_asset(project_id, reference_like)
    target_asset, _ = store.ingest_asset(project_id, target_path)
    assert reference_asset.metadata.frame_rate is not None
    editing = EditingService(store)
    base = editing.initialize_timeline(
        project_id,
        duration_frames=120,
        width=320,
        height=180,
        frame_rate=reference_asset.metadata.frame_rate,
    )
    anatomy = (
        VideoAnatomyPipelineService(store)
        .analyze(
            project_id,
            reference_asset.id,
            profile=VideoAnatomyProfile.RECONSTRUCT,
        )
        .anatomy
    )
    spec = _precise_spec(
        source_sha256=target_asset.sha256,
        reference_sha256=reference_asset.sha256,
        frame_rate=reference_asset.metadata.frame_rate,
        duration_frames=120,
    )
    spec_sha256 = "c" * 64
    correction = _phase_correction(spec, spec_sha256=spec_sha256)
    spec_ref = f"reference-spec-v2:{spec.id}:{spec_sha256}"
    correction_sha256 = canonical_model_sha256(correction)
    correction_ref = f"reference-phase-correction-v2:{correction.id}:{correction_sha256}"

    proposals = VideoProposalService(store)
    proposal = proposals.create_reference_reconstruction(
        anatomy,
        target_asset_id=target_asset.id,
        base_version_id=base.id,
        precise_motion_spec=spec,
        precise_motion_spec_ref=spec_ref,
        phase_correction=correction,
        phase_correction_ref=correction_ref,
    )

    assert proposal.patch_preview is not None
    assert store.load_project(project_id).current_version_id == base.id
    assert correction_ref in proposal.evidence_refs
    assert proposal.provenance.parameters["phase_correction_id"] == correction.id
    assert proposal.provenance.parameters["phase_correction_sha256"] == correction_sha256
    assert (
        proposal.provenance.parameters["phase_correction_comparison_report_sha256"]
        == correction.comparison_report_sha256
    )
    tracks = proposal.patch_preview.patch.operations[-1].value
    assert isinstance(tracks, list)
    keyframes = tracks[0]["clips"][0]["effects"][0]["motion"]["position"]["keyframes"]
    assert keyframes[10]["time"]["frame"] == 10
    expected_source_progress = 8 / (spec.duration_frames - 1)
    assert keyframes[10]["value"]["x"] == pytest.approx(0.5 + 0.05 * expected_source_progress)
    assert keyframes[50]["value"]["x"] == pytest.approx(spec.motion_frames[50].center_x)

    with pytest.raises(
        VideoProposalError,
        match="does not bind the exact correction semantics",
    ):
        proposals.create_reference_reconstruction(
            anatomy,
            target_asset_id=target_asset.id,
            base_version_id=base.id,
            precise_motion_spec=spec,
            precise_motion_spec_ref=spec_ref,
            phase_correction=correction,
            phase_correction_ref=(f"reference-phase-correction-v2:{correction.id}:{'0' * 64}"),
        )


def test_job_spine_records_stages_cache_hit_cancellation_and_retry(tmp_path: Path) -> None:
    source = tmp_path / "hard-cut.mp4"
    _hard_cut_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Job lifecycle")
    asset, _ = store.ingest_asset(project_id, source)
    jobs = VideoAnatomyJobService(store)

    first = jobs.run(project_id, asset.id)
    cached = jobs.run(project_id, asset.id)
    retried_success = jobs.retry(project_id, first.receipt.id)

    assert first.receipt.status is JobStatus.SUCCEEDED
    assert cached.receipt.cache_hit is True
    assert retried_success.receipt.cache_hit is True
    assert retried_success.receipt.resume_from_job_id == first.receipt.id
    assert cached.anatomy.anatomy_sha256 == first.anatomy.anatomy_sha256
    child_phases = {
        item.phase for item in store.list_jobs(project_id) if item.parent_job_id == first.receipt.id
    }
    assert {"structural", "audio", "aggregate"} <= child_phases

    queued = JobReceipt(
        kind="video-anatomy",
        project_id=project_id,
        asset_id=asset.id,
        profile=VideoAnatomyProfile.QUICK.value,
    )
    store.save_job(queued)
    cancelled = jobs.cancel(project_id, queued.id)
    assert cancelled.status is JobStatus.CANCELLED
    resumed = jobs.retry(project_id, cancelled.id)
    assert resumed.receipt.status is JobStatus.SUCCEEDED
    assert resumed.receipt.resume_from_job_id == cancelled.id
    assert resumed.receipt.attempt == 2


def test_job_retry_verifies_checkpoint_and_skips_completed_sibling_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "moving.mp4"
    _moving_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project_id = _project(store, "Crash resume")
    asset, _ = store.ingest_asset(project_id, source)
    jobs = VideoAnatomyJobService(store)
    original = VideoMotionAnalysisService.analyze

    def crash_motion(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected crash after durable audio checkpoint")

    monkeypatch.setattr(VideoMotionAnalysisService, "analyze", crash_motion)
    with pytest.raises(RuntimeError, match="injected crash"):
        jobs.run(
            project_id,
            asset.id,
            profile=VideoAnatomyProfile.MOTION,
            analysis_range=FrameRange(start=0, duration=40),
            pinned_frames=(0, 20, 39),
        )

    failed = next(
        item
        for item in reversed(store.list_jobs(project_id))
        if item.kind == "video.anatomy" and item.parent_job_id is None
    )
    assert failed.status is JobStatus.FAILED
    assert failed.completed_phases == ["structural", "audio"]
    assert failed.plan_id is not None
    assert failed.checkpoint_path is not None
    structural_ids = {
        item.id for item in store.list_evidence(project_id) if item.kind == "video.structure"
    }

    monkeypatch.setattr(VideoMotionAnalysisService, "analyze", original)
    resumed = jobs.retry(project_id, failed.id)

    assert resumed.receipt.status is JobStatus.SUCCEEDED
    assert resumed.receipt.resume_from_job_id == failed.id
    assert resumed.receipt.attempt == 2
    assert resumed.anatomy.plan.id == failed.plan_id
    assert resumed.anatomy.plan.analysis_range == FrameRange(start=0, duration=40)
    assert [item.frame for item in resumed.anatomy.plan.pinned_times] == [0, 20, 39]
    assert {
        item.id for item in store.list_evidence(project_id) if item.kind == "video.structure"
    } == structural_ids
