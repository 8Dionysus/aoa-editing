from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aoa_editing.domain.models import (
    CoverageInterval,
    FrameRange,
    FrameRate,
    MotionTimeV2,
    Provenance,
    ShotEvidence,
    TransitionCandidate,
    VideoAnatomy,
    VideoAnatomyProfile,
    VideoAnatomyStatus,
    VideoCoverageReport,
    VideoFrameSample,
    VideoFrameSampleManifest,
    VideoSampleRole,
    VideoSamplingPlan,
    VideoStructureEvidence,
    VideoTransitionType,
)

SOURCE_HASH = "a" * 64
ARTIFACT_HASH = "b" * 64


def _provenance() -> Provenance:
    return Provenance(
        tool="fixture",
        tool_version="1",
        parameters={"threshold": 0.2},
        deterministic=True,
    )


def _plan(**updates: object) -> VideoSamplingPlan:
    payload: dict[str, object] = {
        "project_id": "project_fixture",
        "asset_id": "asset_fixture",
        "source_sha256": SOURCE_HASH,
        "duration_seconds": 4.0,
        "time_base": FrameRate(numerator=30),
        "profile": VideoAnatomyProfile.STRUCTURAL,
        "analysis_range": FrameRange(start=0, duration=120),
        "strategies": ["scene-score", "keyframe", "uniform-fallback"],
        "minimum_samples_per_shot": 3,
        "global_frame_budget": 20,
        "per_shot_frame_budget": 6,
        "pinned_times": [MotionTimeV2(frame=15)],
        "deduplication": {
            "enabled": True,
            "luminance_delta_min": 0.02,
            "histogram_distance_min": 0.04,
            "perceptual_hash_distance_min": 4,
            "ssim_change_min": 0.01,
            "edge_change_min": 0.02,
            "motion_activity_min": 0.01,
        },
        "required_capabilities": [],
        "focused_rescan_reasons": [],
        "artifact_budget_bytes": 10_000_000,
        "created_at": datetime(2026, 8, 23, tzinfo=UTC),
    }
    payload.update(updates)
    return VideoSamplingPlan.model_validate(payload)


def _sample(
    sample_id: str,
    frame: int,
    *,
    shot_id: str,
    role: VideoSampleRole = VideoSampleRole.INTERIOR,
) -> VideoFrameSample:
    return VideoFrameSample(
        id=sample_id,
        shot_id=shot_id,
        time=MotionTimeV2(frame=frame),
        time_seconds=frame / 30,
        source_frame_index=frame,
        role=role,
        selection_reasons=[role.value],
        artifact_path=f"analysis/video-anatomy/frames/{sample_id}.png",
        artifact_sha256=ARTIFACT_HASH,
        width=320,
        height=180,
        pixel_format="rgb24",
        extraction_parameters={"seek": "accurate"},
        provenance=_provenance(),
        seek_precision="exact-frame",
    )


def _structure(plan: VideoSamplingPlan) -> VideoStructureEvidence:
    shots = [
        ShotEvidence(
            id="shot-0001",
            frame_range=FrameRange(start=0, duration=60),
            representative_sample_ids=["sample-0001"],
            boundary_sample_ids=["sample-0002"],
            confidence=0.9,
        ),
        ShotEvidence(
            id="shot-0002",
            frame_range=FrameRange(start=60, duration=60),
            representative_sample_ids=["sample-0003"],
            boundary_sample_ids=["sample-0002"],
            confidence=0.9,
        ),
    ]
    return VideoStructureEvidence(
        project_id=plan.project_id,
        asset_id=plan.asset_id,
        source_sha256=plan.source_sha256,
        plan_id=plan.id,
        plan_sha256=plan.plan_sha256,
        analysis_range=plan.analysis_range,
        shots=shots,
        transitions=[
            TransitionCandidate(
                id="transition-0001",
                frame=60,
                time_seconds=2.0,
                window=FrameRange(start=57, duration=7),
                transition_type=VideoTransitionType.HARD_CUT,
                confidence=0.9,
                detector_signals=[
                    {
                        "detector": "scene-score",
                        "score": 0.8,
                        "supports": [VideoTransitionType.HARD_CUT],
                    },
                    {
                        "detector": "histogram-distance",
                        "score": 0.7,
                        "supports": [VideoTransitionType.HARD_CUT],
                    },
                ],
                sample_ids=["sample-0002"],
            )
        ],
        coverage=VideoCoverageReport(
            requested_range=plan.analysis_range,
            covered_intervals=[
                CoverageInterval(frame_range=plan.analysis_range, reasons=["shot-partition"])
            ],
            uncovered_ranges=[],
            per_shot_sample_counts={"shot-0001": 2, "shot-0002": 2},
            protected_sample_ids=["sample-0002"],
            coverage_ratio=1.0,
        ),
        ambiguous_ranges=[],
        detector_evidence_refs=["evidence_legacy_scenes"],
        provenance=_provenance(),
    )


def test_sampling_plan_hash_is_deterministic_and_detects_tampering() -> None:
    first = _plan(id="samplingplan_first")
    second = _plan(id="samplingplan_second")

    assert first.plan_sha256 == second.plan_sha256

    payload = first.model_dump(mode="json")
    payload["global_frame_budget"] = 21
    with pytest.raises(ValueError, match="plan_sha256"):
        VideoSamplingPlan.model_validate(payload)


def test_structure_requires_a_contiguous_full_shot_partition() -> None:
    plan = _plan()
    structure = _structure(plan)
    assert structure.shots[-1].frame_range.end == plan.analysis_range.end

    payload = structure.model_dump(mode="json")
    payload["shots"][1]["frame_range"]["start"] = 61
    with pytest.raises(ValueError, match="contiguous"):
        VideoStructureEvidence.model_validate(payload)


def test_boundary_and_pinned_samples_cannot_be_deduplicated_away() -> None:
    sample = _sample(
        "sample-0001",
        59,
        shot_id="shot-0001",
        role=VideoSampleRole.PRE_BOUNDARY,
    )
    payload = sample.model_dump(mode="json")
    payload["kept"] = False
    payload["duplicate_of_sample_id"] = "sample-other"
    with pytest.raises(ValueError, match="protected"):
        VideoFrameSample.model_validate(payload)


def test_video_anatomy_joins_hash_bound_child_evidence_without_becoming_a_proposal() -> None:
    plan = _plan()
    structure = _structure(plan)
    samples = [
        _sample("sample-0001", 0, shot_id="shot-0001"),
        _sample(
            "sample-0002",
            59,
            shot_id="shot-0001",
            role=VideoSampleRole.PRE_BOUNDARY,
        ),
        _sample("sample-0003", 90, shot_id="shot-0002"),
        _sample("sample-0004", 119, shot_id="shot-0002"),
    ]
    manifest = VideoFrameSampleManifest(
        project_id=plan.project_id,
        asset_id=plan.asset_id,
        source_sha256=plan.source_sha256,
        plan_id=plan.id,
        plan_sha256=plan.plan_sha256,
        samples=samples,
        selected_count=4,
        duplicate_count=0,
        artifact_bytes=4096,
        provenance=_provenance(),
    )
    anatomy = VideoAnatomy(
        project_id=plan.project_id,
        asset_id=plan.asset_id,
        source_sha256=plan.source_sha256,
        profile=plan.profile,
        technical_metadata={
            "duration_seconds": 4.0,
            "width": 320,
            "height": 180,
            "frame_rate": {"numerator": 30, "denominator": 1},
            "has_video": True,
            "has_audio": False,
        },
        track_summary={"video_streams": 1, "audio_streams": 0},
        plan=plan,
        frame_manifest=manifest,
        structure=structure,
        visual_observations=[],
        motion_evidence=[],
        audio_timeline=None,
        recurring_motifs=[],
        unresolved_ranges=[],
        contradictions=[],
        coverage_matrix={"structure": 1.0},
        confidence_summary={"structure": 0.9},
        provenance_graph=[],
        evidence_refs=["evidence_legacy_scenes"],
        status=VideoAnatomyStatus.PARTIAL,
        incompleteness_reasons=["semantic provider was not requested"],
        provenance=_provenance(),
    )

    assert anatomy.status is VideoAnatomyStatus.PARTIAL
    assert anatomy.plan.plan_sha256 == anatomy.structure.plan_sha256
    assert not hasattr(anatomy, "patch")
