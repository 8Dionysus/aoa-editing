from __future__ import annotations

import subprocess
from pathlib import Path

from aoa_editing.analysis.video_anatomy import VideoAnatomyService
from aoa_editing.analysis.video_pipeline import VideoAnatomyPipelineService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    EvidenceRecord,
    Intent,
    Provenance,
    Scenario,
    VideoAnatomyProfile,
    VideoAnatomyStatus,
    VideoSampleRole,
    VideoTransitionType,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore


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


def test_structural_anatomy_detects_and_preserves_a_hard_cut(tmp_path: Path) -> None:
    source = tmp_path / "hard-cut.mp4"
    _hard_cut_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project = store.create_project(
        "Structural fixture",
        Intent(text="Measure the source", scenario=Scenario.MEMORY_MONTAGE),
    )
    asset, _ = store.ingest_asset(project.id, source)

    result = VideoAnatomyService(store).analyze_structural(project.id, asset.id)

    assert result.anatomy.status is VideoAnatomyStatus.COMPLETE
    assert result.structure.coverage.coverage_ratio == 1.0
    assert [shot.frame_range.duration for shot in result.structure.shots] == [60, 60]
    hard_cut = next(
        item
        for item in result.structure.transitions
        if item.transition_type is VideoTransitionType.HARD_CUT
    )
    assert hard_cut.frame == 60
    protected = [
        item
        for item in result.frame_manifest.samples
        if item.role
        in {
            VideoSampleRole.PRE_BOUNDARY,
            VideoSampleRole.BOUNDARY,
            VideoSampleRole.POST_BOUNDARY,
        }
    ]
    assert {item.source_frame_index for item in protected} == {59, 60, 61}
    assert all(item.kept and item.artifact_path for item in protected)
    assert result.frame_manifest.duplicate_count > 0
    assert result.contact_sheet_path.is_file()
    assert sha256_file(result.contact_sheet_path)

    restored = store.load_video_anatomy(project.id, result.plan.id)
    assert restored.anatomy_sha256 == result.anatomy.anatomy_sha256
    evidence_kinds = {item.kind for item in store.list_evidence(project.id)}
    assert {
        "video.scenes",
        "video.structure.detectors",
        "video.frame-samples",
        "video.structure",
        "video.anatomy",
    } <= evidence_kinds


def test_structural_pipeline_keeps_no_audio_and_no_ai_as_a_complete_baseline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hard-cut.mp4"
    _hard_cut_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project = store.create_project(
        "Standalone structural profile",
        Intent(text="No AI is required", scenario=Scenario.MEMORY_MONTAGE),
    )
    asset, _ = store.ingest_asset(project.id, source)

    result = VideoAnatomyPipelineService(store).analyze(project.id, asset.id)

    assert result.anatomy.status is VideoAnatomyStatus.COMPLETE
    assert result.anatomy.audio_timeline is not None
    assert result.anatomy.audio_timeline.partial is False
    assert result.anatomy.audio_timeline.events == []
    assert result.anatomy.coverage_matrix == {
        "structure": 1.0,
        "sampling": 1.0,
        "audio": 1.0,
    }
    assert store.load_project(project.id).current_version_id is None


def test_existing_video_scenes_evidence_is_reused_additively(tmp_path: Path) -> None:
    source = tmp_path / "legacy-scenes.mp4"
    _hard_cut_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project = store.create_project(
        "Legacy scene compatibility",
        Intent(text="Reuse prior evidence", scenario=Scenario.MEMORY_MONTAGE),
    )
    asset, _ = store.ingest_asset(project.id, source)
    assert asset.metadata.frame_rate is not None
    legacy = EvidenceRecord(
        project_id=project.id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="video.scenes",
        payload={"cut_times_seconds": [2.0], "threshold": 0.32, "count": 1},
        time_base=asset.metadata.frame_rate,
        ranges=[],
        provenance=Provenance(
            tool="legacy-fixture",
            tool_version="0.1.0",
            deterministic=True,
        ),
    )
    store.save_evidence(legacy)

    result = VideoAnatomyService(store).analyze_structural(project.id, asset.id)

    assert legacy.id in result.anatomy.evidence_refs
    detector = next(
        item
        for item in result.evidence_records
        if item.kind == "video.structure.detectors"
    )
    assert detector.payload["scene_candidate_evidence_id"] == legacy.id
    assert any(
        edge.source_ref == legacy.id and edge.target_ref == detector.id
        for edge in result.anatomy.provenance_graph
    )
    assert sum(item.kind == "video.scenes" for item in store.list_evidence(project.id)) == 1


def test_missing_vision_provider_localizes_failure_without_losing_structure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hard-cut.mp4"
    _hard_cut_video(source)
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    project = store.create_project(
        "Provider-neutral semantic profile",
        Intent(text="Describe when available", scenario=Scenario.MEMORY_MONTAGE),
    )
    asset, _ = store.ingest_asset(project.id, source)

    result = VideoAnatomyPipelineService(store).analyze(
        project.id,
        asset.id,
        profile=VideoAnatomyProfile.SEMANTIC,
    )

    assert result.anatomy.status is VideoAnatomyStatus.PARTIAL
    assert result.anatomy.structure.coverage.coverage_ratio == 1.0
    assert result.anatomy.visual_observations == []
    assert result.anatomy.coverage_matrix["visual"] == 0.0
    visual_record = next(
        item for item in result.evidence_records if item.kind == "video.visual-observations"
    )
    assert visual_record.payload["provider_failure_code"] == "provider_missing"
    assert visual_record.payload["structural_profile_preserved"] is True
    assert store.load_project(project.id).current_version_id is None
