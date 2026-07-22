from __future__ import annotations

import subprocess

from PIL import Image, ImageDraw

from aoa_editing.analysis.ports import FunctionAnalyzerPort
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    EvidenceAuthority,
    EvidenceRecord,
    FrameRate,
    Intent,
    Provenance,
    Scenario,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.scenarios.planners import memory_montage, speech_clean, still_motion


def _settings(tmp_path):  # type: ignore[no-untyped-def]
    return Settings.for_home(tmp_path / "editing-home")


def _still(path) -> None:  # type: ignore[no-untyped-def]
    image = Image.new("RGB", (320, 180), "#14213d")
    draw = ImageDraw.Draw(image)
    draw.ellipse((95, 25, 235, 165), fill="#fca311")
    draw.rectangle((140, 65, 180, 150), fill="#e5e5e5")
    image.save(path)


def _speech_video(path) -> None:  # type: ignore[no-untyped-def]
    audio = "aevalsrc=if(between(t\\,1\\,2)\\,0\\,0.2*sin(2*PI*440*t)):s=48000:d=3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            audio,
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def test_still_analysis_creates_nonsemantic_layers_and_accepted_version(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "shapes.png"
    _still(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Still", Intent(text="Gentle parallax", scenario=Scenario.STILL_MOTION)
    )
    asset, _ = editing.ingest(project.id, source)
    base = editing.initialize_timeline(project.id, duration_frames=180)
    evidence = AnalysisService(store).analyze(project.id, asset.id)
    layers = next(record for record in evidence if record.kind == "image.depth_layers")
    assert layers.payload["semantic_claim"] is False
    assert len(layers.artifacts) == 2
    treatment = still_motion(project.id, base, asset, store.list_evidence(project.id))
    editing.save_treatment(treatment)
    accepted = editing.accept_treatment(project.id, treatment.id)
    assert len(accepted.timeline.tracks) == 3
    assert accepted.applied_patch_id is not None


def test_speech_clean_uses_measured_silence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "speech.mp4"
    _speech_video(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Speech", Intent(text="Remove dead air", scenario=Scenario.SPEECH_CLEAN)
    )
    asset, _ = editing.ingest(project.id, source)
    base = editing.initialize_timeline(
        project.id,
        duration_frames=90,
        width=320,
        height=180,
        frame_rate=FrameRate(numerator=30),
    )
    evidence = AnalysisService(store).analyze(project.id, asset.id)
    transcript = EvidenceRecord(
        project_id=project.id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="speech.transcript",
        payload={
            "segments": [
                    {"start": 0.1, "end": 0.8, "text": "Measured words", "speaker": "A"}
            ]
        },
        provenance=Provenance(tool="fixture-asr", tool_version="1"),
    )
    evidence.append(transcript)
    treatment = speech_clean(project.id, base, asset, evidence)
    assert treatment.patch.evidence_refs
    assert treatment.patch.operations[0].value < 90
    caption_track = next(
        item for item in treatment.patch.operations[1].value if item["kind"] == "caption"
    )
    assert caption_track["clips"][0]["effects"][0]["text"] == "Measured words"
    editing.save_treatment(treatment)
    accepted = editing.accept_treatment(project.id, treatment.id)
    assert accepted.timeline.duration_frames < base.timeline.duration_frames


def test_memory_montage_uses_scene_boundaries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "speech.mp4"
    _speech_video(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Memory", Intent(text="Make a montage", scenario=Scenario.MEMORY_MONTAGE)
    )
    asset, _ = editing.ingest(project.id, source)
    base = editing.initialize_timeline(
        project.id,
        duration_frames=90,
        width=320,
        height=180,
        frame_rate=FrameRate(numerator=30),
    )
    scenes = EvidenceRecord(
        project_id=project.id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="video.scenes",
        payload={"cut_times_seconds": [1.0, 2.0], "threshold": 0.32, "count": 2},
        provenance=Provenance(tool="fixture", tool_version="1"),
    )
    treatment = memory_montage(project.id, base, [asset], [scenes])
    clips = treatment.patch.operations[1].value[0]["clips"]
    assert len(clips) == 3
    assert treatment.patch.evidence_refs == [scenes.id]


def test_analysis_localizes_adapter_failure_and_preserves_sibling_evidence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "speech.mp4"
    _speech_video(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Resilient analysis",
        Intent(text="Keep partial evidence", scenario=Scenario.SPEECH_CLEAN),
    )
    asset, _ = editing.ingest(project.id, source)

    def fail_loudness(_project_id: str, _asset: object) -> EvidenceRecord:
        raise RuntimeError("fixture analyzer unavailable")

    records = AnalysisService(
        store,
        ports=[FunctionAnalyzerPort(kind="audio.loudness", function=fail_loudness)],
    ).analyze(project.id, asset.id)

    assert any(record.kind == "audio.silence" for record in records)
    failure = next(record for record in records if record.kind == "analysis.failure")
    assert failure.payload["failed_kind"] == "audio.loudness"
    assert failure.payload["localized"] is True
    persisted = store.list_effective_evidence(project.id)
    assert any(record.kind == "audio.silence" for record in persisted)
    assert any(record.id == failure.id for record in persisted)


def test_human_evidence_correction_wins_without_erasing_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "speech.mp4"
    _speech_video(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Corrections",
        Intent(text="Correct evidence", scenario=Scenario.SPEECH_CLEAN),
    )
    asset, _ = editing.ingest(project.id, source)
    measured = next(
        record
        for record in AnalysisService(store).analyze(project.id, asset.id)
        if record.kind == "audio.silence"
    )
    correction = editing.correct_evidence(
        project.id,
        asset_id=asset.id,
        kind="audio.silence",
        payload={"intervals": [], "editor_note": "the pause is intentional"},
        supersedes=[measured.id],
        rationale="Reviewed against the source",
    )

    assert correction.authority is EvidenceAuthority.HUMAN_CORRECTION
    assert {item.id for item in store.list_evidence(project.id)} >= {
        measured.id,
        correction.id,
    }
    effective = store.list_effective_evidence(project.id)
    assert correction.id in {item.id for item in effective}
    assert measured.id not in {item.id for item in effective}
