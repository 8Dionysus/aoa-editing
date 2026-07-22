from __future__ import annotations

import subprocess

import pytest
from PIL import Image, ImageDraw

from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import FrameRange, FrameRate, Intent, Scenario
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderError, RenderService
from aoa_editing.scenarios.planners import still_motion


def _settings(tmp_path):  # type: ignore[no-untyped-def]
    return Settings.for_home(tmp_path / "editing-home")


def _render_fixture(tmp_path):  # type: ignore[no-untyped-def]
    source = tmp_path / "scene.png"
    image = Image.new("RGB", (320, 180), "#102030")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 20, 250, 175), fill="#ff9900")
    draw.polygon([(160, 35), (100, 155), (225, 155)], fill="#f0f0f0")
    image.save(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Render fixture", Intent(text="Animate", scenario=Scenario.STILL_MOTION)
    )
    asset, _ = editing.ingest(project.id, source)
    base = editing.initialize_timeline(
        project.id,
        duration_frames=30,
        width=320,
        height=180,
        frame_rate=FrameRate(numerator=30),
    )
    AnalysisService(store).analyze(project.id, asset.id)
    treatment = still_motion(
        project.id, base, asset, store.list_evidence(project.id), duration_seconds=1
    )
    editing.save_treatment(treatment)
    version = editing.accept_treatment(project.id, treatment.id)
    return store, project, version


def test_still_motion_preview_and_qc(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, project, version = _render_fixture(tmp_path)
    receipt = RenderService(store).render(project.id, version.id, profile="preview")
    assert receipt.status.value == "succeeded"
    assert receipt.output_hashes
    report = QualityService(store).inspect(project.id, version.id, profile="preview")
    assert report.overall in {"pass", "warn"}
    assert not [check for check in report.checks if check.status == "fail"]
    kdenlive = KdenliveExporter(store).export(project.id, version.id)
    otio = OTIOExporter(store).export(project.id, version.id)
    assert kdenlive.validator is not None
    assert otio.validator is not None
    assert (store.project_path(project.id) / kdenlive.output_path).is_file()
    assert (store.project_path(project.id) / otio.output_path).is_file()


def test_failed_render_is_receipted_atomic_and_retryable(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    store, project, version = _render_fixture(tmp_path)
    real_run = subprocess.run

    def fail_ffmpeg(*args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="forced failure")

    monkeypatch.setattr("aoa_editing.render.service.subprocess.run", fail_ffmpeg)
    with pytest.raises(RenderError, match="forced failure"):
        RenderService(store).render(project.id, version.id, profile="preview")
    output = store.project_path(project.id) / "renders" / version.id / "preview"
    assert not (output / "video.partial.mp4").exists()
    assert not (output / "video.mp4").exists()
    failed = [job for job in store.list_jobs(project.id) if job.kind == "render.preview"]
    assert len(failed) == 1
    assert failed[0].status.value == "failed"

    monkeypatch.setattr("aoa_editing.render.service.subprocess.run", real_run)
    receipt = RenderService(store).render(project.id, version.id, profile="preview")
    assert receipt.status.value == "succeeded"
    assert (output / "video.mp4").is_file()


def test_point_render_creates_only_requested_frame_range(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, project, version = _render_fixture(tmp_path)
    receipt = RenderService(store).render(
        project.id,
        version.id,
        profile="preview",
        frame_range=FrameRange(start=7, duration=10),
    )
    assert receipt.kind == "render.preview.segment"
    relative = receipt.output_paths[0]
    assert "/segments/000000007-000000010/" in f"/{relative}"
    output = store.project_path(project.id) / relative
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(result.stdout.strip()) == 10
