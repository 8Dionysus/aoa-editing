from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Clip,
    FrameRange,
    FrameRate,
    Intent,
    ScalarKeyframe,
    Scenario,
    Timeline,
    Track,
    TrackKind,
    TransformEffect,
    Vec2Keyframe,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.render.compiler import FFmpegCompiler
from aoa_editing.render.service import RenderService


def _settings(tmp_path: Path) -> Settings:
    return Settings.for_home(tmp_path / "editing-home")


def _square_fixture(path: Path) -> None:
    image = Image.new("RGB", (512, 512), "#dc4020")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 511, 511), outline="#f6dd45", width=24)
    draw.ellipse((144, 144, 368, 368), fill="#205bdc")
    for coordinate in range(32, 512, 16):
        draw.line((coordinate, 24, coordinate, 488), fill="#ef8a32", width=2)
        draw.line((24, coordinate, 488, coordinate), fill="#ef8a32", width=2)
    image.save(path)


def _extract_frame(video: Path, output: Path, frame: int) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-frames:v",
            "1",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_contain_camera_reveals_the_whole_square_on_a_portrait_canvas(tmp_path: Path) -> None:
    source = tmp_path / "transfer-square.png"
    _square_fixture(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Virtual camera transfer fixture",
        Intent(text="Zoom out to the complete square", scenario=Scenario.STILL_MOTION),
    )
    asset, _ = editing.ingest(project.id, source)
    duration = 68
    transform = TransformEffect(
        fit_mode="contain",
        position_mode="canvas_center",
        position=[
            Vec2Keyframe(frame=0, x=0.5, y=0.5),
            Vec2Keyframe(frame=duration - 1, x=0.5, y=0.5, easing="ease_in_out"),
        ],
        scale=[
            ScalarKeyframe(frame=0, value=2.0),
            ScalarKeyframe(frame=duration - 1, value=1.0, easing="ease_in_out"),
        ],
        rotation=[
            ScalarKeyframe(
                frame=frame,
                value=-7.0 * (1 - frame / (duration - 1)),
            )
            for frame in range(duration)
        ],
    )
    version = editing.initialize_timeline(
        project.id,
        duration_frames=duration,
        width=180,
        height=320,
        frame_rate=FrameRate(numerator=10),
    )
    timeline = Timeline(
        width=180,
        height=320,
        frame_rate=FrameRate(numerator=10),
        duration_frames=duration,
        tracks=[
            Track(
                kind=TrackKind.VIDEO,
                name="General contain camera",
                clips=[
                    Clip(
                        asset_id=asset.id,
                        timeline_range=FrameRange(start=0, duration=duration),
                        effects=[transform],
                    )
                ],
            )
        ],
    )
    updated = version.model_copy(update={"timeline": timeline})
    # Initial versions are immutable, so create a normal child version through a patch-sized save.
    child = updated.model_copy(
        update={
            "id": "version_" + "1" * 32,
            "parent_version_id": version.id,
            "message": "Apply general contain camera",
        }
    )
    store.save_version(child)

    plan = FFmpegCompiler(store).compile(
        project.id, child, "preview", tmp_path / "preview.partial.mp4"
    )
    assert "force_original_aspect_ratio=decrease" in plan.filter_graph
    assert "pad=360:640" in plan.filter_graph
    assert "rotate=" in plan.filter_graph
    assert "zoompan=" in plan.filter_graph
    assert plan.filter_graph.index("rotate=") < plan.filter_graph.index("pad=360:640")

    receipt = RenderService(store).render(project.id, child.id, profile="preview")
    video = store.project_path(project.id) / receipt.output_paths[0]
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    _extract_frame(video, first, 0)
    _extract_frame(video, last, duration - 1)
    with Image.open(first) as image:
        assert max(image.convert("RGB").getpixel((90, 20))) > 20
    with Image.open(last) as image:
        rgb = image.convert("RGB")
        assert max(rgb.getpixel((90, 20))) < 12
        assert max(rgb.getpixel((90, 160))) > 40
