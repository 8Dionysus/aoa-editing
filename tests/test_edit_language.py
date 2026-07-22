from __future__ import annotations

import pytest
from PIL import Image

from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    BlendEffect,
    BlurEffect,
    Clip,
    DissolveEffect,
    FrameRange,
    GlowEffect,
    Intent,
    Scenario,
    SpeedEffect,
    TextEffect,
    Timeline,
    Track,
    TrackKind,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.render.compiler import FFmpegCompiler
from aoa_editing.render.service import RenderService


def _settings(tmp_path):  # type: ignore[no-untyped-def]
    return Settings.for_home(tmp_path / "editing-home")


def test_general_effect_language_compiles_without_scenario_branches(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "graphic.png"
    Image.new("RGB", (640, 360), "#36536b").save(source)
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Effect language",
        Intent(text="Exercise reusable effects", scenario=Scenario.STILL_MOTION),
    )
    asset, _ = editing.ingest(project.id, source)
    version = store.create_initial_version(
        project.id,
        Timeline(
            width=640,
            height=360,
            duration_frames=30,
            tracks=[
                Track(
                    kind=TrackKind.VIDEO,
                    name="Reusable effects",
                    clips=[
                        Clip(
                            asset_id=asset.id,
                            timeline_range=FrameRange(start=0, duration=30),
                            effects=[
                                SpeedEffect(rate=32.0),
                                BlurEffect(sigma=1.5),
                                GlowEffect(radius=3, intensity=0.25),
                                BlendEffect(mode="screen", opacity=0.8),
                                DissolveEffect(fade_in_frames=3, fade_out_frames=4),
                                TextEffect(text="Review me", x=0.5, y=0.8),
                            ],
                        )
                    ],
                ),
                Track(
                    kind=TrackKind.CAPTION,
                    name="Captions",
                    clips=[
                        Clip(
                            asset_id=asset.id,
                            timeline_range=FrameRange(start=5, duration=15),
                            role="caption",
                            effects=[TextEffect(text="Measured cue")],
                        )
                    ],
                ),
            ],
        ),
    )

    plan = FFmpegCompiler(store).compile(
        project.id, version, "preview", tmp_path / "effects.partial.mp4"
    )

    assert "setpts=(PTS-STARTPTS)/32.0" in plan.filter_graph
    assert "gblur=sigma=1.5" in plan.filter_graph
    assert "all_mode=screen" in plan.filter_graph
    assert "fade=t=in" in plan.filter_graph
    assert "drawtext=" in plan.filter_graph
    assert "enable='between(t," in plan.filter_graph
    receipt = RenderService(store).render(project.id, version.id, profile="preview")
    assert receipt.status == "succeeded"


def test_speed_effect_supports_bounded_agent_workflow_timelapse() -> None:
    assert SpeedEffect(rate=64.0).rate == 64.0
    with pytest.raises(ValueError):
        SpeedEffect(rate=64.01)
