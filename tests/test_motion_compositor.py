from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Clip,
    DecomposedTransformMotionV2,
    EditPatch,
    FrameRange,
    FrameRate,
    Intent,
    MatrixMotionCurveV2,
    MatrixMotionKeyframeV2,
    MatrixTransformMotionV2,
    MotionCouplingV2,
    MotionPhaseV2,
    MotionSamplingV2,
    MotionTimeV2,
    PatchOperation,
    ScalarMotionCurveV2,
    ScalarMotionKeyframeV2,
    Scenario,
    Track,
    TrackKind,
    TransformEffectV2,
    Vec2MotionCurveV2,
    Vec2MotionKeyframeV2,
    Vec2ValueV2,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.quality.service import QualityService
from aoa_editing.render.compiler import FFmpegCompiler
from aoa_editing.render.service import RenderService


def _motion_effect(duration: int) -> TransformEffectV2:
    start = MotionTimeV2(frame=0)
    end = MotionTimeV2(frame=duration - 1)
    return TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            fit_mode="contain",
            position=Vec2MotionCurveV2(
                continuity="c1",
                keyframes=[
                    Vec2MotionKeyframeV2(
                        time=start,
                        value=Vec2ValueV2(x=0.32, y=0.62),
                        interpolation="monotone_cubic",
                    ),
                    Vec2MotionKeyframeV2(
                        time=end,
                        value=Vec2ValueV2(x=0.5, y=0.5),
                    ),
                ],
            ),
            scale=ScalarMotionCurveV2(
                continuity="c1",
                monotonicity="decreasing",
                overshoot_policy="forbid",
                keyframes=[
                    ScalarMotionKeyframeV2(
                        time=start,
                        value=2.2,
                        interpolation="monotone_cubic",
                    ),
                    ScalarMotionKeyframeV2(time=end, value=1.0),
                ],
            ),
            rotation=ScalarMotionCurveV2(
                angle_unwrap=True,
                keyframes=[
                    ScalarMotionKeyframeV2(
                        time=start,
                        value=-8.0,
                        interpolation="monotone_cubic",
                    ),
                    ScalarMotionKeyframeV2(time=end, value=0.0),
                ],
            ),
            pivot=Vec2MotionCurveV2(
                keyframes=[
                    Vec2MotionKeyframeV2(
                        time=start,
                        value=Vec2ValueV2(x=0.5, y=0.5),
                    ),
                    Vec2MotionKeyframeV2(
                        time=end,
                        value=Vec2ValueV2(x=0.5, y=0.5),
                    ),
                ]
            ),
        ),
        phases=[
            MotionPhaseV2(
                id="main-motion",
                start=start,
                end=end,
                channels=["position", "scale", "rotation"],
                intent="twist",
            )
        ],
        couplings=[
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["rotation"],
                relation="shared_progress",
                tolerance=0.01,
            )
        ],
        sampling=MotionSamplingV2(
            samples_per_frame=3,
            shutter_fraction=1.0,
            distribution="uniform",
        ),
    )


def _source(path: Path) -> None:
    image = Image.new("RGBA", (360, 360), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 355, 355), fill="#f0a020", outline="#fff060", width=8)
    draw.ellipse((90, 70, 290, 270), fill="#2048d0")
    draw.polygon(((180, 35), (90, 320), (315, 305)), fill="#d02030")
    image.save(path)


def _frame_count(path: Path) -> int:
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
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return int(result.stdout.strip())


def test_motion_v2_compiles_to_derived_matrices_and_renders_through_application_core(
    tmp_path: Path,
) -> None:
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    editing = EditingService(store)
    source = tmp_path / "generic-alpha-source.png"
    _source(source)
    project = editing.create_project(
        "Motion v2 compositor fixture",
        Intent(text="Use a smooth coupled reveal", scenario=Scenario.STILL_MOTION),
    )
    asset, _ = editing.ingest(project.id, source)
    duration = 25
    base = editing.initialize_timeline(
        project.id,
        duration_frames=duration,
        width=180,
        height=320,
        frame_rate=FrameRate(numerator=25),
    )
    effect = _motion_effect(duration)
    track = Track(
        kind=TrackKind.VIDEO,
        name="Canonical motion v2",
        clips=[
            Clip(
                asset_id=asset.id,
                timeline_range=FrameRange(start=0, duration=duration),
                effects=[effect],
            )
        ],
    )
    patch = EditPatch(
        project_id=project.id,
        base_version_id=base.id,
        operations=[
            PatchOperation(
                op="replace",
                path="/tracks",
                value=[track.model_dump(mode="json")],
            )
        ],
        rationale="Exercise the generic Motion Language v2 compositor path.",
    )
    version = editing.apply_patch(project.id, patch, "Approve generic motion v2")
    graph = store.load_decision_graph(project.id, version.decision_graph_id or "")
    assert graph.state == "approved"
    assert graph.decisions[0].reversible_operations

    compiler = FFmpegCompiler(store)
    output = tmp_path / "compiled.partial.mp4"
    first_plan = compiler.compile(project.id, version, "preview", output)
    second_plan = compiler.compile(project.id, version, "preview", output)
    assert first_plan.compositor_jobs == second_plan.compositor_jobs
    assert len(first_plan.compositor_jobs) == 1
    assert len(first_plan.compositor_jobs[0].frames) == duration
    assert all(len(frame.matrices_3x3) == 3 for frame in first_plan.compositor_jobs[0].frames)
    assert first_plan.compositor_jobs[0].source_sha256 == asset.sha256
    assert "zoompan=" not in first_plan.filter_graph
    assert "rotate=" not in first_plan.filter_graph
    assert str(first_plan.compositor_jobs[0].output_path) in first_plan.command

    first_receipt = RenderService(store).render(project.id, version.id, profile="preview")
    first_hash = first_receipt.output_hashes[0]
    rendered = store.project_path(project.id) / first_receipt.output_paths[0]
    assert _frame_count(rendered) == duration
    render_root = rendered.parent
    plan_payload = json.loads((render_root / "render-plan.json").read_text(encoding="utf-8"))
    assert plan_payload["compositor_jobs"][0]["source_sha256"] == asset.sha256
    assert not Path(plan_payload["compositor_jobs"][0]["output_path"]).exists()
    lineage = json.loads((render_root / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["input_hashes"] == [asset.sha256]
    assert lineage["forbidden_hashes_present"] == []
    assert lineage["compositor_jobs"][0]["motion_language_version"] == "2.0.0"
    quality = QualityService(store).inspect(project.id, version.id, profile="preview")
    assert not [check for check in quality.checks if check.status == "fail"]

    kdenlive = KdenliveExporter(store).export(project.id, version.id)
    assert kdenlive.validator is not None
    assert (
        next(item for item in kdenlive.items if item.feature == "Motion Language v2 curves").status
        == "approximated"
    )
    kdenlive_root = ET.parse(store.project_path(project.id) / kdenlive.output_path).getroot()
    effects_property = next(
        item
        for item in kdenlive_root.findall(".//property")
        if item.attrib.get("name") == "aoa_editing:effects_json"
    )
    assert json.loads(effects_property.text or "[]") == [effect.model_dump(mode="json")]

    otio_report = OTIOExporter(store).export(project.id, version.id)
    otio_payload = json.loads(
        (store.project_path(project.id) / otio_report.output_path).read_text(encoding="utf-8")
    )
    otio_effects = otio_payload["tracks"]["children"][0]["children"][0]["metadata"]["aoa_editing"][
        "effects"
    ]
    assert [TransformEffectV2.model_validate(item) for item in otio_effects] == [effect]

    second_receipt = RenderService(store).render(project.id, version.id, profile="preview")
    assert second_receipt.output_hashes[0] == first_hash

    segment_range = FrameRange(start=7, duration=5)
    segment_plan = compiler.compile(
        project.id,
        version,
        "preview",
        tmp_path / "segment.partial.mp4",
        frame_range=segment_range,
    )
    segment_job = segment_plan.compositor_jobs[0]
    assert [frame.output_frame for frame in segment_job.frames] == list(range(5))
    assert [frame.timeline_frame for frame in segment_job.frames] == list(range(7, 12))
    assert [frame.matrices_3x3 for frame in segment_job.frames] == [
        frame.matrices_3x3 for frame in first_plan.compositor_jobs[0].frames[7:12]
    ]
    assert segment_job.encoder_command[segment_job.encoder_command.index("-frames:v") + 1] == "5"
    segment_receipt = RenderService(store).render(
        project.id,
        version.id,
        profile="preview",
        frame_range=segment_range,
    )
    segment_render = store.project_path(project.id) / segment_receipt.output_paths[0]
    assert _frame_count(segment_render) == 5
    segment_lineage = json.loads(
        (segment_render.parent / "lineage.json").read_text(encoding="utf-8")
    )
    assert segment_lineage["frame_range"] == {"start": 7, "duration": 5}
    assert segment_lineage["compositor_jobs"][0]["frame_count"] == 5

    restarted = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    loaded = restarted.load_version(project.id, version.id)
    loaded_effect = loaded.timeline.tracks[0].clips[0].effects[0]
    assert isinstance(loaded_effect, TransformEffectV2)
    assert loaded_effect == effect
    reverted = EditingService(restarted).revert_version(project.id, version.id)
    assert reverted.timeline.tracks == []


def test_direct_matrix_motion_scales_from_canonical_output_to_preview_profile(
    tmp_path: Path,
) -> None:
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    editing = EditingService(store)
    source = tmp_path / "direct-matrix-source.png"
    _source(source)
    project = editing.create_project(
        "Direct matrix preview fixture",
        Intent(text="Compile measured matrices", scenario=Scenario.STILL_MOTION),
    )
    asset, _ = editing.ingest(project.id, source)
    duration = 9
    base = editing.initialize_timeline(
        project.id,
        duration_frames=duration,
        width=1920,
        height=1080,
        frame_rate=FrameRate(numerator=25),
    )
    start = MotionTimeV2(frame=0)
    end = MotionTimeV2(frame=duration - 1)
    effect = TransformEffectV2(
        motion=MatrixTransformMotionV2(
            matrix=MatrixMotionCurveV2(
                keyframes=[
                    MatrixMotionKeyframeV2(
                        time=start,
                        matrix_3x3=((2.0, 0.0, 100.0), (0.0, 2.0, 40.0), (0.0, 0.0, 1.0)),
                    ),
                    MatrixMotionKeyframeV2(
                        time=end,
                        matrix_3x3=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                    ),
                ]
            )
        )
    )
    track = Track(
        kind=TrackKind.VIDEO,
        name="Measured direct matrices",
        clips=[
            Clip(
                asset_id=asset.id,
                timeline_range=FrameRange(start=0, duration=duration),
                effects=[effect],
            )
        ],
    )
    version = base.model_copy(
        update={
            "id": "version_" + "5" * 32,
            "parent_version_id": base.id,
            "timeline": base.timeline.model_copy(update={"tracks": [track]}),
            "message": "Direct matrix compile fixture",
        }
    )
    store.save_version(version)

    plan = FFmpegCompiler(store).compile(
        project.id,
        version,
        "preview",
        tmp_path / "direct-matrix.partial.mp4",
    )
    assert (plan.profile.width, plan.profile.height) == (960, 540)
    assert plan.compositor_jobs[0].frames[0].matrices_3x3 == [
        ((1.0, 0.0, 50.0), (0.0, 1.0, 20.0), (0.0, 0.0, 1.0))
    ]


def test_motion_v2_compiler_rejects_a_curve_that_does_not_cover_the_clip(
    tmp_path: Path,
) -> None:
    store = ProjectStore(Settings.for_home(tmp_path / "editing-home"))
    editing = EditingService(store)
    source = tmp_path / "source.png"
    _source(source)
    project = editing.create_project(
        "Invalid curve fixture",
        Intent(text="Reject partial motion", scenario=Scenario.STILL_MOTION),
    )
    asset, _ = editing.ingest(project.id, source)
    base = editing.initialize_timeline(
        project.id,
        duration_frames=25,
        width=180,
        height=320,
        frame_rate=FrameRate(numerator=25),
    )
    effect = _motion_effect(24)
    track = Track(
        kind=TrackKind.VIDEO,
        name="Partial motion",
        clips=[
            Clip(
                asset_id=asset.id,
                timeline_range=FrameRange(start=0, duration=25),
                effects=[effect],
            )
        ],
    )
    version = base.model_copy(
        update={
            "id": "version_" + "4" * 32,
            "parent_version_id": base.id,
            "timeline": base.timeline.model_copy(update={"tracks": [track]}),
            "message": "Invalid partial curve",
        }
    )
    store.save_version(version)

    try:
        FFmpegCompiler(store).compile(
            project.id,
            version,
            "preview",
            tmp_path / "invalid.partial.mp4",
        )
    except ValueError as error:
        assert "cover the complete clip" in str(error)
    else:  # pragma: no cover - contract failure
        raise AssertionError("partial motion curve was accepted")
