"""Executable planner for a reviewed terminal-workflow recording binding."""

from __future__ import annotations

from aoa_editing.domain.models import (
    Asset,
    AudioEffect,
    Clip,
    EditPatch,
    EvidenceRecord,
    FrameRange,
    MediaKind,
    PatchOperation,
    ProjectVersion,
    RationaleItem,
    ScalarKeyframe,
    Scenario,
    ScreenWorkflowEditSpec,
    ScreenWorkflowFocus,
    ScreenWorkflowPlan,
    ScreenWorkflowShotKind,
    SpeedEffect,
    TextEffect,
    Track,
    TrackKind,
    TransformEffect,
    Treatment,
    Vec2Keyframe,
)
from aoa_editing.scenarios.planners import PlanningError


def screen_workflow(
    project_id: str,
    base: ProjectVersion,
    asset: Asset,
    plan: ScreenWorkflowPlan,
    edit_spec: ScreenWorkflowEditSpec,
    binding_evidence: EvidenceRecord,
    *,
    voiceover_asset: Asset | None = None,
    voiceover_evidence: EvidenceRecord | None = None,
) -> Treatment:
    if asset.media_kind is not MediaKind.VIDEO:
        raise PlanningError("screen.workflow requires a video recording")
    if not plan.capture_ready:
        raise PlanningError("screen workflow plan must be reviewed for capture")
    if edit_spec.plan_id != plan.id or edit_spec.script_sha256 != plan.script_sha256:
        raise PlanningError("screen workflow edit spec does not match the capture plan")
    if edit_spec.asset_id != asset.id or binding_evidence.asset_id != asset.id:
        raise PlanningError("screen workflow binding does not match the selected asset")
    asset_rate = asset.metadata.frame_rate
    if asset_rate is None or asset_rate != base.timeline.frame_rate:
        raise PlanningError("screen workflow source and timeline frame rates must match")
    expected_beats = {item.id for item in plan.beats}
    bound_beats = {item.beat_id for item in edit_spec.bindings}
    if expected_beats != bound_beats:
        raise PlanningError("every capture-plan beat must have one or more reviewed bindings")
    duration_seconds = asset.metadata.duration_seconds
    if duration_seconds is None:
        raise PlanningError("screen workflow source duration is unavailable")
    source_frames = round(duration_seconds * asset_rate.fps)
    if any(item.source_range.end > source_frames for item in edit_spec.bindings):
        raise PlanningError("screen workflow binding ends after the source recording")
    voiceover_timing = edit_spec.voiceover_timing
    if (voiceover_timing is None) != (voiceover_asset is None):
        raise PlanningError("voiceover timing and asset must be supplied together")
    if voiceover_timing is not None and voiceover_evidence is None:
        raise PlanningError("reviewed voiceover timing evidence is required")

    templates = {item.kind: item for item in plan.templates}
    picture_clips: list[Clip] = []
    audio_clips: list[Clip] = []
    caption_clips: list[Clip] = []
    cursor = 0
    for binding in edit_spec.bindings:
        timeline_range = FrameRange(start=cursor, duration=binding.timeline_duration_frames)
        effects: list[object] = [
            _transform(
                binding.focus_start,
                binding.focus_end,
                duration_frames=binding.timeline_duration_frames,
                transition_seconds=templates[binding.shot_kind].camera_transition_seconds,
                fps=base.timeline.frame_rate.fps,
                hard_cut=binding.shot_kind
                in {
                    ScreenWorkflowShotKind.ESTABLISH,
                    ScreenWorkflowShotKind.STEP_CUT,
                },
            )
        ]
        if binding.speed != 1.0:
            effects.append(SpeedEffect(rate=binding.speed))
        picture_clips.append(
            Clip(
                asset_id=asset.id,
                timeline_range=timeline_range,
                source_range=binding.source_range,
                role=(
                    f"screen-workflow:{binding.beat_id}:{binding.segment_order}:"
                    f"{binding.shot_kind.value}"
                ),
                effects=effects,  # type: ignore[arg-type]
                evidence_refs=[binding_evidence.id],
            )
        )
        if binding.include_source_audio:
            if not asset.metadata.has_audio:
                raise PlanningError("a binding requests source audio from a silent recording")
            audio_effects: list[object] = [
                AudioEffect(
                    fade_in_frames=min(3, binding.timeline_duration_frames // 4),
                    fade_out_frames=min(3, binding.timeline_duration_frames // 4),
                )
            ]
            if binding.speed != 1.0:
                audio_effects.append(SpeedEffect(rate=binding.speed))
            audio_clips.append(
                Clip(
                    asset_id=asset.id,
                    timeline_range=timeline_range,
                    source_range=binding.source_range,
                    role=f"screen-workflow-audio:{binding.beat_id}:{binding.segment_order}",
                    effects=audio_effects,  # type: ignore[arg-type]
                    evidence_refs=[binding_evidence.id],
                )
            )
        if binding.caption_text:
            caption_clips.append(
                Clip(
                    asset_id=asset.id,
                    timeline_range=timeline_range,
                    role=f"screen-workflow-caption:{binding.beat_id}:{binding.segment_order}",
                    effects=[TextEffect(text=binding.caption_text, y=0.86)],
                    evidence_refs=[binding_evidence.id],
                )
            )
        cursor += binding.timeline_duration_frames

    tracks = [Track(kind=TrackKind.VIDEO, name="Screen workflow", clips=picture_clips)]
    if audio_clips:
        tracks.append(Track(kind=TrackKind.AUDIO, name="Reviewed source audio", clips=audio_clips))
    if voiceover_timing is not None and voiceover_asset is not None and voiceover_evidence:
        narration_duration = voiceover_timing.duration_frames
        if narration_duration != cursor:
            raise PlanningError("reviewed voiceover must cover the complete screen edit")
        tracks.append(
            Track(
                kind=TrackKind.AUDIO,
                name="Reviewed voiceover",
                clips=[
                    Clip(
                        asset_id=voiceover_asset.id,
                        timeline_range=FrameRange(start=0, duration=narration_duration),
                        source_range=FrameRange(start=0, duration=narration_duration),
                        role=f"screen-workflow-voiceover:{voiceover_timing.id}",
                        effects=[
                            AudioEffect(
                                fade_in_frames=min(2, narration_duration // 4),
                                fade_out_frames=min(2, narration_duration // 4),
                                normalize_lufs=voiceover_timing.normalization_target_lufs,
                            )
                        ],
                        evidence_refs=[
                            voiceover_evidence.id,
                            *voiceover_timing.analysis_evidence_refs,
                        ],
                    )
                ],
            )
        )
    if caption_clips:
        tracks.append(
            Track(kind=TrackKind.CAPTION, name="Safe-area workflow captions", clips=caption_clips)
        )
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=[
            PatchOperation(op="replace", path="/duration_frames", value=cursor),
            PatchOperation(op="replace", path="/background", value="#080A0F"),
            PatchOperation(
                op="replace",
                path="/tracks",
                value=[track.model_dump(mode="json") for track in tracks],
            ),
        ],
        rationale=(
            "Assemble the reviewed prompt/action/result bindings and move the camera only "
            "between semantic screen states."
        ),
        evidence_refs=[
            binding_evidence.id,
            *([voiceover_evidence.id] if voiceover_evidence is not None else []),
        ],
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.SCREEN_WORKFLOW,
        title="Reviewed terminal workflow",
        summary=(
            f"Assemble {len(picture_clips)} script-bound screen beats over "
            f"{cursor / base.timeline.frame_rate.fps:.2f}s."
        ),
        structure=[
            "wide workspace orientation",
            "semantic action focus",
            "readable result proof",
            "source-contiguous acceleration within each ongoing workflow process",
            "hard cuts only between genuinely distinct workflow states",
        ],
        duration_frames=cursor,
        used_asset_ids=[
            asset.id,
            *([voiceover_asset.id] if voiceover_asset is not None else []),
        ],
        strengths=[
            "Script and capture stay linked by immutable beat ids",
            "Every source range and focus target is human reviewed",
            "Continuous speed tiers retain every source frame in their reviewed group",
            "Camera motion is ordinary reversible timeline data",
            *(
                ["Narration timing is attributable, reviewed, and bound to immutable audio"]
                if voiceover_timing is not None
                else []
            ),
        ],
        tradeoffs=["Legacy transform curves are used until video-safe Motion v2 is proven"],
        uncertainties=(
            []
            if voiceover_timing is not None
            else ["Fine rhythm still depends on the later narration performance"]
        ),
        risks=[
            "Over-tight focus can hide context or expose low-resolution terminal text",
            "Speeds above 24x can become perceptually indistinguishable from a cut",
        ],
        expected_style="Modern terminal-first agent workflow with restrained semantic camera moves",
        rationale=[
            RationaleItem(
                claim=(
                    "The edit uses only reviewed source bindings and never admits reference "
                    "media into project or render lineage."
                ),
                evidence_refs=[binding_evidence.id],
                confidence=1.0,
            )
        ],
        patch=patch,
        alternatives=[
            "Longer read holds with fewer close-ups",
            "Hard-cut-only technical walkthrough",
            "Narration-led version with source audio removed",
        ],
        planner=edit_spec.provenance,
    )


def _transform(
    start: ScreenWorkflowFocus,
    end: ScreenWorkflowFocus,
    *,
    duration_frames: int,
    transition_seconds: float,
    fps: float,
    hard_cut: bool,
) -> TransformEffect:
    if hard_cut or duration_frames == 1:
        return TransformEffect(
            fit_mode="cover",
            position_mode="canvas_center",
            position=[Vec2Keyframe(frame=0, x=end.center_x, y=end.center_y)],
            scale=[ScalarKeyframe(frame=0, value=end.scale)],
        )
    transition = min(duration_frames - 1, max(1, round(transition_seconds * fps)))
    same_position = start.center_x == end.center_x and start.center_y == end.center_y
    same_scale = start.scale == end.scale
    return TransformEffect(
        fit_mode="cover",
        position_mode="canvas_center",
        position=(
            [Vec2Keyframe(frame=0, x=start.center_x, y=start.center_y)]
            if same_position
            else [
                Vec2Keyframe(frame=0, x=start.center_x, y=start.center_y),
                Vec2Keyframe(
                    frame=transition,
                    x=end.center_x,
                    y=end.center_y,
                    easing="ease_in_out",
                ),
            ]
        ),
        scale=(
            [ScalarKeyframe(frame=0, value=start.scale)]
            if same_scale
            else [
                ScalarKeyframe(frame=0, value=start.scale),
                ScalarKeyframe(frame=transition, value=end.scale, easing="ease_in_out"),
            ]
        ),
    )
