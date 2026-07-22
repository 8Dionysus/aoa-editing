"""Evidence-backed deterministic planners for the three prototype scenarios."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from itertools import pairwise
from typing import Any, Literal

from aoa_editing.domain.models import (
    Asset,
    AudioEffect,
    CheckResult,
    Clip,
    ColorEffect,
    EditPatch,
    EvidenceRecord,
    FrameRange,
    MaskEffect,
    MediaKind,
    PatchOperation,
    ProjectVersion,
    Provenance,
    RationaleItem,
    ReferenceReconstructionSpec,
    ScalarKeyframe,
    Scenario,
    TechniqueApplicabilityDecisionV2,
    TechniqueCompositionEvidenceV2,
    TechniquePacket,
    TextEffect,
    Track,
    TrackKind,
    TransformEffect,
    Treatment,
    Vec2Keyframe,
    new_id,
)
from aoa_editing.knowledge.techniques import portable_motion_effect_v2


class PlanningError(ValueError):
    """The selected scenario lacks the source or evidence it requires."""


def creative_treatment_variants(primary: Treatment) -> list[Treatment]:
    """Materialize two executable alternatives for creatively divergent scenarios."""

    if primary.scenario not in {Scenario.STILL_MOTION, Scenario.MEMORY_MONTAGE}:
        return [primary]
    tracks_operation = next(
        (operation for operation in primary.patch.operations if operation.path == "/tracks"),
        None,
    )
    if tracks_operation is None or not isinstance(tracks_operation.value, list):
        return [primary]
    original_tracks = deepcopy(tracks_operation.value)
    if primary.scenario is Scenario.STILL_MOTION:
        restrained = deepcopy(original_tracks[:1])
        _set_restrained_push(restrained)
        reverse = deepcopy(original_tracks)
        _reverse_motion_curves(reverse)
        variants = [
            _variant(
                primary,
                "Restrained single-plane push",
                "Use one source plane and a quiet centered push without parallax masks.",
                restrained,
            ),
            _variant(
                primary,
                "Reverse parallax release",
                "Reverse the measured layer motion for a settling rather than advancing gesture.",
                reverse,
            ),
        ]
    else:
        reverse = deepcopy(original_tracks)
        _reorder_montage(reverse, reverse_order=True, sparse=False)
        sparse = deepcopy(original_tracks)
        _reorder_montage(sparse, reverse_order=False, sparse=True)
        variants = [
            _variant(
                primary,
                "Reverse chronology",
                "Traverse measured memory beats from the latest source region backward.",
                reverse,
            ),
            _variant(
                primary,
                "Sparse contemplative montage",
                "Use alternating measured beats to create more breathing room.",
                sparse,
            ),
        ]
    return [primary, *variants]


def _planner() -> Provenance:
    return Provenance(
        tool="aoa-editing",
        tool_version="0.1.0",
        parameters={"planner": "deterministic-baseline-v1"},
        deterministic=True,
    )


def _frames(seconds: float, version: ProjectVersion) -> int:
    return max(1, round(seconds * version.timeline.frame_rate.fps))


def _evidence(records: Iterable[EvidenceRecord], kind: str, asset_id: str) -> EvidenceRecord:
    try:
        return next(item for item in records if item.kind == kind and item.asset_id == asset_id)
    except StopIteration as error:
        raise PlanningError(f"missing {kind} evidence for {asset_id}") from error


def speech_clean(
    project_id: str,
    base: ProjectVersion,
    asset: Asset,
    evidence: list[EvidenceRecord],
) -> Treatment:
    if not asset.metadata.has_audio or asset.media_kind is MediaKind.IMAGE:
        raise PlanningError("speech.clean requires an audio-bearing source")
    silence = _evidence(evidence, "audio.silence", asset.id)
    duration = asset.metadata.duration_seconds
    if not duration:
        raise PlanningError("source duration is unavailable")
    intervals = silence.payload.get("intervals", [])
    cursor = 0.0
    kept: list[tuple[float, float]] = []
    padding = 0.08
    for interval in intervals:
        start = max(cursor, float(interval["start_seconds"]) - padding)
        if start - cursor >= 0.12:
            kept.append((cursor, start))
        cursor = min(duration, float(interval["end_seconds"]) + padding)
    if duration - cursor >= 0.12:
        kept.append((cursor, duration))
    if not kept:
        kept = [(0.0, duration)]
    fps = base.timeline.frame_rate.fps
    timeline_cursor = 0
    kept_mapping: list[tuple[float, float, int]] = []
    video_clips: list[Clip] = []
    audio_clips: list[Clip] = []
    for start, end in kept:
        output_start = timeline_cursor
        source_start = round(start * fps)
        clip_duration = max(1, round((end - start) * fps))
        timeline_range = FrameRange(start=timeline_cursor, duration=clip_duration)
        source_range = FrameRange(start=source_start, duration=clip_duration)
        if asset.metadata.has_video:
            video_clips.append(
                Clip(
                    asset_id=asset.id,
                    timeline_range=timeline_range,
                    source_range=source_range,
                    role="speech-picture",
                    evidence_refs=[silence.id],
                )
            )
        audio_clips.append(
            Clip(
                asset_id=asset.id,
                timeline_range=timeline_range,
                source_range=source_range,
                role="speech-audio",
                effects=[
                    AudioEffect(
                        fade_in_frames=min(3, clip_duration // 4),
                        fade_out_frames=min(3, clip_duration // 4),
                        normalize_lufs=-16,
                    )
                ],
                evidence_refs=[silence.id],
            )
        )
        timeline_cursor += clip_duration
        kept_mapping.append((start, end, output_start))
    tracks: list[Track] = []
    if video_clips:
        tracks.append(Track(kind=TrackKind.VIDEO, name="Clean picture", clips=video_clips))
    tracks.append(Track(kind=TrackKind.AUDIO, name="Clean speech", clips=audio_clips))
    transcript = next(
        (
            item
            for item in evidence
            if item.asset_id == asset.id and item.kind == "speech.transcript"
        ),
        None,
    )
    caption_clips: list[Clip] = []
    if transcript is not None:
        for segment in _transcript_segments(transcript):
            for kept_start, kept_end, output_start in kept_mapping:
                start = max(float(segment["start"]), kept_start)
                end = min(float(segment["end"]), kept_end)
                if end <= start:
                    continue
                cue_start = min(
                    timeline_cursor - 1,
                    output_start + round((start - kept_start) * fps),
                )
                cue_duration = max(1, round((end - start) * fps))
                caption_clips.append(
                    Clip(
                        asset_id=asset.id,
                        timeline_range=FrameRange(
                            start=cue_start,
                            duration=max(
                                1, min(cue_duration, timeline_cursor - cue_start)
                            ),
                        ),
                        source_range=FrameRange(
                            start=round(start * fps), duration=cue_duration
                        ),
                        role=f"caption:{segment.get('speaker') or 'speaker'}",
                        effects=[TextEffect(text=str(segment["text"]))],
                        evidence_refs=[transcript.id],
                    )
                )
    if caption_clips:
        tracks.append(
            Track(kind=TrackKind.CAPTION, name="Reviewed transcript", clips=caption_clips)
        )
    operations = [
        PatchOperation(op="replace", path="/duration_frames", value=max(1, timeline_cursor)),
        PatchOperation(
            op="replace",
            path="/tracks",
            value=[track.model_dump(mode="json") for track in tracks],
        ),
    ]
    removed = max(0.0, duration - timeline_cursor / fps)
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=operations,
        rationale=f"Remove measured silence while keeping {padding:.2f}s boundary padding.",
        evidence_refs=[silence.id, *([transcript.id] if transcript else [])],
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.SPEECH_CLEAN,
        title="Clean speech pauses",
        summary=f"Keep {len(kept)} speech regions and remove about {removed:.2f}s of silence.",
        structure=["measured speech regions", "linked picture and speech", "captions if present"],
        duration_frames=max(1, timeline_cursor),
        used_asset_ids=[asset.id],
        strengths=["Measured pause boundaries", "Natural boundary padding", "Live speech retained"],
        tradeoffs=["Repetition removal needs optional transcript evidence"],
        uncertainties=["Silence is not always editorially unwanted"],
        risks=["Aggressive thresholds can damage conversational rhythm"],
        expected_style="Concise, natural spoken delivery",
        rationale=[
            RationaleItem(
                claim="Cuts are derived from measured silence intervals with conservative padding.",
                evidence_refs=[silence.id],
                confidence=0.86,
            )
        ],
        patch=patch,
        alternatives=["Keep all pauses", "Use a larger conversational pause threshold"],
        planner=_planner(),
    )


def _transcript_segments(record: EvidenceRecord) -> list[dict[str, Any]]:
    payload = record.payload
    candidates = payload.get("segments") or payload.get("chunks")
    if candidates is None and isinstance(payload.get("result"), dict):
        result = payload["result"]
        candidates = result.get("segments") or result.get("chunks")
    if not isinstance(candidates, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        start = item.get("start", item.get("start_seconds"))
        end = item.get("end", item.get("end_seconds"))
        if start is None or end is None or float(end) <= float(start):
            continue
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "text": str(item["text"]).strip(),
                "speaker": item.get("speaker"),
            }
        )
    return segments


def memory_montage(
    project_id: str,
    base: ProjectVersion,
    assets: list[Asset],
    evidence: list[EvidenceRecord],
) -> Treatment:
    video_assets = [asset for asset in assets if asset.metadata.has_video]
    if not video_assets:
        raise PlanningError("memory.montage requires at least one video asset")
    fps = base.timeline.frame_rate.fps
    selected: list[Clip] = []
    selected_audio: list[Clip] = []
    evidence_refs: list[str] = []
    cursor = 0
    for asset in video_assets:
        scenes = _evidence(evidence, "video.scenes", asset.id)
        evidence_refs.append(scenes.id)
        duration = asset.metadata.duration_seconds or 0
        boundaries = [0.0, *scenes.payload.get("cut_times_seconds", []), duration]
        candidates = [
            (start, end)
            for start, end in pairwise(boundaries)
            if end - start >= 0.35
        ]
        if not candidates and duration > 0:
            candidates = [(0.0, duration)]
        for start, end in candidates[:6]:
            clip_duration = min(round((end - start) * fps), round(2.5 * fps))
            if clip_duration <= 0:
                continue
            timeline_range = FrameRange(start=cursor, duration=clip_duration)
            source_range = FrameRange(start=round(start * fps), duration=clip_duration)
            selected.append(
                Clip(
                    asset_id=asset.id,
                    timeline_range=timeline_range,
                    source_range=source_range,
                    role="memory-beat",
                    effects=[ColorEffect(saturation=0.92, contrast=1.04, vignette=0.12)],
                    evidence_refs=[scenes.id],
                )
            )
            if asset.metadata.has_audio:
                selected_audio.append(
                    Clip(
                        asset_id=asset.id,
                        timeline_range=timeline_range,
                        source_range=source_range,
                        role="live-sound",
                        effects=[
                            AudioEffect(
                                fade_in_frames=min(3, clip_duration // 4),
                                fade_out_frames=min(3, clip_duration // 4),
                                normalize_lufs=-18,
                            )
                        ],
                        evidence_refs=[scenes.id],
                    )
                )
            cursor += clip_duration
    if not selected:
        raise PlanningError("no usable montage regions were found")
    tracks = [Track(kind=TrackKind.VIDEO, name="Memory beats", clips=selected)]
    if selected_audio:
        tracks.append(
            Track(kind=TrackKind.AUDIO, name="Preserved live sound", clips=selected_audio)
        )
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=[
            PatchOperation(op="replace", path="/duration_frames", value=cursor),
            PatchOperation(
                op="replace",
                path="/tracks",
                value=[track.model_dump(mode="json") for track in tracks],
            ),
        ],
        rationale="Select concise chronological beats at measured scene boundaries.",
        evidence_refs=evidence_refs,
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.MEMORY_MONTAGE,
        title="Measured memory montage",
        summary=f"Build a {cursor / fps:.2f}s montage from {len(selected)} measured beats.",
        structure=["scene-derived beats", "chronological assembly", "preserved live sound"],
        duration_frames=cursor,
        used_asset_ids=sorted({clip.asset_id for clip in selected}),
        strengths=["Measured boundaries", "Bounded beat length", "Synchronized live sound"],
        tradeoffs=["Chronology is a fallback when semantic ranking is unavailable"],
        uncertainties=["Scene change is not the same as emotional importance"],
        risks=["Sparse source material can produce repetitive structure"],
        expected_style="Compact, reflective chronological montage",
        rationale=[
            RationaleItem(
                claim=(
                    "Clip boundaries originate from scene-change evidence, then receive a "
                    "bounded maximum length."
                ),
                evidence_refs=evidence_refs,
                confidence=0.78,
            )
        ],
        patch=patch,
        alternatives=[
            "Longer contemplative beats",
            "Reverse chronology",
            "Music-synchronized cut points",
        ],
        planner=_planner(),
    )


def still_motion(
    project_id: str,
    base: ProjectVersion,
    asset: Asset,
    evidence: list[EvidenceRecord],
    *,
    duration_seconds: float = 6.0,
) -> Treatment:
    if asset.media_kind is not MediaKind.IMAGE:
        raise PlanningError("still.motion requires an image source")
    statistics = _evidence(evidence, "image.statistics", asset.id)
    layers = _evidence(evidence, "image.depth_layers", asset.id)
    duration = _frames(duration_seconds, base)
    layer_payload = {item["role"]: item for item in layers.payload["layers"]}
    background = Clip(
        asset_id=asset.id,
        timeline_range=FrameRange(start=0, duration=duration),
        role="background",
        effects=[
            TransformEffect(
                position=[
                    Vec2Keyframe(frame=0, x=-0.015, y=0),
                    Vec2Keyframe(frame=duration - 1, x=0.015, y=0),
                ],
                scale=[
                    ScalarKeyframe(frame=0, value=1.08),
                    ScalarKeyframe(frame=duration - 1, value=1.16, easing="ease_in_out"),
                ],
            ),
            ColorEffect(contrast=1.04, saturation=1.03, vignette=0.10),
        ],
        evidence_refs=[statistics.id, layers.id],
    )
    tracks = [Track(kind=TrackKind.VIDEO, name="Background drift", clips=[background])]
    motions = {
        "midground": ((0.012, 0.004), (-0.012, -0.004), 1.10, 1.18),
        "foreground": ((0.025, -0.006), (-0.020, 0.006), 1.12, 1.22),
    }
    for role in ("midground", "foreground"):
        payload = layer_payload.get(role)
        if not payload or not payload.get("path"):
            continue
        start, end, start_scale, end_scale = motions[role]
        clip = Clip(
            asset_id=asset.id,
            timeline_range=FrameRange(start=0, duration=duration),
            role=role,
            effects=[
                MaskEffect(mask_asset_path=str(payload["path"]), feather=1.5),
                TransformEffect(
                    position=[
                        Vec2Keyframe(frame=0, x=start[0], y=start[1]),
                        Vec2Keyframe(frame=duration - 1, x=end[0], y=end[1], easing="ease_in_out"),
                    ],
                    scale=[
                        ScalarKeyframe(frame=0, value=start_scale),
                        ScalarKeyframe(
                            frame=duration - 1, value=end_scale, easing="ease_in_out"
                        ),
                    ],
                ),
            ],
            evidence_refs=[layers.id],
        )
        tracks.append(Track(kind=TrackKind.VIDEO, name=role.title(), clips=[clip]))
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=[
            PatchOperation(op="replace", path="/duration_frames", value=duration),
            PatchOperation(
                op="replace",
                path="/tracks",
                value=[track.model_dump(mode="json") for track in tracks],
            ),
        ],
        rationale="Create restrained depth-separated motion from measured image structure.",
        evidence_refs=[statistics.id, layers.id],
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.STILL_MOTION,
        title="Layered still motion",
        summary=(
            f"Animate {len(tracks)} depth layers over {duration_seconds:.2f}s with "
            "reversible keyframes."
        ),
        structure=["background drift", "measured saliency layers", "foreground parallax"],
        duration_frames=duration,
        used_asset_ids=[asset.id],
        strengths=["Source remains immutable", "Reversible keyframes", "Reusable parallax"],
        tradeoffs=["Deterministic masks are non-semantic proxies"],
        uncertainties=["Depth order is heuristic without a depth model"],
        risks=["Large motion can expose mask or source edges"],
        expected_style="Restrained cinematic parallax",
        rationale=[
            RationaleItem(
                claim=(
                    "Motion amplitude increases toward the foreground to create parallax "
                    "without inventing semantic segmentation."
                ),
                evidence_refs=[layers.id],
                confidence=0.68,
            ),
            RationaleItem(
                claim=(
                    "Color treatment is deliberately restrained and informed by measured "
                    "image statistics."
                ),
                evidence_refs=[statistics.id],
                confidence=0.60,
            ),
        ],
        patch=patch,
        alternatives=["Slow push-in only", "Vertical reveal", "Higher-energy orbit"],
        planner=_planner(),
    )


def reference_reconstruct(
    project_id: str,
    base: ProjectVersion,
    asset: Asset,
    evidence: list[EvidenceRecord],
    spec: ReferenceReconstructionSpec,
) -> Treatment:
    """Translate a frozen observation spec into an ordinary reversible treatment."""

    if asset.media_kind is not MediaKind.IMAGE:
        raise PlanningError("reference.reconstruct requires an image source")
    if asset.sha256 != spec.source_sha256:
        raise PlanningError("source hash does not match the frozen reconstruction spec")
    specification = _evidence(evidence, "reference.reconstruction_spec", asset.id)
    measurements = spec.camera_motion.measurements
    if not measurements or measurements[0].frame != 0:
        raise PlanningError("camera measurements must begin at frame zero")
    if measurements[-1].frame != spec.duration_frames - 1:
        raise PlanningError("camera measurements must include the final timeline frame")
    dense_curve = len(measurements) > 2

    def easing(index: int) -> str:
        if index == 0 or dense_curve:
            return "linear"
        return spec.camera_motion.curve_hint

    transform = TransformEffect(
        fit_mode="contain",
        position_mode="canvas_center",
        position=[
            Vec2Keyframe(
                frame=item.frame,
                x=item.center_x,
                y=item.center_y,
                easing=easing(index),  # type: ignore[arg-type]
            )
            for index, item in enumerate(measurements)
        ],
        scale=[
            ScalarKeyframe(
                frame=item.frame,
                value=item.scale_relative_to_contain,
                easing=easing(index),  # type: ignore[arg-type]
            )
            for index, item in enumerate(measurements)
        ],
        rotation=[
            ScalarKeyframe(
                frame=item.frame,
                value=item.rotation_degrees,
                easing=easing(index),  # type: ignore[arg-type]
            )
            for index, item in enumerate(measurements)
        ],
    )
    picture = Track(
        kind=TrackKind.VIDEO,
        name="Measured virtual camera",
        clips=[
            Clip(
                asset_id=asset.id,
                timeline_range=FrameRange(start=0, duration=spec.duration_frames),
                role="reference-reconstruction-primary",
                effects=[transform],
                evidence_refs=[specification.id],
            )
        ],
    )
    operations = [
        PatchOperation(op="replace", path="/width", value=spec.camera_motion.width),
        PatchOperation(op="replace", path="/height", value=spec.camera_motion.height),
        PatchOperation(
            op="replace",
            path="/frame_rate",
            value=spec.camera_motion.frame_rate.model_dump(mode="json"),
        ),
        PatchOperation(op="replace", path="/duration_frames", value=spec.duration_frames),
        PatchOperation(op="replace", path="/background", value="#000000"),
        PatchOperation(
            op="replace",
            path="/tracks",
            value=[picture.model_dump(mode="json")],
        ),
    ]
    confidence = min(item.inlier_ratio for item in measurements)
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=operations,
        rationale=(
            "Rebuild the frozen measured similarity trajectory from the permitted source "
            "image only."
        ),
        evidence_refs=[specification.id],
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.REFERENCE_RECONSTRUCT,
        title="Measured single-source reconstruction",
        summary=(
            f"Apply {len(measurements)} measured camera samples across "
            f"{spec.duration_frames} frames without reference media in render lineage."
        ),
        structure=[
            "single permitted source",
            "dense similarity-transform curve",
            "silent delivery",
        ],
        duration_frames=spec.duration_frames,
        used_asset_ids=[asset.id],
        strengths=["Frozen measurements", "Source-only lineage", "Frame-accurate trajectory"],
        tradeoffs=["A global transform cannot reproduce independent object motion"],
        uncertainties=list(spec.uncertainties),
        risks=["Low-feature frames can reduce registration confidence"],
        expected_style="Continuous detail-to-whole-image reveal",
        rationale=[
            RationaleItem(
                claim=(
                    "A single source-wide similarity transform explains the sampled "
                    "reference frames."
                ),
                evidence_refs=[specification.id],
                confidence=confidence,
            )
        ],
        patch=patch,
        alternatives=[
            "Use only endpoint keyframes with the inferred easing curve",
            "Use a denser sample interval if motion comparison exceeds tolerance",
        ],
        planner=Provenance(
            tool="aoa-editing",
            tool_version="0.1.0",
            parameters={
                "planner": "reference-spec-v1",
                "spec_id": spec.id,
                "reference_hash_used_as_media": False,
            },
            deterministic=True,
        ),
    )


def apply_camera_technique(
    project_id: str,
    base: ProjectVersion,
    asset: Asset,
    evidence: list[EvidenceRecord],
    technique: TechniquePacket,
    *,
    width: int,
    height: int,
) -> Treatment:
    """Apply a source-neutral Tree of Editing camera recipe as still.motion."""

    recipe = technique.camera_motion
    if recipe is None:
        raise PlanningError("technique has no operational camera motion recipe")
    if asset.media_kind is not MediaKind.IMAGE:
        raise PlanningError("camera motion technique requires an image source")
    if width * recipe.aspect_height != height * recipe.aspect_width:
        raise PlanningError("selected canvas does not match the technique aspect ratio")
    evidence_by_kind = {
        record.kind: record
        for record in evidence
        if record.asset_id == asset.id
    }
    missing = [kind for kind in technique.required_evidence if kind not in evidence_by_kind]
    if missing:
        raise PlanningError(f"technique evidence is missing: {missing}")
    evidence_refs = [evidence_by_kind[kind].id for kind in technique.required_evidence]
    transform = TransformEffect(
        fit_mode=recipe.fit_mode,
        position_mode=recipe.position_mode,
        position=recipe.position,
        scale=recipe.scale,
        rotation=recipe.rotation,
    )
    picture = Track(
        kind=TrackKind.VIDEO,
        name="Tree of Editing camera technique",
        clips=[
            Clip(
                asset_id=asset.id,
                timeline_range=FrameRange(start=0, duration=recipe.duration_frames),
                role="technique-primary",
                effects=[transform],
                evidence_refs=evidence_refs,
            )
        ],
    )
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=[
            PatchOperation(op="replace", path="/width", value=width),
            PatchOperation(op="replace", path="/height", value=height),
            PatchOperation(
                op="replace",
                path="/frame_rate",
                value=recipe.frame_rate.model_dump(mode="json"),
            ),
            PatchOperation(
                op="replace", path="/duration_frames", value=recipe.duration_frames
            ),
            PatchOperation(op="replace", path="/background", value="#000000"),
            PatchOperation(
                op="replace",
                path="/tracks",
                value=[picture.model_dump(mode="json")],
            ),
        ],
        rationale=(
            f"Apply source-neutral normalized camera recipe {technique.id} within its "
            "declared constraints."
        ),
        evidence_refs=evidence_refs,
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.STILL_MOTION,
        title=technique.title,
        summary=technique.intent,
        structure=["source-neutral technique packet", "normalized camera recipe"],
        duration_frames=recipe.duration_frames,
        used_asset_ids=[asset.id],
        strengths=list(technique.rules),
        tradeoffs=list(technique.constraints),
        uncertainties=list(technique.promotion_questions),
        risks=list(technique.failure_modes),
        expected_style=technique.title,
        rationale=[
            RationaleItem(
                claim=technique.rules[0],
                evidence_refs=evidence_refs,
                confidence=0.85,
            )
        ],
        patch=patch,
        alternatives=technique.creative_variants,
        planner=Provenance(
            tool="aoa-editing-tree-of-editing",
            tool_version="0.1.0",
            parameters={
                "technique_id": technique.id,
                "technique_revision": technique.revision,
                "status": technique.status,
            },
            deterministic=True,
        ),
    )


def assess_camera_technique_v2(
    project_id: str,
    asset: Asset,
    evidence: list[EvidenceRecord],
    technique: TechniquePacket,
    *,
    width: int,
    height: int,
) -> TechniqueApplicabilityDecisionV2:
    """Evaluate generic evidence and refuse before any Treatment or render exists."""

    recipe = technique.portable_camera_motion_v2
    resolution = technique.resolution_constraint_v2
    if recipe is None or resolution is None:
        raise PlanningError("technique has no portable v2 motion contract")
    if asset.media_kind is not MediaKind.IMAGE:
        raise PlanningError("portable camera motion requires an image source")
    if asset.metadata.width is None or asset.metadata.height is None:
        raise PlanningError("image dimensions are unavailable")

    evidence_by_kind = {
        record.kind: record
        for record in evidence
        if record.asset_id == asset.id and record.source_sha256 == asset.sha256
    }
    missing = [kind for kind in technique.required_evidence if kind not in evidence_by_kind]
    composition_record = evidence_by_kind.get("editing.composition_applicability.v2")
    composition: TechniqueCompositionEvidenceV2 | None = None
    if composition_record is not None:
        try:
            composition = TechniqueCompositionEvidenceV2.model_validate(
                composition_record.payload
            )
        except ValueError:
            missing.append("editing.composition_applicability.v2:malformed")

    aspect_matches = width * recipe.aspect_height == height * recipe.aspect_width
    curve_model = technique.curve_model_v2
    if curve_model is None:
        raise PlanningError("technique lacks its v2 curve model")
    authored_scale_min = min(recipe.scales_relative_to_contain)
    authored_scale_max = max(recipe.scales_relative_to_contain)
    maximum_scale = authored_scale_max + (
        (authored_scale_max - authored_scale_min)
        * curve_model.maximum_interpolation_overshoot_fraction
    )
    contain_fit = min(
        width / asset.metadata.width,
        height / asset.metadata.height,
    )
    effective_upscale = contain_fit * maximum_scale
    composition_ok = (
        composition is not None
        and composition.single_global_transform_sufficient
        and not composition.independent_layer_motion_required
        and composition.authority in {"fixture_ground_truth", "human_assertion"}
    )
    checks = [
        CheckResult(
            id="required-evidence",
            status="pass" if not missing else "fail",
            summary=(
                "all required evidence is present"
                if not missing
                else "required technique evidence is missing or malformed"
            ),
            measured={"missing": sorted(set(missing))},
        ),
        CheckResult(
            id="output-aspect",
            status="pass" if aspect_matches else "fail",
            summary=(
                "output aspect matches the authored technique"
                if aspect_matches
                else "output aspect needs an explicit reframe operation"
            ),
            measured={
                "canvas": [width, height],
                "required_aspect": [recipe.aspect_width, recipe.aspect_height],
            },
        ),
        CheckResult(
            id="composition-applicability",
            status="pass" if composition_ok else "fail",
            summary=(
                "one global transform is sufficient"
                if composition_ok
                else "composition evidence does not permit one global transform"
            ),
            measured=(
                composition.model_dump(mode="json")
                if composition is not None
                else {"available": False}
            ),
        ),
        CheckResult(
            id="resolution-guard",
            status=(
                "pass"
                if effective_upscale <= resolution.maximum_effective_upscale
                else "fail"
            ),
            summary=(
                "source pixel density supports the peak camera scale"
                if effective_upscale <= resolution.maximum_effective_upscale
                else "peak camera scale exceeds available source pixel density"
            ),
            measured={
                "source": [asset.metadata.width, asset.metadata.height],
                "canvas": [width, height],
                "contain_fit": contain_fit,
                "maximum_motion_scale": maximum_scale,
                "maximum_authored_scale": authored_scale_max,
                "interpolation_overshoot_allowance": (
                    curve_model.maximum_interpolation_overshoot_fraction
                ),
                "effective_upscale": effective_upscale,
                "maximum_effective_upscale": resolution.maximum_effective_upscale,
            },
        ),
    ]
    refusal_code: Literal[
        "insufficient_source_resolution",
        "inapplicable_composition",
        "missing_required_evidence",
        "output_aspect_mismatch",
    ] | None = None
    if missing:
        refusal_code = "missing_required_evidence"
    elif not aspect_matches:
        refusal_code = "output_aspect_mismatch"
    elif not composition_ok:
        refusal_code = "inapplicable_composition"
    elif effective_upscale > resolution.maximum_effective_upscale:
        refusal_code = "insufficient_source_resolution"
    eligible = refusal_code is None
    return TechniqueApplicabilityDecisionV2(
        project_id=project_id,
        asset_id=asset.id,
        technique_id=technique.id,
        technique_revision=technique.revision,
        outcome="eligible" if eligible else "refused",
        refusal_code=refusal_code,
        checks=checks,
        evidence_refs=[
            evidence_by_kind[kind].id
            for kind in technique.required_evidence
            if kind in evidence_by_kind
        ],
        measured={
            "source_class": composition.source_class if composition else "unknown",
            "source_sha256": asset.sha256,
            "effective_upscale": effective_upscale,
        },
        provenance=Provenance(
            tool="aoa-editing-technique-applicability-v2",
            tool_version="0.1.0",
            parameters={
                "technique_id": technique.id,
                "technique_revision": technique.revision,
                "fail_closed": True,
                "render_created": False,
            },
            deterministic=True,
        ),
    )


def apply_camera_technique_v2(
    project_id: str,
    base: ProjectVersion,
    asset: Asset,
    technique: TechniquePacket,
    decision: TechniqueApplicabilityDecisionV2,
    *,
    width: int,
    height: int,
) -> Treatment:
    """Create a normal reversible Treatment only after a passing typed decision."""

    recipe = technique.portable_camera_motion_v2
    if recipe is None:
        raise PlanningError("technique has no portable v2 motion contract")
    if (
        decision.outcome != "eligible"
        or decision.project_id != project_id
        or decision.asset_id != asset.id
        or decision.technique_id != technique.id
        or decision.technique_revision != technique.revision
    ):
        raise PlanningError("technique applicability decision is not eligible for this request")
    transform = portable_motion_effect_v2(technique)
    picture = Track(
        kind=TrackKind.VIDEO,
        name="Tree of Editing portable camera technique v2",
        clips=[
            Clip(
                asset_id=asset.id,
                timeline_range=FrameRange(start=0, duration=recipe.duration_frames),
                role="technique-primary",
                effects=[transform],
                evidence_refs=decision.evidence_refs,
            )
        ],
    )
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=[
            PatchOperation(op="replace", path="/width", value=width),
            PatchOperation(op="replace", path="/height", value=height),
            PatchOperation(
                op="replace",
                path="/frame_rate",
                value=recipe.frame_rate.model_dump(mode="json"),
            ),
            PatchOperation(
                op="replace", path="/duration_frames", value=recipe.duration_frames
            ),
            PatchOperation(op="replace", path="/background", value="#000000"),
            PatchOperation(
                op="replace",
                path="/tracks",
                value=[picture.model_dump(mode="json")],
            ),
        ],
        rationale=(
            f"Apply portable normalized motion {technique.id} after its typed "
            "resolution and composition gates passed."
        ),
        evidence_refs=decision.evidence_refs,
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.STILL_MOTION,
        title=technique.title,
        summary=technique.intent,
        structure=[
            "source-neutral technique packet v2",
            "dense phase-aware normalized camera recipe",
            "pre-render applicability decision",
        ],
        duration_frames=recipe.duration_frames,
        used_asset_ids=[asset.id],
        strengths=list(technique.rules),
        tradeoffs=list(technique.constraints),
        uncertainties=list(technique.promotion_questions),
        risks=list(technique.failure_modes),
        expected_style=technique.title,
        rationale=[
            RationaleItem(
                claim="Typed evidence permits one global transform at safe pixel density.",
                evidence_refs=decision.evidence_refs,
                confidence=1.0,
            )
        ],
        patch=patch,
        alternatives=technique.creative_variants,
        planner=Provenance(
            tool="aoa-editing-tree-of-editing-v2",
            tool_version="0.1.0",
            parameters={
                "technique_id": technique.id,
                "technique_revision": technique.revision,
                "curve_model": (
                    technique.curve_model_v2.kind
                    if technique.curve_model_v2 is not None
                    else None
                ),
                "status": technique.status,
                "applicability_decision_id": decision.id,
            },
            deterministic=True,
        ),
    )


def _variant(
    primary: Treatment, title: str, summary: str, tracks: list[dict[str, Any]]
) -> Treatment:
    duration = max(
        (
            int(clip["timeline_range"]["start"])
            + int(clip["timeline_range"]["duration"])
            for track in tracks
            for clip in track.get("clips", [])
        ),
        default=1,
    )
    operations = [
        operation.model_copy(
            update={
                "value": (
                    tracks
                    if operation.path == "/tracks"
                    else duration if operation.path == "/duration_frames" else operation.value
                )
            }
        )
        for operation in primary.patch.operations
    ]
    patch = primary.patch.model_copy(
        update={
            "id": new_id("patch"),
            "operations": operations,
            "inverse_operations": [],
            "rationale": summary,
        }
    )
    return primary.model_copy(
        update={
            "id": new_id("treatment"),
            "title": title,
            "summary": summary,
            "patch": patch,
            "decision_graph_id": None,
            "created_at": primary.created_at,
        }
    )


def _set_restrained_push(tracks: list[dict[str, Any]]) -> None:
    for track in tracks:
        for clip in track.get("clips", []):
            effects = clip.get("effects", [])
            for effect in effects:
                if effect.get("type") != "transform":
                    continue
                positions = effect.get("position", [])
                for item in positions:
                    item["x"] = 0.0
                    item["y"] = 0.0
                scales = effect.get("scale", [])
                if scales:
                    scales[0]["value"] = 1.02
                    scales[-1]["value"] = 1.10


def _reverse_motion_curves(tracks: list[dict[str, Any]]) -> None:
    for track in tracks:
        for clip in track.get("clips", []):
            for effect in clip.get("effects", []):
                if effect.get("type") != "transform":
                    continue
                for name, keys in (("position", ("x", "y")), ("scale", ("value",))):
                    curve = effect.get(name, [])
                    values = [tuple(item[key] for key in keys) for item in curve]
                    for item, replacement in zip(curve, reversed(values), strict=True):
                        for key, value in zip(keys, replacement, strict=True):
                            item[key] = value


def _reorder_montage(
    tracks: list[dict[str, Any]], *, reverse_order: bool, sparse: bool
) -> None:
    if not tracks:
        return
    for track in tracks:
        clips = list(track.get("clips", []))
        if reverse_order:
            clips.reverse()
        if sparse:
            clips = clips[::2]
        cursor = 0
        for clip in clips:
            clip["timeline_range"]["start"] = cursor
            cursor += int(clip["timeline_range"]["duration"])
        track["clips"] = clips
