"""Evidence-backed narration timing for script-bound screen workflows."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import pairwise
from math import floor
from typing import Any, Literal

from aoa_editing.domain.models import (
    Asset,
    EvidenceRecord,
    FrameRange,
    FrameRate,
    Provenance,
    ScreenWorkflowPlan,
    ScreenWorkflowVoiceoverCue,
    ScreenWorkflowVoiceoverTiming,
    new_id,
)


def build_voiceover_timing_draft(
    *,
    plan: ScreenWorkflowPlan,
    asset: Asset,
    evidence: Sequence[EvidenceRecord],
    timeline_frame_rate: FrameRate,
) -> ScreenWorkflowVoiceoverTiming:
    """Build a review-required timing draft from transcript or measured pauses."""

    if not plan.capture_ready:
        raise ValueError("voiceover timing requires a reviewed capture plan")
    if not asset.metadata.has_audio:
        raise ValueError("voiceover asset has no audio stream")
    duration_seconds = asset.metadata.duration_seconds
    if duration_seconds is None or duration_seconds <= 0:
        raise ValueError("voiceover duration is unavailable")
    duration_frames = max(1, floor(duration_seconds * timeline_frame_rate.fps + 1e-6))
    if duration_frames < len(plan.beats):
        raise ValueError("voiceover is too short to assign at least one frame per beat")

    relevant = [
        item
        for item in evidence
        if item.asset_id == asset.id
        and item.source_sha256 == asset.sha256
        and (
            item.kind
            in {
                "media.probe",
                "audio.silence",
                "audio.loudness",
                "speech.transcript",
            }
            or (
                item.kind == "analysis.failure"
                and item.payload.get("failed_kind") == "speech.transcript"
            )
        )
    ]
    transcript = next((item for item in relevant if item.kind == "speech.transcript"), None)
    transcript_segments = _transcript_segments(transcript, duration_seconds)
    groups = _partition_transcript(plan, transcript_segments)
    if groups is not None:
        cues = _transcript_cues(
            plan,
            groups,
            duration_frames=duration_frames,
            frame_rate=timeline_frame_rate,
        )
        method: Literal["transcript-segments", "silence-weighted"] = "transcript-segments"
    else:
        silence = next((item for item in relevant if item.kind == "audio.silence"), None)
        cues = _silence_weighted_cues(
            plan,
            silence,
            duration_frames=duration_frames,
            frame_rate=timeline_frame_rate,
        )
        method = "silence-weighted"
    evidence_refs = [item.id for item in relevant]
    if not evidence_refs:
        raise ValueError("voiceover timing requires measured audio evidence")
    return ScreenWorkflowVoiceoverTiming(
        plan_id=plan.id,
        script_sha256=plan.script_sha256,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        source_duration_seconds=duration_seconds,
        timeline_frame_rate=timeline_frame_rate,
        duration_frames=duration_frames,
        status="alignment-draft",
        edit_ready=False,
        draft_alignment_method=method,
        cues=cues,
        analysis_evidence_refs=evidence_refs,
        provenance=Provenance(
            tool="aoa-editing-voiceover-timing",
            tool_version="0.1.0",
            parameters={
                "alignment_method": method,
                "transcript_segment_count": len(transcript_segments),
                "review_required": True,
            },
            deterministic=True,
        ),
    )


def review_voiceover_timing(
    *,
    plan: ScreenWorkflowPlan,
    draft: ScreenWorkflowVoiceoverTiming,
    cues: Sequence[ScreenWorkflowVoiceoverCue],
    reviewed_by: str,
    review_note: str,
) -> ScreenWorkflowVoiceoverTiming:
    """Create an immutable reviewed revision from explicitly authored cue ranges."""

    reviewer = reviewed_by.strip()
    note = review_note.strip()
    if not reviewer or not note:
        raise ValueError("voiceover review requires reviewer and review note")
    if draft.status != "alignment-draft" or draft.edit_ready:
        raise ValueError("voiceover review input must be an alignment draft")
    if draft.plan_id != plan.id or draft.script_sha256 != plan.script_sha256:
        raise ValueError("voiceover timing draft does not match the capture plan")
    authored = list(cues)
    expected = [item.id for item in plan.beats]
    if [item.beat_id for item in authored] != expected:
        raise ValueError("reviewed voiceover cues must follow every capture-plan beat")
    reviewed_cues = [
        item.model_copy(
            update={
                "order": index,
                "confidence": 1.0,
                "alignment_basis": "human-reviewed",
            }
        )
        for index, item in enumerate(authored, start=1)
    ]
    return ScreenWorkflowVoiceoverTiming(
        **draft.model_dump(
            exclude={
                "id",
                "generated_at",
                "status",
                "edit_ready",
                "cues",
                "reviewed_by",
                "review_note",
                "supersedes_timing_id",
                "provenance",
            }
        ),
        id=new_id("workflowvoiceover"),
        status="reviewed-for-edit",
        edit_ready=True,
        cues=reviewed_cues,
        reviewed_by=reviewer,
        review_note=note,
        supersedes_timing_id=draft.id,
        provenance=Provenance(
            tool="human-editor",
            tool_version="explicit-voiceover-review-v1",
            parameters={
                "draft_timing_id": draft.id,
                "draft_alignment_method": draft.draft_alignment_method,
                "reviewed_cue_count": len(reviewed_cues),
            },
            deterministic=False,
        ),
    )


def _transcript_segments(
    transcript: EvidenceRecord | None,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if transcript is None:
        return []
    selected: list[dict[str, Any]] = []
    for raw in transcript.payload.get("segments", []):
        if not isinstance(raw, dict):
            continue
        try:
            start = max(0.0, float(raw["start"]))
            end = min(duration_seconds, float(raw["end"]))
            text = str(raw["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if end > start and text:
            selected.append({"start": start, "end": end, "text": text})
    selected.sort(key=lambda item: (item["start"], item["end"]))
    return selected


def _partition_transcript(
    plan: ScreenWorkflowPlan,
    segments: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]] | None:
    beat_count = len(plan.beats)
    segment_count = len(segments)
    if segment_count < beat_count:
        return None
    weights = [max(1, len(_normalize_text(item.narration).split())) for item in plan.beats]
    total_weight = sum(weights)
    states: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {
        (0, 0): (0.0, [])
    }
    for beat_index, beat in enumerate(plan.beats):
        remaining_beats = beat_count - beat_index - 1
        next_states: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}
        for (completed, start), (prior_cost, ranges) in states.items():
            if completed != beat_index:
                continue
            maximum_end = segment_count - remaining_beats
            for end in range(start + 1, maximum_end + 1):
                joined = " ".join(str(item["text"]) for item in segments[start:end])
                similarity = SequenceMatcher(
                    None,
                    _normalize_text(beat.narration),
                    _normalize_text(joined),
                ).ratio()
                observed_share = (end - start) / segment_count
                expected_share = weights[beat_index] / total_weight
                cost = prior_cost + (1.0 - similarity) + 0.15 * abs(
                    observed_share - expected_share
                )
                key = (beat_index + 1, end)
                current = next_states.get(key)
                if current is None or cost < current[0]:
                    next_states[key] = (cost, [*ranges, (start, end)])
        states = next_states
    selected = states.get((beat_count, segment_count))
    if selected is None:
        return None
    return [list(segments[start:end]) for start, end in selected[1]]


def _transcript_cues(
    plan: ScreenWorkflowPlan,
    groups: Sequence[Sequence[dict[str, Any]]],
    *,
    duration_frames: int,
    frame_rate: FrameRate,
) -> list[ScreenWorkflowVoiceoverCue]:
    boundaries = [0]
    for left, right in pairwise(groups):
        midpoint_seconds = (float(left[-1]["end"]) + float(right[0]["start"])) / 2
        boundaries.append(round(midpoint_seconds * frame_rate.fps))
    boundaries.append(duration_frames)
    boundaries = _repair_boundaries(boundaries, duration_frames)
    cues: list[ScreenWorkflowVoiceoverCue] = []
    for index, (beat, group) in enumerate(zip(plan.beats, groups, strict=True)):
        timeline = FrameRange(
            start=boundaries[index],
            duration=boundaries[index + 1] - boundaries[index],
        )
        spoken_start = max(timeline.start, round(float(group[0]["start"]) * frame_rate.fps))
        spoken_end = min(timeline.end, round(float(group[-1]["end"]) * frame_rate.fps))
        spoken = (
            FrameRange(start=spoken_start, duration=spoken_end - spoken_start)
            if spoken_end > spoken_start
            else None
        )
        joined = " ".join(str(item["text"]) for item in group).strip()
        confidence = SequenceMatcher(
            None,
            _normalize_text(beat.narration),
            _normalize_text(joined),
        ).ratio()
        cues.append(
            ScreenWorkflowVoiceoverCue(
                order=index + 1,
                beat_id=beat.id,
                timeline_range=timeline,
                spoken_range=spoken,
                transcript_text=joined,
                confidence=confidence,
                alignment_basis="transcript-segments",
                reviewer_notes=["Draft alignment: verify words, pauses, and boundary frames."],
            )
        )
    return cues


def _silence_weighted_cues(
    plan: ScreenWorkflowPlan,
    silence: EvidenceRecord | None,
    *,
    duration_frames: int,
    frame_rate: FrameRate,
) -> list[ScreenWorkflowVoiceoverCue]:
    weights = [max(0.5, item.estimated_duration_seconds) for item in plan.beats]
    total = sum(weights)
    cumulative = 0.0
    targets: list[int] = []
    for weight in weights[:-1]:
        cumulative += weight
        targets.append(round(duration_frames * cumulative / total))
    silence_candidates: list[int] = []
    if silence is not None:
        for interval in silence.payload.get("intervals", []):
            if not isinstance(interval, dict):
                continue
            try:
                start = float(interval["start_seconds"])
                end = float(interval["end_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            midpoint = round(((start + end) / 2) * frame_rate.fps)
            if 0 < midpoint < duration_frames:
                silence_candidates.append(midpoint)
    boundaries = [0]
    for index, target in enumerate(targets):
        minimum = boundaries[-1] + 1
        maximum = duration_frames - (len(targets) - index)
        feasible = [item for item in silence_candidates if minimum <= item <= maximum]
        selected = (
            min(feasible, key=lambda item: (abs(item - target), item))
            if feasible
            else target
        )
        boundaries.append(min(max(selected, minimum), maximum))
    boundaries.append(duration_frames)
    boundaries = _repair_boundaries(boundaries, duration_frames)
    return [
        ScreenWorkflowVoiceoverCue(
            order=index + 1,
            beat_id=beat.id,
            timeline_range=FrameRange(
                start=boundaries[index],
                duration=boundaries[index + 1] - boundaries[index],
            ),
            confidence=0.45 if silence_candidates else 0.25,
            alignment_basis="silence-weighted",
            reviewer_notes=[
                "Transcript alignment unavailable; verify this pause-weighted boundary manually."
            ],
        )
        for index, beat in enumerate(plan.beats)
    ]


def _repair_boundaries(boundaries: Sequence[int], duration_frames: int) -> list[int]:
    repaired = [0]
    internal_count = len(boundaries) - 2
    for index, boundary in enumerate(boundaries[1:-1]):
        minimum = repaired[-1] + 1
        maximum = duration_frames - (internal_count - index)
        repaired.append(min(max(boundary, minimum), maximum))
    repaired.append(duration_frames)
    return repaired


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))
