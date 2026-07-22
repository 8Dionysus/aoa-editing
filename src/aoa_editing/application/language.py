"""Fail-closed natural-language to typed-patch preview adapter."""

from __future__ import annotations

import re

from aoa_editing.domain.models import (
    MAX_SPEED_RATE,
    EditPatch,
    PatchOperation,
    PatchPreview,
    TrackKind,
)
from aoa_editing.domain.patches import apply_timeline_patch
from aoa_editing.infrastructure.store import ProjectStore

HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
DURATION = re.compile(
    r"(?:длительност[ьи]?|duration)(?:\s+(?:до|to|=))?\s+(\d+(?:[.,]\d+)?)"
)
SPEED = re.compile(r"(?:скорост[ьи]?|speed)(?:\s+(?:до|to|=))?\s+(\d+(?:[.,]\d+)?)\s*x?")


class LanguagePatchError(ValueError):
    """The command cannot be mapped to an allowlisted, valid timeline patch."""


class NaturalLanguagePatchService:
    """Small deterministic baseline; an LLM port may only produce this same contract."""

    def __init__(self, store: ProjectStore):
        self.store = store

    def preview(
        self, project_id: str, command: str, *, version_id: str | None = None
    ) -> PatchPreview:
        command = command.strip()
        if not command:
            raise LanguagePatchError("natural-language command is empty")
        version = self.store.load_version(project_id, version_id)
        lowered = command.casefold()
        operations: list[PatchOperation] = []
        summaries: list[str] = []
        warnings: list[str] = []

        if "фон" in lowered or "background" in lowered:
            match = HEX_COLOR.search(command)
            if match:
                color = match.group(0).lower()
            elif any(word in lowered for word in ("бел", "white")):
                color = "#ffffff"
            elif any(word in lowered for word in ("черн", "black")):
                color = "#000000"
            else:
                raise LanguagePatchError("background command requires #RRGGBB, white, or black")
            operations.append(PatchOperation(op="replace", path="/background", value=color))
            summaries.append(f"background → {color}")

        duration_match = DURATION.search(lowered)
        if duration_match:
            seconds = float(duration_match.group(1).replace(",", "."))
            if seconds <= 0:
                raise LanguagePatchError("duration must be positive")
            frames = max(1, round(seconds * version.timeline.frame_rate.fps))
            operations.append(
                PatchOperation(op="replace", path="/duration_frames", value=frames)
            )
            summaries.append(f"duration → {frames} frames")

        speed_match = SPEED.search(lowered)
        if speed_match:
            rate = float(speed_match.group(1).replace(",", "."))
            if not 0.5 <= rate <= MAX_SPEED_RATE:
                raise LanguagePatchError(
                    f"constant speed supports 0.5x through {MAX_SPEED_RATE:.0f}x"
                )
            location = next(
                (
                    (track_index, clip_index)
                    for track_index, track in enumerate(version.timeline.tracks)
                    if track.kind is TrackKind.VIDEO
                    for clip_index, clip in enumerate(track.clips)
                    if clip.enabled
                ),
                None,
            )
            if location is None:
                raise LanguagePatchError("speed command needs an enabled video clip")
            track_index, clip_index = location
            operations.append(
                PatchOperation(
                    op="add",
                    path=f"/tracks/{track_index}/clips/{clip_index}/effects/-",
                    value={"type": "speed", "rate": rate},
                )
            )
            summaries.append(f"first enabled video clip speed → {rate}x")
            warnings.append("Speed preview targets the first enabled video clip; inspect the diff.")

        requested_enable = any(word in lowered for word in ("включи", "enable"))
        requested_disable = any(word in lowered for word in ("отключи", "disable", "exclude"))
        if requested_enable or requested_disable:
            clip_location = next(
                (
                    (track_index, clip_index, clip.id)
                    for track_index, track in enumerate(version.timeline.tracks)
                    for clip_index, clip in enumerate(track.clips)
                    if clip.id.casefold() in lowered
                ),
                None,
            )
            if clip_location is None:
                raise LanguagePatchError("enable/disable command must include an existing clip id")
            track_index, clip_index, clip_id = clip_location
            enabled = requested_enable and not requested_disable
            operations.append(
                PatchOperation(
                    op="replace",
                    path=f"/tracks/{track_index}/clips/{clip_index}/enabled",
                    value=enabled,
                )
            )
            summaries.append(f"{clip_id} enabled → {enabled}")

        requested_pin = any(word in lowered for word in ("закрепи", "pin ", "pin:"))
        requested_unpin = any(word in lowered for word in ("открепи", "unpin"))
        if requested_pin or requested_unpin:
            pin_location = next(
                (
                    (track_index, clip_index, clip.id)
                    for track_index, track in enumerate(version.timeline.tracks)
                    for clip_index, clip in enumerate(track.clips)
                    if clip.id.casefold() in lowered
                ),
                None,
            )
            if pin_location is None:
                raise LanguagePatchError("pin/unpin command must include an existing clip id")
            track_index, clip_index, clip_id = pin_location
            pinned = requested_pin and not requested_unpin
            operations.append(
                PatchOperation(
                    op="replace",
                    path=f"/tracks/{track_index}/clips/{clip_index}/pinned",
                    value=pinned,
                )
            )
            summaries.append(f"{clip_id} pinned → {pinned}")

        if not operations:
            raise LanguagePatchError(
                "unsupported command; baseline supports background, duration, speed, "
                "clip enable/disable, and pin/unpin"
            )
        patch = EditPatch(
            project_id=project_id,
            base_version_id=version.id,
            operations=operations,
            rationale=f"Natural-language preview: {command}",
        )
        try:
            apply_timeline_patch(version.timeline, patch.operations)
        except ValueError as error:
            raise LanguagePatchError(
                f"command would violate timeline invariants: {error}"
            ) from error
        return PatchPreview(
            project_id=project_id,
            base_version_id=version.id,
            natural_language_command=command,
            patch=patch,
            summary="; ".join(summaries),
            warnings=warnings,
        )
