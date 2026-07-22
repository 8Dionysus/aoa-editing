"""Stable invariants that protect project meaning across adapters."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

from aoa_editing.domain.models import ProjectManifest, ProjectVersion, Timeline


class InvariantViolation(ValueError):
    """Raised when canonical project state would become invalid."""


def validate_timeline(timeline: Timeline) -> None:
    ids: list[str] = []
    for track in timeline.tracks:
        ids.append(track.id)
        ids.extend(clip.id for clip in track.clips)
        ordered = sorted(track.clips, key=lambda clip: clip.timeline_range.start)
        if track.kind.value in {"audio", "caption"}:
            continue
        for left, right in pairwise(ordered):
            if left.timeline_range.end > right.timeline_range.start:
                raise InvariantViolation(
                    f"clips {left.id} and {right.id} overlap on non-layered track {track.id}"
                )
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise InvariantViolation(f"duplicate timeline identifiers: {duplicates}")


def validate_project_version(project: ProjectManifest, version: ProjectVersion) -> None:
    if version.project_id != project.id:
        raise InvariantViolation("version belongs to a different project")
    if version.parent_version_id and version.parent_version_id not in project.versions:
        raise InvariantViolation("version parent is not registered by the project")
    validate_timeline(version.timeline)


def validate_manifest(project: ProjectManifest) -> None:
    if len(project.assets) != len(set(project.assets)):
        raise InvariantViolation("project asset ids must be unique")
    if len(project.versions) != len(set(project.versions)):
        raise InvariantViolation("project version ids must be unique")
    if project.current_version_id and project.current_version_id not in project.versions:
        raise InvariantViolation("current version must be registered")
    for digest in project.sealed_reference_hashes:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise InvariantViolation("sealed reference hash must be lowercase SHA-256")
