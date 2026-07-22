"""OpenTimelineIO projection; rich AoA effects stay in metadata."""

from __future__ import annotations

import opentimelineio as otio  # type: ignore[import-untyped]

from aoa_editing.domain.models import CompatibilityItem, InterchangeReport, Provenance
from aoa_editing.infrastructure.store import ProjectStore


class OTIOExporter:
    def __init__(self, store: ProjectStore):
        self.store = store

    def export(self, project_id: str, version_id: str | None = None) -> InterchangeReport:
        version = self.store.load_version(project_id, version_id)
        fps = version.timeline.frame_rate.fps
        timeline = otio.schema.Timeline(name=version.message)
        timeline.metadata["aoa_editing"] = {
            "schema_version": version.schema_version,
            "project_id": project_id,
            "version_id": version.id,
        }
        for source_track in version.timeline.tracks:
            track = otio.schema.Track(name=source_track.name)
            cursor = 0
            for source_clip in sorted(
                source_track.clips, key=lambda item: item.timeline_range.start
            ):
                if source_clip.timeline_range.start > cursor:
                    track.append(
                        otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(0, fps),
                                duration=otio.opentime.RationalTime(
                                    source_clip.timeline_range.start - cursor, fps
                                ),
                            )
                        )
                    )
                path = self.store.asset_source_path(project_id, source_clip.asset_id)
                source_start = source_clip.source_range.start if source_clip.source_range else 0
                source_duration = (
                    source_clip.source_range.duration
                    if source_clip.source_range
                    else source_clip.timeline_range.duration
                )
                clip = otio.schema.Clip(
                    name=source_clip.role,
                    media_reference=otio.schema.ExternalReference(target_url=path.as_uri()),
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(source_start, fps),
                        duration=otio.opentime.RationalTime(source_duration, fps),
                    ),
                )
                clip.metadata["aoa_editing"] = {
                    "clip_id": source_clip.id,
                    "effects": [effect.model_dump(mode="json") for effect in source_clip.effects],
                    "evidence_refs": source_clip.evidence_refs,
                    "decision_refs": source_clip.decision_refs,
                }
                track.append(clip)
                cursor = source_clip.timeline_range.end
            timeline.tracks.append(track)
        output = (
            self.store.project_path(project_id)
            / "exports"
            / version.id
            / f"{version.id}.otio"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        otio.adapters.write_to_file(timeline, str(output), adapter_name="otio_json")
        reread = otio.adapters.read_from_file(str(output), adapter_name="otio_json")
        if len(reread.tracks) != len(timeline.tracks):
            raise RuntimeError("OTIO round-trip lost tracks")
        report = InterchangeReport(
            project_id=project_id,
            version_id=version.id,
            format="opentimelineio",
            output_path=str(output.relative_to(self.store.project_path(project_id))),
            items=[
                CompatibilityItem(
                    feature="tracks, gaps, clips, source ranges",
                    status="native",
                    detail="Represented as OTIO timeline objects.",
                ),
                CompatibilityItem(
                    feature="AoA effects, evidence, and decisions",
                    status="approximated",
                    detail="Preserved as AoA metadata; downstream adapters may ignore it.",
                ),
            ],
            validator=Provenance(
                tool="OpenTimelineIO",
                tool_version=getattr(otio, "__version__", "0.18.1"),
                command=[],
                parameters={"validation": "otio-json-round-trip"},
                deterministic=True,
            ),
        )
        self.store.save_interchange(report)
        return report
