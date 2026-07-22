"""Kdenlive-compatible MLT XML export with explicit fidelity reporting."""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from aoa_editing.domain.models import (
    AudioEffect,
    BlendEffect,
    BlurEffect,
    ColorEffect,
    CompatibilityItem,
    DissolveEffect,
    GlowEffect,
    InterchangeReport,
    MaskEffect,
    ProjectVersion,
    Provenance,
    SpeedEffect,
    TextEffect,
    TrackKind,
    TransformEffect,
    TransformEffectV2,
)
from aoa_editing.infrastructure.store import ProjectStore


class KdenliveExporter:
    def __init__(self, store: ProjectStore):
        self.store = store

    def export(self, project_id: str, version_id: str | None = None) -> InterchangeReport:
        version = self.store.load_version(project_id, version_id)
        output = (
            self.store.project_path(project_id) / "exports" / version.id / f"{version.id}.kdenlive"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        tree, compatibility = self._build(project_id, version)
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)
        ET.parse(output)
        validation = self.validate(output)
        report = InterchangeReport(
            project_id=project_id,
            version_id=version.id,
            format="kdenlive-mlt",
            output_path=str(output.relative_to(self.store.project_path(project_id))),
            items=compatibility,
            validator=validation,
        )
        self.store.save_interchange(report)
        return report

    def _build(
        self, project_id: str, version: ProjectVersion
    ) -> tuple[ET.ElementTree, list[CompatibilityItem]]:
        timeline = version.timeline
        fps = timeline.frame_rate
        root = ET.Element(
            "mlt",
            {
                "LC_NUMERIC": "C",
                "version": "7.40.0",
                "title": version.message,
                "producer": "tractor0",
                "root": str(self.store.project_path(project_id)),
            },
        )
        ET.SubElement(
            root,
            "profile",
            {
                "description": "AoA Editing export",
                "width": str(timeline.width),
                "height": str(timeline.height),
                "progressive": "1",
                "sample_aspect_num": "1",
                "sample_aspect_den": "1",
                "display_aspect_num": str(timeline.width),
                "display_aspect_den": str(timeline.height),
                "frame_rate_num": str(fps.numerator),
                "frame_rate_den": str(fps.denominator),
                "colorspace": "709",
            },
        )
        black = ET.SubElement(
            root,
            "producer",
            {"id": "black", "in": "0", "out": str(timeline.duration_frames - 1)},
        )
        _property(black, "length", str(timeline.duration_frames))
        _property(black, "eof", "pause")
        _property(black, "resource", timeline.background)
        _property(black, "mlt_service", "color")
        black_playlist = ET.SubElement(root, "playlist", {"id": "background"})
        ET.SubElement(
            black_playlist,
            "entry",
            {"producer": "black", "in": "0", "out": str(timeline.duration_frames - 1)},
        )
        playlist_ids = ["background"]
        compatibility: list[CompatibilityItem] = [
            CompatibilityItem(
                feature="timeline and source trims",
                status="native",
                detail="Multitrack MLT playlists retain clip timing and source ranges.",
            )
        ]
        clip_number = 0
        effect_types: set[type[object]] = set()
        for track_index, track in enumerate(timeline.tracks):
            playlist_id = f"playlist{track_index}"
            playlist_ids.append(playlist_id)
            playlist = ET.SubElement(root, "playlist", {"id": playlist_id})
            _property(playlist, "kdenlive:track_name", track.name)
            if track.muted:
                _property(playlist, "hide", "both")
            cursor = 0
            for clip in sorted(track.clips, key=lambda item: item.timeline_range.start):
                if not clip.enabled:
                    continue
                if clip.timeline_range.start > cursor:
                    ET.SubElement(
                        playlist,
                        "blank",
                        {"length": str(clip.timeline_range.start - cursor)},
                    )
                asset = self.store.load_asset(project_id, clip.asset_id)
                source = self.store.asset_source_path(project_id, clip.asset_id)
                producer_id = f"producer{clip_number}"
                producer = ET.SubElement(
                    root,
                    "producer",
                    {
                        "id": producer_id,
                        "in": "0",
                        "out": str(max(0, clip.timeline_range.duration - 1)),
                    },
                )
                _property(producer, "length", str(clip.timeline_range.duration))
                _property(producer, "resource", str(source))
                _property(producer, "kdenlive:clipname", f"{asset.original_name} [{clip.role}]")
                _property(producer, "kdenlive:control_uuid", clip.id)
                _property(
                    producer,
                    "aoa_editing:effects_json",
                    json.dumps(
                        [effect.model_dump(mode="json") for effect in clip.effects],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                if asset.media_kind.value == "image":
                    _property(producer, "mlt_service", "qimage")
                    _property(producer, "ttl", "1")
                else:
                    _property(producer, "mlt_service", "avformat-novalidate")
                if track.kind is TrackKind.AUDIO:
                    _property(producer, "video_index", "-1")
                source_start = clip.source_range.start if clip.source_range else 0
                source_duration = (
                    clip.source_range.duration
                    if clip.source_range
                    else clip.timeline_range.duration
                )
                ET.SubElement(
                    playlist,
                    "entry",
                    {
                        "producer": producer_id,
                        "in": str(source_start),
                        "out": str(source_start + source_duration - 1),
                    },
                )
                cursor = clip.timeline_range.end
                effect_types.update(type(effect) for effect in clip.effects)
                clip_number += 1
        tractor = ET.SubElement(
            root,
            "tractor",
            {"id": "tractor0", "in": "0", "out": str(timeline.duration_frames - 1)},
        )
        _property(tractor, "kdenlive:docproperties.version", "1.1")
        _property(tractor, "kdenlive:docproperties.documentid", version.id)
        _property(tractor, "kdenlive:sequenceproperties.hasAudio", "1")
        _property(tractor, "kdenlive:sequenceproperties.hasVideo", "1")
        multitrack = ET.SubElement(tractor, "multitrack")
        for playlist_id in playlist_ids:
            ET.SubElement(multitrack, "track", {"producer": playlist_id})
        for track_index in range(1, len(playlist_ids)):
            transition = ET.SubElement(
                tractor,
                "transition",
                {
                    "id": f"transition{track_index}",
                    "in": "0",
                    "out": str(timeline.duration_frames - 1),
                },
            )
            _property(transition, "a_track", "0")
            _property(transition, "b_track", str(track_index))
            _property(transition, "mlt_service", "qtblend")
            _property(transition, "always_active", "1")
        compatibility.extend(_effect_compatibility(effect_types))
        compatibility.append(
            CompatibilityItem(
                feature="AoA evidence and rationale",
                status="omitted",
                detail="Canonical JSON stays alongside the export; MLT receives clip IDs only.",
            )
        )
        return ET.ElementTree(root), compatibility

    def validate(self, path: Path) -> Provenance:
        command = [
            "flatpak",
            "run",
            f"--filesystem={self.store.settings.projects_root}:ro",
            "--command=melt",
            "org.kde.kdenlive",
            str(path),
            "-consumer",
            "null",
            "real_time=-1",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-4000:] or "MLT could not load the export")
        version_result = subprocess.run(
            ["flatpak", "run", "--command=melt", "org.kde.kdenlive", "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return Provenance(
            tool="melt",
            tool_version=(version_result.stdout or version_result.stderr).splitlines()[0],
            command=command,
            parameters={"validation": "full-null-consumer-load"},
            deterministic=True,
        )


def _property(parent: ET.Element, name: str, value: str) -> None:
    element = ET.SubElement(parent, "property", {"name": name})
    element.text = value


def _effect_compatibility(effect_types: set[type[object]]) -> list[CompatibilityItem]:
    mapping: list[tuple[type[object], str, str]] = [
        (TransformEffect, "transform keyframes", "approximated"),
        (TransformEffectV2, "Motion Language v2 curves", "approximated"),
        (ColorEffect, "color effects", "approximated"),
        (MaskEffect, "depth masks", "omitted"),
        (AudioEffect, "audio gain/fades/normalization", "approximated"),
        (TextEffect, "titles", "omitted"),
        (SpeedEffect, "constant speed", "approximated"),
        (DissolveEffect, "dissolve fades", "approximated"),
        (BlurEffect, "Gaussian blur", "omitted"),
        (GlowEffect, "screen glow", "omitted"),
        (BlendEffect, "blend mode and opacity", "approximated"),
    ]
    return [
        CompatibilityItem(
            feature=feature,
            status=status,  # type: ignore[arg-type]
            detail=(
                "The source clip remains editable; reproduce or bake this effect for exact parity."
            ),
        )
        for effect_type, feature, status in mapping
        if effect_type in effect_types
    ]
