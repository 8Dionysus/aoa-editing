"""Technical QC for a rendered artifact."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from aoa_editing.domain.models import CheckResult, QCReport
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore

BLACK_INTERVAL = re.compile(r"black_start:([0-9.]+) black_end:([0-9.]+)")


class QualityService:
    def __init__(self, store: ProjectStore):
        self.store = store

    def inspect(
        self,
        project_id: str,
        version_id: str | None = None,
        *,
        profile: str = "preview",
    ) -> QCReport:
        version = self.store.load_version(project_id, version_id)
        render = (
            self.store.project_path(project_id) / "renders" / version.id / profile / "video.mp4"
        )
        probe = _probe(render)
        stream = next(
            item for item in probe.get("streams", []) if item.get("codec_type") == "video"
        )
        format_data = probe.get("format", {})
        actual_duration = float(format_data.get("duration", 0))
        expected_duration = version.timeline.duration_frames / version.timeline.frame_rate.fps
        tolerance = max(0.1, 2 / version.timeline.frame_rate.fps)
        duration_ok = abs(actual_duration - expected_duration) <= tolerance
        checks = [
            CheckResult(
                id="container-readable",
                status="pass",
                summary="ffprobe parsed the render",
                measured={"format": format_data.get("format_name")},
            ),
            CheckResult(
                id="duration",
                status="pass" if duration_ok else "fail",
                summary=(
                    "render duration matches the edit graph" if duration_ok else "duration mismatch"
                ),
                measured={
                    "expected_seconds": expected_duration,
                    "actual_seconds": actual_duration,
                    "tolerance_seconds": tolerance,
                },
            ),
            CheckResult(
                id="video-stream",
                status="pass" if stream.get("width") and stream.get("height") else "fail",
                summary="video stream has valid dimensions",
                measured={
                    "codec": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "pixel_format": stream.get("pix_fmt"),
                },
            ),
        ]
        black = _blackdetect(render)
        checks.append(
            CheckResult(
                id="extended-black",
                status="warn" if black else "pass",
                summary=(
                    "extended black intervals found" if black else "no extended black intervals"
                ),
                measured={"intervals": black},
            )
        )
        failed = any(check.status == "fail" for check in checks)
        warned = any(check.status == "warn" for check in checks)
        report = QCReport(
            project_id=project_id,
            version_id=version.id,
            render_path=str(render.relative_to(self.store.project_path(project_id))),
            render_sha256=sha256_file(render),
            checks=checks,
            overall="fail" if failed else "warn" if warned else "pass",
        )
        self.store.save_qc(report)
        return report


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("ffprobe returned a non-object payload")
    return payload


def _blackdetect(path: Path) -> list[dict[str, float]]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=0.5:pix_th=0.05",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return [
        {"start": float(start), "end": float(end)}
        for start, end in BLACK_INTERVAL.findall(result.stderr)
    ]
