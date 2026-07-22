"""Gated batch study for human-free terminal and workflow reference ranges."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    MediaKind,
    Provenance,
    ReferenceWorkflowRangePlan,
    ReferenceWorkflowRangeResult,
    ReferenceWorkflowSourceResult,
    ReferenceWorkflowStudyPlan,
    ReferenceWorkflowStudyReceipt,
)
from aoa_editing.evals.gate import require_readiness
from aoa_editing.infrastructure.media import ffprobe, sha256_file


class ReferenceWorkflowStudyError(RuntimeError):
    """The reference workflow corpus is unsafe, stale, or not reviewable."""


ReadinessGuard = Callable[[Settings], dict[str, object]]


def run_reference_workflow_study(
    plan_path: Path,
    output: Path,
    *,
    settings: Settings | None = None,
    readiness_guard: ReadinessGuard = require_readiness,
) -> ReferenceWorkflowStudyReceipt:
    selected = settings or Settings.from_env()
    readiness = readiness_guard(selected)
    revision = str(readiness.get("git_revision", ""))
    if readiness.get("overall") != "pass" or len(revision) != 40:
        raise ReferenceWorkflowStudyError(
            "reference workflow study requires passing revision-bound readiness"
        )
    selected_plan_path = plan_path.expanduser().resolve(strict=True)
    plan_bytes = selected_plan_path.read_bytes()
    plan = ReferenceWorkflowStudyPlan.model_validate_json(plan_bytes)
    output_root = output.expanduser().resolve()
    evals_root = selected.evals_root.expanduser().resolve()
    projects_root = selected.projects_root.expanduser().resolve()
    if not output_root.is_relative_to(evals_root):
        raise ReferenceWorkflowStudyError("reference workflow output must stay under evals_root")
    if output_root.exists():
        raise ReferenceWorkflowStudyError("reference workflow output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.partial-", dir=output_root.parent))
    try:
        contacts = partial / "contact-sheets"
        contacts.mkdir()
        source_results: list[ReferenceWorkflowSourceResult] = []
        for source in plan.sources:
            source_path = Path(source.path).expanduser()
            if not source_path.is_absolute():
                source_path = selected_plan_path.parent / source_path
            source_path = source_path.resolve(strict=True)
            if source_path.is_relative_to(projects_root):
                raise ReferenceWorkflowStudyError(
                    f"reference source {source.alias} is inside the project asset tree"
                )
            digest = sha256_file(source_path)
            if digest != source.expected_sha256:
                raise ReferenceWorkflowStudyError(
                    f"reference source hash mismatch for {source.alias}"
                )
            kind, metadata, _raw_probe, _command = ffprobe(source_path)
            if kind is not MediaKind.VIDEO or metadata.duration_seconds is None:
                raise ReferenceWorkflowStudyError(
                    f"reference source {source.alias} is not a duration-bearing video"
                )
            if any(item.end_seconds > metadata.duration_seconds + 0.02 for item in source.ranges):
                raise ReferenceWorkflowStudyError(
                    f"reviewed range ends after source duration for {source.alias}"
                )
            range_results = [
                (
                    _measure_range(source_path, item)
                    if item.decision == "include"
                    else _excluded_range(item)
                )
                for item in source.ranges
            ]
            contact_path = contacts / f"{source.alias}.jpg"
            _contact_sheet(
                source_path,
                source.alias,
                [item for item in source.ranges if item.decision == "include"],
                contact_path,
            )
            source_results.append(
                ReferenceWorkflowSourceResult(
                    alias=source.alias,
                    sha256=digest,
                    size_bytes=source_path.stat().st_size,
                    metadata=metadata,
                    ranges=range_results,
                    contact_sheet=str(contact_path.relative_to(partial)),
                    contact_sheet_sha256=sha256_file(contact_path),
                )
            )
        receipt = ReferenceWorkflowStudyReceipt(
            plan_id=plan.id,
            plan_sha256=_sha256_bytes(plan_bytes),
            readiness_revision=revision,
            sources=source_results,
            candidate_rules=plan.candidate_rules,
            provenance=Provenance(
                tool="aoa-editing-reference-workflow-study",
                tool_version="0.1.0",
                parameters={
                    "activity_sampler": "opencv-gray-absdiff-v1",
                    "sample_rate_hz": 4.0,
                    "maximum_samples_per_range": 240,
                    "contact_sheet_frames": 12,
                    "source_paths_persisted": False,
                    "reference_media_role": "analysis-and-evaluation-only",
                },
                deterministic=True,
            ),
        )
        (partial / "reference-workflow-study.json").write_text(
            receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        forbidden_suffixes = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        copied = [path for path in partial.rglob("*") if path.suffix.lower() in forbidden_suffixes]
        if copied:
            raise ReferenceWorkflowStudyError("reference media appeared in study output")
        partial.replace(output_root)
        return receipt
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _measure_range(path: Path, item: ReferenceWorkflowRangePlan) -> ReferenceWorkflowRangeResult:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ReferenceWorkflowStudyError(f"cannot decode reference range {item.id}")
    duration = item.end_seconds - item.start_seconds
    sample_count = max(2, min(240, round(duration * 4.0) + 1))
    frames: list[np.ndarray] = []
    try:
        for timestamp in np.linspace(item.start_seconds, item.end_seconds, sample_count):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
    finally:
        capture.release()
    changes = [
        float(np.mean(np.abs(right - left)) / 255.0)
        for left, right in pairwise(frames)
    ]
    return ReferenceWorkflowRangeResult(
        range_id=item.id,
        start_seconds=item.start_seconds,
        end_seconds=item.end_seconds,
        content_class=item.content_class,
        decision=item.decision,
        human_present=item.human_present,
        sample_count=len(frames),
        mean_frame_change=float(np.mean(changes)) if changes else 0.0,
        p90_frame_change=float(np.percentile(changes, 90)) if changes else 0.0,
        static_fraction=(
            sum(change < 0.008 for change in changes) / len(changes) if changes else 1.0
        ),
        rationale=item.rationale,
    )


def _excluded_range(item: ReferenceWorkflowRangePlan) -> ReferenceWorkflowRangeResult:
    """Preserve the manual exclusion decision without decoding that range."""

    return ReferenceWorkflowRangeResult(
        range_id=item.id,
        start_seconds=item.start_seconds,
        end_seconds=item.end_seconds,
        content_class=item.content_class,
        decision=item.decision,
        human_present=item.human_present,
        sample_count=0,
        mean_frame_change=0.0,
        p90_frame_change=0.0,
        static_fraction=1.0,
        rationale=item.rationale,
    )


def _contact_sheet(
    path: Path,
    alias: str,
    ranges: list[ReferenceWorkflowRangePlan],
    output: Path,
) -> None:
    timestamps = _representative_timestamps(ranges, limit=12)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ReferenceWorkflowStudyError(f"cannot decode contact sheet for {alias}")
    panels: list[tuple[float, Image.Image]] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((480, 270), Image.Resampling.LANCZOS)
            panels.append((timestamp, image.copy()))
    finally:
        capture.release()
    if not panels:
        raise ReferenceWorkflowStudyError(f"no reviewed frames decoded for {alias}")
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    cell_width, cell_height = 500, 310
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 44), "#0b0d12")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 14), f"{alias} — reviewed human-free workflow ranges", fill="#f3f5f7")
    for index, (timestamp, panel) in enumerate(panels):
        x = (index % columns) * cell_width + 10
        y = (index // columns) * cell_height + 50
        sheet.paste(panel, (x, y))
        draw.text((x, y + 274), _timestamp(timestamp), fill="#c8d2df")
    sheet.save(output, "JPEG", quality=90, optimize=True)


def _representative_timestamps(
    ranges: list[ReferenceWorkflowRangePlan], *, limit: int
) -> list[float]:
    total = sum(item.end_seconds - item.start_seconds for item in ranges)
    if total <= 0:
        return []
    timestamps: list[float] = []
    for item in ranges:
        allocation = max(1, round(limit * (item.end_seconds - item.start_seconds) / total))
        timestamps.extend(
            float(value)
            for value in np.linspace(
                item.start_seconds + min(0.05, (item.end_seconds - item.start_seconds) / 4),
                item.end_seconds - min(0.05, (item.end_seconds - item.start_seconds) / 4),
                allocation,
            )
        )
    if len(timestamps) <= limit:
        return timestamps
    indexes = np.linspace(0, len(timestamps) - 1, limit).round().astype(int)
    return [timestamps[index] for index in indexes]


def _timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
