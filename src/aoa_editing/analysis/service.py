"""Deterministic baseline media analysis with optional host STT."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Callable, Iterable
from importlib.metadata import version
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from aoa_editing import __version__
from aoa_editing.analysis.ports import AnalyzerPort
from aoa_editing.domain.models import (
    Asset,
    EvidenceRecord,
    FrameRange,
    FrameRate,
    MediaKind,
    Provenance,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.service import ProviderService

SILENCE_START = re.compile(r"silence_start:\s*([-0-9.]+)")
SILENCE_END = re.compile(r"silence_end:\s*([-0-9.]+)\s*\|\s*silence_duration:\s*([-0-9.]+)")
SCENE_TIME = re.compile(r"pts_time:([0-9.]+)")


class AnalysisError(RuntimeError):
    """A mandatory evidence operation failed."""


def _run(command: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AnalysisError(f"cannot run {command[0]}: {error}") from error


def _ffmpeg_version() -> str:
    result = _run(["ffmpeg", "-version"], timeout=10)
    return (result.stdout or result.stderr).splitlines()[0]


class AnalysisService:
    def __init__(self, store: ProjectStore, ports: Iterable[AnalyzerPort] = ()):
        self.store = store
        self.ports = {port.kind: port for port in ports}

    def analyze(
        self, project_id: str, asset_id: str, *, transcribe: bool = False
    ) -> list[EvidenceRecord]:
        asset = self.store.load_asset(project_id, asset_id)
        existing = {
            record.kind: record
            for record in self.store.list_effective_evidence(project_id)
            if record.asset_id == asset_id and record.source_sha256 == asset.sha256
        }
        produced: list[EvidenceRecord] = []
        if asset.media_kind is MediaKind.IMAGE:
            still_kinds = ("image.statistics", "image.depth_layers")
            complete = all(
                kind in existing
                and all(
                    (self.store.project_path(project_id) / artifact).is_file()
                    for artifact in existing[kind].artifacts
                )
                for kind in still_kinds
            )
            candidates = (
                [existing[kind] for kind in still_kinds]
                if complete
                else self._analyze_still(project_id, asset)
            )
            for record in candidates:
                selected = existing.get(record.kind, record)
                if selected is record:
                    self.store.save_evidence(record)
                produced.append(selected)
        else:
            if asset.metadata.has_video:
                scene_record = existing.get("video.scenes")
                if scene_record is None:
                    produced.extend(
                        self._execute("video.scenes", project_id, asset, self._scene_cuts)
                    )
                else:
                    produced.append(scene_record)
            if asset.metadata.has_audio:
                factories = {
                    "audio.silence": self._silence,
                    "audio.loudness": self._loudness,
                }
                for kind, factory in factories.items():
                    audio_record = existing.get(kind)
                    if audio_record is None:
                        produced.extend(self._execute(kind, project_id, asset, factory))
                    else:
                        produced.append(audio_record)
                if transcribe:
                    transcript_record = existing.get("speech.transcript")
                    if transcript_record is None:
                        produced.extend(
                            self._execute(
                                "speech.transcript", project_id, asset, self._transcribe
                            )
                        )
                    else:
                        produced.append(transcript_record)
        return produced

    def _execute(
        self,
        kind: str,
        project_id: str,
        asset: Asset,
        fallback: Callable[[str, Asset], EvidenceRecord],
    ) -> list[EvidenceRecord]:
        """Run one capability and persist either its evidence or a localized failure."""

        analyzer = self.ports.get(kind)
        try:
            record = (
                analyzer.analyze(project_id, asset)
                if analyzer is not None
                else fallback(project_id, asset)
            )
            if record.kind != kind:
                raise AnalysisError(
                    f"analyzer contract mismatch: expected {kind}, received {record.kind}"
                )
        except Exception as error:  # capability failure must not discard sibling evidence
            record = EvidenceRecord(
                project_id=project_id,
                asset_id=asset.id,
                source_sha256=asset.sha256,
                kind="analysis.failure",
                confidence=1.0,
                payload={
                    "failed_kind": kind,
                    "error_type": type(error).__name__,
                    "message": str(error)[-2000:],
                    "localized": True,
                    "retryable": True,
                },
                **_temporal_scope(asset),
                provenance=Provenance(
                    tool="aoa-editing-analysis-orchestrator",
                    tool_version=__version__,
                    parameters={
                        "failed_kind": kind,
                        "adapter": type(analyzer).__name__ if analyzer else "builtin",
                    },
                    deterministic=False,
                ),
            )
        self.store.save_evidence(record)
        return [record]

    def _provenance(self, tool: str, command: list[str], **parameters: Any) -> Provenance:
        if tool == "ffmpeg":
            tool_version = _ffmpeg_version()
        else:
            try:
                tool_version = version(tool)
            except Exception:  # package metadata is evidence-only
                tool_version = "unknown"
        return Provenance(
            tool=tool,
            tool_version=tool_version,
            command=command,
            parameters=parameters,
            deterministic=True,
        )

    def _scene_cuts(self, project_id: str, asset: Asset) -> EvidenceRecord:
        source = self.store.asset_source_path(project_id, asset.id)
        threshold = 0.32
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
        result = _run(command)
        if result.returncode != 0:
            raise AnalysisError(result.stderr[-2000:])
        cuts = sorted({round(float(value), 6) for value in SCENE_TIME.findall(result.stderr)})
        return EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="video.scenes",
            payload={"cut_times_seconds": cuts, "threshold": threshold, "count": len(cuts)},
            **_temporal_scope(asset),
            provenance=self._provenance("ffmpeg", command, detector="scene_score"),
        )

    def _silence(self, project_id: str, asset: Asset) -> EvidenceRecord:
        source = self.store.asset_source_path(project_id, asset.id)
        noise = "-35dB"
        minimum = 0.35
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-af",
            f"silencedetect=noise={noise}:d={minimum}",
            "-f",
            "null",
            "-",
        ]
        result = _run(command)
        if result.returncode != 0:
            raise AnalysisError(result.stderr[-2000:])
        starts = [float(value) for value in SILENCE_START.findall(result.stderr)]
        ends = [
            (float(end), float(duration))
            for end, duration in SILENCE_END.findall(result.stderr)
        ]
        intervals: list[dict[str, float]] = []
        for index, start in enumerate(starts):
            if index < len(ends):
                end, duration = ends[index]
            else:
                end = asset.metadata.duration_seconds or start
                duration = max(0.0, end - start)
            intervals.append(
                {"start_seconds": max(0.0, start), "end_seconds": end, "duration_seconds": duration}
            )
        return EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="audio.silence",
            payload={
                "intervals": intervals,
                "noise_threshold": noise,
                "minimum_duration": minimum,
            },
            **_temporal_scope(asset),
            provenance=self._provenance("ffmpeg", command, filter="silencedetect"),
        )

    def _loudness(self, project_id: str, asset: Asset) -> EvidenceRecord:
        source = self.store.asset_source_path(project_id, asset.id)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-af",
            "loudnorm=I=-16:LRA=11:TP=-1.5:print_format=json",
            "-f",
            "null",
            "-",
        ]
        result = _run(command)
        if result.returncode != 0:
            raise AnalysisError(result.stderr[-2000:])
        match = re.search(r"\{\s*\"input_i\".*?\}", result.stderr, flags=re.DOTALL)
        measured: dict[str, Any]
        if match:
            try:
                measured = json.loads(match.group(0))
            except json.JSONDecodeError:
                measured = {"raw_summary": match.group(0)}
        else:
            measured = {"raw_summary": result.stderr[-2000:]}
        return EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="audio.loudness",
            payload=measured,
            **_temporal_scope(asset),
            provenance=self._provenance("ffmpeg", command, filter="loudnorm-analysis"),
        )

    def _analyze_still(self, project_id: str, asset: Asset) -> list[EvidenceRecord]:
        source = self.store.asset_source_path(project_id, asset.id)
        with Image.open(source) as loaded:
            image = loaded.convert("RGB")
        sample = image.copy()
        sample.thumbnail((512, 512), Image.Resampling.LANCZOS)
        pixels = np.asarray(sample, dtype=np.float32) / 255.0
        luminance = (
            pixels[..., 0] * 0.2126 + pixels[..., 1] * 0.7152 + pixels[..., 2] * 0.0722
        )
        saturation = pixels.max(axis=2) - pixels.min(axis=2)
        grad_y, grad_x = np.gradient(luminance)
        edges = np.hypot(grad_x, grad_y)
        height, width = luminance.shape
        yy, xx = np.mgrid[0:height, 0:width]
        radius = np.sqrt(
            ((xx / max(width - 1, 1)) - 0.5) ** 2
            + ((yy / max(height - 1, 1)) - 0.5) ** 2
        )
        center = np.clip(1 - radius / 0.7072, 0, 1)
        saliency = _normalize(edges) * 0.45 + saturation * 0.25 + center * 0.30
        artifacts_root = self.store.project_path(project_id) / "evidence" / "artifacts" / asset.id
        artifacts_root.mkdir(parents=True, exist_ok=True)
        mask_records: list[dict[str, Any]] = []
        quantiles = (("midground", 0.55), ("foreground", 0.78))
        for role, quantile in quantiles:
            threshold = float(np.quantile(saliency, quantile))
            mask_small = Image.fromarray((saliency >= threshold).astype(np.uint8) * 255, mode="L")
            mask = mask_small.resize(image.size, Image.Resampling.LANCZOS).filter(
                ImageFilter.GaussianBlur(radius=max(1, round(min(image.size) * 0.002)))
            )
            path = artifacts_root / f"{role}-mask.png"
            mask.save(path, format="PNG", optimize=True)
            mask_records.append(
                {
                    "role": role,
                    "path": str(path.relative_to(self.store.project_path(project_id))),
                    "coverage": float(np.mean(saliency >= threshold)),
                    "threshold": threshold,
                }
            )
        entropy = _entropy(np.asarray(sample.convert("L"), dtype=np.uint8))
        stats = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="image.statistics",
            payload={
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.width / image.height,
                "mean_rgb": [float(value) for value in pixels.mean(axis=(0, 1))],
                "mean_luminance": float(luminance.mean()),
                "mean_saturation": float(saturation.mean()),
                "edge_energy": float(edges.mean()),
                "entropy": entropy,
            },
            provenance=self._provenance(
                "pillow", [], algorithm="rgb-luminance-edge-statistics", sample_size=sample.size
            ),
        )
        layers = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="image.depth_layers",
            confidence=0.45,
            payload={
                "method": "deterministic-saliency-proxy",
                "semantic_claim": False,
                "layers": [
                    {"role": "background", "path": None, "coverage": 1.0},
                    *mask_records,
                ],
            },
            artifacts=[item["path"] for item in mask_records],
            provenance=self._provenance(
                "pillow", [], algorithm="gradient-saturation-center-saliency", quantiles=quantiles
            ),
        )
        return [stats, layers]

    def _transcribe(self, project_id: str, asset: Asset) -> EvidenceRecord:
        source = self.store.asset_source_path(project_id, asset.id)
        result = ProviderService.for_settings(self.store.settings).invoke(
            "local-ai://speech/transcript/default",
            {
                "media_path": str(source),
                "language": None,
                "profile": "auto",
            },
            privacy_mode="local-only",
            project_id=project_id,
            asset_id=asset.id,
        )
        receipt = result.receipt
        if result.output is None or receipt.outcome not in {"succeeded", "partial"}:
            raise AnalysisError(
                "speech transcript provider failed: "
                f"{receipt.failure_code}; receipt={result.receipt_path}"
            )
        return EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="speech.transcript",
            payload=result.output,
            **_temporal_scope(asset),
            provenance=Provenance(
                tool="aoa-editing-provider-alias",
                tool_version=__version__,
                command=receipt.command,
                parameters={
                    "requested_alias": receipt.requested_alias,
                    "provider_receipt_id": receipt.id,
                    "provider_receipt_path": result.receipt_path,
                    "resolved_owner": receipt.resolved_owner,
                    "adapter_kind": receipt.adapter_kind,
                    "response_sha256": receipt.response_sha256,
                    "privacy_decision": receipt.privacy_decision.decision,
                    "output_authority": receipt.output_authority,
                    "fallback_status": receipt.fallback_status,
                },
                deterministic=receipt.adapter_kind == "deterministic-fallback",
                model_id=receipt.model_id,
                model_revision=receipt.model_revision,
            ),
        )


def _normalize(values: NDArray[np.float32]) -> NDArray[np.float32]:
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _temporal_scope(asset: Asset) -> dict[str, Any]:
    duration = asset.metadata.duration_seconds
    if duration is None or duration <= 0:
        return {}
    if asset.metadata.has_video and asset.metadata.frame_rate is not None:
        time_base = asset.metadata.frame_rate
        units = round(duration * time_base.fps)
    elif asset.metadata.has_audio and asset.metadata.sample_rate is not None:
        time_base = FrameRate(numerator=asset.metadata.sample_rate)
        units = round(duration * asset.metadata.sample_rate)
    else:
        return {}
    return {"time_base": time_base, "ranges": [FrameRange(start=0, duration=max(1, units))]}


def _entropy(values: NDArray[np.uint8]) -> float:
    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())
