"""Idempotent proxy, thumbnail, contact-sheet, and waveform generation."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from aoa_editing.domain.models import (
    Asset,
    DerivedMediaManifest,
    JobReceipt,
    JobStatus,
    MediaKind,
    Provenance,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore


class DerivativeError(RuntimeError):
    """Derived media failed without changing the immutable source."""


class MediaDerivativeService:
    def __init__(self, store: ProjectStore):
        self.store = store

    def derive(self, project_id: str, asset_id: str) -> DerivedMediaManifest:
        """Create or validate all applicable ingest derivatives exactly once."""

        asset = self.store.load_asset(project_id, asset_id)
        source = self.store.asset_source_path(project_id, asset_id)
        existing_path = (
            self.store.project_path(project_id) / "assets" / asset_id / "derivatives.json"
        )
        if existing_path.is_file():
            existing = self.store.load_derivatives(project_id, asset_id)
            try:
                self._validate_existing(existing, source)
                return existing
            except DerivativeError as error:
                if "source hash" in str(error):
                    raise
                project_root = self.store.project_path(project_id).resolve()
                for relative_artifact in existing.artifacts.values():
                    artifact = (project_root / relative_artifact).resolve()
                    if artifact.is_relative_to(project_root):
                        artifact.unlink(missing_ok=True)
                existing_path.unlink(missing_ok=True)

        derived_root = self.store.project_path(project_id) / "assets" / asset_id / "derived"
        derived_root.mkdir(parents=True, exist_ok=True)
        for stale in derived_root.glob(".*.partial.*"):
            stale.unlink(missing_ok=True)
        receipt = JobReceipt(
            kind="media.derive",
            project_id=project_id,
            status=JobStatus.RUNNING,
            started_at=datetime.now(UTC),
            input_hashes=[asset.sha256],
        )
        self.store.save_job(receipt)
        source_before = sha256_file(source)
        commands: list[list[str]] = []
        try:
            if asset.media_kind is MediaKind.IMAGE:
                artifacts = self._image_derivatives(source, derived_root)
                tool = "pillow"
                tool_version = version("pillow")
            else:
                artifacts, commands = self._timed_media_derivatives(
                    asset, source, derived_root
                )
                tool = "ffmpeg"
                tool_version = _ffmpeg_version()
            if sha256_file(source) != source_before or source_before != asset.sha256:
                raise DerivativeError("immutable source changed during media derivation")
            relative_paths = {
                name: str(path.relative_to(self.store.project_path(project_id)))
                for name, path in artifacts.items()
            }
            output_hashes = {name: sha256_file(path) for name, path in artifacts.items()}
            manifest = DerivedMediaManifest(
                project_id=project_id,
                asset_id=asset_id,
                source_sha256=asset.sha256,
                artifacts=relative_paths,
                output_hashes=output_hashes,
                job_id=receipt.id,
                provenance=Provenance(
                    tool=tool,
                    tool_version=tool_version,
                    command=[item for command in commands for item in command],
                    parameters={
                        "proxy_max_side": 1920 if asset.media_kind is MediaKind.IMAGE else 960,
                        "atomic_partial_promotion": True,
                        "source_mutation_allowed": False,
                    },
                    deterministic=True,
                ),
            )
            self.store.save_derivatives(manifest)
            completed = receipt.model_copy(
                update={
                    "status": JobStatus.SUCCEEDED,
                    "finished_at": datetime.now(UTC),
                    "command": manifest.provenance.command,
                    "output_paths": list(relative_paths.values()),
                    "output_hashes": list(output_hashes.values()),
                }
            )
            self.store.save_job(completed)
            return manifest
        except Exception as error:
            for stale in derived_root.glob(".*.partial.*"):
                stale.unlink(missing_ok=True)
            failed = receipt.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "finished_at": datetime.now(UTC),
                    "command": [item for command in commands for item in command],
                    "error": str(error),
                }
            )
            self.store.save_job(failed)
            if isinstance(error, DerivativeError):
                raise
            raise DerivativeError(str(error)) from error

    def _validate_existing(self, manifest: DerivedMediaManifest, source: Path) -> None:
        if manifest.source_sha256 != sha256_file(source):
            raise DerivativeError("derivative manifest source hash is stale")
        project_root = self.store.project_path(manifest.project_id).resolve()
        for name, relative in manifest.artifacts.items():
            path = (project_root / relative).resolve()
            if not path.is_relative_to(project_root) or not path.is_file():
                raise DerivativeError(f"derived artifact is missing: {name}")
            if sha256_file(path) != manifest.output_hashes.get(name):
                raise DerivativeError(f"derived artifact hash changed: {name}")

    def _image_derivatives(self, source: Path, root: Path) -> dict[str, Path]:
        with Image.open(source) as loaded:
            image = loaded.copy()
        proxy = image.copy()
        proxy.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
        proxy_path = root / "proxy.png"
        _save_image_atomic(proxy, proxy_path, "PNG")

        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((480, 270), Image.Resampling.LANCZOS)
        thumbnail_path = root / "thumbnail.jpg"
        _save_image_atomic(thumbnail, thumbnail_path, "JPEG", quality=90)

        contact = Image.new("RGB", (960, 540), "#080a0f")
        panel = image.convert("RGB")
        panel.thumbnail((920, 500), Image.Resampling.LANCZOS)
        contact.paste(
            panel,
            ((contact.width - panel.width) // 2, (contact.height - panel.height) // 2),
        )
        contact_path = root / "contact-sheet.jpg"
        _save_image_atomic(contact, contact_path, "JPEG", quality=90)
        return {
            "proxy": proxy_path,
            "thumbnail": thumbnail_path,
            "contact_sheet": contact_path,
        }

    def _timed_media_derivatives(
        self, asset: Asset, source: Path, root: Path
    ) -> tuple[dict[str, Path], list[list[str]]]:
        commands: list[list[str]] = []
        artifacts: dict[str, Path] = {}
        if asset.metadata.has_video:
            proxy = root / "proxy.mp4"
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(_partial(proxy)),
            ]
            _run_and_promote(command, proxy)
            commands.append(command)
            artifacts["proxy"] = proxy

            thumbnail = root / "thumbnail.jpg"
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=480:270:force_original_aspect_ratio=decrease",
                "-q:v",
                "2",
                str(_partial(thumbnail)),
            ]
            _run_and_promote(command, thumbnail)
            commands.append(command)
            artifacts["thumbnail"] = thumbnail

            contact = root / "contact-sheet.jpg"
            duration = max(asset.metadata.duration_seconds or 1.0, 0.04)
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"fps=6/{duration},scale=320:-2:flags=lanczos,tile=3x2",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(_partial(contact)),
            ]
            _run_and_promote(command, contact)
            commands.append(command)
            artifacts["contact_sheet"] = contact
        else:
            proxy = root / "proxy.m4a"
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(_partial(proxy)),
            ]
            _run_and_promote(command, proxy)
            commands.append(command)
            artifacts["proxy"] = proxy

        if asset.metadata.has_audio:
            waveform = root / "waveform.png"
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-filter_complex",
                "aformat=channel_layouts=mono,showwavespic=s=1280x240:colors=0x75cbd2",
                "-frames:v",
                "1",
                str(_partial(waveform)),
            ]
            _run_and_promote(command, waveform)
            commands.append(command)
            artifacts["waveform"] = waveform
            if not asset.metadata.has_video:
                with Image.open(waveform) as loaded:
                    panel = loaded.convert("RGB")
                audio_thumbnail = panel.copy()
                audio_thumbnail.thumbnail((480, 270), Image.Resampling.LANCZOS)
                thumbnail_path = root / "thumbnail.jpg"
                _save_image_atomic(audio_thumbnail, thumbnail_path, "JPEG", quality=90)
                contact_path = root / "contact-sheet.jpg"
                _save_image_atomic(panel, contact_path, "JPEG", quality=90)
                artifacts["thumbnail"] = thumbnail_path
                artifacts["contact_sheet"] = contact_path
        return artifacts, commands


def _partial(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def _run_and_promote(command: list[str], destination: Path) -> None:
    partial = _partial(destination)
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise DerivativeError(result.stderr[-4000:] or "FFmpeg derivative failed")
    if not partial.is_file() or partial.stat().st_size == 0:
        raise DerivativeError(f"FFmpeg did not create {destination.name}")
    os.replace(partial, destination)


def _save_image_atomic(
    image: Image.Image, destination: Path, format_name: str, **kwargs: int
) -> None:
    partial = _partial(destination)
    image.save(partial, format=format_name, **kwargs)
    os.replace(partial, destination)


def _ffmpeg_version() -> str:
    result = subprocess.run(
        ["ffmpeg", "-version"], check=False, capture_output=True, text=True, timeout=10
    )
    return result.stdout.splitlines()[0] if result.stdout else "unknown"
