"""Media hashing and ffprobe-backed metadata extraction."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from aoa_editing.domain.models import FrameRate, MediaKind, MediaMetadata

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


class MediaProbeError(RuntimeError):
    """Media could not be identified by the baseline probe."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(value: str | None) -> FrameRate | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if rate <= 0:
        return None
    return FrameRate(numerator=rate.numerator, denominator=rate.denominator)


def ffprobe(path: Path) -> tuple[MediaKind, MediaMetadata, dict[str, Any], list[str]]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise MediaProbeError(result.stderr.strip() or f"ffprobe failed for {path}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise MediaProbeError("ffprobe returned invalid JSON") from error
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES and video is not None:
        kind = MediaKind.IMAGE
    elif video is not None:
        kind = MediaKind.VIDEO
    elif audio is not None or suffix in AUDIO_SUFFIXES:
        kind = MediaKind.AUDIO
    else:
        raise MediaProbeError("no supported audio, video, or image stream found")
    raw_duration = payload.get("format", {}).get("duration")
    if raw_duration is None and video is not None:
        raw_duration = video.get("duration")
    try:
        duration = float(raw_duration) if raw_duration not in {None, "N/A"} else None
    except (TypeError, ValueError):
        duration = None
    metadata = MediaMetadata(
        duration_seconds=duration,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        frame_rate=_fraction(video.get("avg_frame_rate")) if video else None,
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        channels=audio.get("channels") if audio else None,
        format_name=payload.get("format", {}).get("format_name"),
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        pixel_format=video.get("pix_fmt") if video else None,
        color_space=video.get("color_space") if video else None,
        has_video=video is not None,
        has_audio=audio is not None,
    )
    return kind, metadata, payload, command

