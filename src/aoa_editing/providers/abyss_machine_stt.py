"""Normalize the abyss-machine dictation bridge into the typed transcript alias."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ABYSS_MACHINE = "/usr/local/bin/abyss-machine"
PROTOCOL_VERSION = "aoa-local-ai-v1"
RunJSON = Callable[[Sequence[str], float], dict[str, Any]]


class HostSTTAdapterError(RuntimeError):
    """The owner CLI failed or returned an unusable contract."""


def host_stt_health(run_json: RunJSON | None = None) -> dict[str, Any]:
    """Return fresh path-free health normalized from three owner read models."""

    runner = run_json or _run_json
    capabilities = runner([ABYSS_MACHINE, "ai", "capabilities", "--json"], 15)
    dictation = runner([ABYSS_MACHINE, "dictation", "status", "--json"], 15)
    inventory = runner([ABYSS_MACHINE, "ai", "models", "--json"], 30)
    return normalize_host_health(capabilities, dictation, inventory)


def normalize_host_health(
    capabilities: dict[str, Any],
    dictation: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    stt = _mapping(_mapping(capabilities.get("capabilities")).get("stt"))
    profiles = _mapping(dictation.get("profiles"))
    quality = _mapping(profiles.get("quality"))
    entries = inventory.get("entries")
    selected_entry = next(
        (
            item
            for item in entries
            if isinstance(item, dict)
            and item.get("category") == "stt_whisper_openvino"
            and item.get("name") == "whisper-large-v3-turbo"
        ),
        None,
    ) if isinstance(entries, list) else None
    inventory_revision = (
        _fingerprint_inventory_entry(selected_entry)
        if isinstance(selected_entry, dict)
        else None
    )
    model_id = str(quality.get("model_id") or "openai/whisper-large-v3-turbo")
    backend = str(stt.get("host_recommended_backend") or quality.get("device") or "unknown")
    ready = all(
        (
            capabilities.get("ok") is True,
            stt.get("status") == "ready",
            quality.get("enabled") is True,
            quality.get("model_dir_exists") is True,
            dictation.get("server_socket_exists") is True,
            inventory_revision is not None,
        )
    )
    return {
        "status": "ready" if ready else "unavailable",
        "observed_at": capabilities.get("generated_at"),
        "protocol_version": PROTOCOL_VERSION,
        "backend": backend,
        "model_id": model_id,
        "model_revision": inventory_revision,
        "owner_version": capabilities.get("version"),
        "evidence": {
            "capabilities_schema": capabilities.get("schema"),
            "dictation_schema": dictation.get("schema"),
            "inventory_schema": inventory.get("schema"),
            "inventory_category": (
                selected_entry.get("category") if isinstance(selected_entry, dict) else None
            ),
            "inventory_name": (
                selected_entry.get("name") if isinstance(selected_entry, dict) else None
            ),
            "inventory_entry_sha256": inventory_revision,
            "server_socket_ready": dictation.get("server_socket_exists") is True,
        },
    }


def transcribe_with_host(
    media_path: Path,
    *,
    profile: str = "auto",
    run_json: RunJSON | None = None,
) -> dict[str, Any]:
    runner = run_json or _run_json
    raw = runner(
        [
            ABYSS_MACHINE,
            "dictation",
            "transcribe",
            str(media_path),
            "--profile",
            profile,
            "--json",
        ],
        1800,
    )
    return normalize_host_transcript(raw)


def normalize_host_transcript(raw: dict[str, Any]) -> dict[str, Any]:
    """Remove host paths and expose only evidence admitted by the product schema."""

    text = str(raw.get("text") or "").strip()
    language = raw.get("language")
    raw_segments = raw.get("segments")
    segments: list[dict[str, Any]] = []
    segment_text_available = False
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            start = _number(item.get("start", item.get("start_sec")))
            end = _number(item.get("end", item.get("end_sec")))
            if start is None or end is None or end <= start:
                continue
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": str(item["text"]).strip(),
                    "speaker": item.get("speaker"),
                }
            )
            segment_text_available = True
    if not segments:
        duration = _duration_seconds(raw, raw_segments)
        if duration is None or duration <= 0:
            raise HostSTTAdapterError("host transcript omitted usable duration evidence")
        segments = [{"start": 0.0, "end": duration, "text": text, "speaker": None}]
    return {
        "text": text,
        "language": str(language) if language is not None else None,
        "segments": segments,
        "partial": bool(raw.get("partial", False)) or not segment_text_available,
        "temporal_precision": "segment" if segment_text_available else "file",
        "host_schema": raw.get("schema"),
        "host_profile": raw.get("profile_name"),
        "host_via": raw.get("via"),
        "host_segment_timing_available": bool(raw_segments),
        "host_segment_text_available": segment_text_available,
    }


def _fingerprint_inventory_entry(entry: dict[str, Any]) -> str:
    permitted = {
        key: entry.get(key)
        for key in (
            "kind",
            "category",
            "name",
            "relative_path",
            "artifacts",
            "file_summary",
            "read_only_source",
        )
    }
    payload = json.dumps(
        permitted,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _duration_seconds(
    raw: dict[str, Any],
    raw_segments: object,
) -> float | None:
    for key in ("raw_audio_duration_sec", "processed_audio_duration_sec", "duration"):
        selected = _number(raw.get(key))
        if selected is not None and selected > 0:
            return selected
    if isinstance(raw_segments, list):
        ends = [
            selected
            for item in raw_segments
            if isinstance(item, dict)
            for selected in [_number(item.get("end", item.get("end_sec")))]
            if selected is not None
        ]
        if ends:
            return max(ends)
    return None


def _number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _run_json(command: Sequence[str], timeout: float) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HostSTTAdapterError(f"cannot run owner command: {error}") from error
    if result.returncode != 0:
        raise HostSTTAdapterError(result.stderr[-2000:] or "owner command failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise HostSTTAdapterError("owner command returned malformed JSON") from error
    if not isinstance(payload, dict):
        raise HostSTTAdapterError("owner command returned a non-object JSON document")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")
    transcribe = subparsers.add_parser("transcribe")
    transcribe.add_argument("media_path", type=Path)
    transcribe.add_argument("--profile", default="auto")
    arguments = parser.parse_args()
    try:
        payload = (
            host_stt_health()
            if arguments.command == "health"
            else transcribe_with_host(arguments.media_path, profile=arguments.profile)
        )
    except HostSTTAdapterError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
