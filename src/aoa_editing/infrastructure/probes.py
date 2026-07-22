"""Read-only host capability probes used by doctor and receipts."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from aoa_editing.config import Settings
from aoa_editing.providers.catalog import ProviderConfigurationError
from aoa_editing.providers.service import ProviderService


def _run(command: list[str], timeout: float = 8) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def _version(command: list[str]) -> dict[str, Any]:
    ok, output = _run(command)
    return {"available": ok, "version": output.splitlines()[0] if output else None}


def _path_status(path: Path, create: bool) -> dict[str, Any]:
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return {"path": str(path), "writable": False, "error": str(error)}
    anchor = path if path.exists() else next((p for p in path.parents if p.exists()), path)
    writable = anchor.exists() and os_access_write(anchor)
    return {"path": str(path), "exists": path.exists(), "writable": writable}


def os_access_write(path: Path) -> bool:
    import os

    return os.access(path, os.W_OK)


def doctor_report(settings: Settings, *, create_roots: bool = False) -> dict[str, Any]:
    if create_roots:
        settings.ensure_roots()
    ffmpeg = _version(["ffmpeg", "-version"])
    ffprobe = _version(["ffprobe", "-version"])
    flatpak = _version(["flatpak", "--version"])
    kdenlive_ok, kdenlive_output = _run(["flatpak", "info", "org.kde.kdenlive"])
    melt_ok, melt_output = _run(
        ["flatpak", "run", "--command=melt", "org.kde.kdenlive", "-version"], timeout=15
    )
    host_ai_ok, host_ai_output = _run(["abyss-machine", "ai", "capabilities", "--json"])
    host_ai: dict[str, Any] | None = None
    if host_ai_ok:
        try:
            parsed = json.loads(host_ai_output)
            capabilities = parsed.get("capabilities", {})
            host_ai = {
                name: value.get("status") if isinstance(value, dict) else None
                for name, value in capabilities.items()
            }
        except json.JSONDecodeError:
            host_ai_ok = False
    provider_contract_ok = True
    provider_status: dict[str, Any]
    try:
        provider_status = ProviderService(settings).inspect_bindings(probe=True)
    except ProviderConfigurationError as error:
        provider_contract_ok = False
        provider_status = {
            "schema": "aoa_editing_provider_status_v1",
            "error": str(error),
            "bindings": [],
        }
    commands = {
        name: shutil.which(name)
        for name in ("ffmpeg", "ffprobe", "git", "flatpak", "abyss-machine")
    }
    mandatory = ffmpeg["available"] and ffprobe["available"] and provider_contract_ok
    return {
        "schema": "aoa_editing_doctor_v2",
        "ok": mandatory,
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release()},
        "commands": commands,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "flatpak": flatpak,
        "kdenlive": {
            "available": kdenlive_ok,
            "summary": next(
                (line.strip() for line in kdenlive_output.splitlines() if "Version:" in line),
                None,
            ),
        },
        "melt": {
            "available": melt_ok,
            "version": melt_output.splitlines()[0] if melt_output else None,
        },
        "host_ai": {"available": host_ai_ok, "capabilities": host_ai},
        "providers": {
            "contract_ok": provider_contract_ok,
            **provider_status,
        },
        "editing_home": {
            "path": str(settings.editing_home),
            "owner": "aoa-editing",
            "resolution_source": settings.layout.resolution_source,
            "local_overlay": str(settings.local_overlay),
            "compatibility_overrides": list(settings.layout.compatibility_overrides),
        },
        "paths": {
            name: {
                **_path_status(path, create_roots and name != "runtime"),
                "owner": "aoa-editing",
            }
            for name, path in {
                "runtime": settings.runtime_root,
                "projects": settings.projects_root,
                "evals": settings.evals_root,
                "artifacts": settings.artifacts_root,
                "cache": settings.cache_root,
                "tmp": settings.tmp_root,
                "state": settings.state_root,
            }.items()
        },
    }
