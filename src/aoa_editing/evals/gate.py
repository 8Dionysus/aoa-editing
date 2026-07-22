"""Fail-closed readiness gate that never decodes sealed reference content."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aoa_editing.config import LEGACY_ROOT_VARIABLES, Settings
from aoa_editing.evals.generic import run_generic_suite
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.probes import doctor_report


class ReadinessError(RuntimeError):
    """Reference content must remain sealed because readiness is not proven."""


def run_readiness_gate(settings: Settings | None = None) -> dict[str, Any]:
    selected = settings or Settings.from_env()
    repo = Path(__file__).resolve().parents[3]
    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{revision[:10]}"
    run_root = selected.evals_root / "readiness" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    checks: list[dict[str, Any]] = []

    status = _git(repo, ["status", "--porcelain"])
    checks.append(
        _result(
            "git-clean",
            not bool(status.strip()),
            "worktree is clean" if not status.strip() else "worktree has uncommitted changes",
            {"status": status.splitlines()},
        )
    )
    executable = Path(sys.executable)
    bin_root = executable.parent
    command_checks = [
        (
            "schemas-current",
            [str(executable), "scripts/generate_schemas.py", "--check"],
            120,
        ),
        ("ruff", [str(bin_root / "ruff"), "check", "."], 180),
        ("mypy", [str(bin_root / "mypy")], 300),
        ("pytest", [str(bin_root / "pytest"), "-q"], 1200),
        (
            "ui-browser-smoke",
            [str(repo / "scripts" / "ui-smoke"), "--output", str(run_root / "ui-smoke")],
            180,
        ),
    ]
    for check_id, command, timeout in command_checks:
        checks.append(_command_check(check_id, command, repo, timeout))

    doctor = doctor_report(selected, create_roots=False)
    checks.append(
        _result(
            "doctor",
            bool(doctor["ok"]),
            "mandatory media runtime is available" if doctor["ok"] else "doctor failed",
            doctor,
        )
    )
    checks.append(_sealed_metadata_check(repo / "evals" / "reference.lock.json"))

    launcher = Path.home() / ".local" / "share" / "applications" / "aoa-editing.desktop"
    checks.append(
        _result(
            "desktop-launcher",
            launcher.is_file() and os.access(launcher, os.R_OK),
            "desktop launcher is installed" if launcher.is_file() else "desktop launcher missing",
            {"path": str(launcher)},
        )
    )

    bootstrap_root = run_root / "clean-bootstrap"
    bootstrap_env = os.environ.copy()
    for variable in LEGACY_ROOT_VARIABLES:
        bootstrap_env.pop(variable, None)
    bootstrap_env.update(
        {
            "AOA_EDITING_HOME": str(bootstrap_root),
        }
    )
    checks.append(
        _command_check(
            "clean-bootstrap",
            [str(repo / "scripts" / "bootstrap")],
            repo,
            1200,
            environment=bootstrap_env,
        )
    )

    try:
        generic = run_generic_suite(run_root / "generic")
        checks.append(
            _result(
                "generic-three-scenario-e2e",
                bool(generic.get("ok")),
                (
                    "all scenarios passed ingest, evidence, treatment, version, preview/final, "
                    "QC, and interchange"
                ),
                {
                    "receipt": str(run_root / "generic" / "generic-eval.json"),
                    "scenarios": sorted(generic["scenarios"]),
                },
            )
        )
    except Exception as error:
        checks.append(
            _result(
                "generic-three-scenario-e2e",
                False,
                "generic suite failed",
                {"error": f"{type(error).__name__}: {error}"},
            )
        )

    ok = all(check["status"] == "pass" for check in checks)
    receipt = {
        "schema": "aoa_editing_readiness_gate_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "overall": "pass" if ok else "fail",
        "reference_content_decoded": False,
        "git_revision": revision,
        "repo": str(repo),
        "run_root": str(run_root),
        "checks": checks,
    }
    receipt_path = run_root / "readiness.json"
    rendered = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    receipt_path.write_text(rendered, encoding="utf-8")
    latest = selected.evals_root / "readiness" / "latest.json"
    latest.write_text(rendered, encoding="utf-8")
    return receipt


def require_readiness(settings: Settings | None = None) -> dict[str, Any]:
    selected = settings or Settings.from_env()
    repo = Path(__file__).resolve().parents[3]
    latest = selected.evals_root / "readiness" / "latest.json"
    if not latest.is_file():
        raise ReadinessError("no readiness receipt; reference content remains sealed")
    receipt = json.loads(latest.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ReadinessError("readiness receipt is not a JSON object")
    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    if receipt.get("overall") != "pass":
        raise ReadinessError("latest readiness receipt failed")
    if receipt.get("git_revision") != revision:
        raise ReadinessError("readiness receipt belongs to a different Git revision")
    if _git(repo, ["status", "--porcelain"]).strip():
        raise ReadinessError(
            "worktree changed after readiness; reference content remains sealed"
        )
    return receipt


def _sealed_metadata_check(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    measured: dict[str, Any] = {"lock": str(lock_path), "files": {}}
    ok = True
    for name in ("source_image", "reference_video"):
        expected = lock[name]
        path = Path(expected["path"])
        exists = path.is_file() and os.access(path, os.R_OK)
        size = path.stat().st_size if exists else None
        digest = sha256_file(path) if exists else None
        matches = (
            exists
            and size == expected["size_bytes"]
            and digest == expected["sha256"]
        )
        ok = ok and matches
        measured["files"][name] = {
            "path": str(path),
            "exists_readable": exists,
            "size_bytes": size,
            "sha256": digest,
            "matches_lock": matches,
        }
    measured["operations_used"] = ["existence", "readability", "byte_size", "sha256"]
    return _result(
        "sealed-input-metadata",
        ok,
        (
            "sealed files match path/size/hash lock without decoding"
            if ok
            else "sealed metadata mismatch"
        ),
        measured,
    )


def _command_check(
    check_id: str,
    command: list[str],
    cwd: Path,
    timeout: int,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = round(time.monotonic() - started, 3)
        output = ((result.stdout or "") + (result.stderr or ""))[-8000:]
        return _result(
            check_id,
            result.returncode == 0,
            "command passed" if result.returncode == 0 else "command failed",
            {
                "command": command,
                "returncode": result.returncode,
                "duration_seconds": duration,
                "output_tail": output,
            },
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return _result(
            check_id,
            False,
            "command could not complete",
            {"command": command, "error": str(error)},
        )


def _result(
    check_id: str, passed: bool, summary: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": check_id,
        "mandatory": True,
        "status": "pass" if passed else "fail",
        "summary": summary,
        "evidence": evidence,
    }


def _git(repo: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True, timeout=30
    )
    return result.stdout
