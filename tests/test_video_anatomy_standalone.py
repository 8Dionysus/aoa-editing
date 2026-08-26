from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / ".venv" / "bin" / "aoa-editing"


def _standalone_environment(home: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AOA_", "ABYSS_", "CODEX_", "CLAUDE_"))
    }
    environment["AOA_EDITING_HOME"] = str(home)
    return environment


def _run(arguments: list[str], *, home: Path, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        [str(CLI), *arguments],
        cwd=cwd,
        env=_standalone_environment(home),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def test_video_anatomy_runs_from_fresh_home_without_abyss_or_provider_host(
    tmp_path: Path,
) -> None:
    source = tmp_path / "standalone.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=24000/1001:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        timeout=60,
    )
    home = tmp_path / "fresh-editing-home"
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()

    project = _run(
        [
            "project",
            "create",
            "--name",
            "Standalone anatomy",
            "--scenario",
            "memory.montage",
            "--intent",
            "Measure before editing",
        ],
        home=home,
        cwd=unrelated_cwd,
    )
    project_id = str(project["id"])
    ingested = _run(
        ["asset", "ingest", project_id, str(source)],
        home=home,
        cwd=unrelated_cwd,
    )
    asset = ingested["asset"]
    assert isinstance(asset, dict)
    asset_id = str(asset["id"])

    estimate = _run(
        ["anatomy", "estimate", project_id, asset_id, "--profile", "structural"],
        home=home,
        cwd=unrelated_cwd,
    )
    assert estimate["admitted"] is True
    result = _run(
        ["anatomy", "run", project_id, asset_id, "--profile", "structural"],
        home=home,
        cwd=unrelated_cwd,
    )
    anatomy = result["anatomy"]
    assert isinstance(anatomy, dict)
    assert anatomy["source_sha256"] == asset["sha256"]
    assert anatomy["canonical_edit_decision"] is False
    assert result["editorial_proposal"] is None
    assert result["reconstruction_proposal"] is None
    assert not (home / ".aoa-editing.local.toml").exists()
