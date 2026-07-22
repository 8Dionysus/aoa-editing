from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from aoa_editing.agent import AgentProtocol, AgentRequest
from aoa_editing.api.app import create_app
from aoa_editing.application.service import EditingService
from aoa_editing.config import LEGACY_ROOT_VARIABLES, Settings
from aoa_editing.domain.models import Intent, Scenario
from aoa_editing.infrastructure.store import ProjectStore

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / ".venv" / "bin" / "aoa-editing"
PYTHON = ROOT / ".venv" / "bin" / "python"


def _environment(home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in LEGACY_ROOT_VARIABLES:
        environment.pop(variable, None)
    environment["AOA_EDITING_HOME"] = str(home)
    return environment


def test_shell_resolver_is_independent_of_cwd(tmp_path: Path) -> None:
    home = tmp_path / "portable-home"
    result = subprocess.run(
        ["python3.12", str(ROOT / "scripts" / "resolve_home.py"), "projects"],
        cwd=tmp_path,
        env=_environment(home),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str((home / "var" / "projects").resolve())


def test_ui_cli_agent_and_restart_share_the_same_projects(tmp_path: Path) -> None:
    home = tmp_path / "shared-home"
    settings = Settings.for_home(home)
    editing = EditingService(ProjectStore(settings))
    project = editing.create_project(
        "Shared surface",
        Intent(text="Persist across every adapter", scenario=Scenario.STILL_MOTION),
    )

    api_projects = TestClient(create_app(settings)).get("/api/projects").json()
    agent = AgentProtocol(settings).execute(AgentRequest(id=1, method="project.list", params={}))
    cli = subprocess.run(
        [str(CLI), "project", "list"],
        cwd=tmp_path,
        env=_environment(home),
        check=True,
        capture_output=True,
        text=True,
    )
    cli_projects = json.loads(cli.stdout)

    assert [item["id"] for item in api_projects] == [project.id]
    assert [item["id"] for item in agent["result"]] == [project.id]
    assert [item["id"] for item in cli_projects] == [project.id]

    restarted = ProjectStore(Settings.from_env(environ=_environment(home), source_home=ROOT))
    assert [item.id for item in restarted.list_projects()] == [project.id]


def test_cleanup_refuses_compatibility_roots_outside_the_home(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("AOA_EDITING_HOME", None)
    for variable in LEGACY_ROOT_VARIABLES:
        environment.pop(variable, None)
    environment["AOA_EDITING_CACHE_ROOT"] = str(tmp_path / "legacy-cache")
    result = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "clean_workspace.py"),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing unsafe cache root" in (result.stdout + result.stderr)
