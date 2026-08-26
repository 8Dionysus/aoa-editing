from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from aoa_editing.agent import AgentProtocol, AgentRequest
from aoa_editing.api.app import create_app
from aoa_editing.application.service import EditingService
from aoa_editing.application.video_jobs import VideoAnatomyJobService
from aoa_editing.config import LEGACY_ROOT_VARIABLES, Settings
from aoa_editing.domain.models import Intent, Scenario
from aoa_editing.infrastructure.store import ProjectStore

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / ".venv" / "bin" / "aoa-editing"


def _environment(home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in LEGACY_ROOT_VARIABLES:
        environment.pop(variable, None)
    environment["AOA_EDITING_HOME"] = str(home)
    return environment


def _hard_cut_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=160x90:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=160x90:rate=24:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_api_cli_and_agent_share_one_video_anatomy_contract(tmp_path: Path) -> None:
    source = tmp_path / "hard-cut.mp4"
    _hard_cut_video(source)
    home = tmp_path / "editing-home"
    settings = Settings.for_home(home)
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "Anatomy surface parity",
        Intent(text="Inspect before editing", scenario=Scenario.MEMORY_MONTAGE),
    )
    asset, _ = editing.ingest(project.id, source)

    client = TestClient(create_app(settings))
    api_response = client.post(
        f"/api/projects/{project.id}/assets/{asset.id}/video-anatomy",
        json={"profile": "structural", "pinned_frames": [0, 24, 47]},
    )
    assert api_response.status_code == 200, api_response.text
    api_payload = api_response.json()
    anatomy = api_payload["anatomy"]
    plan_id = anatomy["plan"]["id"]
    assert api_payload["job"]["status"] == "succeeded"
    assert len(anatomy["structure"]["shots"]) == 2

    agent = AgentProtocol(settings)
    agent_run = agent.execute(
        AgentRequest(
            id="run",
            method="video_anatomy.run",
            params={
                "project_id": project.id,
                "asset_id": asset.id,
                "profile": "structural",
                "pinned_frames": [47, 24, 0],
            },
        )
    )
    assert agent_run["ok"] is True
    assert agent_run["result"]["anatomy"]["plan_id"] == plan_id
    assert agent_run["result"]["job"]["cache_hit"] is True

    agent_shots = agent.execute(
        AgentRequest(
            id="shots",
            method="video_anatomy.shots",
            params={"project_id": project.id, "plan_id": plan_id},
        )
    )
    assert agent_shots["ok"] is True
    assert [item["shot"]["id"] for item in agent_shots["result"]] == [
        item["id"] for item in anatomy["structure"]["shots"]
    ]

    cli = subprocess.run(
        [str(CLI), "anatomy", "show", project.id, plan_id],
        cwd=tmp_path,
        env=_environment(home),
        check=True,
        capture_output=True,
        text=True,
    )
    cli_anatomy = json.loads(cli.stdout)
    assert cli_anatomy["id"] == anatomy["id"]
    assert cli_anatomy["anatomy_sha256"] == anatomy["anatomy_sha256"]

    api_read = client.get(f"/api/projects/{project.id}/video-anatomy/{plan_id}")
    assert api_read.status_code == 200
    assert api_read.json()["anatomy_sha256"] == cli_anatomy["anatomy_sha256"]
    contact_sheet = client.get(f"/api/projects/{project.id}/video-anatomy/{plan_id}/contact-sheet")
    assert contact_sheet.status_code == 200
    assert contact_sheet.headers["content-type"] == "image/jpeg"


def test_rebuildable_cache_prune_is_dry_run_first_and_never_touches_projects(
    tmp_path: Path,
) -> None:
    home = tmp_path / "editing-home"
    settings = Settings.for_home(home)
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "Cache isolation",
        Intent(text="Preserve canonical evidence", scenario=Scenario.MEMORY_MONTAGE),
    )
    canonical = store.project_path(project.id) / "canonical-marker.json"
    canonical.write_text('{"canonical":true}\n', encoding="utf-8")
    cached = settings.cache_root / "video-anatomy" / "vision-describe" / "cached.json"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text('{"rebuildable":true}\n', encoding="utf-8")

    service = VideoAnatomyJobService(store)
    dry_run = service.prune_rebuildable_cache()
    assert dry_run.dry_run is True
    assert dry_run.files_selected == 1
    assert cached.is_file()
    assert canonical.is_file()

    client = TestClient(create_app(settings))
    api_dry_run = client.post("/api/video-anatomy/cache/prune", json={})
    assert api_dry_run.status_code == 200
    assert api_dry_run.json()["dry_run"] is True

    agent_result = AgentProtocol(settings).execute(
        AgentRequest(id="prune", method="video_anatomy.prune_cache", params={"apply": False})
    )
    assert agent_result["ok"] is True
    assert agent_result["result"]["files_selected"] == 1

    cli = subprocess.run(
        [str(CLI), "anatomy", "prune-cache"],
        cwd=tmp_path,
        env=_environment(home),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(cli.stdout)["dry_run"] is True

    applied = service.prune_rebuildable_cache(dry_run=False)
    assert applied.files_removed == 1
    assert not cached.exists()
    assert canonical.is_file()
