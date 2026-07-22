from __future__ import annotations

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from aoa_editing.agent import AgentProtocol, AgentRequest
from aoa_editing.api.app import create_app
from aoa_editing.config import Settings


def _settings(tmp_path):  # type: ignore[no-untyped-def]
    return Settings.for_home(tmp_path / "editing-home")


def _image(path) -> None:  # type: ignore[no-untyped-def]
    image = Image.new("RGB", (320, 180), "#111827")
    draw = ImageDraw.Draw(image)
    draw.ellipse((65, 15, 255, 175), fill="#f59e0b")
    draw.rectangle((145, 50, 180, 165), fill="#fff7ed")
    image.save(path)


def test_http_workbench_full_still_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.png"
    _image(source)
    client = TestClient(create_app(_settings(tmp_path)))
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["local_first"] is True
    created = client.post(
        "/api/projects",
        json={
            "name": "API fixture",
            "intent": "A subtle layered move",
            "scenario": "still.motion",
            "target_duration_seconds": 1,
        },
    )
    assert created.status_code == 201
    project_id = created.json()["id"]
    initial_bundle = client.get(f"/api/projects/{project_id}").json()
    assert len(initial_bundle["brief_revisions"]) == 1
    revised = client.post(
        f"/api/projects/{project_id}/briefs",
        json={
            "intent": "A restrained layered move",
            "scenario": "still.motion",
            "target_duration_seconds": 1,
            "mood": "contemplative",
            "rationale": "Mood confirmed by the editor",
        },
    )
    assert revised.status_code == 201, revised.text
    assert revised.json()["revision"] == 2
    with source.open("rb") as stream:
        uploaded = client.post(
            f"/api/projects/{project_id}/assets",
            files={"file": (source.name, stream, "image/png")},
        )
    assert uploaded.status_code == 201, uploaded.text
    asset_id = uploaded.json()["asset"]["id"]
    analyzed = client.post(
        f"/api/projects/{project_id}/assets/{asset_id}/analyze",
        json={"transcribe": False},
    )
    assert analyzed.status_code == 200
    assert {item["kind"] for item in analyzed.json()} == {
        "image.statistics",
        "image.depth_layers",
    }
    statistics = next(item for item in analyzed.json() if item["kind"] == "image.statistics")
    corrected = client.post(
        f"/api/projects/{project_id}/evidence/corrections",
        json={
            "asset_id": asset_id,
            "kind": "image.statistics",
            "payload": {**statistics["payload"], "editor_note": "reviewed"},
            "supersedes": [statistics["id"]],
            "rationale": "Visual review",
        },
    )
    assert corrected.status_code == 201, corrected.text
    assert corrected.json()["authority"] == "human-correction"
    proposed = client.post(
        f"/api/projects/{project_id}/treatments/options",
        json={"duration_seconds": 1},
    )
    assert proposed.status_code == 201, proposed.text
    options = proposed.json()
    assert len(options) == 3
    assert len({item["patch"]["id"] for item in options}) == 3
    accepted = client.post(
        f"/api/projects/{project_id}/treatments/{options[0]['id']}/accept"
    )
    assert accepted.status_code == 201
    version_id = accepted.json()["id"]
    previewed = client.post(
        f"/api/projects/{project_id}/patches/preview-language",
        json={"command": "Сделай фон белым", "version_id": version_id},
    )
    assert previewed.status_code == 200
    assert previewed.json()["requires_confirmation"] is True
    assert previewed.json()["patch"]["operations"] == [
        {"op": "replace", "path": "/background", "value": "#ffffff"}
    ]
    clip_id = accepted.json()["timeline"]["tracks"][0]["clips"][0]["id"]
    pin_preview = client.post(
        f"/api/projects/{project_id}/patches/preview-language",
        json={"command": f"Закрепи {clip_id}", "version_id": version_id},
    )
    assert pin_preview.status_code == 200
    assert pin_preview.json()["patch"]["operations"][0]["path"].endswith("/pinned")
    assert pin_preview.json()["patch"]["operations"][0]["value"] is True
    assert len(client.get(f"/api/projects/{project_id}").json()["versions"]) == 2
    rendered = client.post(
        f"/api/projects/{project_id}/versions/{version_id}/render?profile=preview"
    )
    assert rendered.status_code == 201, rendered.text
    quality = client.post(
        f"/api/projects/{project_id}/versions/{version_id}/qc?profile=preview"
    )
    assert quality.status_code == 201
    assert quality.json()["overall"] in {"pass", "warn"}
    exported = client.post(
        f"/api/projects/{project_id}/versions/{version_id}/export/otio"
    )
    assert exported.status_code == 200
    bundle = client.get(f"/api/projects/{project_id}").json()
    assert len(bundle["versions"]) == 2
    media = client.get(
        f"/api/projects/{project_id}/file",
        params={"path": f"renders/{version_id}/preview/video.mp4"},
    )
    assert media.status_code == 200
    assert media.headers["content-type"] == "video/mp4"
    style = client.post(
        "/api/style-profiles",
        json={
            "scope": project_id,
            "explicit_opt_in": True,
            "confirmations": [{"preference": "motion", "value": "restrained"}],
        },
    )
    assert style.status_code == 201
    assert client.get("/api/style-profiles").json()[0]["id"] == style.json()["id"]
    rejected_style = client.post(
        "/api/style-profiles",
        json={
            "scope": project_id,
            "explicit_opt_in": False,
            "confirmations": [{"preference": "motion", "value": "aggressive"}],
        },
    )
    assert rejected_style.status_code == 422


def test_agent_protocol_is_allowlisted_and_uses_same_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    protocol = AgentProtocol(_settings(tmp_path))
    created = protocol.execute(
        AgentRequest(
            id=1,
            method="project.create",
            params={
                "name": "Agent fixture",
                "intent": "Keep it inspectable",
                "scenario": "still.motion",
            },
        )
    )
    assert created["ok"] is True
    listed = protocol.execute(AgentRequest(id=2, method="project.list"))
    assert listed["result"][0]["name"] == "Agent fixture"
    preview_without_timeline = protocol.execute(
        AgentRequest(
            id="language",
            method="patch.preview_language",
            params={"project_id": created["result"]["id"], "command": "Сделай фон белым"},
        )
    )
    assert preview_without_timeline["ok"] is False
    rejected = protocol.execute(AgentRequest(id=3, method="shell.exec"))
    assert rejected == {
        "id": 3,
        "ok": False,
        "error": {"code": "method_not_allowed", "message": "shell.exec"},
    }
