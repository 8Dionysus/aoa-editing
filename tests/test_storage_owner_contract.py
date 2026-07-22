from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "decisions" / "ADR-0008-repo-local-home-and-provider-aliases.md"


def test_storage_decision_partially_supersedes_adr_0002() -> None:
    decision = ADR.read_text(encoding="utf-8")
    prior = (
        ROOT / "docs" / "decisions" / "ADR-0002-local-runtime-and-provider-ports.md"
    ).read_text(encoding="utf-8")
    index = (ROOT / "docs" / "decisions" / "README.md").read_text(encoding="utf-8")

    assert "- Status: accepted" in decision
    assert "partially supersedes ADR-0002" in decision
    assert "semantic ownership" in decision
    assert "physical placement" in decision
    assert "filesystem symlink" in decision
    assert "prototype-v0.1.0" in decision
    assert "Partially superseded by ADR-0008" in prior
    assert "ADR-0008-repo-local-home-and-provider-aliases.md" in index


def test_storage_layout_manifest_has_one_product_owned_default_home() -> None:
    payload = json.loads((ROOT / "manifests" / "storage-layout.json").read_text(encoding="utf-8"))
    paths = payload["paths"]

    assert payload["schema"] == "aoa_editing_storage_layout_v1"
    assert payload["default_home"] == "repo-local"
    assert paths["runtime"]["relative"] == ".venv"
    assert paths["projects"]["relative"] == "var/projects"
    assert paths["evals"]["relative"] == "var/evals"
    assert paths["artifacts"]["relative"] == "var/artifacts"
    assert paths["cache"]["relative"] == "var/cache"
    assert paths["tmp"]["relative"] == "var/tmp"
    assert paths["state"]["relative"] == "var/state"
    assert {entry["owner"] for entry in paths.values()} == {"aoa-editing"}
    assert payload["shared_ai"]["owner"] == "abyss-stack_and_abyss-machine"
    assert payload["shared_ai"]["binding"] == "typed_provider_alias"
    assert payload["shared_ai"]["declaration_catalog"] == "manifests/ai-capabilities.json"
    assert payload["shared_ai"]["local_binding_overlay"] == ".aoa-editing.local.toml"
    assert payload["shared_ai"]["receipt_root"] == "var/state/provider-receipts"
    assert payload["shared_ai"]["filesystem_links_allowed"] is False
    assert payload["legacy"]["cleanup_authorized"] is False
    assert payload["readiness_recertification"] == {
        "runtime_receipt": "var/evals/readiness/latest.json",
        "revision_bound": True,
        "required_overall": "pass",
        "required_before_checkpoint_tag": True,
    }
    assert payload["implementation_state"] == "cutover_verified"


def test_owner_docs_no_longer_assign_product_state_to_abyss_machine() -> None:
    documents = [
        ROOT / "AGENTS.md",
        ROOT / "DESIGN.md",
        ROOT / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "boundaries.md",
        ROOT / "docs" / "operations.md",
        ROOT / "docs" / "troubleshooting.md",
        ROOT / "evals" / "AGENTS.md",
    ]
    forbidden_claims = (
        "Default durable project data belongs to the host layer",
        "Project runtime belongs to the host layer",
        "All runtime and project data is routed outside the repository",
    )

    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert "repo-local" in text.lower() or "repository home" in text.lower(), path
        for claim in forbidden_claims:
            assert claim not in text, (path, claim)
