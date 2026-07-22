from __future__ import annotations

import sqlite3
import subprocess

import pytest

from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import Intent, Scenario
from aoa_editing.infrastructure.derivatives import MediaDerivativeService
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ForbiddenSourceError, ProjectStore


def _settings(tmp_path):  # type: ignore[no-untyped-def]
    return Settings.for_home(tmp_path / "editing-home")


@pytest.fixture
def tiny_image(tmp_path):  # type: ignore[no-untyped-def]
    path = tmp_path / "tiny.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x48",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
    )
    return path


def test_project_and_ingest_are_idempotent(tmp_path, tiny_image) -> None:  # type: ignore[no-untyped-def]
    store = ProjectStore(_settings(tmp_path))
    service = EditingService(store)
    project = service.create_project(
        "Fixture", Intent(text="Animate this still", scenario=Scenario.STILL_MOTION)
    )
    first, evidence = service.ingest(project.id, tiny_image)
    second, second_evidence = service.ingest(project.id, tiny_image)
    assert first == second
    assert evidence == second_evidence
    assert store.asset_source_path(project.id, first.id).read_bytes() == tiny_image.read_bytes()
    assert len(store.load_project(project.id).assets) == 1
    derivatives = store.load_derivatives(project.id, first.id)
    assert set(derivatives.artifacts) == {"proxy", "thumbnail", "contact_sheet"}
    for relative in derivatives.artifacts.values():
        assert (store.project_path(project.id) / relative).is_file()
    assert len([job for job in store.list_jobs(project.id) if job.kind == "media.derive"]) == 1


def test_sealed_reference_hash_fails_closed(tmp_path, tiny_image) -> None:  # type: ignore[no-untyped-def]
    store = ProjectStore(_settings(tmp_path))
    project = store.create_project(
        "Fixture", Intent(text="Animate", scenario=Scenario.STILL_MOTION)
    )
    sealed = project.model_copy(
        update={"sealed_reference_hashes": [sha256_file(tiny_image)]}
    )
    store.save_project(sealed)
    with pytest.raises(ForbiddenSourceError, match="sealed for evaluation"):
        store.ingest_asset(project.id, tiny_image)


def test_treatment_acceptance_versions_the_editorial_decision_graph(
    tmp_path, tiny_image
) -> None:  # type: ignore[no-untyped-def]
    store = ProjectStore(_settings(tmp_path))
    editing = EditingService(store)
    project = editing.create_project(
        "Decision fixture",
        Intent(text="Animate with reviewable reasons", scenario=Scenario.STILL_MOTION),
    )
    asset, _ = editing.ingest(project.id, tiny_image)
    AnalysisService(store).analyze(project.id, asset.id)

    treatment = editing.propose(project.id, duration_seconds=1)
    assert treatment.decision_graph_id is not None
    proposed = store.load_decision_graph(project.id, treatment.decision_graph_id)
    assert proposed.state == "proposed"
    assert proposed.decisions
    assert proposed.decisions[0].evidence_refs == treatment.patch.evidence_refs
    assert proposed.decisions[0].approval == "proposed"
    assert proposed.decisions[0].reversible_operations == []

    version = editing.accept_treatment(project.id, treatment.id)
    assert version.decision_graph_id is not None
    approved = store.load_decision_graph(project.id, version.decision_graph_id)
    assert approved.parent_graph_id == proposed.id
    assert approved.version_id == version.id
    assert approved.state == "approved"
    assert [item.id for item in approved.decisions] == [item.id for item in proposed.decisions]
    assert all(item.approval == "approved" for item in approved.decisions)
    assert all(item.reversible_operations for item in approved.decisions)
    assert {entry.action for entry in store.load_decision_log(project.id).entries} >= {
        "proposed",
        "approved",
    }


def test_restart_rebuilds_index_and_recovers_missing_derivative(tmp_path, tiny_image) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "Recovery", Intent(text="Recover safely", scenario=Scenario.STILL_MOTION)
    )
    asset, _ = editing.ingest(project.id, tiny_image)
    derivatives = store.load_derivatives(project.id, asset.id)
    missing = store.project_path(project.id) / derivatives.artifacts["thumbnail"]
    missing.unlink()
    stale = missing.parent / ".abandoned.partial.jpg"
    stale.write_bytes(b"incomplete")

    for suffix in ("", "-wal", "-shm"):
        (settings.state_root / f"index.sqlite3{suffix}").unlink(missing_ok=True)
    restarted = ProjectStore(settings)
    recovered = MediaDerivativeService(restarted).derive(project.id, asset.id)

    assert restarted.load_project(project.id).id == project.id
    assert (restarted.project_path(project.id) / recovered.artifacts["thumbnail"]).is_file()
    assert not stale.exists()
    assert len(
        [job for job in restarted.list_jobs(project.id) if job.kind == "media.derive"]
    ) == 2
    with sqlite3.connect(settings.state_root / "index.sqlite3") as database:
        counts = tuple(
            database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("projects", "assets", "jobs")
        )
    assert counts == (1, 1, 2)
