from __future__ import annotations

import json
from pathlib import Path

from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import Intent, ProjectManifest, Scenario
from aoa_editing.infrastructure.relocation import (
    inspect_legacy_projects,
    migrate_legacy_storage,
    tree_measurement,
)
from aoa_editing.infrastructure.store import ProjectStore


def _exclude_chromium_profile(relative: Path) -> bool:
    return "chromium-profile" in relative.parts


def _legacy_fixture(root: Path) -> tuple[Settings, str]:
    settings = Settings.for_home(root)
    editing = EditingService(ProjectStore(settings))
    project = editing.create_project(
        "Relocatable project",
        Intent(text="Keep semantic identity", scenario=Scenario.STILL_MOTION),
    )
    (settings.evals_root / "generic").mkdir(parents=True)
    (settings.evals_root / "generic" / "receipt.json").write_text(
        '{"overall":"pass"}\n', encoding="utf-8"
    )
    (settings.evals_root / "reference").mkdir()
    (settings.evals_root / "reference" / "proof.json").write_text(
        '{"sealed":true}\n', encoding="utf-8"
    )
    readiness = settings.evals_root / "readiness" / "run-v1"
    (readiness / "clean-bootstrap" / "runtime").mkdir(parents=True)
    chromium_profile = readiness / "ui-smoke" / "run" / "chromium-profile"
    chromium_profile.mkdir(parents=True)
    (readiness / "readiness.json").write_text('{"overall":"pass"}\n', encoding="utf-8")
    (readiness / "clean-bootstrap" / "runtime" / "derived.bin").write_bytes(b"rebuild me")
    (chromium_profile / "SingletonLock").symlink_to("host-process")
    return settings, project.id


def test_dry_run_is_read_only_and_reports_all_copy_classes(tmp_path: Path) -> None:
    legacy, _ = _legacy_fixture(tmp_path / "legacy")
    destination = Settings.for_home(tmp_path / "destination")
    before = (
        tree_measurement(legacy.projects_root),
        tree_measurement(legacy.evals_root, exclude=_exclude_chromium_profile),
    )

    report = migrate_legacy_storage(
        legacy_data_root=legacy.data_root,
        destination=destination,
        inventory_sha256="1" * 64,
        dry_run=True,
    )

    assert report.overall == "pass"
    assert report.dry_run is True
    assert {entry.id for entry in report.entries} == {
        "projects",
        "evals-generic",
        "evals-reference",
        "evals-readiness",
        "sqlite-index",
    }
    assert not destination.layout.var_root.exists()
    assert (
        tree_measurement(legacy.projects_root),
        tree_measurement(legacy.evals_root, exclude=_exclude_chromium_profile),
    ) == before


def test_verified_copy_relocates_meaning_without_links_or_runtime_copies(
    tmp_path: Path,
) -> None:
    legacy, project_id = _legacy_fixture(tmp_path / "legacy")
    destination = Settings.for_home(tmp_path / "destination")
    source_project = legacy.projects_root / project_id
    source_manifest = json.loads((source_project / "project.json").read_text(encoding="utf-8"))
    before = (
        tree_measurement(legacy.projects_root),
        tree_measurement(legacy.evals_root, exclude=_exclude_chromium_profile),
    )

    report = migrate_legacy_storage(
        legacy_data_root=legacy.data_root,
        destination=destination,
        inventory_sha256="2" * 64,
        dry_run=False,
    )

    relocated = ProjectStore(destination)
    destination_project = destination.projects_root / project_id
    destination_manifest = json.loads(
        (destination_project / "project.json").read_text(encoding="utf-8")
    )
    assert report.overall == "pass"
    assert report.cutover_ready is True
    assert report.legacy_cleanup_authorized is False
    assert report.filesystem_links_created is False
    assert relocated.load_project(project_id) == ProjectManifest.model_validate(source_manifest)
    assert destination_manifest == source_manifest
    assert str(destination.editing_home) not in json.dumps(destination_manifest)
    assert source_project.stat().st_ino != destination_project.stat().st_ino
    assert not (
        destination.evals_root / "readiness" / "run-v1" / "clean-bootstrap" / "runtime"
    ).exists()
    assert not (
        destination.evals_root
        / "readiness"
        / "run-v1"
        / "ui-smoke"
        / "run"
        / "chromium-profile"
    ).exists()
    assert (destination.evals_root / "readiness" / "run-v1" / "readiness.json").is_file()
    assert (
        tree_measurement(legacy.projects_root),
        tree_measurement(legacy.evals_root, exclude=_exclude_chromium_profile),
    ) == before
    assert (
        legacy.evals_root
        / "readiness"
        / "run-v1"
        / "ui-smoke"
        / "run"
        / "chromium-profile"
        / "SingletonLock"
    ).is_symlink()
    assert (destination.artifacts_root / "migrations" / f"{report.id}.json").is_file()


def test_legacy_project_manifests_are_readable_without_writing_source(
    tmp_path: Path,
) -> None:
    legacy, _ = _legacy_fixture(tmp_path / "legacy")
    before = tree_measurement(legacy.projects_root)

    manifests = inspect_legacy_projects(legacy.projects_root)

    assert len(manifests) == 1
    assert {manifest.schema_version for manifest in manifests} == {"1.0.0"}
    assert tree_measurement(legacy.projects_root) == before
