"""Measured, non-destructive relocation into the repo-local operational home."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    CheckResult,
    ProjectManifest,
    StorageRelocationEntry,
    StorageRelocationReport,
)
from aoa_editing.infrastructure.store import ProjectStore


class RelocationError(RuntimeError):
    """Legacy state cannot be copied and verified without ambiguity."""


@dataclass(frozen=True, slots=True)
class TreeMeasurement:
    file_count: int
    logical_bytes: int
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class _CopyUnit:
    id: str
    classification: str
    source: Path
    destination: Path
    exclude: Callable[[Path], bool] | None = None
    excluded_paths: tuple[str, ...] = ()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_measurement(
    root: Path,
    *,
    exclude: Callable[[Path], bool] | None = None,
) -> TreeMeasurement:
    """Hash file content, size, and relative identity in deterministic order."""

    selected = root.expanduser().resolve(strict=True)
    if selected.is_symlink():
        raise RelocationError(f"filesystem links are not admitted to relocation: {selected}")
    if selected.is_file():
        paths = [selected]
        relative_root = selected.parent
    else:
        relative_root = selected
        paths = []
        for candidate in selected.rglob("*"):
            relative = candidate.relative_to(selected)
            if exclude is not None and exclude(relative):
                continue
            if candidate.is_symlink():
                raise RelocationError(
                    f"filesystem links are not admitted to relocation: {candidate}"
                )
            if candidate.is_file():
                paths.append(candidate)
        paths.sort(key=lambda item: item.relative_to(relative_root).as_posix())

    digest = hashlib.sha256()
    logical_bytes = 0
    for path in paths:
        size = path.stat().st_size
        relative_path = path.relative_to(relative_root).as_posix()
        digest.update(f"{_sha256_file(path)}  {size}  {relative_path}\n".encode())
        logical_bytes += size
    return TreeMeasurement(
        file_count=len(paths),
        logical_bytes=logical_bytes,
        tree_sha256=digest.hexdigest(),
    )


def inspect_legacy_projects(projects_root: Path) -> list[ProjectManifest]:
    """Validate old project manifests without opening or rebuilding their index."""

    selected = projects_root.expanduser().resolve(strict=True)
    manifests: list[ProjectManifest] = []
    for path in sorted(selected.glob("project_*/project.json")):
        manifest = ProjectManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if path.parent.name != manifest.id:
            raise RelocationError(
                f"project directory identity differs from manifest: {path.parent.name}"
            )
        manifests.append(manifest)
    if not manifests:
        raise RelocationError(f"legacy project root contains no manifests: {selected}")
    return manifests


def _exclude_readiness_derived(relative: Path) -> bool:
    parts = relative.parts
    nested_runtime = any(
        parts[index : index + 2] == ("clean-bootstrap", "runtime")
        for index in range(max(0, len(parts) - 1))
    )
    return nested_runtime or "chromium-profile" in parts


def _copy_ignore_readiness(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    if Path(directory).name == "clean-bootstrap" and "runtime" in names:
        ignored.add("runtime")
    if "chromium-profile" in names:
        ignored.add("chromium-profile")
    return ignored


def _index_source(data_root: Path) -> Path:
    direct = data_root / "index.sqlite3"
    if direct.is_file():
        return direct
    nested = data_root / "state" / "index.sqlite3"
    if nested.is_file():
        return nested
    raise RelocationError(f"legacy SQLite index is missing beneath {data_root}")


def _check(check_id: str, passed: bool, summary: str, measured: dict[str, object]) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if passed else "fail",
        summary=summary,
        measured=measured,
    )


def _copy_units(data_root: Path, destination: Settings) -> list[_CopyUnit]:
    return [
        _CopyUnit(
            id="projects",
            classification="canonical_authoritative_and_immutable_ingest",
            source=data_root / "projects",
            destination=destination.projects_root,
        ),
        _CopyUnit(
            id="evals-generic",
            classification="historical_proof",
            source=data_root / "evals" / "generic",
            destination=destination.evals_root / "generic",
        ),
        _CopyUnit(
            id="evals-reference",
            classification="historical_proof",
            source=data_root / "evals" / "reference",
            destination=destination.evals_root / "reference",
        ),
        _CopyUnit(
            id="evals-readiness",
            classification="historical_proof",
            source=data_root / "evals" / "readiness",
            destination=destination.evals_root / "readiness",
            exclude=_exclude_readiness_derived,
            excluded_paths=(
                "*/clean-bootstrap/runtime/**",
                "*/ui-smoke/*/chromium-profile/**",
            ),
        ),
    ]


def _planned_entry(unit: _CopyUnit, measurement: TreeMeasurement) -> StorageRelocationEntry:
    return StorageRelocationEntry(
        id=unit.id,
        classification=unit.classification,
        method="copy_then_verify",
        source_path=str(unit.source),
        destination_path=str(unit.destination),
        file_count=measurement.file_count,
        logical_bytes=measurement.logical_bytes,
        source_tree_sha256=measurement.tree_sha256,
        excluded_paths=list(unit.excluded_paths),
        status="planned",
    )


def _atomic_report(path: Path, report: StorageRelocationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(report.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def migrate_legacy_storage(
    *,
    legacy_data_root: Path,
    destination: Settings,
    inventory_sha256: str,
    dry_run: bool,
) -> StorageRelocationReport:
    """Plan or perform a verified copy while leaving every legacy byte in place."""

    data_root = legacy_data_root.expanduser().resolve(strict=True)
    if len(inventory_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in inventory_sha256
    ):
        raise RelocationError("inventory SHA-256 must be 64 lowercase hexadecimal characters")
    if destination.layout.compatibility_overrides:
        raise RelocationError("destination cannot use legacy compatibility root overrides")
    if destination.data_root != destination.layout.var_root:
        raise RelocationError("destination is not a unified repo-local home")

    units = _copy_units(data_root, destination)
    source_projects = inspect_legacy_projects(data_root / "projects")
    source_measurements: dict[str, TreeMeasurement] = {}
    for unit in units:
        source_measurements[unit.id] = tree_measurement(unit.source, exclude=unit.exclude)
    index_source = _index_source(data_root)
    index_measurement = tree_measurement(index_source)
    planned = [
        *[_planned_entry(unit, source_measurements[unit.id]) for unit in units],
        StorageRelocationEntry(
            id="sqlite-index",
            classification="derived_reproducible",
            method="rebuild_from_canonical",
            source_path=str(index_source),
            destination_path=str(destination.state_root / "index.sqlite3"),
            file_count=index_measurement.file_count,
            logical_bytes=index_measurement.logical_bytes,
            source_tree_sha256=index_measurement.tree_sha256,
            status="planned",
        ),
    ]
    preflight_checks = [
        _check(
            "source-projects-readable",
            bool(source_projects),
            "legacy project manifests validate without opening the old SQLite index",
            {"project_ids": [project.id for project in source_projects]},
        ),
        _check(
            "destination-owner-layout",
            destination.data_root == destination.layout.var_root,
            "destination derives all product paths from one operational home",
            {"home": str(destination.editing_home)},
        ),
        _check(
            "legacy-cleanup-not-authorized",
            True,
            "relocation does not authorize source cleanup",
            {"authorized": False},
        ),
    ]
    if dry_run:
        return StorageRelocationReport(
            source_inventory_sha256=inventory_sha256,
            source_data_root=str(data_root),
            destination_home=str(destination.editing_home),
            dry_run=True,
            entries=planned,
            checks=preflight_checks,
            cutover_ready=False,
            overall="pass",
        )

    if destination.layout.var_root.exists():
        raise RelocationError(
            f"destination var root already exists; refusing merge: {destination.layout.var_root}"
        )
    stage = destination.editing_home / ".aoa-editing-migration-staging"
    if stage.exists():
        raise RelocationError(f"stale migration staging root exists: {stage}")
    destination.editing_home.mkdir(parents=True, exist_ok=True)
    stage.mkdir()

    verified_entries: list[StorageRelocationEntry] = []
    try:
        for unit in units:
            relative_destination = unit.destination.relative_to(destination.layout.var_root)
            staged_destination = stage / relative_destination
            staged_destination.parent.mkdir(parents=True, exist_ok=True)
            ignore = _copy_ignore_readiness if unit.id == "evals-readiness" else None
            shutil.copytree(
                unit.source,
                staged_destination,
                copy_function=shutil.copy2,
                symlinks=False,
                ignore=ignore,
            )
            destination_measurement = tree_measurement(staged_destination)
            source_measurement = source_measurements[unit.id]
            if destination_measurement != source_measurement:
                raise RelocationError(
                    f"copy verification failed for {unit.id}: "
                    f"{source_measurement} != {destination_measurement}"
                )
            verified_entries.append(
                _planned_entry(unit, source_measurement).model_copy(
                    update={
                        "destination_tree_sha256": destination_measurement.tree_sha256,
                        "status": "verified",
                    }
                )
            )
        for relative in ("artifacts", "cache", "tmp", "state"):
            (stage / relative).mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination.layout.var_root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    relocated_store = ProjectStore(destination)
    destination_projects = relocated_store.list_projects()
    source_by_id = {project.id: project for project in source_projects}
    destination_by_id = {project.id: project for project in destination_projects}
    semantic_match = source_by_id == destination_by_id
    rebuilt_index = tree_measurement(destination.state_root / "index.sqlite3")
    verified_entries.append(
        planned[-1].model_copy(
            update={
                "destination_tree_sha256": rebuilt_index.tree_sha256,
                "status": "rebuilt",
            }
        )
    )
    runtime_copy_absent = not any(
        path.is_dir() for path in destination.evals_root.glob("readiness/*/clean-bootstrap/runtime")
    )
    checks = [
        *preflight_checks,
        _check(
            "copy-tree-digests",
            all(
                entry.source_tree_sha256 == entry.destination_tree_sha256
                for entry in verified_entries
                if entry.method == "copy_then_verify"
            ),
            "every copied tree matches its measured source",
            {
                entry.id: entry.destination_tree_sha256
                for entry in verified_entries
                if entry.method == "copy_then_verify"
            },
        ),
        _check(
            "project-semantic-identity",
            semantic_match,
            "all project manifests retain their typed meaning after relocation",
            {
                "source_ids": sorted(source_by_id),
                "destination_ids": sorted(destination_by_id),
            },
        ),
        _check(
            "sqlite-rebuilt",
            rebuilt_index.file_count == 1,
            "SQLite index was rebuilt from relocated canonical documents",
            {"tree_sha256": rebuilt_index.tree_sha256},
        ),
        _check(
            "readiness-runtime-rebuilt-not-copied",
            runtime_copy_absent,
            "nested clean-bootstrap runtimes were excluded from historical proof copies",
            {"runtime_copy_present": not runtime_copy_absent},
        ),
        _check(
            "filesystem-links-absent",
            True,
            "relocation created no filesystem links",
            {"created": False},
        ),
    ]
    overall: Literal["pass", "fail"] = (
        "pass" if all(check.status == "pass" for check in checks) else "fail"
    )
    report = StorageRelocationReport(
        source_inventory_sha256=inventory_sha256,
        source_data_root=str(data_root),
        destination_home=str(destination.editing_home),
        dry_run=False,
        entries=verified_entries,
        checks=checks,
        cutover_ready=overall == "pass",
        overall=overall,
    )
    _atomic_report(destination.artifacts_root / "migrations" / f"{report.id}.json", report)
    return report
