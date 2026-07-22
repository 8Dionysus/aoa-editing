"""Filesystem canonical store with a rebuildable SQLite index."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aoa_editing.config import Settings
from aoa_editing.domain.decisions import graph_events, initial_graph, proposed_graph
from aoa_editing.domain.invariants import validate_manifest, validate_project_version
from aoa_editing.domain.models import (
    Asset,
    DecisionLog,
    DecisionLogEntry,
    DerivedMediaManifest,
    EditorialBriefRevision,
    EditorialDecisionGraph,
    EditPatch,
    EvidenceRecord,
    FrameRange,
    FrameRate,
    Intent,
    InterchangeReport,
    JobReceipt,
    MotionCorrectionProposalSetV2,
    MotionCorrectionReviewV2,
    ProjectManifest,
    ProjectVersion,
    Provenance,
    QCReport,
    ReferenceMediaBindingV2,
    ReferenceWorkspaceRegistrationV2,
    StyleProfile,
    Timeline,
    Treatment,
    new_id,
)
from aoa_editing.infrastructure.media import ffprobe, sha256_file

ID_PATTERN = re.compile(r"^[a-z]+_[0-9a-f]{32}$")


class StoreError(RuntimeError):
    """Project store operation failed without committing partial canonical state."""


class ForbiddenSourceError(StoreError):
    """An evaluation-only source was offered as a project/render input."""


def _atomic_json(path: Path, model: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(mode="json") if isinstance(model, BaseModel) else model
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StoreError(f"cannot load {path}: {error}") from error


class ProjectStore:
    """Canonical file owner; the index can be deleted and reconstructed."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_roots()
        self._init_index()
        self.rebuild_index()

    def _project_root(self, project_id: str) -> Path:
        if not ID_PATTERN.fullmatch(project_id) or not project_id.startswith("project_"):
            raise StoreError(f"invalid project id: {project_id!r}")
        return self.settings.projects_root / project_id

    def _init_index(self) -> None:
        self.settings.state_root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.settings.state_root / "index.sqlite3") as database:
            database.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, updated_at TEXT NOT NULL,
                    current_version_id TEXT
                );
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                    media_kind TEXT NOT NULL, stored_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, parent_id TEXT,
                    created_at TEXT NOT NULL, message TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL,
                    status TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                """
            )

    def rebuild_index(self) -> dict[str, int]:
        """Reconstruct the disposable lookup index from canonical project files."""

        counts = {"projects": 0, "assets": 0, "versions": 0, "jobs": 0}
        with sqlite3.connect(self.settings.state_root / "index.sqlite3") as database:
            for path in sorted(self.settings.projects_root.glob("project_*/project.json")):
                project = _read_model(path, ProjectManifest)
                database.execute(
                    """INSERT OR REPLACE INTO projects(id, name, updated_at, current_version_id)
                    VALUES (?, ?, ?, ?)""",
                    (
                        project.id,
                        project.name,
                        project.updated_at.isoformat(),
                        project.current_version_id,
                    ),
                )
                counts["projects"] += 1
                root = path.parent
                for asset_path in sorted(root.glob("assets/asset_*/asset.json")):
                    asset = _read_model(asset_path, Asset)
                    database.execute(
                        """INSERT OR REPLACE INTO assets
                        (id, project_id, sha256, media_kind, stored_path)
                        VALUES (?, ?, ?, ?, ?)""",
                        (
                            asset.id,
                            project.id,
                            asset.sha256,
                            asset.media_kind.value,
                            asset.stored_path,
                        ),
                    )
                    counts["assets"] += 1
                for version_path in sorted(root.glob("versions/version_*.json")):
                    version = _read_model(version_path, ProjectVersion)
                    database.execute(
                        """INSERT OR REPLACE INTO versions
                        (id, project_id, parent_id, created_at, message)
                        VALUES (?, ?, ?, ?, ?)""",
                        (
                            version.id,
                            project.id,
                            version.parent_version_id,
                            version.created_at.isoformat(),
                            version.message,
                        ),
                    )
                    counts["versions"] += 1
                for job_path in sorted(root.glob("jobs/job_*.json")):
                    job = _read_model(job_path, JobReceipt)
                    database.execute(
                        """INSERT OR REPLACE INTO jobs
                        (id, project_id, kind, status, updated_at)
                        VALUES (?, ?, ?, ?, ?)""",
                        (
                            job.id,
                            project.id,
                            job.kind,
                            job.status.value,
                            (job.finished_at or job.started_at or job.created_at).isoformat(),
                        ),
                    )
                    counts["jobs"] += 1
        return counts

    def create_project(self, name: str, intent: Intent) -> ProjectManifest:
        manifest = ProjectManifest(name=name, intent=intent)
        brief = EditorialBriefRevision(
            project_id=manifest.id,
            revision=1,
            intent=intent,
            rationale="Initial editorial brief",
        )
        manifest = manifest.model_copy(update={"brief_revision_ids": [brief.id]})
        root = self._project_root(manifest.id)
        if root.exists():  # practically impossible, but do not merge identities
            raise StoreError(f"project already exists: {manifest.id}")
        for directory in (
            "assets",
            "evidence",
            "treatments",
            "patches",
            "versions",
            "renders",
            "qc",
            "exports",
            "jobs",
            "comparison",
            "decisions/graphs",
            "decisions/log",
            "briefs",
        ):
            (root / directory).mkdir(parents=True, exist_ok=False)
        _atomic_json(root / "briefs" / f"{brief.id}.json", brief)
        _atomic_json(root / "project.json", manifest)
        self._index_project(manifest)
        return manifest

    def list_projects(self) -> list[ProjectManifest]:
        projects: list[ProjectManifest] = []
        for path in sorted(self.settings.projects_root.glob("project_*/project.json")):
            projects.append(_read_model(path, ProjectManifest))
        return projects

    def load_project(self, project_id: str) -> ProjectManifest:
        return _read_model(self._project_root(project_id) / "project.json", ProjectManifest)

    def save_project(self, manifest: ProjectManifest) -> None:
        validate_manifest(manifest)
        _atomic_json(self._project_root(manifest.id) / "project.json", manifest)
        self._index_project(manifest)

    def revise_brief(
        self, project_id: str, intent: Intent, rationale: str
    ) -> EditorialBriefRevision:
        manifest = self.load_project(project_id)
        revision = EditorialBriefRevision(
            project_id=project_id,
            revision=len(manifest.brief_revision_ids) + 1,
            intent=intent,
            rationale=rationale,
        )
        path = self._project_root(project_id) / "briefs" / f"{revision.id}.json"
        _atomic_json(path, revision)
        updated = manifest.model_copy(
            update={
                "intent": intent,
                "brief_revision_ids": [*manifest.brief_revision_ids, revision.id],
                "updated_at": datetime.now(UTC),
            }
        )
        self.save_project(updated)
        return revision

    def list_brief_revisions(self, project_id: str) -> list[EditorialBriefRevision]:
        manifest = self.load_project(project_id)
        return [
            _read_model(
                self._project_root(project_id) / "briefs" / f"{brief_id}.json",
                EditorialBriefRevision,
            )
            for brief_id in manifest.brief_revision_ids
        ]

    def save_style_profile(self, profile: StyleProfile) -> None:
        root = self.settings.state_root / "style-profiles"
        path = root / f"{profile.id}.json"
        if path.exists():
            existing = _read_model(path, StyleProfile)
            if existing != profile:
                raise StoreError(f"immutable style profile already differs: {profile.id}")
            return
        _atomic_json(path, profile)

    def list_style_profiles(self) -> list[StyleProfile]:
        root = self.settings.state_root / "style-profiles"
        return [_read_model(path, StyleProfile) for path in sorted(root.glob("style_*.json"))]

    def _index_project(self, manifest: ProjectManifest) -> None:
        with sqlite3.connect(self.settings.state_root / "index.sqlite3") as database:
            database.execute(
                """INSERT INTO projects(id, name, updated_at, current_version_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                  updated_at=excluded.updated_at, current_version_id=excluded.current_version_id""",
                (
                    manifest.id,
                    manifest.name,
                    manifest.updated_at.isoformat(),
                    manifest.current_version_id,
                ),
            )

    def ingest_asset(self, project_id: str, source: Path) -> tuple[Asset, EvidenceRecord]:
        source = source.expanduser().resolve(strict=True)
        if not source.is_file() or not os.access(source, os.R_OK):
            raise StoreError(f"source is not a readable file: {source}")
        manifest = self.load_project(project_id)
        digest = sha256_file(source)
        forbidden = set(manifest.sealed_reference_hashes)
        forbidden.update(
            item.strip().lower()
            for item in os.environ.get("AOA_EDITING_FORBIDDEN_HASHES", "").split(",")
            if item.strip()
        )
        if digest in forbidden:
            raise ForbiddenSourceError(f"source hash is sealed for evaluation: {digest}")
        for asset_id in manifest.assets:
            existing = self.load_asset(project_id, asset_id)
            if existing.sha256 == digest:
                evidence = next(
                    (
                        item
                        for item in self.list_evidence(project_id)
                        if item.asset_id == existing.id and item.kind == "media.probe"
                    ),
                    None,
                )
                if evidence is None:
                    raise StoreError("indexed asset is missing mandatory probe evidence")
                return existing, evidence
        media_kind, metadata, payload, command = ffprobe(source)
        asset = Asset(
            sha256=digest,
            original_name=source.name,
            media_kind=media_kind,
            stored_path="",
            size_bytes=source.stat().st_size,
            metadata=metadata,
        )
        asset_dir = self._project_root(project_id) / "assets" / asset.id
        asset_dir.mkdir(parents=False, exist_ok=False)
        suffix = source.suffix.lower() or ".bin"
        destination = asset_dir / f"source{suffix}"
        partial = asset_dir / f".source{suffix}.partial"
        try:
            shutil.copyfile(source, partial)
            if sha256_file(partial) != digest:
                raise StoreError("copied source hash does not match input")
            os.replace(partial, destination)
            destination.chmod(0o444)
            stored = asset.model_copy(
                update={"stored_path": str(destination.relative_to(self._project_root(project_id)))}
            )
            _atomic_json(asset_dir / "asset.json", stored)
            evidence = EvidenceRecord(
                project_id=project_id,
                asset_id=stored.id,
                source_sha256=digest,
                kind="media.probe",
                payload=payload,
                **_asset_temporal_scope(stored),
                provenance=Provenance(
                    tool="ffprobe",
                    tool_version=_tool_version("ffprobe"),
                    command=command,
                    deterministic=True,
                ),
            )
            self.save_evidence(evidence)
        except Exception:
            if partial.exists():
                partial.unlink()
            shutil.rmtree(asset_dir, ignore_errors=True)
            raise
        updated = manifest.model_copy(
            update={
                "assets": [*manifest.assets, stored.id],
                "updated_at": datetime.now(UTC),
            }
        )
        self.save_project(updated)
        with sqlite3.connect(self.settings.state_root / "index.sqlite3") as database:
            database.execute(
                """INSERT INTO assets(id, project_id, sha256, media_kind, stored_path)
                VALUES (?, ?, ?, ?, ?)""",
                (stored.id, project_id, digest, stored.media_kind.value, stored.stored_path),
            )
        return stored, evidence

    def load_asset(self, project_id: str, asset_id: str) -> Asset:
        return _read_model(
            self._project_root(project_id) / "assets" / asset_id / "asset.json", Asset
        )

    def list_assets(self, project_id: str) -> list[Asset]:
        manifest = self.load_project(project_id)
        return [self.load_asset(project_id, asset_id) for asset_id in manifest.assets]

    def asset_source_path(self, project_id: str, asset_id: str) -> Path:
        asset = self.load_asset(project_id, asset_id)
        path = (self._project_root(project_id) / asset.stored_path).resolve()
        root = self._project_root(project_id).resolve()
        if not path.is_relative_to(root):
            raise StoreError("asset path escapes project root")
        if sha256_file(path) != asset.sha256:
            raise StoreError(f"immutable asset hash changed: {asset_id}")
        return path

    def save_derivatives(self, manifest: DerivedMediaManifest) -> None:
        _atomic_json(
            self._project_root(manifest.project_id)
            / "assets"
            / manifest.asset_id
            / "derivatives.json",
            manifest,
        )

    def load_derivatives(self, project_id: str, asset_id: str) -> DerivedMediaManifest:
        return _read_model(
            self._project_root(project_id) / "assets" / asset_id / "derivatives.json",
            DerivedMediaManifest,
        )

    def list_derivatives(self, project_id: str) -> list[DerivedMediaManifest]:
        root = self._project_root(project_id) / "assets"
        return [
            _read_model(path, DerivedMediaManifest)
            for path in sorted(root.glob("asset_*/derivatives.json"))
        ]

    def save_evidence(self, evidence: EvidenceRecord) -> None:
        if evidence.supersedes:
            known = {item.id: item for item in self.list_evidence(evidence.project_id)}
            for superseded_id in evidence.supersedes:
                superseded = known.get(superseded_id)
                if superseded is None:
                    raise StoreError(f"superseded evidence does not exist: {superseded_id}")
                if (
                    superseded.asset_id != evidence.asset_id
                    or superseded.kind != evidence.kind
                ):
                    raise StoreError("a correction may only supersede the same asset and kind")
        _atomic_json(
            self._project_root(evidence.project_id) / "evidence" / f"{evidence.id}.json",
            evidence,
        )

    def list_evidence(self, project_id: str) -> list[EvidenceRecord]:
        root = self._project_root(project_id) / "evidence"
        return [_read_model(path, EvidenceRecord) for path in sorted(root.glob("*.json"))]

    def list_effective_evidence(self, project_id: str) -> list[EvidenceRecord]:
        """Resolve supersession and prefer explicit human corrections per capability."""

        records = self.list_evidence(project_id)
        superseded = {item for record in records for item in record.supersedes}
        active = [record for record in records if record.id not in superseded]
        winners: dict[tuple[str, str, str], EvidenceRecord] = {}
        for record in active:
            failure_capability = (
                str(record.payload.get("failed_kind", ""))
                if record.kind == "analysis.failure"
                else ""
            )
            key = (record.asset_id, record.kind, failure_capability)
            current = winners.get(key)
            rank = (
                record.authority.value == "human-correction",
                record.provenance.generated_at,
                record.id,
            )
            if current is None:
                winners[key] = record
                continue
            current_rank = (
                current.authority.value == "human-correction",
                current.provenance.generated_at,
                current.id,
            )
            if rank > current_rank:
                winners[key] = record
        return sorted(winners.values(), key=lambda item: (item.asset_id, item.kind))

    def save_treatment(self, treatment: Treatment) -> Treatment:
        graph_path: Path | None = None
        if treatment.decision_graph_id is None:
            graph = proposed_graph(treatment)
            treatment = treatment.model_copy(update={"decision_graph_id": graph.id})
            graph_path = self.save_decision_graph(graph)
            for event in graph_events(graph, rationale=treatment.patch.rationale):
                self.append_decision_event(event)
        else:
            candidate = (
                self._project_root(treatment.project_id)
                / "decisions"
                / "graphs"
                / f"{treatment.decision_graph_id}.json"
            )
            if not candidate.exists():
                graph = proposed_graph(treatment, graph_id=treatment.decision_graph_id)
                graph_path = self.save_decision_graph(graph)
                for event in graph_events(graph, rationale=treatment.patch.rationale):
                    self.append_decision_event(event)
        _atomic_json(
            self._project_root(treatment.project_id)
            / "treatments"
            / f"{treatment.id}.json",
            treatment,
        )
        self.save_patch(treatment.patch)
        if graph_path is not None and not graph_path.is_file():  # pragma: no cover - fs guard
            raise StoreError("decision graph was not persisted")
        return treatment

    def load_treatment(self, project_id: str, treatment_id: str) -> Treatment:
        return _read_model(
            self._project_root(project_id) / "treatments" / f"{treatment_id}.json",
            Treatment,
        )

    def list_treatments(self, project_id: str) -> list[Treatment]:
        root = self._project_root(project_id) / "treatments"
        return [_read_model(path, Treatment) for path in sorted(root.glob("*.json"))]

    def save_patch(self, patch: EditPatch) -> None:
        _atomic_json(
            self._project_root(patch.project_id) / "patches" / f"{patch.id}.json", patch
        )

    def save_decision_graph(self, graph: EditorialDecisionGraph) -> Path:
        path = (
            self._project_root(graph.project_id)
            / "decisions"
            / "graphs"
            / f"{graph.id}.json"
        )
        if path.exists():
            existing = _read_model(path, EditorialDecisionGraph)
            if existing != graph:
                raise StoreError(f"immutable decision graph already differs: {graph.id}")
            return path
        _atomic_json(path, graph)
        return path

    def load_decision_graph(
        self, project_id: str, graph_id: str
    ) -> EditorialDecisionGraph:
        return _read_model(
            self._project_root(project_id) / "decisions" / "graphs" / f"{graph_id}.json",
            EditorialDecisionGraph,
        )

    def list_decision_graphs(self, project_id: str) -> list[EditorialDecisionGraph]:
        root = self._project_root(project_id) / "decisions" / "graphs"
        return [
            _read_model(path, EditorialDecisionGraph) for path in sorted(root.glob("*.json"))
        ]

    def append_decision_event(self, event: DecisionLogEntry) -> None:
        path = (
            self._project_root(event.project_id)
            / "decisions"
            / "log"
            / f"{event.id}.json"
        )
        if path.exists():
            existing = _read_model(path, DecisionLogEntry)
            if existing != event:
                raise StoreError(f"immutable decision event already differs: {event.id}")
            return
        _atomic_json(path, event)

    def load_decision_log(self, project_id: str) -> DecisionLog:
        root = self._project_root(project_id) / "decisions" / "log"
        entries = [_read_model(path, DecisionLogEntry) for path in sorted(root.glob("*.json"))]
        entries.sort(key=lambda item: (item.created_at, item.id))
        return DecisionLog(project_id=project_id, entries=entries)

    def save_version(self, version: ProjectVersion) -> ProjectManifest:
        manifest = self.load_project(version.project_id)
        validate_project_version(manifest, version)
        target = self._project_root(version.project_id) / "versions" / f"{version.id}.json"
        if target.exists():
            existing = _read_model(target, ProjectVersion)
            if existing != version:
                raise StoreError(
                    f"immutable version already exists with different content: {version.id}"
                )
            return manifest
        _atomic_json(target, version)
        updated = manifest.model_copy(
            update={
                "versions": [*manifest.versions, version.id],
                "current_version_id": version.id,
                "updated_at": datetime.now(UTC),
            }
        )
        self.save_project(updated)
        with sqlite3.connect(self.settings.state_root / "index.sqlite3") as database:
            database.execute(
                """INSERT INTO versions(id, project_id, parent_id, created_at, message)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    version.id,
                    version.project_id,
                    version.parent_version_id,
                    version.created_at.isoformat(),
                    version.message,
                ),
            )
        return updated

    def create_initial_version(self, project_id: str, timeline: Timeline) -> ProjectVersion:
        manifest = self.load_project(project_id)
        if manifest.versions:
            raise StoreError("project already has an initial version")
        version_id = new_id("version")
        graph = initial_graph(project_id, version_id)
        version = ProjectVersion(
            id=version_id,
            project_id=project_id,
            message="Initial timeline",
            timeline=timeline,
            decision_graph_id=graph.id,
        )
        self.save_decision_graph(graph)
        for event in graph_events(graph, rationale="Create an empty initial timeline"):
            self.append_decision_event(event)
        self.save_version(version)
        return version

    def load_version(self, project_id: str, version_id: str | None = None) -> ProjectVersion:
        manifest = self.load_project(project_id)
        selected = version_id or manifest.current_version_id
        if selected is None:
            raise StoreError("project has no versions")
        return _read_model(
            self._project_root(project_id) / "versions" / f"{selected}.json",
            ProjectVersion,
        )

    def list_versions(self, project_id: str) -> list[ProjectVersion]:
        manifest = self.load_project(project_id)
        return [self.load_version(project_id, version_id) for version_id in manifest.versions]

    def save_job(self, receipt: JobReceipt) -> None:
        _atomic_json(
            self._project_root(receipt.project_id) / "jobs" / f"{receipt.id}.json", receipt
        )
        with sqlite3.connect(self.settings.state_root / "index.sqlite3") as database:
            database.execute(
                """INSERT INTO jobs(id, project_id, kind, status, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                  updated_at=excluded.updated_at""",
                (
                    receipt.id,
                    receipt.project_id,
                    receipt.kind,
                    receipt.status.value,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def list_jobs(self, project_id: str) -> list[JobReceipt]:
        root = self._project_root(project_id) / "jobs"
        return [_read_model(path, JobReceipt) for path in sorted(root.glob("*.json"))]

    def save_qc(self, report: QCReport) -> None:
        _atomic_json(
            self._project_root(report.project_id) / "qc" / f"{report.id}.json", report
        )

    def list_qc(self, project_id: str) -> list[QCReport]:
        root = self._project_root(project_id) / "qc"
        return [_read_model(path, QCReport) for path in sorted(root.glob("*.json"))]

    def save_interchange(self, report: InterchangeReport) -> None:
        _atomic_json(
            self._project_root(report.project_id)
            / "exports"
            / f"{report.id}.report.json",
            report,
        )

    def list_interchange(self, project_id: str) -> list[InterchangeReport]:
        root = self._project_root(project_id) / "exports"
        return [
            _read_model(path, InterchangeReport)
            for path in sorted(root.glob("*.report.json"))
        ]

    def save_reference_workspace(
        self,
        workspace: ReferenceWorkspaceRegistrationV2,
    ) -> ReferenceWorkspaceRegistrationV2:
        """Persist an immutable project-local reference workbench registration."""

        self.load_project(workspace.project_id)
        self.load_version(workspace.project_id, workspace.base_version_id)
        self.load_asset(workspace.project_id, workspace.asset_id)
        path = (
            self._project_root(workspace.project_id)
            / "comparison"
            / "workspaces"
            / f"{workspace.id}.json"
        )
        if path.exists():
            existing = _read_model(path, ReferenceWorkspaceRegistrationV2)
            if existing != workspace:
                raise StoreError(
                    f"immutable reference workspace already differs: {workspace.id}"
                )
            return existing
        _atomic_json(path, workspace)
        return workspace

    def load_reference_workspace(
        self,
        project_id: str,
        workspace_id: str,
    ) -> ReferenceWorkspaceRegistrationV2:
        _require_id(workspace_id, "referenceworkspace")
        return _read_model(
            self._project_root(project_id)
            / "comparison"
            / "workspaces"
            / f"{workspace_id}.json",
            ReferenceWorkspaceRegistrationV2,
        )

    def list_reference_workspaces(
        self,
        project_id: str,
    ) -> list[ReferenceWorkspaceRegistrationV2]:
        root = self._project_root(project_id) / "comparison" / "workspaces"
        return [
            _read_model(path, ReferenceWorkspaceRegistrationV2)
            for path in sorted(root.glob("referenceworkspace_*.json"))
        ]

    def save_motion_correction_proposal(
        self,
        proposal: MotionCorrectionProposalSetV2,
    ) -> MotionCorrectionProposalSetV2:
        workspace = self.load_reference_workspace(
            proposal.project_id,
            proposal.workspace_id,
        )
        self.load_version(proposal.project_id, proposal.base_version_id)
        if proposal.target_effect_path != workspace.target_effect_path:
            raise StoreError("motion proposal targets a different workspace effect")
        path = (
            self._project_root(proposal.project_id)
            / "comparison"
            / "corrections"
            / "proposals"
            / f"{proposal.id}.json"
        )
        if path.exists():
            existing = _read_model(path, MotionCorrectionProposalSetV2)
            if existing != proposal:
                raise StoreError(f"immutable motion proposal already differs: {proposal.id}")
            return existing
        _atomic_json(path, proposal)
        return proposal

    def load_motion_correction_proposal(
        self,
        project_id: str,
        proposal_id: str,
    ) -> MotionCorrectionProposalSetV2:
        _require_id(proposal_id, "motionproposal")
        return _read_model(
            self._project_root(project_id)
            / "comparison"
            / "corrections"
            / "proposals"
            / f"{proposal_id}.json",
            MotionCorrectionProposalSetV2,
        )

    def list_motion_correction_proposals(
        self,
        project_id: str,
        *,
        workspace_id: str | None = None,
    ) -> list[MotionCorrectionProposalSetV2]:
        root = (
            self._project_root(project_id)
            / "comparison"
            / "corrections"
            / "proposals"
        )
        proposals = [
            _read_model(path, MotionCorrectionProposalSetV2)
            for path in sorted(root.glob("motionproposal_*.json"))
        ]
        if workspace_id is None:
            return proposals
        return [item for item in proposals if item.workspace_id == workspace_id]

    def save_motion_correction_review(
        self,
        review: MotionCorrectionReviewV2,
    ) -> MotionCorrectionReviewV2:
        proposal = self.load_motion_correction_proposal(
            review.project_id,
            review.proposal_id,
        )
        if (
            review.workspace_id != proposal.workspace_id
            or review.base_version_id != proposal.base_version_id
        ):
            raise StoreError("motion review does not match its immutable proposal")
        path = (
            self._project_root(review.project_id)
            / "comparison"
            / "corrections"
            / "reviews"
            / f"{review.id}.json"
        )
        if path.exists():
            existing = _read_model(path, MotionCorrectionReviewV2)
            if existing != review:
                raise StoreError(f"immutable motion review already differs: {review.id}")
            return existing
        _atomic_json(path, review)
        return review

    def list_motion_correction_reviews(
        self,
        project_id: str,
        *,
        workspace_id: str | None = None,
    ) -> list[MotionCorrectionReviewV2]:
        root = (
            self._project_root(project_id)
            / "comparison"
            / "corrections"
            / "reviews"
        )
        reviews = [
            _read_model(path, MotionCorrectionReviewV2)
            for path in sorted(root.glob("motionreview_*.json"))
        ]
        if workspace_id is None:
            return reviews
        return [item for item in reviews if item.workspace_id == workspace_id]

    def bind_reference_media(
        self,
        reference_sha256: str,
        source: Path,
    ) -> ReferenceMediaBindingV2:
        """Write only an untracked local binding; the hash remains canonical identity."""

        source = source.expanduser().resolve(strict=True)
        if not source.is_file() or not os.access(source, os.R_OK):
            raise StoreError(f"reference binding is not a readable file: {source}")
        actual = sha256_file(source)
        if actual != reference_sha256:
            raise StoreError(
                f"reference binding hash mismatch: expected {reference_sha256}, got {actual}"
            )
        binding = ReferenceMediaBindingV2(
            reference_sha256=reference_sha256,
            physical_path=str(source),
            size_bytes=source.stat().st_size,
            provenance=Provenance(
                tool="aoa-editing",
                tool_version="reference-workspace-v2",
                parameters={"operation": "bind-sealed-reference-media"},
                deterministic=True,
            ),
        )
        _atomic_json(
            self.settings.state_root
            / "reference-media-bindings"
            / f"{reference_sha256}.json",
            binding,
        )
        return binding

    def resolve_reference_media(self, reference_sha256: str) -> Path:
        binding = _read_model(
            self.settings.state_root
            / "reference-media-bindings"
            / f"{reference_sha256}.json",
            ReferenceMediaBindingV2,
        )
        source = Path(binding.physical_path).expanduser().resolve(strict=True)
        if not source.is_file() or sha256_file(source) != reference_sha256:
            raise StoreError("sealed reference binding is missing or its hash changed")
        return source

    def project_path(self, project_id: str) -> Path:
        return self._project_root(project_id)


def _require_id(value: str, prefix: str) -> None:
    if not ID_PATTERN.fullmatch(value) or not value.startswith(f"{prefix}_"):
        raise StoreError(f"invalid {prefix} id: {value!r}")


def _tool_version(command: str) -> str:
    import subprocess

    result = subprocess.run(
        [command, "-version"], check=False, capture_output=True, text=True, timeout=10
    )
    line = (result.stdout or result.stderr).splitlines()
    return line[0] if line else "unknown"


def _asset_temporal_scope(asset: Asset) -> dict[str, Any]:
    duration = asset.metadata.duration_seconds
    if duration is None or duration <= 0:
        return {}
    if asset.metadata.has_video and asset.metadata.frame_rate is not None:
        time_base = asset.metadata.frame_rate
        units = round(duration * time_base.fps)
    elif asset.metadata.has_audio and asset.metadata.sample_rate is not None:
        time_base = FrameRate(numerator=asset.metadata.sample_rate)
        units = round(duration * asset.metadata.sample_rate)
    else:
        return {}
    return {"time_base": time_base, "ranges": [FrameRange(start=0, duration=max(1, units))]}
