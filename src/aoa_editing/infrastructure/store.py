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
    EditorialStructureProposal,
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
    ReferenceReconstructionProposal,
    ReferenceWorkspaceRegistrationV2,
    StyleProfile,
    Timeline,
    Treatment,
    VideoAnatomy,
    VideoAudioTimelineEvidence,
    VideoFrameSampleManifest,
    VideoMotionEvidence,
    VideoProposalAcceptanceReceipt,
    VideoProposalReview,
    VideoSamplingPlan,
    VideoStructureEvidence,
    VideoVisualObservation,
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

    def register_evaluation_reference_asset(
        self,
        project_id: str,
        reference_sha256: str,
        *,
        readiness_revision: str,
    ) -> tuple[Asset, EvidenceRecord]:
        """Register a sealed external binding for post-gate analysis without copying media."""

        if not re.fullmatch(r"[0-9a-f]{40}", readiness_revision):
            raise StoreError("evaluation reference registration requires a full readiness revision")
        # Local import keeps ordinary storage independent from the eval module,
        # while this exceptional external-media route remains impossible before
        # a live, revision-matched, clean-worktree readiness decision.
        from aoa_editing.evals.gate import require_readiness

        readiness = require_readiness(self.settings)
        if readiness.get("git_revision") != readiness_revision:
            raise ForbiddenSourceError("evaluation reference readiness revision differs")
        manifest = self.load_project(project_id)
        if reference_sha256 not in manifest.sealed_reference_hashes:
            raise ForbiddenSourceError(
                "evaluation reference hash must be sealed before registration"
            )
        for asset_id in manifest.assets:
            existing = self.load_asset(project_id, asset_id)
            if existing.sha256 == reference_sha256:
                evidence = next(
                    item
                    for item in self.list_evidence(project_id)
                    if item.asset_id == asset_id and item.kind == "media.probe"
                )
                return existing, evidence
        source = self.resolve_reference_media(reference_sha256)
        media_kind, metadata, payload, command = ffprobe(source)
        asset = Asset(
            sha256=reference_sha256,
            original_name="sealed-reference-video",
            media_kind=media_kind,
            stored_path=f"@evaluation-reference/{reference_sha256}",
            size_bytes=source.stat().st_size,
            metadata=metadata,
        )
        asset_dir = self._project_root(project_id) / "assets" / asset.id
        asset_dir.mkdir(parents=False, exist_ok=False)
        _atomic_json(asset_dir / "asset.json", asset)
        probe_payload = dict(payload)
        probe_payload["evaluation_reference"] = {
            "role": "analysis-and-comparison-only",
            "media_copied_into_project": False,
            "allowed_as_render_input": False,
            "readiness_revision": readiness_revision,
        }
        evidence = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=reference_sha256,
            kind="media.probe",
            payload=probe_payload,
            **_asset_temporal_scope(asset),
            provenance=Provenance(
                tool="ffprobe",
                tool_version=_tool_version("ffprobe"),
                command=[
                    item if item != str(source) else "<sealed-reference-binding>"
                    for item in command
                ],
                parameters={
                    "evaluation_only": True,
                    "readiness_revision": readiness_revision,
                    "media_copied_into_project": False,
                },
                deterministic=True,
            ),
        )
        self.save_evidence(evidence)
        self.save_project(
            manifest.model_copy(
                update={
                    "assets": [*manifest.assets, asset.id],
                    "updated_at": datetime.now(UTC),
                }
            )
        )
        with sqlite3.connect(self.settings.state_root / "index.sqlite3") as database:
            database.execute(
                """INSERT INTO assets(id, project_id, sha256, media_kind, stored_path)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    asset.id,
                    project_id,
                    reference_sha256,
                    asset.media_kind.value,
                    asset.stored_path,
                ),
            )
        return asset, evidence

    def load_asset(self, project_id: str, asset_id: str) -> Asset:
        return _read_model(
            self._project_root(project_id) / "assets" / asset_id / "asset.json", Asset
        )

    def list_assets(self, project_id: str) -> list[Asset]:
        manifest = self.load_project(project_id)
        return [self.load_asset(project_id, asset_id) for asset_id in manifest.assets]

    def asset_source_path(self, project_id: str, asset_id: str) -> Path:
        asset = self.load_asset(project_id, asset_id)
        if asset.stored_path == f"@evaluation-reference/{asset.sha256}":
            project = self.load_project(project_id)
            if asset.sha256 not in project.sealed_reference_hashes:
                raise ForbiddenSourceError("evaluation reference binding is no longer sealed")
            return self.resolve_reference_media(asset.sha256)
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
                if superseded.asset_id != evidence.asset_id or superseded.kind != evidence.kind:
                    raise StoreError("a correction may only supersede the same asset and kind")
        _atomic_json(
            self._project_root(evidence.project_id) / "evidence" / f"{evidence.id}.json",
            evidence,
        )

    def list_evidence(self, project_id: str) -> list[EvidenceRecord]:
        root = self._project_root(project_id) / "evidence"
        return [_read_model(path, EvidenceRecord) for path in sorted(root.glob("*.json"))]

    def video_anatomy_path(self, project_id: str, plan_id: str) -> Path:
        """Return the project-owned artifact root for one immutable sampling plan."""

        self.load_project(project_id)
        _require_id(plan_id, "samplingplan")
        root = self._project_root(project_id) / "analysis" / "video-anatomy" / plan_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def save_video_frame_timestamps(
        self,
        project_id: str,
        asset_id: str,
        source_sha256: str,
        payload: dict[str, object],
    ) -> Path:
        """Persist the exact decoded-order PTS map for a VFR source."""

        asset = self.load_asset(project_id, asset_id)
        if asset.sha256 != source_sha256:
            raise StoreError("frame timestamp map source identity differs from the asset")
        path = (
            self._project_root(project_id)
            / "analysis"
            / "video-timing"
            / f"{asset_id}-{source_sha256[:16]}.json"
        )
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise StoreError("immutable frame timestamp map already differs")
            return path
        _atomic_json(path, payload)
        return path

    def _save_video_anatomy_model(
        self,
        project_id: str,
        plan_id: str,
        filename: str,
        model: BaseModel,
    ) -> Path:
        path = self.video_anatomy_path(project_id, plan_id) / filename
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            incoming = model.model_dump(mode="json")
            if existing != incoming:
                raise StoreError(f"immutable Video Anatomy artifact already differs: {path}")
            return path
        _atomic_json(path, model)
        return path

    def save_video_sampling_plan(self, plan: VideoSamplingPlan) -> Path:
        if plan.project_id != self.load_project(plan.project_id).id:
            raise StoreError("sampling plan belongs to an unknown project")
        asset = self.load_asset(plan.project_id, plan.asset_id)
        if asset.sha256 != plan.source_sha256:
            raise StoreError("sampling plan source identity differs from the asset")
        return self._save_video_anatomy_model(plan.project_id, plan.id, "sampling-plan.json", plan)

    def load_video_sampling_plan(self, project_id: str, plan_id: str) -> VideoSamplingPlan:
        return _read_model(
            self.video_anatomy_path(project_id, plan_id) / "sampling-plan.json",
            VideoSamplingPlan,
        )

    def save_video_frame_manifest(self, manifest: VideoFrameSampleManifest) -> Path:
        plan = self.load_video_sampling_plan(manifest.project_id, manifest.plan_id)
        if (
            manifest.asset_id != plan.asset_id
            or manifest.source_sha256 != plan.source_sha256
            or manifest.plan_sha256 != plan.plan_sha256
        ):
            raise StoreError("frame manifest differs from its persisted sampling plan")
        return self._save_video_anatomy_model(
            manifest.project_id, manifest.plan_id, "frame-manifest.json", manifest
        )

    def load_video_frame_manifest(self, project_id: str, plan_id: str) -> VideoFrameSampleManifest:
        return _read_model(
            self.video_anatomy_path(project_id, plan_id) / "frame-manifest.json",
            VideoFrameSampleManifest,
        )

    def save_video_structure(self, structure: VideoStructureEvidence) -> Path:
        plan = self.load_video_sampling_plan(structure.project_id, structure.plan_id)
        if (
            structure.asset_id != plan.asset_id
            or structure.source_sha256 != plan.source_sha256
            or structure.plan_sha256 != plan.plan_sha256
        ):
            raise StoreError("structure evidence differs from its persisted sampling plan")
        return self._save_video_anatomy_model(
            structure.project_id, structure.plan_id, "structure.json", structure
        )

    def load_video_structure(self, project_id: str, plan_id: str) -> VideoStructureEvidence:
        return _read_model(
            self.video_anatomy_path(project_id, plan_id) / "structure.json",
            VideoStructureEvidence,
        )

    def save_video_audio_timeline(self, plan_id: str, timeline: VideoAudioTimelineEvidence) -> Path:
        plan = self.load_video_sampling_plan(timeline.project_id, plan_id)
        if timeline.asset_id != plan.asset_id or timeline.source_sha256 != plan.source_sha256:
            raise StoreError("audio timeline differs from its sampling plan source")
        return self._save_video_anatomy_model(
            timeline.project_id, plan_id, "audio-timeline.json", timeline
        )

    def load_video_audio_timeline(
        self, project_id: str, plan_id: str
    ) -> VideoAudioTimelineEvidence:
        return _read_model(
            self.video_anatomy_path(project_id, plan_id) / "audio-timeline.json",
            VideoAudioTimelineEvidence,
        )

    def save_video_visual_observation(
        self, plan_id: str, observation: VideoVisualObservation
    ) -> Path:
        plan = self.load_video_sampling_plan(observation.project_id, plan_id)
        if observation.asset_id != plan.asset_id or observation.source_sha256 != plan.source_sha256:
            raise StoreError("visual observation differs from its sampling plan source")
        return self._save_video_anatomy_model(
            observation.project_id,
            plan_id,
            f"visual/{observation.id}.json",
            observation,
        )

    def list_video_visual_observations(
        self, project_id: str, plan_id: str
    ) -> list[VideoVisualObservation]:
        root = self.video_anatomy_path(project_id, plan_id) / "visual"
        return [_read_model(path, VideoVisualObservation) for path in sorted(root.glob("*.json"))]

    def save_video_motion_evidence(self, plan_id: str, motion: VideoMotionEvidence) -> Path:
        plan = self.load_video_sampling_plan(motion.project_id, plan_id)
        if motion.asset_id != plan.asset_id or motion.source_sha256 != plan.source_sha256:
            raise StoreError("motion evidence differs from its sampling plan source")
        return self._save_video_anatomy_model(
            motion.project_id,
            plan_id,
            f"motion/{motion.id}.json",
            motion,
        )

    def list_video_motion_evidence(
        self, project_id: str, plan_id: str
    ) -> list[VideoMotionEvidence]:
        root = self.video_anatomy_path(project_id, plan_id) / "motion"
        return [_read_model(path, VideoMotionEvidence) for path in sorted(root.glob("*.json"))]

    def save_video_anatomy(self, anatomy: VideoAnatomy) -> Path:
        plan = self.load_video_sampling_plan(anatomy.project_id, anatomy.plan.id)
        if plan != anatomy.plan:
            raise StoreError("Video Anatomy embeds a different sampling plan")
        path = self._save_video_anatomy_model(
            anatomy.project_id,
            anatomy.plan.id,
            f"anatomies/{anatomy.id}.json",
            anatomy,
        )
        _atomic_json(
            self.video_anatomy_path(anatomy.project_id, anatomy.plan.id) / "latest-anatomy.json",
            {
                "anatomy_id": anatomy.id,
                "anatomy_sha256": anatomy.anatomy_sha256,
                "artifact_path": str(path.relative_to(self._project_root(anatomy.project_id))),
            },
        )
        return path

    def load_video_anatomy(self, project_id: str, plan_id: str) -> VideoAnatomy:
        root = self.video_anatomy_path(project_id, plan_id)
        latest_path = root / "latest-anatomy.json"
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            anatomy_id = str(latest["anatomy_id"])
        except (OSError, ValueError, KeyError) as error:
            raise StoreError(f"cannot resolve latest Video Anatomy: {latest_path}") from error
        return _read_model(root / "anatomies" / f"{anatomy_id}.json", VideoAnatomy)

    def list_video_anatomies(self, project_id: str) -> list[VideoAnatomy]:
        root = self._project_root(project_id) / "analysis" / "video-anatomy"
        results: list[VideoAnatomy] = []
        for latest_path in sorted(root.glob("samplingplan_*/latest-anatomy.json")):
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                anatomy_id = str(latest["anatomy_id"])
                results.append(
                    _read_model(
                        latest_path.parent / "anatomies" / f"{anatomy_id}.json",
                        VideoAnatomy,
                    )
                )
            except (OSError, ValueError, KeyError) as error:
                raise StoreError(f"cannot enumerate Video Anatomy: {latest_path}") from error
        return results

    def _save_project_immutable_model(
        self,
        project_id: str,
        relative_path: Path,
        model: BaseModel,
    ) -> Path:
        self.load_project(project_id)
        path = self._project_root(project_id) / relative_path
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != model.model_dump(mode="json"):
                raise StoreError(f"immutable project artifact already differs: {path}")
            return path
        _atomic_json(path, model)
        return path

    def save_editorial_structure_proposal(self, proposal: EditorialStructureProposal) -> Path:
        _require_id(proposal.id, "editorialproposal")
        return self._save_project_immutable_model(
            proposal.project_id,
            Path("proposals/video-anatomy/editorial") / f"{proposal.id}.json",
            proposal,
        )

    def load_editorial_structure_proposal(
        self, project_id: str, proposal_id: str
    ) -> EditorialStructureProposal:
        _require_id(proposal_id, "editorialproposal")
        return _read_model(
            self._project_root(project_id)
            / "proposals/video-anatomy/editorial"
            / f"{proposal_id}.json",
            EditorialStructureProposal,
        )

    def list_editorial_structure_proposals(
        self, project_id: str
    ) -> list[EditorialStructureProposal]:
        root = self._project_root(project_id) / "proposals/video-anatomy/editorial"
        return [
            _read_model(path, EditorialStructureProposal) for path in sorted(root.glob("*.json"))
        ]

    def save_reference_reconstruction_proposal(
        self, proposal: ReferenceReconstructionProposal
    ) -> Path:
        _require_id(proposal.id, "reconstructionproposal")
        return self._save_project_immutable_model(
            proposal.project_id,
            Path("proposals/video-anatomy/reconstruction") / f"{proposal.id}.json",
            proposal,
        )

    def load_reference_reconstruction_proposal(
        self, project_id: str, proposal_id: str
    ) -> ReferenceReconstructionProposal:
        _require_id(proposal_id, "reconstructionproposal")
        return _read_model(
            self._project_root(project_id)
            / "proposals/video-anatomy/reconstruction"
            / f"{proposal_id}.json",
            ReferenceReconstructionProposal,
        )

    def list_reference_reconstruction_proposals(
        self, project_id: str
    ) -> list[ReferenceReconstructionProposal]:
        root = self._project_root(project_id) / "proposals/video-anatomy/reconstruction"
        return [
            _read_model(path, ReferenceReconstructionProposal)
            for path in sorted(root.glob("*.json"))
        ]

    def save_video_proposal_review(self, review: VideoProposalReview) -> Path:
        _require_id(review.id, "proposalreview")
        return self._save_project_immutable_model(
            review.project_id,
            Path("proposals/video-anatomy/reviews") / f"{review.id}.json",
            review,
        )

    def load_video_proposal_review(self, project_id: str, review_id: str) -> VideoProposalReview:
        _require_id(review_id, "proposalreview")
        return _read_model(
            self._project_root(project_id)
            / "proposals/video-anatomy/reviews"
            / f"{review_id}.json",
            VideoProposalReview,
        )

    def list_video_proposal_reviews(self, project_id: str) -> list[VideoProposalReview]:
        root = self._project_root(project_id) / "proposals/video-anatomy/reviews"
        return [_read_model(path, VideoProposalReview) for path in sorted(root.glob("*.json"))]

    def save_video_proposal_acceptance(self, receipt: VideoProposalAcceptanceReceipt) -> Path:
        _require_id(receipt.id, "proposalacceptance")
        return self._save_project_immutable_model(
            receipt.project_id,
            Path("proposals/video-anatomy/acceptances") / f"{receipt.id}.json",
            receipt,
        )

    def list_video_proposal_acceptances(
        self, project_id: str
    ) -> list[VideoProposalAcceptanceReceipt]:
        root = self._project_root(project_id) / "proposals/video-anatomy/acceptances"
        return [
            _read_model(path, VideoProposalAcceptanceReceipt)
            for path in sorted(root.glob("*.json"))
        ]

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
            self._project_root(treatment.project_id) / "treatments" / f"{treatment.id}.json",
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
        _atomic_json(self._project_root(patch.project_id) / "patches" / f"{patch.id}.json", patch)

    def save_decision_graph(self, graph: EditorialDecisionGraph) -> Path:
        path = self._project_root(graph.project_id) / "decisions" / "graphs" / f"{graph.id}.json"
        if path.exists():
            existing = _read_model(path, EditorialDecisionGraph)
            if existing != graph:
                raise StoreError(f"immutable decision graph already differs: {graph.id}")
            return path
        _atomic_json(path, graph)
        return path

    def load_decision_graph(self, project_id: str, graph_id: str) -> EditorialDecisionGraph:
        return _read_model(
            self._project_root(project_id) / "decisions" / "graphs" / f"{graph_id}.json",
            EditorialDecisionGraph,
        )

    def list_decision_graphs(self, project_id: str) -> list[EditorialDecisionGraph]:
        root = self._project_root(project_id) / "decisions" / "graphs"
        return [_read_model(path, EditorialDecisionGraph) for path in sorted(root.glob("*.json"))]

    def append_decision_event(self, event: DecisionLogEntry) -> None:
        path = self._project_root(event.project_id) / "decisions" / "log" / f"{event.id}.json"
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

    def load_job(self, project_id: str, job_id: str) -> JobReceipt:
        _require_id(job_id, "job")
        return _read_model(
            self._project_root(project_id) / "jobs" / f"{job_id}.json",
            JobReceipt,
        )

    def save_job_checkpoint(
        self,
        project_id: str,
        job_id: str,
        phase: str,
        payload: dict[str, object],
    ) -> Path:
        """Atomically promote a restart checkpoint under the owning project."""

        _require_id(job_id, "job")
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", phase) is None:
            raise StoreError("job checkpoint phase is not path-safe")
        path = self._project_root(project_id) / "jobs" / "checkpoints" / job_id / f"{phase}.json"
        _atomic_json(path, payload)
        return path

    def save_qc(self, report: QCReport) -> None:
        _atomic_json(self._project_root(report.project_id) / "qc" / f"{report.id}.json", report)

    def list_qc(self, project_id: str) -> list[QCReport]:
        root = self._project_root(project_id) / "qc"
        return [_read_model(path, QCReport) for path in sorted(root.glob("*.json"))]

    def save_interchange(self, report: InterchangeReport) -> None:
        _atomic_json(
            self._project_root(report.project_id) / "exports" / f"{report.id}.report.json",
            report,
        )

    def list_interchange(self, project_id: str) -> list[InterchangeReport]:
        root = self._project_root(project_id) / "exports"
        return [_read_model(path, InterchangeReport) for path in sorted(root.glob("*.report.json"))]

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
                raise StoreError(f"immutable reference workspace already differs: {workspace.id}")
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
            self._project_root(project_id) / "comparison" / "workspaces" / f"{workspace_id}.json",
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
        root = self._project_root(project_id) / "comparison" / "corrections" / "proposals"
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
        root = self._project_root(project_id) / "comparison" / "corrections" / "reviews"
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
            self.settings.state_root / "reference-media-bindings" / f"{reference_sha256}.json",
            binding,
        )
        return binding

    def resolve_reference_media(self, reference_sha256: str) -> Path:
        binding = _read_model(
            self.settings.state_root / "reference-media-bindings" / f"{reference_sha256}.json",
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
