"""Canonical receipt, cache, cancellation, and retry spine for Video Anatomy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aoa_editing import __version__
from aoa_editing.analysis.video_pipeline import (
    VideoAnatomyPipelineService,
    VideoAnatomyRunResult,
)
from aoa_editing.application.video_proposals import VideoProposalService
from aoa_editing.domain.models import (
    EditorialStructureProposal,
    FrameRange,
    JobReceipt,
    JobStatus,
    Provenance,
    ReferenceReconstructionProposal,
    VideoAnatomy,
    VideoAnatomyCachePruneReceipt,
    VideoAnatomyProfile,
    VideoAnatomyResourceEstimate,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.service import ProviderService


class VideoAnatomyCancelled(RuntimeError):
    """A cooperative cancellation was observed at a durable phase boundary."""


class VideoAnatomyResourceError(RuntimeError):
    """A deep pass was refused before decode because its resource gate failed."""


@dataclass(frozen=True, slots=True)
class VideoAnatomyJobResult:
    receipt: JobReceipt
    anatomy: VideoAnatomy
    run: VideoAnatomyRunResult | None
    editorial_proposal: EditorialStructureProposal | None = None
    reconstruction_proposal: ReferenceReconstructionProposal | None = None


class VideoAnatomyJobService:
    def __init__(
        self,
        store: ProjectStore,
        *,
        providers: ProviderService | None = None,
    ):
        self.store = store
        self.providers = providers

    def prune_rebuildable_cache(self, *, dry_run: bool = True) -> VideoAnatomyCachePruneReceipt:
        """Remove only provider/result caches; immutable project evidence is out of scope."""

        cache_root = (self.store.settings.cache_root / "video-anatomy").resolve()
        configured_cache = self.store.settings.cache_root.resolve()
        if not cache_root.is_relative_to(configured_cache):
            raise ValueError("Video Anatomy cache root escapes the configured cache")
        files = (
            sorted(path for path in cache_root.rglob("*") if path.is_file())
            if cache_root.exists()
            else []
        )
        selected_bytes = sum(path.stat().st_size for path in files)
        removed_files = 0
        removed_bytes = 0
        if not dry_run and cache_root.exists():
            self.store.settings.tmp_root.mkdir(parents=True, exist_ok=True)
            quarantine = Path(
                tempfile.mkdtemp(
                    prefix="video-anatomy-cache-prune-",
                    dir=self.store.settings.tmp_root,
                )
            ) / "payload"
            os.replace(cache_root, quarantine)
            removed_files = len(files)
            removed_bytes = selected_bytes
            shutil.rmtree(quarantine.parent)
        receipt = VideoAnatomyCachePruneReceipt(
            dry_run=dry_run,
            cache_root=str(cache_root),
            files_selected=len(files),
            bytes_selected=selected_bytes,
            files_removed=removed_files,
            bytes_removed=removed_bytes,
            provenance=Provenance(
                tool="aoa-editing-video-anatomy-cache-prune",
                tool_version=__version__,
                parameters={
                    "scope": "configured-cache-root/video-anatomy",
                    "canonical_project_roots_touched": False,
                },
                deterministic=False,
            ),
        )
        receipt_root = self.store.settings.state_root / "video-anatomy" / "cache-prune"
        receipt_root.mkdir(parents=True, exist_ok=True)
        target = receipt_root / f"{receipt.id}.json"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=receipt_root)
        try:
            with open(descriptor, "w", encoding="utf-8", closefd=True) as stream:
                stream.write(receipt.model_dump_json(indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return receipt

    def estimate(
        self,
        project_id: str,
        asset_id: str,
        *,
        profile: VideoAnatomyProfile = VideoAnatomyProfile.STRUCTURAL,
        analysis_range: FrameRange | None = None,
        pinned_frames: tuple[int, ...] = (),
    ) -> VideoAnatomyResourceEstimate:
        asset = self.store.load_asset(project_id, asset_id)
        metadata = asset.metadata
        if metadata.frame_rate is None or metadata.duration_seconds is None:
            raise ValueError("Video Anatomy resource estimate requires timed video metadata")
        source_frames = metadata.video_frame_count or max(
            1, round(metadata.duration_seconds * metadata.frame_rate.fps)
        )
        selected = analysis_range or FrameRange(start=0, duration=source_frames)
        if selected.end > source_frames + 1:
            raise ValueError("resource estimate range exceeds the measured source")
        pinned = set(pinned_frames)
        if any(frame < selected.start or frame >= selected.end for frame in pinned):
            raise ValueError("resource estimate pinned frame is outside the selected range")
        pixel_bytes = max(1, (metadata.width or 1920) * (metadata.height or 1080) * 3)
        speed_fps = {
            VideoAnatomyProfile.QUICK: 160.0,
            VideoAnatomyProfile.STRUCTURAL: 90.0,
            VideoAnatomyProfile.SEMANTIC: 45.0,
            VideoAnatomyProfile.MOTION: 16.0,
            VideoAnatomyProfile.RECONSTRUCT: 12.0,
        }[profile]
        estimated_decode_frames = (
            min(selected.duration, 48 + len(pinned))
            if profile is VideoAnatomyProfile.QUICK
            else selected.duration
        )
        artifact_cap = (
            {
                VideoAnatomyProfile.QUICK: 64,
                VideoAnatomyProfile.STRUCTURAL: 256,
                VideoAnatomyProfile.SEMANTIC: 384,
                VideoAnatomyProfile.MOTION: 512,
                VideoAnatomyProfile.RECONSTRUCT: 768,
            }[profile]
            * 1024
            * 1024
        )
        estimated_artifacts = min(
            artifact_cap,
            max(8 * 1024 * 1024, estimated_decode_frames * 96 * 1024),
        )
        peak_temporary = max(64 * 1024 * 1024, pixel_bytes * 12)
        reserve = 512 * 1024 * 1024
        required = estimated_artifacts + peak_temporary + reserve
        project_root = self.store.project_path(project_id)
        free = shutil.disk_usage(project_root).free
        warnings: list[str] = []
        if selected.duration / metadata.frame_rate.fps >= 300:
            warnings.append("long source: run quick or a focused range before a deep profile")
        if profile in {VideoAnatomyProfile.MOTION, VideoAnatomyProfile.RECONSTRUCT}:
            warnings.append("deep motion analysis decodes every requested frame")
        if profile is VideoAnatomyProfile.QUICK:
            warnings.append(
                "quick uses bounded keyframe/uniform probes; transition shapes remain approximate"
            )
        admitted = free >= required
        return VideoAnatomyResourceEstimate(
            project_id=project_id,
            asset_id=asset_id,
            source_sha256=asset.sha256,
            profile=profile,
            analysis_range=selected,
            source_duration_seconds=metadata.duration_seconds,
            requested_frame_count=selected.duration,
            estimated_decode_frame_count=estimated_decode_frames,
            estimated_runtime_seconds=max(0.05, estimated_decode_frames / speed_fps),
            estimated_artifact_bytes=estimated_artifacts,
            estimated_peak_temporary_bytes=peak_temporary,
            free_bytes_at_preflight=free,
            required_free_bytes=required,
            admitted=admitted,
            warnings=warnings,
            refusal_reason=(
                None
                if admitted
                else f"need {required} free bytes but measured {free} at the project store"
            ),
            provenance=Provenance(
                tool="aoa-editing-video-anatomy-resource-estimator",
                tool_version=__version__,
                parameters={
                    "reserve_bytes": reserve,
                    "speed_model_fps": speed_fps,
                    "pixel_bytes": pixel_bytes,
                },
                deterministic=False,
            ),
        )

    def run(
        self,
        project_id: str,
        asset_id: str,
        *,
        profile: VideoAnatomyProfile = VideoAnatomyProfile.STRUCTURAL,
        analysis_range: FrameRange | None = None,
        pinned_frames: tuple[int, ...] = (),
        explicit_provider_opt_in: bool = False,
        resume_from_job_id: str | None = None,
    ) -> VideoAnatomyJobResult:
        asset = self.store.load_asset(project_id, asset_id)
        estimate = self.estimate(
            project_id,
            asset_id,
            profile=profile,
            analysis_range=analysis_range,
            pinned_frames=pinned_frames,
        )
        if not estimate.admitted:
            raise VideoAnatomyResourceError(estimate.refusal_reason or "resource preflight refused")
        provider_revision = self._provider_revision(profile)
        cache_key = self._cache_key(
            asset.sha256,
            profile,
            analysis_range,
            pinned_frames,
            explicit_provider_opt_in,
            provider_revision,
        )
        attempt = 1
        previous: JobReceipt | None = None
        resume_plan_id: str | None = None
        resume_completed_phases: tuple[str, ...] = ()
        resume_evidence_refs: tuple[str, ...] = ()
        if resume_from_job_id is not None:
            previous = self.store.load_job(project_id, resume_from_job_id)
            attempt = previous.attempt + 1
            resume_plan_id = previous.plan_id or self._plan_id_from_evidence(
                project_id, previous.partial_evidence_refs
            )
            resume_completed_phases = tuple(previous.completed_phases)
            resume_evidence_refs = tuple(previous.partial_evidence_refs)
            # A successful job points at its final VideoAnatomy packet, whereas an
            # interrupted job points at the phase-manifest used for true resume.
            # Successful retries are satisfied by the normal cache path below and
            # must not be parsed as phase manifests.
            if previous.status is not JobStatus.SUCCEEDED:
                self._verify_resume_checkpoint(project_id, previous)
        request_parameters: dict[str, object] = {
            "analysis_range": (
                analysis_range.model_dump(mode="json") if analysis_range is not None else None
            ),
            "pinned_frames": sorted(set(pinned_frames)),
            "explicit_provider_opt_in": explicit_provider_opt_in,
            "resource_estimate": estimate.model_dump(mode="json"),
            "pipeline_revision": f"video-anatomy-v1@{__version__}",
        }
        cached = next(
            (
                item
                for item in reversed(self.store.list_jobs(project_id))
                if item.asset_id == asset_id
                and item.cache_key == cache_key
                and item.status is JobStatus.SUCCEEDED
                and item.checkpoint_path is not None
                and item.checkpoint_sha256 is not None
            ),
            None,
        )
        if cached is not None:
            assert cached.checkpoint_path is not None
            assert cached.checkpoint_sha256 is not None
            anatomy = self._load_checkpoint(project_id, cached)
            now = datetime.now(UTC)
            receipt = JobReceipt(
                kind="video.anatomy",
                project_id=project_id,
                asset_id=asset_id,
                plan_id=anatomy.plan.id,
                plan_sha256=anatomy.plan.plan_sha256,
                profile=profile.value,
                phase="cache-hit",
                progress=1.0,
                status=JobStatus.SUCCEEDED,
                started_at=now,
                finished_at=now,
                input_hashes=[asset.sha256, anatomy.plan.plan_sha256],
                output_paths=[cached.checkpoint_path],
                output_hashes=[cached.checkpoint_sha256],
                checkpoint_path=cached.checkpoint_path,
                checkpoint_sha256=cached.checkpoint_sha256,
                cache_key=cache_key,
                cache_hit=True,
                resume_from_job_id=resume_from_job_id,
                attempt=attempt,
                request_parameters=request_parameters,
                completed_phases=["cache-hit"],
                partial_evidence_refs=anatomy.evidence_refs,
            )
            self.store.save_job(receipt)
            editorial, reconstruction = self._profile_proposals(anatomy)
            return VideoAnatomyJobResult(
                receipt=receipt,
                anatomy=anatomy,
                run=None,
                editorial_proposal=editorial,
                reconstruction_proposal=reconstruction,
            )

        queued = JobReceipt(
            kind="video.anatomy",
            project_id=project_id,
            asset_id=asset_id,
            profile=profile.value,
            phase="queued",
            status=JobStatus.QUEUED,
            input_hashes=[asset.sha256],
            cache_key=cache_key,
            resume_from_job_id=resume_from_job_id,
            attempt=attempt,
            request_parameters=request_parameters,
            plan_id=resume_plan_id,
            plan_sha256=(previous.plan_sha256 if previous is not None else None),
            completed_phases=list(resume_completed_phases),
            partial_evidence_refs=list(resume_evidence_refs),
        )
        self.store.save_job(queued)
        running = queued.model_copy(
            update={
                "status": JobStatus.RUNNING,
                "phase": "starting",
                "started_at": datetime.now(UTC),
            }
        )
        self.store.save_job(running)
        partial_refs: list[str] = list(resume_evidence_refs)

        def progress(phase: str, value: float, evidence_refs: list[str]) -> None:
            current = self.store.load_job(project_id, queued.id)
            if current.cancellation_requested or current.status is JobStatus.CANCELLED:
                raise VideoAnatomyCancelled(f"cancelled before phase promotion: {phase}")
            partial_refs.extend(evidence_refs)
            completed = list(dict.fromkeys([*current.completed_phases, phase]))
            checkpoint_path, checkpoint_hash, output_paths, output_hashes, plan_id = (
                self._write_phase_checkpoint(
                    project_id,
                    queued.id,
                    phase,
                    cache_key,
                    asset.sha256,
                    completed,
                    sorted(set(partial_refs)),
                )
            )
            plan_sha256 = (
                self.store.load_video_sampling_plan(project_id, plan_id).plan_sha256
                if plan_id is not None
                else current.plan_sha256
            )
            child_now = datetime.now(UTC)
            child = JobReceipt(
                kind=f"video.anatomy.stage.{phase}",
                project_id=project_id,
                asset_id=asset_id,
                profile=profile.value,
                phase=phase,
                progress=1.0,
                status=JobStatus.SUCCEEDED,
                started_at=child_now,
                finished_at=child_now,
                input_hashes=[asset.sha256],
                output_paths=output_paths,
                output_hashes=output_hashes,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                cache_key=hashlib.sha256(f"{cache_key}:{phase}".encode()).hexdigest(),
                parent_job_id=queued.id,
                attempt=attempt,
                request_parameters=request_parameters,
                completed_phases=[phase],
                partial_evidence_refs=evidence_refs,
            )
            self.store.save_job(child)
            self.store.save_job(
                current.model_copy(
                    update={
                        "phase": phase,
                        "progress": value,
                        "plan_id": plan_id or current.plan_id,
                        "plan_sha256": plan_sha256,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_sha256": checkpoint_hash,
                        "completed_phases": completed,
                        "partial_evidence_refs": sorted(set(partial_refs)),
                    }
                )
            )

        try:
            run = VideoAnatomyPipelineService(
                self.store,
                providers=self.providers,
            ).analyze(
                project_id,
                asset_id,
                profile=profile,
                analysis_range=analysis_range,
                pinned_frames=pinned_frames,
                explicit_provider_opt_in=explicit_provider_opt_in,
                phase_hook=progress,
                resume_plan_id=resume_plan_id,
                resume_completed_phases=resume_completed_phases,
                resume_evidence_refs=resume_evidence_refs,
            )
            anatomy_path = self._anatomy_path(project_id, run.anatomy)
            anatomy_hash = sha256_file(anatomy_path)
            relative = str(anatomy_path.relative_to(self.store.project_path(project_id)))
            current = self.store.load_job(project_id, queued.id)
            receipt = current.model_copy(
                update={
                    "plan_id": run.anatomy.plan.id,
                    "plan_sha256": run.anatomy.plan.plan_sha256,
                    "phase": "complete",
                    "progress": 1.0,
                    "status": JobStatus.SUCCEEDED,
                    "finished_at": datetime.now(UTC),
                    "input_hashes": [asset.sha256, run.anatomy.plan.plan_sha256],
                    "output_paths": [relative],
                    "output_hashes": [anatomy_hash],
                    "checkpoint_path": relative,
                    "checkpoint_sha256": anatomy_hash,
                    "partial_evidence_refs": run.anatomy.evidence_refs,
                }
            )
            self.store.save_job(receipt)
            editorial, reconstruction = self._profile_proposals(run.anatomy)
            return VideoAnatomyJobResult(
                receipt=receipt,
                anatomy=run.anatomy,
                run=run,
                editorial_proposal=editorial,
                reconstruction_proposal=reconstruction,
            )
        except VideoAnatomyCancelled:
            current = self.store.load_job(project_id, queued.id)
            cancelled = current.model_copy(
                update={
                    "status": JobStatus.CANCELLED,
                    "cancellation_requested": True,
                    "finished_at": datetime.now(UTC),
                    "error": "cancelled at a durable phase boundary",
                    "partial_evidence_refs": sorted(set(partial_refs)),
                }
            )
            self.store.save_job(cancelled)
            raise
        except Exception as error:
            current = self.store.load_job(project_id, queued.id)
            failed = current.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "finished_at": datetime.now(UTC),
                    "error": f"{type(error).__name__}: {str(error)[-1800:]}",
                    "partial_evidence_refs": sorted(set(partial_refs)),
                }
            )
            self.store.save_job(failed)
            raise

    def cancel(self, project_id: str, job_id: str) -> JobReceipt:
        current = self.store.load_job(project_id, job_id)
        if current.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return current
        cancelled = current.model_copy(
            update={
                "status": JobStatus.CANCELLED,
                "cancellation_requested": True,
                "finished_at": datetime.now(UTC),
                "error": "cancellation requested by operator",
            }
        )
        self.store.save_job(cancelled)
        return cancelled

    def retry(self, project_id: str, job_id: str) -> VideoAnatomyJobResult:
        previous = self.store.load_job(project_id, job_id)
        if previous.asset_id is None or previous.profile is None:
            raise ValueError("selected job lacks a resumable Video Anatomy identity")
        raw_range = previous.request_parameters.get("analysis_range")
        analysis_range = (
            FrameRange.model_validate(raw_range) if isinstance(raw_range, dict) else None
        )
        raw_pins = previous.request_parameters.get("pinned_frames", [])
        pinned_frames = tuple(int(item) for item in raw_pins) if isinstance(raw_pins, list) else ()
        return self.run(
            project_id,
            previous.asset_id,
            profile=VideoAnatomyProfile(previous.profile),
            analysis_range=analysis_range,
            pinned_frames=pinned_frames,
            explicit_provider_opt_in=bool(
                previous.request_parameters.get("explicit_provider_opt_in", False)
            ),
            resume_from_job_id=previous.id,
        )

    def _write_phase_checkpoint(
        self,
        project_id: str,
        job_id: str,
        phase: str,
        cache_key: str,
        source_sha256: str,
        completed_phases: list[str],
        evidence_refs: list[str],
    ) -> tuple[str, str, list[str], list[str], str | None]:
        project_root = self.store.project_path(project_id)
        records = {
            item.id: item
            for item in self.store.list_evidence(project_id)
            if item.id in set(evidence_refs)
        }
        paths: list[Path] = []
        for evidence_id in evidence_refs:
            record = records.get(evidence_id)
            if record is None:
                raise ValueError(f"phase checkpoint cannot resolve evidence: {evidence_id}")
            paths.append(project_root / "evidence" / f"{evidence_id}.json")
            for artifact in record.artifacts:
                selected = (project_root / artifact).resolve()
                if not selected.is_relative_to(project_root.resolve()):
                    raise ValueError("phase checkpoint artifact escapes the project")
                paths.append(selected)
        unique_paths = sorted(set(paths))
        if any(not path.is_file() for path in unique_paths):
            raise ValueError("phase checkpoint includes a missing artifact")
        relative_paths = [str(path.relative_to(project_root)) for path in unique_paths]
        hashes = [sha256_file(path) for path in unique_paths]
        plan_id = self._plan_id_from_evidence(project_id, evidence_refs)
        payload: dict[str, object] = {
            "schema_version": "1.0.0",
            "job_id": job_id,
            "phase": phase,
            "cache_key": cache_key,
            "source_sha256": source_sha256,
            "plan_id": plan_id,
            "completed_phases": completed_phases,
            "evidence_refs": evidence_refs,
            "artifact_paths": relative_paths,
            "artifact_sha256": hashes,
        }
        path = self.store.save_job_checkpoint(project_id, job_id, phase, payload)
        relative = str(path.relative_to(project_root))
        return relative, sha256_file(path), relative_paths, hashes, plan_id

    def _verify_resume_checkpoint(self, project_id: str, receipt: JobReceipt) -> None:
        if receipt.checkpoint_path is None and not receipt.completed_phases:
            return
        if receipt.checkpoint_path is None or receipt.checkpoint_sha256 is None:
            raise ValueError("resumable job has completed phases without a hashed checkpoint")
        project_root = self.store.project_path(project_id).resolve()
        checkpoint = (project_root / receipt.checkpoint_path).resolve()
        if (
            not checkpoint.is_relative_to(project_root)
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != receipt.checkpoint_sha256
        ):
            raise ValueError("resume checkpoint is missing, outside the project, or changed")
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        paths = payload.get("artifact_paths")
        hashes = payload.get("artifact_sha256")
        if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes):
            raise ValueError("resume checkpoint artifact manifest is malformed")
        for relative, expected in zip(paths, hashes, strict=True):
            path = (project_root / str(relative)).resolve()
            if (
                not path.is_relative_to(project_root)
                or not path.is_file()
                or sha256_file(path) != str(expected)
            ):
                raise ValueError(f"resume artifact is missing or changed: {relative}")

    def _plan_id_from_evidence(self, project_id: str, evidence_refs: list[str]) -> str | None:
        wanted = set(evidence_refs)
        for record in reversed(self.store.list_evidence(project_id)):
            if record.id not in wanted:
                continue
            plan_id = record.payload.get("plan_id")
            if isinstance(plan_id, str) and plan_id.startswith("samplingplan_"):
                return plan_id
        return None

    def _load_checkpoint(self, project_id: str, receipt: JobReceipt) -> VideoAnatomy:
        assert receipt.checkpoint_path is not None
        assert receipt.checkpoint_sha256 is not None
        path = self.store.project_path(project_id) / Path(receipt.checkpoint_path)
        if not path.is_file() or sha256_file(path) != receipt.checkpoint_sha256:
            raise ValueError("cached Video Anatomy checkpoint is missing or changed")
        return VideoAnatomy.model_validate_json(path.read_text(encoding="utf-8"))

    def _anatomy_path(self, project_id: str, anatomy: VideoAnatomy) -> Path:
        return (
            self.store.video_anatomy_path(project_id, anatomy.plan.id)
            / "anatomies"
            / f"{anatomy.id}.json"
        )

    @staticmethod
    def _cache_key(
        source_sha256: str,
        profile: VideoAnatomyProfile,
        analysis_range: FrameRange | None,
        pinned_frames: tuple[int, ...],
        explicit_provider_opt_in: bool,
        provider_revision: str,
    ) -> str:
        payload = json.dumps(
            {
                "source_sha256": source_sha256,
                "profile": profile.value,
                "analysis_range": (
                    analysis_range.model_dump(mode="json") if analysis_range is not None else None
                ),
                "pinned_frames": sorted(set(pinned_frames)),
                "explicit_provider_opt_in": explicit_provider_opt_in,
                "provider_revision": provider_revision,
                "pipeline_revision": f"video-anatomy-v1@{__version__}",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _provider_revision(self, profile: VideoAnatomyProfile) -> str:
        capabilities: set[str] = set()
        if profile in {VideoAnatomyProfile.QUICK, VideoAnatomyProfile.RECONSTRUCT}:
            capabilities.add("speech.transcript")
        if profile in {VideoAnatomyProfile.SEMANTIC, VideoAnatomyProfile.RECONSTRUCT}:
            capabilities.add("vision.describe")
        if not capabilities:
            return "not-applicable"
        providers = self.providers or ProviderService.for_settings(self.store.settings)
        status = providers.inspect_bindings(probe=False)
        selected = sorted(
            (
                item
                for item in status.get("bindings", [])
                if item.get("capability_id") in capabilities
            ),
            key=lambda item: str(item.get("capability_id", "")),
        )
        found = {str(item.get("capability_id")) for item in selected}
        selected.extend(
            {"capability_id": capability, "binding_state": "unbound"}
            for capability in sorted(capabilities - found)
        )
        return hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _profile_proposals(
        self,
        anatomy: VideoAnatomy,
    ) -> tuple[EditorialStructureProposal | None, ReferenceReconstructionProposal | None]:
        if anatomy.profile is not VideoAnatomyProfile.RECONSTRUCT:
            return None, None
        editorial = next(
            (
                item
                for item in self.store.list_editorial_structure_proposals(anatomy.project_id)
                if item.anatomy_id == anatomy.id
                and item.anatomy_sha256 == anatomy.anatomy_sha256
            ),
            None,
        )
        reconstruction = next(
            (
                item
                for item in self.store.list_reference_reconstruction_proposals(
                    anatomy.project_id
                )
                if item.reference_anatomy_id == anatomy.id
                and item.reference_anatomy_sha256 == anatomy.anatomy_sha256
                and item.target_asset_id is None
            ),
            None,
        )
        proposals = VideoProposalService(self.store)
        return (
            editorial or proposals.create_editorial_structure(anatomy),
            reconstruction or proposals.create_reference_reconstruction(anatomy),
        )
