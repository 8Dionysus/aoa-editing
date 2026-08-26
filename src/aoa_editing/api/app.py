"""FastAPI workbench over the shared application core."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from aoa_editing import __version__
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.analysis.video_pipeline import VideoAnatomyPipelineService
from aoa_editing.application.language import NaturalLanguagePatchService
from aoa_editing.application.reference_workspace import (
    MotionCorrectionService,
    ReadinessGuard,
    ReferenceWorkspaceService,
)
from aoa_editing.application.service import EditingService
from aoa_editing.application.video_jobs import VideoAnatomyJobService
from aoa_editing.application.video_proposals import VideoProposalService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    EditPatch,
    FrameRange,
    FrameRate,
    Intent,
    MotionCorrectionOperationV2,
    MotionCorrectionReviewDecisionV2,
    PatchOperation,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionSpecV2,
    Scenario,
    ScreenWorkflowBeatInput,
    ScreenWorkflowEditSpec,
    ScreenWorkflowPlan,
    ScreenWorkflowVoiceoverCue,
    ScreenWorkflowVoiceoverTiming,
    VideoAnatomyProfile,
)
from aoa_editing.infrastructure.probes import doctor_report
from aoa_editing.infrastructure.store import ProjectStore, StoreError
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCreate(RequestModel):
    name: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    scenario: Scenario
    target_duration_seconds: float | None = Field(default=None, gt=0)
    audience: str | None = None
    frame_format: str | None = None
    tempo: str | None = None
    mood: str | None = None
    required_elements: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    sound_requirements: list[str] = Field(default_factory=list)
    privacy_mode: Literal["local-only", "explicit-provider-opt-in"] = "local-only"
    automation_level: Literal["suggest", "review-before-apply", "approved-auto"] = (
        "review-before-apply"
    )

    def to_intent(self) -> Intent:
        return Intent(
            text=self.intent,
            scenario=self.scenario,
            target_duration_seconds=self.target_duration_seconds,
            audience=self.audience,
            frame_format=self.frame_format,
            tempo=self.tempo,
            mood=self.mood,
            required_elements=self.required_elements,
            forbidden_elements=self.forbidden_elements,
            sound_requirements=self.sound_requirements,
            privacy_mode=self.privacy_mode,
            automation_level=self.automation_level,
        )


class BriefRevisionRequest(ProjectCreate):
    name: str = Field(default="unchanged", exclude=True)
    rationale: str = Field(min_length=1)


class StyleConfirmationRequest(RequestModel):
    preference: str = Field(min_length=1)
    value: Any


class StyleProfileRequest(RequestModel):
    scope: str = Field(min_length=1)
    explicit_opt_in: Literal[True]
    confirmations: list[StyleConfirmationRequest] = Field(min_length=1)


class AnalyzeRequest(RequestModel):
    transcribe: bool = False


class VideoAnatomyRequest(RequestModel):
    profile: VideoAnatomyProfile = VideoAnatomyProfile.STRUCTURAL
    start_frame: int | None = Field(default=None, ge=0)
    duration_frames: int | None = Field(default=None, gt=0)
    pinned_frames: list[int] = Field(default_factory=list)
    explicit_provider_opt_in: bool = False

    def frame_range(self) -> FrameRange | None:
        if self.start_frame is None and self.duration_frames is None:
            return None
        if self.start_frame is None or self.duration_frames is None:
            raise ValueError("start_frame and duration_frames must appear together")
        return FrameRange(start=self.start_frame, duration=self.duration_frames)


class VideoAnatomyCachePruneRequest(RequestModel):
    apply: bool = False


class VideoReconstructionProposalRequest(RequestModel):
    target_asset_id: str | None = None
    base_version_id: str | None = None
    precise_motion_spec: ReferenceReconstructionSpecV2 | None = None
    precise_motion_spec_ref: str | None = None
    phase_correction: ReferencePhaseCorrectionV2 | None = None
    phase_correction_ref: str | None = None


class VideoProposalReviewRequest(RequestModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class ProposeRequest(RequestModel):
    scenario: Scenario | None = None
    asset_id: str | None = None
    duration_seconds: float | None = Field(default=None, gt=0)


class ScreenWorkflowPlanRequest(RequestModel):
    title: str = Field(min_length=1)
    script: str = Field(min_length=1)
    beats: list[ScreenWorkflowBeatInput] | None = None
    reviewed_by: str | None = None
    reference_study_ids: list[str] = Field(default_factory=list)


class ScreenWorkflowTreatmentRequest(RequestModel):
    plan: ScreenWorkflowPlan
    edit_spec: ScreenWorkflowEditSpec


class ScreenWorkflowVoiceoverTimingRequest(RequestModel):
    plan: ScreenWorkflowPlan
    asset_id: str = Field(min_length=1)
    timeline_frame_rate: FrameRate = Field(default_factory=FrameRate)
    transcribe: bool = True


class ScreenWorkflowVoiceoverReviewRequest(RequestModel):
    plan: ScreenWorkflowPlan
    draft: ScreenWorkflowVoiceoverTiming
    cues: list[ScreenWorkflowVoiceoverCue] = Field(min_length=1)
    reviewed_by: str = Field(min_length=1)
    review_note: str = Field(min_length=1)


class PatchRequest(RequestModel):
    base_version_id: str
    operations: list[PatchOperation]
    rationale: str = Field(min_length=1)
    message: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class LanguagePatchRequest(RequestModel):
    command: str = Field(min_length=1)
    version_id: str | None = None


class EvidenceCorrectionRequest(RequestModel):
    asset_id: str
    kind: str = Field(min_length=1)
    payload: dict[str, Any]
    supersedes: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)


class ReferenceWorkspaceAttachRequest(RequestModel):
    study_path: str = Field(min_length=1)
    reference_media_path: str | None = None


class MotionCorrectionLanguageRequest(RequestModel):
    command: str = Field(min_length=1)
    version_id: str | None = None


class MotionCorrectionOperationsRequest(RequestModel):
    operations: list[MotionCorrectionOperationV2] = Field(min_length=1)
    version_id: str | None = None


class MotionCorrectionReviewRequest(RequestModel):
    decisions: list[MotionCorrectionReviewDecisionV2] = Field(min_length=1)


def create_app(
    settings: Settings | None = None,
    *,
    readiness_guard: ReadinessGuard | None = None,
) -> FastAPI:
    selected = settings or Settings.from_env()
    store = ProjectStore(selected)
    editing = EditingService(store)
    video_jobs = VideoAnatomyJobService(store)
    video_proposals = VideoProposalService(store)
    reference_workspaces = (
        ReferenceWorkspaceService(store, readiness_guard=readiness_guard)
        if readiness_guard is not None
        else ReferenceWorkspaceService(store)
    )
    motion_corrections = (
        MotionCorrectionService(store, readiness_guard=readiness_guard)
        if readiness_guard is not None
        else MotionCorrectionService(store)
    )
    app = FastAPI(
        title="AoA Editing",
        version=__version__,
        description="Local-first evidence-backed AI video co-editor",
    )
    app.state.settings = selected
    app.state.store = store
    app.state.editing = editing
    app.state.video_jobs = video_jobs
    app.state.video_proposals = video_proposals
    app.state.reference_workspaces = reference_workspaces
    app.state.motion_corrections = motion_corrections
    static_root = Path(__file__).resolve().parents[1] / "web"
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.exception_handler(StoreError)
    async def store_error_handler(_request: Any, error: StoreError):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "local_first": True}

    @app.get("/api/doctor")
    def doctor() -> dict[str, Any]:
        return doctor_report(selected, create_roots=False)

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        return [project.model_dump(mode="json") for project in store.list_projects()]

    @app.post("/api/projects", status_code=201)
    def create_project(request: ProjectCreate) -> dict[str, Any]:
        project = editing.create_project(
            request.name,
            request.to_intent(),
        )
        return project.model_dump(mode="json")

    @app.post("/api/projects/{project_id}/briefs", status_code=201)
    def revise_brief(project_id: str, request: BriefRevisionRequest) -> dict[str, Any]:
        return editing.revise_brief(project_id, request.to_intent(), request.rationale).model_dump(
            mode="json"
        )

    @app.get("/api/style-profiles")
    def style_profiles() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in store.list_style_profiles()]

    @app.post("/api/style-profiles", status_code=201)
    def confirm_style(request: StyleProfileRequest) -> dict[str, Any]:
        return editing.confirm_style(
            scope=request.scope,
            confirmations=[(item.preference, item.value) for item in request.confirmations],
        ).model_dump(mode="json")

    @app.get("/api/projects/{project_id}")
    def project_detail(project_id: str) -> dict[str, Any]:
        return _project_bundle(store, project_id)

    @app.post("/api/projects/{project_id}/assets", status_code=201)
    def upload_asset(
        project_id: str,
        file: Annotated[UploadFile, File(description="Local source media")],
    ) -> dict[str, Any]:
        suffix = Path(file.filename or "upload.bin").suffix[:12]
        selected.tmp_root.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix="upload-", suffix=suffix, dir=selected.tmp_root
        )
        try:
            with open(descriptor, "wb", closefd=True) as target:
                shutil.copyfileobj(file.file, target)
            asset, evidence = editing.ingest(project_id, Path(temp_name))
            return {
                "asset": asset.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "derivatives": store.load_derivatives(project_id, asset.id).model_dump(mode="json"),
            }
        finally:
            Path(temp_name).unlink(missing_ok=True)

    @app.post("/api/projects/{project_id}/assets/{asset_id}/analyze")
    def analyze_asset(
        project_id: str, asset_id: str, request: AnalyzeRequest
    ) -> list[dict[str, Any]]:
        records = AnalysisService(store).analyze(
            project_id, asset_id, transcribe=request.transcribe
        )
        return [record.model_dump(mode="json") for record in records]

    @app.post("/api/projects/{project_id}/assets/{asset_id}/video-anatomy")
    def analyze_video_anatomy(
        project_id: str,
        asset_id: str,
        request: VideoAnatomyRequest,
    ) -> dict[str, Any]:
        result = video_jobs.run(
            project_id,
            asset_id,
            profile=request.profile,
            analysis_range=request.frame_range(),
            pinned_frames=tuple(sorted(set(request.pinned_frames))),
            explicit_provider_opt_in=request.explicit_provider_opt_in,
        )
        return {
            "job": result.receipt.model_dump(mode="json"),
            "anatomy": result.anatomy.model_dump(mode="json"),
            "editorial_proposal": (
                result.editorial_proposal.model_dump(mode="json")
                if result.editorial_proposal is not None
                else None
            ),
            "reconstruction_proposal": (
                result.reconstruction_proposal.model_dump(mode="json")
                if result.reconstruction_proposal is not None
                else None
            ),
        }

    @app.post("/api/projects/{project_id}/assets/{asset_id}/video-anatomy/estimate")
    def estimate_video_anatomy(
        project_id: str,
        asset_id: str,
        request: VideoAnatomyRequest,
    ) -> dict[str, Any]:
        return video_jobs.estimate(
            project_id,
            asset_id,
            profile=request.profile,
            analysis_range=request.frame_range(),
            pinned_frames=tuple(sorted(set(request.pinned_frames))),
        ).model_dump(mode="json")

    @app.post("/api/video-anatomy/cache/prune")
    def prune_video_anatomy_cache(
        request: VideoAnatomyCachePruneRequest,
    ) -> dict[str, Any]:
        return video_jobs.prune_rebuildable_cache(
            dry_run=not request.apply
        ).model_dump(mode="json")

    @app.get("/api/projects/{project_id}/video-anatomy/{plan_id}")
    def get_video_anatomy(project_id: str, plan_id: str) -> dict[str, Any]:
        return store.load_video_anatomy(project_id, plan_id).model_dump(mode="json")

    @app.get("/api/projects/{project_id}/video-anatomy/{plan_id}/contact-sheet")
    def get_video_anatomy_contact_sheet(project_id: str, plan_id: str) -> FileResponse:
        path = store.video_anatomy_path(project_id, plan_id) / "contact-sheet.jpg"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="contact sheet not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/projects/{project_id}/video-anatomy/{plan_id}/focused-plans")
    def create_video_anatomy_focused_plans(
        project_id: str,
        plan_id: str,
    ) -> list[dict[str, Any]]:
        anatomy = store.load_video_anatomy(project_id, plan_id)
        return [
            item.model_dump(mode="json")
            for item in VideoAnatomyPipelineService(store).create_focused_plans(anatomy)
        ]

    @app.post("/api/projects/{project_id}/video-anatomy/{plan_id}/shots/{shot_id}/deepen")
    def deepen_video_anatomy_shot(
        project_id: str,
        plan_id: str,
        shot_id: str,
    ) -> dict[str, Any]:
        anatomy = store.load_video_anatomy(project_id, plan_id)
        shot = next((item for item in anatomy.structure.shots if item.id == shot_id), None)
        if shot is None:
            raise HTTPException(status_code=404, detail="shot not found")
        result = video_jobs.run(
            project_id,
            anatomy.asset_id,
            profile=VideoAnatomyProfile.MOTION,
            analysis_range=shot.frame_range,
            pinned_frames=(shot.frame_range.start, shot.frame_range.end - 1),
        )
        return {
            "job": result.receipt.model_dump(mode="json"),
            "anatomy": result.anatomy.model_dump(mode="json"),
        }

    @app.get("/api/projects/{project_id}/jobs/{job_id}")
    def get_job(project_id: str, job_id: str) -> dict[str, Any]:
        return store.load_job(project_id, job_id).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/jobs/{job_id}/cancel")
    def cancel_job(project_id: str, job_id: str) -> dict[str, Any]:
        return video_jobs.cancel(project_id, job_id).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/jobs/{job_id}/retry")
    def retry_job(project_id: str, job_id: str) -> dict[str, Any]:
        result = video_jobs.retry(project_id, job_id)
        return {
            "job": result.receipt.model_dump(mode="json"),
            "anatomy": result.anatomy.model_dump(mode="json"),
        }

    @app.post(
        "/api/projects/{project_id}/video-anatomy/{plan_id}/editorial-proposals",
        status_code=201,
    )
    def create_video_editorial_proposal(
        project_id: str,
        plan_id: str,
    ) -> dict[str, Any]:
        anatomy = store.load_video_anatomy(project_id, plan_id)
        return video_proposals.create_editorial_structure(anatomy).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/video-anatomy/{plan_id}/reconstruction-proposals",
        status_code=201,
    )
    def create_video_reconstruction_proposal(
        project_id: str,
        plan_id: str,
        request: VideoReconstructionProposalRequest,
    ) -> dict[str, Any]:
        anatomy = store.load_video_anatomy(project_id, plan_id)
        if (request.precise_motion_spec is None) != (
            request.precise_motion_spec_ref is None
        ):
            raise HTTPException(
                status_code=422,
                detail="precise motion spec and immutable ref must appear together",
            )
        if (request.phase_correction is None) != (request.phase_correction_ref is None):
            raise HTTPException(
                status_code=422,
                detail="phase correction and immutable ref must appear together",
            )
        return video_proposals.create_reference_reconstruction(
            anatomy,
            target_asset_id=request.target_asset_id,
            base_version_id=request.base_version_id,
            precise_motion_spec=request.precise_motion_spec,
            precise_motion_spec_ref=request.precise_motion_spec_ref,
            phase_correction=request.phase_correction,
            phase_correction_ref=request.phase_correction_ref,
        ).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/video-editorial-proposals/{proposal_id}/review",
        status_code=201,
    )
    def review_video_editorial_proposal(
        project_id: str,
        proposal_id: str,
        request: VideoProposalReviewRequest,
    ) -> dict[str, Any]:
        return video_proposals.review_editorial(
            project_id,
            proposal_id,
            decision=request.decision,
            reviewer=request.reviewer,
            rationale=request.rationale,
        ).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/video-reconstruction-proposals/{proposal_id}/review",
        status_code=201,
    )
    def review_video_reconstruction_proposal(
        project_id: str,
        proposal_id: str,
        request: VideoProposalReviewRequest,
    ) -> dict[str, Any]:
        return video_proposals.review_reconstruction(
            project_id,
            proposal_id,
            decision=request.decision,
            reviewer=request.reviewer,
            rationale=request.rationale,
        ).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/video-reconstruction-proposals/"
        "{proposal_id}/accept/{review_id}",
        status_code=201,
    )
    def accept_video_reconstruction_proposal(
        project_id: str,
        proposal_id: str,
        review_id: str,
    ) -> dict[str, Any]:
        return video_proposals.accept_reconstruction(
            project_id,
            proposal_id,
            review_id,
        ).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/evidence/corrections", status_code=201)
    def correct_evidence(project_id: str, request: EvidenceCorrectionRequest) -> dict[str, Any]:
        return editing.correct_evidence(
            project_id,
            asset_id=request.asset_id,
            kind=request.kind,
            payload=request.payload,
            supersedes=request.supersedes,
            rationale=request.rationale,
            confidence=request.confidence,
        ).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/reference-workspaces", status_code=201)
    def attach_reference_workspace(
        project_id: str,
        request: ReferenceWorkspaceAttachRequest,
    ) -> dict[str, Any]:
        return reference_workspaces.attach(
            project_id,
            study_path=Path(request.study_path),
            reference_media_path=(
                Path(request.reference_media_path)
                if request.reference_media_path is not None
                else None
            ),
        ).model_dump(mode="json")

    @app.get("/api/projects/{project_id}/reference-workspaces/{workspace_id}")
    def reference_workspace_bundle(
        project_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        return reference_workspaces.bundle(project_id, workspace_id).model_dump(mode="json")

    @app.get("/api/projects/{project_id}/reference-workspaces/{workspace_id}/artifacts/{role}")
    def reference_workspace_artifact(
        project_id: str,
        workspace_id: str,
        role: str,
    ) -> FileResponse:
        path, media_type = reference_workspaces.resolve_artifact(
            project_id,
            workspace_id,
            role,
        )
        return FileResponse(path, media_type=media_type)

    @app.post(
        "/api/projects/{project_id}/reference-workspaces/"
        "{workspace_id}/corrections/preview-language",
        status_code=201,
    )
    def preview_motion_correction_language(
        project_id: str,
        workspace_id: str,
        request: MotionCorrectionLanguageRequest,
    ) -> dict[str, Any]:
        return motion_corrections.preview_language(
            project_id,
            workspace_id,
            request.command,
            version_id=request.version_id,
        ).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/reference-workspaces/"
        "{workspace_id}/corrections/preview-operations",
        status_code=201,
    )
    def preview_motion_correction_operations(
        project_id: str,
        workspace_id: str,
        request: MotionCorrectionOperationsRequest,
    ) -> dict[str, Any]:
        return motion_corrections.preview_operations(
            project_id,
            workspace_id,
            request.operations,
            version_id=request.version_id,
        ).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/motion-correction-proposals/{proposal_id}/review",
        status_code=201,
    )
    def review_motion_correction_proposal(
        project_id: str,
        proposal_id: str,
        request: MotionCorrectionReviewRequest,
    ) -> dict[str, Any]:
        return motion_corrections.review(
            project_id,
            proposal_id,
            request.decisions,
        ).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/treatments/propose", status_code=201)
    def propose(project_id: str, request: ProposeRequest) -> dict[str, Any]:
        treatment = editing.propose(
            project_id,
            scenario=request.scenario,
            asset_id=request.asset_id,
            duration_seconds=request.duration_seconds,
        )
        return treatment.model_dump(mode="json")

    @app.post("/api/workflows/screen/plan", status_code=201)
    def plan_screen_workflow(request: ScreenWorkflowPlanRequest) -> dict[str, Any]:
        if request.reviewed_by is not None and request.beats is None:
            raise ValueError("reviewed_by requires authored beats")
        return editing.plan_screen_workflow(
            title=request.title,
            script=request.script,
            beats=request.beats,
            reviewed_by=request.reviewed_by,
            reference_study_ids=request.reference_study_ids,
        ).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/workflows/screen/voiceover-timing",
        status_code=201,
    )
    def time_screen_workflow_voiceover(
        project_id: str,
        request: ScreenWorkflowVoiceoverTimingRequest,
    ) -> dict[str, Any]:
        return editing.time_screen_workflow_voiceover(
            project_id,
            plan=request.plan,
            asset_id=request.asset_id,
            timeline_frame_rate=request.timeline_frame_rate,
            transcribe=request.transcribe,
        ).model_dump(mode="json")

    @app.post("/api/workflows/screen/voiceover-review", status_code=201)
    def review_screen_workflow_voiceover(
        request: ScreenWorkflowVoiceoverReviewRequest,
    ) -> dict[str, Any]:
        return editing.review_screen_workflow_voiceover(
            plan=request.plan,
            draft=request.draft,
            cues=request.cues,
            reviewed_by=request.reviewed_by,
            review_note=request.review_note,
        ).model_dump(mode="json")

    @app.post(
        "/api/projects/{project_id}/treatments/screen-workflow",
        status_code=201,
    )
    def propose_screen_workflow(
        project_id: str,
        request: ScreenWorkflowTreatmentRequest,
    ) -> dict[str, Any]:
        return editing.propose_screen_workflow(
            project_id,
            plan=request.plan,
            edit_spec=request.edit_spec,
        ).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/treatments/options", status_code=201)
    def propose_options(project_id: str, request: ProposeRequest) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in editing.propose_options(
                project_id,
                scenario=request.scenario,
                asset_id=request.asset_id,
                duration_seconds=request.duration_seconds,
            )
        ]

    @app.post("/api/projects/{project_id}/treatments/{treatment_id}/accept", status_code=201)
    def accept_treatment(project_id: str, treatment_id: str) -> dict[str, Any]:
        return editing.accept_treatment(project_id, treatment_id).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/patches", status_code=201)
    def apply_patch(project_id: str, request: PatchRequest) -> dict[str, Any]:
        patch = EditPatch(
            project_id=project_id,
            base_version_id=request.base_version_id,
            operations=request.operations,
            rationale=request.rationale,
            evidence_refs=request.evidence_refs,
        )
        return editing.apply_patch(project_id, patch, request.message).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/patches/preview-language")
    def preview_language_patch(project_id: str, request: LanguagePatchRequest) -> dict[str, Any]:
        return (
            NaturalLanguagePatchService(store)
            .preview(project_id, request.command, version_id=request.version_id)
            .model_dump(mode="json")
        )

    @app.post("/api/projects/{project_id}/versions/{version_id}/revert", status_code=201)
    def revert(project_id: str, version_id: str) -> dict[str, Any]:
        return editing.revert_version(project_id, version_id).model_dump(mode="json")

    @app.post("/api/projects/{project_id}/versions/{version_id}/render", status_code=201)
    def render(
        project_id: str,
        version_id: str,
        profile: Literal["preview", "final"] = Query(default="preview"),
        start_frame: int | None = Query(default=None, ge=0),
        duration_frames: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        if (start_frame is None) != (duration_frames is None):
            raise HTTPException(
                status_code=422,
                detail="start_frame and duration_frames must be supplied together",
            )
        return (
            RenderService(store)
            .render(
                project_id,
                version_id,
                profile=profile,
                frame_range=(
                    FrameRange(start=start_frame, duration=duration_frames)
                    if start_frame is not None and duration_frames is not None
                    else None
                ),
            )
            .model_dump(mode="json")
        )

    @app.post("/api/projects/{project_id}/versions/{version_id}/qc", status_code=201)
    def qc(
        project_id: str,
        version_id: str,
        profile: Literal["preview", "final"] = Query(default="preview"),
    ) -> dict[str, Any]:
        return (
            QualityService(store)
            .inspect(project_id, version_id, profile=profile)
            .model_dump(mode="json")
        )

    @app.post("/api/projects/{project_id}/versions/{version_id}/export/{format_name}")
    def export(
        project_id: str, version_id: str, format_name: Literal["kdenlive", "otio"]
    ) -> dict[str, Any]:
        exporter = KdenliveExporter(store) if format_name == "kdenlive" else OTIOExporter(store)
        return exporter.export(project_id, version_id).model_dump(mode="json")

    @app.get("/api/projects/{project_id}/file")
    def project_file(project_id: str, path: str) -> FileResponse:
        root = store.project_path(project_id).resolve()
        selected_path = (root / path).resolve()
        if not selected_path.is_relative_to(root) or not selected_path.is_file():
            raise HTTPException(status_code=404, detail="project artifact not found")
        return FileResponse(selected_path)

    return app


def _project_bundle(store: ProjectStore, project_id: str) -> dict[str, Any]:
    project = store.load_project(project_id)
    return {
        "project": project.model_dump(mode="json"),
        "brief_revisions": [
            item.model_dump(mode="json") for item in store.list_brief_revisions(project_id)
        ],
        "assets": [item.model_dump(mode="json") for item in store.list_assets(project_id)],
        "derivatives": [
            item.model_dump(mode="json") for item in store.list_derivatives(project_id)
        ],
        "evidence": [item.model_dump(mode="json") for item in store.list_evidence(project_id)],
        "effective_evidence": [
            item.model_dump(mode="json") for item in store.list_effective_evidence(project_id)
        ],
        "treatments": [item.model_dump(mode="json") for item in store.list_treatments(project_id)],
        "versions": [item.model_dump(mode="json") for item in store.list_versions(project_id)],
        "decision_graphs": [
            item.model_dump(mode="json") for item in store.list_decision_graphs(project_id)
        ],
        "decision_log": store.load_decision_log(project_id).model_dump(mode="json"),
        "jobs": [item.model_dump(mode="json") for item in store.list_jobs(project_id)],
        "video_anatomies": [
            item.model_dump(mode="json") for item in store.list_video_anatomies(project_id)
        ],
        "video_editorial_proposals": [
            item.model_dump(mode="json")
            for item in store.list_editorial_structure_proposals(project_id)
        ],
        "video_reconstruction_proposals": [
            item.model_dump(mode="json")
            for item in store.list_reference_reconstruction_proposals(project_id)
        ],
        "video_proposal_reviews": [
            item.model_dump(mode="json") for item in store.list_video_proposal_reviews(project_id)
        ],
        "video_proposal_acceptances": [
            item.model_dump(mode="json")
            for item in store.list_video_proposal_acceptances(project_id)
        ],
        "qc": [item.model_dump(mode="json") for item in store.list_qc(project_id)],
        "interchange": [
            item.model_dump(mode="json") for item in store.list_interchange(project_id)
        ],
        "reference_workspaces": [
            item.model_dump(mode="json") for item in store.list_reference_workspaces(project_id)
        ],
        "motion_correction_proposals": [
            item.model_dump(mode="json")
            for item in store.list_motion_correction_proposals(project_id)
        ],
        "motion_correction_reviews": [
            item.model_dump(mode="json")
            for item in store.list_motion_correction_reviews(project_id)
        ],
    }
