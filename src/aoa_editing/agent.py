"""Allowlisted line-delimited JSON agent protocol over the application core."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

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
    FrameRange,
    FrameRate,
    Intent,
    MotionCorrectionOperationV2,
    MotionCorrectionReviewDecisionV2,
    ReferenceReconstructionSpecV2,
    Scenario,
    ScreenWorkflowBeatInput,
    ScreenWorkflowEditSpec,
    ScreenWorkflowPlan,
    ScreenWorkflowVoiceoverCue,
    ScreenWorkflowVoiceoverTiming,
    VideoAnatomy,
    VideoAnatomyProfile,
)
from aoa_editing.evals.workflow_reference import run_reference_workflow_study
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.providers.service import ProviderService
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class AgentProtocol:
    def __init__(
        self,
        settings: Settings,
        *,
        readiness_guard: ReadinessGuard | None = None,
    ):
        self.store = ProjectStore(settings)
        self.editing = EditingService(self.store)
        self.providers = ProviderService(settings)
        self.video_jobs = VideoAnatomyJobService(self.store, providers=self.providers)
        self.video_proposals = VideoProposalService(self.store)
        self.reference_workspaces = (
            ReferenceWorkspaceService(self.store, readiness_guard=readiness_guard)
            if readiness_guard is not None
            else ReferenceWorkspaceService(self.store)
        )
        self.motion_corrections = (
            MotionCorrectionService(self.store, readiness_guard=readiness_guard)
            if readiness_guard is not None
            else MotionCorrectionService(self.store)
        )
        self.methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "project.list": self._project_list,
            "project.create": self._project_create,
            "project.show": self._project_show,
            "project.revise_brief": self._project_revise_brief,
            "asset.ingest": self._asset_ingest,
            "asset.analyze": self._asset_analyze,
            "video_anatomy.run": self._video_anatomy_run,
            "video_anatomy.estimate": self._video_anatomy_estimate,
            "video_anatomy.show": self._video_anatomy_show,
            "video_anatomy.shots": self._video_anatomy_shots,
            "video_anatomy.deepen_shot": self._video_anatomy_deepen_shot,
            "video_anatomy.focused_plans": self._video_anatomy_focused_plans,
            "video_anatomy.contact_sheet": self._video_anatomy_contact_sheet,
            "video_anatomy.job": self._video_anatomy_job,
            "video_anatomy.cancel": self._video_anatomy_cancel,
            "video_anatomy.retry": self._video_anatomy_retry,
            "video_anatomy.prune_cache": self._video_anatomy_prune_cache,
            "video_anatomy.editorial_proposal": self._video_editorial_proposal,
            "video_anatomy.reconstruction_proposal": (self._video_reconstruction_proposal),
            "video_anatomy.review_reconstruction": (self._video_review_reconstruction),
            "video_anatomy.accept_reconstruction": (self._video_accept_reconstruction),
            "evidence.correct": self._evidence_correct,
            "style.confirm": self._style_confirm,
            "style.list": self._style_list,
            "treatment.propose": self._treatment_propose,
            "treatment.options": self._treatment_options,
            "treatment.accept": self._treatment_accept,
            "workflow.plan_script": self._workflow_plan_script,
            "workflow.time_voiceover": self._workflow_time_voiceover,
            "workflow.review_voiceover": self._workflow_review_voiceover,
            "workflow.propose_treatment": self._workflow_propose_treatment,
            "reference.workflow_study.run": self._reference_workflow_study_run,
            "patch.preview_language": self._patch_preview_language,
            "reference.workspace.list": self._reference_workspace_list,
            "reference.workspace.attach": self._reference_workspace_attach,
            "reference.workspace.show": self._reference_workspace_show,
            "reference.correction.preview_language": (self._reference_correction_preview_language),
            "reference.correction.preview_operations": (
                self._reference_correction_preview_operations
            ),
            "reference.correction.review": self._reference_correction_review,
            "version.revert": self._version_revert,
            "render.run": self._render,
            "render.qc": self._qc,
            "export.kdenlive": self._kdenlive,
            "export.otio": self._otio,
            "provider.status": self._provider_status,
            "provider.invoke": self._provider_invoke,
        }

    def execute(self, request: AgentRequest) -> dict[str, Any]:
        method = self.methods.get(request.method)
        if method is None:
            return {
                "id": request.id,
                "ok": False,
                "error": {"code": "method_not_allowed", "message": request.method},
            }
        try:
            result = method(request.params)
            if isinstance(result, BaseModel):
                result = result.model_dump(mode="json")
            elif isinstance(result, list):
                result = [
                    item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                    for item in result
                ]
            return {"id": request.id, "ok": True, "result": result}
        except Exception as error:
            return {
                "id": request.id,
                "ok": False,
                "error": {"code": type(error).__name__, "message": str(error)},
            }

    def _project_list(self, _params: dict[str, Any]) -> list[Any]:
        return self.store.list_projects()

    def _project_create(self, params: dict[str, Any]) -> Any:
        return self.editing.create_project(
            str(params["name"]),
            Intent(
                text=str(params["intent"]),
                scenario=Scenario(str(params["scenario"])),
                target_duration_seconds=params.get("target_duration_seconds"),
            ),
        )

    def _project_show(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = str(params["project_id"])
        return {
            "project": self.store.load_project(project_id).model_dump(mode="json"),
            "assets": [item.model_dump(mode="json") for item in self.store.list_assets(project_id)],
            "versions": [
                item.model_dump(mode="json") for item in self.store.list_versions(project_id)
            ],
            "brief_revisions": [
                item.model_dump(mode="json") for item in self.store.list_brief_revisions(project_id)
            ],
            "reference_workspaces": [
                item.model_dump(mode="json")
                for item in self.store.list_reference_workspaces(project_id)
            ],
        }

    def _project_revise_brief(self, params: dict[str, Any]) -> Any:
        project = self.store.load_project(str(params["project_id"]))
        intent_payload = {**project.intent.model_dump(mode="json"), **params["intent"]}
        return self.editing.revise_brief(
            project.id,
            Intent.model_validate(intent_payload),
            str(params["rationale"]),
        )

    def _asset_ingest(self, params: dict[str, Any]) -> list[Any]:
        return list(self.editing.ingest(str(params["project_id"]), Path(str(params["source"]))))

    def _asset_analyze(self, params: dict[str, Any]) -> list[Any]:
        return AnalysisService(self.store).analyze(
            str(params["project_id"]),
            str(params["asset_id"]),
            transcribe=bool(params.get("transcribe", False)),
        )

    def _video_anatomy_run(self, params: dict[str, Any]) -> dict[str, Any]:
        frame_range = self._video_frame_range(params)
        result = self.video_jobs.run(
            str(params["project_id"]),
            str(params["asset_id"]),
            profile=VideoAnatomyProfile(str(params.get("profile", "structural"))),
            analysis_range=frame_range,
            pinned_frames=tuple(sorted({int(item) for item in params.get("pinned_frames", [])})),
            explicit_provider_opt_in=bool(params.get("explicit_provider_opt_in", False)),
        )
        return {
            "job": result.receipt.model_dump(mode="json"),
            "anatomy": self._anatomy_projection(result.anatomy),
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

    def _video_anatomy_estimate(self, params: dict[str, Any]) -> Any:
        return self.video_jobs.estimate(
            str(params["project_id"]),
            str(params["asset_id"]),
            profile=VideoAnatomyProfile(str(params.get("profile", "structural"))),
            analysis_range=self._video_frame_range(params),
        )

    def _video_anatomy_show(self, params: dict[str, Any]) -> dict[str, Any]:
        anatomy = self.store.load_video_anatomy(
            str(params["project_id"]),
            str(params["plan_id"]),
        )
        if bool(params.get("include_children", False)):
            return anatomy.model_dump(mode="json")
        return self._anatomy_projection(anatomy)

    def _video_anatomy_shots(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        anatomy = self.store.load_video_anatomy(
            str(params["project_id"]),
            str(params["plan_id"]),
        )
        selected = anatomy.structure.shots
        if params.get("shot_id") is not None:
            selected = [item for item in selected if item.id == str(params["shot_id"])]
        start = int(params.get("start_frame", anatomy.plan.analysis_range.start))
        end = int(params.get("end_frame", anatomy.plan.analysis_range.end))
        motion = {item.shot_id: item for item in anatomy.motion_evidence}
        visual = {item.shot_id: item for item in anatomy.visual_observations}
        return [
            {
                "shot": item.model_dump(mode="json"),
                "transition_in": next(
                    (
                        transition.model_dump(mode="json")
                        for transition in anatomy.structure.transitions
                        if abs(transition.frame - item.frame_range.start) <= 1
                    ),
                    None,
                ),
                "visual_observation": (
                    visual[item.id].model_dump(mode="json") if item.id in visual else None
                ),
                "motion_evidence": (
                    motion[item.id].model_dump(mode="json") if item.id in motion else None
                ),
            }
            for item in selected
            if item.frame_range.end > start and item.frame_range.start < end
        ]

    def _video_anatomy_deepen_shot(self, params: dict[str, Any]) -> dict[str, Any]:
        anatomy = self.store.load_video_anatomy(
            str(params["project_id"]),
            str(params["plan_id"]),
        )
        shot_id = str(params["shot_id"])
        shot = next((item for item in anatomy.structure.shots if item.id == shot_id), None)
        if shot is None:
            raise ValueError(f"unknown shot: {shot_id}")
        result = self.video_jobs.run(
            anatomy.project_id,
            anatomy.asset_id,
            profile=VideoAnatomyProfile.MOTION,
            analysis_range=shot.frame_range,
            pinned_frames=(shot.frame_range.start, shot.frame_range.end - 1),
        )
        return {
            "job": result.receipt.model_dump(mode="json"),
            "anatomy": self._anatomy_projection(result.anatomy),
        }

    def _video_anatomy_focused_plans(self, params: dict[str, Any]) -> list[Any]:
        anatomy = self.store.load_video_anatomy(
            str(params["project_id"]),
            str(params["plan_id"]),
        )
        return VideoAnatomyPipelineService(self.store).create_focused_plans(
            anatomy,
            maximum_plans=int(params.get("maximum_plans", 8)),
        )

    def _video_anatomy_contact_sheet(self, params: dict[str, Any]) -> dict[str, Any]:
        project_id = str(params["project_id"])
        plan_id = str(params["plan_id"])
        path = self.store.video_anatomy_path(project_id, plan_id) / "contact-sheet.jpg"
        if not path.is_file():
            raise ValueError("contact sheet not found")
        return {
            "artifact_path": str(path.relative_to(self.store.project_path(project_id))),
            "sha256": sha256_file(path),
        }

    def _video_anatomy_job(self, params: dict[str, Any]) -> Any:
        return self.store.load_job(str(params["project_id"]), str(params["job_id"]))

    def _video_anatomy_cancel(self, params: dict[str, Any]) -> Any:
        return self.video_jobs.cancel(
            str(params["project_id"]),
            str(params["job_id"]),
        )

    def _video_anatomy_retry(self, params: dict[str, Any]) -> dict[str, Any]:
        result = self.video_jobs.retry(
            str(params["project_id"]),
            str(params["job_id"]),
        )
        return {
            "job": result.receipt.model_dump(mode="json"),
            "anatomy": self._anatomy_projection(result.anatomy),
        }

    def _video_anatomy_prune_cache(self, params: dict[str, Any]) -> Any:
        return self.video_jobs.prune_rebuildable_cache(
            dry_run=not bool(params.get("apply", False))
        )

    def _video_editorial_proposal(self, params: dict[str, Any]) -> Any:
        anatomy = self.store.load_video_anatomy(
            str(params["project_id"]),
            str(params["plan_id"]),
        )
        return self.video_proposals.create_editorial_structure(anatomy)

    def _video_reconstruction_proposal(self, params: dict[str, Any]) -> Any:
        anatomy = self.store.load_video_anatomy(
            str(params["project_id"]),
            str(params["plan_id"]),
        )
        raw_spec = params.get("precise_motion_spec")
        precise_spec = (
            ReferenceReconstructionSpecV2.model_validate(raw_spec)
            if raw_spec is not None
            else None
        )
        precise_ref = (
            str(params["precise_motion_spec_ref"])
            if params.get("precise_motion_spec_ref") is not None
            else None
        )
        if (precise_spec is None) != (precise_ref is None):
            raise ValueError("precise motion spec and immutable ref must appear together")
        return self.video_proposals.create_reference_reconstruction(
            anatomy,
            target_asset_id=(
                str(params["target_asset_id"])
                if params.get("target_asset_id") is not None
                else None
            ),
            base_version_id=(
                str(params["base_version_id"])
                if params.get("base_version_id") is not None
                else None
            ),
            precise_motion_spec=precise_spec,
            precise_motion_spec_ref=precise_ref,
        )

    def _video_review_reconstruction(self, params: dict[str, Any]) -> Any:
        return self.video_proposals.review_reconstruction(
            str(params["project_id"]),
            str(params["proposal_id"]),
            decision=str(params["decision"]),
            reviewer=str(params["reviewer"]),
            rationale=str(params["rationale"]),
        )

    def _video_accept_reconstruction(self, params: dict[str, Any]) -> Any:
        return self.video_proposals.accept_reconstruction(
            str(params["project_id"]),
            str(params["proposal_id"]),
            str(params["review_id"]),
        )

    @staticmethod
    def _video_frame_range(params: dict[str, Any]) -> FrameRange | None:
        start = params.get("start_frame")
        duration = params.get("duration_frames")
        if start is None and duration is None:
            return None
        if start is None or duration is None:
            raise ValueError("start_frame and duration_frames must appear together")
        return FrameRange(start=int(start), duration=int(duration))

    @staticmethod
    def _anatomy_projection(anatomy: VideoAnatomy) -> dict[str, Any]:
        return {
            "id": anatomy.id,
            "anatomy_sha256": anatomy.anatomy_sha256,
            "project_id": anatomy.project_id,
            "asset_id": anatomy.asset_id,
            "source_sha256": anatomy.source_sha256,
            "plan_id": anatomy.plan.id,
            "plan_sha256": anatomy.plan.plan_sha256,
            "profile": anatomy.profile.value,
            "status": anatomy.status.value,
            "shot_count": len(anatomy.structure.shots),
            "transition_count": len(anatomy.structure.transitions),
            "visual_observation_count": len(anatomy.visual_observations),
            "motion_evidence_count": len(anatomy.motion_evidence),
            "coverage_matrix": anatomy.coverage_matrix,
            "confidence_summary": anatomy.confidence_summary,
            "unresolved_ranges": [
                item.model_dump(mode="json") for item in anatomy.unresolved_ranges
            ],
            "incompleteness_reasons": anatomy.incompleteness_reasons,
            "child_refs": anatomy.evidence_refs,
        }

    def _evidence_correct(self, params: dict[str, Any]) -> Any:
        return self.editing.correct_evidence(
            str(params["project_id"]),
            asset_id=str(params["asset_id"]),
            kind=str(params["kind"]),
            payload=dict(params["payload"]),
            supersedes=[str(item) for item in params["supersedes"]],
            rationale=str(params["rationale"]),
            confidence=float(params.get("confidence", 1.0)),
        )

    def _style_confirm(self, params: dict[str, Any]) -> Any:
        if params.get("explicit_opt_in") is not True:
            raise ValueError("style memory requires explicit_opt_in=true")
        return self.editing.confirm_style(
            scope=str(params["scope"]),
            confirmations=[
                (str(item["preference"]), item["value"]) for item in params["confirmations"]
            ],
        )

    def _style_list(self, _params: dict[str, Any]) -> list[Any]:
        return self.store.list_style_profiles()

    def _treatment_propose(self, params: dict[str, Any]) -> Any:
        scenario = params.get("scenario")
        return self.editing.propose(
            str(params["project_id"]),
            scenario=Scenario(str(scenario)) if scenario else None,
            asset_id=str(params["asset_id"]) if params.get("asset_id") else None,
            duration_seconds=(
                float(params["duration_seconds"])
                if params.get("duration_seconds") is not None
                else None
            ),
        )

    def _treatment_accept(self, params: dict[str, Any]) -> Any:
        return self.editing.accept_treatment(str(params["project_id"]), str(params["treatment_id"]))

    def _workflow_plan_script(self, params: dict[str, Any]) -> Any:
        beats_payload = params.get("beats")
        beats = (
            TypeAdapter(list[ScreenWorkflowBeatInput]).validate_python(beats_payload)
            if beats_payload is not None
            else None
        )
        reviewed_by = str(params["reviewed_by"]) if params.get("reviewed_by") is not None else None
        if reviewed_by is not None and beats is None:
            raise ValueError("reviewed_by requires authored beats")
        return self.editing.plan_screen_workflow(
            title=str(params["title"]),
            script=str(params["script"]),
            beats=beats,
            reviewed_by=reviewed_by,
            reference_study_ids=[str(item) for item in params.get("reference_study_ids", [])],
        )

    def _workflow_propose_treatment(self, params: dict[str, Any]) -> Any:
        return self.editing.propose_screen_workflow(
            str(params["project_id"]),
            plan=ScreenWorkflowPlan.model_validate(params["plan"]),
            edit_spec=ScreenWorkflowEditSpec.model_validate(params["edit_spec"]),
        )

    def _workflow_time_voiceover(self, params: dict[str, Any]) -> Any:
        return self.editing.time_screen_workflow_voiceover(
            str(params["project_id"]),
            plan=ScreenWorkflowPlan.model_validate(params["plan"]),
            asset_id=str(params["asset_id"]),
            timeline_frame_rate=FrameRate.model_validate(params.get("timeline_frame_rate", {})),
            transcribe=bool(params.get("transcribe", True)),
        )

    def _workflow_review_voiceover(self, params: dict[str, Any]) -> Any:
        return self.editing.review_screen_workflow_voiceover(
            plan=ScreenWorkflowPlan.model_validate(params["plan"]),
            draft=ScreenWorkflowVoiceoverTiming.model_validate(params["draft"]),
            cues=TypeAdapter(list[ScreenWorkflowVoiceoverCue]).validate_python(params["cues"]),
            reviewed_by=str(params["reviewed_by"]),
            review_note=str(params["review_note"]),
        )

    def _reference_workflow_study_run(self, params: dict[str, Any]) -> Any:
        return run_reference_workflow_study(
            Path(str(params["plan_path"])),
            Path(str(params["output"])),
            settings=self.store.settings,
            readiness_guard=self.reference_workspaces.readiness_guard,
        )

    def _treatment_options(self, params: dict[str, Any]) -> Any:
        scenario = params.get("scenario")
        return self.editing.propose_options(
            str(params["project_id"]),
            scenario=Scenario(str(scenario)) if scenario else None,
            asset_id=str(params["asset_id"]) if params.get("asset_id") else None,
            duration_seconds=(
                float(params["duration_seconds"])
                if params.get("duration_seconds") is not None
                else None
            ),
        )

    def _patch_preview_language(self, params: dict[str, Any]) -> Any:
        return NaturalLanguagePatchService(self.store).preview(
            str(params["project_id"]),
            str(params["command"]),
            version_id=str(params["version_id"]) if params.get("version_id") else None,
        )

    def _provider_status(self, params: dict[str, Any]) -> Any:
        return self.providers.inspect_bindings(probe=bool(params.get("probe", True)))

    def _provider_invoke(self, params: dict[str, Any]) -> Any:
        return self.providers.invoke(
            str(params["alias"]),
            dict(params["payload"]),
            privacy_mode=str(params.get("privacy_mode", "local-only")),
            explicit_opt_in=bool(params.get("explicit_opt_in", False)),
            allow_fallback=bool(params.get("allow_fallback", True)),
            project_id=(
                str(params["project_id"]) if params.get("project_id") is not None else None
            ),
            asset_id=(str(params["asset_id"]) if params.get("asset_id") is not None else None),
        )

    def _reference_workspace_list(self, params: dict[str, Any]) -> list[Any]:
        return self.store.list_reference_workspaces(str(params["project_id"]))

    def _reference_workspace_attach(self, params: dict[str, Any]) -> Any:
        return self.reference_workspaces.attach(
            str(params["project_id"]),
            study_path=Path(str(params["study_path"])),
            reference_media_path=(
                Path(str(params["reference_media_path"]))
                if params.get("reference_media_path")
                else None
            ),
        )

    def _reference_workspace_show(self, params: dict[str, Any]) -> Any:
        return self.reference_workspaces.bundle(
            str(params["project_id"]),
            str(params["workspace_id"]),
        )

    def _reference_correction_preview_language(
        self,
        params: dict[str, Any],
    ) -> Any:
        return self.motion_corrections.preview_language(
            str(params["project_id"]),
            str(params["workspace_id"]),
            str(params["command"]),
            version_id=(str(params["version_id"]) if params.get("version_id") else None),
        )

    def _reference_correction_preview_operations(
        self,
        params: dict[str, Any],
    ) -> Any:
        operations = TypeAdapter(list[MotionCorrectionOperationV2]).validate_python(
            params["operations"]
        )
        return self.motion_corrections.preview_operations(
            str(params["project_id"]),
            str(params["workspace_id"]),
            operations,
            version_id=(str(params["version_id"]) if params.get("version_id") else None),
        )

    def _reference_correction_review(self, params: dict[str, Any]) -> Any:
        decisions = TypeAdapter(list[MotionCorrectionReviewDecisionV2]).validate_python(
            params["decisions"]
        )
        return self.motion_corrections.review(
            str(params["project_id"]),
            str(params["proposal_id"]),
            decisions,
        )

    def _version_revert(self, params: dict[str, Any]) -> Any:
        return self.editing.revert_version(str(params["project_id"]), str(params["version_id"]))

    def _render(self, params: dict[str, Any]) -> Any:
        frame_range = None
        if params.get("start_frame") is not None or params.get("duration_frames") is not None:
            frame_range = FrameRange(
                start=int(params["start_frame"]),
                duration=int(params["duration_frames"]),
            )
        return RenderService(self.store).render(
            str(params["project_id"]),
            str(params["version_id"]) if params.get("version_id") else None,
            profile=str(params.get("profile", "preview")),
            frame_range=frame_range,
        )

    def _qc(self, params: dict[str, Any]) -> Any:
        return QualityService(self.store).inspect(
            str(params["project_id"]),
            str(params["version_id"]) if params.get("version_id") else None,
            profile=str(params.get("profile", "preview")),
        )

    def _kdenlive(self, params: dict[str, Any]) -> Any:
        return KdenliveExporter(self.store).export(
            str(params["project_id"]),
            str(params["version_id"]) if params.get("version_id") else None,
        )

    def _otio(self, params: dict[str, Any]) -> Any:
        return OTIOExporter(self.store).export(
            str(params["project_id"]),
            str(params["version_id"]) if params.get("version_id") else None,
        )


def run_stdio(
    settings: Settings | None = None,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    protocol = AgentProtocol(settings or Settings.from_env())
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = AgentRequest.model_validate_json(line)
            response = protocol.execute(request)
        except (ValidationError, json.JSONDecodeError) as error:
            response = {
                "id": None,
                "ok": False,
                "error": {"code": "invalid_request", "message": str(error)},
            }
        output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        output_stream.flush()
