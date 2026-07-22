"""Allowlisted line-delimited JSON agent protocol over the application core."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.language import NaturalLanguagePatchService
from aoa_editing.application.reference_workspace import (
    MotionCorrectionService,
    ReadinessGuard,
    ReferenceWorkspaceService,
)
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    FrameRange,
    FrameRate,
    Intent,
    MotionCorrectionOperationV2,
    MotionCorrectionReviewDecisionV2,
    Scenario,
    ScreenWorkflowBeatInput,
    ScreenWorkflowEditSpec,
    ScreenWorkflowPlan,
    ScreenWorkflowVoiceoverCue,
    ScreenWorkflowVoiceoverTiming,
)
from aoa_editing.evals.workflow_reference import run_reference_workflow_study
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
            "reference.correction.preview_language": (
                self._reference_correction_preview_language
            ),
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
            "assets": [
                item.model_dump(mode="json") for item in self.store.list_assets(project_id)
            ],
            "versions": [
                item.model_dump(mode="json") for item in self.store.list_versions(project_id)
            ],
            "brief_revisions": [
                item.model_dump(mode="json")
                for item in self.store.list_brief_revisions(project_id)
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
        return list(
            self.editing.ingest(str(params["project_id"]), Path(str(params["source"])))
        )

    def _asset_analyze(self, params: dict[str, Any]) -> list[Any]:
        return AnalysisService(self.store).analyze(
            str(params["project_id"]),
            str(params["asset_id"]),
            transcribe=bool(params.get("transcribe", False)),
        )

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
                (str(item["preference"]), item["value"])
                for item in params["confirmations"]
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
        return self.editing.accept_treatment(
            str(params["project_id"]), str(params["treatment_id"])
        )

    def _workflow_plan_script(self, params: dict[str, Any]) -> Any:
        beats_payload = params.get("beats")
        beats = (
            TypeAdapter(list[ScreenWorkflowBeatInput]).validate_python(beats_payload)
            if beats_payload is not None
            else None
        )
        reviewed_by = (
            str(params["reviewed_by"]) if params.get("reviewed_by") is not None else None
        )
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
            timeline_frame_rate=FrameRate.model_validate(
                params.get("timeline_frame_rate", {})
            ),
            transcribe=bool(params.get("transcribe", True)),
        )

    def _workflow_review_voiceover(self, params: dict[str, Any]) -> Any:
        return self.editing.review_screen_workflow_voiceover(
            plan=ScreenWorkflowPlan.model_validate(params["plan"]),
            draft=ScreenWorkflowVoiceoverTiming.model_validate(params["draft"]),
            cues=TypeAdapter(list[ScreenWorkflowVoiceoverCue]).validate_python(
                params["cues"]
            ),
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
            asset_id=(
                str(params["asset_id"]) if params.get("asset_id") is not None else None
            ),
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
            version_id=(
                str(params["version_id"]) if params.get("version_id") else None
            ),
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
            version_id=(
                str(params["version_id"]) if params.get("version_id") else None
            ),
        )

    def _reference_correction_review(self, params: dict[str, Any]) -> Any:
        decisions = TypeAdapter(
            list[MotionCorrectionReviewDecisionV2]
        ).validate_python(params["decisions"])
        return self.motion_corrections.review(
            str(params["project_id"]),
            str(params["proposal_id"]),
            decisions,
        )

    def _version_revert(self, params: dict[str, Any]) -> Any:
        return self.editing.revert_version(
            str(params["project_id"]), str(params["version_id"])
        )

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
