"""Reference-understanding workbench and human-gated motion corrections."""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Asset,
    EditPatch,
    EvidenceAuthority,
    EvidenceRecord,
    FrameRange,
    MotionCorrectionContextV2,
    MotionCorrectionOperationV2,
    MotionCorrectionProposalItemV2,
    MotionCorrectionProposalSetV2,
    MotionCorrectionReviewDecisionV2,
    MotionCorrectionReviewV2,
    MotionRecoveryFrame,
    MotionRecoveryReport,
    MovePhaseBoundaryCorrectionV2,
    PatchOperation,
    ProjectVersion,
    Provenance,
    ReconstructionStudyPassV2,
    ReferenceComparisonReportV2,
    ReferenceCurveSampleV2,
    ReferenceMotionEvidenceV2,
    ReferenceMotionHumanCorrectionV2,
    ReferenceMotionInterpretationV2,
    ReferenceReconstructionPassReceiptV2,
    ReferenceReconstructionSpecV2,
    ReferenceReconstructionStudyV2,
    ReferenceWorkbenchBundleV2,
    ReferenceWorkbenchSampleV2,
    ReferenceWorkspaceArtifactV2,
    ReferenceWorkspaceRegistrationV2,
    ReferenceWorkspaceVersionV2,
    TransformEffectV2,
    new_id,
)
from aoa_editing.domain.motion_corrections import (
    MotionCorrectionCompileError,
    compile_motion_corrections,
    infer_motion_correction_operations,
    motion_effect_sha256,
)
from aoa_editing.domain.patches import apply_timeline_patch
from aoa_editing.evals.gate import require_readiness
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore, StoreError

ReadinessGuard = Callable[[Settings], dict[str, Any]]
ReferenceArtifactAuthority = Literal[
    "reference_evidence",
    "candidate_render",
    "comparison_diagnostic",
]


class ReferenceWorkspaceError(ValueError):
    """Reference workbench evidence is stale, inconsistent, or outside its owner root."""


class MotionCorrectionReviewError(ValueError):
    """A correction review cannot cross the explicit human/version boundary."""


class ReferenceWorkspaceService:
    """Join frozen evaluation evidence to a project without importing reference media."""

    def __init__(
        self,
        store: ProjectStore,
        *,
        readiness_guard: ReadinessGuard = require_readiness,
    ):
        self.store = store
        self.readiness_guard = readiness_guard

    def attach(
        self,
        project_id: str,
        *,
        study_path: Path,
        reference_media_path: Path | None = None,
    ) -> ReferenceWorkspaceRegistrationV2:
        readiness = self._readiness()
        home = self.store.settings.editing_home.resolve()
        study_file = _home_file(home, study_path)
        study = _read_model(study_file, ReferenceReconstructionStudyV2)
        selected = next(
            item for item in study.passes if item.sequence == study.selected_sequence
        )
        if selected.receipt_path is None:
            raise ReferenceWorkspaceError("selected reconstruction pass has no application receipt")
        if selected.receipt_id is None or selected.receipt_sha256 is None:
            raise ReferenceWorkspaceError(
                "selected reconstruction pass has no sealed receipt identity"
            )
        receipt_file = _recorded_home_file(home, selected.receipt_path)
        if sha256_file(receipt_file) != selected.receipt_sha256:
            raise ReferenceWorkspaceError("selected reconstruction receipt hash changed")
        receipt = _read_model(receipt_file, ReferenceReconstructionPassReceiptV2)
        if receipt.project_id != project_id:
            raise ReferenceWorkspaceError("selected reconstruction belongs to another project")
        project = self.store.load_project(project_id)
        if study.reference_sha256 not in project.sealed_reference_hashes:
            raise ReferenceWorkspaceError("workspace reference hash is not sealed by the project")
        version = self.store.load_version(project_id, receipt.version_id)
        asset = self.store.load_asset(project_id, receipt.asset_id)
        target_path, effect = _find_transform_effect(version, asset.id)

        spec_file = _recorded_home_file(home, study.spec_path)
        evidence_file = spec_file.with_name("reference-motion-evidence-v2.json")
        comparison_file = _recorded_home_file(home, study.final_comparison_path)
        candidate_motion_value = selected.artifacts.get("candidate_motion")
        if candidate_motion_value is None:
            raise ReferenceWorkspaceError("selected pass has no candidate motion evidence")
        candidate_motion_file = _recorded_home_file(home, candidate_motion_value)
        spec = _read_model(spec_file, ReferenceReconstructionSpecV2)
        evidence = _read_model(evidence_file, ReferenceMotionEvidenceV2)
        comparison = _read_model(comparison_file, ReferenceComparisonReportV2)
        candidate_motion = _read_model(candidate_motion_file, MotionRecoveryReport)

        self._validate_join(
            study=study,
            selected=selected,
            selected_receipt=receipt,
            spec=spec,
            evidence=evidence,
            comparison=comparison,
            candidate_motion=candidate_motion,
            study_file=study_file,
            spec_file=spec_file,
            comparison_file=comparison_file,
            candidate_motion_file=candidate_motion_file,
        )
        expected_candidate_file = (
            self.store.project_path(project_id)
            / "renders"
            / receipt.version_id
            / "final"
            / "video.mp4"
        ).resolve(strict=True)
        candidate_file = Path(comparison.candidate_path).resolve(strict=True)
        if (
            asset.sha256 != study.source_sha256
            or candidate_file != expected_candidate_file
            or sha256_file(candidate_file) != study.selected_candidate_sha256
        ):
            raise ReferenceWorkspaceError(
                "selected reconstruction does not match its project asset and final render"
            )
        existing = next(
            (
                item
                for item in self.store.list_reference_workspaces(project_id)
                if item.study_id == study.id
                and item.base_version_id == receipt.version_id
                and item.comparison_id == comparison.id
            ),
            None,
        )
        if reference_media_path is not None:
            self.store.bind_reference_media(study.reference_sha256, reference_media_path)
        if existing is not None:
            return existing

        phase_markers = _phase_markers(evidence)
        artifacts = self._artifacts(
            project_id=project_id,
            evidence=evidence,
            comparison=comparison,
            phase_markers=phase_markers,
        )
        workspace_id = new_id("referenceworkspace")
        baseline = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="reference.motion_interpretation.v2",
            authority=EvidenceAuthority.ANALYZER,
            confidence=evidence.curve_confidence,
            payload={
                "workspace_id": workspace_id,
                "study_id": study.id,
                "reference_evidence_id": evidence.id,
                "comparison_id": comparison.id,
                "phase_markers": phase_markers,
                "pivot_assessment": (
                    "identified"
                    if all(
                        frame.pivot_identifiability != "unidentifiable"
                        for frame in evidence.frames
                    )
                    else "unidentifiable"
                ),
                "leading_explanation": evidence.phase_model.get("channel_coupling", {}),
                "uncertainties": evidence.uncertainties,
                "alternative_explanations": evidence.alternative_explanations,
                "reference_used_as_render_input": False,
            },
            artifacts=[
                artifact.relative_path
                for artifact in artifacts.values()
                if artifact.relative_path is not None
            ],
            time_base=evidence.frame_rate,
            ranges=[FrameRange(start=0, duration=evidence.frame_count)],
            provenance=Provenance(
                tool="aoa-editing",
                tool_version="reference-workspace-v2",
                parameters={
                    "operation": "attach-frozen-reference-understanding",
                    "reference_pixels_persisted_in_project": False,
                },
                deterministic=True,
            ),
        )
        workspace = ReferenceWorkspaceRegistrationV2(
            id=workspace_id,
            project_id=project_id,
            base_version_id=version.id,
            asset_id=asset.id,
            target_effect_path=target_path,
            target_effect_sha256=motion_effect_sha256(effect),
            baseline_evidence_id=baseline.id,
            study_id=study.id,
            study_path=_relative_to_home(home, study_file),
            study_sha256=sha256_file(study_file),
            reference_evidence_id=evidence.id,
            reference_evidence_path=_relative_to_home(home, evidence_file),
            reference_evidence_sha256=sha256_file(evidence_file),
            spec_id=spec.id,
            spec_path=_relative_to_home(home, spec_file),
            spec_sha256=sha256_file(spec_file),
            comparison_id=comparison.id,
            comparison_path=_relative_to_home(home, comparison_file),
            comparison_sha256=sha256_file(comparison_file),
            candidate_motion_id=candidate_motion.id,
            candidate_motion_path=_relative_to_home(home, candidate_motion_file),
            candidate_motion_sha256=sha256_file(candidate_motion_file),
            source_sha256=study.source_sha256,
            reference_sha256=study.reference_sha256,
            candidate_sha256=study.selected_candidate_sha256,
            readiness_revision=str(readiness["git_revision"]),
            frame_count=evidence.frame_count,
            frame_rate=evidence.frame_rate,
            width=evidence.width,
            height=evidence.height,
            phase_markers=phase_markers,
            pivot_identifiable=all(
                frame.pivot_identifiability != "unidentifiable"
                for frame in evidence.frames
            ),
            artifacts=artifacts,
            provenance=Provenance(
                tool="aoa-editing",
                tool_version="reference-workspace-v2",
                parameters={
                    "operation": "register-reference-workspace",
                    "readiness_revision": readiness["git_revision"],
                    "reference_media_copied_into_project": False,
                },
                deterministic=True,
            ),
        )
        self.store.save_evidence(baseline)
        return self.store.save_reference_workspace(workspace)

    def bundle(
        self,
        project_id: str,
        workspace_id: str,
    ) -> ReferenceWorkbenchBundleV2:
        readiness = self._readiness()
        workspace = self.store.load_reference_workspace(project_id, workspace_id)
        home = self.store.settings.editing_home.resolve()
        evidence = _read_model(
            _recorded_home_file(home, workspace.reference_evidence_path),
            ReferenceMotionEvidenceV2,
        )
        candidate = _read_model(
            _recorded_home_file(home, workspace.candidate_motion_path),
            MotionRecoveryReport,
        )
        comparison = _read_model(
            _recorded_home_file(home, workspace.comparison_path),
            ReferenceComparisonReportV2,
        )
        self._verify_workspace_documents(workspace)
        if (
            len(evidence.frames) != len(candidate.frames)
            or len(evidence.frames) != workspace.frame_count
        ):
            raise ReferenceWorkspaceError("reference and candidate curves are not frame-aligned")
        expected_frames = list(range(workspace.frame_count))
        if (
            [frame.frame for frame in evidence.frames] != expected_frames
            or [frame.frame for frame in candidate.frames] != expected_frames
            or evidence.frame_rate != workspace.frame_rate
            or candidate.frame_rate != workspace.frame_rate
            or (evidence.width, evidence.height) != (workspace.width, workspace.height)
            or (candidate.width, candidate.height) != (workspace.width, workspace.height)
        ):
            raise ReferenceWorkspaceError(
                "reference and candidate curves do not share one contiguous time base"
            )
        if any(
            abs(reference_frame.time_seconds - candidate_frame.time_seconds) > 1e-6
            for reference_frame, candidate_frame in zip(
                evidence.frames,
                candidate.frames,
                strict=True,
            )
        ):
            raise ReferenceWorkspaceError(
                "reference and candidate curve sample times are not aligned"
            )
        samples = [
            ReferenceWorkbenchSampleV2(
                frame=reference_frame.frame,
                time_seconds=reference_frame.time_seconds,
                reference=ReferenceCurveSampleV2(
                    center_x=reference_frame.center_x,
                    center_y=reference_frame.center_y,
                    scale=reference_frame.scale_relative_to_contain,
                    rotation_degrees=reference_frame.rotation_degrees,
                    velocity=reference_frame.velocity,
                    acceleration=reference_frame.acceleration,
                    jerk=reference_frame.jerk,
                    confidence=reference_frame.confidence,
                    residual_flow_mean=reference_frame.residual_flow_mean,
                    uncertainty=reference_frame.uncertainty,
                ),
                candidate=_candidate_sample(candidate_frame),
            )
            for reference_frame, candidate_frame in zip(
                evidence.frames,
                candidate.frames,
                strict=True,
            )
        ]
        versions = [
            ReferenceWorkspaceVersionV2(
                id=version.id,
                message=version.message,
                created_at=version.created_at,
                current=version.id == self.store.load_project(project_id).current_version_id,
                preview_available=(
                    self.store.project_path(project_id)
                    / "renders"
                    / version.id
                    / "preview"
                    / "video.mp4"
                ).is_file(),
                final_available=(
                    self.store.project_path(project_id)
                    / "renders"
                    / version.id
                    / "final"
                    / "video.mp4"
                ).is_file(),
            )
            for version in self.store.list_versions(project_id)
        ]
        human_evidence = [
            item
            for item in self.store.list_evidence(project_id)
            if item.kind == "reference.motion_interpretation.v2"
            and item.authority is EvidenceAuthority.HUMAN_CORRECTION
            and item.payload.get("workspace_id") == workspace.id
        ]
        warnings = list(evidence.uncertainties)
        if workspace.readiness_revision != readiness["git_revision"]:
            warnings.append(
                "workspace attachment revision differs from the current passing readiness; "
                "all hashes were revalidated before serving"
            )
        return ReferenceWorkbenchBundleV2(
            workspace=workspace,
            samples=samples,
            interpretations=_interpretations(workspace, evidence),
            versions=versions,
            proposals=self.store.list_motion_correction_proposals(
                project_id,
                workspace_id=workspace.id,
            ),
            reviews=self.store.list_motion_correction_reviews(
                project_id,
                workspace_id=workspace.id,
            ),
            human_correction_evidence_ids=[item.id for item in human_evidence],
            objective_summary={
                "objective_overall": comparison.objective_overall,
                "overall": comparison.overall,
                "technical": comparison.technical,
                "geometric": comparison.geometric,
                "temporal": comparison.temporal,
                "perceptual": comparison.perceptual,
                "checks": [
                    item.model_dump(mode="json") for item in comparison.checks
                ],
                "human_review": (
                    comparison.human_review.model_dump(mode="json")
                    if comparison.human_review is not None
                    else None
                ),
            },
            warnings=warnings,
        )

    def resolve_artifact(
        self,
        project_id: str,
        workspace_id: str,
        role: str,
    ) -> tuple[Path, str]:
        self._readiness()
        workspace = self.store.load_reference_workspace(project_id, workspace_id)
        artifact = workspace.artifacts.get(role)
        if artifact is None:
            raise StoreError(f"unknown reference workspace artifact role: {role}")
        if artifact.scope == "reference_binding":
            path = self.store.resolve_reference_media(workspace.reference_sha256)
        elif artifact.scope == "project":
            if artifact.relative_path is None:  # model invariant
                raise StoreError("project artifact has no relative path")
            root = self.store.project_path(project_id).resolve()
            path = (root / artifact.relative_path).resolve(strict=True)
            if not path.is_relative_to(root):
                raise StoreError("workspace project artifact escapes the project root")
        else:
            if artifact.relative_path is None:  # model invariant
                raise StoreError("editing-home artifact has no relative path")
            root = self.store.settings.editing_home.resolve()
            path = (root / artifact.relative_path).resolve(strict=True)
            if not path.is_relative_to(root):
                raise StoreError("workspace artifact escapes the editing home")
        if not path.is_file() or sha256_file(path) != artifact.sha256:
            raise StoreError(f"workspace artifact is missing or changed: {role}")
        return path, artifact.media_type

    def _readiness(self) -> dict[str, Any]:
        receipt = self.readiness_guard(self.store.settings)
        if receipt.get("overall") != "pass":
            raise ReferenceWorkspaceError("reference workspace requires passing readiness")
        revision = receipt.get("git_revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ReferenceWorkspaceError("readiness guard returned no valid Git revision")
        return receipt

    def _validate_join(
        self,
        *,
        study: ReferenceReconstructionStudyV2,
        selected: ReconstructionStudyPassV2,
        selected_receipt: ReferenceReconstructionPassReceiptV2,
        spec: ReferenceReconstructionSpecV2,
        evidence: ReferenceMotionEvidenceV2,
        comparison: ReferenceComparisonReportV2,
        candidate_motion: MotionRecoveryReport,
        study_file: Path,
        spec_file: Path,
        comparison_file: Path,
        candidate_motion_file: Path,
    ) -> None:
        if (
            selected.disposition != "selected"
            or selected.objective_overall != "pass"
            or selected.comparison_id != comparison.id
            or selected.comparison_sha256 != sha256_file(comparison_file)
            or study.final_comparison_id != selected.comparison_id
            or study.final_comparison_sha256 != selected.comparison_sha256
            or selected.receipt_id != selected_receipt.id
            or selected_receipt.overall != "pass"
            or str(selected_receipt.sequence) != selected.sequence
            or selected_receipt.pass_kind != selected.pass_kind
        ):
            raise ReferenceWorkspaceError(
                "selected reconstruction pass is not a sealed passing application result"
            )
        values = {
            study.source_sha256,
            spec.source_sha256,
            evidence.source_sha256,
            comparison.source_sha256,
            selected_receipt.source_sha256,
        }
        references = {
            study.reference_sha256,
            spec.reference_sha256,
            evidence.reference_sha256,
            comparison.reference_sha256,
            selected_receipt.reference_sha256,
        }
        if len(values) != 1 or len(references) != 1:
            raise ReferenceWorkspaceError("workspace evidence hashes do not join")
        if (
            study.spec_id != spec.id
            or spec.evidence_id != evidence.id
            or study.final_comparison_id != comparison.id
            or comparison.spec_id != spec.id
            or comparison.spec_sha256 != study.spec_sha256
            or selected_receipt.spec_id != spec.id
            or selected_receipt.spec_sha256 != study.spec_sha256
            or _recorded_home_file(
                self.store.settings.editing_home.resolve(),
                selected_receipt.spec_path,
            )
            != spec_file
            or comparison.objective_overall != "pass"
            or comparison.mandatory_skips != 0
            or candidate_motion.video_sha256 != study.selected_candidate_sha256
            or comparison.candidate_sha256 != study.selected_candidate_sha256
            or candidate_motion.source_sha256 != study.source_sha256
            or candidate_motion.frame_count != spec.duration_frames
            or candidate_motion.analyzed_frame_count != spec.duration_frames
            or candidate_motion.frame_rate != spec.frame_rate
            or (candidate_motion.width, candidate_motion.height)
            != (spec.width, spec.height)
            or evidence.frame_count != spec.duration_frames
            or evidence.frame_rate != spec.frame_rate
            or (evidence.width, evidence.height) != (spec.width, spec.height)
            or evidence.readiness_revision != spec.readiness_revision
            or evidence.motion_gate_revision != spec.motion_gate_revision
        ):
            raise ReferenceWorkspaceError("workspace evidence identities do not join")
        selected_comparison_file = _recorded_home_file(
            self.store.settings.editing_home.resolve(),
            selected.comparison_path,
        )
        if selected_comparison_file != comparison_file:
            raise ReferenceWorkspaceError(
                "selected reconstruction comparison path does not join"
            )
        selected_candidate_motion = selected.artifacts.get("candidate_motion")
        if selected_candidate_motion is None or _recorded_home_file(
            self.store.settings.editing_home.resolve(),
            selected_candidate_motion,
        ) != candidate_motion_file:
            raise ReferenceWorkspaceError(
                "selected reconstruction candidate motion path does not join"
            )
        expected = {
            study_file: None,
            spec_file: study.spec_sha256,
            comparison_file: study.final_comparison_sha256,
        }
        for path, expected_hash in expected.items():
            digest = sha256_file(path)
            if expected_hash is not None and digest != expected_hash:
                raise ReferenceWorkspaceError(f"workspace document hash mismatch: {path}")

    def _verify_workspace_documents(
        self,
        workspace: ReferenceWorkspaceRegistrationV2,
    ) -> None:
        home = self.store.settings.editing_home.resolve()
        documents = {
            workspace.study_path: workspace.study_sha256,
            workspace.reference_evidence_path: workspace.reference_evidence_sha256,
            workspace.spec_path: workspace.spec_sha256,
            workspace.comparison_path: workspace.comparison_sha256,
            workspace.candidate_motion_path: workspace.candidate_motion_sha256,
        }
        for relative, expected in documents.items():
            path = _recorded_home_file(home, relative)
            if sha256_file(path) != expected:
                raise ReferenceWorkspaceError(f"workspace document changed: {relative}")

    def _artifacts(
        self,
        *,
        project_id: str,
        evidence: ReferenceMotionEvidenceV2,
        comparison: ReferenceComparisonReportV2,
        phase_markers: dict[str, int],
    ) -> dict[str, ReferenceWorkspaceArtifactV2]:
        home = self.store.settings.editing_home.resolve()
        project_root = self.store.project_path(project_id).resolve()
        candidate_path = Path(comparison.candidate_path).resolve(strict=True)
        if not candidate_path.is_relative_to(project_root):
            raise ReferenceWorkspaceError("candidate video is outside its project root")
        artifacts: dict[str, ReferenceWorkspaceArtifactV2] = {
            "reference_media": ReferenceWorkspaceArtifactV2(
                role="reference_media",
                scope="reference_binding",
                sha256=evidence.reference_sha256,
                media_type="video/mp4",
                authority="reference_evidence",
                reference_derived=True,
            ),
            "candidate_media": _project_artifact(
                "candidate_media",
                candidate_path,
                project_root,
                authority="candidate_render",
                reference_derived=False,
            ),
        }
        evidence_roles = {
            "transform_curves": "position_scale_rotation_plot",
            "velocity": "velocity_plot",
            "acceleration": "acceleration_plot",
            "jerk": "jerk_plot",
            "pivot": "pivot_identifiability_plot",
            "phase_contact_sheet": "phase_contact_sheet",
        }
        for role, key in evidence_roles.items():
            value = evidence.artifacts.get(key)
            if value is None:
                raise ReferenceWorkspaceError(f"reference evidence lacks artifact {key}")
            artifacts[role] = _home_artifact(
                role,
                _recorded_home_file(home, value),
                home,
                authority="reference_evidence",
                reference_derived=True,
            )
        comparison_roles = {
            "side_by_side": "side_by_side",
            "aligned_difference": "aligned_difference",
            "motion_error": "motion_error_plot",
            "derivative_error": "derivative_error_plot",
        }
        for role, key in comparison_roles.items():
            value = comparison.artifacts.get(key)
            if value is None:
                raise ReferenceWorkspaceError(f"comparison lacks artifact {key}")
            artifacts[role] = _home_artifact(
                role,
                _recorded_home_file(home, value),
                home,
                authority="comparison_diagnostic",
                reference_derived=True,
            )
        diagnostics_value = evidence.artifacts.get("diagnostic_overlays")
        if diagnostics_value is None:
            raise ReferenceWorkspaceError("reference evidence lacks frame diagnostics")
        diagnostics = _recorded_home_path(home, diagnostics_value)
        if not diagnostics.is_dir():
            raise ReferenceWorkspaceError("reference diagnostic artifact is not a directory")
        overlays = sorted(diagnostics.glob("frame-*-overlay.*"))
        residuals = sorted(diagnostics.glob("frame-*-residual.*"))
        if not overlays or not residuals:
            raise ReferenceWorkspaceError("reference diagnostic frames are incomplete")
        pivot_frame = phase_markers["twist_peak"]
        selected_overlay = _nearest_frame_artifact(overlays, pivot_frame)
        selected_residual = _nearest_frame_artifact(residuals, pivot_frame)
        artifacts["frame_overlay"] = _home_artifact(
            "frame_overlay",
            selected_overlay,
            home,
            authority="reference_evidence",
            reference_derived=True,
        )
        artifacts["optical_flow_residual"] = _home_artifact(
            "optical_flow_residual",
            selected_residual,
            home,
            authority="reference_evidence",
            reference_derived=True,
        )
        for path in overlays:
            frame = _artifact_frame(path)
            role = f"frame_overlay_{frame:04d}"
            artifacts[role] = _home_artifact(
                role,
                path,
                home,
                authority="reference_evidence",
                reference_derived=True,
            )
        for path in residuals:
            frame = _artifact_frame(path)
            role = f"optical_flow_residual_{frame:04d}"
            artifacts[role] = _home_artifact(
                role,
                path,
                home,
                authority="reference_evidence",
                reference_derived=True,
            )
        return artifacts


class MotionCorrectionService:
    """Preview semantic motion diffs and apply only individually reviewed items."""

    def __init__(
        self,
        store: ProjectStore,
        *,
        readiness_guard: ReadinessGuard = require_readiness,
    ):
        self.store = store
        self.readiness_guard = readiness_guard

    def preview_language(
        self,
        project_id: str,
        workspace_id: str,
        command: str,
        *,
        version_id: str | None = None,
    ) -> MotionCorrectionProposalSetV2:
        workspace = self._workspace(project_id, workspace_id)
        operations = infer_motion_correction_operations(
            command,
            _correction_context(workspace),
        )
        return self.preview_operations(
            project_id,
            workspace_id,
            operations,
            version_id=version_id,
            natural_language_command=command,
        )

    def preview_operations(
        self,
        project_id: str,
        workspace_id: str,
        operations: list[MotionCorrectionOperationV2],
        *,
        version_id: str | None = None,
        natural_language_command: str | None = None,
    ) -> MotionCorrectionProposalSetV2:
        workspace = self._workspace(project_id, workspace_id)
        version = self.store.load_version(project_id, version_id)
        effect = _effect_at_path(version, workspace.target_effect_path)
        asset = self.store.load_asset(project_id, workspace.asset_id)
        width, height = _source_dimensions(asset)
        items: list[MotionCorrectionProposalItemV2] = []
        for operation in operations:
            try:
                _updated, diff = compile_motion_corrections(
                    effect,
                    [operation],
                    source_width=width,
                    source_height=height,
                    pivot_identifiable=workspace.pivot_identifiable,
                    target_path=workspace.target_effect_path,
                )
                items.append(
                    MotionCorrectionProposalItemV2(
                        operation=operation,
                        rationale=_operation_rationale(operation.kind),
                        evidence_refs=[
                            workspace.baseline_evidence_id,
                            workspace.reference_evidence_id,
                            workspace.comparison_id,
                        ],
                        executable=True,
                        diff=diff,
                    )
                )
            except MotionCorrectionCompileError as error:
                items.append(
                    MotionCorrectionProposalItemV2(
                        operation=operation,
                        rationale=_operation_rationale(operation.kind),
                        evidence_refs=[
                            workspace.baseline_evidence_id,
                            workspace.reference_evidence_id,
                            workspace.comparison_id,
                        ],
                        executable=False,
                        blockers=[str(error)],
                    )
                )
        proposal = MotionCorrectionProposalSetV2(
            project_id=project_id,
            workspace_id=workspace.id,
            base_version_id=version.id,
            target_effect_path=workspace.target_effect_path,
            base_effect_sha256=motion_effect_sha256(effect),
            natural_language_command=natural_language_command,
            items=items,
            warnings=[
                "Each correction requires an explicit approve/reject decision.",
                "The reference remains evaluation-only and is never added to render lineage.",
            ],
            provenance=Provenance(
                tool="aoa-editing",
                tool_version="motion-correction-language-v2",
                parameters={
                    "operation": "preview-semantic-motion-corrections",
                    "writes_canonical_state": False,
                    "operation_count": len(operations),
                },
                deterministic=True,
            ),
        )
        return self.store.save_motion_correction_proposal(proposal)

    def review(
        self,
        project_id: str,
        proposal_id: str,
        decisions: list[MotionCorrectionReviewDecisionV2],
    ) -> MotionCorrectionReviewV2:
        self._readiness()
        proposal = self.store.load_motion_correction_proposal(project_id, proposal_id)
        workspace = self.store.load_reference_workspace(project_id, proposal.workspace_id)
        if self.store.list_motion_correction_reviews(
            project_id,
            workspace_id=workspace.id,
        ):
            reviewed = {
                item.proposal_id
                for item in self.store.list_motion_correction_reviews(
                    project_id,
                    workspace_id=workspace.id,
                )
            }
            if proposal.id in reviewed:
                raise MotionCorrectionReviewError("motion proposal was already reviewed")
        expected_ids = [item.id for item in proposal.items]
        decision_by_id = {item.correction_id: item for item in decisions}
        if len(decision_by_id) != len(decisions) or set(decision_by_id) != set(expected_ids):
            raise MotionCorrectionReviewError(
                "review must approve or reject every correction exactly once"
            )
        approved_items = [
            item
            for item in proposal.items
            if decision_by_id[item.id].decision == "approve"
        ]
        blocked_approvals = [item.id for item in approved_items if not item.executable]
        if blocked_approvals:
            raise MotionCorrectionReviewError(
                f"non-executable corrections cannot be approved: {blocked_approvals}"
            )
        project = self.store.load_project(project_id)
        if project.current_version_id != proposal.base_version_id:
            raise MotionCorrectionReviewError(
                "proposal base is stale; create a new preview from the current version"
            )
        version = self.store.load_version(project_id, proposal.base_version_id)
        effect = _effect_at_path(version, proposal.target_effect_path)
        if motion_effect_sha256(effect) != proposal.base_effect_sha256:
            raise MotionCorrectionReviewError("proposal base effect hash changed")
        asset = self.store.load_asset(project_id, workspace.asset_id)
        width, height = _source_dimensions(asset)
        updated = effect
        for item in approved_items:
            updated, _diff = compile_motion_corrections(
                updated,
                [item.operation],
                source_width=width,
                source_height=height,
                pivot_identifiable=workspace.pivot_identifiable,
                target_path=proposal.target_effect_path,
            )

        accepted_ids = [item.id for item in approved_items]
        rejected_ids = [item.id for item in proposal.items if item.id not in accepted_ids]
        phase_markers = dict(workspace.phase_markers)
        for item in approved_items:
            operation = item.operation
            if isinstance(operation, MovePhaseBoundaryCorrectionV2):
                phase_markers[operation.boundary] = (
                    operation.original_frame + operation.delta_frames
                )
        try:
            MotionCorrectionContextV2(
                duration_frames=workspace.frame_count,
                onset_frame=phase_markers["onset"],
                twist_start_frame=phase_markers["twist_start"],
                twist_peak_frame=phase_markers["twist_peak"],
                twist_end_frame=phase_markers["twist_end"],
                settle_frame=phase_markers["settle"],
                pivot_identifiable=workspace.pivot_identifiable,
            )
        except ValueError as error:
            raise MotionCorrectionReviewError(
                "approved phase corrections break canonical phase ordering"
            ) from error
        interpretation = ReferenceMotionHumanCorrectionV2(
            workspace_id=workspace.id,
            proposal_id=proposal.id,
            phase_markers=phase_markers,
            pivot_assessment=(
                "identified" if workspace.pivot_identifiable else "unidentifiable"
            ),
            accepted_correction_ids=accepted_ids,
            rejected_correction_ids=rejected_ids,
            note="; ".join(item.rationale for item in decisions),
        )
        prior = [
            item
            for item in self.store.list_evidence(project_id)
            if item.kind == "reference.motion_interpretation.v2"
            and item.payload.get("workspace_id") == workspace.id
        ]
        prior.sort(key=lambda item: (item.provenance.generated_at, item.id))
        supersedes = [prior[-1].id] if prior else [workspace.baseline_evidence_id]
        human_evidence = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="reference.motion_interpretation.v2",
            authority=EvidenceAuthority.HUMAN_CORRECTION,
            confidence=1.0,
            payload=interpretation.model_dump(mode="json"),
            supersedes=supersedes,
            time_base=workspace.frame_rate,
            ranges=[FrameRange(start=0, duration=workspace.frame_count)],
            provenance=Provenance(
                tool="human-editor",
                tool_version="explicit-motion-correction-review-v2",
                parameters={
                    "proposal_id": proposal.id,
                    "approved_count": len(accepted_ids),
                    "rejected_count": len(rejected_ids),
                },
                deterministic=False,
            ),
        )

        patch: EditPatch | None = None
        if approved_items:
            patch = EditPatch(
                project_id=project_id,
                base_version_id=version.id,
                operations=[
                    PatchOperation(
                        op="replace",
                        path=proposal.target_effect_path,
                        value=updated.model_dump(mode="json"),
                    )
                ],
                rationale=(
                    "Apply individually approved Reference Workspace motion corrections: "
                    + ", ".join(item.operation.kind for item in approved_items)
                ),
                evidence_refs=[
                    workspace.baseline_evidence_id,
                    workspace.reference_evidence_id,
                    workspace.comparison_id,
                    human_evidence.id,
                ],
            )
            apply_timeline_patch(version.timeline, patch.operations)

        self.store.save_evidence(human_evidence)
        created: ProjectVersion | None = None
        if patch is not None:
            created = EditingService(self.store).apply_patch(
                project_id,
                patch,
                "Apply reviewed reference-motion corrections",
            )
        review = MotionCorrectionReviewV2(
            project_id=project_id,
            workspace_id=workspace.id,
            proposal_id=proposal.id,
            base_version_id=proposal.base_version_id,
            decisions=decisions,
            applied_patch_id=patch.id if patch is not None else None,
            created_version_id=created.id if created is not None else None,
            human_evidence_id=human_evidence.id,
            outcome="applied" if created is not None else "rejected",
            provenance=Provenance(
                tool="human-editor",
                tool_version="explicit-motion-correction-review-v2",
                parameters={
                    "all_items_reviewed": True,
                    "canonical_write_via_editing_service": created is not None,
                },
                deterministic=False,
            ),
        )
        return self.store.save_motion_correction_review(review)

    def _workspace(
        self,
        project_id: str,
        workspace_id: str,
    ) -> ReferenceWorkspaceRegistrationV2:
        self._readiness()
        return self.store.load_reference_workspace(project_id, workspace_id)

    def _readiness(self) -> dict[str, Any]:
        receipt = self.readiness_guard(self.store.settings)
        if receipt.get("overall") != "pass":
            raise MotionCorrectionReviewError(
                "motion correction requires passing readiness"
            )
        revision = receipt.get("git_revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise MotionCorrectionReviewError(
                "readiness guard returned no valid Git revision"
            )
        return receipt


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReferenceWorkspaceError(f"cannot load {path}: {error}") from error


def _home_file(home: Path, value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = home / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(home) or not path.is_file():
        raise ReferenceWorkspaceError("reference workspace document is outside editing home")
    return path


def _recorded_home_path(home: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = home / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(home):
        raise ReferenceWorkspaceError("recorded reference artifact is outside editing home")
    return path


def _recorded_home_file(home: Path, value: str) -> Path:
    path = _recorded_home_path(home, value)
    if not path.is_file():
        raise ReferenceWorkspaceError("recorded reference artifact is not a file")
    return path


def _relative_to_home(home: Path, path: Path) -> str:
    return str(path.resolve(strict=True).relative_to(home))


def _find_transform_effect(
    version: ProjectVersion,
    asset_id: str,
) -> tuple[str, TransformEffectV2]:
    matches: list[tuple[str, TransformEffectV2]] = []
    for track_index, track in enumerate(version.timeline.tracks):
        for clip_index, clip in enumerate(track.clips):
            if clip.asset_id != asset_id:
                continue
            for effect_index, effect in enumerate(clip.effects):
                if isinstance(effect, TransformEffectV2):
                    matches.append(
                        (
                            f"/tracks/{track_index}/clips/{clip_index}/effects/{effect_index}",
                            effect,
                        )
                    )
    if len(matches) != 1:
        raise ReferenceWorkspaceError(
            "reference workspace requires exactly one Motion Language v2 source effect"
        )
    return matches[0]


_EFFECT_PATH = re.compile(r"^/tracks/(\d+)/clips/(\d+)/effects/(\d+)$")


def _effect_at_path(version: ProjectVersion, path: str) -> TransformEffectV2:
    match = _EFFECT_PATH.fullmatch(path)
    if match is None:
        raise MotionCorrectionReviewError("workspace effect path is not allowlisted")
    track_index, clip_index, effect_index = (int(item) for item in match.groups())
    try:
        effect = version.timeline.tracks[track_index].clips[clip_index].effects[
            effect_index
        ]
    except IndexError as error:
        raise MotionCorrectionReviewError("workspace effect path is absent") from error
    if not isinstance(effect, TransformEffectV2):
        raise MotionCorrectionReviewError("workspace target is not Motion Language v2")
    return effect


def _phase_markers(evidence: ReferenceMotionEvidenceV2) -> dict[str, int]:
    phase = evidence.phase_model
    coupling = cast(dict[str, Any], phase.get("channel_coupling", {}))
    twist = cast(dict[str, Any], phase.get("twist_candidate", {}))
    markers = {
        "onset": _required_int(phase, "onset_frame", "phase model"),
        "twist_start": _fallback_int(
            twist,
            "start_frame",
            coupling,
            "start_frame",
        ),
        "twist_peak": _fallback_int(
            twist,
            "peak_frame",
            coupling,
            "maximum_phase_offset_frame",
        ),
        "twist_end": _fallback_int(
            twist,
            "end_frame",
            coupling,
            "end_frame",
        ),
        "settle": _required_int(phase, "settle_frame", "phase model"),
    }
    if any(left >= right for left, right in pairwise(markers.values())):
        raise ReferenceWorkspaceError("reference phase markers are not strictly ordered")
    return markers


def _required_int(values: dict[str, Any], key: str, source: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReferenceWorkspaceError(f"{source} lacks integer {key}")
    return value


def _fallback_int(
    preferred: dict[str, Any],
    preferred_key: str,
    fallback: dict[str, Any],
    fallback_key: str,
) -> int:
    if preferred_key in preferred:
        return _required_int(preferred, preferred_key, "twist candidate")
    return _required_int(fallback, fallback_key, "channel coupling")


def _source_dimensions(asset: Asset) -> tuple[int, int]:
    width = asset.metadata.width
    height = asset.metadata.height
    if width is None or height is None:
        raise MotionCorrectionReviewError("source dimensions are unavailable")
    return width, height


def _candidate_sample(frame: MotionRecoveryFrame) -> ReferenceCurveSampleV2:
    values = (
        frame.center_x,
        frame.center_y,
        frame.scale_relative_to_contain,
        frame.rotation_degrees,
    )
    if any(value is None for value in values):
        raise ReferenceWorkspaceError(
            f"candidate motion frame {frame.frame} lacks similarity components"
        )
    return ReferenceCurveSampleV2(
        center_x=cast(float, frame.center_x),
        center_y=cast(float, frame.center_y),
        scale=cast(float, frame.scale_relative_to_contain),
        rotation_degrees=cast(float, frame.rotation_degrees),
        velocity=frame.velocity,
        acceleration=frame.acceleration,
        jerk=frame.jerk,
        confidence=frame.confidence,
        uncertainty=["candidate motion recovery outlier"] if frame.outlier else [],
    )


def _interpretations(
    workspace: ReferenceWorkspaceRegistrationV2,
    evidence: ReferenceMotionEvidenceV2,
) -> list[ReferenceMotionInterpretationV2]:
    coupling = cast(dict[str, Any], evidence.phase_model.get("channel_coupling", {}))
    coupling_confidence = float(coupling.get("confidence", evidence.curve_confidence))
    refs = [workspace.reference_evidence_id, workspace.comparison_id, workspace.study_id]
    return [
        ReferenceMotionInterpretationV2(
            id="staggered-coupled-motion",
            label="Нелинейное связанное движение",
            status="supported",
            explanation=(
                "Scale и rotation синхронны; изогнутый центр при этом запаздывает, "
                "это наблюдаемая основа характерного «подвыверта»."
            ),
            confidence=coupling_confidence,
            evidence_refs=refs,
            editable_operation_kinds=[
                "adjust_tangent",
                "adjust_rotation_coupling",
                "retime_settle",
                "move_phase_boundary",
            ],
        ),
        ReferenceMotionInterpretationV2(
            id="moving-pivot-gauge",
            label="Движущийся pivot",
            status="ambiguous",
            explanation=(
                "Pivot и компенсирующий translation наблюдаемо эквивалентны; "
                "числовой pivot нельзя честно восстановить без авторского ограничения."
            ),
            confidence=1.0,
            evidence_refs=refs,
            editable_operation_kinds=["adjust_pivot_curve"],
        ),
        ReferenceMotionInterpretationV2(
            id="affine-homography",
            label="Affine / perspective",
            status="rejected",
            explanation=(
                "Similarity объясняет все кадры без материального выигрыша "
                "affine или homography."
            ),
            confidence=evidence.curve_confidence,
            evidence_refs=refs,
        ),
        ReferenceMotionInterpretationV2(
            id="local-warp",
            label="Локальный warp",
            status="rejected",
            explanation=(
                "Остаточный flow не требует отдельной деформации и может быть "
                "объяснён codec/resampling."
            ),
            confidence=evidence.curve_confidence,
            evidence_refs=refs,
        ),
    ]


def _correction_context(
    workspace: ReferenceWorkspaceRegistrationV2,
) -> MotionCorrectionContextV2:
    markers = workspace.phase_markers
    return MotionCorrectionContextV2(
        duration_frames=workspace.frame_count,
        onset_frame=markers["onset"],
        twist_start_frame=markers["twist_start"],
        twist_peak_frame=markers["twist_peak"],
        twist_end_frame=markers["twist_end"],
        settle_frame=markers["settle"],
        pivot_identifiable=workspace.pivot_identifiable,
    )


def _operation_rationale(kind: str) -> str:
    return {
        "adjust_tangent": "Soften the local motion derivative around the twist.",
        "move_phase_boundary": "Move one explicit evidence-bounded phase marker.",
        "retime_settle": "Retime the settle tail while preserving range endpoints.",
        "adjust_pivot_curve": "Test an explicit pivot interpretation only if identifiable.",
        "adjust_rotation_coupling": "Adjust rotation timing relative to coupled motion.",
        "change_easing": "Change the bounded curve timing without a renderer-specific command.",
    }[kind]


def _project_artifact(
    role: str,
    path: Path,
    project_root: Path,
    *,
    authority: ReferenceArtifactAuthority,
    reference_derived: bool,
) -> ReferenceWorkspaceArtifactV2:
    return ReferenceWorkspaceArtifactV2(
        role=role,
        scope="project",
        relative_path=str(path.relative_to(project_root)),
        sha256=sha256_file(path),
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        authority=authority,
        reference_derived=reference_derived,
    )


def _home_artifact(
    role: str,
    path: Path,
    home: Path,
    *,
    authority: ReferenceArtifactAuthority,
    reference_derived: bool,
) -> ReferenceWorkspaceArtifactV2:
    return ReferenceWorkspaceArtifactV2(
        role=role,
        scope="editing_home",
        relative_path=_relative_to_home(home, path),
        sha256=sha256_file(path),
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        authority=authority,
        reference_derived=reference_derived,
    )


def _artifact_frame(path: Path) -> int:
    match = re.search(r"frame-(\d+)-", path.name)
    if match is None:
        raise ReferenceWorkspaceError(f"diagnostic artifact has no frame index: {path}")
    return int(match.group(1))


def _nearest_frame_artifact(paths: list[Path], target: int) -> Path:
    return min(paths, key=lambda path: (abs(_artifact_frame(path) - target), path.name))
