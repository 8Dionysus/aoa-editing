"""Application service shared by CLI, HTTP, and agent transports."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from aoa_editing import __version__
from aoa_editing.domain.decisions import (
    approve_graph,
    approved_patch_graph,
    graph_events,
)
from aoa_editing.domain.models import (
    Asset,
    DecisionOrigin,
    EditorialBriefRevision,
    EditPatch,
    EvidenceAuthority,
    EvidenceRecord,
    FrameRange,
    FrameRate,
    Intent,
    ProjectManifest,
    ProjectVersion,
    Provenance,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionPassKindV2,
    ReferenceReconstructionSpec,
    ReferenceReconstructionSpecV2,
    Scenario,
    ScreenWorkflowBeatInput,
    ScreenWorkflowEditSpec,
    ScreenWorkflowPlan,
    ScreenWorkflowVoiceoverCue,
    ScreenWorkflowVoiceoverTiming,
    StyleConfirmation,
    StyleProfile,
    TechniqueCompositionEvidenceV2,
    TechniquePacket,
    TechniqueProposalResultV2,
    Timeline,
    Treatment,
    new_id,
)
from aoa_editing.domain.patches import apply_timeline_patch
from aoa_editing.infrastructure.derivatives import MediaDerivativeService
from aoa_editing.infrastructure.store import ProjectStore


class EditingService:
    def __init__(self, store: ProjectStore):
        self.store = store

    def create_project(self, name: str, intent: Intent) -> ProjectManifest:
        return self.store.create_project(name, intent)

    def revise_brief(
        self, project_id: str, intent: Intent, rationale: str
    ) -> EditorialBriefRevision:
        return self.store.revise_brief(project_id, intent, rationale)

    def confirm_style(
        self,
        *,
        scope: str,
        confirmations: list[tuple[str, object]],
    ) -> StyleProfile:
        """Persist only preferences provided through an explicit confirmation action."""

        profile = StyleProfile(
            scope=scope,
            explicit_opt_in=True,
            confirmations=[
                StyleConfirmation(preference=preference, value=value)
                for preference, value in confirmations
            ],
        )
        self.store.save_style_profile(profile)
        return profile

    def seal_reference(self, project_id: str, reference_sha256: str) -> ProjectManifest:
        """Forbid a comparison source before any project ingest or render can see it."""

        manifest = self.store.load_project(project_id)
        if reference_sha256 in manifest.sealed_reference_hashes:
            return manifest
        updated = manifest.model_copy(
            update={
                "sealed_reference_hashes": [
                    *manifest.sealed_reference_hashes,
                    reference_sha256,
                ],
                "updated_at": datetime.now(UTC),
            }
        )
        self.store.save_project(updated)
        return updated

    def ingest(self, project_id: str, source: Path) -> tuple[Asset, EvidenceRecord]:
        asset, evidence = self.store.ingest_asset(project_id, source)
        MediaDerivativeService(self.store).derive(project_id, asset.id)
        return asset, evidence

    def correct_evidence(
        self,
        project_id: str,
        *,
        asset_id: str,
        kind: str,
        payload: dict[str, object],
        supersedes: list[str],
        rationale: str,
        confidence: float = 1.0,
    ) -> EvidenceRecord:
        """Persist an explicit human correction without erasing analyzer history."""

        asset = self.store.load_asset(project_id, asset_id)
        if not supersedes:
            raise ValueError("a human correction must name evidence it supersedes")
        existing = {item.id: item for item in self.store.list_evidence(project_id)}
        original = existing.get(supersedes[0])
        if original is None:
            raise ValueError(f"superseded evidence does not exist: {supersedes[0]}")
        correction = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind=kind,
            authority=EvidenceAuthority.HUMAN_CORRECTION,
            confidence=confidence,
            payload=payload,
            supersedes=supersedes,
            time_base=original.time_base,
            ranges=original.ranges,
            provenance=Provenance(
                tool="human-editor",
                tool_version="explicit-confirmation",
                parameters={"rationale": rationale},
                deterministic=False,
            ),
        )
        self.store.save_evidence(correction)
        return correction

    def initialize_timeline(
        self,
        project_id: str,
        *,
        duration_frames: int,
        width: int = 1920,
        height: int = 1080,
        frame_rate: FrameRate | None = None,
    ) -> ProjectVersion:
        timeline = Timeline(
            width=width,
            height=height,
            frame_rate=frame_rate or FrameRate(),
            duration_frames=duration_frames,
        )
        return self.store.create_initial_version(project_id, timeline)

    def accept_treatment(self, project_id: str, treatment_id: str) -> ProjectVersion:
        treatment = self.store.load_treatment(project_id, treatment_id)
        if treatment.decision_graph_id is None:
            raise ValueError("treatment has no editorial decision graph")
        proposed = self.store.load_decision_graph(project_id, treatment.decision_graph_id)
        base = self.store.load_version(project_id, treatment.base_version_id)
        timeline, inverses = apply_timeline_patch(base.timeline, treatment.patch.operations)
        patch = treatment.patch.model_copy(update={"inverse_operations": inverses})
        self.store.save_patch(patch)
        graph_id = new_id("decisiongraph")
        version = ProjectVersion(
            project_id=project_id,
            parent_version_id=base.id,
            message=f"Accept treatment: {treatment.title}",
            timeline=timeline,
            accepted_treatment_id=treatment.id,
            applied_patch_id=patch.id,
            decision_graph_id=graph_id,
        )
        approved = approve_graph(
            proposed,
            version_id=version.id,
            inverse_operations=inverses,
            graph_id=graph_id,
        )
        self.store.save_decision_graph(approved)
        for event in graph_events(approved, rationale=patch.rationale):
            self.store.append_decision_event(event)
        self.store.save_version(version)
        return version

    def apply_patch(self, project_id: str, patch: EditPatch, message: str) -> ProjectVersion:
        base = self.store.load_version(project_id, patch.base_version_id)
        timeline, inverses = apply_timeline_patch(base.timeline, patch.operations)
        completed = patch.model_copy(update={"inverse_operations": inverses})
        self.store.save_patch(completed)
        graph_id = new_id("decisiongraph")
        version = ProjectVersion(
            project_id=project_id,
            parent_version_id=base.id,
            message=message,
            timeline=timeline,
            applied_patch_id=completed.id,
            decision_graph_id=graph_id,
        )
        graph = approved_patch_graph(
            project_id=project_id,
            base_version_id=base.id,
            parent_graph_id=base.decision_graph_id,
            version_id=version.id,
            patch=completed,
            origin=DecisionOrigin.HUMAN,
            intent=message,
        ).model_copy(update={"id": graph_id})
        self.store.save_decision_graph(graph)
        for event in graph_events(graph, rationale=completed.rationale):
            self.store.append_decision_event(event)
        self.store.save_version(version)
        return version

    def revert_version(self, project_id: str, version_id: str) -> ProjectVersion:
        version = self.store.load_version(project_id, version_id)
        if version.applied_patch_id is None:
            raise ValueError("selected version has no reversible patch")
        patch_path = (
            self.store.project_path(project_id)
            / "patches"
            / f"{version.applied_patch_id}.json"
        )
        patch = EditPatch.model_validate_json(patch_path.read_text(encoding="utf-8"))
        if not patch.inverse_operations:
            raise ValueError("patch has no inverse operations")
        inverse = EditPatch(
            project_id=project_id,
            base_version_id=version.id,
            operations=patch.inverse_operations,
            rationale=f"Revert {version.id}",
            evidence_refs=patch.evidence_refs,
        )
        return self.apply_patch(project_id, inverse, f"Revert: {version.message}")

    def save_treatment(self, treatment: Treatment) -> Treatment:
        return self.store.save_treatment(treatment)

    def plan_screen_workflow(
        self,
        *,
        title: str,
        script: str,
        beats: list[ScreenWorkflowBeatInput] | None = None,
        reviewed_by: str | None = None,
        reference_study_ids: list[str] | None = None,
    ) -> ScreenWorkflowPlan:
        """Create a media-free capture plan from a script and optional authored beats."""

        from aoa_editing.application.screen_workflow import build_screen_workflow_plan

        return build_screen_workflow_plan(
            title=title,
            script=script,
            beats=beats,
            reviewed_by=reviewed_by,
            reference_study_ids=reference_study_ids or [],
        )

    def propose_screen_workflow(
        self,
        project_id: str,
        *,
        plan: ScreenWorkflowPlan,
        edit_spec: ScreenWorkflowEditSpec,
    ) -> Treatment:
        """Create a normal reversible Treatment from reviewed post-capture bindings."""

        from aoa_editing.scenarios.screen_workflow import screen_workflow

        project = self.store.load_project(project_id)
        if project.intent.scenario is not Scenario.SCREEN_WORKFLOW:
            raise ValueError("project intent must use screen.workflow")
        if not plan.capture_ready:
            raise ValueError("screen workflow plan must be reviewed for capture")
        asset = self.store.load_asset(project_id, edit_spec.asset_id)
        if asset.metadata.frame_rate is None:
            raise ValueError("screen workflow asset has no usable frame rate")
        expected_beats = {item.id for item in plan.beats}
        if {item.beat_id for item in edit_spec.bindings} != expected_beats:
            raise ValueError("every workflow beat must have one or more reviewed source bindings")
        timing = edit_spec.voiceover_timing
        voiceover_asset = None
        voiceover_evidence = None
        if timing is not None:
            if not timing.edit_ready:
                raise ValueError("screen workflow voiceover timing must be reviewed for edit")
            if timing.plan_id != plan.id or timing.script_sha256 != plan.script_sha256:
                raise ValueError("screen workflow voiceover timing does not match the plan")
            plan_beat_ids = [item.id for item in plan.beats]
            if [item.beat_id for item in timing.cues] != plan_beat_ids:
                raise ValueError("voiceover timing must cover every workflow beat in order")
            binding_beat_order = list(
                dict.fromkeys(item.beat_id for item in edit_spec.bindings)
            )
            if binding_beat_order != plan_beat_ids:
                raise ValueError("voiceover-bound source bindings must follow plan beat order")
            bound_duration_by_beat = {
                beat_id: sum(
                    item.timeline_duration_frames
                    for item in edit_spec.bindings
                    if item.beat_id == beat_id
                )
                for beat_id in plan_beat_ids
            }
            if [bound_duration_by_beat[item.beat_id] for item in timing.cues] != [
                item.timeline_range.duration for item in timing.cues
            ]:
                raise ValueError("source binding durations must match reviewed voiceover cues")
            voiceover_asset = self.store.load_asset(project_id, timing.asset_id)
            if voiceover_asset.sha256 != timing.source_sha256:
                raise ValueError("voiceover timing source hash does not match the ingested asset")
            if not voiceover_asset.metadata.has_audio:
                raise ValueError("voiceover timing asset has no audio stream")
        if project.current_version_id is None:
            duration_frames = sum(
                item.timeline_duration_frames for item in edit_spec.bindings
            )
            width = asset.metadata.width or 1920
            height = asset.metadata.height or 1080
            if width % 2:
                width += 1
            if height % 2:
                height += 1
            base = self.initialize_timeline(
                project_id,
                duration_frames=duration_frames,
                width=width,
                height=height,
                frame_rate=asset.metadata.frame_rate,
            )
        else:
            base = self.store.load_version(project_id)
        if timing is not None:
            if timing.timeline_frame_rate != base.timeline.frame_rate:
                raise ValueError("voiceover timing and screen timeline frame rates must match")
            if timing.duration_frames != sum(
                item.timeline_duration_frames for item in edit_spec.bindings
            ):
                raise ValueError("voiceover timing duration must equal the bound edit duration")
        existing = next(
            (
                item
                for item in self.store.list_evidence(project_id)
                if item.kind == "screen.workflow.binding"
                and item.asset_id == asset.id
                and item.payload.get("edit_spec_id") == edit_spec.id
            ),
            None,
        )
        if existing is None:
            from aoa_editing.application.screen_workflow import (
                screen_workflow_temporal_checks,
            )

            temporal_checks = screen_workflow_temporal_checks(edit_spec.bindings)
            binding_evidence = EvidenceRecord(
                project_id=project_id,
                asset_id=asset.id,
                source_sha256=asset.sha256,
                kind="screen.workflow.binding",
                authority=EvidenceAuthority.HUMAN_CORRECTION,
                confidence=1.0,
                payload={
                    "edit_spec_id": edit_spec.id,
                    "plan_id": plan.id,
                    "script_sha256": plan.script_sha256,
                    "reviewed_by": edit_spec.reviewed_by,
                    "review_note": edit_spec.review_note,
                    "bindings": [
                        item.model_dump(mode="json") for item in edit_spec.bindings
                    ],
                    "temporal_integrity_checks": [
                        item.model_dump(mode="json") for item in temporal_checks
                    ],
                },
                ranges=[item.source_range for item in edit_spec.bindings],
                time_base=asset.metadata.frame_rate,
                provenance=edit_spec.provenance,
            )
            self.store.save_evidence(binding_evidence)
        else:
            binding_evidence = existing
        if timing is not None and voiceover_asset is not None:
            voiceover_evidence = next(
                (
                    item
                    for item in self.store.list_evidence(project_id)
                    if item.kind == "screen.workflow.voiceover-timing"
                    and item.asset_id == voiceover_asset.id
                    and item.payload.get("timing_id") == timing.id
                ),
                None,
            )
            if voiceover_evidence is None:
                voiceover_evidence = EvidenceRecord(
                    project_id=project_id,
                    asset_id=voiceover_asset.id,
                    source_sha256=voiceover_asset.sha256,
                    kind="screen.workflow.voiceover-timing",
                    authority=EvidenceAuthority.HUMAN_CORRECTION,
                    confidence=1.0,
                    payload={
                        "timing_id": timing.id,
                        "plan_id": timing.plan_id,
                        "script_sha256": timing.script_sha256,
                        "draft_alignment_method": timing.draft_alignment_method,
                        "reviewed_by": timing.reviewed_by,
                        "review_note": timing.review_note,
                        "normalization_target_lufs": timing.normalization_target_lufs,
                        "cues": [item.model_dump(mode="json") for item in timing.cues],
                        "analysis_evidence_refs": timing.analysis_evidence_refs,
                    },
                    ranges=[item.timeline_range for item in timing.cues],
                    time_base=timing.timeline_frame_rate,
                    provenance=timing.provenance,
                )
                self.store.save_evidence(voiceover_evidence)
        treatment = screen_workflow(
            project_id,
            base,
            asset,
            plan,
            edit_spec,
            binding_evidence,
            voiceover_asset=voiceover_asset,
            voiceover_evidence=voiceover_evidence,
        )
        return self.store.save_treatment(treatment)

    def time_screen_workflow_voiceover(
        self,
        project_id: str,
        *,
        plan: ScreenWorkflowPlan,
        asset_id: str,
        timeline_frame_rate: FrameRate | None = None,
        transcribe: bool = True,
    ) -> ScreenWorkflowVoiceoverTiming:
        """Analyze one ingested narration asset and return a review-required timing draft."""

        from aoa_editing.analysis.service import AnalysisService
        from aoa_editing.application.voiceover import build_voiceover_timing_draft

        project = self.store.load_project(project_id)
        if project.intent.scenario is not Scenario.SCREEN_WORKFLOW:
            raise ValueError("project intent must use screen.workflow")
        asset = self.store.load_asset(project_id, asset_id)
        AnalysisService(self.store).analyze(project_id, asset_id, transcribe=transcribe)
        evidence = [
            item
            for item in self.store.list_effective_evidence(project_id)
            if item.asset_id == asset_id and item.source_sha256 == asset.sha256
        ]
        return build_voiceover_timing_draft(
            plan=plan,
            asset=asset,
            evidence=evidence,
            timeline_frame_rate=timeline_frame_rate or FrameRate(),
        )

    def review_screen_workflow_voiceover(
        self,
        *,
        plan: ScreenWorkflowPlan,
        draft: ScreenWorkflowVoiceoverTiming,
        cues: list[ScreenWorkflowVoiceoverCue],
        reviewed_by: str,
        review_note: str,
    ) -> ScreenWorkflowVoiceoverTiming:
        """Create an attributable reviewed timing revision without mutating the draft."""

        from aoa_editing.application.voiceover import review_voiceover_timing

        return review_voiceover_timing(
            plan=plan,
            draft=draft,
            cues=cues,
            reviewed_by=reviewed_by,
            review_note=review_note,
        )

    def propose(
        self,
        project_id: str,
        *,
        scenario: Scenario | None = None,
        asset_id: str | None = None,
        duration_seconds: float | None = None,
    ) -> Treatment:
        from aoa_editing.scenarios.planners import (
            memory_montage,
            speech_clean,
            still_motion,
        )

        project = self.store.load_project(project_id)
        selected_scenario = scenario or project.intent.scenario
        if selected_scenario is Scenario.SCREEN_WORKFLOW:
            raise ValueError(
                "screen.workflow requires a reviewed capture plan and edit spec; "
                "use propose_screen_workflow"
            )
        assets = self.store.list_assets(project_id)
        if not assets:
            raise ValueError("ingest at least one asset before planning")
        selected = next((asset for asset in assets if asset.id == asset_id), None)
        if asset_id and selected is None:
            raise ValueError(f"asset does not belong to project: {asset_id}")
        if project.current_version_id is None:
            primary = selected or assets[0]
            fps = primary.metadata.frame_rate or FrameRate()
            duration = (
                duration_seconds
                or project.intent.target_duration_seconds
                or primary.metadata.duration_seconds
                or 6.0
            )
            width = primary.metadata.width or 1920
            height = primary.metadata.height or 1080
            if width % 2:
                width += 1
            if height % 2:
                height += 1
            base = self.initialize_timeline(
                project_id,
                duration_frames=max(1, round(duration * fps.fps)),
                width=width,
                height=height,
                frame_rate=fps,
            )
        else:
            base = self.store.load_version(project_id)
        evidence = self.store.list_effective_evidence(project_id)
        if selected_scenario is Scenario.STILL_MOTION:
            primary_image = selected or next(
                (asset for asset in assets if asset.media_kind.value == "image"), None
            )
            if primary_image is None:
                raise ValueError("still.motion needs an image asset")
            treatment = still_motion(
                project_id,
                base,
                primary_image,
                evidence,
                duration_seconds=(
                    duration_seconds or project.intent.target_duration_seconds or 6.0
                ),
            )
        elif selected_scenario is Scenario.SPEECH_CLEAN:
            primary_audio = selected or next(
                (asset for asset in assets if asset.metadata.has_audio), None
            )
            if primary_audio is None:
                raise ValueError("speech.clean needs an audio-bearing asset")
            treatment = speech_clean(project_id, base, primary_audio, evidence)
        elif selected_scenario is Scenario.MEMORY_MONTAGE:
            treatment = memory_montage(project_id, base, assets, evidence)
        elif selected_scenario is Scenario.REFERENCE_RECONSTRUCT:
            raise ValueError(
                "reference.reconstruct requires a frozen spec; use propose_reference"
            )
        else:  # pragma: no cover - enum constrains this branch
            raise ValueError(f"unsupported scenario: {selected_scenario}")
        return self.store.save_treatment(treatment)

    def propose_options(
        self,
        project_id: str,
        *,
        scenario: Scenario | None = None,
        asset_id: str | None = None,
        duration_seconds: float | None = None,
    ) -> list[Treatment]:
        """Persist two or three executable interpretations when the task is creative."""

        from aoa_editing.scenarios.planners import creative_treatment_variants

        primary = self.propose(
            project_id,
            scenario=scenario,
            asset_id=asset_id,
            duration_seconds=duration_seconds,
        )
        variants = creative_treatment_variants(primary)
        return [primary, *(self.store.save_treatment(item) for item in variants[1:])]

    def propose_reference(
        self,
        project_id: str,
        *,
        asset_id: str,
        spec: ReferenceReconstructionSpec,
    ) -> Treatment:
        """Create a normal treatment from a frozen parameter-only reference spec."""

        from aoa_editing.scenarios.planners import reference_reconstruct

        project = self.store.load_project(project_id)
        if spec.reference_sha256 not in project.sealed_reference_hashes:
            raise ValueError("reference hash must be sealed before reconstruction planning")
        asset = self.store.load_asset(project_id, asset_id)
        if asset.sha256 != spec.source_sha256:
            raise ValueError("selected source does not match the reconstruction spec")
        if project.current_version_id is None:
            base = self.initialize_timeline(
                project_id,
                duration_frames=spec.duration_frames,
                width=spec.camera_motion.width,
                height=spec.camera_motion.height,
                frame_rate=spec.camera_motion.frame_rate,
            )
        else:
            base = self.store.load_version(project_id)
        evidence = self.store.list_effective_evidence(project_id)
        specification = next(
            (
                record
                for record in evidence
                if record.asset_id == asset.id
                and record.kind == "reference.reconstruction_spec"
                and record.payload.get("spec_id") == spec.id
            ),
            None,
        )
        if specification is None:
            serialized = spec.model_dump_json().encode("utf-8")
            specification = EvidenceRecord(
                project_id=project_id,
                asset_id=asset.id,
                source_sha256=asset.sha256,
                kind="reference.reconstruction_spec",
                confidence=min(
                    item.inlier_ratio for item in spec.camera_motion.measurements
                ),
                payload={
                    "spec_id": spec.id,
                    "spec_sha256": hashlib.sha256(serialized).hexdigest(),
                    "reference_sha256": spec.reference_sha256,
                    "reference_used_as_media": False,
                    "frozen_spec": spec.model_dump(mode="json"),
                },
                time_base=spec.camera_motion.frame_rate,
                ranges=[FrameRange(start=0, duration=spec.duration_frames)],
                provenance=Provenance(
                    tool="aoa-editing",
                    tool_version=__version__,
                    parameters={
                        "operation": "persist-frozen-reference-parameters",
                        "reference_pixels_persisted": False,
                    },
                    deterministic=True,
                ),
            )
            self.store.save_evidence(specification)
            evidence.append(specification)
        treatment = reference_reconstruct(project_id, base, asset, evidence, spec)
        return self.store.save_treatment(treatment)

    def propose_reference_v2(
        self,
        project_id: str,
        *,
        asset_id: str,
        spec: ReferenceReconstructionSpecV2,
        pass_kind: ReferenceReconstructionPassKindV2,
        phase_correction: ReferencePhaseCorrectionV2 | None = None,
    ) -> Treatment:
        """Create one bounded Motion Language v2 treatment from a frozen spec."""

        from aoa_editing.scenarios.reference_v2 import reference_reconstruct_v2

        project = self.store.load_project(project_id)
        if spec.reference_sha256 not in project.sealed_reference_hashes:
            raise ValueError("reference hash must be sealed before v2 reconstruction planning")
        asset = self.store.load_asset(project_id, asset_id)
        if asset.sha256 != spec.source_sha256:
            raise ValueError("selected source does not match the reconstruction spec v2")
        if project.current_version_id is None:
            base = self.initialize_timeline(
                project_id,
                duration_frames=spec.duration_frames,
                width=spec.width,
                height=spec.height,
                frame_rate=spec.frame_rate,
            )
        else:
            base = self.store.load_version(project_id)
        evidence = self.store.list_effective_evidence(project_id)
        specification = next(
            (
                record
                for record in evidence
                if record.asset_id == asset.id
                and record.kind == "reference.reconstruction_spec.v2"
                and record.payload.get("spec_id") == spec.id
            ),
            None,
        )
        if specification is None:
            serialized = spec.model_dump_json().encode("utf-8")
            specification = EvidenceRecord(
                project_id=project_id,
                asset_id=asset.id,
                source_sha256=asset.sha256,
                kind="reference.reconstruction_spec.v2",
                confidence=min(frame.confidence for frame in spec.motion_frames),
                payload={
                    "spec_id": spec.id,
                    "spec_sha256": hashlib.sha256(serialized).hexdigest(),
                    "reference_sha256": spec.reference_sha256,
                    "reference_used_as_media": False,
                    "frozen_spec": spec.model_dump(mode="json"),
                },
                time_base=spec.frame_rate,
                ranges=[FrameRange(start=0, duration=spec.duration_frames)],
                provenance=Provenance(
                    tool="aoa-editing",
                    tool_version=__version__,
                    parameters={
                        "operation": "persist-frozen-reference-parameters-v2",
                        "reference_pixels_persisted": False,
                        "pass_kind": pass_kind,
                    },
                    deterministic=True,
                ),
            )
            self.store.save_evidence(specification)
            evidence.append(specification)
        if pass_kind == "phase_compensated_matrix":
            if phase_correction is None:
                raise ValueError("phase-compensated pass requires phase correction")
            correction_record = next(
                (
                    record
                    for record in evidence
                    if record.asset_id == asset.id
                    and record.kind == "reference.phase_correction.v2"
                    and record.payload.get("correction_id") == phase_correction.id
                ),
                None,
            )
            if correction_record is None:
                correction_record = EvidenceRecord(
                    project_id=project_id,
                    asset_id=asset.id,
                    source_sha256=asset.sha256,
                    kind="reference.phase_correction.v2",
                    authority=EvidenceAuthority.ANALYZER,
                    confidence=1.0,
                    payload={
                        "correction_id": phase_correction.id,
                        "spec_id": phase_correction.spec_id,
                        "comparison_report_id": phase_correction.comparison_report_id,
                        "reference_used_as_media": False,
                        "candidate_used_as_media": False,
                        "phase_correction": phase_correction.model_dump(mode="json"),
                    },
                    artifacts=[
                        phase_correction.comparison_report_path,
                        phase_correction.candidate_motion_path,
                    ],
                    time_base=spec.frame_rate,
                    ranges=list(phase_correction.identity_ranges),
                    provenance=Provenance(
                        tool="aoa-editing",
                        tool_version=__version__,
                        parameters={
                            "operation": "persist-reference-phase-correction-v2",
                            "algorithm_revision": phase_correction.algorithm_revision,
                            "reference_pixels_persisted": False,
                            "candidate_pixels_persisted": False,
                        },
                        deterministic=True,
                    ),
                )
                self.store.save_evidence(correction_record)
                evidence.append(correction_record)
        treatment = reference_reconstruct_v2(
            project_id,
            base,
            asset,
            evidence,
            spec,
            pass_kind=pass_kind,
            phase_correction=phase_correction,
        )
        return self.store.save_treatment(treatment)

    def propose_technique(
        self,
        project_id: str,
        *,
        asset_id: str,
        technique: TechniquePacket,
        width: int,
        height: int,
    ) -> Treatment:
        """Apply an operational Tree of Editing packet through normal contracts."""

        from aoa_editing.scenarios.planners import apply_camera_technique

        recipe = technique.camera_motion
        if recipe is None:
            raise ValueError("technique has no camera motion recipe")
        asset = self.store.load_asset(project_id, asset_id)
        project = self.store.load_project(project_id)
        if project.current_version_id is None:
            base = self.initialize_timeline(
                project_id,
                duration_frames=recipe.duration_frames,
                width=width,
                height=height,
                frame_rate=recipe.frame_rate,
            )
        else:
            base = self.store.load_version(project_id)
        treatment = apply_camera_technique(
            project_id,
            base,
            asset,
            self.store.list_effective_evidence(project_id),
            technique,
            width=width,
            height=height,
        )
        return self.store.save_treatment(treatment)

    def record_technique_composition_v2(
        self,
        project_id: str,
        *,
        asset_id: str,
        assessment: TechniqueCompositionEvidenceV2,
    ) -> EvidenceRecord:
        """Persist explicit composition applicability as evidence, never edit authority."""

        asset = self.store.load_asset(project_id, asset_id)
        existing = [
            record
            for record in self.store.list_effective_evidence(project_id)
            if record.asset_id == asset.id
            and record.source_sha256 == asset.sha256
            and record.kind == "editing.composition_applicability.v2"
        ]
        payload = assessment.model_dump(mode="json")
        for record in existing:
            if record.payload == payload:
                return record
        evidence = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="editing.composition_applicability.v2",
            authority=(
                EvidenceAuthority.HUMAN_CORRECTION
                if assessment.authority == "human_assertion"
                else EvidenceAuthority.ANALYZER
            ),
            confidence=(
                1.0
                if assessment.authority in {"fixture_ground_truth", "human_assertion"}
                else 0.5
            ),
            payload=payload,
            supersedes=[record.id for record in existing],
            provenance=Provenance(
                tool="aoa-editing-composition-applicability-v2",
                tool_version=__version__,
                parameters={
                    "authority": assessment.authority,
                    "semantic_claim": True,
                    "edit_authority": False,
                },
                deterministic=assessment.authority == "fixture_ground_truth",
            ),
        )
        self.store.save_evidence(evidence)
        return evidence

    def propose_technique_v2(
        self,
        project_id: str,
        *,
        asset_id: str,
        technique: TechniquePacket,
        width: int,
        height: int,
    ) -> TechniqueProposalResultV2:
        """Apply or honestly refuse a portable packet through one application boundary."""

        from aoa_editing.scenarios.planners import (
            apply_camera_technique_v2,
            assess_camera_technique_v2,
        )

        recipe = technique.portable_camera_motion_v2
        if recipe is None:
            raise ValueError("technique has no portable v2 camera motion recipe")
        asset = self.store.load_asset(project_id, asset_id)
        evidence = self.store.list_effective_evidence(project_id)
        decision = assess_camera_technique_v2(
            project_id,
            asset,
            evidence,
            technique,
            width=width,
            height=height,
        )
        decision_record = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="editing.technique_applicability_decision.v2",
            payload=decision.model_dump(mode="json"),
            provenance=Provenance(
                tool="aoa-editing-application-service",
                tool_version=__version__,
                parameters={
                    "operation": "propose-technique-v2",
                    "outcome": decision.outcome,
                    "treatment_created": decision.outcome == "eligible",
                },
                deterministic=True,
            ),
        )
        self.store.save_evidence(decision_record)
        if decision.outcome == "refused":
            return TechniqueProposalResultV2(decision=decision)

        project = self.store.load_project(project_id)
        if project.current_version_id is None:
            base = self.initialize_timeline(
                project_id,
                duration_frames=recipe.duration_frames,
                width=width,
                height=height,
                frame_rate=recipe.frame_rate,
            )
        else:
            base = self.store.load_version(project_id)
        treatment = apply_camera_technique_v2(
            project_id,
            base,
            asset,
            technique,
            decision,
            width=width,
            height=height,
        )
        return TechniqueProposalResultV2(
            decision=decision,
            treatment=self.store.save_treatment(treatment),
        )

    def touch_project(self, project_id: str) -> ProjectManifest:
        manifest = self.store.load_project(project_id)
        updated = manifest.model_copy(update={"updated_at": datetime.now(UTC)})
        self.store.save_project(updated)
        return updated
