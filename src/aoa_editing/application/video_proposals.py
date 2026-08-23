"""Evidence-bound Video Anatomy proposals and explicit human acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aoa_editing import __version__
from aoa_editing.application.service import EditingService
from aoa_editing.domain.models import (
    Clip,
    DecomposedTransformMotionV2,
    DissolveEffect,
    EditorialStructureProposal,
    EditorialStructureSection,
    EditPatch,
    FrameRange,
    FrameRate,
    MotionSamplingV2,
    MotionTimeV2,
    PatchOperation,
    PatchPreview,
    Provenance,
    RationaleItem,
    ReconstructionShotSkeleton,
    ReconstructionTransformSample,
    ReferenceReconstructionProposal,
    ReferenceReconstructionSpecV2,
    ScalarKeyframe,
    ScalarMotionCurveV2,
    ScalarMotionKeyframeV2,
    Track,
    TrackKind,
    TransformEffect,
    TransformEffectV2,
    Vec2Keyframe,
    Vec2MotionCurveV2,
    Vec2MotionKeyframeV2,
    Vec2ValueV2,
    VideoAnatomy,
    VideoMotionEvidence,
    VideoProposalAcceptanceReceipt,
    VideoProposalReview,
    VideoTransitionType,
)
from aoa_editing.infrastructure.store import ProjectStore


class VideoProposalError(ValueError):
    """A proposal or human review is not safe to compile or accept."""


def _model_sha256(model: Any) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class VideoProposalService:
    def __init__(self, store: ProjectStore):
        self.store = store

    def create_editorial_structure(
        self,
        anatomy: VideoAnatomy,
    ) -> EditorialStructureProposal:
        if anatomy.status.value in {"failed", "insufficient"}:
            raise VideoProposalError("insufficient Video Anatomy cannot support a proposal")
        shots = anatomy.structure.shots
        climax_id = self._climax_shot(anatomy)
        sections: list[EditorialStructureSection] = []
        shot_roles: dict[str, str] = {}
        for index, shot in enumerate(shots):
            if index == 0:
                role = "hook"
            elif shot.id == climax_id:
                role = "climax"
            elif index == len(shots) - 1:
                role = "conclusion"
            elif index == 1:
                role = "setup"
            else:
                role = "development"
            observation = next(
                (item for item in anatomy.visual_observations if item.shot_id == shot.id),
                None,
            )
            summary = (
                observation.description
                if observation is not None
                else f"Measured shot {shot.id}; semantic description remains unavailable."
            )
            evidence_refs = [*anatomy.evidence_refs, *shot.representative_sample_ids]
            sections.append(
                EditorialStructureSection(
                    role=role,  # type: ignore[arg-type]
                    frame_range=shot.frame_range,
                    shot_ids=[shot.id],
                    summary=summary,
                    confidence=min(shot.confidence, observation.confidence if observation else 0.5),
                    evidence_refs=evidence_refs,
                )
            )
            shot_roles[shot.id] = role
        techniques = sorted(
            {
                item.transition_type.value
                for item in anatomy.structure.transitions
                if item.transition_type
                not in {VideoTransitionType.UNKNOWN, VideoTransitionType.CONTINUOUS_MOTION}
            }
        )
        if any(item.camera_hypotheses for item in anatomy.motion_evidence):
            techniques.append("measured-transform-motion")
        proposal = EditorialStructureProposal(
            project_id=anatomy.project_id,
            asset_id=anatomy.asset_id,
            anatomy_id=anatomy.id,
            anatomy_sha256=anatomy.anatomy_sha256,
            sections=sections,
            pacing_phases=[f"{item.role}:{item.frame_range.duration}f" for item in sections],
            shot_roles=shot_roles,
            likely_techniques=techniques,
            rationale=[
                RationaleItem(
                    claim=(
                        "Editorial roles are a proposal derived from measured shot order, "
                        "not evidence."
                    ),
                    evidence_refs=anatomy.evidence_refs,
                    confidence=min(anatomy.confidence_summary.values()),
                )
            ],
            evidence_refs=anatomy.evidence_refs,
            uncertainties=[
                *anatomy.incompleteness_reasons,
                *[
                    f"unresolved range {item.start}:{item.end}"
                    for item in anatomy.unresolved_ranges
                ],
            ],
            alternative_interpretations=[
                "The first shot may be setup rather than hook.",
                "The maximum measured motion may not coincide with the semantic climax.",
            ],
            provenance=self._provenance(
                operation="editorial-structure-proposal",
                anatomy_sha256=anatomy.anatomy_sha256,
            ),
        )
        self.store.save_editorial_structure_proposal(proposal)
        return proposal

    def create_reference_reconstruction(
        self,
        anatomy: VideoAnatomy,
        *,
        target_asset_id: str | None = None,
        base_version_id: str | None = None,
        precise_motion_spec: ReferenceReconstructionSpecV2 | None = None,
        precise_motion_spec_ref: str | None = None,
    ) -> ReferenceReconstructionProposal:
        if anatomy.status.value in {"failed", "insufficient"}:
            raise VideoProposalError("insufficient Video Anatomy cannot support reconstruction")
        if (precise_motion_spec is None) != (precise_motion_spec_ref is None):
            raise VideoProposalError(
                "precise motion spec and its immutable evidence ref must appear together"
            )
        if precise_motion_spec is not None:
            self._validate_precise_motion_spec(
                anatomy,
                precise_motion_spec,
                target_asset_id=target_asset_id,
            )
        skeleton: list[ReconstructionShotSkeleton] = []
        cursor = 0
        motion_by_shot = {item.shot_id: item for item in anatomy.motion_evidence}
        for index, shot in enumerate(anatomy.structure.shots, start=1):
            timeline_range = FrameRange(start=cursor, duration=shot.frame_range.duration)
            motion = motion_by_shot.get(shot.id)
            observation = next(
                (item for item in anatomy.visual_observations if item.shot_id == shot.id),
                None,
            )
            transition_in = self._transition_at(anatomy, shot.frame_range.start)
            transition_out = self._transition_at(anatomy, shot.frame_range.end)
            skeleton.append(
                ReconstructionShotSkeleton(
                    id=f"reconstruction-shot-{index:04d}",
                    source_shot_ids=[shot.id],
                    timeline_range=timeline_range,
                    composition_summary=(
                        observation.description
                        if observation is not None
                        else "Source-neutral composition is unresolved; preserve target framing."
                    ),
                    transforms=self._transform_samples(
                        motion,
                        timeline_range,
                        precise_motion_spec=precise_motion_spec,
                    ),
                    transition_in=transition_in,
                    transition_out=transition_out,
                    audio_events=self._audio_events(anatomy, shot.frame_range),
                    overlays=(observation.overlays if observation is not None else []),
                    technique_ids=self._techniques(motion, transition_in, transition_out),
                    evidence_refs=sorted(
                        set(
                            [
                                *anatomy.evidence_refs,
                                *(
                                    [precise_motion_spec_ref]
                                    if precise_motion_spec_ref is not None
                                    else []
                                ),
                                *shot.representative_sample_ids,
                                *(
                                    [motion.all_frame_evidence_ref]
                                    if motion is not None
                                    and motion.all_frame_evidence_ref is not None
                                    else []
                                ),
                            ]
                        )
                    ),
                    confidence=min(
                        shot.confidence,
                        motion.confidence if motion is not None else 0.5,
                        observation.confidence if observation is not None else 0.5,
                    ),
                )
            )
            cursor += shot.frame_range.duration
        width = (
            precise_motion_spec.width
            if precise_motion_spec is not None
            else anatomy.technical_metadata.width or 1920
        )
        height = (
            precise_motion_spec.height
            if precise_motion_spec is not None
            else anatomy.technical_metadata.height or 1080
        )
        preview = None
        if target_asset_id is not None or base_version_id is not None:
            if target_asset_id is None or base_version_id is None:
                raise VideoProposalError(
                    "patch preview requires both target_asset_id and base_version_id"
                )
            preview = self._compile_preview(
                anatomy,
                skeleton,
                target_asset_id=target_asset_id,
                base_version_id=base_version_id,
                width=width,
                height=height,
                precise_motion=precise_motion_spec is not None,
                frame_rate=(
                    precise_motion_spec.frame_rate
                    if precise_motion_spec is not None
                    else anatomy.plan.time_base
                ),
            )
        unsupported = list(anatomy.incompleteness_reasons)
        if not anatomy.motion_evidence and precise_motion_spec is None:
            unsupported.append("measured transform curves are unavailable")
        proposal = ReferenceReconstructionProposal(
            project_id=anatomy.project_id,
            reference_anatomy_id=anatomy.id,
            reference_anatomy_sha256=anatomy.anatomy_sha256,
            output_width=width + width % 2,
            output_height=height + height % 2,
            output_frame_rate=(
                precise_motion_spec.frame_rate
                if precise_motion_spec is not None
                else anatomy.plan.time_base
            ),
            duration_frames=cursor,
            shot_skeleton=skeleton,
            motion_phases=[
                f"{item.shot_id}:{phase.role}:{phase.frame_range.start}-{phase.frame_range.end}"
                for item in anatomy.motion_evidence
                for phase in item.phases
            ],
            audio_structure=(
                [
                    f"{item.kind}:{item.start_seconds:.3f}-{item.end_seconds:.3f}"
                    for item in anatomy.audio_timeline.events
                ]
                if anatomy.audio_timeline is not None
                else []
            ),
            unsupported_elements=sorted(set(unsupported)),
            unresolved_elements=[
                f"frames {item.start}:{item.end}" for item in anatomy.unresolved_ranges
            ],
            evidence_refs=[
                *anatomy.evidence_refs,
                *(
                    [precise_motion_spec_ref]
                    if precise_motion_spec_ref is not None
                    else []
                ),
            ],
            confidence=min(anatomy.confidence_summary.values()),
            refinement_plan=[
                "Run focused analysis for every unresolved range.",
                "Review boundary triplets and measured transform curves.",
                "Approve only after patch preview and source-lineage inspection.",
            ],
            patch_preview=preview,
            target_asset_id=target_asset_id,
            provenance=self._provenance(
                operation="source-neutral-reference-reconstruction-proposal",
                anatomy_sha256=anatomy.anatomy_sha256,
                reference_pixels_embedded=False,
                precise_motion_spec_id=(
                    precise_motion_spec.id if precise_motion_spec is not None else None
                ),
                precise_motion_spec_ref=precise_motion_spec_ref,
                dense_normalized_transform_samples=(
                    precise_motion_spec is not None
                ),
            ),
        )
        self.store.save_reference_reconstruction_proposal(proposal)
        return proposal

    def review_editorial(
        self,
        project_id: str,
        proposal_id: str,
        *,
        decision: str,
        reviewer: str,
        rationale: str,
    ) -> VideoProposalReview:
        proposal = self.store.load_editorial_structure_proposal(project_id, proposal_id)
        return self._review(
            project_id,
            "editorial-structure",
            proposal,
            decision,
            reviewer,
            rationale,
        )

    def review_reconstruction(
        self,
        project_id: str,
        proposal_id: str,
        *,
        decision: str,
        reviewer: str,
        rationale: str,
    ) -> VideoProposalReview:
        proposal = self.store.load_reference_reconstruction_proposal(project_id, proposal_id)
        return self._review(
            project_id,
            "reference-reconstruction",
            proposal,
            decision,
            reviewer,
            rationale,
        )

    def accept_reconstruction(
        self,
        project_id: str,
        proposal_id: str,
        review_id: str,
    ) -> VideoProposalAcceptanceReceipt:
        proposal = self.store.load_reference_reconstruction_proposal(project_id, proposal_id)
        review = self.store.load_video_proposal_review(project_id, review_id)
        proposal_hash = _model_sha256(proposal)
        if (
            review.proposal_kind != "reference-reconstruction"
            or review.proposal_id != proposal.id
            or review.proposal_sha256 != proposal_hash
            or review.decision != "approved"
        ):
            raise VideoProposalError("reconstruction proposal lacks a matching approval")
        if proposal.patch_preview is None:
            raise VideoProposalError("approved proposal has no canonical patch preview")
        version = EditingService(self.store).apply_patch(
            project_id,
            proposal.patch_preview.patch,
            f"Accept Video Anatomy reconstruction {proposal.id}",
        )
        patch_path = (
            self.store.project_path(project_id)
            / "patches"
            / f"{proposal.patch_preview.patch.id}.json"
        )
        applied = EditPatch.model_validate_json(patch_path.read_text(encoding="utf-8"))
        receipt = VideoProposalAcceptanceReceipt(
            project_id=project_id,
            proposal_id=proposal.id,
            proposal_sha256=proposal_hash,
            review_id=review.id,
            patch_id=applied.id,
            version_id=version.id,
            inverse_operation_count=len(applied.inverse_operations),
            accepted_by=review.reviewer,
        )
        self.store.save_video_proposal_acceptance(receipt)
        return receipt

    def _review(
        self,
        project_id: str,
        proposal_kind: str,
        proposal: Any,
        decision: str,
        reviewer: str,
        rationale: str,
    ) -> VideoProposalReview:
        if decision not in {"approved", "rejected"}:
            raise VideoProposalError("proposal review must be approved or rejected")
        review = VideoProposalReview(
            project_id=project_id,
            proposal_kind=proposal_kind,  # type: ignore[arg-type]
            proposal_id=proposal.id,
            proposal_sha256=_model_sha256(proposal),
            decision=decision,  # type: ignore[arg-type]
            reviewer=reviewer,
            rationale=rationale,
        )
        self.store.save_video_proposal_review(review)
        return review

    def _compile_preview(
        self,
        anatomy: VideoAnatomy,
        skeleton: list[ReconstructionShotSkeleton],
        *,
        target_asset_id: str,
        base_version_id: str,
        width: int,
        height: int,
        precise_motion: bool,
        frame_rate: FrameRate,
    ) -> PatchPreview:
        target = self.store.load_asset(anatomy.project_id, target_asset_id)
        base = self.store.load_version(anatomy.project_id, base_version_id)
        clips: list[Clip] = []
        for shot in skeleton:
            transforms = shot.transforms or [
                ReconstructionTransformSample(
                    time=MotionTimeV2(frame=shot.timeline_range.start),
                    center_x=0.5,
                    center_y=0.5,
                    contain_relative_scale=1.0,
                    rotation_degrees=0,
                )
            ]
            compiled_transforms = (
                self._clip_local_transforms(transforms, shot.timeline_range.start)
                if precise_motion
                else transforms
            )
            effects: list[Any] = [
                self._precise_transform_effect(compiled_transforms)
                if precise_motion
                else TransformEffect(
                    fit_mode="contain",
                    position_mode="canvas_center",
                    position=[
                        Vec2Keyframe(
                            frame=int(item.time.as_fraction),
                            x=item.center_x * width,
                            y=item.center_y * height,
                            easing="ease_in_out",
                        )
                        for item in transforms
                    ],
                    scale=[
                        ScalarKeyframe(
                            frame=int(item.time.as_fraction),
                            value=item.contain_relative_scale,
                            easing="ease_in_out",
                        )
                        for item in transforms
                    ],
                    rotation=[
                        ScalarKeyframe(
                            frame=int(item.time.as_fraction),
                            value=item.rotation_degrees,
                            easing="ease_in_out",
                        )
                        for item in transforms
                    ],
                )
            ]
            if shot.transition_in in {
                VideoTransitionType.CROSS_DISSOLVE,
                VideoTransitionType.FADE_IN,
            }:
                effects.append(DissolveEffect(fade_in_frames=min(12, shot.timeline_range.duration)))
            clips.append(
                Clip(
                    asset_id=target.id,
                    timeline_range=shot.timeline_range,
                    role="reconstruction-target",
                    effects=effects,
                    evidence_refs=anatomy.evidence_refs,
                )
            )
        track = Track(
            kind=TrackKind.VIDEO,
            name="Reviewed Video Anatomy reconstruction",
            clips=clips,
        )
        patch = EditPatch(
            project_id=anatomy.project_id,
            base_version_id=base.id,
            operations=[
                PatchOperation(
                    op="replace",
                    path="/duration_frames",
                    value=sum(item.timeline_range.duration for item in skeleton),
                ),
                PatchOperation(op="replace", path="/width", value=width + width % 2),
                PatchOperation(op="replace", path="/height", value=height + height % 2),
                PatchOperation(
                    op="replace",
                    path="/frame_rate",
                    value=frame_rate.model_dump(mode="json"),
                ),
                PatchOperation(
                    op="replace",
                    path="/tracks",
                    value=[track.model_dump(mode="json")],
                ),
            ],
            rationale=(
                "Compile a source-neutral reconstruction proposal for explicit human review; "
                "no reference media is present in the patch."
            ),
            evidence_refs=anatomy.evidence_refs,
        )
        return PatchPreview(
            project_id=anatomy.project_id,
            base_version_id=base.id,
            natural_language_command="Reconstruct the reviewed Video Anatomy with the target asset",
            patch=patch,
            summary=f"{len(skeleton)} source-neutral shots over {patch.operations[0].value} frames",
            warnings=[
                "This preview remains inert until a named human approval is accepted.",
                *anatomy.incompleteness_reasons,
            ],
        )

    def _validate_precise_motion_spec(
        self,
        anatomy: VideoAnatomy,
        spec: ReferenceReconstructionSpecV2,
        *,
        target_asset_id: str | None,
    ) -> None:
        if target_asset_id is None:
            raise VideoProposalError(
                "precise focused motion requires the explicitly permitted target asset"
            )
        target = self.store.load_asset(anatomy.project_id, target_asset_id)
        if anatomy.source_sha256 != spec.reference_sha256:
            raise VideoProposalError(
                "precise motion spec reference hash differs from Video Anatomy"
            )
        if target.sha256 != spec.source_sha256:
            raise VideoProposalError(
                "precise motion spec visual-source hash differs from the target asset"
            )
        if anatomy.plan.analysis_range.start != 0:
            raise VideoProposalError("precise reconstruction requires a full-range anatomy")
        if anatomy.plan.analysis_range.duration != spec.duration_frames:
            raise VideoProposalError(
                "precise motion spec duration differs from Video Anatomy coverage"
            )
        if anatomy.plan.time_base != spec.frame_rate:
            raise VideoProposalError(
                "precise motion spec frame rate differs from Video Anatomy"
            )
        if (
            anatomy.technical_metadata.width != spec.width
            or anatomy.technical_metadata.height != spec.height
        ):
            raise VideoProposalError(
                "precise motion spec canvas differs from Video Anatomy metadata"
            )
        if not spec.frozen_before_first_v2_render or spec.reference_media_allowed_in_render:
            raise VideoProposalError("precise motion spec does not preserve reference isolation")

    def _transform_samples(
        self,
        motion: VideoMotionEvidence | None,
        timeline_range: FrameRange,
        *,
        precise_motion_spec: ReferenceReconstructionSpecV2 | None = None,
    ) -> list[ReconstructionTransformSample]:
        if precise_motion_spec is not None:
            frames = precise_motion_spec.motion_frames[
                timeline_range.start : timeline_range.end
            ]
            if len(frames) != timeline_range.duration:
                raise VideoProposalError(
                    "precise motion spec does not cover the reconstruction shot"
                )
            return [
                ReconstructionTransformSample(
                    time=MotionTimeV2(frame=frame.frame),
                    center_x=frame.center_x,
                    center_y=frame.center_y,
                    contain_relative_scale=frame.scale_relative_to_contain,
                    rotation_degrees=frame.rotation_degrees,
                )
                for frame in frames
            ]
        if motion is None or motion.all_frame_evidence_ref is None:
            return [
                ReconstructionTransformSample(
                    time=MotionTimeV2(frame=timeline_range.start),
                    center_x=0.5,
                    center_y=0.5,
                    contain_relative_scale=1.0,
                    rotation_degrees=0,
                ),
                ReconstructionTransformSample(
                    time=MotionTimeV2(frame=timeline_range.end - 1),
                    center_x=0.5,
                    center_y=0.5,
                    contain_relative_scale=1.0,
                    rotation_degrees=0,
                ),
            ]
        raw_path = self.store.project_path(motion.project_id) / Path(motion.all_frame_evidence_ref)
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        steps = [item for item in payload.get("steps", []) if isinstance(item, dict)]
        checkpoints = {
            0,
            max(0, timeline_range.duration // 4),
            max(0, timeline_range.duration // 2),
            max(0, 3 * timeline_range.duration // 4),
            timeline_range.duration - 1,
        }
        center_x, center_y, scale, rotation = 0.5, 0.5, 1.0, 0.0
        output: list[ReconstructionTransformSample] = []
        for offset in range(timeline_range.duration):
            if offset > 0 and offset - 1 < len(steps):
                step = steps[offset - 1]
                center_x = max(0.0, min(1.0, center_x + float(step.get("translation_x", 0))))
                center_y = max(0.0, min(1.0, center_y + float(step.get("translation_y", 0))))
                scale = max(0.25, min(4.0, scale * float(step.get("scale", 1))))
                rotation += float(step.get("rotation_degrees", 0))
            if offset in checkpoints:
                output.append(
                    ReconstructionTransformSample(
                        time=MotionTimeV2(frame=timeline_range.start + offset),
                        center_x=center_x,
                        center_y=center_y,
                        contain_relative_scale=scale,
                        rotation_degrees=rotation,
                    )
                )
        return output

    @staticmethod
    def _clip_local_transforms(
        transforms: list[ReconstructionTransformSample],
        timeline_start: int,
    ) -> list[ReconstructionTransformSample]:
        return [
            item.model_copy(
                update={
                    "time": MotionTimeV2(
                        frame=item.time.frame - timeline_start,
                        subframe_numerator=item.time.subframe_numerator,
                        subframe_denominator=item.time.subframe_denominator,
                    )
                }
            )
            for item in transforms
        ]

    @staticmethod
    def _precise_transform_effect(
        transforms: list[ReconstructionTransformSample],
    ) -> TransformEffectV2:
        if not transforms:
            raise VideoProposalError("precise reconstruction requires transform samples")
        start = transforms[0].time
        end = transforms[-1].time
        pivot = Vec2ValueV2(x=0.5, y=0.5)
        return TransformEffectV2(
            motion=DecomposedTransformMotionV2(
                fit_mode="contain",
                position=Vec2MotionCurveV2(
                    keyframes=[
                        Vec2MotionKeyframeV2(
                            time=item.time,
                            value=Vec2ValueV2(x=item.center_x, y=item.center_y),
                        )
                        for item in transforms
                    ]
                ),
                scale=ScalarMotionCurveV2(
                    keyframes=[
                        ScalarMotionKeyframeV2(
                            time=item.time,
                            value=item.contain_relative_scale,
                        )
                        for item in transforms
                    ]
                ),
                rotation=ScalarMotionCurveV2(
                    keyframes=[
                        ScalarMotionKeyframeV2(
                            time=item.time,
                            value=item.rotation_degrees,
                        )
                        for item in transforms
                    ],
                    angle_unwrap=True,
                ),
                pivot=Vec2MotionCurveV2(
                    keyframes=[
                        Vec2MotionKeyframeV2(time=start, value=pivot),
                        *(
                            [Vec2MotionKeyframeV2(time=end, value=pivot)]
                            if end != start
                            else []
                        ),
                    ]
                ),
            ),
            sampling=MotionSamplingV2(),
        )

    @staticmethod
    def _transition_at(anatomy: VideoAnatomy, frame: int) -> VideoTransitionType | None:
        candidate = next(
            (item for item in anatomy.structure.transitions if abs(item.frame - frame) <= 1),
            None,
        )
        return candidate.transition_type if candidate is not None else None

    @staticmethod
    def _audio_events(anatomy: VideoAnatomy, frame_range: FrameRange) -> list[str]:
        if anatomy.audio_timeline is None:
            return []
        fps = anatomy.plan.time_base.fps
        return [
            f"{item.kind}:{item.start_seconds:.3f}-{item.end_seconds:.3f}"
            for item in anatomy.audio_timeline.events
            if round(item.end_seconds * fps) >= frame_range.start
            and round(item.start_seconds * fps) < frame_range.end
        ]

    @staticmethod
    def _techniques(
        motion: VideoMotionEvidence | None,
        transition_in: VideoTransitionType | None,
        transition_out: VideoTransitionType | None,
    ) -> list[str]:
        result = [item.value for item in (transition_in, transition_out) if item is not None]
        if motion is not None:
            result.extend(motion.camera_hypotheses)
            if motion.independent_motion_detected:
                result.append("independent-motion")
        return sorted(set(result))

    @staticmethod
    def _climax_shot(anatomy: VideoAnatomy) -> str:
        if anatomy.motion_evidence:
            return max(
                anatomy.motion_evidence,
                key=lambda item: item.motion_magnitude_peak,
            ).shot_id
        return anatomy.structure.shots[max(0, len(anatomy.structure.shots) - 2)].id

    @staticmethod
    def _provenance(**parameters: Any) -> Provenance:
        return Provenance(
            tool="aoa-editing-video-proposals",
            tool_version=__version__,
            parameters=parameters,
            deterministic=True,
        )
