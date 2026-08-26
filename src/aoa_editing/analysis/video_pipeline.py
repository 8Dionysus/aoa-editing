"""Application-facing orchestration for progressive Video Anatomy profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aoa_editing import __version__
from aoa_editing.analysis.video_anatomy import (
    StructuralAnatomyResult,
    VideoAnatomyService,
)
from aoa_editing.analysis.video_audio import VideoAudioTimelineService
from aoa_editing.analysis.video_motion import VideoMotionAnalysisService
from aoa_editing.analysis.video_semantics import VideoSemanticService
from aoa_editing.domain.models import (
    EvidenceLineageEdge,
    EvidenceRecord,
    FrameRange,
    Provenance,
    VideoAnatomy,
    VideoAnatomyProfile,
    VideoAnatomyStatus,
    VideoAudioTimelineEvidence,
    VideoMotionEvidence,
    VideoSamplingPlan,
    VideoVisualObservation,
    new_id,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.service import ProviderService


@dataclass(frozen=True, slots=True)
class VideoAnatomyRunResult:
    structural: StructuralAnatomyResult
    anatomy: VideoAnatomy
    evidence_records: tuple[EvidenceRecord, ...]


class VideoAnatomyPipelineService:
    """Run all stages required by a selected profile without mutating a timeline."""

    def __init__(
        self,
        store: ProjectStore,
        *,
        providers: ProviderService | None = None,
    ):
        self.store = store
        self.providers = providers

    def analyze(
        self,
        project_id: str,
        asset_id: str,
        *,
        profile: VideoAnatomyProfile = VideoAnatomyProfile.STRUCTURAL,
        analysis_range: FrameRange | None = None,
        pinned_frames: tuple[int, ...] = (),
        global_frame_budget: int | None = None,
        per_shot_frame_budget: int | None = None,
        artifact_budget_bytes: int | None = None,
        explicit_provider_opt_in: bool = False,
        focused_rescan_reasons: tuple[str, ...] = (),
        phase_hook: Callable[[str, float, list[str]], None] | None = None,
        resume_plan_id: str | None = None,
        resume_completed_phases: tuple[str, ...] = (),
        resume_evidence_refs: tuple[str, ...] = (),
    ) -> VideoAnatomyRunResult:
        completed = set(resume_completed_phases)
        resumed_records = self._resume_records(project_id, resume_evidence_refs)
        if resume_plan_id is not None and "structural" in completed:
            structural = self._load_structural_resume(
                project_id,
                asset_id,
                profile,
                resume_plan_id,
                resumed_records,
            )
            latest = structural.anatomy
            if latest.provenance.tool == "aoa-editing-video-anatomy-pipeline":
                return VideoAnatomyRunResult(
                    structural=structural,
                    anatomy=latest,
                    evidence_records=tuple(resumed_records),
                )
        else:
            structural = VideoAnatomyService(self.store).analyze_structural(
                project_id,
                asset_id,
                profile=profile,
                analysis_range=analysis_range,
                pinned_frames=pinned_frames,
                global_frame_budget=global_frame_budget,
                per_shot_frame_budget=per_shot_frame_budget,
                artifact_budget_bytes=artifact_budget_bytes,
                focused_rescan_reasons=focused_rescan_reasons,
            )
            self._notify(
                phase_hook,
                "structural",
                0.45,
                [item.id for item in structural.evidence_records],
            )
        plan = structural.plan
        asset = self.store.load_asset(project_id, asset_id)
        records = list(structural.evidence_records)
        visual: list[VideoVisualObservation] = []
        motion: list[VideoMotionEvidence] = []
        audio: VideoAudioTimelineEvidence | None = None
        coverage = dict(structural.anatomy.coverage_matrix)
        confidence = dict(structural.anatomy.confidence_summary)
        unresolved = list(structural.structure.ambiguous_ranges)
        reasons: list[str] = []

        audio_record = self._record_for_phase(resumed_records, plan.id, "video.audio-timeline")
        if "audio" in completed and audio_record is not None:
            audio = self.store.load_video_audio_timeline(project_id, plan.id)
        else:
            audio, audio_record = VideoAudioTimelineService(
                self.store,
                providers=self.providers,
            ).analyze(
                plan,
                transcribe=profile
                in {
                    VideoAnatomyProfile.QUICK,
                    VideoAnatomyProfile.SEMANTIC,
                    VideoAnatomyProfile.RECONSTRUCT,
                },
            )
            self._notify(phase_hook, "audio", 0.6, [audio_record.id])
        records.append(audio_record)
        audio_coverage = 1.0
        if audio.partial:
            unknown_duration = sum(end - start for start, end in audio.unknown_intervals)
            audio_coverage = max(0.0, 1.0 - unknown_duration / audio.duration_seconds)
        coverage["audio"] = audio_coverage
        confidence["audio"] = 0.65 if audio.partial else 1.0

        if profile in {VideoAnatomyProfile.SEMANTIC, VideoAnatomyProfile.RECONSTRUCT}:
            visual_record = self._record_for_phase(
                resumed_records, plan.id, "video.visual-observations"
            )
            if "semantic" in completed and visual_record is not None:
                visual = self.store.list_video_visual_observations(project_id, plan.id)
            else:
                visual, visual_record = VideoSemanticService(
                    self.store,
                    providers=self.providers,
                ).analyze(
                    plan,
                    structural.structure,
                    structural.frame_manifest,
                    explicit_opt_in=explicit_provider_opt_in,
                )
                self._notify(phase_hook, "semantic", 0.76, [visual_record.id])
            records.append(visual_record)
            visual_shots = {item.shot_id for item in visual}
            visual_coverage = len(visual_shots) / len(structural.structure.shots)
            coverage["visual"] = visual_coverage
            confidence["visual"] = (
                sum(item.confidence for item in visual) / len(visual) if visual else 0.0
            )
            for shot in structural.structure.shots:
                if shot.id not in visual_shots:
                    unresolved.append(shot.frame_range)
            if visual_coverage < 1:
                reasons.append("semantic vision did not cover every shot")

        if profile in {VideoAnatomyProfile.MOTION, VideoAnatomyProfile.RECONSTRUCT}:
            motion_record = self._record_for_phase(resumed_records, plan.id, "video.motion")
            if "motion" in completed and motion_record is not None:
                motion = self.store.list_video_motion_evidence(project_id, plan.id)
            else:
                motion, motion_record = VideoMotionAnalysisService(self.store).analyze(
                    plan,
                    structural.structure,
                )
                self._notify(phase_hook, "motion", 0.9, [motion_record.id])
            records.append(motion_record)
            motion_shots = {item.shot_id for item in motion}
            motion_coverage = len(motion_shots) / len(structural.structure.shots)
            coverage["motion"] = motion_coverage
            confidence["motion"] = (
                sum(item.confidence for item in motion) / len(motion) if motion else 0.0
            )
            if motion_coverage < 1:
                reasons.append("motion analysis did not cover every requested shot")

        if profile is VideoAnatomyProfile.RECONSTRUCT and audio.partial:
            reasons.append("reconstruct profile retains unresolved transcript intervals")
            for start, end in audio.unknown_intervals:
                frame_start = max(plan.analysis_range.start, round(start * plan.time_base.fps))
                frame_end = min(plan.analysis_range.end, round(end * plan.time_base.fps))
                if frame_end > frame_start:
                    unresolved.append(
                        FrameRange(start=frame_start, duration=frame_end - frame_start)
                    )

        unresolved = self._merge_ranges(unresolved)
        status = (
            VideoAnatomyStatus.COMPLETE
            if not reasons and not unresolved
            else (VideoAnatomyStatus.PARTIAL)
        )
        child_ids = [item.id for item in records]
        lineage = list(structural.anatomy.provenance_graph)
        final_id = new_id("videoanatomy")
        for record in records[len(structural.evidence_records) :]:
            lineage.append(
                EvidenceLineageEdge(
                    source_ref=record.id,
                    target_ref=final_id,
                    relation="supports",
                )
            )
        final = VideoAnatomy(
            id=final_id,
            project_id=project_id,
            asset_id=asset_id,
            source_sha256=asset.sha256,
            profile=profile,
            technical_metadata=asset.metadata,
            track_summary=structural.anatomy.track_summary,
            plan=plan,
            frame_manifest=structural.frame_manifest,
            structure=structural.structure,
            visual_observations=visual,
            motion_evidence=motion,
            audio_timeline=audio,
            recurring_motifs=self._motifs(visual, audio),
            unresolved_ranges=unresolved,
            contradictions=[],
            coverage_matrix=coverage,
            confidence_summary=confidence,
            provenance_graph=lineage,
            evidence_refs=sorted(set(child_ids)),
            status=status,
            incompleteness_reasons=(
                sorted(set([*reasons, "ambiguous structural ranges remain"]))
                if unresolved and not reasons
                else sorted(set(reasons))
            ),
            provenance=Provenance(
                tool="aoa-editing-video-anatomy-pipeline",
                tool_version=__version__,
                parameters={
                    "profile": profile.value,
                    "stage_count": len(records),
                    "focused": bool(focused_rescan_reasons),
                },
                deterministic=not bool(visual),
            ),
        )
        path = self.store.save_video_anatomy(final)
        record = EvidenceRecord(
            project_id=project_id,
            asset_id=asset_id,
            source_sha256=asset.sha256,
            kind="video.anatomy",
            confidence=min(final.confidence_summary.values()),
            payload={
                "anatomy_id": final.id,
                "anatomy_sha256": final.anatomy_sha256,
                "profile": profile.value,
                "status": final.status.value,
                "coverage_matrix": final.coverage_matrix,
                "incompleteness_reasons": final.incompleteness_reasons,
                "evidence_refs": final.evidence_refs,
            },
            artifacts=[str(path.relative_to(self.store.project_path(project_id)))],
            time_base=plan.time_base,
            ranges=[plan.analysis_range],
            provenance=final.provenance,
        )
        self.store.save_evidence(record)
        records.append(record)
        self._notify(phase_hook, "aggregate", 1.0, [record.id])
        return VideoAnatomyRunResult(
            structural=structural,
            anatomy=final,
            evidence_records=tuple(records),
        )

    def _load_structural_resume(
        self,
        project_id: str,
        asset_id: str,
        profile: VideoAnatomyProfile,
        plan_id: str,
        records: list[EvidenceRecord],
    ) -> StructuralAnatomyResult:
        plan = self.store.load_video_sampling_plan(project_id, plan_id)
        if plan.asset_id != asset_id or plan.profile is not profile:
            raise ValueError("resume plan differs from the requested asset or profile")
        manifest = self.store.load_video_frame_manifest(project_id, plan_id)
        structure = self.store.load_video_structure(project_id, plan_id)
        anatomy = self.store.load_video_anatomy(project_id, plan_id)
        structural_records = tuple(
            item
            for item in records
            if item.kind
            in {
                "video.scenes",
                "video.structure.detectors",
                "video.frame-samples",
                "video.structure",
                "video.anatomy",
            }
        )
        if not structural_records:
            raise ValueError("resume checkpoint lacks structural evidence records")
        contact_sheet = self.store.video_anatomy_path(project_id, plan_id) / "contact-sheet.jpg"
        if not contact_sheet.is_file():
            raise ValueError("resume checkpoint lacks its contact sheet")
        return StructuralAnatomyResult(
            plan=plan,
            frame_manifest=manifest,
            structure=structure,
            anatomy=anatomy,
            evidence_records=structural_records,
            contact_sheet_path=contact_sheet,
        )

    def _resume_records(
        self, project_id: str, evidence_refs: tuple[str, ...]
    ) -> list[EvidenceRecord]:
        wanted = set(evidence_refs)
        return [item for item in self.store.list_evidence(project_id) if item.id in wanted]

    @staticmethod
    def _record_for_phase(
        records: list[EvidenceRecord], plan_id: str, kind: str
    ) -> EvidenceRecord | None:
        return next(
            (
                item
                for item in reversed(records)
                if item.kind == kind and item.payload.get("plan_id") == plan_id
            ),
            None,
        )

    @staticmethod
    def _notify(
        hook: Callable[[str, float, list[str]], None] | None,
        phase: str,
        progress: float,
        evidence_refs: list[str],
    ) -> None:
        if hook is not None:
            hook(phase, progress, evidence_refs)

    def create_focused_plans(
        self,
        anatomy: VideoAnatomy,
        *,
        maximum_plans: int = 8,
    ) -> list[VideoSamplingPlan]:
        ranges = self._merge_ranges(
            [*anatomy.unresolved_ranges, *anatomy.structure.ambiguous_ranges]
        )[:maximum_plans]
        plans: list[VideoSamplingPlan] = []
        service = VideoAnatomyService(self.store)
        for frame_range in ranges:
            padded_start = max(anatomy.plan.analysis_range.start, frame_range.start - 6)
            padded_end = min(anatomy.plan.analysis_range.end, frame_range.end + 6)
            plans.append(
                service.create_plan(
                    anatomy.project_id,
                    anatomy.asset_id,
                    profile=anatomy.profile,
                    analysis_range=FrameRange(
                        start=padded_start,
                        duration=padded_end - padded_start,
                    ),
                    pinned_frames=(
                        frame_range.start,
                        max(frame_range.start, frame_range.end - 1),
                    ),
                    global_frame_budget=max(24, anatomy.plan.per_shot_frame_budget * 3),
                    per_shot_frame_budget=anatomy.plan.per_shot_frame_budget,
                    artifact_budget_bytes=min(
                        anatomy.plan.artifact_budget_bytes,
                        128 * 1024 * 1024,
                    ),
                    focused_rescan_reasons=(
                        f"resolve range {frame_range.start}:{frame_range.end}",
                        f"parent anatomy {anatomy.id}@{anatomy.anatomy_sha256}",
                    ),
                )
            )
        return plans

    @staticmethod
    def _merge_ranges(ranges: list[FrameRange]) -> list[FrameRange]:
        if not ranges:
            return []
        ordered = sorted(ranges, key=lambda item: (item.start, item.end))
        merged: list[FrameRange] = []
        start, end = ordered[0].start, ordered[0].end
        for item in ordered[1:]:
            if item.start <= end:
                end = max(end, item.end)
            else:
                merged.append(FrameRange(start=start, duration=end - start))
                start, end = item.start, item.end
        merged.append(FrameRange(start=start, duration=end - start))
        return merged

    @staticmethod
    def _motifs(
        visual: list[VideoVisualObservation],
        audio: VideoAudioTimelineEvidence | None,
    ) -> list[str]:
        motifs: list[str] = []
        descriptions = [item.description.casefold() for item in visual]
        for term in {word for text in descriptions for word in text.split() if len(word) >= 5}:
            if sum(term in text for text in descriptions) >= 2:
                motifs.append(f"recurring visual term: {term}")
        if audio is not None:
            accents = sum(item.kind in {"accent", "onset"} for item in audio.events)
            if accents >= 2:
                motifs.append(f"recurring audio accents: {accents}")
        return sorted(motifs)[:20]
