"""Standalone, evidence-first video decomposition for AoA Editing.

The structural pass deliberately has no model-provider dependency.  It scans the
requested source range, joins several conservative transition signals, extracts
budgeted review frames, and persists hash-bound evidence before any editorial
proposal exists.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from aoa_editing import __version__
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.domain.models import (
    CoverageInterval,
    EvidenceLineageEdge,
    EvidenceRecord,
    FrameRange,
    MotionTimeV2,
    Provenance,
    ShotEvidence,
    TransitionCandidate,
    TransitionDetectorSignal,
    VideoAnatomy,
    VideoAnatomyProfile,
    VideoAnatomyStatus,
    VideoCoverageReport,
    VideoFrameSample,
    VideoFrameSampleManifest,
    VideoSampleRole,
    VideoSamplingPlan,
    VideoStructureEvidence,
    VideoTransitionType,
    new_id,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore


class VideoAnatomyError(RuntimeError):
    """A structural pass cannot produce truthful, complete evidence."""


@dataclass(frozen=True, slots=True)
class StructuralAnatomyResult:
    plan: VideoSamplingPlan
    frame_manifest: VideoFrameSampleManifest
    structure: VideoStructureEvidence
    anatomy: VideoAnatomy
    evidence_records: tuple[EvidenceRecord, ...]
    contact_sheet_path: Path


@dataclass(frozen=True, slots=True)
class _FrameFeature:
    frame: int
    luminance: float
    histogram: NDArray[Any]
    edge_density: float
    pixel_change: float
    histogram_distance: float
    luminance_change: float
    edge_change: float
    directional_asymmetry: float
    combined_change: float


@dataclass(frozen=True, slots=True)
class _SampleCandidate:
    frame: int
    role: VideoSampleRole
    reasons: tuple[str, ...]
    shot_id: str
    sample_id: str


@dataclass(frozen=True, slots=True)
class _DedupMetrics:
    luminance_delta: float
    histogram_distance: float
    perceptual_hash_distance: int
    ssim_change: float
    edge_change: float
    motion_activity: float

    def as_dict(self) -> dict[str, float]:
        return {
            "luminance_delta": self.luminance_delta,
            "histogram_distance": self.histogram_distance,
            "perceptual_hash_distance": float(self.perceptual_hash_distance),
            "ssim_change": self.ssim_change,
            "edge_change": self.edge_change,
            "motion_activity": self.motion_activity,
        }


_PROTECTED_ROLES = {
    VideoSampleRole.FIRST_FRAME,
    VideoSampleRole.LAST_FRAME,
    VideoSampleRole.PRE_BOUNDARY,
    VideoSampleRole.BOUNDARY,
    VideoSampleRole.POST_BOUNDARY,
    VideoSampleRole.PINNED,
}

_ROLE_PRIORITY = {
    VideoSampleRole.INTERIOR: 0,
    VideoSampleRole.REPRESENTATIVE: 1,
    VideoSampleRole.POST_BOUNDARY: 2,
    VideoSampleRole.PRE_BOUNDARY: 2,
    VideoSampleRole.BOUNDARY: 3,
    VideoSampleRole.LAST_FRAME: 4,
    VideoSampleRole.FIRST_FRAME: 4,
    VideoSampleRole.PINNED: 5,
}

_PROFILE_DEFAULTS: dict[VideoAnatomyProfile, dict[str, int]] = {
    VideoAnatomyProfile.QUICK: {
        "global_frame_budget": 24,
        "per_shot_frame_budget": 4,
        "artifact_budget_bytes": 64 * 1024 * 1024,
    },
    VideoAnatomyProfile.STRUCTURAL: {
        "global_frame_budget": 96,
        "per_shot_frame_budget": 8,
        "artifact_budget_bytes": 256 * 1024 * 1024,
    },
    VideoAnatomyProfile.SEMANTIC: {
        "global_frame_budget": 128,
        "per_shot_frame_budget": 10,
        "artifact_budget_bytes": 384 * 1024 * 1024,
    },
    VideoAnatomyProfile.MOTION: {
        "global_frame_budget": 192,
        "per_shot_frame_budget": 14,
        "artifact_budget_bytes": 512 * 1024 * 1024,
    },
    VideoAnatomyProfile.RECONSTRUCT: {
        "global_frame_budget": 256,
        "per_shot_frame_budget": 18,
        "artifact_budget_bytes": 768 * 1024 * 1024,
    },
}


class VideoAnatomyService:
    """Create reproducible structural evidence inside one AoA Editing project."""

    def __init__(self, store: ProjectStore):
        self.store = store

    def create_plan(
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
        focused_rescan_reasons: tuple[str, ...] = (),
    ) -> VideoSamplingPlan:
        asset = self.store.load_asset(project_id, asset_id)
        metadata = asset.metadata
        if not metadata.has_video or metadata.frame_rate is None:
            raise VideoAnatomyError("Video Anatomy requires a video stream and frame rate")
        if metadata.duration_seconds is None or metadata.duration_seconds <= 0:
            raise VideoAnatomyError("Video Anatomy requires a positive measured duration")
        frame_timestamps: list[float] = []
        timing_path: Path | None = None
        timing_sha256: str | None = None
        if metadata.variable_frame_rate:
            source = self.store.asset_source_path(project_id, asset_id)
            frame_timestamps = self._probe_frame_timestamps(source)
            timing_payload: dict[str, object] = {
                "schema_version": "1.0.0",
                "project_id": project_id,
                "asset_id": asset.id,
                "source_sha256": asset.sha256,
                "stream_time_base": (
                    metadata.video_time_base.model_dump(mode="json")
                    if metadata.video_time_base is not None
                    else None
                ),
                "timestamps_seconds": frame_timestamps,
            }
            timing_path = self.store.save_video_frame_timestamps(
                project_id,
                asset.id,
                asset.sha256,
                timing_payload,
            )
            timing_sha256 = sha256_file(timing_path)
        source_frames = (
            len(frame_timestamps)
            or metadata.video_frame_count
            or max(1, round(metadata.duration_seconds * metadata.frame_rate.fps))
        )
        selected_range = analysis_range or FrameRange(start=0, duration=source_frames)
        if selected_range.end > source_frames + 1:
            raise VideoAnatomyError("requested analysis range exceeds measured source frames")
        defaults = _PROFILE_DEFAULTS[profile]
        pinned = tuple(sorted(set(pinned_frames)))
        plan = VideoSamplingPlan(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            duration_seconds=metadata.duration_seconds,
            time_base=metadata.frame_rate,
            stream_time_base=metadata.video_time_base,
            variable_frame_rate=metadata.variable_frame_rate,
            frame_timestamp_count=(len(frame_timestamps) if frame_timestamps else None),
            frame_timestamps_artifact=(
                self._relative(project_id, timing_path) if timing_path is not None else None
            ),
            frame_timestamps_sha256=timing_sha256,
            profile=profile,
            analysis_range=selected_range,
            strategies=[
                "scene-score",
                "keyframe",
                "histogram-change",
                "luma-progression",
                "motion-activity",
                "uniform-fallback",
            ],
            minimum_samples_per_shot=3,
            global_frame_budget=(
                global_frame_budget
                if global_frame_budget is not None
                else defaults["global_frame_budget"]
            ),
            per_shot_frame_budget=(
                per_shot_frame_budget
                if per_shot_frame_budget is not None
                else defaults["per_shot_frame_budget"]
            ),
            pinned_times=[MotionTimeV2(frame=frame) for frame in pinned],
            required_capabilities=[],
            focused_rescan_reasons=list(focused_rescan_reasons),
            artifact_budget_bytes=(
                artifact_budget_bytes
                if artifact_budget_bytes is not None
                else defaults["artifact_budget_bytes"]
            ),
        )
        self.store.save_video_sampling_plan(plan)
        return plan

    def analyze_structural(
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
        focused_rescan_reasons: tuple[str, ...] = (),
    ) -> StructuralAnatomyResult:
        plan = self.create_plan(
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
        asset = self.store.load_asset(project_id, asset_id)
        source = self.store.asset_source_path(project_id, asset_id)
        frame_timestamps = self._load_frame_timestamps(plan)
        if plan.profile is VideoAnatomyProfile.QUICK:
            features = self._scan_quick_features(source, plan, frame_timestamps)
            transitions, thresholds = self._detect_quick_transitions(
                features,
                plan,
                frame_timestamps,
            )
            scene_evidence = self._quick_candidate_evidence(
                project_id,
                asset_id,
                plan,
                features,
                transitions,
                thresholds,
            )
            scan_operation = "sparse-keyframe-uniform-structural-scan"
        else:
            scene_evidence = self._legacy_scene_evidence(project_id, asset_id)
            features = self._scan_features(source, plan.analysis_range)
            transitions, thresholds = self._detect_transitions(
                features,
                plan,
                scene_evidence,
                frame_timestamps,
            )
            scan_operation = "all-frame-structural-scan"
        boundaries = self._shot_boundaries(transitions, plan)
        shot_ranges = self._shot_ranges(plan.analysis_range, boundaries)
        candidates = self._select_samples(plan, shot_ranges, transitions)
        frames = self._decode_selected_frames(source, plan, candidates)

        root = self.store.video_anatomy_path(project_id, plan.id)
        signals_path = root / "detector-signals.json"
        signal_payload = {
            "schema_version": "1.0.0",
            "project_id": project_id,
            "asset_id": asset_id,
            "source_sha256": asset.sha256,
            "plan_id": plan.id,
            "plan_sha256": plan.plan_sha256,
            "thresholds": thresholds,
            "frames": [self._feature_payload(item) for item in features],
        }
        self._atomic_json(signals_path, signal_payload)
        signal_relative = self._relative(project_id, signals_path)
        plan_relative = self._relative(project_id, root / "sampling-plan.json")
        timing_artifacts = (
            [plan.frame_timestamps_artifact] if plan.frame_timestamps_artifact is not None else []
        )
        detector_record = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="video.structure.detectors",
            confidence=1.0,
            payload={
                "plan_id": plan.id,
                "plan_sha256": plan.plan_sha256,
                "decoded_frame_count": len(features),
                "candidate_count": len(transitions),
                "thresholds": thresholds,
                "scene_candidate_evidence_id": scene_evidence.id,
                "scan_mode": "sparse" if plan.profile is VideoAnatomyProfile.QUICK else "all-frame",
                "decoded_ratio": round(len(features) / plan.analysis_range.duration, 8),
                "variable_frame_rate": plan.variable_frame_rate,
                "frame_timestamp_count": plan.frame_timestamp_count,
                "frame_timestamps_sha256": plan.frame_timestamps_sha256,
            },
            artifacts=[plan_relative, signal_relative, *timing_artifacts],
            time_base=plan.time_base,
            ranges=[plan.analysis_range],
            provenance=self._provenance(
                operation=scan_operation,
                algorithms=[
                    "pixel-change",
                    "hsv-histogram-distance",
                    "luma-progression",
                    "edge-change",
                    "directional-asymmetry",
                    "ffmpeg-scene-score",
                ],
            ),
        )
        self.store.save_evidence(detector_record)

        samples, kept_frames, artifact_bytes = self._materialize_samples(
            project_id,
            plan,
            candidates,
            frames,
            frame_timestamps,
        )
        contact_sheet = root / "contact-sheet.jpg"
        self._write_contact_sheet(contact_sheet, samples, kept_frames, plan)
        contact_relative = self._relative(project_id, contact_sheet)
        manifest = VideoFrameSampleManifest(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            plan_id=plan.id,
            plan_sha256=plan.plan_sha256,
            samples=samples,
            selected_count=sum(item.kept for item in samples),
            duplicate_count=sum(not item.kept for item in samples),
            artifact_bytes=artifact_bytes,
            provenance=self._provenance(
                operation="budgeted-frame-extraction",
                frame_budget=plan.global_frame_budget,
                artifact_budget_bytes=plan.artifact_budget_bytes,
                protected_sample_count=sum(item.role in _PROTECTED_ROLES for item in samples),
            ),
        )
        manifest_path = self.store.save_video_frame_manifest(manifest)
        manifest_relative = self._relative(project_id, manifest_path)
        retained_artifacts = [
            item.artifact_path for item in samples if item.artifact_path is not None
        ]
        frame_record = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="video.frame-samples",
            confidence=1.0,
            payload={
                "plan_id": plan.id,
                "plan_sha256": plan.plan_sha256,
                "manifest_path": manifest_relative,
                "manifest_sha256": manifest.manifest_sha256,
                "selected_count": manifest.selected_count,
                "duplicate_count": manifest.duplicate_count,
                "artifact_bytes": manifest.artifact_bytes,
                "contact_sheet_sha256": sha256_file(contact_sheet),
            },
            artifacts=[manifest_relative, contact_relative, *retained_artifacts],
            time_base=plan.time_base,
            ranges=[plan.analysis_range],
            provenance=manifest.provenance,
        )
        self.store.save_evidence(frame_record)

        structure = self._build_structure(
            plan,
            shot_ranges,
            transitions,
            samples,
            detector_record.id,
            thresholds,
        )
        structure_path = self.store.save_video_structure(structure)
        structure_relative = self._relative(project_id, structure_path)
        structure_record = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="video.structure",
            confidence=self._structure_confidence(structure),
            payload={
                "plan_id": plan.id,
                "plan_sha256": plan.plan_sha256,
                "structure_id": structure.id,
                "shot_count": len(structure.shots),
                "transition_count": len(structure.transitions),
                "coverage_ratio": structure.coverage.coverage_ratio,
                "ambiguous_range_count": len(structure.ambiguous_ranges),
                "scene_candidate_evidence_id": scene_evidence.id,
            },
            artifacts=[structure_relative],
            time_base=plan.time_base,
            ranges=[plan.analysis_range],
            provenance=structure.provenance,
        )
        self.store.save_evidence(structure_record)

        evidence_records = (
            scene_evidence,
            detector_record,
            frame_record,
            structure_record,
        )
        structural_only = plan.profile in {
            VideoAnatomyProfile.QUICK,
            VideoAnatomyProfile.STRUCTURAL,
        }
        anatomy = VideoAnatomy(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            profile=plan.profile,
            technical_metadata=asset.metadata,
            track_summary={
                "video_streams": 1 if asset.metadata.has_video else 0,
                "audio_streams": 1 if asset.metadata.has_audio else 0,
                "video_codec": asset.metadata.video_codec,
                "audio_codec": asset.metadata.audio_codec,
                "pixel_format": asset.metadata.pixel_format,
                "sample_rate": asset.metadata.sample_rate,
                "channels": asset.metadata.channels,
                "nominal_frame_rate": (
                    asset.metadata.nominal_frame_rate.model_dump(mode="json")
                    if asset.metadata.nominal_frame_rate is not None
                    else None
                ),
                "stream_time_base": (
                    asset.metadata.video_time_base.model_dump(mode="json")
                    if asset.metadata.video_time_base is not None
                    else None
                ),
                "variable_frame_rate": asset.metadata.variable_frame_rate,
                "video_frame_count": asset.metadata.video_frame_count,
            },
            plan=plan,
            frame_manifest=manifest,
            structure=structure,
            visual_observations=[],
            motion_evidence=[],
            audio_timeline=None,
            recurring_motifs=[],
            unresolved_ranges=[],
            contradictions=[],
            coverage_matrix={
                "structure": structure.coverage.coverage_ratio,
                "sampling": 1.0,
            },
            confidence_summary={
                "structure": self._structure_confidence(structure),
                "coverage": structure.coverage.coverage_ratio,
                "sampling": 1.0 if manifest.selected_count else 0.0,
            },
            provenance_graph=[
                EvidenceLineageEdge(
                    source_ref=scene_evidence.id,
                    target_ref=detector_record.id,
                    relation="supports",
                ),
                EvidenceLineageEdge(
                    source_ref=detector_record.id,
                    target_ref=structure_record.id,
                    relation="supports",
                ),
                EvidenceLineageEdge(
                    source_ref=frame_record.id,
                    target_ref=structure_record.id,
                    relation="supports",
                ),
            ],
            evidence_refs=[item.id for item in evidence_records],
            status=(VideoAnatomyStatus.COMPLETE if structural_only else VideoAnatomyStatus.PARTIAL),
            incompleteness_reasons=(
                [] if structural_only else ["deeper profile stages have not run yet"]
            ),
            provenance=self._provenance(operation="structural-anatomy-aggregation"),
        )
        anatomy_path = self.store.save_video_anatomy(anatomy)
        anatomy_record = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="video.anatomy",
            confidence=min(anatomy.confidence_summary.values()),
            payload={
                "anatomy_id": anatomy.id,
                "anatomy_sha256": anatomy.anatomy_sha256,
                "profile": anatomy.profile.value,
                "status": anatomy.status.value,
                "evidence_refs": anatomy.evidence_refs,
            },
            artifacts=[self._relative(project_id, anatomy_path)],
            time_base=plan.time_base,
            ranges=[plan.analysis_range],
            provenance=anatomy.provenance,
        )
        self.store.save_evidence(anatomy_record)
        return StructuralAnatomyResult(
            plan=plan,
            frame_manifest=manifest,
            structure=structure,
            anatomy=anatomy,
            evidence_records=(*evidence_records, anatomy_record),
            contact_sheet_path=contact_sheet,
        )

    def _legacy_scene_evidence(self, project_id: str, asset_id: str) -> EvidenceRecord:
        asset = self.store.load_asset(project_id, asset_id)
        existing = next(
            (
                item
                for item in self.store.list_effective_evidence(project_id)
                if item.asset_id == asset_id
                and item.source_sha256 == asset.sha256
                and item.kind == "video.scenes"
            ),
            None,
        )
        if existing is not None:
            return existing
        records = AnalysisService(self.store).analyze(project_id, asset_id)
        scene = next((item for item in records if item.kind == "video.scenes"), None)
        if scene is None:
            failure = next(
                (
                    item
                    for item in records
                    if item.kind == "analysis.failure"
                    and item.payload.get("failed_kind") == "video.scenes"
                ),
                None,
            )
            detail = failure.payload.get("message") if failure is not None else "missing result"
            raise VideoAnatomyError(f"legacy scene detector failed: {detail}")
        return scene

    def _scan_quick_features(
        self,
        source: Path,
        plan: VideoSamplingPlan,
        frame_timestamps: list[float],
    ) -> list[_FrameFeature]:
        """Decode only bounded keyframe and uniform samples for the quick profile."""

        uniform = self._even_frames(plan.analysis_range, plan.global_frame_budget)
        keyframes = self._probe_keyframe_indices(source, plan, frame_timestamps)
        if len(keyframes) > plan.global_frame_budget:
            positions = self._even_frames(
                FrameRange(start=0, duration=len(keyframes)),
                plan.global_frame_budget,
            )
            keyframes = [keyframes[index] for index in positions]
        requested = sorted(
            {
                plan.analysis_range.start,
                plan.analysis_range.end - 1,
                *uniform,
                *keyframes,
                *(item.frame for item in plan.pinned_times),
            }
        )
        decoded = self._decode_frame_indices(source, requested)
        features: list[_FrameFeature] = []
        previous_gray: NDArray[Any] | None = None
        previous_hist: NDArray[Any] | None = None
        previous_luminance: float | None = None
        previous_edges: NDArray[Any] | None = None
        for frame_index in requested:
            frame = decoded[frame_index]
            feature, previous_gray, previous_hist, previous_luminance, previous_edges = (
                self._measure_feature(
                    frame_index,
                    frame,
                    previous_gray,
                    previous_hist,
                    previous_luminance,
                    previous_edges,
                )
            )
            features.append(feature)
        return features

    def _probe_keyframe_indices(
        self,
        source: Path,
        plan: VideoSamplingPlan,
        frame_timestamps: list[float],
    ) -> list[int]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(source),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise VideoAnatomyError("cannot inspect keyframes: " + result.stderr[-2000:])
        try:
            times = [
                float(item["best_effort_timestamp_time"])
                for item in json.loads(result.stdout).get("frames", [])
                if item.get("best_effort_timestamp_time") not in {None, "N/A"}
            ]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise VideoAnatomyError("ffprobe returned malformed keyframe timestamps") from error
        if frame_timestamps:
            frames = [self._nearest_timestamp_index(frame_timestamps, value) for value in times]
        else:
            frames = [round(value * plan.time_base.fps) for value in times]
        return sorted(
            {
                frame
                for frame in frames
                if plan.analysis_range.start <= frame < plan.analysis_range.end
            }
        )

    @staticmethod
    def _nearest_timestamp_index(timestamps: list[float], value: float) -> int:
        right = bisect_left(timestamps, value)
        if right <= 0:
            return 0
        if right >= len(timestamps):
            return len(timestamps) - 1
        left = right - 1
        return left if abs(timestamps[left] - value) <= abs(timestamps[right] - value) else right

    def _quick_candidate_evidence(
        self,
        project_id: str,
        asset_id: str,
        plan: VideoSamplingPlan,
        features: list[_FrameFeature],
        transitions: list[TransitionCandidate],
        thresholds: dict[str, float],
    ) -> EvidenceRecord:
        asset = self.store.load_asset(project_id, asset_id)
        record = EvidenceRecord(
            project_id=project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="video.structure.quick-candidates",
            confidence=0.45 if transitions else 1.0,
            payload={
                "plan_id": plan.id,
                "plan_sha256": plan.plan_sha256,
                "decoded_frame_count": len(features),
                "source_frame_count": plan.analysis_range.duration,
                "candidate_frames": [item.frame for item in transitions],
                "thresholds": thresholds,
                "limitations": [
                    "boundaries between sparse observations are approximate",
                    "run structural or reconstruct for dense transition classification",
                ],
            },
            time_base=plan.time_base,
            ranges=[plan.analysis_range],
            provenance=self._provenance(
                operation="quick-keyframe-uniform-candidate-scan",
                deterministic=True,
                keyframe_budget=plan.global_frame_budget,
                uniform_budget=plan.global_frame_budget,
            ),
        )
        self.store.save_evidence(record)
        return record

    def _detect_quick_transitions(
        self,
        features: list[_FrameFeature],
        plan: VideoSamplingPlan,
        frame_timestamps: list[float],
    ) -> tuple[list[TransitionCandidate], dict[str, float]]:
        changes = np.asarray([item.combined_change for item in features[1:]], dtype=np.float64)
        median = float(np.median(changes)) if changes.size else 0.0
        mad = float(np.median(np.abs(changes - median))) if changes.size else 0.0
        threshold = min(0.72, max(0.2, median + max(0.1, 5.0 * mad)))
        transitions: list[TransitionCandidate] = []
        for previous, item in pairwise(features):
            if item.combined_change < threshold or item.histogram_distance < 0.12:
                continue
            frame = max(
                plan.analysis_range.start + 1,
                min(plan.analysis_range.end - 1, round((previous.frame + item.frame) / 2)),
            )
            transitions.append(
                TransitionCandidate(
                    frame=frame,
                    time_seconds=self._frame_time_seconds(plan, frame, frame_timestamps),
                    window=FrameRange(
                        start=previous.frame,
                        duration=max(1, item.frame - previous.frame + 1),
                    ),
                    transition_type=VideoTransitionType.UNKNOWN,
                    confidence=min(0.5, max(0.3, item.combined_change)),
                    detector_signals=[
                        TransitionDetectorSignal(
                            detector="sparse-hsv-histogram-change",
                            score=item.histogram_distance,
                            supports=[
                                VideoTransitionType.HARD_CUT,
                                VideoTransitionType.CROSS_DISSOLVE,
                                VideoTransitionType.CONTINUOUS_MOTION,
                                VideoTransitionType.UNKNOWN,
                            ],
                            limitations=[
                                "no frames were decoded inside interval "
                                f"{previous.frame}:{item.frame}"
                            ],
                        ),
                        TransitionDetectorSignal(
                            detector="sparse-combined-change",
                            score=item.combined_change,
                            supports=[
                                VideoTransitionType.HARD_CUT,
                                VideoTransitionType.CROSS_DISSOLVE,
                                VideoTransitionType.CONTINUOUS_MOTION,
                                VideoTransitionType.UNKNOWN,
                            ],
                            limitations=["sparse observations cannot classify transition shape"],
                        ),
                    ],
                    competing_types=[
                        VideoTransitionType.HARD_CUT,
                        VideoTransitionType.CROSS_DISSOLVE,
                        VideoTransitionType.CONTINUOUS_MOTION,
                    ],
                    ambiguity_reasons=[
                        "quick profile localizes a change interval but does not classify it"
                    ],
                )
            )
        return transitions, {
            "sparse_combined_change": round(threshold, 8),
            "sparse_median": round(median, 8),
            "sparse_median_absolute_deviation": round(mad, 8),
            "sparse_histogram_minimum": 0.12,
        }

    def _measure_feature(
        self,
        frame_index: int,
        frame: NDArray[Any],
        previous_gray: NDArray[Any] | None,
        previous_hist: NDArray[Any] | None,
        previous_luminance: float | None,
        previous_edges: NDArray[Any] | None,
    ) -> tuple[
        _FrameFeature,
        NDArray[Any],
        NDArray[Any],
        float,
        NDArray[Any],
    ]:
        small = self._analysis_frame(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten().astype(np.float32)
        edges = cv2.Canny(gray, 80, 160)
        luminance = float(np.mean(gray) / 255.0)
        edge_density = float(np.mean(edges > 0))
        if previous_gray is None:
            pixel_change = 0.0
            histogram_distance = 0.0
            luminance_change = 0.0
            edge_change = 0.0
            directional_asymmetry = 0.0
        else:
            assert previous_hist is not None
            assert previous_luminance is not None
            assert previous_edges is not None
            absolute = cv2.absdiff(gray, previous_gray).astype(np.float32) / 255.0
            pixel_change = float(np.mean(absolute))
            histogram_distance = float(
                max(
                    0.0,
                    min(
                        1.0,
                        cv2.compareHist(previous_hist, histogram, cv2.HISTCMP_BHATTACHARYYA),
                    ),
                )
            )
            luminance_change = abs(luminance - previous_luminance)
            edge_change = float(np.mean(cv2.absdiff(edges, previous_edges)) / 255.0)
            midpoint = absolute.shape[1] // 2
            left = float(np.mean(absolute[:, :midpoint])) if midpoint else pixel_change
            right = float(np.mean(absolute[:, midpoint:]))
            directional_asymmetry = min(1.0, abs(left - right) * 4.0)
        combined = min(
            1.0,
            0.45 * pixel_change
            + 0.35 * histogram_distance
            + 0.12 * luminance_change
            + 0.08 * edge_change,
        )
        return (
            _FrameFeature(
                frame=frame_index,
                luminance=luminance,
                histogram=histogram,
                edge_density=edge_density,
                pixel_change=pixel_change,
                histogram_distance=histogram_distance,
                luminance_change=luminance_change,
                edge_change=edge_change,
                directional_asymmetry=directional_asymmetry,
                combined_change=combined,
            ),
            gray,
            histogram,
            luminance,
            edges,
        )

    @staticmethod
    def _decode_frame_indices(
        source: Path,
        requested: list[int],
    ) -> dict[int, NDArray[Any]]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoAnatomyError(f"cannot open video for sparse decode: {source}")
        frames: dict[int, NDArray[Any]] = {}
        try:
            for frame_index in requested:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if ok:
                    frames[frame_index] = cast(NDArray[Any], frame)
        finally:
            capture.release()
        missing = sorted(set(requested) - set(frames))
        if missing:
            raise VideoAnatomyError(f"could not decode sparse frames: {missing[:10]}")
        return frames

    def _scan_features(self, source: Path, frame_range: FrameRange) -> list[_FrameFeature]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoAnatomyError(f"cannot open video for structural scan: {source}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start)
        features: list[_FrameFeature] = []
        previous_gray: NDArray[Any] | None = None
        previous_hist: NDArray[Any] | None = None
        previous_luminance: float | None = None
        previous_edges: NDArray[Any] | None = None
        try:
            for frame_index in range(frame_range.start, frame_range.end):
                ok, frame = capture.read()
                if not ok:
                    break
                decoded = cast(NDArray[Any], frame)
                small = self._analysis_frame(decoded)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
                histogram = cv2.normalize(histogram, histogram).flatten().astype(np.float32)
                edges = cv2.Canny(gray, 80, 160)
                luminance = float(np.mean(gray) / 255.0)
                edge_density = float(np.mean(edges > 0))
                if previous_gray is None:
                    pixel_change = 0.0
                    histogram_distance = 0.0
                    luminance_change = 0.0
                    edge_change = 0.0
                    directional_asymmetry = 0.0
                else:
                    assert previous_hist is not None
                    assert previous_luminance is not None
                    assert previous_edges is not None
                    absolute = cv2.absdiff(gray, previous_gray).astype(np.float32) / 255.0
                    pixel_change = float(np.mean(absolute))
                    histogram_distance = float(
                        max(
                            0.0,
                            min(
                                1.0,
                                cv2.compareHist(
                                    previous_hist, histogram, cv2.HISTCMP_BHATTACHARYYA
                                ),
                            ),
                        )
                    )
                    luminance_change = abs(luminance - float(previous_luminance))
                    edge_change = float(np.mean(cv2.absdiff(edges, previous_edges)) / 255.0)
                    midpoint = absolute.shape[1] // 2
                    left = float(np.mean(absolute[:, :midpoint])) if midpoint else pixel_change
                    right = float(np.mean(absolute[:, midpoint:]))
                    directional_asymmetry = min(1.0, abs(left - right) * 4.0)
                combined = min(
                    1.0,
                    0.45 * pixel_change
                    + 0.35 * histogram_distance
                    + 0.12 * luminance_change
                    + 0.08 * edge_change,
                )
                features.append(
                    _FrameFeature(
                        frame=frame_index,
                        luminance=luminance,
                        histogram=histogram,
                        edge_density=edge_density,
                        pixel_change=pixel_change,
                        histogram_distance=histogram_distance,
                        luminance_change=luminance_change,
                        edge_change=edge_change,
                        directional_asymmetry=directional_asymmetry,
                        combined_change=combined,
                    )
                )
                previous_gray = gray
                previous_hist = histogram
                previous_luminance = luminance
                previous_edges = edges
        finally:
            capture.release()
        if len(features) != frame_range.duration:
            raise VideoAnatomyError(
                "decoded structural coverage is incomplete: "
                f"expected {frame_range.duration}, got {len(features)}"
            )
        return features

    def _detect_transitions(
        self,
        features: list[_FrameFeature],
        plan: VideoSamplingPlan,
        legacy_scene: EvidenceRecord,
        frame_timestamps: list[float],
    ) -> tuple[list[TransitionCandidate], dict[str, float]]:
        changes = np.asarray([item.combined_change for item in features[1:]], dtype=np.float64)
        median = float(np.median(changes)) if changes.size else 0.0
        mad = float(np.median(np.abs(changes - median))) if changes.size else 0.0
        threshold = min(0.65, max(0.16, median + max(0.08, 6.0 * mad)))
        legacy_frames = {
            round(float(value) * plan.time_base.fps)
            for value in legacy_scene.payload.get("cut_times_seconds", [])
        }
        raw: set[int] = set()
        impulse_candidates: set[int] = set()
        for index in range(1, len(features)):
            item = features[index]
            left = features[index - 1].combined_change
            right = features[index + 1].combined_change if index + 1 < len(features) else 0.0
            impulse_candidate = (
                item.combined_change >= threshold
                and item.combined_change >= left
                and item.combined_change >= right
            ) or item.frame in legacy_frames
            if impulse_candidate:
                raw.add(index)
                impulse_candidates.add(index)

        # Gradual transitions and whip-like moves often remain below the impulse
        # threshold on every individual frame.  Admit a dense candidate window
        # only when change is sustained; this is intentionally different from
        # lowering the global cut threshold and avoids turning ordinary noise
        # into confident cuts.
        sustained_threshold = 0.035
        sustained: list[int] = []
        for index in range(1, len(features)):
            if features[index].combined_change >= sustained_threshold:
                sustained.append(index)
                continue
            if (
                len(sustained) >= 4
                and sum(features[item].combined_change for item in sustained) >= 0.25
            ):
                raw.update(sustained)
            sustained = []
        if (
            len(sustained) >= 4
            and sum(features[item].combined_change for item in sustained) >= 0.25
        ):
            raw.update(sustained)
        isolated_impulses = {
            index
            for index in impulse_candidates
            if features[index].combined_change >= 0.35
            and max(
                features[index - 1].combined_change if index > 0 else 0.0,
                (features[index + 1].combined_change if index + 1 < len(features) else 0.0),
            )
            <= max(0.12, features[index].combined_change * 0.55)
        }
        grouped: list[list[int]] = []
        for index in sorted(raw - isolated_impulses):
            if grouped and index - grouped[-1][-1] <= 1:
                grouped[-1].append(index)
            else:
                grouped.append([index])
        grouped.extend([[index] for index in isolated_impulses])
        grouped.sort(key=lambda item: item[0])
        transitions: list[TransitionCandidate] = []
        for group in grouped:
            transition = self._classify_transition(
                features,
                group,
                plan,
                legacy_frames,
                frame_timestamps,
            )
            transitions.append(transition)
        return transitions, {
            "combined_change": round(threshold, 8),
            "median": round(median, 8),
            "median_absolute_deviation": round(mad, 8),
            "sustained_change_threshold": sustained_threshold,
            "sustained_change_minimum_frames": 4.0,
            "sustained_change_minimum_sum": 0.25,
            "minimum_candidate_separation_frames": 1.0,
        }

    def _classify_transition(
        self,
        features: list[_FrameFeature],
        group: list[int],
        plan: VideoSamplingPlan,
        legacy_frames: set[int],
        frame_timestamps: list[float],
    ) -> TransitionCandidate:
        if not group:
            raise VideoAnatomyError("transition classifier received an empty window")
        midpoint = group[len(group) // 2]
        index = (
            midpoint
            if len(group) >= 4
            else max(group, key=lambda candidate: features[candidate].combined_change)
        )
        item = features[index]
        around_start = max(1, group[0] - 3)
        around_end = min(len(features), group[-1] + 4)
        around = features[around_start:around_end]
        group_frames = {features[candidate].frame for candidate in group}
        neighbors = [value.combined_change for value in around if value.frame not in group_frames]
        neighbor_mean = float(np.mean(neighbors)) if neighbors else 0.0
        lumas = [value.luminance for value in around]
        group_features = [features[candidate] for candidate in group]
        group_span = group[-1] - group[0] + 1
        pre = features[group[0] - 1] if group[0] > 0 else None
        post = features[group[-1] + 1] if group[-1] + 1 < len(features) else None
        maximum_luma = max(value.luminance for value in group_features)
        directional_peak = max(value.directional_asymmetry for value in group_features)
        group_histogram_mean = float(
            np.mean([value.histogram_distance for value in group_features])
        )
        group_pixel_mean = float(np.mean([value.pixel_change for value in group_features]))
        supports: list[VideoTransitionType]
        transition_type = VideoTransitionType.UNKNOWN
        ambiguity: list[str] = []
        competing: list[VideoTransitionType] = []
        if (
            group_span <= 3
            and pre is not None
            and post is not None
            and maximum_luma - max(pre.luminance, post.luminance) >= 0.3
            and abs(pre.luminance - post.luminance) <= 0.08
            and item.pixel_change >= 0.15
        ):
            transition_type = VideoTransitionType.FLASH
            confidence = min(
                0.98,
                0.65
                + 0.2 * (maximum_luma - max(pre.luminance, post.luminance))
                + 0.1 * item.pixel_change,
            )
        elif (
            group_span >= 4
            and len(lumas) >= 5
            and self._monotonic_ratio(lumas) >= 0.75
            and min(lumas) < 0.12
        ):
            transition_type = (
                VideoTransitionType.FADE_OUT
                if lumas[-1] < lumas[0]
                else VideoTransitionType.FADE_IN
            )
            confidence = min(0.9, 0.55 + self._monotonic_ratio(lumas) * 0.35)
        elif group_span >= 24:
            transition_type = VideoTransitionType.CONTINUOUS_MOTION
            confidence = min(0.82, 0.5 + group_pixel_mean + group_histogram_mean * 0.2)
        elif group_span >= 4 and directional_peak >= 0.25 and group_pixel_mean >= 0.04:
            transition_type = VideoTransitionType.WIPE_OR_DIRECTIONAL
            confidence = min(0.88, 0.52 + directional_peak * 0.32)
        elif group_span >= 4 and group_histogram_mean >= 0.08:
            transition_type = VideoTransitionType.CROSS_DISSOLVE
            confidence = min(0.88, 0.52 + group_histogram_mean * 0.25)
        elif (
            group_span <= 2
            and item.histogram_distance >= 0.35
            and neighbor_mean <= max(0.08, item.combined_change * 0.45)
        ):
            transition_type = VideoTransitionType.HARD_CUT
            confidence = min(
                0.99,
                0.55
                + 0.25 * item.histogram_distance
                + 0.15 * item.pixel_change
                + (0.05 if any(abs(item.frame - frame) <= 1 for frame in legacy_frames) else 0),
            )
        elif item.directional_asymmetry >= 0.3 and item.pixel_change >= 0.05:
            transition_type = VideoTransitionType.WIPE_OR_DIRECTIONAL
            confidence = min(0.85, 0.5 + item.directional_asymmetry * 0.35)
        elif neighbor_mean >= 0.035 and item.histogram_distance >= 0.12:
            transition_type = VideoTransitionType.CROSS_DISSOLVE
            confidence = min(0.85, 0.5 + item.histogram_distance * 0.25 + neighbor_mean)
        elif item.pixel_change >= 0.04 and item.histogram_distance < 0.12:
            transition_type = VideoTransitionType.CONTINUOUS_MOTION
            confidence = min(0.75, 0.45 + item.pixel_change)
        else:
            confidence = min(0.6, max(0.2, item.combined_change))
            ambiguity.append(
                "available change signals do not separate a transition from content motion"
            )
            competing = [
                VideoTransitionType.HARD_CUT,
                VideoTransitionType.CROSS_DISSOLVE,
                VideoTransitionType.CONTINUOUS_MOTION,
            ]
        supports = [
            VideoTransitionType.HARD_CUT,
            VideoTransitionType.CROSS_DISSOLVE,
            VideoTransitionType.FLASH,
            VideoTransitionType.WIPE_OR_DIRECTIONAL,
            VideoTransitionType.CONTINUOUS_MOTION,
            VideoTransitionType.UNKNOWN,
        ]
        luma_supports = [
            VideoTransitionType.FADE_IN,
            VideoTransitionType.FADE_OUT,
            VideoTransitionType.FLASH,
            VideoTransitionType.UNKNOWN,
        ]
        start = max(plan.analysis_range.start, features[group[0]].frame - 4)
        end = min(plan.analysis_range.end, features[group[-1]].frame + 5)
        return TransitionCandidate(
            frame=item.frame,
            time_seconds=self._frame_time_seconds(plan, item.frame, frame_timestamps),
            window=FrameRange(start=start, duration=end - start),
            transition_type=transition_type,
            confidence=confidence,
            detector_signals=[
                TransitionDetectorSignal(
                    detector="combined-frame-change",
                    score=item.combined_change,
                    supports=supports,
                    limitations=["content motion can resemble a transition"],
                ),
                TransitionDetectorSignal(
                    detector="hsv-histogram-distance",
                    score=item.histogram_distance,
                    supports=supports,
                    limitations=["spatial layout is not represented by this signal"],
                ),
                TransitionDetectorSignal(
                    detector="luminance-progression",
                    score=min(1.0, item.luminance_change * 4.0),
                    supports=luma_supports,
                    limitations=["dark content can resemble a fade"],
                ),
                TransitionDetectorSignal(
                    detector="directional-change-asymmetry",
                    score=item.directional_asymmetry,
                    supports=[
                        VideoTransitionType.WIPE_OR_DIRECTIONAL,
                        VideoTransitionType.CONTINUOUS_MOTION,
                        VideoTransitionType.UNKNOWN,
                    ],
                ),
            ],
            competing_types=competing,
            ambiguity_reasons=ambiguity,
        )

    @staticmethod
    def _shot_boundaries(
        transitions: list[TransitionCandidate], plan: VideoSamplingPlan
    ) -> list[int]:
        accepted = {
            VideoTransitionType.HARD_CUT,
            VideoTransitionType.FADE_IN,
            VideoTransitionType.FADE_OUT,
            VideoTransitionType.CROSS_DISSOLVE,
            VideoTransitionType.WIPE_OR_DIRECTIONAL,
        }
        return sorted(
            {
                item.frame
                for item in transitions
                if (
                    (
                        item.transition_type in accepted
                        and item.confidence >= 0.55
                    )
                    or (
                        plan.profile is VideoAnatomyProfile.QUICK
                        and item.transition_type is VideoTransitionType.UNKNOWN
                        and item.confidence >= 0.3
                    )
                )
                and plan.analysis_range.start < item.frame < plan.analysis_range.end
            }
        )

    @staticmethod
    def _shot_ranges(frame_range: FrameRange, boundaries: list[int]) -> list[FrameRange]:
        points = [frame_range.start, *boundaries, frame_range.end]
        return [
            FrameRange(start=start, duration=end - start)
            for start, end in pairwise(points)
            if end > start
        ]

    def _select_samples(
        self,
        plan: VideoSamplingPlan,
        shot_ranges: list[FrameRange],
        transitions: list[TransitionCandidate],
    ) -> list[_SampleCandidate]:
        selected: dict[int, tuple[VideoSampleRole, set[str]]] = {}

        def add(frame: int, role: VideoSampleRole, reason: str) -> None:
            if frame < plan.analysis_range.start or frame >= plan.analysis_range.end:
                return
            previous = selected.get(frame)
            if previous is None:
                selected[frame] = (role, {reason})
                return
            previous_role, reasons = previous
            selected[frame] = (
                role if _ROLE_PRIORITY[role] > _ROLE_PRIORITY[previous_role] else previous_role,
                {*reasons, reason},
            )

        add(plan.analysis_range.start, VideoSampleRole.FIRST_FRAME, "analysis-range-start")
        add(plan.analysis_range.end - 1, VideoSampleRole.LAST_FRAME, "analysis-range-end")
        for shot in shot_ranges:
            target = min(
                plan.per_shot_frame_budget,
                max(plan.minimum_samples_per_shot, min(5, shot.duration)),
            )
            for frame in self._even_frames(shot, target):
                add(frame, VideoSampleRole.REPRESENTATIVE, "shot-distributed-coverage")
        for transition in transitions:
            if transition.transition_type is VideoTransitionType.CONTINUOUS_MOTION:
                continue
            add(transition.frame - 1, VideoSampleRole.PRE_BOUNDARY, transition.id)
            add(transition.frame, VideoSampleRole.BOUNDARY, transition.id)
            add(transition.frame + 1, VideoSampleRole.POST_BOUNDARY, transition.id)
        for pinned in plan.pinned_times:
            add(pinned.frame, VideoSampleRole.PINNED, "operator-pinned")
        if len(selected) < plan.global_frame_budget:
            for frame in self._even_frames(plan.analysis_range, plan.global_frame_budget):
                if len(selected) >= plan.global_frame_budget:
                    break
                add(frame, VideoSampleRole.INTERIOR, "global-uniform-fallback")

        protected = sum(role in _PROTECTED_ROLES for role, _ in selected.values())
        if len(selected) > plan.global_frame_budget and protected <= plan.global_frame_budget:
            removable = [
                frame
                for frame, (role, _) in sorted(selected.items(), reverse=True)
                if role not in _PROTECTED_ROLES
            ]
            while len(selected) > plan.global_frame_budget and removable:
                del selected[removable.pop(0)]

        def shot_id(frame: int) -> str:
            for index, shot in enumerate(shot_ranges, start=1):
                if shot.start <= frame < shot.end:
                    return f"shot-{index:04d}"
            raise VideoAnatomyError(f"selected sample {frame} lies outside every shot")

        return [
            _SampleCandidate(
                frame=frame,
                role=role,
                reasons=tuple(sorted(reasons)),
                shot_id=shot_id(frame),
                sample_id=new_id("sample"),
            )
            for frame, (role, reasons) in sorted(selected.items())
        ]

    @staticmethod
    def _even_frames(frame_range: FrameRange, count: int) -> list[int]:
        if count <= 1 or frame_range.duration == 1:
            return [frame_range.start]
        actual = min(count, frame_range.duration)
        return sorted(
            {
                frame_range.start + round(index * (frame_range.duration - 1) / (actual - 1))
                for index in range(actual)
            }
        )

    def _decode_selected_frames(
        self,
        source: Path,
        plan: VideoSamplingPlan,
        candidates: list[_SampleCandidate],
    ) -> dict[int, NDArray[Any]]:
        requested = {item.frame for item in candidates}
        if plan.profile is VideoAnatomyProfile.QUICK:
            return self._decode_frame_indices(source, sorted(requested))
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoAnatomyError(f"cannot open video for sample extraction: {source}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, plan.analysis_range.start)
        frames: dict[int, NDArray[Any]] = {}
        try:
            for frame_index in range(plan.analysis_range.start, plan.analysis_range.end):
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in requested:
                    frames[frame_index] = cast(NDArray[Any], frame)
        finally:
            capture.release()
        missing = sorted(requested - set(frames))
        if missing:
            raise VideoAnatomyError(f"could not decode selected frames: {missing[:10]}")
        return frames

    def _materialize_samples(
        self,
        project_id: str,
        plan: VideoSamplingPlan,
        candidates: list[_SampleCandidate],
        frames: dict[int, NDArray[Any]],
        frame_timestamps: list[float],
    ) -> tuple[list[VideoFrameSample], dict[str, NDArray[Any]], int]:
        root = self.store.video_anatomy_path(project_id, plan.id)
        frame_root = root / "frames"
        frame_root.mkdir(parents=True, exist_ok=True)
        kept_by_shot: dict[str, list[tuple[_SampleCandidate, NDArray[Any]]]] = {}
        output: list[VideoFrameSample] = []
        retained_frames: dict[str, NDArray[Any]] = {}
        artifact_bytes = 0
        for candidate in candidates:
            frame = frames[candidate.frame]
            survivor: _SampleCandidate | None = None
            duplicate_metrics: _DedupMetrics | None = None
            if plan.deduplication.enabled and candidate.role not in _PROTECTED_ROLES:
                for kept_candidate, kept_frame in kept_by_shot.get(candidate.shot_id, []):
                    metrics = self._dedup_metrics(kept_frame, frame)
                    if self._is_duplicate(metrics, plan):
                        survivor = kept_candidate
                        duplicate_metrics = metrics
                        break
            artifact_path: str | None = None
            artifact_hash: str | None = None
            if survivor is None:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise VideoAnatomyError(f"cannot encode selected frame {candidate.frame}")
                payload = encoded.tobytes()
                if artifact_bytes + len(payload) > plan.artifact_budget_bytes:
                    raise VideoAnatomyError(
                        "frame artifact budget exhausted before protected coverage completed"
                    )
                target = frame_root / f"{candidate.sample_id}.jpg"
                self._atomic_bytes(target, payload)
                artifact_path = self._relative(project_id, target)
                artifact_hash = sha256_file(target)
                artifact_bytes += len(payload)
                kept_by_shot.setdefault(candidate.shot_id, []).append((candidate, frame))
                retained_frames[candidate.sample_id] = frame
            height, width = frame.shape[:2]
            output.append(
                VideoFrameSample(
                    id=candidate.sample_id,
                    shot_id=candidate.shot_id,
                    time=MotionTimeV2(frame=candidate.frame),
                    time_seconds=self._frame_time_seconds(
                        plan,
                        candidate.frame,
                        frame_timestamps,
                    ),
                    source_frame_index=candidate.frame,
                    role=candidate.role,
                    selection_reasons=list(candidate.reasons),
                    artifact_path=artifact_path,
                    artifact_sha256=artifact_hash,
                    width=width,
                    height=height,
                    pixel_format="bgr24-decoded-to-jpeg",
                    extraction_parameters={
                        "decoder": "opencv-video-capture",
                        "jpeg_quality": 92,
                        "requested_frame": candidate.frame,
                    },
                    provenance=self._sample_provenance(
                        plan,
                        candidate.frame,
                        frame_timestamps,
                    ),
                    seek_precision="decoded-nearest",
                    seek_limitations=(
                        []
                        if frame_timestamps
                        else [
                            "constant-rate time derives from decoded order and average rate"
                        ]
                    ),
                    kept=survivor is None,
                    duplicate_of_sample_id=(survivor.sample_id if survivor is not None else None),
                    duplicate_metrics=(
                        duplicate_metrics.as_dict() if duplicate_metrics is not None else {}
                    ),
                )
            )
        return output, retained_frames, artifact_bytes

    def _build_structure(
        self,
        plan: VideoSamplingPlan,
        shot_ranges: list[FrameRange],
        transitions: list[TransitionCandidate],
        samples: list[VideoFrameSample],
        detector_record_id: str,
        thresholds: dict[str, float],
    ) -> VideoStructureEvidence:
        kept = [item for item in samples if item.kept]
        shots: list[ShotEvidence] = []
        for index, frame_range in enumerate(shot_ranges, start=1):
            shot_id = f"shot-{index:04d}"
            shot_samples = [item for item in kept if item.shot_id == shot_id]
            if not shot_samples:
                raise VideoAnatomyError(f"sampling left {shot_id} without review evidence")
            representatives = [
                item.id
                for item in shot_samples
                if item.role
                in {
                    VideoSampleRole.FIRST_FRAME,
                    VideoSampleRole.LAST_FRAME,
                    VideoSampleRole.INTERIOR,
                    VideoSampleRole.REPRESENTATIVE,
                    VideoSampleRole.PINNED,
                }
            ] or [shot_samples[len(shot_samples) // 2].id]
            boundary_samples = [
                item.id
                for item in shot_samples
                if item.role
                in {
                    VideoSampleRole.PRE_BOUNDARY,
                    VideoSampleRole.BOUNDARY,
                    VideoSampleRole.POST_BOUNDARY,
                }
            ]
            shots.append(
                ShotEvidence(
                    id=shot_id,
                    frame_range=frame_range,
                    representative_sample_ids=representatives,
                    boundary_sample_ids=boundary_samples,
                    confidence=0.95,
                )
            )
        attached_transitions = []
        for transition in transitions:
            sample_ids = [
                item.id
                for item in kept
                if transition.window.start <= item.source_frame_index < transition.window.end
            ]
            attached_transitions.append(transition.model_copy(update={"sample_ids": sample_ids}))
        counts = {
            shot.id: sum(item.kept and item.shot_id == shot.id for item in samples)
            for shot in shots
        }
        protected = [item.id for item in samples if item.role in _PROTECTED_ROLES]
        ambiguous = [
            item.window
            for item in attached_transitions
            if item.transition_type is VideoTransitionType.UNKNOWN
        ]
        coverage_reason = (
            "sparse keyframe/uniform scan with inferred contiguous shot partition"
            if plan.profile is VideoAnatomyProfile.QUICK
            else "all-frame structural scan with contiguous shot partition"
        )
        return VideoStructureEvidence(
            project_id=plan.project_id,
            asset_id=plan.asset_id,
            source_sha256=plan.source_sha256,
            plan_id=plan.id,
            plan_sha256=plan.plan_sha256,
            analysis_range=plan.analysis_range,
            shots=shots,
            transitions=attached_transitions,
            coverage=VideoCoverageReport(
                requested_range=plan.analysis_range,
                covered_intervals=[
                    CoverageInterval(
                        frame_range=plan.analysis_range,
                        reasons=[coverage_reason],
                        sample_ids=[item.id for item in kept],
                    )
                ],
                uncovered_ranges=[],
                per_shot_sample_counts=counts,
                protected_sample_ids=protected,
                coverage_ratio=1.0,
            ),
            ambiguous_ranges=ambiguous,
            detector_evidence_refs=[detector_record_id],
            provenance=self._provenance(
                operation="shot-partition-and-transition-classification",
                thresholds=thresholds,
                conservative_unknown=True,
            ),
        )

    def _write_contact_sheet(
        self,
        path: Path,
        samples: list[VideoFrameSample],
        frames: dict[str, NDArray[Any]],
        plan: VideoSamplingPlan,
    ) -> None:
        kept = [item for item in samples if item.kept]
        columns = min(4, max(1, len(kept)))
        rows = math.ceil(len(kept) / columns)
        tile_width, tile_height, label_height = 320, 180, 42
        sheet = Image.new(
            "RGB", (columns * tile_width, rows * (tile_height + label_height)), "#111"
        )
        draw = ImageDraw.Draw(sheet)
        for index, sample in enumerate(kept):
            frame = cv2.cvtColor(frames[sample.id], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = (index % columns) * tile_width
            y = (index // columns) * (tile_height + label_height)
            tile = Image.new("RGB", (tile_width, tile_height), "black")
            tile.paste(image, ((tile_width - image.width) // 2, (tile_height - image.height) // 2))
            sheet.paste(tile, (x, y))
            label = (
                f"{sample.shot_id}  f{sample.source_frame_index}  "
                f"{sample.time_seconds:.3f}s\n{sample.role.value}"
            )
            draw.text((x + 6, y + tile_height + 3), label, fill="white")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            sheet.save(temporary_path, format="JPEG", quality=90)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        if sha256_file(path) == "0" * 64:  # impossible sentinel, keeps write verification explicit
            raise VideoAnatomyError("contact sheet hashing failed")
        if not plan.plan_sha256:
            raise VideoAnatomyError("contact sheet lost its sampling-plan identity")

    @staticmethod
    def _analysis_frame(frame: NDArray[Any]) -> NDArray[Any]:
        height, width = frame.shape[:2]
        scale = min(1.0, 320 / width, 180 / height)
        if scale == 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _monotonic_ratio(values: list[float]) -> float:
        differences = np.diff(np.asarray(values, dtype=np.float64))
        if differences.size == 0:
            return 0.0
        rising = float(np.mean(differences >= -0.005))
        falling = float(np.mean(differences <= 0.005))
        return max(rising, falling)

    @staticmethod
    def _dedup_metrics(first: NDArray[Any], second: NDArray[Any]) -> _DedupMetrics:
        first_small = VideoAnatomyService._analysis_frame(first)
        second_small = VideoAnatomyService._analysis_frame(second)
        first_gray = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second_small, cv2.COLOR_BGR2GRAY)
        luminance_delta = abs(float(np.mean(first_gray) - np.mean(second_gray))) / 255.0
        first_hist = cv2.calcHist([first_small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        second_hist = cv2.calcHist([second_small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        first_hist = cv2.normalize(first_hist, first_hist)
        second_hist = cv2.normalize(second_hist, second_hist)
        histogram_distance = float(
            max(0.0, min(1.0, cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_BHATTACHARYYA)))
        )
        hash_distance = VideoAnatomyService._dhash_distance(first_gray, second_gray)
        first_float = first_gray.astype(np.float64)
        second_float = second_gray.astype(np.float64)
        mean_first, mean_second = float(np.mean(first_float)), float(np.mean(second_float))
        variance_first, variance_second = float(np.var(first_float)), float(np.var(second_float))
        covariance = float(np.mean((first_float - mean_first) * (second_float - mean_second)))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
            (mean_first**2 + mean_second**2 + c1) * (variance_first + variance_second + c2)
        )
        first_edges = cv2.Canny(first_gray, 80, 160)
        second_edges = cv2.Canny(second_gray, 80, 160)
        edge_change = float(np.mean(cv2.absdiff(first_edges, second_edges)) / 255.0)
        motion_activity = float(np.mean(cv2.absdiff(first_gray, second_gray)) / 255.0)
        return _DedupMetrics(
            luminance_delta=luminance_delta,
            histogram_distance=histogram_distance,
            perceptual_hash_distance=hash_distance,
            ssim_change=max(0.0, min(1.0, 1.0 - ssim)),
            edge_change=edge_change,
            motion_activity=motion_activity,
        )

    @staticmethod
    def _dhash_distance(first: NDArray[Any], second: NDArray[Any]) -> int:
        def digest(image: NDArray[Any]) -> NDArray[Any]:
            resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
            return resized[:, 1:] > resized[:, :-1]

        return int(np.count_nonzero(digest(first) != digest(second)))

    @staticmethod
    def _is_duplicate(metrics: _DedupMetrics, plan: VideoSamplingPlan) -> bool:
        policy = plan.deduplication
        return (
            metrics.luminance_delta < policy.luminance_delta_min
            and metrics.histogram_distance < policy.histogram_distance_min
            and metrics.perceptual_hash_distance < policy.perceptual_hash_distance_min
            and metrics.ssim_change < policy.ssim_change_min
            and metrics.edge_change < policy.edge_change_min
            and metrics.motion_activity < policy.motion_activity_min
        )

    @staticmethod
    def _probe_frame_timestamps(source: Path) -> list[float]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(source),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise VideoAnatomyError(
                "cannot inspect variable-frame-rate timestamps: " + result.stderr[-2000:]
            )
        try:
            frames = json.loads(result.stdout).get("frames", [])
            timestamps = [
                round(float(item["best_effort_timestamp_time"]), 9)
                for item in frames
                if item.get("best_effort_timestamp_time") not in {None, "N/A"}
            ]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise VideoAnatomyError("ffprobe returned a malformed frame timestamp map") from error
        if not timestamps or timestamps != sorted(timestamps):
            raise VideoAnatomyError("frame timestamps are empty or not monotonic")
        return timestamps

    def _load_frame_timestamps(self, plan: VideoSamplingPlan) -> list[float]:
        if plan.frame_timestamps_artifact is None:
            return []
        if plan.frame_timestamps_sha256 is None or plan.frame_timestamp_count is None:
            raise VideoAnatomyError("sampling plan has an incomplete frame timestamp identity")
        project_root = self.store.project_path(plan.project_id).resolve()
        path = (project_root / plan.frame_timestamps_artifact).resolve()
        if (
            not path.is_relative_to(project_root)
            or not path.is_file()
            or sha256_file(path) != plan.frame_timestamps_sha256
        ):
            raise VideoAnatomyError(
                "frame timestamp map is missing, outside the project, or changed"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            timestamps = [float(item) for item in payload["timestamps_seconds"]]
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise VideoAnatomyError("frame timestamp map is malformed") from error
        if len(timestamps) != plan.frame_timestamp_count or timestamps != sorted(timestamps):
            raise VideoAnatomyError("frame timestamp map count or ordering differs from the plan")
        return timestamps

    @staticmethod
    def _frame_time_seconds(
        plan: VideoSamplingPlan,
        frame: int,
        frame_timestamps: list[float],
    ) -> float:
        if frame_timestamps:
            try:
                return frame_timestamps[frame]
            except IndexError as error:
                raise VideoAnatomyError("frame index is outside the timestamp map") from error
        return frame / plan.time_base.fps

    def _sample_provenance(
        self,
        plan: VideoSamplingPlan,
        frame: int,
        frame_timestamps: list[float],
    ) -> Provenance:
        parameters: dict[str, Any] = {
            "operation": "exact-frame-sample",
            "source_frame_index": frame,
        }
        if frame_timestamps:
            parameters.update(
                {
                    "exact_pts_seconds": self._frame_time_seconds(
                        plan,
                        frame,
                        frame_timestamps,
                    ),
                    "frame_timestamps_sha256": plan.frame_timestamps_sha256,
                    "timestamp_basis": "ffprobe-best-effort-timestamp-time",
                }
            )
        else:
            parameters["time_base_fps"] = plan.time_base.fps
            parameters["timestamp_basis"] = "constant-frame-rate"
        return self._provenance(**parameters)

    @staticmethod
    def _feature_payload(item: _FrameFeature) -> dict[str, float | int]:
        return {
            "frame": item.frame,
            "luminance": round(item.luminance, 8),
            "edge_density": round(item.edge_density, 8),
            "pixel_change": round(item.pixel_change, 8),
            "histogram_distance": round(item.histogram_distance, 8),
            "luminance_change": round(item.luminance_change, 8),
            "edge_change": round(item.edge_change, 8),
            "directional_asymmetry": round(item.directional_asymmetry, 8),
            "combined_change": round(item.combined_change, 8),
        }

    @staticmethod
    def _structure_confidence(structure: VideoStructureEvidence) -> float:
        values = [item.confidence for item in structure.shots]
        values.extend(item.confidence for item in structure.transitions)
        return float(np.mean(values)) if values else 1.0

    @staticmethod
    def _provenance(**parameters: Any) -> Provenance:
        return Provenance(
            tool="aoa-editing-video-anatomy",
            tool_version=__version__,
            parameters=parameters,
            deterministic=True,
        )

    def _relative(self, project_id: str, path: Path) -> str:
        return str(path.relative_to(self.store.project_path(project_id)))

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
