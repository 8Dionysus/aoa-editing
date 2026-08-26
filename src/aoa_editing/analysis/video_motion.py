"""Dense, source-neutral motion evidence for Video Anatomy shots."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from aoa_editing import __version__
from aoa_editing.domain.models import (
    EvidenceRecord,
    FrameRange,
    Provenance,
    ShotEvidence,
    VideoMotionClassification,
    VideoMotionEvidence,
    VideoMotionPhase,
    VideoSamplingPlan,
    VideoStructureEvidence,
)
from aoa_editing.infrastructure.store import ProjectStore


class VideoMotionAnalysisError(RuntimeError):
    """Motion evidence cannot be produced for the requested shot range."""


@dataclass(frozen=True, slots=True)
class _MotionStep:
    frame: int
    magnitude_mean: float
    magnitude_p95: float
    direction_degrees: float
    residual_mean: float
    transform_found: bool
    translation_x: float
    translation_y: float
    scale: float
    rotation_degrees: float

    def payload(self) -> dict[str, float | int | bool]:
        return {
            "frame": self.frame,
            "magnitude_mean": self.magnitude_mean,
            "magnitude_p95": self.magnitude_p95,
            "direction_degrees": self.direction_degrees,
            "residual_mean": self.residual_mean,
            "transform_found": self.transform_found,
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "scale": self.scale,
            "rotation_degrees": self.rotation_degrees,
        }


class VideoMotionAnalysisService:
    def __init__(self, store: ProjectStore):
        self.store = store

    def analyze(
        self,
        plan: VideoSamplingPlan,
        structure: VideoStructureEvidence,
        *,
        shot_ids: set[str] | None = None,
    ) -> tuple[list[VideoMotionEvidence], EvidenceRecord]:
        source = self.store.asset_source_path(plan.project_id, plan.asset_id)
        selected = [shot for shot in structure.shots if shot_ids is None or shot.id in shot_ids]
        if not selected:
            raise VideoMotionAnalysisError("motion analysis selected no known shots")
        motions: list[VideoMotionEvidence] = []
        artifact_paths: list[str] = []
        for shot in selected:
            motion, raw_path = self._analyze_shot(source, plan, shot)
            typed_path = self.store.save_video_motion_evidence(plan.id, motion)
            motions.append(motion)
            artifact_paths.extend(
                [
                    str(raw_path.relative_to(self.store.project_path(plan.project_id))),
                    str(typed_path.relative_to(self.store.project_path(plan.project_id))),
                ]
            )
        record = EvidenceRecord(
            project_id=plan.project_id,
            asset_id=plan.asset_id,
            source_sha256=plan.source_sha256,
            kind="video.motion",
            confidence=float(np.mean([item.confidence for item in motions])),
            payload={
                "plan_id": plan.id,
                "shot_ids": [item.shot_id for item in motions],
                "motion_evidence_ids": [item.id for item in motions],
                "all_frame_shots": [item.shot_id for item in motions],
                "classifications": {item.shot_id: item.classification.value for item in motions},
            },
            artifacts=artifact_paths,
            time_base=plan.time_base,
            ranges=[item.frame_range for item in motions],
            provenance=self._provenance(
                operation="per-shot-dense-motion-analysis",
                algorithm="farneback-plus-sparse-similarity",
            ),
        )
        self.store.save_evidence(record)
        return motions, record

    def _analyze_shot(
        self,
        source: Path,
        plan: VideoSamplingPlan,
        shot: ShotEvidence,
    ) -> tuple[VideoMotionEvidence, Path]:
        grays = self._decode_gray_range(source, shot)
        if not grays:
            raise VideoMotionAnalysisError(f"no frames decoded for {shot.id}")
        height, width = grays[0].shape
        diagonal = math.hypot(width, height)
        steps: list[_MotionStep] = []
        for offset in range(1, len(grays)):
            previous = grays[offset - 1]
            current = grays[offset]
            flow = cast(Any, cv2.calcOpticalFlowFarneback)(
                previous,
                current,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            dx = flow[..., 0]
            dy = flow[..., 1]
            magnitude = np.hypot(dx, dy) / diagonal
            median_dx = float(np.median(dx))
            median_dy = float(np.median(dy))
            residual = np.hypot(dx - median_dx, dy - median_dy) / diagonal
            transform = self._sparse_similarity(previous, current)
            steps.append(
                _MotionStep(
                    frame=shot.frame_range.start + offset,
                    magnitude_mean=float(np.mean(magnitude)),
                    magnitude_p95=float(np.percentile(magnitude, 95)),
                    direction_degrees=math.degrees(math.atan2(median_dy, median_dx)),
                    residual_mean=float(np.mean(residual)),
                    transform_found=transform is not None,
                    translation_x=(float(transform[0, 2] / width) if transform is not None else 0),
                    translation_y=(float(transform[1, 2] / height) if transform is not None else 0),
                    scale=(
                        float(math.hypot(transform[0, 0], transform[1, 0]))
                        if transform is not None
                        else 1.0
                    ),
                    rotation_degrees=(
                        math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
                        if transform is not None
                        else 0.0
                    ),
                )
            )
        magnitudes = [item.magnitude_mean for item in steps]
        peak_values = [item.magnitude_p95 for item in steps]
        magnitude_mean = float(np.mean(magnitudes)) if magnitudes else 0.0
        magnitude_peak = max(peak_values, default=0.0)
        residual_mean = float(np.mean([item.residual_mean for item in steps])) if steps else 0.0
        transform_ratio = sum(item.transform_found for item in steps) / len(steps) if steps else 0.0
        residual_ratio = residual_mean / max(magnitude_mean, 1e-9)
        classification, independent, uncertainty = self._classification(
            magnitude_mean,
            residual_ratio,
            transform_ratio,
            len(grays),
        )
        translations_x = [item.translation_x for item in steps if item.transform_found]
        translations_y = [item.translation_y for item in steps if item.transform_found]
        scales = [item.scale for item in steps if item.transform_found]
        rotations = [item.rotation_degrees for item in steps if item.transform_found]
        hypotheses: list[str] = []
        if translations_x and abs(float(np.median(translations_x))) >= 0.0005:
            hypotheses.append("pan")
        if translations_y and abs(float(np.median(translations_y))) >= 0.0005:
            hypotheses.append("tilt")
        if scales and abs(float(np.median(scales)) - 1.0) >= 0.0005:
            hypotheses.append("zoom")
        if rotations and abs(float(np.median(rotations))) >= 0.03:
            hypotheses.append("rotation")
        if hypotheses and independent:
            uncertainty.append(
                "global transform and independent residual motion coexist; "
                "camera causality is ambiguous"
            )
        phases = self._phases(shot, magnitudes)
        direction = self._circular_mean([item.direction_degrees for item in steps])
        raw_root = self.store.video_anatomy_path(plan.project_id, plan.id) / "motion-raw"
        raw_path = raw_root / f"{shot.id}.json"
        self._atomic_json(
            raw_path,
            {
                "schema_version": "1.0.0",
                "project_id": plan.project_id,
                "asset_id": plan.asset_id,
                "source_sha256": plan.source_sha256,
                "plan_id": plan.id,
                "shot_id": shot.id,
                "frame_range": shot.frame_range.model_dump(mode="json"),
                "decoded_frame_count": len(grays),
                "steps": [item.payload() for item in steps],
            },
        )
        relative = str(raw_path.relative_to(self.store.project_path(plan.project_id)))
        motion = VideoMotionEvidence(
            project_id=plan.project_id,
            asset_id=plan.asset_id,
            source_sha256=plan.source_sha256,
            shot_id=shot.id,
            frame_range=shot.frame_range,
            classification=classification,
            global_transform_model=(
                "none"
                if classification is VideoMotionClassification.STATIC
                else "similarity"
                if transform_ratio >= 0.4
                else "translation"
            ),
            camera_hypotheses=hypotheses,  # type: ignore[arg-type]
            independent_motion_detected=independent,
            motion_magnitude_mean=magnitude_mean,
            motion_magnitude_peak=magnitude_peak,
            direction_degrees=direction,
            phases=phases,
            uncertainty=uncertainty,
            competing_models=(
                ["camera-motion", "layer-or-object-motion"] if hypotheses and independent else []
            ),
            analyzed_frame_count=len(grays),
            source_frame_count=shot.frame_range.duration,
            all_frame_evidence_ref=relative,
            confidence=max(0.25, min(0.98, 0.55 + 0.35 * transform_ratio)),
            provenance=self._provenance(
                operation="all-frame-shot-motion",
                decoded_frame_count=len(grays),
                transform_success_ratio=transform_ratio,
                residual_ratio=residual_ratio,
                normalized_to_frame_diagonal=True,
            ),
        )
        return motion, raw_path

    @staticmethod
    def _decode_gray_range(source: Path, shot: ShotEvidence) -> list[NDArray[Any]]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoMotionAnalysisError(f"cannot open video for motion analysis: {source}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, shot.frame_range.start)
        grays: list[NDArray[Any]] = []
        try:
            for _ in range(shot.frame_range.duration):
                ok, frame = capture.read()
                if not ok:
                    break
                decoded = cast(NDArray[Any], frame)
                height, width = decoded.shape[:2]
                scale = min(1.0, 320 / width, 180 / height)
                if scale < 1:
                    decoded = cv2.resize(
                        decoded,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                grays.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY))
        finally:
            capture.release()
        if len(grays) != shot.frame_range.duration:
            raise VideoMotionAnalysisError(
                f"motion coverage for {shot.id} is incomplete: "
                f"expected {shot.frame_range.duration}, decoded {len(grays)}"
            )
        return grays

    @staticmethod
    def _sparse_similarity(previous: NDArray[Any], current: NDArray[Any]) -> NDArray[Any] | None:
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=5,
        )
        if points is None or len(points) < 6:
            return None
        tracked, status, _errors = cast(Any, cv2.calcOpticalFlowPyrLK)(
            previous, current, points, None
        )
        if tracked is None or status is None:
            return None
        valid = status.reshape(-1).astype(bool)
        if int(np.count_nonzero(valid)) < 6:
            return None
        matrix, _mask = cv2.estimateAffinePartial2D(
            points.reshape(-1, 2)[valid],
            tracked.reshape(-1, 2)[valid],
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        return cast(NDArray[Any] | None, matrix)

    @staticmethod
    def _classification(
        magnitude_mean: float,
        residual_ratio: float,
        transform_ratio: float,
        frame_count: int,
    ) -> tuple[VideoMotionClassification, bool, list[str]]:
        if frame_count < 2:
            return (
                VideoMotionClassification.UNKNOWN,
                False,
                ["a one-frame shot cannot establish temporal motion"],
            )
        if magnitude_mean < 0.00015:
            return VideoMotionClassification.STATIC, False, []
        if transform_ratio >= 0.5 and residual_ratio < 0.45:
            return VideoMotionClassification.GLOBAL, False, []
        if transform_ratio < 0.35 and residual_ratio >= 0.55:
            return VideoMotionClassification.INDEPENDENT, True, []
        return (
            VideoMotionClassification.MIXED,
            residual_ratio >= 0.45,
            ["global and residual flow do not support one exclusive motion model"],
        )

    @staticmethod
    def _phases(shot: ShotEvidence, magnitudes: list[float]) -> list[VideoMotionPhase]:
        if shot.frame_range.duration < 4:
            return []
        points = [
            shot.frame_range.start,
            shot.frame_range.start + shot.frame_range.duration // 4,
            shot.frame_range.start + shot.frame_range.duration // 2,
            shot.frame_range.start + 3 * shot.frame_range.duration // 4,
            shot.frame_range.end,
        ]
        roles = ["onset", "development", "peak", "settle"]
        peak_frame = (
            shot.frame_range.start + int(np.argmax(magnitudes)) + 1
            if magnitudes
            else shot.frame_range.start
        )
        return [
            VideoMotionPhase(
                role=role,  # type: ignore[arg-type]
                frame_range=FrameRange(start=start, duration=end - start),
                confidence=(0.8 if role == "peak" and start <= peak_frame < end else 0.65),
            )
            for role, (start, end) in zip(roles, pairwise(points), strict=True)
            if end > start
        ]

    @staticmethod
    def _circular_mean(values: list[float]) -> float | None:
        if not values:
            return None
        radians = np.radians(values)
        return float(
            math.degrees(
                math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))
            )
        )

    @staticmethod
    def _provenance(**parameters: Any) -> Provenance:
        return Provenance(
            tool="aoa-editing-video-motion",
            tool_version=__version__,
            parameters=parameters,
            deterministic=True,
        )

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
