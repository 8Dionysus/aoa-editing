"""General source-to-reference affine motion measurement.

The adapter returns transform parameters only. It never exposes reference pixels as
project assets and does not own reconstruction policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from aoa_editing.domain.models import FrameRate

CurveHint = Literal[
    "linear", "ease_in", "ease_out", "ease_in_out", "hold", "underdetermined"
]


class ReferenceAnalysisError(RuntimeError):
    """A reference frame could not be related to the permitted source image."""


@dataclass(frozen=True, slots=True)
class AffineFrameMeasurement:
    scale_relative_to_contain: float
    rotation_degrees: float
    center_x: float
    center_y: float
    match_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse: float
    frame: int = 0
    time_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class VideoMotionMeasurement:
    frame_count: int
    frame_rate: FrameRate
    width: int
    height: int
    measurements: tuple[AffineFrameMeasurement, ...]
    curve_hint: CurveHint


def estimate_affine_frame(
    source: NDArray[Any],
    frame: NDArray[Any],
    *,
    max_source_side: int = 1024,
    max_frame_side: int = 960,
) -> AffineFrameMeasurement:
    """Estimate a similarity transform from a source still into one output frame."""

    source_gray = _gray(source)
    frame_gray = _gray(frame)
    source_sample, source_factor = _bounded(source_gray, max_source_side)
    frame_sample, frame_factor = _bounded(frame_gray, max_frame_side)
    detector = cv2.SIFT_create(  # type: ignore[attr-defined]
        nfeatures=8000, contrastThreshold=0.02
    )
    source_points, source_descriptors = detector.detectAndCompute(source_sample, None)
    frame_points, frame_descriptors = detector.detectAndCompute(frame_sample, None)
    if source_descriptors is None or frame_descriptors is None:
        raise ReferenceAnalysisError("source or frame has no measurable local features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(source_descriptors, frame_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(matches) < 8:
        raise ReferenceAnalysisError(f"too few reliable feature matches: {len(matches)}")
    source_coordinates = np.asarray(
        [source_points[match.queryIdx].pt for match in matches], dtype=np.float32
    )
    frame_coordinates = np.asarray(
        [frame_points[match.trainIdx].pt for match in matches], dtype=np.float32
    )
    matrix_value, inlier_mask_value = cv2.estimateAffinePartial2D(
        source_coordinates,
        frame_coordinates,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=50,
    )
    if matrix_value is None or inlier_mask_value is None:
        raise ReferenceAnalysisError("robust affine estimation did not converge")
    matrix = np.asarray(matrix_value, dtype=np.float64)
    inlier_mask = np.asarray(inlier_mask_value, dtype=np.uint8)
    linear = matrix[:, :2] * (source_factor / frame_factor)
    translation = matrix[:, 2] / frame_factor
    scale = math.hypot(float(linear[0, 0]), float(linear[1, 0]))
    rotation = math.degrees(math.atan2(float(linear[1, 0]), float(linear[0, 0])))
    source_height, source_width = source_gray.shape
    frame_height, frame_width = frame_gray.shape
    source_center = np.array([source_width / 2, source_height / 2], dtype=np.float64)
    output_center = linear @ source_center + translation
    contain_scale = min(frame_width / source_width, frame_height / source_height)
    inliers = inlier_mask.ravel().astype(bool)
    predicted = (
        source_coordinates @ matrix[:, :2].T + matrix[:, 2]
    )
    residuals = np.linalg.norm(predicted[inliers] - frame_coordinates[inliers], axis=1)
    inlier_count = int(inliers.sum())
    return AffineFrameMeasurement(
        scale_relative_to_contain=scale / contain_scale,
        rotation_degrees=rotation,
        center_x=float(output_center[0] / frame_width),
        center_y=float(output_center[1] / frame_height),
        match_count=len(matches),
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / len(matches),
        reprojection_rmse=float(math.sqrt(float(np.mean(np.square(residuals))))),
    )


def measure_video_motion(
    source_path: Path,
    reference_path: Path,
    *,
    sample_step: int = 5,
) -> VideoMotionMeasurement:
    """Measure a source still's similarity transform through a reference video."""

    if sample_step < 1:
        raise ValueError("sample_step must be positive")
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise ReferenceAnalysisError(f"cannot decode source image: {source_path}")
    capture = cv2.VideoCapture(str(reference_path))
    if not capture.isOpened():
        raise ReferenceAnalysisError(f"cannot decode reference video: {reference_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < 1 or fps_value <= 0 or width < 1 or height < 1:
            raise ReferenceAnalysisError("reference video reports invalid stream geometry")
        rational = Fraction(fps_value).limit_denominator(1001)
        frame_rate = FrameRate(numerator=rational.numerator, denominator=rational.denominator)
        selected = set(range(0, frame_count, sample_step))
        selected.add(frame_count - 1)
        measurements: list[AffineFrameMeasurement] = []
        frame_index = 0
        while True:
            available, frame = capture.read()
            if not available:
                break
            if frame_index in selected:
                measured = estimate_affine_frame(source, frame)
                measurements.append(
                    replace(
                        measured,
                        frame=frame_index,
                        time_seconds=frame_index / frame_rate.fps,
                    )
                )
            frame_index += 1
    finally:
        capture.release()
    if not measurements or measurements[-1].frame != frame_count - 1:
        raise ReferenceAnalysisError("reference video ended before its declared final frame")
    return VideoMotionMeasurement(
        frame_count=frame_count,
        frame_rate=frame_rate,
        width=width,
        height=height,
        measurements=tuple(measurements),
        curve_hint=_classify_curve(measurements),
    )


def _classify_curve(measurements: list[AffineFrameMeasurement]) -> CurveHint:
    if len(measurements) < 3:
        return "underdetermined"
    start = measurements[0]
    end = measurements[-1]
    scale_delta = end.scale_relative_to_contain - start.scale_relative_to_contain
    if abs(scale_delta) < 1e-6 or end.frame == start.frame:
        return "hold"
    candidates: dict[CurveHint, Callable[[float], float]] = {
        "linear": lambda progress: progress,
        "ease_in": lambda progress: progress * progress,
        "ease_out": lambda progress: 1 - (1 - progress) ** 2,
        "ease_in_out": lambda progress: progress * progress * (3 - 2 * progress),
    }
    errors: dict[CurveHint, float] = {}
    for name, curve in candidates.items():
        residuals = []
        for item in measurements:
            time_progress = (item.frame - start.frame) / (end.frame - start.frame)
            value_progress = (
                item.scale_relative_to_contain - start.scale_relative_to_contain
            ) / scale_delta
            residuals.append((value_progress - curve(time_progress)) ** 2)
        errors[name] = math.sqrt(sum(residuals) / len(residuals))
    return min(errors, key=errors.__getitem__)


def _gray(image: NDArray[Any]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cast(NDArray[np.uint8], image.astype(np.uint8, copy=False))
    if image.ndim == 3 and image.shape[2] == 3:
        return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if image.ndim == 3 and image.shape[2] == 4:
        return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY))
    raise ReferenceAnalysisError(f"unsupported image shape: {image.shape}")


def _bounded(
    image: NDArray[np.uint8], maximum_side: int
) -> tuple[NDArray[np.uint8], float]:
    factor = min(1.0, maximum_side / max(image.shape))
    if factor == 1.0:
        return image, factor
    width = max(1, round(image.shape[1] * factor))
    height = max(1, round(image.shape[0] * factor))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cast(NDArray[np.uint8], resized), factor
