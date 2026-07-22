"""Source-neutral all-frame motion recovery with explicit model uncertainty."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from aoa_editing.domain.models import (
    FrameRate,
    MotionCurve,
    MotionModel,
    MotionRecoveryFrame,
    MotionRecoveryReport,
    Provenance,
)
from aoa_editing.infrastructure.media import sha256_file


class MotionRecoveryError(RuntimeError):
    """Motion evidence is unavailable or too weak for a defensible estimate."""


GlobalMotionModel = Literal["similarity", "affine", "homography"]


@dataclass(frozen=True, slots=True)
class DerivativeSeries:
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]
    jerk: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OutlierRejection:
    values: NDArray[np.float64]
    outlier_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CurveClassification:
    name: MotionCurve
    confidence: float
    errors: dict[str, float]


@dataclass(frozen=True, slots=True)
class SimilarityCoupling:
    """Observable phase relationship between similarity and center-path channels."""

    leading_explanation: str
    center_phase_relation: Literal[
        "center_lags",
        "center_leads",
        "synchronized",
    ]
    start_frame: int
    maximum_phase_offset_frame: int
    end_frame: int
    scale_rotation_progress_rmse: float
    center_similarity_progress_rmse: float
    maximum_phase_offset: float
    similarity_peak_velocity_frame: int
    center_peak_velocity_frame: int
    peak_velocity_lag_frames: int
    center_path_max_deviation: float
    center_path_max_deviation_frame: int
    center_heading_change_degrees: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _Features:
    gray: NDArray[np.uint8]
    original_shape: tuple[int, ...]
    sample_factor: float
    keypoints: Sequence[Any]
    descriptors: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _Fit:
    model: GlobalMotionModel
    matrix: NDArray[np.float64]
    inlier_mask: NDArray[np.bool_]
    inlier_count: int
    inlier_ratio: float
    rmse: float
    median_error: float


def unwrap_degrees(values: Sequence[float]) -> list[float]:
    """Unwrap the circular angle channel while preserving its first observation."""

    if not values:
        return []
    radians = np.deg2rad(np.asarray(values, dtype=np.float64))
    return cast(list[float], np.rad2deg(np.unwrap(radians)).tolist())


def finite_derivatives(values: Sequence[float], *, fps: float) -> DerivativeSeries:
    """Return frame-aligned first, second, and third temporal derivatives."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if len(array) == 1:
        zeros = (0.0,)
        return DerivativeSeries(zeros, zeros, zeros)
    edge_order: Literal[1, 2] = 2 if len(array) >= 3 else 1
    velocity = cast(
        NDArray[np.float64],
        np.gradient(array.tolist(), 1.0 / fps, edge_order=edge_order),
    )
    acceleration = cast(
        NDArray[np.float64],
        np.gradient(velocity.tolist(), 1.0 / fps, edge_order=edge_order),
    )
    jerk = cast(
        NDArray[np.float64],
        np.gradient(acceleration.tolist(), 1.0 / fps, edge_order=edge_order),
    )
    return DerivativeSeries(
        tuple(float(item) for item in velocity),
        tuple(float(item) for item in acceleration),
        tuple(float(item) for item in jerk),
    )


def local_polynomial_derivatives(
    values: Sequence[float] | NDArray[np.float64],
    *,
    fps: float,
    window: int = 15,
    degree: int = 3,
) -> DerivativeSeries:
    """Estimate stable frame-aligned derivatives with local polynomial fits."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer of at least three")
    if degree < 1:
        raise ValueError("degree must be positive")
    if len(array) <= degree:
        return finite_derivatives(array.tolist(), fps=fps)

    selected_window = min(window, len(array))
    if selected_window % 2 == 0:
        selected_window -= 1
    if selected_window <= degree:
        return finite_derivatives(array.tolist(), fps=fps)

    velocity: list[float] = []
    acceleration: list[float] = []
    jerk: list[float] = []
    for index in range(len(array)):
        start, stop = _bounded_window(index, len(array), selected_window)
        offsets = (
            np.arange(start, stop, dtype=np.float64) - float(index)
        ) / fps
        coefficients = np.polynomial.polynomial.polyfit(
            offsets,
            array[start:stop],
            degree,
        )
        velocity.append(float(coefficients[1]))
        acceleration.append(
            float(2.0 * coefficients[2]) if len(coefficients) > 2 else 0.0
        )
        jerk.append(
            float(6.0 * coefficients[3]) if len(coefficients) > 3 else 0.0
        )
    return DerivativeSeries(tuple(velocity), tuple(acceleration), tuple(jerk))


def characterize_similarity_coupling(
    *,
    center_x: Sequence[float] | NDArray[np.float64],
    center_y: Sequence[float] | NDArray[np.float64],
    scale: Sequence[float] | NDArray[np.float64],
    rotation: Sequence[float] | NDArray[np.float64],
    fps: float,
) -> SimilarityCoupling:
    """Explain stagger and curvature without inventing an unobservable pivot."""

    arrays = {
        "center_x": np.asarray(center_x, dtype=np.float64),
        "center_y": np.asarray(center_y, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "rotation": np.asarray(rotation, dtype=np.float64),
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 5:
        raise ValueError("coupled channels must have one shared length of at least five")
    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("coupled channels must be one-dimensional")
    if fps <= 0:
        raise ValueError("fps must be positive")

    smoothed = {
        name: _local_polynomial_values(values, window=15, degree=3)
        for name, values in arrays.items()
    }
    similarity_progresses = [
        progress
        for name in ("scale", "rotation")
        if (progress := _endpoint_progress(smoothed[name])) is not None
    ]
    if not similarity_progresses:
        raise ValueError("scale and rotation are both endpoint-degenerate")
    similarity_progress = np.mean(np.vstack(similarity_progresses), axis=0)
    scale_progress = _endpoint_progress(smoothed["scale"])
    rotation_progress = _endpoint_progress(smoothed["rotation"])
    scale_rotation_rmse = (
        _array_rmse(scale_progress - rotation_progress)
        if scale_progress is not None and rotation_progress is not None
        else 0.0
    )

    center = np.column_stack((smoothed["center_x"], smoothed["center_y"]))
    displacement = center[-1] - center[0]
    displacement_squared = float(np.dot(displacement, displacement))
    if displacement_squared <= 1e-12:
        raise ValueError("center path is endpoint-degenerate")
    center_progress = ((center - center[0]) @ displacement) / displacement_squared
    phase_offset = similarity_progress - center_progress
    maximum_index = int(np.argmax(np.abs(phase_offset)))
    maximum_offset = float(phase_offset[maximum_index])
    maximum_absolute_offset = abs(maximum_offset)
    relation: Literal["center_lags", "center_leads", "synchronized"]
    if maximum_absolute_offset < 0.02:
        relation = "synchronized"
    elif maximum_offset > 0:
        relation = "center_lags"
    else:
        relation = "center_leads"

    threshold = max(0.02, maximum_absolute_offset * 0.75)
    strong = np.flatnonzero(np.abs(phase_offset) >= threshold)
    start_frame = int(strong[0]) if len(strong) else maximum_index
    end_frame = int(strong[-1]) if len(strong) else maximum_index

    similarity_derivatives = local_polynomial_derivatives(
        similarity_progress,
        fps=fps,
        window=_preferred_derivative_window(len(similarity_progress)),
        degree=3,
    )
    center_derivatives = local_polynomial_derivatives(
        center_progress,
        fps=fps,
        window=_preferred_derivative_window(len(center_progress)),
        degree=3,
    )
    similarity_peak = int(np.argmax(np.abs(similarity_derivatives.velocity)))
    center_peak = int(np.argmax(np.abs(center_derivatives.velocity)))

    straight_path = center[0] + np.outer(center_progress, displacement)
    path_deviation = np.linalg.norm(center - straight_path, axis=1)
    path_deviation_frame = int(np.argmax(path_deviation))
    path_deviation_max = float(path_deviation[path_deviation_frame])

    x_derivatives = local_polynomial_derivatives(
        smoothed["center_x"],
        fps=fps,
        window=_preferred_derivative_window(len(center)),
        degree=3,
    )
    y_derivatives = local_polynomial_derivatives(
        smoothed["center_y"],
        fps=fps,
        window=_preferred_derivative_window(len(center)),
        degree=3,
    )
    velocity = np.column_stack((x_derivatives.velocity, y_derivatives.velocity))
    speed = np.linalg.norm(velocity, axis=1)
    active = np.flatnonzero(speed >= max(1e-12, float(np.max(speed)) * 0.15))
    if len(active) >= 2:
        headings = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
        heading_change = math.degrees(
            float(headings[int(active[-1])] - headings[int(active[0])])
        )
    else:
        heading_change = 0.0

    if scale_rotation_rmse > 0.05:
        explanation = "scale_rotation_phase_offset"
    elif maximum_absolute_offset >= 0.05 and path_deviation_max >= 0.001:
        explanation = "staggered_center_path_with_synchronized_scale_rotation"
    elif maximum_absolute_offset >= 0.05:
        explanation = "staggered_translation_with_synchronized_scale_rotation"
    elif path_deviation_max >= 0.001:
        explanation = "curved_center_path_with_shared_similarity_timing"
    else:
        explanation = "shared_nonlinear_similarity_progress"
    synchronization_confidence = math.exp(-20.0 * scale_rotation_rmse)
    structural_strength = min(
        1.0,
        maximum_absolute_offset / 0.08 + path_deviation_max / 0.005,
    )
    confidence = min(
        1.0,
        max(0.0, synchronization_confidence * (0.70 + 0.30 * structural_strength)),
    )
    return SimilarityCoupling(
        leading_explanation=explanation,
        center_phase_relation=relation,
        start_frame=start_frame,
        maximum_phase_offset_frame=maximum_index,
        end_frame=end_frame,
        scale_rotation_progress_rmse=scale_rotation_rmse,
        center_similarity_progress_rmse=_array_rmse(
            center_progress - similarity_progress
        ),
        maximum_phase_offset=maximum_offset,
        similarity_peak_velocity_frame=similarity_peak,
        center_peak_velocity_frame=center_peak,
        peak_velocity_lag_frames=center_peak - similarity_peak,
        center_path_max_deviation=path_deviation_max,
        center_path_max_deviation_frame=path_deviation_frame,
        center_heading_change_degrees=heading_change,
        confidence=confidence,
    )


def reject_temporal_outliers(
    values: Sequence[float] | NDArray[np.float64],
    *,
    half_window: int = 2,
    sigma: float = 3.5,
) -> OutlierRejection:
    """Hampel-filter isolated spikes and retain their exact frame indices."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if half_window < 1:
        raise ValueError("half_window must be positive")
    candidates: list[int] = []
    global_range = float(np.ptp(array))
    minimum_delta = max(1e-9, global_range * 0.02)
    for index, value in enumerate(array):
        start = max(0, index - half_window)
        stop = min(len(array), index + half_window + 1)
        window = array[start:stop]
        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        threshold = max(minimum_delta, sigma * 1.4826 * mad)
        if abs(float(value) - median) > threshold:
            candidates.append(index)
    if not candidates:
        return OutlierRejection(array.copy(), ())

    rejected = set(candidates)
    filtered = array.copy()
    for index in candidates:
        left = next((item for item in range(index - 1, -1, -1) if item not in rejected), None)
        right = next(
            (item for item in range(index + 1, len(array)) if item not in rejected),
            None,
        )
        if left is not None and right is not None:
            fraction = (index - left) / (right - left)
            filtered[index] = array[left] + (array[right] - array[left]) * fraction
        elif left is not None:
            filtered[index] = array[left]
        elif right is not None:
            filtered[index] = array[right]
    return OutlierRejection(filtered, tuple(candidates))


def evaluate_curve(name: MotionCurve, progress: float) -> float:
    """Evaluate the canonical synthetic curve catalog at normalized time."""

    time = min(1.0, max(0.0, float(progress)))
    if name == "hold":
        return 0.0
    if name == "linear":
        return time
    if name == "ease_in":
        return time**2
    if name == "ease_out":
        return 1.0 - (1.0 - time) ** 2
    if name == "ease_in_out":
        return time * time * (3.0 - 2.0 * time)
    if name == "cubic_bezier":
        return _cubic_bezier(time, 0.18, 0.82, 0.38, 1.0)
    if name == "hermite":
        return _hermite(time, start_tangent=0.68, end_tangent=0.12)
    if name == "overshoot":
        overshoot = 1.70158
        shifted = time - 1.0
        return 1.0 + (overshoot + 1.0) * shifted**3 + overshoot * shifted**2
    if name == "settle":
        raw = 1.0 - math.exp(-5.5 * time) * (
            math.cos(10.0 * time) + 0.28 * math.sin(10.0 * time)
        )
        final = 1.0 - math.exp(-5.5) * (math.cos(10.0) + 0.28 * math.sin(10.0))
        return raw / final
    return time


def classify_curve(values: Sequence[float]) -> CurveClassification:
    """Classify a scalar progress channel and expose ambiguity, not only a label."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if len(array) < 4:
        return CurveClassification("underdetermined", 0.0, {})
    delta = float(array[-1] - array[0])
    if abs(delta) < 1e-8:
        return CurveClassification("hold", 1.0, {"hold": 0.0})
    observed = (array - array[0]) / delta
    time = np.linspace(0.0, 1.0, len(array), dtype=np.float64)
    names: tuple[MotionCurve, ...] = (
        "linear",
        "ease_in",
        "ease_out",
        "ease_in_out",
        "cubic_bezier",
        "hermite",
        "overshoot",
        "settle",
    )
    errors: dict[str, float] = {
        str(name): float(
            math.sqrt(
                float(
                    np.mean(
                        np.square(
                            observed
                            - np.asarray([evaluate_curve(name, item) for item in time])
                        )
                    )
                )
            )
        )
        for name in names
    }
    ordered = sorted(errors.items(), key=lambda item: item[1])
    best_name, best_error = ordered[0]
    second_error = ordered[1][1]
    separation = max(0.0, (second_error - best_error) / max(second_error, 1e-9))
    absolute = math.exp(-12.0 * best_error)
    confidence = min(1.0, max(0.0, 0.55 * separation + 0.45 * absolute))
    return CurveClassification(cast(MotionCurve, best_name), confidence, errors)


class MotionAnalyzer:
    """Estimate global transform families from source-to-frame correspondences."""

    def __init__(
        self,
        *,
        maximum_features: int = 5000,
        ratio_threshold: float = 0.78,
        ransac_threshold: float = 2.5,
        maximum_source_side: int = 1200,
        maximum_frame_side: int = 960,
    ) -> None:
        self.maximum_features = maximum_features
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.maximum_source_side = maximum_source_side
        self.maximum_frame_side = maximum_frame_side

    def analyze_frame(
        self,
        source: NDArray[Any],
        frame: NDArray[Any],
        *,
        frame_index: int = 0,
        fps: float = 30.0,
    ) -> MotionRecoveryFrame:
        source_features = self._features(source, maximum_side=self.maximum_source_side)
        return self._analyze_prepared_frame(
            source_features,
            frame,
            frame_index=frame_index,
            fps=fps,
        )

    def analyze_video(
        self,
        source_path: Path,
        video_path: Path,
        *,
        sample_step: int = 1,
        output_path: Path | None = None,
    ) -> MotionRecoveryReport:
        if sample_step < 1:
            raise ValueError("sample_step must be positive")
        selected_source = source_path.expanduser().resolve(strict=True)
        selected_video = video_path.expanduser().resolve(strict=True)
        source = cv2.imread(str(selected_source), cv2.IMREAD_UNCHANGED)
        if source is None:
            raise MotionRecoveryError(f"cannot decode source image: {selected_source}")
        source_features = self._features(source, maximum_side=self.maximum_source_side)
        capture = cv2.VideoCapture(str(selected_video))
        if not capture.isOpened():
            raise MotionRecoveryError(f"cannot decode motion video: {selected_video}")
        declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if declared_count < 1 or fps_value <= 0 or width < 1 or height < 1:
            capture.release()
            raise MotionRecoveryError("motion video reports invalid stream geometry")
        selected_frames = set(range(0, declared_count, sample_step))
        selected_frames.add(declared_count - 1)
        raw: list[MotionRecoveryFrame] = []
        decoded_count = 0
        try:
            while True:
                available, frame = capture.read()
                if not available:
                    break
                if decoded_count in selected_frames:
                    raw.append(
                        self._analyze_prepared_frame(
                            source_features,
                            frame,
                            frame_index=decoded_count,
                            fps=fps_value,
                        )
                    )
                decoded_count += 1
        finally:
            capture.release()
        if decoded_count != declared_count or not raw or raw[-1].frame != declared_count - 1:
            raise MotionRecoveryError(
                f"decoded {decoded_count} frames from stream declaring {declared_count}"
            )
        rational = Fraction(fps_value).limit_denominator(1001)
        report = self._temporal_report(
            source_path=selected_source,
            video_path=selected_video,
            frame_count=declared_count,
            frame_rate=FrameRate(
                numerator=rational.numerator,
                denominator=rational.denominator,
            ),
            width=width,
            height=height,
            frames=raw,
            sample_step=sample_step,
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as stream:
                stream.write(report.model_dump_json(indent=2))
                stream.write("\n")
        return report

    def _features(self, image: NDArray[Any], *, maximum_side: int) -> _Features:
        original = _gray(image)
        gray, factor = _bounded(original, maximum_side)
        detector = cv2.SIFT_create(  # type: ignore[attr-defined]
            nfeatures=self.maximum_features,
            contrastThreshold=0.012,
            edgeThreshold=14,
        )
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 12:
            raise MotionRecoveryError("source has fewer than 12 measurable local features")
        return _Features(
            gray=gray,
            original_shape=original.shape,
            sample_factor=factor,
            keypoints=keypoints,
            descriptors=np.asarray(descriptors, dtype=np.float32),
        )

    def _analyze_prepared_frame(
        self,
        source: _Features,
        frame: NDArray[Any],
        *,
        frame_index: int,
        fps: float,
    ) -> MotionRecoveryFrame:
        frame_features = self._features(frame, maximum_side=self.maximum_frame_side)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(source.descriptors, frame_features.descriptors, k=2)
        matches = [
            first
            for first, second in pairs
            if first.distance < self.ratio_threshold * second.distance
        ]
        if len(matches) < 10:
            raise MotionRecoveryError(
                f"frame {frame_index} has only {len(matches)} reliable correspondences"
            )
        source_points = np.asarray(
            [source.keypoints[item.queryIdx].pt for item in matches],
            dtype=np.float64,
        ) / source.sample_factor
        frame_points = np.asarray(
            [frame_features.keypoints[item.trainIdx].pt for item in matches],
            dtype=np.float64,
        ) / frame_features.sample_factor
        models: tuple[GlobalMotionModel, ...] = (
            "similarity",
            "affine",
            "homography",
        )
        fits = {
            model: self._fit(
                model,
                source_points,
                frame_points,
                output_sample_factor=frame_features.sample_factor,
            )
            for model in models
        }
        selected_model, mismatch = _select_model(
            fits,
            source_shape=source.original_shape,
        )
        selected_fit = fits["homography"] if selected_model == "unknown" else fits[selected_model]
        parameters = _matrix_parameters(
            selected_fit.matrix,
            source_shape=source.original_shape,
            frame_shape=frame_features.original_shape,
        )
        confidence = _confidence(selected_fit, len(matches))
        if mismatch:
            confidence = min(confidence, 0.40)
        return MotionRecoveryFrame(
            frame=frame_index,
            time_seconds=frame_index / fps,
            selected_model=selected_model,
            matrix_3x3=selected_fit.matrix.tolist(),
            scale_relative_to_contain=parameters["scale_relative_to_contain"],
            rotation_degrees=parameters["rotation_degrees"],
            center_x=parameters["center_x"],
            center_y=parameters["center_y"],
            affine_anisotropy=parameters["affine_anisotropy"],
            affine_shear=parameters["affine_shear"],
            perspective_strength=parameters["perspective_strength"],
            match_count=len(matches),
            inlier_count=selected_fit.inlier_count,
            inlier_ratio=selected_fit.inlier_ratio,
            reprojection_rmse=selected_fit.rmse,
            confidence=confidence,
            model_scores={name: fit.rmse for name, fit in fits.items()},
        )

    def _fit(
        self,
        model: GlobalMotionModel,
        source_points: NDArray[np.float64],
        frame_points: NDArray[np.float64],
        *,
        output_sample_factor: float,
    ) -> _Fit:
        threshold = self.ransac_threshold / output_sample_factor
        if model == "similarity":
            affine, mask = cv2.estimateAffinePartial2D(
                source_points,
                frame_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=threshold,
                maxIters=10000,
                confidence=0.999,
                refineIters=40,
            )
            matrix = _affine_to_homography(affine)
        elif model == "affine":
            affine, mask = cv2.estimateAffine2D(
                source_points,
                frame_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=threshold,
                maxIters=10000,
                confidence=0.999,
                refineIters=40,
            )
            matrix = _affine_to_homography(affine)
        else:
            homography, mask = cv2.findHomography(
                source_points,
                frame_points,
                cv2.RANSAC,
                threshold,
                maxIters=10000,
                confidence=0.999,
            )
            if homography is None:
                raise MotionRecoveryError("homography estimation did not converge")
            matrix = np.asarray(homography, dtype=np.float64)
            matrix /= matrix[2, 2]
        if mask is None:
            raise MotionRecoveryError(f"{model} estimation did not produce inliers")
        inlier_mask = np.asarray(mask, dtype=np.uint8).ravel().astype(bool)
        if int(inlier_mask.sum()) < 6:
            raise MotionRecoveryError(f"{model} estimation retained fewer than six inliers")
        projected = cv2.perspectiveTransform(
            source_points.reshape(-1, 1, 2).astype(np.float64),
            matrix,
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - frame_points, axis=1)
        inlier_errors = errors[inlier_mask] * output_sample_factor
        return _Fit(
            model=model,
            matrix=matrix,
            inlier_mask=inlier_mask,
            inlier_count=int(inlier_mask.sum()),
            inlier_ratio=float(inlier_mask.mean()),
            rmse=float(math.sqrt(float(np.mean(np.square(inlier_errors))))),
            median_error=float(np.median(inlier_errors)),
        )

    def _temporal_report(
        self,
        *,
        source_path: Path,
        video_path: Path,
        frame_count: int,
        frame_rate: FrameRate,
        width: int,
        height: int,
        frames: list[MotionRecoveryFrame],
        sample_step: int,
    ) -> MotionRecoveryReport:
        channels = _observable_channels(frames)
        filtered: dict[str, NDArray[np.float64]] = {}
        outliers: set[int] = set()
        for name, values in channels.items():
            selected_values = unwrap_degrees(values) if name == "rotation" else values
            rejection = reject_temporal_outliers(selected_values)
            filtered[name] = rejection.values
            outliers.update(rejection.outlier_indices)

        fps_for_samples = frame_rate.fps / sample_step
        derivatives = {
            name: local_polynomial_derivatives(
                values,
                fps=fps_for_samples,
                window=_preferred_derivative_window(len(values)),
                degree=3,
            )
            for name, values in filtered.items()
        }
        enriched: list[MotionRecoveryFrame] = []
        for index, item in enumerate(frames):
            velocity = {
                name: series.velocity[index] for name, series in derivatives.items()
            }
            acceleration = {
                name: series.acceleration[index] for name, series in derivatives.items()
            }
            jerk = {name: series.jerk[index] for name, series in derivatives.items()}
            enriched.append(
                item.model_copy(
                    update={
                        "scale_relative_to_contain": float(filtered["scale"][index]),
                        "rotation_degrees": float(filtered["rotation"][index]),
                        "center_x": float(filtered["center_x"][index]),
                        "center_y": float(filtered["center_y"][index]),
                        "velocity": velocity,
                        "acceleration": acceleration,
                        "jerk": jerk,
                        "outlier": index in outliers,
                    }
                )
            )

        driver = _curve_driver(filtered)
        curve = classify_curve(driver.tolist())
        phase_boundaries = _phase_boundaries(driver.tolist())
        distribution = Counter(item.selected_model for item in enriched)
        unknown_fraction = distribution.get("unknown", 0) / len(enriched)
        non_unknown = {
            name: count for name, count in distribution.items() if name != "unknown"
        }
        selected_model: MotionModel
        if unknown_fraction >= 0.25 or not non_unknown:
            selected_model = "unknown"
        else:
            selected_model = cast(
                MotionModel,
                max(non_unknown.items(), key=lambda item: item[1])[0],
            )
        minority = len(enriched) - distribution.get(selected_model, 0)
        model_mismatch = selected_model == "unknown" or minority / len(enriched) >= 0.30
        uncertainties = [
            (
                "Pivot is not identifiable from a single global transform: moving pivot "
                "and compensating translation are observationally equivalent."
            )
        ]
        alternatives = [
            "The recovered global matrix can be authored as layer motion or a virtual camera.",
            (
                "Pivot and translation remain a gauge-equivalent decomposition without "
                "an authoring constraint."
            ),
        ]
        if model_mismatch:
            uncertainties.append(
                "No single global similarity, affine, or homography model explains "
                "the full sequence."
            )
            alternatives.append(
                "Structured local warp, rolling deformation, or correspondence loss "
                "may explain residuals."
            )
        if curve.confidence < 0.55:
            uncertainties.append("Temporal curve family is ambiguous at the recovered precision.")
        mean_confidence = float(np.mean([item.confidence for item in enriched]))
        return MotionRecoveryReport(
            source_sha256=sha256_file(source_path),
            video_sha256=sha256_file(video_path),
            frame_count=frame_count,
            analyzed_frame_count=len(enriched),
            frame_rate=frame_rate,
            width=width,
            height=height,
            selected_model=selected_model,
            model_distribution=dict(sorted(distribution.items())),
            curve_hint=curve.name,
            curve_confidence=curve.confidence,
            mean_confidence=mean_confidence,
            model_mismatch=model_mismatch,
            outlier_frames=[enriched[index].frame for index in sorted(outliers)],
            phase_boundaries=phase_boundaries,
            uncertainties=uncertainties,
            alternative_explanations=alternatives,
            frames=enriched,
            provenance=Provenance(
                tool="aoa-editing-motion-recovery",
                tool_version=version("aoa-editing"),
                command=[],
                parameters={
                    "sample_step": sample_step,
                    "feature": "SIFT",
                    "matcher": "ratio-test BF L2",
                    "models": ["similarity", "affine", "homography"],
                    "pivot_claim": "unidentifiable_without_authoring_constraint",
                    "kinematics": "local polynomial degree 3",
                },
                deterministic=True,
            ),
        )


def _gray(image: NDArray[Any]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cast(NDArray[np.uint8], image.astype(np.uint8, copy=False))
    if image.ndim == 3 and image.shape[2] == 3:
        return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if image.ndim == 3 and image.shape[2] == 4:
        return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY))
    raise MotionRecoveryError(f"unsupported image shape: {image.shape}")


def _bounded(
    image: NDArray[np.uint8],
    maximum_side: int,
) -> tuple[NDArray[np.uint8], float]:
    if maximum_side < 64:
        raise ValueError("maximum feature-analysis side must be at least 64 pixels")
    factor = min(1.0, maximum_side / max(image.shape))
    if factor == 1.0:
        return image, factor
    width = max(1, round(image.shape[1] * factor))
    height = max(1, round(image.shape[0] * factor))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cast(NDArray[np.uint8], resized), factor


def _affine_to_homography(value: NDArray[Any] | None) -> NDArray[np.float64]:
    if value is None:
        raise MotionRecoveryError("affine estimation did not converge")
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = np.asarray(value, dtype=np.float64)
    return matrix


def _select_model(
    fits: Mapping[GlobalMotionModel, _Fit],
    *,
    source_shape: tuple[int, ...],
) -> tuple[MotionModel, bool]:
    similarity = fits["similarity"]
    affine = fits["affine"]
    homography = fits["homography"]
    affine_parameters = _linear_structure(affine.matrix[:2, :2])
    source_height, source_width = source_shape[:2]
    perspective = math.hypot(
        float(homography.matrix[2, 0]) * source_width,
        float(homography.matrix[2, 1]) * source_height,
    )
    if (
        homography.rmse > 2.35
        or homography.median_error > 1.75
        or homography.inlier_ratio < 0.38
    ):
        return "unknown", True
    homography_improvement = (affine.rmse - homography.rmse) / max(affine.rmse, 1e-9)
    if perspective >= 0.012 and homography_improvement >= 0.08:
        return "homography", False
    affine_improvement = (similarity.rmse - affine.rmse) / max(similarity.rmse, 1e-9)
    if (
        (
            affine_parameters["anisotropy"] >= 0.025
            or affine_parameters["shear"] >= 0.025
        )
        and affine_improvement >= 0.08
    ):
        return "affine", False
    if similarity.rmse > 2.35 or similarity.inlier_ratio < 0.38:
        return "unknown", True
    return "similarity", False


def _matrix_parameters(
    matrix: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    frame_shape: tuple[int, ...],
) -> dict[str, float]:
    source_height, source_width = source_shape[:2]
    frame_height, frame_width = frame_shape[:2]
    source_center = np.asarray([source_width / 2, source_height / 2], dtype=np.float64)
    center = cv2.perspectiveTransform(
        source_center.reshape(1, 1, 2),
        matrix,
    ).reshape(2)
    jacobian = _homography_jacobian(matrix, source_center)
    structure = _linear_structure(jacobian)
    contain = min(frame_width / source_width, frame_height / source_height)
    perspective = math.hypot(
        float(matrix[2, 0]) * source_width,
        float(matrix[2, 1]) * source_height,
    )
    return {
        "scale_relative_to_contain": structure["scale"] / contain,
        "rotation_degrees": structure["rotation_degrees"],
        "center_x": float(center[0] / frame_width),
        "center_y": float(center[1] / frame_height),
        "affine_anisotropy": structure["anisotropy"],
        "affine_shear": structure["shear"],
        "perspective_strength": perspective,
    }


def _homography_jacobian(
    matrix: NDArray[np.float64],
    point: NDArray[np.float64],
) -> NDArray[np.float64]:
    x, y = float(point[0]), float(point[1])
    numerator_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    numerator_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    denominator_squared = denominator * denominator
    return np.asarray(
        [
            [
                (matrix[0, 0] * denominator - numerator_x * matrix[2, 0])
                / denominator_squared,
                (matrix[0, 1] * denominator - numerator_x * matrix[2, 1])
                / denominator_squared,
            ],
            [
                (matrix[1, 0] * denominator - numerator_y * matrix[2, 0])
                / denominator_squared,
                (matrix[1, 1] * denominator - numerator_y * matrix[2, 1])
                / denominator_squared,
            ],
        ],
        dtype=np.float64,
    )


def _linear_structure(linear: NDArray[np.float64]) -> dict[str, float]:
    left, singular, right = np.linalg.svd(linear)
    rotation_matrix = left @ right
    if np.linalg.det(rotation_matrix) < 0:
        left[:, -1] *= -1
        rotation_matrix = left @ right
    first = linear[:, 0]
    second = linear[:, 1]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    mean_scale = max(1e-12, (float(singular[0]) + float(singular[1])) / 2)
    anisotropy = abs(float(singular[0]) - float(singular[1])) / mean_scale
    shear = abs(float(np.dot(first, second))) / max(1e-12, first_norm * second_norm)
    return {
        "scale": math.sqrt(abs(float(np.linalg.det(linear)))),
        "rotation_degrees": math.degrees(
            math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))
        ),
        "anisotropy": anisotropy,
        "shear": shear,
    }


def _confidence(fit: _Fit, match_count: int) -> float:
    evidence = min(1.0, match_count / 120.0)
    precision = math.exp(-fit.rmse / 2.5)
    return min(
        1.0,
        max(0.0, 0.42 * fit.inlier_ratio + 0.28 * evidence + 0.30 * precision),
    )


def _observable_channels(frames: Sequence[MotionRecoveryFrame]) -> dict[str, list[float]]:
    names = {
        "scale": "scale_relative_to_contain",
        "rotation": "rotation_degrees",
        "center_x": "center_x",
        "center_y": "center_y",
    }
    channels: dict[str, list[float]] = {}
    for name, attribute in names.items():
        values = [getattr(frame, attribute) for frame in frames]
        if any(value is None for value in values):
            raise MotionRecoveryError(f"observable channel {name} contains missing values")
        channels[name] = [float(cast(float, value)) for value in values]
    return channels


def _curve_driver(channels: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    candidates = [
        (float(np.ptp(values)), name, values)
        for name, values in channels.items()
        if abs(float(values[-1] - values[0])) > 1e-6
    ]
    if not candidates:
        return np.zeros(len(next(iter(channels.values()))), dtype=np.float64)
    _, _, selected = max(candidates, key=lambda item: item[0])
    return selected


def _phase_boundaries(values: Sequence[float]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return {"onset": 0, "peak_velocity": 0, "settle": 0}
    velocity = np.abs(np.gradient(array))
    peak = int(np.argmax(velocity))
    threshold = max(1e-9, float(velocity[peak]) * 0.08)
    active = np.flatnonzero(velocity >= threshold)
    return {
        "onset": int(active[0]) if len(active) else 0,
        "peak_velocity": peak,
        "settle": int(active[-1]) if len(active) else len(array) - 1,
    }


def _preferred_derivative_window(length: int) -> int:
    selected = min(15, length if length % 2 else length - 1)
    return max(3, selected)


def _bounded_window(index: int, length: int, window: int) -> tuple[int, int]:
    half = window // 2
    start = max(0, index - half)
    stop = min(length, start + window)
    start = max(0, stop - window)
    return start, stop


def _local_polynomial_values(
    values: NDArray[np.float64],
    *,
    window: int,
    degree: int,
) -> NDArray[np.float64]:
    if len(values) <= degree:
        return values.copy()
    selected_window = min(window, len(values))
    if selected_window % 2 == 0:
        selected_window -= 1
    if selected_window <= degree:
        return values.copy()
    smoothed = np.empty_like(values)
    for index in range(len(values)):
        start, stop = _bounded_window(index, len(values), selected_window)
        offsets = np.arange(start, stop, dtype=np.float64) - float(index)
        coefficients = np.polynomial.polynomial.polyfit(
            offsets,
            values[start:stop],
            degree,
        )
        smoothed[index] = float(coefficients[0])
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    return smoothed


def _endpoint_progress(values: NDArray[np.float64]) -> NDArray[np.float64] | None:
    delta = float(values[-1] - values[0])
    if abs(delta) <= 1e-9:
        return None
    return cast(NDArray[np.float64], (values - values[0]) / delta)


def _array_rmse(values: NDArray[np.float64]) -> float:
    return float(math.sqrt(float(np.mean(np.square(values)))))


def _cubic_bezier(
    time: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    parameter = time
    for _ in range(8):
        x = _bezier_coordinate(parameter, x1, x2)
        derivative = _bezier_derivative(parameter, x1, x2)
        if abs(derivative) < 1e-8:
            break
        parameter = min(1.0, max(0.0, parameter - (x - time) / derivative))
    return _bezier_coordinate(parameter, y1, y2)


def _bezier_coordinate(parameter: float, first: float, second: float) -> float:
    inverse = 1.0 - parameter
    return (
        3.0 * inverse * inverse * parameter * first
        + 3.0 * inverse * parameter * parameter * second
        + parameter**3
    )


def _bezier_derivative(parameter: float, first: float, second: float) -> float:
    inverse = 1.0 - parameter
    return (
        3.0 * inverse * inverse * first
        + 6.0 * inverse * parameter * (second - first)
        + 3.0 * parameter * parameter * (1.0 - second)
    )


def _hermite(time: float, *, start_tangent: float, end_tangent: float) -> float:
    h10 = time**3 - 2.0 * time**2 + time
    h01 = -2.0 * time**3 + 3.0 * time**2
    h11 = time**3 - time**2
    return h10 * start_tangent + h01 + h11 * end_tangent


def write_recovery_report(report: MotionRecoveryReport, path: Path) -> None:
    """Persist a recovery report exclusively for evaluation adapters."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
        stream.write("\n")
