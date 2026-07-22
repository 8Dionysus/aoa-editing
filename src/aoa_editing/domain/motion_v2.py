"""Deterministic evaluation and retiming for canonical Motion Language v2."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise
from typing import cast

import numpy as np
from numpy.typing import NDArray

from aoa_editing.domain.models import (
    BezierHandleV2,
    DecomposedTransformMotionV2,
    Matrix3x3V2,
    MatrixMotionCurveV2,
    MatrixMotionKeyframeV2,
    MatrixTransformMotionV2,
    MotionTimeV2,
    ScalarMotionCurveV2,
    ScalarMotionKeyframeV2,
    TransformEffectV2,
    Vec2MotionCurveV2,
    Vec2MotionKeyframeV2,
    Vec2ValueV2,
)


def evaluate_scalar_curve(
    curve: ScalarMotionCurveV2,
    time: Fraction | int | float,
) -> float:
    """Evaluate a scalar curve at an exact or floating subframe."""

    selected = _fraction(time)
    keyframes = curve.keyframes
    values = _unwrapped_values(curve)
    if selected <= keyframes[0].time.as_fraction:
        return values[0]
    if selected >= keyframes[-1].time.as_fraction:
        return values[-1]
    index = _segment_index(keyframes, selected)
    left = keyframes[index]
    right = keyframes[index + 1]
    start = left.time.as_fraction
    end = right.time.as_fraction
    progress = float((selected - start) / (end - start))
    left_value = values[index]
    right_value = values[index + 1]
    if left.interpolation == "hold":
        return left_value
    if left.interpolation == "linear":
        return left_value + (right_value - left_value) * progress
    if left.interpolation == "cubic_bezier":
        outgoing = left.outgoing_handle
        incoming = right.incoming_handle
        if outgoing is None or incoming is None:  # pragma: no cover - model invariant
            raise ValueError("cubic Bezier segment lacks handles")
        return _bezier_value(
            selected=float(selected),
            start=float(start),
            end=float(end),
            start_value=left_value,
            end_value=right_value,
            outgoing_time=outgoing.time_offset_frames,
            incoming_time=incoming.time_offset_frames,
            outgoing_value=outgoing.value_offset,
            incoming_value=incoming.value_offset,
        )
    if left.interpolation == "cubic_hermite":
        if left.outgoing_tangent is None or right.incoming_tangent is None:
            raise ValueError("cubic Hermite segment lacks tangents")
        return _hermite_value(
            progress,
            left_value,
            right_value,
            left.outgoing_tangent,
            right.incoming_tangent,
            float(end - start),
        )
    tangents = _monotone_tangents(
        [item.time.as_fraction for item in keyframes],
        values,
    )
    return _hermite_value(
        progress,
        left_value,
        right_value,
        tangents[index],
        tangents[index + 1],
        float(end - start),
    )


def evaluate_vec2_curve(
    curve: Vec2MotionCurveV2,
    time: Fraction | int | float,
) -> Vec2ValueV2:
    """Evaluate one path curve without splitting its temporal handles."""

    selected = _fraction(time)
    keyframes = curve.keyframes
    if selected <= keyframes[0].time.as_fraction:
        return keyframes[0].value
    if selected >= keyframes[-1].time.as_fraction:
        return keyframes[-1].value
    index = _segment_index(keyframes, selected)
    left = keyframes[index]
    right = keyframes[index + 1]
    start = left.time.as_fraction
    end = right.time.as_fraction
    progress = float((selected - start) / (end - start))
    if left.interpolation == "hold":
        return left.value
    if left.interpolation == "linear":
        return Vec2ValueV2(
            x=left.value.x + (right.value.x - left.value.x) * progress,
            y=left.value.y + (right.value.y - left.value.y) * progress,
        )
    if left.interpolation == "cubic_bezier":
        outgoing = left.outgoing_handle
        incoming = right.incoming_handle
        if outgoing is None or incoming is None:  # pragma: no cover - model invariant
            raise ValueError("vector cubic Bezier segment lacks handles")
        common = {
            "selected": float(selected),
            "start": float(start),
            "end": float(end),
            "outgoing_time": outgoing.time_offset_frames,
            "incoming_time": incoming.time_offset_frames,
        }
        return Vec2ValueV2(
            x=_bezier_value(
                **common,
                start_value=left.value.x,
                end_value=right.value.x,
                outgoing_value=outgoing.x_offset,
                incoming_value=incoming.x_offset,
            ),
            y=_bezier_value(
                **common,
                start_value=left.value.y,
                end_value=right.value.y,
                outgoing_value=outgoing.y_offset,
                incoming_value=incoming.y_offset,
            ),
        )
    if left.interpolation == "cubic_hermite":
        outgoing_tangent = left.outgoing_tangent
        incoming_tangent = right.incoming_tangent
        if outgoing_tangent is None or incoming_tangent is None:
            raise ValueError("vector cubic Hermite segment lacks tangents")
        duration = float(end - start)
        return Vec2ValueV2(
            x=_hermite_value(
                progress,
                left.value.x,
                right.value.x,
                outgoing_tangent.x,
                incoming_tangent.x,
                duration,
            ),
            y=_hermite_value(
                progress,
                left.value.y,
                right.value.y,
                outgoing_tangent.y,
                incoming_tangent.y,
                duration,
            ),
        )
    times = [item.time.as_fraction for item in keyframes]
    x_values = [item.value.x for item in keyframes]
    y_values = [item.value.y for item in keyframes]
    x_tangents = _monotone_tangents(times, x_values)
    y_tangents = _monotone_tangents(times, y_values)
    duration = float(end - start)
    return Vec2ValueV2(
        x=_hermite_value(
            progress,
            left.value.x,
            right.value.x,
            x_tangents[index],
            x_tangents[index + 1],
            duration,
        ),
        y=_hermite_value(
            progress,
            left.value.y,
            right.value.y,
            y_tangents[index],
            y_tangents[index + 1],
            duration,
        ),
    )


def evaluate_matrix_curve(
    curve: MatrixMotionCurveV2,
    time: Fraction | int | float,
) -> NDArray[np.float64]:
    """Evaluate a source-to-output matrix curve coefficient-wise."""

    selected = _fraction(time)
    keyframes = curve.keyframes
    if selected <= keyframes[0].time.as_fraction:
        return np.asarray(keyframes[0].matrix_3x3, dtype=np.float64)
    if selected >= keyframes[-1].time.as_fraction:
        return np.asarray(keyframes[-1].matrix_3x3, dtype=np.float64)
    index = _segment_index(keyframes, selected)
    left = keyframes[index]
    right = keyframes[index + 1]
    start = left.time.as_fraction
    end = right.time.as_fraction
    progress = float((selected - start) / (end - start))
    left_matrix = np.asarray(left.matrix_3x3, dtype=np.float64)
    right_matrix = np.asarray(right.matrix_3x3, dtype=np.float64)
    if left.interpolation == "hold":
        result = left_matrix
    elif left.interpolation == "linear":
        result = left_matrix + (right_matrix - left_matrix) * progress
    else:
        outgoing = left.outgoing_tangent_3x3
        incoming = right.incoming_tangent_3x3
        if outgoing is None or incoming is None:  # pragma: no cover - model invariant
            raise ValueError("matrix cubic Hermite segment lacks tangents")
        duration = float(end - start)
        result = _hermite_matrix(
            progress,
            left_matrix,
            right_matrix,
            np.asarray(outgoing, dtype=np.float64),
            np.asarray(incoming, dtype=np.float64),
            duration,
        )
    normalizer = float(result[2, 2])
    if abs(normalizer) > 1e-12:
        result = result / normalizer
    if abs(float(np.linalg.det(result))) <= 1e-12:
        raise ValueError("interpolated motion matrix is singular")
    return cast(NDArray[np.float64], result)


def evaluate_transform_matrix(
    effect: TransformEffectV2,
    time: Fraction | int | float,
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> Matrix3x3V2:
    """Compile canonical motion meaning into one source-to-output homography."""

    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("source and output dimensions must be positive")
    selected = _fraction(time)
    motion = effect.motion
    if isinstance(motion, MatrixTransformMotionV2):
        return _matrix_tuple(evaluate_matrix_curve(motion.matrix, selected))
    return _decomposed_transform_matrix(
        motion,
        selected,
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
    )


def scalar_curve_extrema(curve: ScalarMotionCurveV2) -> tuple[float, float]:
    """Bound authored overshoot with deterministic dense segment samples."""

    observed: list[float] = []
    for left, right in pairwise(curve.keyframes):
        start = left.time.as_fraction
        duration = right.time.as_fraction - start
        observed.extend(
            evaluate_scalar_curve(curve, start + duration * Fraction(index, 64))
            for index in range(64)
        )
    observed.append(evaluate_scalar_curve(curve, curve.keyframes[-1].time.as_fraction))
    return min(observed), max(observed)


def retime_scalar_curve(
    curve: ScalarMotionCurveV2,
    *,
    source_start: Fraction,
    source_end: Fraction,
    target_start: Fraction,
    target_end: Fraction,
) -> ScalarMotionCurveV2:
    """Retime a complete scalar curve while preserving its values and derivatives."""

    if source_start >= source_end or target_start >= target_end:
        raise ValueError("retime ranges must have positive duration")
    if (
        curve.keyframes[0].time.as_fraction != source_start
        or curve.keyframes[-1].time.as_fraction != source_end
    ):
        raise ValueError("retime source range must equal the curve endpoints")
    scale = (target_end - target_start) / (source_end - source_start)
    tangent_scale = float(1 / scale)
    keyframes: list[ScalarMotionKeyframeV2] = []
    for item in curve.keyframes:
        mapped = target_start + (item.time.as_fraction - source_start) * scale
        keyframes.append(
            item.model_copy(
                update={
                    "time": motion_time_from_fraction(mapped),
                    "incoming_tangent": (
                        item.incoming_tangent * tangent_scale
                        if item.incoming_tangent is not None
                        else None
                    ),
                    "outgoing_tangent": (
                        item.outgoing_tangent * tangent_scale
                        if item.outgoing_tangent is not None
                        else None
                    ),
                    "incoming_handle": _retime_handle(item.incoming_handle, float(scale)),
                    "outgoing_handle": _retime_handle(item.outgoing_handle, float(scale)),
                }
            )
        )
    return curve.model_copy(update={"keyframes": keyframes})


def motion_time_from_fraction(value: Fraction) -> MotionTimeV2:
    if value < 0:
        raise ValueError("motion time cannot be negative")
    frame = value.numerator // value.denominator
    remainder = value - frame
    return MotionTimeV2(
        frame=frame,
        subframe_numerator=remainder.numerator,
        subframe_denominator=remainder.denominator,
    )


def _decomposed_transform_matrix(
    motion: DecomposedTransformMotionV2,
    time: Fraction,
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> Matrix3x3V2:
    position = evaluate_vec2_curve(motion.position, time)
    pivot = evaluate_vec2_curve(motion.pivot, time)
    scale = evaluate_scalar_curve(motion.scale, time)
    rotation = math.radians(evaluate_scalar_curve(motion.rotation, time))
    fit = (
        min(output_width / source_width, output_height / source_height)
        if motion.fit_mode == "contain"
        else max(output_width / source_width, output_height / source_height)
    )
    operations: dict[str, NDArray[np.float64]] = {
        "fit": np.asarray(((fit, 0.0, 0.0), (0.0, fit, 0.0), (0.0, 0.0, 1.0))),
        "scale": np.asarray(((scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, 1.0))),
        "rotate": np.asarray(
            (
                (math.cos(rotation), -math.sin(rotation), 0.0),
                (math.sin(rotation), math.cos(rotation), 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        "affine": (
            evaluate_matrix_curve(motion.affine, time) if motion.affine is not None else np.eye(3)
        ),
        "homography": (
            evaluate_matrix_curve(motion.homography, time)
            if motion.homography is not None
            else np.eye(3)
        ),
    }
    composed = np.eye(3)
    for operation in motion.transform_order:
        if operation == "translate":
            continue
        composed = operations[operation] @ composed
    pivot_source = np.asarray(
        (pivot.x * source_width, pivot.y * source_height, 1.0),
        dtype=np.float64,
    )
    mapped = composed @ pivot_source
    if abs(float(mapped[2])) <= 1e-12:
        raise ValueError("decomposed transform maps pivot to infinity")
    mapped_xy = mapped[:2] / mapped[2]
    target = np.asarray((position.x * output_width, position.y * output_height))
    translation = target - mapped_xy
    translated = np.asarray(
        (
            (1.0, 0.0, translation[0]),
            (0.0, 1.0, translation[1]),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return _matrix_tuple(translated @ composed)


def _segment_index[
    KeyframeT: (
        ScalarMotionKeyframeV2,
        Vec2MotionKeyframeV2,
        MatrixMotionKeyframeV2,
    )
](keyframes: list[KeyframeT], time: Fraction) -> int:
    for index, (left, right) in enumerate(pairwise(keyframes)):
        if left.time.as_fraction <= time <= right.time.as_fraction:
            return index
    raise ValueError("time does not fall inside the curve")  # pragma: no cover


def _unwrapped_values(curve: ScalarMotionCurveV2) -> list[float]:
    values = [item.value for item in curve.keyframes]
    if not curve.angle_unwrap or len(values) < 2:
        return values
    unwrapped = [values[0]]
    for value in values[1:]:
        previous = unwrapped[-1]
        turns = round((previous - value) / 360.0)
        candidate = value + 360.0 * turns
        if candidate - previous > 180.0:
            candidate -= 360.0
        elif candidate - previous < -180.0:
            candidate += 360.0
        unwrapped.append(candidate)
    return unwrapped


def _monotone_tangents(times: list[Fraction], values: list[float]) -> list[float]:
    if len(values) == 1:
        return [0.0]
    widths = [float(right - left) for left, right in pairwise(times)]
    slopes = [
        (right_value - left_value) / width
        for left_value, right_value, width in zip(
            values[:-1],
            values[1:],
            widths,
            strict=True,
        )
    ]
    if len(values) == 2:
        return [slopes[0], slopes[0]]
    tangents = [slopes[0]]
    for index in range(1, len(values) - 1):
        left_slope = slopes[index - 1]
        right_slope = slopes[index]
        if left_slope == 0.0 or right_slope == 0.0 or left_slope * right_slope < 0:
            tangents.append(0.0)
            continue
        left_width = widths[index - 1]
        right_width = widths[index]
        first_weight = 2.0 * right_width + left_width
        second_weight = right_width + 2.0 * left_width
        tangents.append(
            (first_weight + second_weight)
            / (first_weight / left_slope + second_weight / right_slope)
        )
    tangents.append(slopes[-1])
    return tangents


def _bezier_value(
    *,
    selected: float,
    start: float,
    end: float,
    start_value: float,
    end_value: float,
    outgoing_time: float,
    incoming_time: float,
    outgoing_value: float,
    incoming_value: float,
) -> float:
    control_times = (start, start + outgoing_time, end + incoming_time, end)
    parameter = _solve_bezier_time(selected, control_times)
    return _cubic(
        parameter,
        start_value,
        start_value + outgoing_value,
        end_value + incoming_value,
        end_value,
    )


def _solve_bezier_time(
    selected: float,
    control_times: tuple[float, float, float, float],
) -> float:
    low = 0.0
    high = 1.0
    duration = control_times[-1] - control_times[0]
    parameter = min(1.0, max(0.0, (selected - control_times[0]) / duration))
    for _ in range(12):
        value = _cubic(parameter, *control_times)
        derivative = _cubic_derivative(parameter, *control_times)
        if abs(value - selected) <= 1e-12:
            return parameter
        if value < selected:
            low = parameter
        else:
            high = parameter
        candidate = (
            parameter - (value - selected) / derivative
            if abs(derivative) > 1e-12
            else (low + high) / 2.0
        )
        if not low < candidate < high:
            candidate = (low + high) / 2.0
        parameter = candidate
    return parameter


def _cubic(parameter: float, p0: float, p1: float, p2: float, p3: float) -> float:
    inverse = 1.0 - parameter
    return (
        inverse**3 * p0
        + 3.0 * inverse * inverse * parameter * p1
        + 3.0 * inverse * parameter * parameter * p2
        + parameter**3 * p3
    )


def _cubic_derivative(
    parameter: float,
    p0: float,
    p1: float,
    p2: float,
    p3: float,
) -> float:
    inverse = 1.0 - parameter
    return (
        3.0 * inverse * inverse * (p1 - p0)
        + 6.0 * inverse * parameter * (p2 - p1)
        + 3.0 * parameter * parameter * (p3 - p2)
    )


def _hermite_value(
    progress: float,
    start: float,
    end: float,
    start_tangent: float,
    end_tangent: float,
    duration: float,
) -> float:
    squared = progress * progress
    cubed = squared * progress
    return (
        (2.0 * cubed - 3.0 * squared + 1.0) * start
        + (cubed - 2.0 * squared + progress) * duration * start_tangent
        + (-2.0 * cubed + 3.0 * squared) * end
        + (cubed - squared) * duration * end_tangent
    )


def _hermite_matrix(
    progress: float,
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    start_tangent: NDArray[np.float64],
    end_tangent: NDArray[np.float64],
    duration: float,
) -> NDArray[np.float64]:
    squared = progress * progress
    cubed = squared * progress
    return (
        (2.0 * cubed - 3.0 * squared + 1.0) * start
        + (cubed - 2.0 * squared + progress) * duration * start_tangent
        + (-2.0 * cubed + 3.0 * squared) * end
        + (cubed - squared) * duration * end_tangent
    )


def _retime_handle(handle: BezierHandleV2 | None, scale: float) -> BezierHandleV2 | None:
    if handle is None:
        return None
    return handle.model_copy(update={"time_offset_frames": handle.time_offset_frames * scale})


def _fraction(value: Fraction | int | float) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("motion time must be finite")
    return Fraction(str(value))


def _matrix_tuple(matrix: NDArray[np.float64]) -> Matrix3x3V2:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )
