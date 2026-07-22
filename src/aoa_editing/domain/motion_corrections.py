"""Typed, deterministic motion corrections above canonical Motion Language v2."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from aoa_editing.domain.models import (
    AdjustPivotCurveCorrectionV2,
    AdjustRotationCouplingCorrectionV2,
    AdjustTangentCorrectionV2,
    ChangeEasingCorrectionV2,
    FrameRange,
    Matrix3x3V2,
    MatrixMotionCurveV2,
    MatrixMotionKeyframeV2,
    MatrixTransformMotionV2,
    MotionCorrectionContextV2,
    MotionCorrectionDiffV2,
    MotionCorrectionOperationV2,
    MovePhaseBoundaryCorrectionV2,
    RetimeSettleCorrectionV2,
    TransformEffectV2,
)
from aoa_editing.domain.motion_v2 import evaluate_matrix_curve


class MotionCorrectionLanguageError(ValueError):
    """A command cannot be mapped to an allowlisted semantic correction."""


class MotionCorrectionCompileError(ValueError):
    """A semantic correction is unsupported or violates motion invariants."""


def infer_motion_correction_operations(
    command: str,
    context: MotionCorrectionContextV2,
) -> list[MotionCorrectionOperationV2]:
    """Map bounded Russian/English editing language to independent operations."""

    lowered = command.strip().casefold()
    if not lowered:
        raise MotionCorrectionLanguageError("motion correction command is empty")
    operations: list[MotionCorrectionOperationV2] = []
    twist_words = (
        "подкруч",
        "подкрут",
        "поворот",
        "вращ",
        "rotation",
        "twist",
    )
    soft_words = ("мягч", "плавн", "soft", "smooth")
    settle_words = ("оседа", "settle", "затух", "успока")
    later_words = ("позже", "later", "дольше")
    earlier_words = ("раньше", "earlier", "быстрее осед")
    twist_requested = any(word in lowered for word in twist_words)
    soften_requested = any(word in lowered for word in soft_words)
    twist_range = FrameRange(
        start=context.twist_start_frame,
        duration=context.twist_end_frame - context.twist_start_frame + 1,
    )

    if soften_requested and twist_requested:
        operations.append(
            AdjustTangentCorrectionV2(
                channel="rotation",
                anchor_frame=context.twist_peak_frame,
                strength=0.35,
                affected_range=twist_range,
            )
        )
    if twist_requested:
        operations.append(
            AdjustRotationCouplingCorrectionV2(
                lag_delta_frames=0,
                smoothing_strength=0.2,
                affected_range=twist_range,
            )
        )
    if any(word in lowered for word in settle_words):
        tail_start = max(context.twist_end_frame, context.settle_frame - 10)
        direction: Literal["earlier", "later"]
        earlier_requested = any(word in lowered for word in earlier_words)
        later_requested = any(word in lowered for word in later_words)
        direction = "earlier" if earlier_requested and not later_requested else "later"
        strength = 0.2 if "чуть" in lowered or "slightly" in lowered else 0.35
        operations.append(
            RetimeSettleCorrectionV2(
                direction=direction,
                strength=strength,
                affected_range=FrameRange(
                    start=tail_start,
                    duration=context.duration_frames - tail_start,
                ),
            )
        )

    phase_match = re.search(
        r"(onset|старт|twist[_ ]?peak|пик|settle|оседание)\s*([+-]\d+)",
        lowered,
    )
    if phase_match:
        label, delta_text = phase_match.groups()
        boundary, original = {
            "onset": ("onset", context.onset_frame),
            "старт": ("onset", context.onset_frame),
            "twist_peak": ("twist_peak", context.twist_peak_frame),
            "twist peak": ("twist_peak", context.twist_peak_frame),
            "пик": ("twist_peak", context.twist_peak_frame),
            "settle": ("settle", context.settle_frame),
            "оседание": ("settle", context.settle_frame),
        }[label]
        start = 0 if boundary == "onset" else context.twist_start_frame
        end = (
            context.twist_end_frame + 1
            if boundary == "twist_peak"
            else context.duration_frames
        )
        operations.append(
            MovePhaseBoundaryCorrectionV2(
                boundary=cast(
                    Literal[
                        "onset",
                        "twist_start",
                        "twist_peak",
                        "twist_end",
                        "settle",
                    ],
                    boundary,
                ),
                original_frame=original,
                delta_frames=int(delta_text),
                affected_range=FrameRange(start=start, duration=end - start),
            )
        )

    if "pivot" in lowered or "опорн" in lowered:
        operations.append(
            AdjustPivotCurveCorrectionV2(
                delta_x=0.01,
                delta_y=0.0,
                affected_range=twist_range,
            )
        )
    if "ease" in lowered or "easing" in lowered or "крив" in lowered:
        easing: Literal["ease_in_out", "ease_out"] = (
            "ease_in_out"
            if any(word in lowered for word in ("in out", "in-out", "туда", "плавн"))
            else "ease_out"
        )
        operations.append(
            ChangeEasingCorrectionV2(
                channel="matrix",
                easing=easing,
                strength=0.3,
                affected_range=FrameRange(
                    start=context.onset_frame,
                    duration=context.settle_frame - context.onset_frame + 1,
                ),
            )
        )
    if not operations:
        raise MotionCorrectionLanguageError(
            "unsupported motion command; describe tangent softness, rotation/twist, "
            "phase timing, settle, pivot, or easing"
        )
    return operations


def compile_motion_corrections(
    effect: TransformEffectV2,
    operations: list[MotionCorrectionOperationV2],
    *,
    source_width: int,
    source_height: int,
    pivot_identifiable: bool,
    target_path: str = "/motion",
) -> tuple[TransformEffectV2, MotionCorrectionDiffV2]:
    """Compile semantic operations to a validated effect and compact typed diff."""

    if not operations:
        raise MotionCorrectionCompileError("at least one motion correction is required")
    if source_width <= 0 or source_height <= 0:
        raise MotionCorrectionCompileError("source dimensions must be positive")
    if not isinstance(effect.motion, MatrixTransformMotionV2):
        raise MotionCorrectionCompileError(
            "the current correction compiler requires observable matrix motion"
        )
    _require_dense_integer_curve(effect.motion.matrix)
    before = effect
    updated = effect
    for operation in operations:
        updated = _apply_operation(
            updated,
            operation,
            source_width=source_width,
            source_height=source_height,
            pivot_identifiable=pivot_identifiable,
        )
    before_motion = cast(MatrixTransformMotionV2, before.motion)
    updated_motion = cast(MatrixTransformMotionV2, updated.motion)
    _validate_interpolated_matrices(updated_motion.matrix)
    before_matrices = [item.matrix_3x3 for item in before_motion.matrix.keyframes]
    after_matrices = [item.matrix_3x3 for item in updated_motion.matrix.keyframes]
    deltas = [
        float(
            np.max(
                np.abs(
                    np.asarray(after_matrix, dtype=np.float64)
                    - np.asarray(before_matrix, dtype=np.float64)
                )
            )
        )
        for before_matrix, after_matrix in zip(
            before_matrices,
            after_matrices,
            strict=True,
        )
    ]
    start = min(item.affected_range.start for item in operations)
    end = max(item.affected_range.end for item in operations)
    diff = MotionCorrectionDiffV2(
        target_path=target_path,
        before_sha256=_effect_sha256(before),
        after_sha256=_effect_sha256(updated),
        affected_range=FrameRange(start=start, duration=end - start),
        changed_frame_count=sum(delta > 1e-12 for delta in deltas),
        maximum_matrix_coefficient_delta=max(deltas, default=0.0),
        endpoint_preserved=(
            before_matrices[0] == after_matrices[0]
            and before_matrices[-1] == after_matrices[-1]
        ),
        operation_kinds=[item.kind for item in operations],
    )
    if diff.before_sha256 == diff.after_sha256 or diff.changed_frame_count == 0:
        raise MotionCorrectionCompileError("motion correction produced no canonical change")
    return updated, diff


def motion_effect_sha256(effect: TransformEffectV2) -> str:
    """Stable canonical identity used to reject stale correction proposals."""

    return _effect_sha256(effect)


def _apply_operation(
    effect: TransformEffectV2,
    operation: MotionCorrectionOperationV2,
    *,
    source_width: int,
    source_height: int,
    pivot_identifiable: bool,
) -> TransformEffectV2:
    if isinstance(operation, AdjustPivotCurveCorrectionV2):
        if not pivot_identifiable:
            raise MotionCorrectionCompileError(
                "pivot is not identifiable; matrix/translation evidence is gauge-equivalent"
            )
        raise MotionCorrectionCompileError(
            "observable matrix motion has no separately editable pivot curve"
        )
    if isinstance(operation, AdjustTangentCorrectionV2):
        if operation.channel not in {"rotation", "matrix"}:
            raise MotionCorrectionCompileError(
                f"matrix motion cannot separately adjust {operation.channel} tangents"
            )
        return _adjust_rotation_tangent(
            effect,
            operation,
            source_width=source_width,
            source_height=source_height,
        )
    if isinstance(operation, AdjustRotationCouplingCorrectionV2):
        return _adjust_rotation_coupling(
            effect,
            operation,
            source_width=source_width,
            source_height=source_height,
        )
    if isinstance(operation, RetimeSettleCorrectionV2):
        if operation.direction == "later":
            mapping = lambda progress: _blend(  # noqa: E731
                progress,
                _smoothstep(progress),
                operation.strength,
            )
        else:
            mapping = lambda progress: _blend(  # noqa: E731
                progress,
                1.0 - (1.0 - progress) ** 2,
                operation.strength,
            )
        return _retime_matrix_range(effect, operation.affected_range, mapping)
    if isinstance(operation, MovePhaseBoundaryCorrectionV2):
        start = operation.affected_range.start
        end = operation.affected_range.end - 1
        original = operation.original_frame
        target = original + operation.delta_frames

        def phase_mapping(progress: float) -> float:
            output_frame = start + progress * (end - start)
            if output_frame <= target:
                source_frame = start + (original - start) * (
                    (output_frame - start) / (target - start)
                )
            else:
                source_frame = original + (end - original) * (
                    (output_frame - target) / (end - target)
                )
            return (source_frame - start) / (end - start)

        return _retime_matrix_range(effect, operation.affected_range, phase_mapping)
    if isinstance(operation, ChangeEasingCorrectionV2):
        easing = {
            "linear": lambda value: value,
            "ease_in": lambda value: value * value,
            "ease_out": lambda value: 1.0 - (1.0 - value) ** 2,
            "ease_in_out": _smoothstep,
        }[operation.easing]
        return _retime_matrix_range(
            effect,
            operation.affected_range,
            lambda progress: _blend(
                progress,
                easing(progress),
                operation.strength,
            ),
        )
    raise MotionCorrectionCompileError(f"unsupported correction: {operation.kind}")


def _retime_matrix_range(
    effect: TransformEffectV2,
    affected: FrameRange,
    mapping: Callable[[float], float],
) -> TransformEffectV2:
    motion = cast(MatrixTransformMotionV2, effect.motion)
    curve = motion.matrix
    first = affected.start
    last = affected.end - 1
    _validate_range(curve, affected)
    duration = last - first
    keyframes: list[MatrixMotionKeyframeV2] = []
    for item in curve.keyframes:
        frame = item.time.frame
        if first <= frame <= last:
            progress = 0.0 if duration == 0 else (frame - first) / duration
            mapped_progress = mapping(progress)
            mapped_progress = min(1.0, max(0.0, float(mapped_progress)))
            matrix = evaluate_matrix_curve(
                curve,
                first + mapped_progress * duration,
            )
            item = item.model_copy(
                update={
                    "matrix_3x3": _matrix_tuple(matrix),
                    "interpolation": "linear",
                    "incoming_tangent_3x3": None,
                    "outgoing_tangent_3x3": None,
                }
            )
        keyframes.append(item)
    return effect.model_copy(
        update={
            "motion": motion.model_copy(
                update={"matrix": curve.model_copy(update={"keyframes": keyframes})}
            )
        }
    )


def _adjust_rotation_tangent(
    effect: TransformEffectV2,
    operation: AdjustTangentCorrectionV2,
    *,
    source_width: int,
    source_height: int,
) -> TransformEffectV2:
    motion = cast(MatrixTransformMotionV2, effect.motion)
    curve = motion.matrix
    _validate_range(curve, operation.affected_range)
    first = operation.affected_range.start
    last = operation.affected_range.end - 1
    duration = last - first
    keyframes: list[MatrixMotionKeyframeV2] = []
    for item in curve.keyframes:
        frame = item.time.frame
        if first < frame < last:
            progress = (frame - first) / duration
            window = math.sin(math.pi * progress)
            offset = operation.strength * 0.08 * duration * window
            sampled = evaluate_matrix_curve(curve, min(last, frame + offset))
            current = np.asarray(item.matrix_3x3, dtype=np.float64)
            _, sampled_angle, _ = _similarity_components(
                sampled,
                source_width=source_width,
                source_height=source_height,
            )
            scale, _, center = _similarity_components(
                current,
                source_width=source_width,
                source_height=source_height,
            )
            matrix = _compose_similarity(
                scale,
                sampled_angle,
                center,
                source_width=source_width,
                source_height=source_height,
            )
            item = item.model_copy(update={"matrix_3x3": matrix})
        keyframes.append(item)
    return effect.model_copy(
        update={
            "motion": motion.model_copy(
                update={"matrix": curve.model_copy(update={"keyframes": keyframes})}
            )
        }
    )


def _adjust_rotation_coupling(
    effect: TransformEffectV2,
    operation: AdjustRotationCouplingCorrectionV2,
    *,
    source_width: int,
    source_height: int,
) -> TransformEffectV2:
    motion = cast(MatrixTransformMotionV2, effect.motion)
    curve = motion.matrix
    _validate_range(curve, operation.affected_range)
    first = operation.affected_range.start
    last = operation.affected_range.end - 1
    duration = last - first
    keyframes: list[MatrixMotionKeyframeV2] = []
    for item in curve.keyframes:
        frame = item.time.frame
        if first < frame < last:
            progress = (frame - first) / duration
            window = math.sin(math.pi * progress) ** 2
            current = np.asarray(item.matrix_3x3, dtype=np.float64)
            if operation.lag_delta_frames:
                sample_frame = min(
                    last,
                    max(first, frame - operation.lag_delta_frames),
                )
                sampled = evaluate_matrix_curve(curve, sample_frame)
                _, desired_angle, _ = _similarity_components(
                    sampled,
                    source_width=source_width,
                    source_height=source_height,
                )
            else:
                left = evaluate_matrix_curve(curve, max(first, frame - 1))
                right = evaluate_matrix_curve(curve, min(last, frame + 1))
                _, left_angle, _ = _similarity_components(
                    left,
                    source_width=source_width,
                    source_height=source_height,
                )
                _, right_angle, _ = _similarity_components(
                    right,
                    source_width=source_width,
                    source_height=source_height,
                )
                desired_angle = (left_angle + right_angle) / 2.0
            scale, current_angle, center = _similarity_components(
                current,
                source_width=source_width,
                source_height=source_height,
            )
            angle = _blend(
                current_angle,
                desired_angle,
                operation.smoothing_strength * window,
            )
            item = item.model_copy(
                update={
                    "matrix_3x3": _compose_similarity(
                        scale,
                        angle,
                        center,
                        source_width=source_width,
                        source_height=source_height,
                    )
                }
            )
        keyframes.append(item)
    return effect.model_copy(
        update={
            "motion": motion.model_copy(
                update={"matrix": curve.model_copy(update={"keyframes": keyframes})}
            )
        }
    )


def _similarity_components(
    matrix: NDArray[np.float64],
    *,
    source_width: int,
    source_height: int,
) -> tuple[float, float, tuple[float, float]]:
    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    scale = math.hypot(a, b)
    if scale <= 1e-12:
        raise MotionCorrectionCompileError("motion matrix has zero similarity scale")
    angle = math.atan2(-b, a)
    source_center = np.asarray(
        [source_width / 2.0, source_height / 2.0, 1.0],
        dtype=np.float64,
    )
    center = matrix @ source_center
    return scale, angle, (float(center[0]), float(center[1]))


def _compose_similarity(
    scale: float,
    angle: float,
    center: tuple[float, float],
    *,
    source_width: int,
    source_height: int,
) -> Matrix3x3V2:
    a = scale * math.cos(angle)
    b = -scale * math.sin(angle)
    source_x = source_width / 2.0
    source_y = source_height / 2.0
    tx = center[0] - a * source_x - b * source_y
    ty = center[1] - (-b * source_x + a * source_y)
    return (
        (a, b, tx),
        (-b, a, ty),
        (0.0, 0.0, 1.0),
    )


def _require_dense_integer_curve(curve: MatrixMotionCurveV2) -> None:
    frames = []
    for item in curve.keyframes:
        if item.time.subframe_numerator != 0:
            raise MotionCorrectionCompileError(
                "bounded correction requires integer-frame matrix evidence"
            )
        frames.append(item.time.frame)
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise MotionCorrectionCompileError(
            "bounded correction requires one matrix keyframe per frame"
        )


def _validate_range(curve: MatrixMotionCurveV2, affected: FrameRange) -> None:
    first = curve.keyframes[0].time.frame
    last = curve.keyframes[-1].time.frame
    if affected.start < first or affected.end - 1 > last or affected.duration < 2:
        raise MotionCorrectionCompileError("correction range is outside the matrix curve")


def _validate_interpolated_matrices(curve: MatrixMotionCurveV2) -> None:
    for left, right in zip(curve.keyframes, curve.keyframes[1:], strict=False):
        start = left.time.as_fraction
        end = right.time.as_fraction
        for fraction in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            matrix = evaluate_matrix_curve(curve, start + (end - start) * fraction)
            if abs(float(np.linalg.det(matrix))) <= 1e-10:
                raise MotionCorrectionCompileError(
                    "correction creates a singular interpolated matrix"
                )


def _effect_sha256(effect: TransformEffectV2) -> str:
    return hashlib.sha256(effect.model_dump_json().encode("utf-8")).hexdigest()


def _matrix_tuple(matrix: NDArray[np.float64]) -> Matrix3x3V2:
    return cast(
        Matrix3x3V2,
        tuple(tuple(float(value) for value in row) for row in matrix),
    )


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _blend(left: float, right: float, amount: float) -> float:
    return left + (right - left) * amount
