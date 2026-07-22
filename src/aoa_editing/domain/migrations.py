"""Explicit additive migrations into canonical schema v1."""

from __future__ import annotations

from copy import deepcopy
from itertools import pairwise
from typing import Any

from aoa_editing.domain.models import (
    SCHEMA_MODELS,
    SCHEMA_VERSION,
    BezierHandleV2,
    DecomposedTransformMotionV2,
    MotionTimeV2,
    ScalarKeyframe,
    ScalarMotionCurveV2,
    ScalarMotionKeyframeV2,
    TransformEffect,
    TransformEffectV2,
    Vec2BezierHandleV2,
    Vec2Keyframe,
    Vec2MotionCurveV2,
    Vec2MotionKeyframeV2,
    Vec2ValueV2,
)


class MigrationError(ValueError):
    """Document version is unknown or cannot be migrated without guessing."""


def migrate_document(kind: str, document: dict[str, Any]) -> dict[str, Any]:
    """Migrate a legacy baseline document and validate it against the current model."""

    model = SCHEMA_MODELS.get(kind)
    if model is None:
        raise MigrationError(f"unknown document kind: {kind}")
    payload = deepcopy(document)
    source_version = str(payload.get("schema_version", "0.1.0"))
    if source_version == SCHEMA_VERSION:
        return model.model_validate(payload).model_dump(mode="json")
    if source_version != "0.1.0":
        raise MigrationError(f"no migration from {source_version} for {kind}")
    if kind == "timeline":
        payload.setdefault("background", "#000000")
        payload.setdefault("audio_sample_rate", 48000)
        payload.setdefault("tracks", [])
    elif kind == "project":
        payload.setdefault("assets", [])
        payload.setdefault("versions", [])
        payload.setdefault("current_version_id", None)
        payload.setdefault("sealed_reference_hashes", [])
    else:
        raise MigrationError(
            f"legacy 0.1.0 migration is intentionally unsupported for {kind}; preserve source"
        )
    payload["schema_version"] = SCHEMA_VERSION
    return model.model_validate(payload).model_dump(mode="json")


def migrate_transform_effect_v1(effect: TransformEffect) -> TransformEffectV2:
    """Upgrade a v1 virtual camera without changing its integer/subframe curve."""

    if effect.position_mode != "canvas_center":
        raise MigrationError(
            "v1 offset-position transforms remain valid but need an explicit "
            "pivot-target migration decision"
        )
    frames = [
        item.frame for curve in (effect.position, effect.scale, effect.rotation) for item in curve
    ]
    end = max(frames, default=0)
    position = _migrate_vector_curve(
        effect.position,
        default=Vec2ValueV2(x=0.5, y=0.5),
        end=end,
    )
    scale = _migrate_scalar_curve(effect.scale, default=1.0, end=end)
    scale_values = [item.value for item in scale.keyframes]
    increasing = all(left <= right for left, right in pairwise(scale_values))
    decreasing = all(left >= right for left, right in pairwise(scale_values))
    if increasing:
        scale = scale.model_copy(
            update={"monotonicity": "increasing", "overshoot_policy": "forbid"}
        )
    elif decreasing:
        scale = scale.model_copy(
            update={"monotonicity": "decreasing", "overshoot_policy": "forbid"}
        )
    rotation = _migrate_scalar_curve(effect.rotation, default=0.0, end=end).model_copy(
        update={"angle_unwrap": True}
    )
    pivot = _constant_vector_curve(
        Vec2ValueV2(x=effect.anchor_x, y=effect.anchor_y),
        end=end,
    )
    return TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            fit_mode=effect.fit_mode,
            position=position,
            scale=scale,
            rotation=rotation,
            pivot=pivot,
        )
    )


def _migrate_scalar_curve(
    source: list[ScalarKeyframe],
    *,
    default: float,
    end: int,
) -> ScalarMotionCurveV2:
    if not source:
        times = [0] if end == 0 else [0, end]
        return ScalarMotionCurveV2(
            keyframes=[
                ScalarMotionKeyframeV2(time=MotionTimeV2(frame=frame), value=default)
                for frame in times
            ]
        )
    keyframes = [
        ScalarMotionKeyframeV2(time=MotionTimeV2(frame=item.frame), value=item.value)
        for item in source
    ]
    for index, (left, right) in enumerate(pairwise(source)):
        easing = right.easing
        if easing == "linear":
            continue
        if easing == "hold":
            keyframes[index] = keyframes[index].model_copy(update={"interpolation": "hold"})
            continue
        duration = right.frame - left.frame
        delta = right.value - left.value
        outgoing_y, incoming_y = _legacy_bezier_offsets(easing, delta)
        keyframes[index] = keyframes[index].model_copy(
            update={
                "interpolation": "cubic_bezier",
                "outgoing_handle": BezierHandleV2(
                    time_offset_frames=duration / 3,
                    value_offset=outgoing_y,
                ),
            }
        )
        keyframes[index + 1] = keyframes[index + 1].model_copy(
            update={
                "incoming_handle": BezierHandleV2(
                    time_offset_frames=-duration / 3,
                    value_offset=incoming_y,
                )
            }
        )
    return ScalarMotionCurveV2(keyframes=keyframes)


def _migrate_vector_curve(
    source: list[Vec2Keyframe],
    *,
    default: Vec2ValueV2,
    end: int,
) -> Vec2MotionCurveV2:
    if not source:
        return _constant_vector_curve(default, end=end)
    keyframes = [
        Vec2MotionKeyframeV2(
            time=MotionTimeV2(frame=item.frame),
            value=Vec2ValueV2(x=item.x, y=item.y),
        )
        for item in source
    ]
    for index, (left, right) in enumerate(pairwise(source)):
        easing = right.easing
        if easing == "linear":
            continue
        if easing == "hold":
            keyframes[index] = keyframes[index].model_copy(update={"interpolation": "hold"})
            continue
        duration = right.frame - left.frame
        outgoing_x, incoming_x = _legacy_bezier_offsets(easing, right.x - left.x)
        outgoing_y, incoming_y = _legacy_bezier_offsets(easing, right.y - left.y)
        keyframes[index] = keyframes[index].model_copy(
            update={
                "interpolation": "cubic_bezier",
                "outgoing_handle": Vec2BezierHandleV2(
                    time_offset_frames=duration / 3,
                    x_offset=outgoing_x,
                    y_offset=outgoing_y,
                ),
            }
        )
        keyframes[index + 1] = keyframes[index + 1].model_copy(
            update={
                "incoming_handle": Vec2BezierHandleV2(
                    time_offset_frames=-duration / 3,
                    x_offset=incoming_x,
                    y_offset=incoming_y,
                )
            }
        )
    return Vec2MotionCurveV2(keyframes=keyframes)


def _constant_vector_curve(value: Vec2ValueV2, *, end: int) -> Vec2MotionCurveV2:
    times = [0] if end == 0 else [0, end]
    return Vec2MotionCurveV2(
        keyframes=[
            Vec2MotionKeyframeV2(time=MotionTimeV2(frame=frame), value=value) for frame in times
        ]
    )


def _legacy_bezier_offsets(easing: str, delta: float) -> tuple[float, float]:
    if easing == "ease_in":
        return 0.0, -2.0 * delta / 3.0
    if easing == "ease_out":
        return 2.0 * delta / 3.0, 0.0
    if easing == "ease_in_out":
        return 0.0, 0.0
    raise MigrationError(f"unsupported v1 easing: {easing}")
