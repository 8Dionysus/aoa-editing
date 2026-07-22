from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from aoa_editing.domain.migrations import migrate_transform_effect_v1
from aoa_editing.domain.models import (
    BezierHandleV2,
    DecomposedTransformMotionV2,
    MatrixMotionCurveV2,
    MatrixMotionKeyframeV2,
    MotionCouplingV2,
    MotionPhaseV2,
    MotionSamplingV2,
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
from aoa_editing.domain.motion_v2 import (
    evaluate_matrix_curve,
    evaluate_scalar_curve,
    evaluate_transform_matrix,
    evaluate_vec2_curve,
    retime_scalar_curve,
)


def _time(frame: int, numerator: int = 0, denominator: int = 1) -> MotionTimeV2:
    return MotionTimeV2(
        frame=frame,
        subframe_numerator=numerator,
        subframe_denominator=denominator,
    )


def _constant_scalar(value: float, end: int) -> ScalarMotionCurveV2:
    return ScalarMotionCurveV2(
        keyframes=[
            ScalarMotionKeyframeV2(time=_time(0), value=value),
            ScalarMotionKeyframeV2(time=_time(end), value=value),
        ]
    )


def _constant_vec2(x: float, y: float, end: int) -> Vec2MotionCurveV2:
    return Vec2MotionCurveV2(
        keyframes=[
            Vec2MotionKeyframeV2(time=_time(0), value=Vec2ValueV2(x=x, y=y)),
            Vec2MotionKeyframeV2(time=_time(end), value=Vec2ValueV2(x=x, y=y)),
        ]
    )


def test_motion_time_is_canonical_and_subframe_evaluation_is_deterministic() -> None:
    half = _time(4, 1, 2)
    assert half.as_fraction == Fraction(9, 2)
    assert MotionTimeV2.model_validate_json(half.model_dump_json()) == half

    with pytest.raises(ValueError, match="reduced"):
        _time(4, 2, 4)
    with pytest.raises(ValueError, match="smaller than denominator"):
        _time(4, 2, 2)

    curve = ScalarMotionCurveV2(
        keyframes=[
            ScalarMotionKeyframeV2(time=_time(0), value=0.0),
            ScalarMotionKeyframeV2(time=_time(10), value=20.0),
        ]
    )
    assert evaluate_scalar_curve(curve, half.as_fraction) == pytest.approx(9.0)
    assert evaluate_scalar_curve(curve, half.as_fraction) == evaluate_scalar_curve(
        curve,
        half.as_fraction,
    )


def test_cubic_bezier_handles_encode_smoothstep_without_endpoint_drift() -> None:
    curve = ScalarMotionCurveV2(
        continuity="c1",
        keyframes=[
            ScalarMotionKeyframeV2(
                time=_time(0),
                value=0.0,
                interpolation="cubic_bezier",
                outgoing_handle=BezierHandleV2(
                    time_offset_frames=10.0 / 3.0,
                    value_offset=0.0,
                ),
            ),
            ScalarMotionKeyframeV2(
                time=_time(10),
                value=1.0,
                incoming_handle=BezierHandleV2(
                    time_offset_frames=-10.0 / 3.0,
                    value_offset=0.0,
                ),
            ),
        ],
    )

    for frame in range(11):
        progress = frame / 10
        expected = progress * progress * (3.0 - 2.0 * progress)
        assert evaluate_scalar_curve(curve, Fraction(frame)) == pytest.approx(
            expected,
            abs=1e-10,
        )


def test_hermite_c1_and_monotone_scale_do_not_introduce_overshoot() -> None:
    hermite = ScalarMotionCurveV2(
        continuity="c1",
        keyframes=[
            ScalarMotionKeyframeV2(
                time=_time(0),
                value=0.0,
                interpolation="cubic_hermite",
                outgoing_tangent=0.0,
            ),
            ScalarMotionKeyframeV2(
                time=_time(10),
                value=1.0,
                interpolation="cubic_hermite",
                incoming_tangent=0.1,
                outgoing_tangent=0.1,
            ),
            ScalarMotionKeyframeV2(
                time=_time(20),
                value=2.0,
                incoming_tangent=0.0,
            ),
        ],
    )
    epsilon = Fraction(1, 10_000)
    left = (
        evaluate_scalar_curve(hermite, Fraction(10) - epsilon)
        - evaluate_scalar_curve(hermite, Fraction(10) - 2 * epsilon)
    ) / float(epsilon)
    right = (
        evaluate_scalar_curve(hermite, Fraction(10) + 2 * epsilon)
        - evaluate_scalar_curve(hermite, Fraction(10) + epsilon)
    ) / float(epsilon)
    assert left == pytest.approx(right, abs=5e-4)

    monotone = ScalarMotionCurveV2(
        continuity="c1",
        monotonicity="decreasing",
        overshoot_policy="forbid",
        keyframes=[
            ScalarMotionKeyframeV2(
                time=_time(0),
                value=2.8,
                interpolation="monotone_cubic",
            ),
            ScalarMotionKeyframeV2(
                time=_time(8),
                value=2.25,
                interpolation="monotone_cubic",
            ),
            ScalarMotionKeyframeV2(time=_time(20), value=1.0),
        ],
    )
    samples = [evaluate_scalar_curve(monotone, Fraction(frame, 4)) for frame in range(81)]
    assert samples[0] == 2.8
    assert samples[-1] == 1.0
    assert max(np.diff(samples)) <= 1e-10
    assert min(samples) >= 1.0

    with pytest.raises(ValueError, match="overshoot"):
        ScalarMotionCurveV2(
            overshoot_policy="forbid",
            keyframes=[
                ScalarMotionKeyframeV2(
                    time=_time(0),
                    value=0.0,
                    interpolation="cubic_hermite",
                    outgoing_tangent=1.0,
                ),
                ScalarMotionKeyframeV2(
                    time=_time(10),
                    value=1.0,
                    incoming_tangent=-1.0,
                ),
            ],
        )


def test_explicit_hermite_can_enforce_c2_when_the_authoring_model_justifies_it() -> None:
    quadratic = ScalarMotionCurveV2(
        continuity="c2",
        keyframes=[
            ScalarMotionKeyframeV2(
                time=_time(0),
                value=0.0,
                interpolation="cubic_hermite",
                outgoing_tangent=0.0,
            ),
            ScalarMotionKeyframeV2(
                time=_time(10),
                value=100.0,
                interpolation="cubic_hermite",
                incoming_tangent=20.0,
                outgoing_tangent=20.0,
            ),
            ScalarMotionKeyframeV2(
                time=_time(20),
                value=400.0,
                incoming_tangent=40.0,
            ),
        ],
    )
    assert evaluate_scalar_curve(quadratic, Fraction(15)) == pytest.approx(225.0)

    with pytest.raises(ValueError, match="C2"):
        ScalarMotionCurveV2(
            continuity="c2",
            keyframes=[
                ScalarMotionKeyframeV2(
                    time=_time(0),
                    value=0.0,
                    interpolation="cubic_hermite",
                    outgoing_tangent=0.0,
                ),
                ScalarMotionKeyframeV2(
                    time=_time(10),
                    value=100.0,
                    interpolation="cubic_hermite",
                    incoming_tangent=20.0,
                    outgoing_tangent=20.0,
                ),
                ScalarMotionKeyframeV2(
                    time=_time(20),
                    value=400.0,
                    incoming_tangent=30.0,
                ),
            ],
        )


def test_random_monotone_curves_remain_bounded_and_deterministic() -> None:
    rng = np.random.default_rng(20260719)
    for _ in range(20):
        times = [0, 7, 19, 31, 48]
        values = sorted(rng.uniform(0.8, 3.0, len(times)), reverse=True)
        curve = ScalarMotionCurveV2(
            continuity="c1",
            monotonicity="decreasing",
            overshoot_policy="forbid",
            keyframes=[
                ScalarMotionKeyframeV2(
                    time=_time(frame),
                    value=float(value),
                    interpolation=(
                        "monotone_cubic" if index < len(times) - 1 else "linear"
                    ),
                )
                for index, (frame, value) in enumerate(zip(times, values, strict=True))
            ],
        )
        samples = [
            evaluate_scalar_curve(curve, Fraction(frame, 8))
            for frame in range(times[-1] * 8 + 1)
        ]
        assert min(samples) >= values[-1] - 1e-10
        assert max(samples) <= values[0] + 1e-10
        assert max(np.diff(samples)) <= 1e-10
        assert samples == [
            evaluate_scalar_curve(curve, Fraction(frame, 8))
            for frame in range(times[-1] * 8 + 1)
        ]


def test_angle_unwrap_and_vector_path_curve_preserve_motion_semantics() -> None:
    rotation = ScalarMotionCurveV2(
        angle_unwrap=True,
        keyframes=[
            ScalarMotionKeyframeV2(time=_time(0), value=170.0),
            ScalarMotionKeyframeV2(time=_time(10), value=-170.0),
        ],
    )
    assert evaluate_scalar_curve(rotation, Fraction(5)) == pytest.approx(180.0)
    assert evaluate_scalar_curve(rotation, Fraction(10)) == pytest.approx(190.0)

    position = Vec2MotionCurveV2(
        continuity="c1",
        keyframes=[
            Vec2MotionKeyframeV2(
                time=_time(0),
                value=Vec2ValueV2(x=0.2, y=0.7),
                interpolation="cubic_bezier",
                outgoing_handle=Vec2BezierHandleV2(
                    time_offset_frames=3.0,
                    x_offset=0.0,
                    y_offset=-0.08,
                ),
            ),
            Vec2MotionKeyframeV2(
                time=_time(10),
                value=Vec2ValueV2(x=0.5, y=0.5),
                incoming_handle=Vec2BezierHandleV2(
                    time_offset_frames=-3.0,
                    x_offset=-0.08,
                    y_offset=0.0,
                ),
            ),
        ],
    )
    middle = evaluate_vec2_curve(position, Fraction(5))
    assert 0.2 < middle.x < 0.5
    assert 0.5 < middle.y < 0.7
    assert not math.isclose(
        middle.y,
        0.7 + (0.5 - 0.7) * ((middle.x - 0.2) / (0.5 - 0.2)),
        abs_tol=1e-4,
    )


def test_matrix_curve_and_decomposed_pivot_have_explicit_transform_semantics() -> None:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    translated = ((1.0, 0.0, 10.0), (0.0, 1.0, 20.0), (0.0, 0.0, 1.0))
    matrix_curve = MatrixMotionCurveV2(
        keyframes=[
            MatrixMotionKeyframeV2(
                time=_time(0),
                matrix_3x3=identity,
            ),
            MatrixMotionKeyframeV2(
                time=_time(10),
                matrix_3x3=translated,
            ),
        ]
    )
    assert evaluate_matrix_curve(matrix_curve, Fraction(5)) == pytest.approx(
        np.asarray(((1.0, 0.0, 5.0), (0.0, 1.0, 10.0), (0.0, 0.0, 1.0)))
    )

    duration = 10
    effect = TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            fit_mode="contain",
            position=_constant_vec2(0.5, 0.5, duration),
            scale=_constant_scalar(2.0, duration),
            rotation=_constant_scalar(0.0, duration),
            pivot=Vec2MotionCurveV2(
                keyframes=[
                    Vec2MotionKeyframeV2(
                        time=_time(0),
                        value=Vec2ValueV2(x=0.5, y=0.5),
                    ),
                    Vec2MotionKeyframeV2(
                        time=_time(duration),
                        value=Vec2ValueV2(x=0.25, y=0.5),
                    ),
                ]
            ),
        ),
        sampling=MotionSamplingV2(samples_per_frame=1),
        phases=[
            MotionPhaseV2(
                id="settle",
                start=_time(7),
                end=_time(10),
                channels=["position", "scale", "rotation"],
                intent="settle",
            )
        ],
        couplings=[
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["rotation"],
                relation="shared_progress",
                tolerance=0.01,
            )
        ],
    )
    start = evaluate_transform_matrix(
        effect,
        Fraction(0),
        source_width=100,
        source_height=100,
        output_width=100,
        output_height=200,
    )
    end = evaluate_transform_matrix(
        effect,
        Fraction(duration),
        source_width=100,
        source_height=100,
        output_width=100,
        output_height=200,
    )
    assert np.asarray(start) @ np.asarray([50.0, 50.0, 1.0]) == pytest.approx([50.0, 100.0, 1.0])
    assert np.asarray(end) @ np.asarray([25.0, 50.0, 1.0]) == pytest.approx([50.0, 100.0, 1.0])
    assert not np.allclose(start, end)

    shear = ((1.0, 0.25, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    affine = MatrixMotionCurveV2(
        keyframes=[
            MatrixMotionKeyframeV2(time=_time(0), matrix_3x3=shear),
            MatrixMotionKeyframeV2(time=_time(duration), matrix_3x3=shear),
        ]
    )
    base = {
        "fit_mode": "contain",
        "position": _constant_vec2(0.5, 0.5, duration),
        "scale": _constant_scalar(1.0, duration),
        "rotation": _constant_scalar(30.0, duration),
        "pivot": _constant_vec2(0.5, 0.5, duration),
        "affine": affine,
    }
    rotate_then_affine = TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            **base,
            transform_order=["fit", "rotate", "affine", "scale", "translate"],
        )
    )
    affine_then_rotate = TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            **base,
            transform_order=["fit", "affine", "rotate", "scale", "translate"],
        )
    )
    first_order = evaluate_transform_matrix(
        rotate_then_affine,
        Fraction(0),
        source_width=100,
        source_height=100,
        output_width=100,
        output_height=100,
    )
    second_order = evaluate_transform_matrix(
        affine_then_rotate,
        Fraction(0),
        source_width=100,
        source_height=100,
        output_width=100,
        output_height=100,
    )
    assert not np.allclose(first_order, second_order)


def test_retime_and_v1_migration_preserve_curve_values() -> None:
    source = ScalarMotionCurveV2(
        keyframes=[
            ScalarMotionKeyframeV2(
                time=_time(0),
                value=0.0,
                interpolation="cubic_hermite",
                outgoing_tangent=0.2,
            ),
            ScalarMotionKeyframeV2(
                time=_time(10),
                value=1.0,
                incoming_tangent=0.0,
            ),
        ]
    )
    retimed = retime_scalar_curve(
        source,
        source_start=Fraction(0),
        source_end=Fraction(10),
        target_start=Fraction(5),
        target_end=Fraction(25),
    )
    for source_frame in range(11):
        target_frame = 5 + 2 * source_frame
        assert evaluate_scalar_curve(retimed, Fraction(target_frame)) == pytest.approx(
            evaluate_scalar_curve(source, Fraction(source_frame))
        )

    legacy = TransformEffect(
        fit_mode="contain",
        position_mode="canvas_center",
        position=[
            Vec2Keyframe(frame=0, x=0.3, y=0.6),
            Vec2Keyframe(frame=10, x=0.5, y=0.5, easing="ease_in_out"),
        ],
        scale=[
            ScalarKeyframe(frame=0, value=2.8),
            ScalarKeyframe(frame=10, value=1.0, easing="ease_in_out"),
        ],
        rotation=[
            ScalarKeyframe(frame=0, value=-7.2),
            ScalarKeyframe(frame=10, value=0.0, easing="ease_in_out"),
        ],
    )
    migrated = migrate_transform_effect_v1(legacy)
    assert migrated.motion.kind == "decomposed"
    serialized = TransformEffectV2.model_validate_json(migrated.model_dump_json())
    assert serialized == migrated
    for frame in range(11):
        progress = frame / 10
        eased = progress * progress * (3.0 - 2.0 * progress)
        motion = migrated.motion
        assert motion.kind == "decomposed"
        assert evaluate_scalar_curve(motion.scale, Fraction(frame)) == pytest.approx(
            2.8 + (1.0 - 2.8) * eased,
            abs=1e-10,
        )
