from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from aoa_editing.domain.models import (
    AdjustPivotCurveCorrectionV2,
    AdjustRotationCouplingCorrectionV2,
    AdjustTangentCorrectionV2,
    FrameRange,
    MatrixMotionCurveV2,
    MatrixMotionKeyframeV2,
    MatrixTransformMotionV2,
    MotionCorrectionContextV2,
    MotionSamplingV2,
    MotionTimeV2,
    RetimeSettleCorrectionV2,
    TransformEffectV2,
)
from aoa_editing.domain.motion_corrections import (
    MotionCorrectionCompileError,
    compile_motion_corrections,
    infer_motion_correction_operations,
)
from aoa_editing.domain.motion_v2 import evaluate_matrix_curve


def _matrix(frame: int, *, duration: int) -> tuple[tuple[float, float, float], ...]:
    progress = frame / (duration - 1)
    eased = progress * progress * (3.0 - 2.0 * progress)
    scale = 1.45 - 0.45 * eased
    rotation = math.radians(-8.0 * (1.0 - eased))
    a = scale * math.cos(rotation)
    b = -scale * math.sin(rotation)
    center_x = 160.0 + 18.0 * eased + 4.0 * math.sin(math.pi * progress)
    center_y = 90.0 - 8.0 * eased
    source_x = 160.0
    source_y = 90.0
    tx = center_x - a * source_x - b * source_y
    ty = center_y - (-b * source_x + a * source_y)
    return (
        (a, b, tx),
        (-b, a, ty),
        (0.0, 0.0, 1.0),
    )


def _effect(duration: int = 41) -> TransformEffectV2:
    return TransformEffectV2(
        motion=MatrixTransformMotionV2(
            matrix=MatrixMotionCurveV2(
                keyframes=[
                    MatrixMotionKeyframeV2(
                        time=MotionTimeV2(frame=frame),
                        matrix_3x3=_matrix(frame, duration=duration),
                    )
                    for frame in range(duration)
                ]
            )
        ),
        sampling=MotionSamplingV2(),
    )


def _context() -> MotionCorrectionContextV2:
    return MotionCorrectionContextV2(
        duration_frames=41,
        onset_frame=2,
        twist_start_frame=8,
        twist_peak_frame=20,
        twist_end_frame=30,
        settle_frame=40,
        pivot_identifiable=False,
    )


def test_example_language_becomes_three_independent_semantic_operations() -> None:
    operations = infer_motion_correction_operations(
        "переход должен мягче подкручиваться и чуть позже оседать",
        _context(),
    )

    assert [operation.kind for operation in operations] == [
        "adjust_tangent",
        "adjust_rotation_coupling",
        "retime_settle",
    ]
    assert operations[0].affected_range == FrameRange(start=8, duration=23)
    assert operations[-1].affected_range.end == 41


def test_matrix_corrections_change_only_bounded_samples_and_preserve_endpoints() -> None:
    effect = _effect()
    operations = [
        AdjustTangentCorrectionV2(
            channel="rotation",
            anchor_frame=20,
            strength=0.35,
            affected_range=FrameRange(start=8, duration=23),
        ),
        AdjustRotationCouplingCorrectionV2(
            lag_delta_frames=1,
            smoothing_strength=0.2,
            affected_range=FrameRange(start=8, duration=23),
        ),
        RetimeSettleCorrectionV2(
            direction="later",
            strength=0.25,
            affected_range=FrameRange(start=30, duration=11),
        ),
    ]

    updated, diff = compile_motion_corrections(
        effect,
        operations,
        source_width=320,
        source_height=180,
        pivot_identifiable=False,
    )

    assert updated != effect
    assert diff.changed_frame_count > 0
    assert diff.affected_range == FrameRange(start=8, duration=33)
    assert diff.before_sha256 != diff.after_sha256
    before = effect.motion.matrix.keyframes
    after = updated.motion.matrix.keyframes
    assert after[0].matrix_3x3 == before[0].matrix_3x3
    assert after[7].matrix_3x3 == before[7].matrix_3x3
    assert after[-1].matrix_3x3 == before[-1].matrix_3x3
    assert after[20].matrix_3x3 != before[20].matrix_3x3
    for sample in (Fraction(17, 2), Fraction(41, 2), Fraction(159, 4)):
        matrix = evaluate_matrix_curve(updated.motion.matrix, sample)
        assert abs(float(np.linalg.det(matrix))) > 1e-8


def test_pivot_correction_fails_closed_when_matrix_evidence_is_gauge_equivalent() -> None:
    operation = AdjustPivotCurveCorrectionV2(
        delta_x=0.01,
        delta_y=-0.01,
        affected_range=FrameRange(start=8, duration=23),
    )
    with pytest.raises(MotionCorrectionCompileError, match="pivot is not identifiable"):
        compile_motion_corrections(
            _effect(),
            [operation],
            source_width=320,
            source_height=180,
            pivot_identifiable=False,
        )
