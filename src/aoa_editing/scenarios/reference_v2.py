"""Evidence-bounded reconstruction planners for frozen Reference Spec v2."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np

from aoa_editing.domain.models import (
    Asset,
    BezierHandleV2,
    Clip,
    DecomposedTransformMotionV2,
    EditPatch,
    EvidenceRecord,
    FrameRange,
    MatrixMotionCurveV2,
    MatrixMotionKeyframeV2,
    MatrixTransformMotionV2,
    MediaKind,
    MotionCouplingV2,
    MotionPhaseV2,
    MotionSamplingV2,
    MotionTimeV2,
    PatchOperation,
    ProjectVersion,
    Provenance,
    RationaleItem,
    ReconstructionHypothesisDecisionV2,
    ReferenceMotionFrameV2,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionPassKindV2,
    ReferenceReconstructionPassSequenceV2,
    ReferenceReconstructionSpecV2,
    ScalarMotionCurveV2,
    ScalarMotionKeyframeV2,
    Scenario,
    Track,
    TrackKind,
    TransformEffectV2,
    Treatment,
    Vec2BezierHandleV2,
    Vec2MotionCurveV2,
    Vec2MotionKeyframeV2,
    Vec2ValueV2,
)
from aoa_editing.domain.motion_v2 import (
    evaluate_scalar_curve,
    evaluate_transform_matrix,
)
from aoa_editing.scenarios.planners import PlanningError

PASS_SEQUENCE: dict[
    ReferenceReconstructionPassKindV2,
    ReferenceReconstructionPassSequenceV2,
] = {
    "timing_easing": 2,
    "continuous_tangents": 3,
    "moving_pivot_hypothesis": 4,
    "coupled_scale_rotation": 5,
    "phase_compensated_matrix": "5r",
}
MATERIAL_RESIDUAL_FLOW_P95_PIXELS = 0.5


def classify_reconstruction_hypotheses_v2(
    spec: ReferenceReconstructionSpecV2,
) -> dict[str, ReconstructionHypothesisDecisionV2]:
    """Gate added transform families from frozen evidence, never candidate scores."""

    selected_models = [frame.selected_model for frame in spec.motion_frames]
    nested_frames = sum(model in {"affine", "homography"} for model in selected_models)
    material_residual_frames = sum(
        frame.residual_flow_p95 > MATERIAL_RESIDUAL_FLOW_P95_PIXELS
        for frame in spec.motion_frames
    )
    valid_material_residual_frames = sum(
        frame.residual_flow_p95 > MATERIAL_RESIDUAL_FLOW_P95_PIXELS
        and frame.residual_valid_fraction >= 0.5
        for frame in spec.motion_frames
    )
    all_pivots_unidentifiable = all(
        frame.pivot_identifiability == "unidentifiable" for frame in spec.motion_frames
    )
    moving_pivot = ReconstructionHypothesisDecisionV2(
        hypothesis="moving_pivot",
        status=(
            "experimental_gauge_only"
            if all_pivots_unidentifiable
            else "evidence_supported"
        ),
        rationale=(
            "A moving pivot may be rendered only as a gauge-equivalent decomposition "
            "because the frozen global matrices cannot distinguish it from compensated "
            "translation."
            if all_pivots_unidentifiable
            else "The frozen evidence contains an identifiable pivot observation."
        ),
        measurements={
            "unidentifiable_frame_count": sum(
                frame.pivot_identifiability == "unidentifiable"
                for frame in spec.motion_frames
            ),
            "frame_count": spec.duration_frames,
            "transform_order_conclusion": spec.transform_order.get("conclusion"),
        },
        evidence_refs=[spec.evidence_id, spec.id],
        permitted_execution=(
            "render_observable_matrix_only"
            if all_pivots_unidentifiable
            else "render_as_bounded_hypothesis"
        ),
    )
    affine = ReconstructionHypothesisDecisionV2(
        hypothesis="affine_homography",
        status="evidence_supported" if nested_frames else "rejected_by_evidence",
        rationale=(
            "At least one frozen frame selected a higher-order global model."
            if nested_frames
            else "Every frozen frame selected similarity; nested affine and homography "
            "fits did not earn their additional degrees of freedom."
        ),
        measurements={
            "selected_affine_or_homography_frames": nested_frames,
            "frame_count": spec.duration_frames,
            "selected_model_fraction": nested_frames / spec.duration_frames,
            "frozen_motion_model": spec.motion_model,
        },
        evidence_refs=[spec.evidence_id, spec.id],
        permitted_execution=(
            "render_as_bounded_hypothesis"
            if nested_frames
            else "do_not_add_operation"
        ),
    )
    local_warp_supported = valid_material_residual_frames > max(
        2, math.floor(spec.duration_frames * 0.05)
    )
    local_warp = ReconstructionHypothesisDecisionV2(
        hypothesis="local_warp",
        status=(
            "evidence_supported" if local_warp_supported else "rejected_by_evidence"
        ),
        rationale=(
            "A temporally persistent, spatially valid residual exceeds the generic "
            "analysis-pixel materiality gate."
            if local_warp_supported
            else "No persistent residual exceeds the generic 0.5 analysis-pixel p95 "
            "materiality gate, so local warp would add unsupported complexity."
        ),
        measurements={
            "material_p95_threshold_analysis_pixels": (
                MATERIAL_RESIDUAL_FLOW_P95_PIXELS
            ),
            "frames_over_material_p95": material_residual_frames,
            "valid_frames_over_material_p95": valid_material_residual_frames,
            "required_persistent_frames_exclusive": max(
                2, math.floor(spec.duration_frames * 0.05)
            ),
            "maximum_residual_flow_p95": max(
                frame.residual_flow_p95 for frame in spec.motion_frames
            ),
        },
        evidence_refs=[spec.evidence_id, spec.id],
        permitted_execution=(
            "render_as_bounded_hypothesis"
            if local_warp_supported
            else "do_not_add_operation"
        ),
    )
    return {
        "moving_pivot": moving_pivot,
        "affine_homography": affine,
        "local_warp": local_warp,
    }


def reference_reconstruct_v2(
    project_id: str,
    base: ProjectVersion,
    asset: Asset,
    evidence: list[EvidenceRecord],
    spec: ReferenceReconstructionSpecV2,
    *,
    pass_kind: ReferenceReconstructionPassKindV2,
    phase_correction: ReferencePhaseCorrectionV2 | None = None,
) -> Treatment:
    """Translate one bounded v2 pass into an ordinary reversible treatment."""

    if asset.media_kind is not MediaKind.IMAGE:
        raise PlanningError("reference.reconstruct v2 requires an image source")
    if asset.sha256 != spec.source_sha256:
        raise PlanningError("source hash does not match the frozen reconstruction spec v2")
    if asset.metadata.width is None or asset.metadata.height is None:
        raise PlanningError("source dimensions are required for Motion Language v2")
    specification = _spec_evidence(evidence, asset.id, spec.id)
    hypotheses = classify_reconstruction_hypotheses_v2(spec)
    correction_evidence: EvidenceRecord | None = None
    if pass_kind == "phase_compensated_matrix":
        if phase_correction is None:
            raise PlanningError("phase-compensated pass requires correction evidence")
        if (
            phase_correction.spec_id != spec.id
            or phase_correction.source_sha256 != spec.source_sha256
            or phase_correction.reference_sha256 != spec.reference_sha256
            or phase_correction.duration_frames != spec.duration_frames
        ):
            raise PlanningError("phase correction does not match the frozen spec v2")
        correction_evidence = _phase_correction_evidence(
            evidence,
            asset.id,
            phase_correction.id,
        )
    elif phase_correction is not None:
        raise PlanningError("phase correction is only valid for phase-compensated pass")
    if pass_kind == "timing_easing":
        effect = _timing_easing_effect(spec)
        title = "Timing and easing without new spatial operations"
        summary = (
            "Replace the linear v1 feeling with one shared nonlinear progress curve "
            "while retaining an endpoint-only similarity path."
        )
        strengths = ["Shared nonlinear timing", "No added spatial transform family"]
        tradeoffs = ["The center remains constrained to a straight endpoint path"]
        rationale_claim = (
            "A shared ease-in/ease-out progress curve tests timing before adding spatial "
            "complexity."
        )
    elif pass_kind == "continuous_tangents":
        effect = _continuous_tangents_effect(spec)
        title = "Phase-aware continuous tangents"
        summary = (
            "Use evidence-selected phase knots and explicit C1 tangents for center, "
            "scale, and rotation."
        )
        strengths = ["C1 temporal continuity", "Observed curved center path"]
        tradeoffs = ["Sparse phase knots still approximate the all-frame matrix curve"]
        rationale_claim = (
            "Frozen phase boundaries and observed derivatives justify a sparse C1 "
            "similarity trajectory."
        )
    elif pass_kind == "moving_pivot_hypothesis":
        if hypotheses["moving_pivot"].status != "experimental_gauge_only":
            raise PlanningError("moving-pivot pass requires a bounded gauge hypothesis")
        effect = _moving_pivot_effect(
            spec,
            source_width=asset.metadata.width,
            source_height=asset.metadata.height,
        )
        title = "Gauge-equivalent moving-pivot hypothesis"
        summary = (
            "Render an explicitly compensated moving pivot without changing the "
            "observable sparse C1 matrix trajectory."
        )
        strengths = ["Directly tests the moving-pivot explanation"]
        tradeoffs = [
            "Pivot and translation are observationally gauge-equivalent",
            "The extra authoring complexity must be rejected if output does not improve",
        ]
        rationale_claim = hypotheses["moving_pivot"].rationale
    elif pass_kind == "coupled_scale_rotation":
        effect = _coupled_matrix_effect(spec)
        title = "Dense coupled scale and rotation matrix"
        summary = (
            "Execute every frozen source-to-output matrix while retaining the observed "
            "scale/rotation coupling and center lag as explicit interpretation metadata."
        )
        strengths = [
            "All-frame observable trajectory",
            "Evidence-backed scale/rotation coupling",
            "No unsupported pivot or transform-order claim",
        ]
        tradeoffs = [
            "The matrix curve preserves output meaning rather than a unique authoring decomposition"
        ]
        rationale_claim = (
            "The frozen evidence supports synchronized scale/rotation and a lagged curved "
            "center path; the observable dense matrix is the uniquely honest execution."
        )
    else:
        if phase_correction is None or correction_evidence is None:
            raise AssertionError("validated phase correction is missing")
        effect = _phase_compensated_matrix_effect(spec, phase_correction)
        title = "Evidence-bounded phase-compensated matrix"
        summary = (
            "Apply a localized monotonic retime inferred from the prior objective "
            "comparison while leaving the protected twist interval unchanged."
        )
        strengths = [
            "One-frame safety margin at both detected phase boundaries",
            "Central scale/rotation and curved-center coupling remains identity-mapped",
            "No new spatial transform family",
        ]
        tradeoffs = [
            "The retime compensates measured recovery and resampling bias rather than "
            "claiming a new reference-space operation"
        ]
        rationale_claim = (
            "The prior candidate passed every non-phase objective check; its remaining "
            "onset and settle bias supports a bounded temporal correction only."
        )

    evidence_refs = [specification.id]
    if correction_evidence is not None:
        evidence_refs.append(correction_evidence.id)

    picture = Track(
        kind=TrackKind.VIDEO,
        name=f"Reference reconstruction v2 pass {PASS_SEQUENCE[pass_kind]}",
        clips=[
            Clip(
                asset_id=asset.id,
                timeline_range=FrameRange(start=0, duration=spec.duration_frames),
                role=f"reference-reconstruction-v2-{pass_kind}",
                effects=[effect],
                evidence_refs=evidence_refs,
            )
        ],
    )
    operations = [
        PatchOperation(op="replace", path="/width", value=spec.width),
        PatchOperation(op="replace", path="/height", value=spec.height),
        PatchOperation(
            op="replace",
            path="/frame_rate",
            value=spec.frame_rate.model_dump(mode="json"),
        ),
        PatchOperation(op="replace", path="/duration_frames", value=spec.duration_frames),
        PatchOperation(op="replace", path="/background", value="#000000"),
        PatchOperation(
            op="replace",
            path="/tracks",
            value=[picture.model_dump(mode="json")],
        ),
    ]
    patch = EditPatch(
        project_id=project_id,
        base_version_id=base.id,
        operations=operations,
        rationale=(
            f"Execute bounded reconstruction v2 pass {PASS_SEQUENCE[pass_kind]} "
            f"({pass_kind}) from frozen parameter evidence only."
        ),
        evidence_refs=evidence_refs,
    )
    return Treatment(
        project_id=project_id,
        base_version_id=base.id,
        scenario=Scenario.REFERENCE_RECONSTRUCT,
        title=title,
        summary=summary,
        structure=[
            "single permitted source",
            f"reconstruction pass {PASS_SEQUENCE[pass_kind]}",
            "Motion Language v2",
            "silent delivery",
        ],
        duration_frames=spec.duration_frames,
        used_asset_ids=[asset.id],
        strengths=strengths,
        tradeoffs=tradeoffs,
        uncertainties=list(spec.uncertainties),
        risks=[
            "Sub-pixel resampling and codec differences bound pixel identity",
            "Human editorial judgment remains a separate authority",
        ],
        expected_style="Smooth nonlinear detail-to-whole-image reveal with twist and settle",
        rationale=[
            RationaleItem(
                claim=rationale_claim,
                evidence_refs=evidence_refs,
                confidence=min(frame.confidence for frame in spec.motion_frames),
            )
        ],
        patch=patch,
        alternatives=[
            "Keep the previous bounded pass",
            "Reject added complexity if objective and diagnostic visual evidence do not improve",
        ],
        planner=Provenance(
            tool="aoa-editing-reference-planner-v2",
            tool_version="0.1.0",
            parameters={
                "planner": "reference-spec-v2",
                "spec_id": spec.id,
                "pass_kind": pass_kind,
                "pass_sequence": PASS_SEQUENCE[pass_kind],
                "reference_hash_used_as_media": False,
                "phase_correction_id": (
                    phase_correction.id if phase_correction is not None else None
                ),
                "hypothesis_gates": {
                    key: value.model_dump(mode="json")
                    for key, value in hypotheses.items()
                },
            },
            deterministic=True,
        ),
    )


def _spec_evidence(
    evidence: list[EvidenceRecord],
    asset_id: str,
    spec_id: str,
) -> EvidenceRecord:
    for record in evidence:
        if (
            record.asset_id == asset_id
            and record.kind == "reference.reconstruction_spec.v2"
            and record.payload.get("spec_id") == spec_id
        ):
            return record
    raise PlanningError("reference reconstruction spec v2 evidence is missing")


def _phase_correction_evidence(
    evidence: list[EvidenceRecord],
    asset_id: str,
    correction_id: str,
) -> EvidenceRecord:
    for record in evidence:
        if (
            record.asset_id == asset_id
            and record.kind == "reference.phase_correction.v2"
            and record.payload.get("correction_id") == correction_id
        ):
            return record
    raise PlanningError("reference phase correction v2 evidence is missing")


def _timing_easing_effect(spec: ReferenceReconstructionSpecV2) -> TransformEffectV2:
    first = spec.motion_frames[0]
    last = spec.motion_frames[-1]
    start = MotionTimeV2(frame=0)
    end = MotionTimeV2(frame=spec.duration_frames - 1)
    duration = float(spec.duration_frames - 1)
    outgoing_time = duration / 3.0
    incoming_time = -duration / 3.0

    def scalar(start_value: float, end_value: float, *, angle: bool = False) -> ScalarMotionCurveV2:
        return ScalarMotionCurveV2(
            continuity="c0",
            angle_unwrap=angle,
            keyframes=[
                ScalarMotionKeyframeV2(
                    time=start,
                    value=start_value,
                    interpolation="cubic_bezier",
                    outgoing_handle=BezierHandleV2(
                        time_offset_frames=outgoing_time,
                        value_offset=0.0,
                    ),
                ),
                ScalarMotionKeyframeV2(
                    time=end,
                    value=end_value,
                    incoming_handle=BezierHandleV2(
                        time_offset_frames=incoming_time,
                        value_offset=0.0,
                    ),
                ),
            ],
        )

    position = Vec2MotionCurveV2(
        continuity="c0",
        keyframes=[
            Vec2MotionKeyframeV2(
                time=start,
                value=Vec2ValueV2(x=first.center_x, y=first.center_y),
                interpolation="cubic_bezier",
                outgoing_handle=Vec2BezierHandleV2(
                    time_offset_frames=outgoing_time,
                    x_offset=0.0,
                    y_offset=0.0,
                ),
            ),
            Vec2MotionKeyframeV2(
                time=end,
                value=Vec2ValueV2(x=last.center_x, y=last.center_y),
                incoming_handle=Vec2BezierHandleV2(
                    time_offset_frames=incoming_time,
                    x_offset=0.0,
                    y_offset=0.0,
                ),
            ),
        ],
    )
    return TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            fit_mode="contain",
            position=position,
            scale=scalar(
                first.scale_relative_to_contain,
                last.scale_relative_to_contain,
            ),
            rotation=scalar(
                first.rotation_degrees,
                last.rotation_degrees,
                angle=True,
            ),
            pivot=_fixed_pivot_curve(spec.duration_frames),
        ),
        sampling=MotionSamplingV2(),
        phases=_phases(spec, channels=["position", "scale", "rotation"]),
        couplings=[
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["rotation", "position"],
                relation="shared_progress",
                tolerance=0.0,
            )
        ],
    )


def _continuous_tangents_effect(
    spec: ReferenceReconstructionSpecV2,
) -> TransformEffectV2:
    frames = [spec.motion_frames[index] for index in _phase_frame_indices(spec)]
    fps = spec.frame_rate.fps

    def scalar(
        getter: Callable[[ReferenceMotionFrameV2], float],
        velocity_key: str,
        *,
        angle: bool = False,
    ) -> ScalarMotionCurveV2:
        values = [getter(frame) for frame in frames]
        keyframes: list[ScalarMotionKeyframeV2] = []
        for index, frame in enumerate(frames):
            tangent = frame.velocity[velocity_key] / fps
            keyframes.append(
                ScalarMotionKeyframeV2(
                    time=MotionTimeV2(frame=frame.frame),
                    value=values[index],
                    interpolation=(
                        "cubic_hermite" if index < len(frames) - 1 else "linear"
                    ),
                    incoming_tangent=tangent if index > 0 else None,
                    outgoing_tangent=tangent if index < len(frames) - 1 else None,
                )
            )
        return ScalarMotionCurveV2(
            keyframes=keyframes,
            continuity="c1",
            monotonicity=_monotonicity(values),
            overshoot_policy="allow",
            angle_unwrap=angle,
        )

    positions: list[Vec2MotionKeyframeV2] = []
    for index, frame in enumerate(frames):
        tangent = Vec2ValueV2(
            x=frame.velocity["center_x"] / fps,
            y=frame.velocity["center_y"] / fps,
        )
        positions.append(
            Vec2MotionKeyframeV2(
                time=MotionTimeV2(frame=frame.frame),
                value=Vec2ValueV2(x=frame.center_x, y=frame.center_y),
                interpolation=(
                    "cubic_hermite" if index < len(frames) - 1 else "linear"
                ),
                incoming_tangent=tangent if index > 0 else None,
                outgoing_tangent=tangent if index < len(frames) - 1 else None,
            )
        )
    coupling = cast(dict[str, Any], spec.phase_model.get("channel_coupling", {}))
    return TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            fit_mode="contain",
            position=Vec2MotionCurveV2(keyframes=positions, continuity="c1"),
            scale=scalar(
                lambda frame: frame.scale_relative_to_contain,
                "scale",
            ),
            rotation=scalar(
                lambda frame: frame.rotation_degrees,
                "rotation",
                angle=True,
            ),
            pivot=_fixed_pivot_curve(spec.duration_frames),
        ),
        sampling=MotionSamplingV2(),
        phases=_phases(spec, channels=["position", "scale", "rotation"]),
        couplings=[
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["rotation"],
                relation="shared_progress",
                tolerance=float(coupling.get("scale_rotation_progress_rmse", 0.0)),
            ),
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["position"],
                relation="phase_lag",
                tolerance=float(coupling.get("maximum_phase_offset", 0.0)),
            ),
        ],
    )


def _moving_pivot_effect(
    spec: ReferenceReconstructionSpecV2,
    *,
    source_width: int,
    source_height: int,
) -> TransformEffectV2:
    base_effect = _continuous_tangents_effect(spec)
    if not isinstance(base_effect.motion, DecomposedTransformMotionV2):
        raise AssertionError("continuous-tangent pass must be decomposed")
    first = spec.motion_frames[0]
    last = spec.motion_frames[-1]
    residuals = []
    for frame in spec.motion_frames:
        progress = frame.frame / (spec.duration_frames - 1)
        residuals.append(
            (
                frame.center_x
                - (first.center_x + (last.center_x - first.center_x) * progress),
                frame.center_y
                - (first.center_y + (last.center_y - first.center_y) * progress),
            )
        )
    if max(math.hypot(x, y) for x, y in residuals) <= 1e-12:
        residuals = [
            (0.005 * math.sin(math.pi * frame.frame / (spec.duration_frames - 1)), 0.0)
            for frame in spec.motion_frames
        ]
    positions: list[Vec2MotionKeyframeV2] = []
    scales: list[ScalarMotionKeyframeV2] = []
    rotations: list[ScalarMotionKeyframeV2] = []
    pivots: list[Vec2MotionKeyframeV2] = []
    for frame, residual in zip(spec.motion_frames, residuals, strict=True):
        time = MotionTimeV2(frame=frame.frame)
        pivot = Vec2ValueV2(
            x=min(0.95, max(0.05, 0.5 + residual[0])),
            y=min(0.95, max(0.05, 0.5 + residual[1])),
        )
        desired = np.asarray(
            evaluate_transform_matrix(
                base_effect,
                frame.frame,
                source_width=source_width,
                source_height=source_height,
                output_width=spec.width,
                output_height=spec.height,
            ),
            dtype=np.float64,
        )
        mapped = desired @ np.asarray(
            (pivot.x * source_width, pivot.y * source_height, 1.0),
            dtype=np.float64,
        )
        mapped /= mapped[2]
        positions.append(
            Vec2MotionKeyframeV2(
                time=time,
                value=Vec2ValueV2(
                    x=float(mapped[0] / spec.width),
                    y=float(mapped[1] / spec.height),
                ),
            )
        )
        scales.append(
            ScalarMotionKeyframeV2(
                time=time,
                value=evaluate_scalar_curve(base_effect.motion.scale, frame.frame),
            )
        )
        rotations.append(
            ScalarMotionKeyframeV2(
                time=time,
                value=evaluate_scalar_curve(base_effect.motion.rotation, frame.frame),
            )
        )
        pivots.append(Vec2MotionKeyframeV2(time=time, value=pivot))
    return TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            fit_mode="contain",
            position=Vec2MotionCurveV2(keyframes=positions),
            scale=ScalarMotionCurveV2(
                keyframes=scales,
                angle_unwrap=False,
            ),
            rotation=ScalarMotionCurveV2(
                keyframes=rotations,
                angle_unwrap=True,
            ),
            pivot=Vec2MotionCurveV2(keyframes=pivots),
        ),
        sampling=MotionSamplingV2(),
        phases=_phases(spec, channels=["position", "scale", "rotation", "pivot"]),
        couplings=[
            MotionCouplingV2(
                driver_channel="pivot",
                follower_channels=["position"],
                relation="explicit_matrix",
                tolerance=1e-9,
            )
        ],
    )


def _coupled_matrix_effect(
    spec: ReferenceReconstructionSpecV2,
) -> TransformEffectV2:
    coupling = cast(dict[str, Any], spec.phase_model.get("channel_coupling", {}))
    return TransformEffectV2(
        motion=MatrixTransformMotionV2(
            matrix=MatrixMotionCurveV2(
                keyframes=[
                    MatrixMotionKeyframeV2(
                        time=MotionTimeV2(frame=frame.frame),
                        matrix_3x3=cast(Any, frame.matrix_3x3),
                    )
                    for frame in spec.motion_frames
                ]
            )
        ),
        sampling=MotionSamplingV2(),
        phases=_phases(spec, channels=["matrix"]),
        couplings=[
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["rotation"],
                relation="shared_progress",
                tolerance=float(coupling.get("scale_rotation_progress_rmse", 0.0)),
            )
        ],
    )


def _phase_compensated_matrix_effect(
    spec: ReferenceReconstructionSpecV2,
    correction: ReferencePhaseCorrectionV2,
) -> TransformEffectV2:
    """Sample the frozen matrix curve through a bounded monotonic time map."""

    coupling = cast(dict[str, Any], spec.phase_model.get("channel_coupling", {}))
    keyframes: list[MatrixMotionKeyframeV2] = []
    for output_frame in range(spec.duration_frames):
        source_frame = evaluate_time_warp_v2(correction, output_frame)
        lower = math.floor(source_frame)
        upper = math.ceil(source_frame)
        progress = source_frame - lower
        left = np.asarray(spec.motion_frames[lower].matrix_3x3, dtype=np.float64)
        right = np.asarray(spec.motion_frames[upper].matrix_3x3, dtype=np.float64)
        matrix = left + (right - left) * progress
        keyframes.append(
            MatrixMotionKeyframeV2(
                time=MotionTimeV2(frame=output_frame),
                matrix_3x3=cast(Any, matrix.tolist()),
            )
        )
    return TransformEffectV2(
        motion=MatrixTransformMotionV2(
            matrix=MatrixMotionCurveV2(keyframes=keyframes)
        ),
        sampling=MotionSamplingV2(),
        phases=_phases(spec, channels=["matrix"]),
        couplings=[
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["rotation"],
                relation="shared_progress",
                tolerance=float(coupling.get("scale_rotation_progress_rmse", 0.0)),
            )
        ],
    )


def evaluate_time_warp_v2(
    correction: ReferencePhaseCorrectionV2,
    output_frame: int,
) -> float:
    """Evaluate a validated piecewise-linear correction at one output frame."""

    if output_frame < 0 or output_frame >= correction.duration_frames:
        raise ValueError("output frame is outside the phase correction duration")
    for left, right in pairwise(correction.anchors):
        if left.output_frame <= output_frame <= right.output_frame:
            progress = (output_frame - left.output_frame) / (
                right.output_frame - left.output_frame
            )
            return left.source_frame + progress * (
                right.source_frame - left.source_frame
            )
    raise AssertionError("validated phase correction has no matching segment")


def _fixed_pivot_curve(duration_frames: int) -> Vec2MotionCurveV2:
    pivot = Vec2ValueV2(x=0.5, y=0.5)
    return Vec2MotionCurveV2(
        keyframes=[
            Vec2MotionKeyframeV2(time=MotionTimeV2(frame=0), value=pivot),
            Vec2MotionKeyframeV2(
                time=MotionTimeV2(frame=duration_frames - 1),
                value=pivot,
            ),
        ]
    )


def _phase_frame_indices(spec: ReferenceReconstructionSpecV2) -> list[int]:
    twist = cast(dict[str, Any], spec.phase_model.get("twist_candidate", {}))
    candidates = [
        0,
        int(spec.phase_model.get("onset_frame", 0)),
        int(twist.get("start_frame", 0)),
        int(twist.get("peak_frame", spec.phase_model.get("peak_velocity_frame", 0))),
        int(twist.get("end_frame", spec.duration_frames - 1)),
        int(spec.phase_model.get("settle_frame", spec.duration_frames - 1)),
        spec.duration_frames - 1,
    ]
    return sorted({min(spec.duration_frames - 1, max(0, value)) for value in candidates})


def _phases(
    spec: ReferenceReconstructionSpecV2,
    *,
    channels: list[Literal["position", "scale", "rotation", "pivot", "matrix"]],
) -> list[MotionPhaseV2]:
    twist = cast(dict[str, Any], spec.phase_model.get("twist_candidate", {}))
    last = spec.duration_frames - 1
    onset = min(last, max(0, int(spec.phase_model.get("onset_frame", 0))))
    peak = min(
        last,
        max(
            onset,
            int(
                twist.get(
                    "peak_frame",
                    spec.phase_model.get("peak_velocity_frame", onset),
                )
            ),
        ),
    )
    twist_start = min(last, max(0, int(twist.get("start_frame", onset))))
    twist_end = min(last, max(twist_start, int(twist.get("end_frame", peak))))
    settle = min(last, max(twist_end, int(spec.phase_model.get("settle_frame", last))))
    definitions = [
        ("onset", 0, max(1, onset), "onset"),
        ("accelerate", onset, peak, "accelerate"),
        ("twist", twist_start, twist_end, "twist"),
        ("settle", twist_end, settle, "settle"),
    ]
    phases: list[MotionPhaseV2] = []
    for identifier, start, end, intent in definitions:
        bounded_start = min(last, max(0, start))
        bounded_end = min(last, max(0, end))
        if bounded_start >= bounded_end:
            continue
        phases.append(
            MotionPhaseV2(
                id=identifier,
                start=MotionTimeV2(frame=bounded_start),
                end=MotionTimeV2(frame=bounded_end),
                channels=channels,
                intent=cast(Any, intent),
            )
        )
    return phases


def _monotonicity(
    values: list[float],
) -> Literal["none", "increasing", "decreasing", "auto"]:
    if all(left <= right for left, right in pairwise(values)):
        return "increasing"
    if all(left >= right for left, right in pairwise(values)):
        return "decreasing"
    return "none"
