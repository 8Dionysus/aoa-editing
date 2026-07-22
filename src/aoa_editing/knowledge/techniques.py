"""Extract and revise source-neutral technique packets from evaluation evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from aoa_editing.domain.models import (
    CameraMotionRecipe,
    DecomposedTransformMotionV2,
    MotionCouplingV2,
    MotionInterpolationV2,
    MotionPhaseV2,
    MotionSamplingV2,
    MotionTimeV2,
    PortableCameraMotionRecipeV2,
    Provenance,
    ReferenceComparisonReport,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionPassReceiptV2,
    ReferenceReconstructionSpec,
    ReferenceReconstructionSpecV2,
    ScalarKeyframe,
    ScalarMotionCurveV2,
    ScalarMotionKeyframeV2,
    TechniqueApplicationRecord,
    TechniqueCurveModelV2,
    TechniquePacket,
    TechniquePhaseModelV2,
    TechniqueResolutionConstraintV2,
    TransformEffectV2,
    Vec2Keyframe,
    Vec2MotionCurveV2,
    Vec2MotionKeyframeV2,
    Vec2ValueV2,
)


class TechniqueError(RuntimeError):
    """A packet cannot be supported by the supplied evaluation evidence."""


def extract_contain_reveal_technique(
    spec: ReferenceReconstructionSpec,
    comparison: ReferenceComparisonReport,
) -> TechniquePacket:
    """Extract normalized motion only after the frozen comparison has passed."""

    if comparison.overall != "pass" or comparison.spec_id != spec.id:
        raise TechniqueError("technique extraction requires a passing matching comparison")
    divisor = math.gcd(spec.camera_motion.width, spec.camera_motion.height)
    dense = len(spec.camera_motion.measurements) > 2

    def easing(index: int) -> str:
        return "linear" if index == 0 or dense else spec.camera_motion.curve_hint

    recipe = CameraMotionRecipe(
        duration_frames=spec.duration_frames,
        frame_rate=spec.camera_motion.frame_rate,
        aspect_width=spec.camera_motion.width // divisor,
        aspect_height=spec.camera_motion.height // divisor,
        fit_mode="contain",
        position_mode="canvas_center",
        position=[
            Vec2Keyframe(
                frame=item.frame,
                x=item.center_x,
                y=item.center_y,
                easing=easing(index),  # type: ignore[arg-type]
            )
            for index, item in enumerate(spec.camera_motion.measurements)
        ],
        scale=[
            ScalarKeyframe(
                frame=item.frame,
                value=item.scale_relative_to_contain,
                easing=easing(index),  # type: ignore[arg-type]
            )
            for index, item in enumerate(spec.camera_motion.measurements)
        ],
        rotation=[
            ScalarKeyframe(
                frame=item.frame,
                value=item.rotation_degrees,
                easing=easing(index),  # type: ignore[arg-type]
            )
            for index, item in enumerate(spec.camera_motion.measurements)
        ],
    )
    packet = TechniquePacket(
        id="technique_continuous_contain_reveal",
        title="Continuous close-detail to whole-image reveal",
        intent=(
            "Begin on a rotated close detail, then continuously settle into a centered "
            "whole-image contain view while preserving source texture."
        ),
        applicability=[
            "A single still image can support the complete visual result.",
            "The desired reveal uses one source-wide similarity transform.",
            "A portrait canvas intentionally exposes background around the contained image.",
        ],
        contraindications=[
            "Independent subjects or depth layers must move separately.",
            "The source lacks enough resolution for the opening crop.",
            "The composition requires semantic object replacement or generated pixels.",
        ],
        required_evidence=["image.statistics", "image.depth_layers"],
        allowed_operations=[
            "contain",
            "normalized-position-keyframes",
            "scale-keyframes",
            "rotation-keyframes",
            "virtual-camera",
        ],
        rules=[
            "Fit the complete source before camera scaling and rotate before transparent padding.",
            "Express placement as normalized source-center coordinates on the output canvas.",
            "Preserve a virtual source canvas large enough for the maximum camera scale.",
        ],
        heuristics=[
            "Use dense linear segments when measurements exist; endpoint easing is a fallback.",
            "Keep background explicit so the final contain reveal is intentional and reviewable.",
        ],
        creative_variants=[
            "Mirror the opening center while preserving the settle point.",
            "Reduce opening scale for lower-resolution sources.",
            "Reverse the curve for a whole-image to detail ending.",
        ],
        constraints=[
            "No reference pixels, audio, masks, or optical flow may enter render lineage.",
            "The output aspect ratio must match the packet unless the recipe is retimed/reframed.",
            "Every curve begins at frame zero and includes the final frame.",
        ],
        failure_modes=[
            "Pre-scaling to output resolution destroys opening-crop detail.",
            "Deeply nested FFmpeg expressions fail on dense curves.",
            "Rotating after padding changes the observed center path and exposes wrong edges.",
            "Feature-poor sources prevent objective camera-path remeasurement.",
        ],
        qc=[
            "Validate frame count, frame rate, aspect ratio, and source-only lineage.",
            "Remeasure scale, rotation, and normalized center across the rendered video.",
            "Review paired contact-sheet samples for blur, edge exposure, and temporal jumps.",
        ],
        positive_examples=[
            "A textured square illustration revealed inside a 9:16 canvas.",
            "A high-resolution photograph where one close detail motivates the opening.",
        ],
        negative_examples=[
            "A portrait requiring independently moving face, foreground, and background layers.",
            "A low-resolution icon enlarged beyond its available detail.",
        ],
        application_history=[
            TechniqueApplicationRecord(
                case_id=comparison.id,
                source_class="high-resolution textured square still",
                outcome="pass",
                evidence_path="runtime-eval:reference/comparison.json",
                metrics={
                    "vmaf_mean": float(comparison.perceptual["vmaf_mean"]),
                    "ssim_all": float(comparison.perceptual["ssim_all"]),
                    "scale_rmse": float(comparison.motion["scale_rmse"]),
                    "rotation_rmse_degrees": float(
                        comparison.motion["rotation_rmse_degrees"]
                    ),
                    "center_rmse_normalized": float(
                        comparison.motion["center_rmse_normalized"]
                    ),
                },
            )
        ],
        status="candidate",
        camera_motion=recipe,
        promotion_questions=[
            "Does the technique remain stable across photographic, graphic, and alpha sources?",
            "What automatic resolution guard should cap opening scale?",
            "Which aspect-ratio adaptations preserve the intended composition?",
            "How many successful independent applications justify evaluated status?",
        ],
        provenance=Provenance(
            tool="aoa-editing-technique-extractor",
            tool_version="0.1.0",
            parameters={
                "source_contract": "passing frozen comparison",
                "spec_id": spec.id,
                "comparison_id": comparison.id,
                "media_identifiers_retained": False,
            },
            deterministic=True,
        ),
    )
    serialized = packet.model_dump_json()
    if spec.source_sha256 in serialized or spec.reference_sha256 in serialized:
        raise TechniqueError("source-specific media hashes leaked into technique packet")
    return packet


def add_application_record(
    packet: TechniquePacket, record: TechniqueApplicationRecord
) -> TechniquePacket:
    if any(item.case_id == record.case_id for item in packet.application_history):
        raise TechniqueError(f"application case already exists: {record.case_id}")
    return packet.model_copy(
        update={
            "revision": packet.revision + 1,
            "application_history": [*packet.application_history, record],
        }
    )


def upgrade_contain_reveal_technique_v2(
    packet: TechniquePacket,
    spec: ReferenceReconstructionSpecV2,
    selected_pass: ReferenceReconstructionPassReceiptV2,
) -> TechniquePacket:
    """Upgrade the v1 candidate from approved semantics, never from media pixels."""

    if packet.id != "technique_continuous_contain_reveal":
        raise TechniqueError("unexpected technique family for v2 upgrade")
    if packet.schema_version != "1.0.0" or packet.revision != 2:
        raise TechniqueError("v2 upgrade requires the checked-in revision-two packet")
    correction = selected_pass.phase_correction
    if (
        selected_pass.overall != "pass"
        or selected_pass.pass_kind != "phase_compensated_matrix"
        or selected_pass.sequence != "5r"
        or correction is None
        or selected_pass.spec_id != spec.id
        or selected_pass.spec_sha256 != correction.spec_sha256
        or correction.spec_id != spec.id
        or correction.duration_frames != spec.duration_frames
    ):
        raise TechniqueError("v2 upgrade requires the matching passing selected semantics")

    coupling = cast(dict[str, Any], spec.phase_model.get("channel_coupling", {}))
    twist = cast(dict[str, Any], spec.phase_model.get("twist_candidate", {}))
    if (
        coupling.get("center_phase_relation") != "center_lags"
        or coupling.get("leading_explanation")
        != "staggered_center_path_with_synchronized_scale_rotation"
    ):
        raise TechniqueError("selected semantics do not support the portable phase model")

    centers: list[tuple[float, float]] = []
    scales: list[float] = []
    rotations: list[float] = []
    for output_frame in range(spec.duration_frames):
        source_frame = _evaluate_time_warp(correction, output_frame)
        lower = math.floor(source_frame)
        upper = math.ceil(source_frame)
        progress = source_frame - lower
        left = spec.motion_frames[lower]
        right = spec.motion_frames[upper]
        centers.append(
            (
                _lerp(left.center_x, right.center_x, progress),
                _lerp(left.center_y, right.center_y, progress),
            )
        )
        scales.append(
            _lerp(
                left.scale_relative_to_contain,
                right.scale_relative_to_contain,
                progress,
            )
        )
        rotations.append(
            _lerp(left.rotation_degrees, right.rotation_degrees, progress)
        )

    divisor = math.gcd(spec.width, spec.height)
    phase = TechniquePhaseModelV2(
        onset_frame=int(correction.predicted_phase["onset"]),
        twist_start_frame=int(twist["start_frame"]),
        twist_peak_frame=int(twist["peak_frame"]),
        twist_end_frame=int(twist["end_frame"]),
        settle_frame=int(correction.predicted_phase["settle"]),
        center_phase_relation="center_lags",
        scale_rotation_relation="synchronized",
        scale_rotation_progress_tolerance=float(
            coupling["scale_rotation_progress_rmse"]
        ),
        maximum_center_phase_offset=float(coupling["maximum_phase_offset"]),
    )
    recipe = PortableCameraMotionRecipeV2(
        duration_frames=spec.duration_frames,
        frame_rate=spec.frame_rate,
        aspect_width=spec.width // divisor,
        aspect_height=spec.height // divisor,
        centers=centers,
        scales_relative_to_contain=scales,
        rotations_degrees=rotations,
        sampling=MotionSamplingV2(),
    )
    upgraded = TechniquePacket.model_validate(
        {
            **packet.model_dump(mode="python"),
            "schema_version": "2.0.0",
            "revision": 3,
            "required_evidence": [
                "image.statistics",
                "image.depth_layers",
                "editing.composition_applicability.v2",
            ],
            "allowed_operations": [
                "contain",
                "normalized-output-center",
                "c1-hermite-motion",
                "phase-aware-retiming",
                "coupled-scale-rotation",
                "fixed-normalized-pivot",
                "virtual-camera",
            ],
            "rules": [
                "Evaluate applicability and pixel density before creating a Treatment.",
                "Fit the complete source before camera scaling.",
                "Express placement as normalized output-center coordinates.",
                "Keep the pivot fixed at normalized source center because moving pivot "
                "is not identifiable from the retained global transform.",
                "Preserve the lagging center path while scale and rotation share progress.",
            ],
            "heuristics": [
                "Retain one normalized sample per output frame and derive C1 Hermite tangents.",
                "Adapt contain fit to source aspect without changing normalized editorial timing.",
                "Refuse rather than soften the opening when pixel density exceeds the packet cap.",
            ],
            "constraints": [
                "No reference pixels, audio, masks, optical flow, hashes, or physical paths "
                "may enter the packet or render lineage.",
                "Output canvas aspect must match the authored 9:16 technique contract.",
                "The source composition must be explainable by one global similarity transform.",
                "Maximum effective upscale is measured after contain fit and peak camera scale.",
                "Every channel has exactly one sample for every output frame.",
            ],
            "failure_modes": [
                "A low-resolution source would require invented opening-crop detail.",
                "A layered composition needs independent motion and cannot use "
                "one global transform.",
                "Changing output aspect without an authored reframe changes composition.",
                "Replacing the lagging center path with shared linear progress loses the twist.",
                "A moving pivot claim would overstate what global-transform evidence identifies.",
                "Feature-poor sources can prevent independent media-path remeasurement.",
            ],
            "qc": [
                "Validate frame count, frame rate, aspect, source-only lineage, and fixed pivot.",
                "Compare every compiled compositor matrix with the normalized packet samples.",
                "Check discrete position, scale, and rotation continuity across all frames.",
                "Render preview and final, then validate QC, Kdenlive/MLT, and OTIO.",
                "Replay each positive case in a distinct workspace and compare final hashes.",
                "Require typed pre-render refusal for resolution and composition "
                "contraindications.",
            ],
            "positive_examples": [
                "A square illustration with enough source pixels for the opening detail.",
                "A licensed photograph whose composition tolerates one global camera move.",
                "A portrait source reframed by normalized contain geometry.",
                "An alpha image composited over an explicit background.",
                "A complex texture with sufficient resolution.",
            ],
            "negative_examples": [
                "A small icon whose peak effective upscale exceeds the declared cap.",
                "A multi-panel or layered scene requiring independently timed subject motion.",
                "An output aspect change without an authored composition adaptation.",
            ],
            "camera_motion": None,
            "curve_model_v2": TechniqueCurveModelV2(),
            "phase_model_v2": phase,
            "resolution_constraint_v2": TechniqueResolutionConstraintV2(
                maximum_effective_upscale=2.0
            ),
            "portable_camera_motion_v2": recipe,
            "promotion_questions": [
                "Does a human Tree of Editing review accept the phase model as editorially "
                "portable rather than reference-bound?",
                "Should the two-times effective-upscale cap vary by source texture or delivery?",
                "Which explicit reframe operation can safely adapt the technique beyond 9:16?",
                "Do alpha-edge and photographic-detail reviews justify a quality-specific guard?",
                "How many human-reviewed productions are required before evaluated status?",
            ],
            "provenance": Provenance(
                tool="aoa-editing-technique-upgrade-v2",
                tool_version="0.1.0",
                parameters={
                    "source_contract": "passing_selected_v2_semantics",
                    "media_identifiers_retained": False,
                    "physical_paths_retained": False,
                    "curve_model": "dense_phase_compensated_c1_similarity_v2",
                },
                deterministic=True,
            ),
        }
    )
    serialized = upgraded.model_dump_json()
    forbidden = (spec.source_sha256, spec.reference_sha256)
    if any(value in serialized for value in forbidden):
        raise TechniqueError("source-specific media hash leaked into v2 technique packet")
    return upgraded


def portable_motion_effect_v2(packet: TechniquePacket) -> TransformEffectV2:
    """Compile a v2 packet into canonical Motion Language v2 authoring meaning."""

    recipe = packet.portable_camera_motion_v2
    phase = packet.phase_model_v2
    curve_model = packet.curve_model_v2
    if recipe is None or phase is None or curve_model is None:
        raise TechniqueError("packet lacks the portable v2 camera recipe")
    duration = recipe.duration_frames
    center_tangents = _vector_tangents(recipe.centers)
    scale_tangents = _scalar_tangents(recipe.scales_relative_to_contain)
    rotation_tangents = _scalar_tangents(recipe.rotations_degrees)

    position_keys: list[Vec2MotionKeyframeV2] = []
    scale_keys: list[ScalarMotionKeyframeV2] = []
    rotation_keys: list[ScalarMotionKeyframeV2] = []
    for frame in range(duration):
        final = frame == duration - 1
        interpolation: MotionInterpolationV2 = (
            "linear" if final else "cubic_hermite"
        )
        center = recipe.centers[frame]
        center_tangent = center_tangents[frame]
        position_keys.append(
            Vec2MotionKeyframeV2(
                time=MotionTimeV2(frame=frame),
                value=Vec2ValueV2(x=center[0], y=center[1]),
                interpolation=interpolation,
                incoming_tangent=(
                    Vec2ValueV2(x=center_tangent[0], y=center_tangent[1])
                    if frame > 0
                    else None
                ),
                outgoing_tangent=(
                    Vec2ValueV2(x=center_tangent[0], y=center_tangent[1])
                    if not final
                    else None
                ),
            )
        )
        scale_keys.append(
            ScalarMotionKeyframeV2(
                time=MotionTimeV2(frame=frame),
                value=recipe.scales_relative_to_contain[frame],
                interpolation=interpolation,
                incoming_tangent=scale_tangents[frame] if frame > 0 else None,
                outgoing_tangent=scale_tangents[frame] if not final else None,
            )
        )
        rotation_keys.append(
            ScalarMotionKeyframeV2(
                time=MotionTimeV2(frame=frame),
                value=recipe.rotations_degrees[frame],
                interpolation=interpolation,
                incoming_tangent=rotation_tangents[frame] if frame > 0 else None,
                outgoing_tangent=rotation_tangents[frame] if not final else None,
            )
        )

    channels = ["position", "scale", "rotation"]
    phases = [
        MotionPhaseV2(
            id="onset",
            start=MotionTimeV2(frame=0),
            end=MotionTimeV2(frame=max(1, phase.onset_frame)),
            channels=cast(Any, channels),
            intent="onset",
        ),
        MotionPhaseV2(
            id="accelerate",
            start=MotionTimeV2(frame=phase.onset_frame),
            end=MotionTimeV2(frame=phase.twist_peak_frame),
            channels=cast(Any, channels),
            intent="accelerate",
        ),
        MotionPhaseV2(
            id="twist",
            start=MotionTimeV2(frame=phase.twist_start_frame),
            end=MotionTimeV2(frame=phase.twist_end_frame),
            channels=cast(Any, channels),
            intent="twist",
        ),
        MotionPhaseV2(
            id="settle",
            start=MotionTimeV2(frame=phase.twist_end_frame),
            end=MotionTimeV2(frame=phase.settle_frame),
            channels=cast(Any, channels),
            intent="settle",
        ),
    ]
    return TransformEffectV2(
        motion=DecomposedTransformMotionV2(
            fit_mode=recipe.fit_mode,
            position=Vec2MotionCurveV2(
                keyframes=position_keys,
                continuity="c1",
            ),
            scale=ScalarMotionCurveV2(
                keyframes=scale_keys,
                continuity="c1",
                overshoot_policy="bounded",
                overshoot_limit_fraction=(
                    curve_model.maximum_interpolation_overshoot_fraction
                ),
            ),
            rotation=ScalarMotionCurveV2(
                keyframes=rotation_keys,
                continuity="c1",
                overshoot_policy="bounded",
                overshoot_limit_fraction=(
                    curve_model.maximum_interpolation_overshoot_fraction
                ),
                angle_unwrap=True,
            ),
            pivot=Vec2MotionCurveV2(
                keyframes=[
                    Vec2MotionKeyframeV2(
                        time=MotionTimeV2(frame=0),
                        value=Vec2ValueV2(x=recipe.pivot[0], y=recipe.pivot[1]),
                    ),
                    Vec2MotionKeyframeV2(
                        time=MotionTimeV2(frame=duration - 1),
                        value=Vec2ValueV2(x=recipe.pivot[0], y=recipe.pivot[1]),
                    ),
                ]
            ),
            transform_order=curve_model.transform_order,
        ),
        sampling=recipe.sampling,
        phases=phases,
        couplings=[
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["rotation"],
                relation="shared_progress",
                tolerance=phase.scale_rotation_progress_tolerance,
            ),
            MotionCouplingV2(
                driver_channel="scale",
                follower_channels=["position"],
                relation="phase_lag",
                tolerance=phase.maximum_center_phase_offset,
            ),
        ],
    )


def add_application_records(
    packet: TechniquePacket,
    records: Sequence[TechniqueApplicationRecord],
) -> TechniquePacket:
    """Append one corpus atomically and advance the packet revision exactly once."""

    case_ids = [item.case_id for item in records]
    existing = {item.case_id for item in packet.application_history}
    if not records:
        raise TechniqueError("application record batch cannot be empty")
    if len(case_ids) != len(set(case_ids)):
        raise TechniqueError("application record batch contains duplicate case IDs")
    if existing.intersection(case_ids):
        raise TechniqueError("application record batch repeats an existing case")
    return TechniquePacket.model_validate(
        {
            **packet.model_dump(mode="python"),
            "revision": packet.revision + 1,
            "application_history": [*packet.application_history, *records],
        }
    )


def load_packet(path: Path) -> TechniquePacket:
    return TechniquePacket.model_validate_json(path.read_text(encoding="utf-8"))


def _lerp(left: float, right: float, progress: float) -> float:
    return left + (right - left) * progress


def _evaluate_time_warp(
    correction: ReferencePhaseCorrectionV2,
    output_frame: int,
) -> float:
    for left, right in zip(correction.anchors, correction.anchors[1:], strict=False):
        if left.output_frame <= output_frame <= right.output_frame:
            progress = (output_frame - left.output_frame) / (
                right.output_frame - left.output_frame
            )
            return left.source_frame + progress * (
                right.source_frame - left.source_frame
            )
    raise TechniqueError("phase correction does not cover the portable output frame")


def _scalar_tangents(values: Sequence[float]) -> list[float]:
    return [
        (
            values[1] - values[0]
            if index == 0
            else values[-1] - values[-2]
            if index == len(values) - 1
            else (values[index + 1] - values[index - 1]) / 2.0
        )
        for index in range(len(values))
    ]


def _vector_tangents(
    values: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    x = _scalar_tangents([item[0] for item in values])
    y = _scalar_tangents([item[1] for item in values])
    return list(zip(x, y, strict=True))
