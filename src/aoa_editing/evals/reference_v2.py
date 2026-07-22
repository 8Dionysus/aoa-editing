"""All-frame Reference Motion Evidence v2 and immutable reconstruction spec."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from aoa_editing.analysis.motion import MotionAnalyzer, characterize_similarity_coupling
from aoa_editing.domain.models import (
    MediaKind,
    MotionRecoveryFrame,
    MotionRecoveryReport,
    Provenance,
    ReferenceMotionEvidenceV2,
    ReferenceMotionFrameV2,
    ReferenceReconstructionSpecV2,
)
from aoa_editing.evals.reference import _audio_structure, _scene_cuts
from aoa_editing.infrastructure.media import ffprobe, sha256_file


class ReferenceV2GateError(RuntimeError):
    """Generic evidence does not authorize v2 reference decoding."""


class ReferenceV2AnalysisError(RuntimeError):
    """Reference v2 evidence could not be completed without guessing."""


def analyze_reference_v2(
    source_path: Path,
    reference_path: Path,
    output_root: Path,
    *,
    readiness_path: Path,
    motion_gate_path: Path,
    baseline_comparison_path: Path | None = None,
) -> tuple[ReferenceMotionEvidenceV2, ReferenceReconstructionSpecV2]:
    """Recover every frame, write diagnostics, and freeze criteria before v2 rendering."""

    readiness_path = readiness_path.expanduser().resolve(strict=True)
    motion_gate_path = motion_gate_path.expanduser().resolve(strict=True)
    readiness = _object(readiness_path)
    motion_gate = _object(motion_gate_path)
    revision = _validate_gates(readiness, motion_gate)
    source_path = source_path.expanduser().resolve(strict=True)
    reference_path = reference_path.expanduser().resolve(strict=True)
    if output_root.exists():
        raise FileExistsError(f"reference v2 root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    kind, metadata, probe_payload, probe_command = ffprobe(reference_path)
    if kind is not MediaKind.VIDEO or not metadata.has_video:
        raise ReferenceV2AnalysisError("reference v2 input must contain a video stream")
    raw_recovery_path = output_root / "raw-motion-recovery.json"
    recovery = MotionAnalyzer().analyze_video(
        source_path,
        reference_path,
        output_path=raw_recovery_path,
    )
    if recovery.analyzed_frame_count != recovery.frame_count:
        raise ReferenceV2AnalysisError("all-frame analysis did not cover the declared stream")

    residuals, diagnostic_paths = _residual_diagnostics(
        source_path,
        reference_path,
        recovery,
        output_root / "diagnostics",
    )
    source_image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source_image is None:
        raise ReferenceV2AnalysisError(f"cannot decode permitted source: {source_path}")
    frames = _evidence_frames(
        recovery,
        residuals,
        source_width=source_image.shape[1],
        source_height=source_image.shape[0],
    )
    ambiguous = _ambiguous_frames(frames)
    phase_model = _phase_model(frames, recovery)
    transform_order = _transform_order_assessment(recovery)
    audio, audio_commands = _audio_structure(reference_path, metadata.duration_seconds)
    cuts, scene_command = _scene_cuts(reference_path)
    if cuts:
        phase_model["scene_cut_candidates_seconds"] = cuts

    artifacts = _plots_and_contact_sheet(
        frames,
        diagnostic_paths,
        output_root,
    )
    artifacts["raw_motion_recovery"] = str(raw_recovery_path)
    artifacts["residual_summary"] = str(
        _write_residual_summary(residuals, output_root / "residual-summary.json")
    )
    baseline = (
        _baseline_observation(baseline_comparison_path)
        if baseline_comparison_path is not None
        else None
    )
    uncertainties = list(recovery.uncertainties)
    uncertainties.extend(
        (
            "Dense optical flow measures residual image motion, not semantic object intent.",
            "Sub-pixel interpolation and codec reconstruction bound achievable pixel identity.",
            (
                "Transform order is only identifiable where competing matrix families "
                "produce measurably different projections."
            ),
        )
    )
    alternative_explanations = list(recovery.alternative_explanations)
    alternative_explanations.extend(
        (
            "A nonlinear timing curve can mimic a short moving-pivot impression.",
            "Coupled scale and rotation can mimic weak perspective over a narrow field.",
            "Structured residual flow can arise from codec motion compensation or local warp.",
        )
    )
    source_hash = sha256_file(source_path)
    reference_hash = sha256_file(reference_path)
    evidence = ReferenceMotionEvidenceV2(
        source_sha256=source_hash,
        reference_sha256=reference_hash,
        readiness_revision=revision,
        readiness_receipt_path=str(readiness_path),
        readiness_receipt_sha256=sha256_file(readiness_path),
        motion_gate_revision=revision,
        motion_gate_path=str(motion_gate_path),
        motion_gate_sha256=sha256_file(motion_gate_path),
        reference_metadata=metadata,
        audio=audio,
        frame_count=recovery.frame_count,
        frame_rate=recovery.frame_rate,
        width=recovery.width,
        height=recovery.height,
        selected_model=recovery.selected_model,
        model_distribution=recovery.model_distribution,
        model_mismatch=recovery.model_mismatch,
        curve_hint=recovery.curve_hint,
        curve_confidence=recovery.curve_confidence,
        phase_model=phase_model,
        transform_order_assessment=transform_order,
        frames=frames,
        ambiguous_frames=ambiguous,
        uncertainties=uncertainties,
        alternative_explanations=alternative_explanations,
        artifacts=artifacts,
        provenance=Provenance(
            tool="aoa-editing-reference-motion-analysis-v2",
            tool_version=version("aoa-editing"),
            command=[*probe_command, *scene_command, *audio_commands],
            parameters={
                "motion_algorithm": (
                    "all-frame SIFT correspondences + competing RANSAC "
                    "similarity/affine/homography"
                ),
                "residual_algorithm": "downscaled Farneback flow after recovered global warp",
                "opencv_version": version("opencv-python-headless"),
                "frame_sampling": "every_frame",
                "probe_stream_count": len(probe_payload.get("streams", [])),
                "readiness_sha256": sha256_file(readiness_path),
                "motion_gate_sha256": sha256_file(motion_gate_path),
            },
            deterministic=True,
        ),
    )
    criteria = _comparison_criteria(
        source_hash=source_hash,
        reference_hash=reference_hash,
        width=recovery.width,
        height=recovery.height,
        frame_rate=recovery.frame_rate.model_dump(mode="json"),
        baseline=baseline,
        motion_gate_sha256=sha256_file(motion_gate_path),
    )
    duration_seconds = metadata.duration_seconds or recovery.frame_count / recovery.frame_rate.fps
    spec = ReferenceReconstructionSpecV2(
        evidence_id=evidence.id,
        source_sha256=source_hash,
        reference_sha256=reference_hash,
        readiness_revision=revision,
        motion_gate_revision=revision,
        duration_frames=recovery.frame_count,
        duration_seconds=duration_seconds,
        frame_rate=recovery.frame_rate,
        width=recovery.width,
        height=recovery.height,
        audio=audio,
        motion_model=recovery.selected_model,
        motion_frames=frames,
        curve_representation={
            "kind": "dense_observed_matrix_curve",
            "sampling": "every_frame",
            "sample_count": len(frames),
            "interpolation_between_integer_frames": "subframe_motion_language_v2",
            "curve_hint": recovery.curve_hint,
            "curve_confidence": recovery.curve_confidence,
            "angle_unwrapped": True,
            "outliers_preserved_and_filtered": True,
        },
        phase_model=phase_model,
        transform_order=transform_order,
        uncertainties=uncertainties,
        alternative_explanations=alternative_explanations,
        comparison_criteria=criteria,
        artifacts=artifacts,
        provenance=Provenance(
            tool="aoa-editing-reference-spec-freezer-v2",
            tool_version=version("aoa-editing"),
            command=[],
            parameters={
                "evidence_schema": "reference-motion-evidence-v2",
                "threshold_revision": "reference-comparison-v2-frozen-1",
                "frozen_before_render": True,
            },
            deterministic=True,
        ),
    )
    evidence_path = output_root / "reference-motion-evidence-v2.json"
    spec_path = output_root / "reference-spec-v2.json"
    _write_model(evidence, evidence_path)
    _write_model(spec, spec_path)
    return evidence, spec


def _validate_gates(
    readiness: dict[str, Any],
    motion_gate: dict[str, Any],
) -> str:
    if readiness.get("overall") != "pass":
        raise ReferenceV2GateError("readiness receipt is not passing")
    if readiness.get("reference_content_decoded") is not False:
        raise ReferenceV2GateError("readiness does not prove sealed pre-analysis state")
    if motion_gate.get("overall") != "pass":
        raise ReferenceV2GateError("motion recovery gate is not passing")
    if motion_gate.get("git_clean") is not True:
        raise ReferenceV2GateError("motion recovery gate is not bound to a clean revision")
    if int(motion_gate.get("mandatory_skips", -1)) != 0:
        raise ReferenceV2GateError("motion recovery gate contains mandatory skips")
    readiness_revision = str(readiness.get("git_revision", ""))
    motion_revision = str(motion_gate.get("git_revision", ""))
    if (
        len(readiness_revision) != 40
        or len(motion_revision) != 40
        or readiness_revision != motion_revision
    ):
        raise ReferenceV2GateError(
            "readiness and motion recovery must belong to the same Git revision"
        )
    return readiness_revision


def _residual_diagnostics(
    source_path: Path,
    reference_path: Path,
    recovery: MotionRecoveryReport,
    output_root: Path,
) -> tuple[list[dict[str, float]], dict[int, dict[str, Path]]]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ReferenceV2AnalysisError(f"cannot decode permitted source: {source_path}")
    selected_source = cast(NDArray[np.uint8], source)
    capture = cv2.VideoCapture(str(reference_path))
    if not capture.isOpened():
        raise ReferenceV2AnalysisError(f"cannot decode reference: {reference_path}")
    by_frame = {item.frame: item for item in recovery.frames}
    metrics: list[dict[str, float]] = []
    decoded = 0
    try:
        while True:
            available, frame = capture.read()
            if not available:
                break
            motion = by_frame.get(decoded)
            if motion is None:
                raise ReferenceV2AnalysisError(f"missing recovery for frame {decoded}")
            mean, percentile, valid = _residual_flow(
                selected_source,
                cast(NDArray[np.uint8], frame),
                motion,
            )
            metrics.append(
                {
                    "frame": float(decoded),
                    "flow_mean": mean,
                    "flow_p95": percentile,
                    "valid_fraction": valid,
                }
            )
            decoded += 1
    finally:
        capture.release()
    if decoded != recovery.frame_count:
        raise ReferenceV2AnalysisError(
            f"residual pass decoded {decoded}, expected {recovery.frame_count}"
        )

    selected = _diagnostic_frames(metrics, recovery)
    output_root.mkdir(parents=True, exist_ok=False)
    artifacts: dict[int, dict[str, Path]] = {}
    capture = cv2.VideoCapture(str(reference_path))
    try:
        decoded = 0
        while True:
            available, frame = capture.read()
            if not available:
                break
            if decoded in selected:
                motion = by_frame[decoded]
                overlay, heatmap = _diagnostic_images(
                    selected_source,
                    cast(NDArray[np.uint8], frame),
                    motion,
                )
                overlay_path = output_root / f"frame-{decoded:04d}-overlay.jpg"
                residual_path = output_root / f"frame-{decoded:04d}-residual.png"
                if not cv2.imwrite(str(overlay_path), overlay):
                    raise ReferenceV2AnalysisError(f"cannot write overlay {overlay_path}")
                if not cv2.imwrite(str(residual_path), heatmap):
                    raise ReferenceV2AnalysisError(
                        f"cannot write residual map {residual_path}"
                    )
                artifacts[decoded] = {
                    "overlay": overlay_path,
                    "residual": residual_path,
                }
            decoded += 1
    finally:
        capture.release()
    return metrics, artifacts


def _residual_flow(
    source: NDArray[np.uint8],
    frame: NDArray[np.uint8],
    motion: MotionRecoveryFrame,
) -> tuple[float, float, float]:
    predicted, actual, valid = _aligned_small_frames(source, frame, motion)
    flow = cast(
        NDArray[np.float32],
        cast(Any, cv2.calcOpticalFlowFarneback)(
            predicted,
            actual,
            None,
            0.5,
            3,
            21,
            4,
            7,
            1.5,
            0,
        ),
    )
    magnitude = np.linalg.norm(flow, axis=2)
    selected = magnitude[valid]
    if not len(selected):
        return math.inf, math.inf, 0.0
    return (
        float(np.mean(selected)),
        float(np.percentile(selected, 95)),
        float(np.mean(valid)),
    )


def _aligned_small_frames(
    source: NDArray[np.uint8],
    frame: NDArray[np.uint8],
    motion: MotionRecoveryFrame,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.bool_]]:
    source_height, source_width = source.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    output_factor = min(1.0, 480.0 / max(frame_height, frame_width))
    source_factor = min(1.0, 640.0 / max(source_height, source_width))
    source_small = cv2.resize(
        source,
        (
            max(1, round(source_width * source_factor)),
            max(1, round(source_height * source_factor)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    output_size = (
        max(1, round(frame_width * output_factor)),
        max(1, round(frame_height * output_factor)),
    )
    actual = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
    source_to_original = np.diag([1.0 / source_factor, 1.0 / source_factor, 1.0])
    output_scale = np.diag([output_factor, output_factor, 1.0])
    matrix = (
        output_scale
        @ np.asarray(motion.matrix_3x3, dtype=np.float64)
        @ source_to_original
    )
    predicted = cv2.warpPerspective(
        source_small,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask_source = np.full(source_small.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        mask_source,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    predicted_gray = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)
    actual_gray = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY)
    valid = np.asarray(mask > 0, dtype=np.bool_)
    return (
        cast(NDArray[np.uint8], predicted_gray),
        cast(NDArray[np.uint8], actual_gray),
        valid,
    )


def _diagnostic_images(
    source: NDArray[np.uint8],
    frame: NDArray[np.uint8],
    motion: MotionRecoveryFrame,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    predicted, actual, valid = _aligned_small_frames(source, frame, motion)
    flow = cast(
        NDArray[np.float32],
        cast(Any, cv2.calcOpticalFlowFarneback)(
            predicted,
            actual,
            None,
            0.5,
            3,
            21,
            4,
            7,
            1.5,
            0,
        ),
    )
    magnitude = np.linalg.norm(flow, axis=2)
    scale = max(0.25, float(np.percentile(magnitude[valid], 98))) if np.any(valid) else 1.0
    heat = np.asarray(np.clip(magnitude / scale * 255.0, 0, 255), dtype=np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    overlay = cv2.resize(
        frame,
        (actual.shape[1], actual.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    source_height, source_width = source.shape[:2]
    points = np.asarray(
        [[0, 0], [source_width, 0], [source_width, source_height], [0, source_height]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(
        points,
        np.asarray(motion.matrix_3x3, dtype=np.float64),
    ).reshape(-1, 2)
    output_factor = actual.shape[1] / frame.shape[1]
    polygon = np.asarray(transformed * output_factor, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [polygon], True, (40, 245, 255), 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"frame {motion.frame} {motion.selected_model} conf={motion.confidence:.3f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cast(NDArray[np.uint8], overlay), cast(NDArray[np.uint8], heatmap)


def _evidence_frames(
    recovery: MotionRecoveryReport,
    residuals: list[dict[str, float]],
    *,
    source_width: int,
    source_height: int,
) -> list[ReferenceMotionFrameV2]:
    residual_by_frame = {round(item["frame"]): item for item in residuals}
    frames: list[ReferenceMotionFrameV2] = []
    for item in recovery.frames:
        if (
            item.center_x is None
            or item.center_y is None
            or item.scale_relative_to_contain is None
            or item.rotation_degrees is None
        ):
            raise ReferenceV2AnalysisError(f"frame {item.frame} lacks observable parameters")
        residual = residual_by_frame[item.frame]
        uncertainty = ["pivot_translation_gauge"]
        ordered_scores = sorted(item.model_scores.items(), key=lambda pair: pair[1])
        if len(ordered_scores) > 1 and ordered_scores[1][1] - ordered_scores[0][1] < 0.15:
            uncertainty.append("nested_global_models_close_without_material_gain")
        if item.confidence < 0.60:
            uncertainty.append("low_correspondence_confidence")
        if item.selected_model == "unknown":
            uncertainty.append("no_supported_global_model")
        if item.outlier:
            uncertainty.append("temporal_outlier_filtered")
        frames.append(
            ReferenceMotionFrameV2(
                frame=item.frame,
                time_seconds=item.time_seconds,
                selected_model=item.selected_model,
                matrix_3x3=item.matrix_3x3,
                visible_source_boundary=_visible_boundary(
                    item,
                    source_width,
                    source_height,
                    recovery.width,
                    recovery.height,
                ),
                center_x=item.center_x,
                center_y=item.center_y,
                scale_relative_to_contain=item.scale_relative_to_contain,
                rotation_degrees=item.rotation_degrees,
                confidence=item.confidence,
                confidence_interval={
                    "center_normalized_radius": max(
                        0.0002,
                        item.reprojection_rmse / max(recovery.width, recovery.height),
                    ),
                    "scale_relative_radius": max(0.0005, (1.0 - item.confidence) * 0.03),
                    "rotation_degrees_radius": max(
                        0.02,
                        (1.0 - item.confidence) * 1.2,
                    ),
                },
                match_count=item.match_count,
                inlier_count=item.inlier_count,
                inlier_ratio=item.inlier_ratio,
                reprojection_rmse=item.reprojection_rmse,
                residual_flow_mean=residual["flow_mean"],
                residual_flow_p95=residual["flow_p95"],
                residual_valid_fraction=residual["valid_fraction"],
                velocity=item.velocity,
                acceleration=item.acceleration,
                jerk=item.jerk,
                model_scores=item.model_scores,
                outlier=item.outlier,
                uncertainty=uncertainty,
            )
        )
    return frames


def _visible_boundary(
    frame: MotionRecoveryFrame,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> list[list[float]]:
    matrix = np.asarray(frame.matrix_3x3, dtype=np.float64)
    source_corners = np.asarray(
        [
            [0.0, 0.0],
            [source_width, 0.0],
            [source_width, source_height],
            [0.0, source_height],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(source_corners, matrix).reshape(-1, 2)
    return [[float(point[0] / width), float(point[1] / height)] for point in corners]


def _ambiguous_frames(frames: list[ReferenceMotionFrameV2]) -> list[int]:
    residual_values = np.asarray([item.residual_flow_p95 for item in frames])
    median = float(np.median(residual_values))
    mad = float(np.median(np.abs(residual_values - median)))
    residual_limit = median + max(0.15, 3.5 * 1.4826 * mad)
    return [
        item.frame
        for item in frames
        if item.confidence < 0.60
        or item.selected_model == "unknown"
        or item.outlier
        or item.residual_flow_p95 > residual_limit
    ]


def _phase_model(
    frames: list[ReferenceMotionFrameV2],
    recovery: MotionRecoveryReport,
) -> dict[str, Any]:
    coupling = characterize_similarity_coupling(
        center_x=[item.center_x for item in frames],
        center_y=[item.center_y for item in frames],
        scale=[item.scale_relative_to_contain for item in frames],
        rotation=[item.rotation_degrees for item in frames],
        fps=recovery.frame_rate.fps,
    )
    angular_acceleration = np.asarray(
        [abs(item.acceleration.get("rotation", 0.0)) for item in frames],
        dtype=np.float64,
    )
    angular_jerk = np.asarray(
        [abs(item.jerk.get("rotation", 0.0)) for item in frames],
        dtype=np.float64,
    )
    scale_velocity = np.asarray(
        [abs(item.velocity.get("scale", 0.0)) for item in frames],
        dtype=np.float64,
    )
    center_curvature = _center_curvature(frames)
    peak = coupling.maximum_phase_offset_frame
    perspective_fraction = sum(
        item.selected_model == "homography" for item in frames
    ) / len(frames)
    unknown_fraction = sum(item.selected_model == "unknown" for item in frames) / len(frames)
    if perspective_fraction >= 0.25:
        explanation = "perspective_component_with_coupled_rotation_timing"
    elif unknown_fraction >= 0.20:
        explanation = "structured_residual_or_local_warp"
    else:
        explanation = coupling.leading_explanation
    twist_score = coupling.confidence * min(
        1.0,
        abs(coupling.maximum_phase_offset) / 0.20
        + coupling.center_path_max_deviation / 0.02,
    )
    return {
        "onset_frame": recovery.phase_boundaries["onset"],
        "peak_velocity_frame": recovery.phase_boundaries["peak_velocity"],
        "settle_frame": recovery.phase_boundaries["settle"],
        "overshoot_detected": _overshoot_detected(frames),
        "channel_coupling": asdict(coupling),
        "twist_candidate": {
            "start_frame": coupling.start_frame,
            "peak_frame": peak,
            "end_frame": coupling.end_frame,
            "score": float(twist_score),
            "leading_explanation": explanation,
            "evidence": {
                "angular_acceleration": float(angular_acceleration[peak]),
                "angular_jerk": float(angular_jerk[peak]),
                "scale_velocity": float(scale_velocity[peak]),
                "center_path_curvature": float(center_curvature[peak]),
                "maximum_phase_offset": coupling.maximum_phase_offset,
                "peak_velocity_lag_frames": coupling.peak_velocity_lag_frames,
                "center_path_max_deviation": coupling.center_path_max_deviation,
                "perspective_frame_fraction": perspective_fraction,
                "unknown_model_frame_fraction": unknown_fraction,
            },
        },
    }


def _transform_order_assessment(recovery: MotionRecoveryReport) -> dict[str, Any]:
    return {
        "observable": "per-frame source-to-output matrix",
        "conclusion": "global_matrix_observable_but_authoring_order_not_unique",
        "supported_model": recovery.selected_model,
        "rotation_scale_commute_for_uniform_similarity": True,
        "pivot_translation_order_identifiable": False,
        "perspective_order_identifiable": (
            recovery.selected_model == "homography" and not recovery.model_mismatch
        ),
        "required_execution_rule": (
            "reproduce the frozen matrix curve; do not claim a unique authoring order "
            "where matrices are observationally equivalent"
        ),
    }


def _plots_and_contact_sheet(
    frames: list[ReferenceMotionFrameV2],
    diagnostic_paths: dict[int, dict[str, Path]],
    output_root: Path,
) -> dict[str, str]:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=False)
    position = _plot(
        frames,
        plot_root / "position-scale-rotation.png",
        "Reference v2: position, scale, rotation",
        (
            ("center_x", lambda item: item.center_x),
            ("center_y", lambda item: item.center_y),
            ("scale", lambda item: item.scale_relative_to_contain),
            ("rotation_deg", lambda item: item.rotation_degrees),
        ),
    )
    velocity = _derivative_plot(frames, plot_root / "velocity.png", "velocity")
    acceleration = _derivative_plot(
        frames,
        plot_root / "acceleration.png",
        "acceleration",
    )
    jerk = _derivative_plot(frames, plot_root / "jerk.png", "jerk")
    pivot = _pivot_plot(plot_root / "pivot-identifiability.png")
    sheet = _contact_sheet(diagnostic_paths, output_root / "phase-contact-sheet.jpg")
    return {
        "position_scale_rotation_plot": str(position),
        "velocity_plot": str(velocity),
        "acceleration_plot": str(acceleration),
        "jerk_plot": str(jerk),
        "pivot_identifiability_plot": str(pivot),
        "phase_contact_sheet": str(sheet),
        "diagnostic_overlays": str(output_root / "diagnostics"),
    }


def _plot(
    frames: list[ReferenceMotionFrameV2],
    path: Path,
    title: str,
    channels: tuple[tuple[str, Any], ...],
) -> Path:
    width = 1500
    panel_height = 210
    margin = 70
    image = Image.new("RGB", (width, panel_height * len(channels) + 80), "#10141c")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 18), title, fill="#f2f5f8", font=ImageFont.load_default())
    for panel, (label, getter) in enumerate(channels):
        top = 60 + panel * panel_height
        bottom = top + panel_height - 40
        left = margin
        right = width - 30
        values = np.asarray([float(getter(item)) for item in frames], dtype=np.float64)
        low = float(np.min(values))
        high = float(np.max(values))
        if math.isclose(low, high):
            high = low + 1.0
        draw.rectangle((left, top, right, bottom), outline="#394454")
        draw.text((left + 5, top + 5), f"{label} [{low:.6g}, {high:.6g}]", fill="#a9b8c8")
        points = [
            (
                left + (right - left) * index / max(1, len(values) - 1),
                bottom - (bottom - top) * (float(value) - low) / (high - low),
            )
            for index, value in enumerate(values)
        ]
        draw.line(points, fill="#63d4c6", width=2)
    image.save(path)
    return path


def _derivative_plot(
    frames: list[ReferenceMotionFrameV2],
    path: Path,
    order: str,
) -> Path:
    return _plot(
        frames,
        path,
        f"Reference v2: {order}",
        tuple(
            (
                f"{order}.{channel}",
                lambda item, channel=channel: getattr(item, order).get(channel, 0.0),
            )
            for channel in ("center_x", "center_y", "scale", "rotation")
        ),
    )


def _pivot_plot(path: Path) -> Path:
    image = Image.new("RGB", (1500, 340), "#10141c")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 55),
        "Pivot identifiability",
        fill="#f2f5f8",
        font=ImageFont.load_default(),
    )
    lines = (
        "No unique pivot curve is observable from one global matrix.",
        "Moving pivot + compensating translation produces the same pixels.",
        "Evidence preserves pivot=null and requires an authoring constraint or human correction.",
    )
    for index, line in enumerate(lines):
        draw.text((70, 120 + 50 * index), line, fill="#efb366")
    image.save(path)
    return path


def _contact_sheet(
    diagnostics: dict[int, dict[str, Path]],
    path: Path,
) -> Path:
    frames = sorted(diagnostics)
    if not frames:
        raise ReferenceV2AnalysisError("no diagnostic frames selected")
    columns = 4
    cell_width, cell_height = 300, 560
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#080a0e")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        image = Image.open(diagnostics[frame]["overlay"]).convert("RGB")
        image.thumbnail((cell_width - 16, cell_height - 42))
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + 28
        sheet.paste(image, (x, y))
        draw.text(
            ((index % columns) * cell_width + 8, (index // columns) * cell_height + 8),
            f"frame {frame}",
            fill="#ffffff",
        )
    sheet.save(path, quality=92)
    return path


def _diagnostic_frames(
    metrics: list[dict[str, float]],
    recovery: MotionRecoveryReport,
) -> set[int]:
    selected = {
        0,
        recovery.frame_count - 1,
        recovery.phase_boundaries["onset"],
        recovery.phase_boundaries["peak_velocity"],
        recovery.phase_boundaries["settle"],
        *recovery.outlier_frames,
    }
    selected.update(
        round(item["frame"])
        for item in sorted(metrics, key=lambda item: item["flow_p95"], reverse=True)[:8]
    )
    return {min(recovery.frame_count - 1, max(0, item)) for item in selected}


def _write_residual_summary(
    residuals: list[dict[str, float]],
    path: Path,
) -> Path:
    means = np.asarray([item["flow_mean"] for item in residuals], dtype=np.float64)
    percentiles = np.asarray([item["flow_p95"] for item in residuals], dtype=np.float64)
    payload = {
        "schema": "aoa_editing_reference_residual_summary_v2",
        "frame_count": len(residuals),
        "flow_mean_over_frames": float(np.mean(means)),
        "flow_p95_over_frames": float(np.percentile(percentiles, 95)),
        "maximum_frame_flow_p95": float(np.max(percentiles)),
        "frames": residuals,
    }
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def _comparison_criteria(
    *,
    source_hash: str,
    reference_hash: str,
    width: int,
    height: int,
    frame_rate: dict[str, Any],
    baseline: dict[str, Any] | None,
    motion_gate_sha256: str,
) -> dict[str, Any]:
    return {
        "threshold_revision": "reference-comparison-v2-frozen-1",
        "frozen_before_first_v2_render": True,
        "threshold_basis": {
            "ground_truth_motion_gate_sha256": motion_gate_sha256,
            "v1_baseline_observation": baseline,
            "human_evidence": (
                "v1 is geometrically close but perceived as too linear and missing "
                "the smoother twist/settle character"
            ),
        },
        "technical": {
            "resolution": [width, height],
            "frame_rate": frame_rate,
            "frame_count_tolerance": 0,
            "audio_structure_exact": True,
        },
        "geometric": {
            "all_frame_center_rmse_normalized_max": 0.004,
            "all_frame_scale_rmse_max": 0.010,
            "all_frame_rotation_rmse_degrees_max": 0.12,
            "visible_boundary_rmse_normalized_max": 0.006,
            "homography_corner_rmse_normalized_max": 0.006,
        },
        "temporal": {
            "position_velocity_rmse_max": 0.020,
            "scale_velocity_rmse_max": 0.040,
            "angular_velocity_rmse_degrees_max": 0.50,
            "position_acceleration_rmse_max": 0.10,
            "scale_acceleration_rmse_max": 0.20,
            "angular_acceleration_rmse_degrees_max": 2.0,
            "jerk_rmse_max": 12.0,
            "phase_onset_tolerance_frames": 2,
            "phase_settle_tolerance_frames": 2,
            "temporal_flow_rmse_max": 0.35,
        },
        "perceptual": {
            "vmaf_mean_min": 92.0,
            "ssim_all_min": 0.975,
            "psnr_diagnostic_only": True,
            "no_black_frames": True,
            "no_edge_exposure": True,
            "no_unwanted_blur": True,
        },
        "editorial": {
            "human_review_required": True,
            "rubric": [
                "smoothness",
                "weight",
                "natural_acceleration",
                "rotation_character",
                "twist_character",
                "settle_timing",
                "absence_of_mechanical_linearity",
                "compositional_match",
            ],
            "required_verdict": "perceptually_and_editorially_indistinguishable",
        },
        "lineage": {
            "only_visual_source_hash": source_hash,
            "reference_hash_forbidden": reference_hash,
        },
    }


def _baseline_observation(path: Path) -> dict[str, Any]:
    selected = path.expanduser().resolve(strict=True)
    payload = _object(selected)
    return {
        "path": str(selected),
        "sha256": sha256_file(selected),
        "motion": payload.get("motion"),
        "perceptual": payload.get("perceptual"),
        "visual_review": payload.get("visual_review"),
    }


def _center_curvature(frames: list[ReferenceMotionFrameV2]) -> NDArray[np.float64]:
    x = np.asarray([item.center_x for item in frames], dtype=np.float64)
    y = np.asarray([item.center_y for item in frames], dtype=np.float64)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    return np.divide(
        np.abs(dx * ddy - dy * ddx),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )


def _overshoot_detected(frames: list[ReferenceMotionFrameV2]) -> bool:
    getters: tuple[Callable[[ReferenceMotionFrameV2], float], ...] = (
        lambda item: item.center_x,
        lambda item: item.center_y,
        lambda item: item.scale_relative_to_contain,
        lambda item: item.rotation_degrees,
    )
    for getter in getters:
        values = np.asarray([getter(item) for item in frames], dtype=np.float64)
        low = min(float(values[0]), float(values[-1]))
        high = max(float(values[0]), float(values[-1]))
        tolerance = max(1e-6, (high - low) * 0.02)
        if float(np.min(values)) < low - tolerance or float(np.max(values)) > high + tolerance:
            return True
    return False


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReferenceV2GateError(f"receipt is not a JSON object: {path}")
    return payload


def _write_model(model: Any, path: Path) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(model.model_dump_json(indent=2))
        stream.write("\n")
