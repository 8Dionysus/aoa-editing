"""Evidence-bounded temporal correction for an otherwise passing v2 candidate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from aoa_editing.analysis.motion import (
    _curve_driver,
    _phase_boundaries,
    _preferred_derivative_window,
    characterize_similarity_coupling,
    local_polynomial_derivatives,
)
from aoa_editing.domain.models import (
    CheckResult,
    ComparisonCriteriaV2,
    FrameRange,
    MotionRecoveryReport,
    Provenance,
    ReferenceComparisonReportV2,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionSpecV2,
    TimeWarpAnchorV2,
)

ALGORITHM_REVISION = "phase-correction-v2-localized-search-1"
PHASE_FAILURE_IDS = {
    "temporal-phase-onset",
    "temporal-phase-duration",
    "temporal-phase-settle",
}
CHANNELS = (
    "center_x",
    "center_y",
    "scale_relative_to_contain",
    "rotation_degrees",
)


class PhaseCorrectionV2Error(RuntimeError):
    """The prior evidence cannot support a bounded temporal-only correction."""


@dataclass(frozen=True)
class _PartialWarp:
    anchors: tuple[tuple[int, float], ...]
    phase_error: int
    safety_penalty: int
    maximum_slope_deviation: float
    mean_absolute_warp: float


@dataclass(frozen=True)
class _PassingWarp:
    anchors: tuple[tuple[int, float], ...]
    phase: dict[str, int]
    metrics: dict[str, Any]
    checks: tuple[CheckResult, ...]
    safety_penalty: int
    maximum_slope_deviation: float
    regularization: float
    maximum_threshold_ratio: float


def derive_phase_correction_v2(
    spec: ReferenceReconstructionSpecV2,
    comparison: ReferenceComparisonReportV2,
    recovery: MotionRecoveryReport,
    *,
    spec_path: str,
    spec_sha256: str,
    comparison_path: str,
    comparison_sha256: str,
    candidate_motion_path: str,
    candidate_motion_sha256: str,
) -> ReferencePhaseCorrectionV2:
    """Derive a localized monotonic retime without admitting comparison pixels."""

    criteria = _validate_inputs(
        spec,
        comparison,
        recovery,
        spec_sha256=spec_sha256,
    )
    expected = _expected_channels(spec)
    candidate = _candidate_channels(recovery)
    expected_onset = int(spec.phase_model["onset_frame"])
    expected_settle = int(spec.phase_model["settle_frame"])
    actual_onset = int(recovery.phase_boundaries["onset"])
    actual_settle = int(recovery.phase_boundaries["settle"])
    protected_start, protected_end = _protected_coupling_bounds(spec)

    early = _search_early_warps(
        candidate,
        expected_onset=expected_onset,
        actual_onset=actual_onset,
        tolerance=criteria.temporal.phase_onset_tolerance_frames,
        protected_start=protected_start,
    )
    tail = _search_tail_warps(
        candidate,
        expected_settle=expected_settle,
        actual_settle=actual_settle,
        tolerance=criteria.temporal.phase_settle_tolerance_frames,
        protected_end=protected_end,
    )
    if not early or not tail:
        raise PhaseCorrectionV2Error(
            "no localized phase-boundary warp satisfies the frozen tolerance"
        )

    passing: list[_PassingWarp] = []
    evaluated_pairs = 0
    for early_warp in early[:10]:
        for tail_warp in tail[:10]:
            anchors = _join_partial_warps(early_warp, tail_warp)
            if anchors is None:
                continue
            evaluated_pairs += 1
            evaluated = _evaluate_full_warp(
                spec,
                criteria,
                expected,
                candidate,
                anchors,
                protected_start=protected_start,
                protected_end=protected_end,
            )
            if evaluated is not None:
                passing.append(evaluated)
    if not passing:
        raise PhaseCorrectionV2Error(
            "no localized phase correction passes the frozen predicted checks"
        )
    passing.sort(
        key=lambda item: (
            item.safety_penalty,
            item.maximum_slope_deviation,
            item.regularization,
            item.maximum_threshold_ratio,
            item.anchors,
        )
    )
    selected = passing[0]
    identity_start = selected.anchors[2][0]
    identity_end = selected.anchors[-3][0]
    failed_ids = [
        item.id for item in comparison.checks if item.status == "fail"
    ]
    return ReferencePhaseCorrectionV2(
        spec_id=spec.id,
        spec_sha256=spec_sha256,
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        comparison_report_id=comparison.id,
        comparison_report_path=comparison_path,
        comparison_report_sha256=comparison_sha256,
        candidate_motion_report_id=recovery.id,
        candidate_motion_path=candidate_motion_path,
        candidate_motion_sha256=candidate_motion_sha256,
        prior_candidate_sha256=comparison.candidate_sha256,
        duration_frames=spec.duration_frames,
        anchors=[
            TimeWarpAnchorV2(output_frame=output, source_frame=source)
            for output, source in selected.anchors
        ],
        identity_ranges=[
            FrameRange(
                start=identity_start,
                duration=identity_end - identity_start + 1,
            )
        ],
        protected_coupling_range=FrameRange(
            start=protected_start,
            duration=protected_end - protected_start + 1,
        ),
        previous_phase={
            "onset": actual_onset,
            "peak_velocity": int(recovery.phase_boundaries["peak_velocity"]),
            "settle": actual_settle,
        },
        predicted_phase=selected.phase,
        predicted_metrics=selected.metrics,
        predicted_checks=list(selected.checks),
        failed_check_ids=cast(Any, failed_ids),
        selection={
            "early_candidate_count": len(early),
            "tail_candidate_count": len(tail),
            "combined_candidate_count": evaluated_pairs,
            "passing_candidate_count": len(passing),
            "selected_maximum_slope_deviation": selected.maximum_slope_deviation,
            "selected_regularization": selected.regularization,
            "selected_maximum_threshold_ratio": selected.maximum_threshold_ratio,
            "protected_identity_start": identity_start,
            "protected_identity_end": identity_end,
            "selection_order": [
                "one-frame phase safety margin when available",
                "minimum maximum segment slope deviation",
                "minimum slope-energy plus localized-warp regularization",
                "minimum frozen-threshold utilization",
                "lexical anchor tie break",
            ],
        },
        provenance=Provenance(
            tool="aoa-editing-phase-correction-v2",
            tool_version="0.1.0",
            parameters={
                "algorithm_revision": ALGORITHM_REVISION,
                "spec_path": spec_path,
                "comparison_path": comparison_path,
                "candidate_motion_path": candidate_motion_path,
                "reference_media_used_as_render_input": False,
                "candidate_media_used_as_render_input": False,
                "protected_coupling_range": [protected_start, protected_end],
            },
            deterministic=True,
        ),
    )


def _validate_inputs(
    spec: ReferenceReconstructionSpecV2,
    comparison: ReferenceComparisonReportV2,
    recovery: MotionRecoveryReport,
    *,
    spec_sha256: str,
) -> ComparisonCriteriaV2:
    if comparison.spec_id != spec.id or comparison.spec_sha256 != spec_sha256:
        raise PhaseCorrectionV2Error("comparison is not bound to the selected frozen spec")
    if (
        comparison.source_sha256 != spec.source_sha256
        or comparison.reference_sha256 != spec.reference_sha256
    ):
        raise PhaseCorrectionV2Error("comparison hashes do not match the frozen spec")
    if (
        recovery.source_sha256 != spec.source_sha256
        or recovery.video_sha256 != comparison.candidate_sha256
    ):
        raise PhaseCorrectionV2Error("candidate recovery is not bound to the comparison")
    if (
        recovery.frame_count != spec.duration_frames
        or recovery.analyzed_frame_count != spec.duration_frames
        or len(recovery.frames) != spec.duration_frames
        or [item.frame for item in recovery.frames]
        != list(range(spec.duration_frames))
    ):
        raise PhaseCorrectionV2Error("candidate recovery is not an all-frame artifact")
    failed = {item.id for item in comparison.checks if item.status == "fail"}
    unexpected = failed - PHASE_FAILURE_IDS
    if unexpected:
        raise PhaseCorrectionV2Error(
            "non-phase objective failure forbids temporal-only correction: "
            + ", ".join(sorted(unexpected))
        )
    if not failed:
        raise PhaseCorrectionV2Error("comparison has no failed phase check to correct")
    if comparison.objective_overall != "fail":
        raise PhaseCorrectionV2Error("phase correction requires a failed objective report")
    try:
        criteria = ComparisonCriteriaV2.model_validate(spec.comparison_criteria)
    except ValueError as exc:
        raise PhaseCorrectionV2Error("frozen comparison criteria are invalid") from exc
    if criteria.threshold_revision != "reference-comparison-v2-frozen-1":
        raise PhaseCorrectionV2Error("unknown frozen comparison threshold revision")
    return criteria


def _expected_channels(
    spec: ReferenceReconstructionSpecV2,
) -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray([getattr(frame, name) for frame in spec.motion_frames])
        for name in CHANNELS
    }


def _candidate_channels(
    recovery: MotionRecoveryReport,
) -> dict[str, NDArray[np.float64]]:
    channels: dict[str, NDArray[np.float64]] = {}
    for name in CHANNELS:
        values = [getattr(frame, name) for frame in recovery.frames]
        if any(value is None for value in values):
            raise PhaseCorrectionV2Error(
                f"candidate recovery has missing observable channel: {name}"
            )
        channels[name] = np.asarray(values, dtype=np.float64)
    return channels


def _protected_coupling_bounds(
    spec: ReferenceReconstructionSpecV2,
) -> tuple[int, int]:
    coupling = cast(dict[str, Any], spec.phase_model.get("channel_coupling", {}))
    twist = cast(dict[str, Any], spec.phase_model.get("twist_candidate", {}))
    start = int(coupling.get("start_frame", twist.get("start_frame", 1)))
    end = int(
        coupling.get(
            "end_frame",
            twist.get("end_frame", spec.duration_frames - 2),
        )
    )
    start = min(spec.duration_frames - 2, max(1, start))
    end = min(spec.duration_frames - 2, max(start, end))
    return start, end


def _search_early_warps(
    candidate: dict[str, NDArray[np.float64]],
    *,
    expected_onset: int,
    actual_onset: int,
    tolerance: int,
    protected_start: int,
) -> list[_PartialWarp]:
    count = len(candidate["rotation_degrees"])
    last = count - 1
    target = min(protected_start - 2, max(1, expected_onset))
    span = max(3, abs(actual_onset - expected_onset) + tolerance + 2)
    minimum_source = max(1, min(target, actual_onset) - span)
    maximum_source = min(protected_start - 2, max(target, actual_onset) + span)
    results: list[_PartialWarp] = []
    for source in range(minimum_source, maximum_source + 1):
        if source == target:
            continue
        first_return = max(target, source) + 1
        for identity_return in range(first_return, protected_start + 1):
            anchors = (
                (0, 0.0),
                (target, float(source)),
                (identity_return, float(identity_return)),
                (last, float(last)),
            )
            if not _strictly_monotonic(anchors):
                continue
            mapped = _warp_channels(candidate, anchors)
            phase = _phase(mapped)
            error = abs(phase["onset"] - expected_onset)
            if error > tolerance:
                continue
            slopes = _slopes(anchors)
            results.append(
                _PartialWarp(
                    anchors=anchors[:3],
                    phase_error=error,
                    safety_penalty=_safety_penalty(error, tolerance),
                    maximum_slope_deviation=max(abs(value - 1.0) for value in slopes),
                    mean_absolute_warp=_mean_absolute_warp(anchors, count),
                )
            )
    results.sort(
        key=lambda item: (
            item.safety_penalty,
            item.maximum_slope_deviation,
            _partial_regularization(item),
            item.phase_error,
            item.anchors,
        )
    )
    return results


def _search_tail_warps(
    candidate: dict[str, NDArray[np.float64]],
    *,
    expected_settle: int,
    actual_settle: int,
    tolerance: int,
    protected_end: int,
) -> list[_PartialWarp]:
    count = len(candidate["rotation_degrees"])
    last = count - 1
    span = max(8, abs(actual_settle - expected_settle) + tolerance + 3)
    output_minimum = max(protected_end + 2, expected_settle - span)
    output_maximum = min(last - 1, expected_settle + tolerance)
    source_minimum = max(protected_end + 1, actual_settle - span)
    source_maximum = min(last - 1, actual_settle + span)
    results: list[_PartialWarp] = []
    for output in range(output_minimum, output_maximum + 1):
        for source in range(source_minimum, source_maximum + 1):
            if source == output:
                continue
            start_minimum = max(protected_end, output - 32)
            for identity_start in range(start_minimum, output):
                anchors = (
                    (0, 0.0),
                    (identity_start, float(identity_start)),
                    (output, float(source)),
                    (last, float(last)),
                )
                if not _strictly_monotonic(anchors):
                    continue
                mapped = _warp_channels(candidate, anchors)
                phase = _phase(mapped)
                error = abs(phase["settle"] - expected_settle)
                if error > tolerance:
                    continue
                slopes = _slopes(anchors)
                results.append(
                    _PartialWarp(
                        anchors=anchors[1:],
                        phase_error=error,
                        safety_penalty=_safety_penalty(error, tolerance),
                        maximum_slope_deviation=max(
                            abs(value - 1.0) for value in slopes
                        ),
                        mean_absolute_warp=_mean_absolute_warp(anchors, count),
                    )
                )
    results.sort(
        key=lambda item: (
            item.safety_penalty,
            item.maximum_slope_deviation,
            _partial_regularization(item),
            item.phase_error,
            item.anchors,
        )
    )
    return results


def _join_partial_warps(
    early: _PartialWarp,
    tail: _PartialWarp,
) -> tuple[tuple[int, float], ...] | None:
    anchors = (*early.anchors, *tail.anchors)
    if not _strictly_monotonic(anchors):
        return None
    if early.anchors[-1][0] > tail.anchors[0][0]:
        return None
    return anchors


def _evaluate_full_warp(
    spec: ReferenceReconstructionSpecV2,
    criteria: ComparisonCriteriaV2,
    expected: dict[str, NDArray[np.float64]],
    candidate: dict[str, NDArray[np.float64]],
    anchors: tuple[tuple[int, float], ...],
    *,
    protected_start: int,
    protected_end: int,
) -> _PassingWarp | None:
    mapped = _warp_channels(candidate, anchors)
    phase = _phase(mapped)
    expected_onset = int(spec.phase_model["onset_frame"])
    expected_settle = int(spec.phase_model["settle_frame"])
    onset_error = abs(phase["onset"] - expected_onset)
    settle_error = abs(phase["settle"] - expected_settle)
    duration_error = abs(
        (phase["settle"] - phase["onset"])
        - (expected_settle - expected_onset)
    )
    geometry = {
        "center_rmse_normalized": _rmse(
            np.hypot(
                mapped["center_x"] - expected["center_x"],
                mapped["center_y"] - expected["center_y"],
            )
        ),
        "scale_rmse": _rmse(
            mapped["scale_relative_to_contain"]
            - expected["scale_relative_to_contain"]
        ),
        "rotation_rmse_degrees": _rmse(
            mapped["rotation_degrees"] - expected["rotation_degrees"]
        ),
    }
    derivatives = _derivative_metrics(spec, mapped)
    expected_coupling = cast(
        dict[str, Any],
        spec.phase_model.get("channel_coupling", {}),
    )
    actual_coupling = asdict(
        characterize_similarity_coupling(
            center_x=mapped["center_x"],
            center_y=mapped["center_y"],
            scale=mapped["scale_relative_to_contain"],
            rotation=mapped["rotation_degrees"],
            fps=spec.frame_rate.fps,
        )
    )
    required_coupling_keys = {
        "leading_explanation",
        "center_phase_relation",
        "scale_rotation_progress_rmse",
        "maximum_phase_offset",
        "peak_velocity_lag_frames",
        "center_path_max_deviation",
    }
    if not required_coupling_keys.issubset(expected_coupling):
        raise PhaseCorrectionV2Error("frozen coupling evidence is incomplete")
    coupling_errors = {
        "scale_rotation_progress_rmse": abs(
            float(actual_coupling["scale_rotation_progress_rmse"])
            - float(expected_coupling["scale_rotation_progress_rmse"])
        ),
        "maximum_phase_offset": abs(
            float(actual_coupling["maximum_phase_offset"])
            - float(expected_coupling["maximum_phase_offset"])
        ),
        "peak_velocity_lag_frames": abs(
            int(actual_coupling["peak_velocity_lag_frames"])
            - int(expected_coupling["peak_velocity_lag_frames"])
        ),
        "center_path_max_deviation": abs(
            float(actual_coupling["center_path_max_deviation"])
            - float(expected_coupling["center_path_max_deviation"])
        ),
    }
    geometric = criteria.geometric
    temporal = criteria.temporal
    coupling_pass = (
        actual_coupling["leading_explanation"]
        == expected_coupling["leading_explanation"]
        and actual_coupling["center_phase_relation"]
        == expected_coupling["center_phase_relation"]
        and coupling_errors["scale_rotation_progress_rmse"]
        <= geometric.all_frame_scale_rmse_max
        and coupling_errors["maximum_phase_offset"]
        <= temporal.position_velocity_rmse_max
        and coupling_errors["peak_velocity_lag_frames"]
        <= temporal.phase_onset_tolerance_frames
        and coupling_errors["center_path_max_deviation"]
        <= geometric.visible_boundary_rmse_normalized_max
    )
    identity_pass = all(
        math.isclose(_map_frame(anchors, frame), float(frame), abs_tol=1e-9)
        for frame in range(protected_start, protected_end + 1)
    )
    checks = (
        _check(
            "predicted-geometric-center",
            geometry["center_rmse_normalized"]
            <= geometric.all_frame_center_rmse_normalized_max,
            geometry["center_rmse_normalized"],
            geometric.all_frame_center_rmse_normalized_max,
        ),
        _check(
            "predicted-geometric-scale",
            geometry["scale_rmse"] <= geometric.all_frame_scale_rmse_max,
            geometry["scale_rmse"],
            geometric.all_frame_scale_rmse_max,
        ),
        _check(
            "predicted-geometric-rotation",
            geometry["rotation_rmse_degrees"]
            <= geometric.all_frame_rotation_rmse_degrees_max,
            geometry["rotation_rmse_degrees"],
            geometric.all_frame_rotation_rmse_degrees_max,
        ),
        _check(
            "predicted-temporal-velocity-position",
            derivatives["velocity_position"]
            <= temporal.position_velocity_rmse_max,
            derivatives["velocity_position"],
            temporal.position_velocity_rmse_max,
        ),
        _check(
            "predicted-temporal-velocity-scale",
            derivatives["velocity_scale"] <= temporal.scale_velocity_rmse_max,
            derivatives["velocity_scale"],
            temporal.scale_velocity_rmse_max,
        ),
        _check(
            "predicted-temporal-velocity-rotation",
            derivatives["velocity_rotation"]
            <= temporal.angular_velocity_rmse_degrees_max,
            derivatives["velocity_rotation"],
            temporal.angular_velocity_rmse_degrees_max,
        ),
        _check(
            "predicted-temporal-acceleration-position",
            derivatives["acceleration_position"]
            <= temporal.position_acceleration_rmse_max,
            derivatives["acceleration_position"],
            temporal.position_acceleration_rmse_max,
        ),
        _check(
            "predicted-temporal-acceleration-scale",
            derivatives["acceleration_scale"]
            <= temporal.scale_acceleration_rmse_max,
            derivatives["acceleration_scale"],
            temporal.scale_acceleration_rmse_max,
        ),
        _check(
            "predicted-temporal-acceleration-rotation",
            derivatives["acceleration_rotation"]
            <= temporal.angular_acceleration_rmse_degrees_max,
            derivatives["acceleration_rotation"],
            temporal.angular_acceleration_rmse_degrees_max,
        ),
        _check(
            "predicted-temporal-jerk",
            derivatives["combined_jerk"] <= temporal.jerk_rmse_max,
            derivatives["combined_jerk"],
            temporal.jerk_rmse_max,
        ),
        _check(
            "predicted-temporal-phase-onset",
            onset_error <= temporal.phase_onset_tolerance_frames,
            float(onset_error),
            float(temporal.phase_onset_tolerance_frames),
        ),
        _check(
            "predicted-temporal-phase-duration",
            duration_error
            <= (
                temporal.phase_onset_tolerance_frames
                + temporal.phase_settle_tolerance_frames
            ),
            float(duration_error),
            float(
                temporal.phase_onset_tolerance_frames
                + temporal.phase_settle_tolerance_frames
            ),
        ),
        _check(
            "predicted-temporal-phase-settle",
            settle_error <= temporal.phase_settle_tolerance_frames,
            float(settle_error),
            float(temporal.phase_settle_tolerance_frames),
        ),
        CheckResult(
            id="predicted-temporal-coupling",
            status="pass" if coupling_pass else "fail",
            summary="predicted coupling remains within frozen comparison bounds",
            measured={
                "actual": actual_coupling,
                "expected": expected_coupling,
                "errors": coupling_errors,
            },
        ),
        CheckResult(
            id="predicted-protected-identity-range",
            status="pass" if identity_pass else "fail",
            summary="central coupling range remains identity-mapped",
            measured={"start": protected_start, "end": protected_end},
        ),
    )
    if any(item.status != "pass" for item in checks):
        return None
    ratios = (
        geometry["center_rmse_normalized"]
        / geometric.all_frame_center_rmse_normalized_max,
        geometry["scale_rmse"] / geometric.all_frame_scale_rmse_max,
        geometry["rotation_rmse_degrees"]
        / geometric.all_frame_rotation_rmse_degrees_max,
        derivatives["velocity_position"] / temporal.position_velocity_rmse_max,
        derivatives["velocity_scale"] / temporal.scale_velocity_rmse_max,
        derivatives["velocity_rotation"]
        / temporal.angular_velocity_rmse_degrees_max,
        derivatives["acceleration_position"]
        / temporal.position_acceleration_rmse_max,
        derivatives["acceleration_scale"]
        / temporal.scale_acceleration_rmse_max,
        derivatives["acceleration_rotation"]
        / temporal.angular_acceleration_rmse_degrees_max,
        derivatives["combined_jerk"] / temporal.jerk_rmse_max,
    )
    slopes = _slopes(anchors)
    mean_warp = _mean_absolute_warp(anchors, spec.duration_frames)
    slope_energy = sum((value - 1.0) ** 2 for value in slopes)
    safety_penalty = _safety_penalty(
        onset_error,
        temporal.phase_onset_tolerance_frames,
    ) + _safety_penalty(
        settle_error,
        temporal.phase_settle_tolerance_frames,
    )
    return _PassingWarp(
        anchors=anchors,
        phase=phase,
        metrics={
            "geometric": geometry,
            "temporal": {
                "derivative_rmse": derivatives,
                "phase": {
                    "expected_onset": expected_onset,
                    "actual_onset": phase["onset"],
                    "onset_error_frames": onset_error,
                    "expected_settle": expected_settle,
                    "actual_settle": phase["settle"],
                    "settle_error_frames": settle_error,
                    "duration_error_frames": duration_error,
                },
            },
            "coupling": {
                "expected": expected_coupling,
                "actual": actual_coupling,
                "errors": coupling_errors,
            },
            "time_warp": {
                "slopes": slopes,
                "mean_absolute_warp_frames": mean_warp,
                "maximum_absolute_warp_frames": max(
                    abs(_map_frame(anchors, frame) - frame)
                    for frame in range(spec.duration_frames)
                ),
            },
        },
        checks=checks,
        safety_penalty=safety_penalty,
        maximum_slope_deviation=max(abs(value - 1.0) for value in slopes),
        regularization=slope_energy + 2.0 * mean_warp,
        maximum_threshold_ratio=max(ratios),
    )


def _derivative_metrics(
    spec: ReferenceReconstructionSpecV2,
    mapped: dict[str, NDArray[np.float64]],
) -> dict[str, float]:
    errors: dict[str, NDArray[np.float64]] = {}
    window = _preferred_derivative_window(spec.duration_frames)
    for channel in CHANNELS:
        key = {
            "scale_relative_to_contain": "scale",
            "rotation_degrees": "rotation",
        }.get(channel, channel)
        actual = local_polynomial_derivatives(
            mapped[channel],
            fps=spec.frame_rate.fps,
            window=window,
            degree=3,
        )
        for order in ("velocity", "acceleration", "jerk"):
            expected = np.asarray(
                [frame.__getattribute__(order)[key] for frame in spec.motion_frames],
                dtype=np.float64,
            )
            errors[f"{order}_{channel}"] = expected - np.asarray(
                getattr(actual, order),
                dtype=np.float64,
            )
    metrics = {
        "velocity_position": _rmse(
            np.hypot(errors["velocity_center_x"], errors["velocity_center_y"])
        ),
        "velocity_scale": _rmse(errors["velocity_scale_relative_to_contain"]),
        "velocity_rotation": _rmse(errors["velocity_rotation_degrees"]),
        "acceleration_position": _rmse(
            np.hypot(
                errors["acceleration_center_x"],
                errors["acceleration_center_y"],
            )
        ),
        "acceleration_scale": _rmse(
            errors["acceleration_scale_relative_to_contain"]
        ),
        "acceleration_rotation": _rmse(errors["acceleration_rotation_degrees"]),
        "jerk_position": _rmse(
            np.hypot(errors["jerk_center_x"], errors["jerk_center_y"])
        ),
        "jerk_scale": _rmse(errors["jerk_scale_relative_to_contain"]),
        "jerk_rotation": _rmse(errors["jerk_rotation_degrees"]),
    }
    metrics["combined_jerk"] = max(
        metrics["jerk_position"],
        metrics["jerk_scale"],
        metrics["jerk_rotation"],
    )
    return metrics


def _phase(channels: dict[str, NDArray[np.float64]]) -> dict[str, int]:
    driver = _curve_driver(
        {
            "center_x": channels["center_x"],
            "center_y": channels["center_y"],
            "scale": channels["scale_relative_to_contain"],
            "rotation": channels["rotation_degrees"],
        }
    )
    return _phase_boundaries(driver.tolist())


def _warp_channels(
    channels: dict[str, NDArray[np.float64]],
    anchors: tuple[tuple[int, float], ...],
) -> dict[str, NDArray[np.float64]]:
    count = len(channels["rotation_degrees"])
    output = np.arange(count, dtype=np.float64)
    source = np.interp(
        output,
        [item[0] for item in anchors],
        [item[1] for item in anchors],
    )
    original = np.arange(count, dtype=np.float64)
    return {
        name: np.interp(source, original, values)
        for name, values in channels.items()
    }


def _map_frame(
    anchors: tuple[tuple[int, float], ...],
    frame: int,
) -> float:
    for left, right in pairwise(anchors):
        if left[0] <= frame <= right[0]:
            progress = (frame - left[0]) / (right[0] - left[0])
            return left[1] + progress * (right[1] - left[1])
    raise AssertionError("validated time warp has no segment")


def _strictly_monotonic(
    anchors: tuple[tuple[int, float], ...],
) -> bool:
    return all(
        right[0] > left[0] and right[1] > left[1]
        for left, right in pairwise(anchors)
    )


def _slopes(
    anchors: tuple[tuple[int, float], ...],
) -> list[float]:
    return [
        (right[1] - left[1]) / (right[0] - left[0])
        for left, right in pairwise(anchors)
    ]


def _mean_absolute_warp(
    anchors: tuple[tuple[int, float], ...],
    count: int,
) -> float:
    return float(
        np.mean(
            [
                abs(_map_frame(anchors, frame) - frame)
                for frame in range(count)
            ]
        )
    )


def _partial_regularization(
    warp: _PartialWarp,
) -> float:
    slopes = _slopes(warp.anchors)
    return sum((value - 1.0) ** 2 for value in slopes) + (
        2.0 * warp.mean_absolute_warp
    )


def _safety_penalty(error: int, tolerance: int) -> int:
    safe_limit = max(0, tolerance - 1)
    return 0 if error <= safe_limit else 1


def _check(
    identifier: str,
    passed: bool,
    actual: float,
    maximum: float,
) -> CheckResult:
    return CheckResult(
        id=identifier,
        status="pass" if passed else "fail",
        summary="predicted value is within the frozen maximum",
        measured={"actual": actual, "maximum": maximum},
    )


def _rmse(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))
