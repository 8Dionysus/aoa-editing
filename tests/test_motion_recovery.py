from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoa_editing.analysis.motion import (
    MotionAnalyzer,
    characterize_similarity_coupling,
    classify_curve,
    finite_derivatives,
    local_polynomial_derivatives,
    reject_temporal_outliers,
    unwrap_degrees,
)
from aoa_editing.evals.motion_fixtures import (
    fixture_definitions,
    render_motion_fixture,
)
from aoa_editing.evals.motion_recovery import (
    compare_recovery_to_truth,
    run_motion_recovery_gate,
)


def test_motion_fixture_catalog_covers_the_required_independent_corpus() -> None:
    definitions = fixture_definitions()
    tags = {tag for definition in definitions for tag in definition.requirement_tags}

    assert {
        "linear",
        "ease_in",
        "ease_out",
        "ease_in_out",
        "cubic_bezier",
        "hermite",
        "overshoot",
        "settle",
        "scale_rotation",
        "curved_translation",
        "moving_pivot",
        "coupled_scale_rotation",
        "affine",
        "homography",
        "local_warp",
        "compression",
        "blur",
        "texture_loss",
        "partial_offscreen",
        "different_aspect_ratios",
        "alpha_source",
        "low_resolution",
    } <= tags
    assert len({definition.id for definition in definitions}) == len(definitions)
    assert all(definition.content_origin == "synthetic_ground_truth" for definition in definitions)


@pytest.mark.integration
def test_generate_render_recover_similarity_curve(tmp_path: Path) -> None:
    definition = next(item for item in fixture_definitions() if item.id == "ease-in-out")
    fixture = render_motion_fixture(definition, tmp_path / definition.id)

    report = MotionAnalyzer().analyze_video(fixture.source_path, fixture.video_path)
    comparison = compare_recovery_to_truth(fixture.truth, report)

    assert report.frame_count == definition.frame_count
    assert len(report.frames) == definition.frame_count
    assert report.selected_model == "similarity"
    assert report.curve_hint == "ease_in_out"
    assert report.model_mismatch is False
    assert report.mean_confidence >= 0.70
    assert comparison["center_rmse_normalized"] <= 0.012
    assert comparison["scale_rmse"] <= 0.025
    assert comparison["rotation_rmse_degrees"] <= 0.35
    assert comparison["corner_rmse_normalized"] <= 0.012


@pytest.mark.integration
@pytest.mark.parametrize(
    ("fixture_id", "expected_model"),
    [("affine-shear", "affine"), ("perspective", "homography")],
)
def test_model_selection_distinguishes_transform_families(
    fixture_id: str,
    expected_model: str,
    tmp_path: Path,
) -> None:
    definition = next(item for item in fixture_definitions() if item.id == fixture_id)
    fixture = render_motion_fixture(definition, tmp_path / fixture_id)

    report = MotionAnalyzer().analyze_video(fixture.source_path, fixture.video_path)
    comparison = compare_recovery_to_truth(fixture.truth, report)

    assert report.selected_model == expected_model
    assert report.model_mismatch is False
    assert comparison["corner_rmse_normalized"] <= 0.018


def test_random_similarity_property_recovery() -> None:
    rng = np.random.default_rng(20260719)
    source = _feature_source()
    analyzer = MotionAnalyzer()
    for _ in range(12):
        width, height = 360, 480
        zoom = float(rng.uniform(0.85, 1.75))
        rotation = float(rng.uniform(-18.0, 18.0))
        center = (
            float(rng.uniform(width * 0.38, width * 0.62)),
            float(rng.uniform(height * 0.38, height * 0.62)),
        )
        contain = min(width / source.shape[1], height / source.shape[0])
        matrix = cv2.getRotationMatrix2D(
            (source.shape[1] / 2, source.shape[0] / 2),
            -rotation,
            contain * zoom,
        )
        transformed_center = matrix[:, :2] @ np.array(
            [source.shape[1] / 2, source.shape[0] / 2]
        ) + matrix[:, 2]
        matrix[:, 2] += np.asarray(center) - transformed_center
        frame = cv2.warpAffine(
            source,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        recovered = analyzer.analyze_frame(source, frame)

        assert recovered.selected_model == "similarity"
        assert recovered.scale_relative_to_contain == pytest.approx(zoom, abs=0.025)
        assert recovered.rotation_degrees == pytest.approx(rotation, abs=0.35)
        assert recovered.center_x == pytest.approx(center[0] / width, abs=0.012)
        assert recovered.center_y == pytest.approx(center[1] / height, abs=0.012)
        assert recovered.confidence >= 0.65


def test_random_admissible_cubic_curves_recover_without_endpoint_or_continuity_drift() -> None:
    rng = np.random.default_rng(8675309)
    source = _feature_source()
    analyzer = MotionAnalyzer()
    width, height = 360, 480
    contain = min(width / source.shape[1], height / source.shape[0])
    time = np.linspace(0.0, 1.0, 9)
    for _ in range(5):
        first, second = sorted(rng.uniform(0.0, 1.0, size=2))
        inverse = 1.0 - time
        progress = (
            3.0 * inverse**2 * time * first
            + 3.0 * inverse * time**2 * second
            + time**3
        )
        recovered_scales: list[float] = []
        for frame_index, value in enumerate(progress):
            zoom = 1.05 + 0.55 * float(value)
            rotation = -8.0 + 14.0 * float(value)
            matrix = cv2.getRotationMatrix2D(
                (source.shape[1] / 2, source.shape[0] / 2),
                -rotation,
                contain * zoom,
            )
            frame = cv2.warpAffine(source, matrix, (width, height), borderValue=0)
            recovered = analyzer.analyze_frame(
                source,
                frame,
                frame_index=frame_index,
                fps=8.0,
            )
            assert recovered.scale_relative_to_contain is not None
            recovered_scales.append(recovered.scale_relative_to_contain)
            assert recovered.scale_relative_to_contain == pytest.approx(zoom, abs=0.025)
        derivatives = finite_derivatives(recovered_scales, fps=8.0)
        assert recovered_scales[0] == pytest.approx(1.05, abs=0.025)
        assert recovered_scales[-1] == pytest.approx(1.60, abs=0.025)
        assert all(math.isfinite(item) for item in derivatives.jerk)
        assert min(np.diff(recovered_scales)) >= -0.01


def test_recovery_is_stable_under_encoding_noise() -> None:
    source = _feature_source()
    width, height = 360, 480
    contain = min(width / source.shape[1], height / source.shape[0])
    matrix = cv2.getRotationMatrix2D(
        (source.shape[1] / 2, source.shape[0] / 2),
        7.5,
        contain * 1.35,
    )
    frame = cv2.warpAffine(source, matrix, (width, height), borderValue=0)
    available, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 28])
    assert available
    noisy = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    recovered = MotionAnalyzer().analyze_frame(source, noisy)

    assert recovered.selected_model == "similarity"
    assert recovered.scale_relative_to_contain == pytest.approx(1.35, abs=0.04)
    assert recovered.rotation_degrees == pytest.approx(-7.5, abs=0.55)
    assert recovered.confidence >= 0.50


def test_bounded_feature_sampling_preserves_full_resolution_coordinates() -> None:
    source = cv2.resize(
        _feature_source(),
        (2048, 1536),
        interpolation=cv2.INTER_CUBIC,
    )
    width, height = 1080, 1920
    contain = min(width / source.shape[1], height / source.shape[0])
    zoom = 1.42
    rotation = -6.25
    center = np.asarray([width * 0.46, height * 0.57])
    matrix = cv2.getRotationMatrix2D(
        (source.shape[1] / 2, source.shape[0] / 2),
        -rotation,
        contain * zoom,
    )
    mapped_center = matrix[:, :2] @ np.asarray(
        [source.shape[1] / 2, source.shape[0] / 2]
    ) + matrix[:, 2]
    matrix[:, 2] += center - mapped_center
    frame = cv2.warpAffine(source, matrix, (width, height), borderValue=0)

    recovered = MotionAnalyzer(
        maximum_source_side=640,
        maximum_frame_side=480,
    ).analyze_frame(source, frame)

    assert recovered.scale_relative_to_contain == pytest.approx(zoom, abs=0.035)
    assert recovered.rotation_degrees == pytest.approx(rotation, abs=0.45)
    assert recovered.center_x == pytest.approx(center[0] / width, abs=0.012)
    assert recovered.center_y == pytest.approx(center[1] / height, abs=0.012)


def test_angle_unwrap_derivatives_and_curve_classification_are_continuous() -> None:
    angles = [170.0, 178.0, -176.0, -169.0]
    assert unwrap_degrees(angles) == pytest.approx([170.0, 178.0, 184.0, 191.0])

    values = [0.0, 0.04, 0.16, 0.36, 0.64, 1.0]
    derivatives = finite_derivatives(values, fps=5.0)
    assert len(derivatives.velocity) == len(values)
    assert len(derivatives.acceleration) == len(values)
    assert len(derivatives.jerk) == len(values)
    assert max(abs(value) for value in derivatives.jerk) < 125
    assert classify_curve(values).name == "ease_in"
    assert classify_curve(values).confidence >= 0.70


def test_temporal_outlier_rejection_localizes_spike_without_flattening_curve() -> None:
    clean = np.asarray([0.0, 0.03, 0.12, 0.27, 0.48, 0.75, 0.91, 1.0])
    contaminated = clean.copy()
    contaminated[4] = 5.0

    result = reject_temporal_outliers(contaminated)

    assert result.outlier_indices == (4,)
    assert result.values[4] == pytest.approx((clean[3] + clean[5]) / 2, abs=0.08)
    retained = [0, 1, 2, 3, 5, 6, 7]
    assert np.max(np.abs(result.values[retained] - clean[retained])) < 1e-9


def test_local_polynomial_derivatives_suppress_subpixel_noise_without_phase_drift() -> None:
    frame_count = 333
    fps = 25.0
    duration = (frame_count - 1) / fps
    progress = np.linspace(0.0, 1.0, frame_count)
    clean = progress * progress * (3.0 - 2.0 * progress)
    noisy = clean + 0.0015 * np.sin(np.arange(frame_count) * 2.37)
    expected_velocity = (6.0 * progress - 6.0 * progress * progress) / duration

    raw = finite_derivatives(noisy, fps=fps)
    robust = local_polynomial_derivatives(noisy, fps=fps, window=21, degree=3)
    raw_error = float(
        np.sqrt(np.mean(np.square(np.asarray(raw.velocity) - expected_velocity)))
    )
    robust_error = float(
        np.sqrt(np.mean(np.square(np.asarray(robust.velocity) - expected_velocity)))
    )

    assert len(robust.velocity) == frame_count
    assert len(robust.acceleration) == frame_count
    assert len(robust.jerk) == frame_count
    assert robust_error < raw_error * 0.25
    assert int(np.argmax(robust.velocity)) == pytest.approx((frame_count - 1) / 2, abs=2)


def test_similarity_coupling_localizes_staggered_curved_center_motion() -> None:
    frame_count = 333
    time = np.linspace(0.0, 1.0, frame_count)
    shared = time * time * (3.0 - 2.0 * time)
    center_progress = shared**1.55
    center_x = 0.286 + 0.214 * center_progress
    center_y = 0.628 - 0.128 * (
        center_progress + 0.05 * np.sin(np.pi * center_progress)
    )

    coupling = characterize_similarity_coupling(
        center_x=center_x,
        center_y=center_y,
        scale=2.8 - 1.8 * shared,
        rotation=-7.2 + 7.2 * shared,
        fps=25.0,
    )

    assert (
        coupling.leading_explanation
        == "staggered_center_path_with_synchronized_scale_rotation"
    )
    assert coupling.center_phase_relation == "center_lags"
    assert coupling.scale_rotation_progress_rmse < 1e-9
    assert coupling.maximum_phase_offset > 0.10
    assert coupling.peak_velocity_lag_frames > 20
    assert coupling.start_frame < coupling.maximum_phase_offset_frame < coupling.end_frame
    assert coupling.center_path_max_deviation > 0.001
    assert coupling.confidence >= 0.90


@pytest.mark.integration
def test_unknown_local_warp_and_pivot_ambiguity_preserve_uncertainty(
    tmp_path: Path,
) -> None:
    definitions = {item.id: item for item in fixture_definitions()}
    warped = render_motion_fixture(definitions["local-warp"], tmp_path / "local-warp")
    warp_report = MotionAnalyzer().analyze_video(warped.source_path, warped.video_path)
    pivoted = render_motion_fixture(definitions["moving-pivot"], tmp_path / "moving-pivot")
    pivot_report = MotionAnalyzer().analyze_video(pivoted.source_path, pivoted.video_path)

    assert warp_report.model_mismatch is True
    assert warp_report.selected_model == "unknown"
    assert any(
        "local" in item.lower() or "global" in item.lower()
        for item in warp_report.uncertainties
    )
    assert pivot_report.selected_model == "similarity"
    assert all(frame.pivot_identifiability == "unidentifiable" for frame in pivot_report.frames)
    assert any("pivot" in item.lower() for item in pivot_report.uncertainties)


@pytest.mark.integration
def test_representative_motion_recovery_gate_is_fail_closed_and_green(
    tmp_path: Path,
) -> None:
    report = run_motion_recovery_gate(
        tmp_path / "gate",
        fixture_ids={"linear", "local-warp"},
    )

    assert report.overall == "pass"
    assert report.mandatory_skips == 0
    assert len(report.git_revision) == 40
    assert all(item.status == "pass" for item in report.checks)
    assert report.metrics["linear"]["mean_confidence"] >= 0.62
    assert report.metrics["local-warp"]["model_mismatch"] is True


def _feature_source() -> np.ndarray:
    rng = np.random.default_rng(4815162342)
    source = rng.integers(0, 256, size=(384, 512, 3), dtype=np.uint8)
    cv2.putText(
        source,
        "GROUND TRUTH",
        (30, 195),
        cv2.FONT_HERSHEY_DUPLEX,
        1.25,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    for index in range(18):
        angle = 2 * math.pi * index / 18
        center = (
            round(256 + 150 * math.cos(angle)),
            round(192 + 115 * math.sin(angle)),
        )
        cv2.circle(source, center, 7 + index % 5, (20, 20, 20), 2, cv2.LINE_AA)
    return source
