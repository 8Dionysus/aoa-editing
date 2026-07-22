from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoa_editing.analysis.reference import estimate_affine_frame, measure_video_motion
from aoa_editing.evals.reference import analyze_reference


def test_affine_measurement_recovers_general_source_transform() -> None:
    rng = np.random.default_rng(42)
    source = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
    cv2.circle(source, (340, 175), 55, 255, 8)
    cv2.rectangle(source, (70, 300), (220, 430), 0, 10)
    output_width, output_height = 270, 480
    contain_scale = output_width / source.shape[1]
    expected_zoom = 1.8
    expected_scale = contain_scale * expected_zoom
    expected_rotation = -4.0
    expected_center = (112.0, 286.0)
    angle = math.radians(expected_rotation)
    matrix = np.array(
        [
            [
                expected_scale * math.cos(angle),
                -expected_scale * math.sin(angle),
                0.0,
            ],
            [
                expected_scale * math.sin(angle),
                expected_scale * math.cos(angle),
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    source_center = np.array([256.0, 256.0])
    matrix[:, 2] = np.array(expected_center) - matrix[:, :2] @ source_center
    frame = cv2.warpAffine(
        source,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    measured = estimate_affine_frame(source, frame)

    assert measured.inlier_ratio > 0.8
    assert measured.scale_relative_to_contain == pytest.approx(expected_zoom, abs=0.025)
    assert measured.rotation_degrees == pytest.approx(expected_rotation, abs=0.2)
    assert measured.center_x == pytest.approx(expected_center[0] / output_width, abs=0.015)
    assert measured.center_y == pytest.approx(expected_center[1] / output_height, abs=0.015)


def test_video_motion_measurement_samples_first_last_and_intermediate_frames(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    source = rng.integers(0, 255, size=(512, 512), dtype=np.uint8)
    cv2.putText(source, "TRANSFER", (55, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 5)
    source_path = tmp_path / "source.png"
    assert cv2.imwrite(str(source_path), source)
    video_path = tmp_path / "motion.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (270, 480)
    )
    assert writer.isOpened()
    for frame_index in range(12):
        progress = frame_index / 11
        eased = progress * progress * (3 - 2 * progress)
        zoom = 2.0 + (1.1 - 2.0) * eased
        rotation = -3.0 * (1 - eased)
        center = (105 + 30 * eased, 285 - 45 * eased)
        scale = 270 / 512 * zoom
        angle = math.radians(rotation)
        matrix = np.array(
            [
                [scale * math.cos(angle), -scale * math.sin(angle), 0.0],
                [scale * math.sin(angle), scale * math.cos(angle), 0.0],
            ],
            dtype=np.float64,
        )
        matrix[:, 2] = np.array(center) - matrix[:, :2] @ np.array([256.0, 256.0])
        frame = cv2.warpAffine(source, matrix, (270, 480), borderValue=0)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    motion = measure_video_motion(source_path, video_path, sample_step=5)

    assert [item.frame for item in motion.measurements] == [0, 5, 10, 11]
    assert motion.frame_count == 12
    assert motion.frame_rate.numerator == 10
    assert motion.curve_hint == "ease_in_out"
    assert min(item.inlier_ratio for item in motion.measurements) > 0.75

    output_root = tmp_path / "reference-eval"
    spec = analyze_reference(
        source_path,
        video_path,
        output_root,
        readiness_receipt={
            "overall": "pass",
            "reference_content_decoded": False,
            "git_revision": "synthetic-test-revision",
            "run_root": str(tmp_path / "readiness"),
        },
        sample_step=5,
    )
    assert spec.frozen_before_first_render is True
    assert spec.reference_media_allowed_in_render is False
    assert spec.reference_sha256 != spec.source_sha256
    assert spec.camera_motion.curve_hint == "ease_in_out"
    assert spec.composition.model == "single_source_similarity_transform"
    assert (output_root / "reference-spec.json").is_file()
