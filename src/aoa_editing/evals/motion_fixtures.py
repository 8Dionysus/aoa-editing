"""Deterministic ground-truth motion fixtures independent of user media."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from aoa_editing.analysis.motion import evaluate_curve, finite_derivatives
from aoa_editing.domain.models import (
    FrameRate,
    MotionCurve,
    MotionGroundTruthCorpus,
    MotionGroundTruthFixture,
    MotionGroundTruthFrame,
    MotionModel,
    Provenance,
)
from aoa_editing.infrastructure.media import sha256_file


@dataclass(frozen=True, slots=True)
class MotionFixtureDefinition:
    id: str
    expected_model: MotionModel
    curve: MotionCurve
    requirement_tags: tuple[str, ...]
    transform_kind: Literal[
        "similarity",
        "curved_translation",
        "moving_pivot",
        "affine",
        "homography",
        "local_warp",
    ] = "similarity"
    degradations: tuple[str, ...] = ()
    source_kind: Literal["rgb", "alpha", "low_resolution"] = "rgb"
    output_kind: Literal["portrait", "landscape", "square"] = "portrait"
    frame_count: int = 24
    fps: int = 12
    content_origin: Literal["synthetic_ground_truth"] = "synthetic_ground_truth"


@dataclass(frozen=True, slots=True)
class RenderedMotionFixture:
    source_path: Path
    video_path: Path
    truth_path: Path
    truth: MotionGroundTruthFixture


def fixture_definitions() -> tuple[MotionFixtureDefinition, ...]:
    """Return coverage-oriented fixtures; none is derived from the sealed reference."""

    return (
        _definition("linear", "linear", "linear"),
        _definition("ease-in", "ease_in", "ease_in"),
        _definition("ease-out", "ease_out", "ease_out"),
        _definition("ease-in-out", "ease_in_out", "ease_in_out"),
        _definition("cubic-bezier", "cubic_bezier", "cubic_bezier"),
        _definition("hermite", "hermite", "hermite"),
        _definition("overshoot", "overshoot", "overshoot"),
        _definition("settle", "settle", "settle"),
        MotionFixtureDefinition(
            id="scale-rotation",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("scale_rotation",),
        ),
        MotionFixtureDefinition(
            id="curved-translation",
            expected_model="similarity",
            curve="linear",
            requirement_tags=("curved_translation",),
            transform_kind="curved_translation",
        ),
        MotionFixtureDefinition(
            id="moving-pivot",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("moving_pivot",),
            transform_kind="moving_pivot",
        ),
        MotionFixtureDefinition(
            id="coupled-scale-rotation",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("coupled_scale_rotation",),
        ),
        MotionFixtureDefinition(
            id="affine-shear",
            expected_model="affine",
            curve="ease_in_out",
            requirement_tags=("affine",),
            transform_kind="affine",
        ),
        MotionFixtureDefinition(
            id="perspective",
            expected_model="homography",
            curve="ease_in_out",
            requirement_tags=("homography",),
            transform_kind="homography",
        ),
        MotionFixtureDefinition(
            id="local-warp",
            expected_model="unknown",
            curve="ease_in_out",
            requirement_tags=("local_warp",),
            transform_kind="local_warp",
        ),
        MotionFixtureDefinition(
            id="compression",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("compression",),
            degradations=("jpeg_quality_28",),
        ),
        MotionFixtureDefinition(
            id="blur",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("blur",),
            degradations=("gaussian_blur",),
        ),
        MotionFixtureDefinition(
            id="texture-loss",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("texture_loss",),
            degradations=("quarter_resolution_reconstruction",),
        ),
        MotionFixtureDefinition(
            id="partial-offscreen",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("partial_offscreen",),
            degradations=("partial_offscreen_crop",),
        ),
        MotionFixtureDefinition(
            id="landscape-output",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("different_aspect_ratios",),
            output_kind="landscape",
        ),
        MotionFixtureDefinition(
            id="square-output",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("different_aspect_ratios",),
            output_kind="square",
        ),
        MotionFixtureDefinition(
            id="alpha-source",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("alpha_source",),
            source_kind="alpha",
        ),
        MotionFixtureDefinition(
            id="low-resolution",
            expected_model="similarity",
            curve="ease_in_out",
            requirement_tags=("low_resolution",),
            source_kind="low_resolution",
        ),
    )


def render_motion_fixture(
    definition: MotionFixtureDefinition,
    root: Path,
) -> RenderedMotionFixture:
    """Render one video and its exact matrix/derivative truth sidecar."""

    if root.exists():
        raise FileExistsError(f"motion fixture root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    source = _source(definition.source_kind)
    source_path = root / ("source.png")
    if not cv2.imwrite(str(source_path), source):
        raise RuntimeError(f"cannot write synthetic motion source: {source_path}")
    output_width, output_height = _output_geometry(definition.output_kind)
    video_path = root / "motion.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cast(Any, cv2).VideoWriter_fourcc(*"MJPG"),
        float(definition.fps),
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open synthetic motion writer: {video_path}")

    matrices: list[NDArray[np.float64]] = []
    pivots: list[tuple[float, float]] = []
    parameters: list[dict[str, float]] = []
    try:
        for frame_index in range(definition.frame_count):
            time = frame_index / max(1, definition.frame_count - 1)
            progress = evaluate_curve(definition.curve, time)
            matrix, pivot = _matrix(
                definition,
                progress=progress,
                time=time,
                source_shape=source.shape,
                output_shape=(output_height, output_width),
            )
            frame = _render_frame(
                source,
                matrix,
                output_shape=(output_height, output_width),
                local_warp=definition.transform_kind == "local_warp",
                time=time,
            )
            frame = _degrade(frame, definition.degradations)
            writer.write(frame)
            matrices.append(matrix)
            pivots.append(pivot)
            parameters.append(
                _truth_parameters(
                    matrix,
                    source_shape=source.shape,
                    output_shape=(output_height, output_width),
                )
            )
    finally:
        writer.release()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"synthetic video is empty: {video_path}")

    derivatives = {
        name: finite_derivatives(
            [item[name] for item in parameters],
            fps=float(definition.fps),
        )
        for name in ("center_x", "center_y", "scale", "rotation")
    }
    truth_frames: list[MotionGroundTruthFrame] = []
    for index, (matrix, pivot, item) in enumerate(
        zip(matrices, pivots, parameters, strict=True)
    ):
        truth_frames.append(
            MotionGroundTruthFrame(
                frame=index,
                time_seconds=index / definition.fps,
                matrix_3x3=matrix.tolist(),
                center_x=item["center_x"],
                center_y=item["center_y"],
                scale_relative_to_contain=item["scale"],
                rotation_degrees=item["rotation"],
                pivot_x=pivot[0],
                pivot_y=pivot[1],
                velocity={
                    name: series.velocity[index] for name, series in derivatives.items()
                },
                acceleration={
                    name: series.acceleration[index]
                    for name, series in derivatives.items()
                },
                jerk={name: series.jerk[index] for name, series in derivatives.items()},
                phase=_phase(index, definition.frame_count),
                transform_order=_transform_order(definition),
            )
        )
    expected_uncertainties = ["pivot_translation_gauge"]
    if definition.transform_kind == "local_warp":
        expected_uncertainties.append("no_single_global_model")
    truth = MotionGroundTruthFixture(
        id=definition.id,
        source_path=str(source_path),
        source_sha256=sha256_file(source_path),
        video_path=str(video_path),
        video_sha256=sha256_file(video_path),
        source_width=source.shape[1],
        source_height=source.shape[0],
        output_width=output_width,
        output_height=output_height,
        frame_count=definition.frame_count,
        frame_rate=FrameRate(numerator=definition.fps),
        expected_model=definition.expected_model,
        expected_curve=definition.curve,
        requirement_tags=list(definition.requirement_tags),
        degradations=list(definition.degradations),
        expected_uncertainties=expected_uncertainties,
        frames=truth_frames,
        provenance=Provenance(
            tool="aoa-editing-ground-truth-motion-renderer",
            tool_version=version("aoa-editing"),
            command=[],
            parameters={
                "fixture_id": definition.id,
                "transform_kind": definition.transform_kind,
                "curve": definition.curve,
                "content_origin": definition.content_origin,
            },
            deterministic=True,
        ),
    )
    truth_path = root / "ground-truth.json"
    with truth_path.open("x", encoding="utf-8") as stream:
        stream.write(truth.model_dump_json(indent=2))
        stream.write("\n")
    return RenderedMotionFixture(
        source_path=source_path,
        video_path=video_path,
        truth_path=truth_path,
        truth=truth,
    )


def create_ground_truth_corpus(
    root: Path,
    *,
    fixture_ids: set[str] | None = None,
) -> tuple[MotionGroundTruthCorpus, Path]:
    """Render a selected or complete corpus and persist a joined manifest."""

    if root.exists():
        raise FileExistsError(f"motion corpus root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    definitions = [
        item
        for item in fixture_definitions()
        if fixture_ids is None or item.id in fixture_ids
    ]
    if fixture_ids is not None:
        missing = fixture_ids - {item.id for item in definitions}
        if missing:
            raise ValueError(f"unknown motion fixtures: {sorted(missing)}")
    rendered = [
        render_motion_fixture(definition, root / "fixtures" / definition.id)
        for definition in definitions
    ]
    coverage: dict[str, list[str]] = {}
    for definition in definitions:
        for tag in definition.requirement_tags:
            coverage.setdefault(tag, []).append(definition.id)
    corpus = MotionGroundTruthCorpus(
        fixtures=[item.truth for item in rendered],
        requirement_coverage=dict(sorted(coverage.items())),
        provenance=Provenance(
            tool="aoa-editing-ground-truth-motion-corpus",
            tool_version=version("aoa-editing"),
            command=[],
            parameters={
                "fixture_count": len(rendered),
                "selection": sorted(fixture_ids) if fixture_ids is not None else "complete",
                "sealed_reference_used": False,
            },
            deterministic=True,
        ),
    )
    manifest = root / "motion-ground-truth.json"
    with manifest.open("x", encoding="utf-8") as stream:
        stream.write(corpus.model_dump_json(indent=2))
        stream.write("\n")
    return corpus, manifest


def _definition(
    fixture_id: str,
    curve: MotionCurve,
    requirement: str,
) -> MotionFixtureDefinition:
    return MotionFixtureDefinition(
        id=fixture_id,
        expected_model="similarity",
        curve=curve,
        requirement_tags=(requirement,),
    )


def _source(
    kind: Literal["rgb", "alpha", "low_resolution"],
) -> NDArray[np.uint8]:
    width, height = (160, 120) if kind == "low_resolution" else (512, 384)
    rng = np.random.default_rng(0xA0A_2026 + width)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    font_scale = max(0.45, width / 410)
    cv2.putText(
        image,
        "SYNTHETIC MOTION",
        (max(4, width // 24), height // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (255, 255, 255),
        max(1, round(width / 180)),
        cv2.LINE_AA,
    )
    for index in range(24):
        angle = 2.0 * math.pi * index / 24
        center = (
            round(width / 2 + width * 0.36 * math.cos(angle)),
            round(height / 2 + height * 0.36 * math.sin(angle)),
        )
        radius = max(2, round(width * (0.008 + 0.002 * (index % 4))))
        cv2.circle(image, center, radius, (8, 8, 8), 2, cv2.LINE_AA)
    cv2.line(
        image,
        (width // 14, height * 5 // 6),
        (width * 13 // 14, height // 7),
        (245, 198, 37),
        max(2, width // 120),
        cv2.LINE_AA,
    )
    if kind != "alpha":
        return image
    alpha = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        alpha,
        (width // 2, height // 2),
        (width * 47 // 100, height * 46 // 100),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    return np.dstack((image, alpha))


def _output_geometry(
    kind: Literal["portrait", "landscape", "square"],
) -> tuple[int, int]:
    if kind == "landscape":
        return 480, 270
    if kind == "square":
        return 384, 384
    return 360, 480


def _matrix(
    definition: MotionFixtureDefinition,
    *,
    progress: float,
    time: float,
    source_shape: tuple[int, ...],
    output_shape: tuple[int, int],
) -> tuple[NDArray[np.float64], tuple[float, float]]:
    source_height, source_width = source_shape[:2]
    output_height, output_width = output_shape
    contain = min(output_width / source_width, output_height / source_height)
    scale = contain * (1.08 + 0.54 * progress)
    rotation = -9.0 + 17.0 * progress
    center_x = 0.43 + 0.13 * progress
    center_y = 0.61 - 0.17 * progress
    if definition.id == "partial-offscreen":
        center_x = 0.16 + 0.16 * progress
        center_y = 0.72 - 0.08 * progress
        scale *= 1.25
    if definition.transform_kind == "curved_translation":
        center_x = 0.34 + 0.32 * time
        center_y = 0.54 + 0.11 * math.sin(math.pi * time)
        scale = contain * (1.12 + 0.36 * time)
        rotation = -5.0 + 10.0 * time

    pivot_x = 0.5
    pivot_y = 0.5
    if definition.transform_kind == "moving_pivot":
        pivot_x = 0.28 + 0.34 * progress
        pivot_y = 0.64 - 0.24 * progress
    pivot_pixels = np.asarray(
        [source_width * pivot_x, source_height * pivot_y],
        dtype=np.float64,
    )
    angle = math.radians(rotation)
    linear = np.asarray(
        [
            [scale * math.cos(angle), -scale * math.sin(angle)],
            [scale * math.sin(angle), scale * math.cos(angle)],
        ],
        dtype=np.float64,
    )
    if definition.transform_kind == "affine":
        deformation = np.asarray(
            [
                [1.0 + 0.22 * progress, 0.16 * progress],
                [0.04 * progress, 1.0 - 0.12 * progress],
            ],
            dtype=np.float64,
        )
        linear = linear @ deformation
    output_anchor = np.asarray(
        [center_x * output_width, center_y * output_height],
        dtype=np.float64,
    )
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = output_anchor - linear @ pivot_pixels
    if definition.transform_kind == "homography":
        perspective = np.eye(3, dtype=np.float64)
        perspective[2, 0] = 0.00048 * progress
        perspective[2, 1] = -0.00026 * progress
        matrix = perspective @ matrix
        matrix /= matrix[2, 2]
    return matrix, (pivot_x, pivot_y)


def _render_frame(
    source: NDArray[np.uint8],
    matrix: NDArray[np.float64],
    *,
    output_shape: tuple[int, int],
    local_warp: bool,
    time: float,
) -> NDArray[np.uint8]:
    output_height, output_width = output_shape
    transformed = cv2.warpPerspective(
        source,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    if transformed.ndim == 3 and transformed.shape[2] == 4:
        alpha = transformed[:, :, 3:4].astype(np.float32) / 255.0
        background = np.zeros((output_height, output_width, 3), dtype=np.float32)
        background[:, :] = (16, 22, 31)
        transformed = np.asarray(
            transformed[:, :, :3].astype(np.float32) * alpha
            + background * (1.0 - alpha),
            dtype=np.uint8,
        )
    if local_warp:
        y, x = np.indices((output_height, output_width), dtype=np.float32)
        amplitude = 8.0 + 3.0 * math.sin(math.pi * time)
        map_x = x + amplitude * np.sin(2.0 * math.pi * y / 91.0 + time * 1.7)
        map_y = y + amplitude * 0.65 * np.sin(2.0 * math.pi * x / 117.0 - time)
        transformed = cv2.remap(
            transformed,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    return cast(NDArray[np.uint8], transformed)


def _degrade(
    frame: NDArray[np.uint8],
    degradations: tuple[str, ...],
) -> NDArray[np.uint8]:
    result: NDArray[np.uint8] = frame
    if "gaussian_blur" in degradations:
        result = cast(NDArray[np.uint8], cv2.GaussianBlur(result, (7, 7), 1.6))
    if "quarter_resolution_reconstruction" in degradations:
        height, width = result.shape[:2]
        small = cv2.resize(result, (max(1, width // 4), max(1, height // 4)))
        result = cast(
            NDArray[np.uint8],
            cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST),
        )
    if "jpeg_quality_28" in degradations:
        available, encoded = cv2.imencode(
            ".jpg",
            result,
            [cv2.IMWRITE_JPEG_QUALITY, 28],
        )
        if not available:
            raise RuntimeError("cannot JPEG-degrade synthetic motion frame")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("cannot decode synthetic JPEG degradation")
        result = cast(NDArray[np.uint8], decoded)
    return result


def _truth_parameters(
    matrix: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    output_shape: tuple[int, int],
) -> dict[str, float]:
    source_height, source_width = source_shape[:2]
    output_height, output_width = output_shape
    source_center = np.asarray([source_width / 2, source_height / 2], dtype=np.float64)
    mapped = cv2.perspectiveTransform(
        source_center.reshape(1, 1, 2),
        matrix,
    ).reshape(2)
    jacobian = _jacobian(matrix, source_center)
    left, _, right = np.linalg.svd(jacobian)
    rotation_matrix = left @ right
    rotation = math.degrees(
        math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))
    )
    contain = min(output_width / source_width, output_height / source_height)
    return {
        "center_x": float(mapped[0] / output_width),
        "center_y": float(mapped[1] / output_height),
        "scale": math.sqrt(abs(float(np.linalg.det(jacobian)))) / contain,
        "rotation": rotation,
    }


def _jacobian(
    matrix: NDArray[np.float64],
    point: NDArray[np.float64],
) -> NDArray[np.float64]:
    x, y = float(point[0]), float(point[1])
    first = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    second = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    squared = denominator**2
    return np.asarray(
        [
            [
                (matrix[0, 0] * denominator - first * matrix[2, 0]) / squared,
                (matrix[0, 1] * denominator - first * matrix[2, 1]) / squared,
            ],
            [
                (matrix[1, 0] * denominator - second * matrix[2, 0]) / squared,
                (matrix[1, 1] * denominator - second * matrix[2, 1]) / squared,
            ],
        ],
        dtype=np.float64,
    )


def _phase(frame: int, frame_count: int) -> str:
    progress = frame / max(1, frame_count - 1)
    if progress <= 0.12:
        return "onset"
    if progress >= 0.86:
        return "settle"
    return "active"


def _transform_order(definition: MotionFixtureDefinition) -> list[str]:
    order = ["translate_to_output_anchor"]
    if definition.transform_kind == "homography":
        order.append("perspective")
    if definition.transform_kind == "affine":
        order.append("affine_deformation")
    order.extend(("rotate", "scale", "translate_negative_pivot"))
    if definition.transform_kind == "local_warp":
        order.append("local_warp_after_global_transform")
    return order
