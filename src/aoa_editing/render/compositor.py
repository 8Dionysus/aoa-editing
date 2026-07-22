"""General matrix compositor adapter for derived Motion Language v2 samples."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from aoa_editing.domain.models import CompositorJobV2
from aoa_editing.infrastructure.media import sha256_file


class CompositorError(RuntimeError):
    """A derived matrix job failed before the final FFmpeg composition."""


class MatrixCompositor:
    """Warp immutable source pixels through compiler-owned matrix samples."""

    def render(self, job: CompositorJobV2) -> Path:
        source = job.source_path.expanduser().resolve(strict=True)
        if sha256_file(source) != job.source_sha256:
            raise CompositorError("compositor source hash differs from the render plan")
        decoded = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise CompositorError(f"cannot decode compositor source: {source}")
        premultiplied = _premultiplied_bgra(cast(NDArray[np.uint8], decoded))
        output = job.output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        process = subprocess.Popen(
            job.encoder_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None:  # pragma: no cover - subprocess contract
                raise CompositorError("compositor encoder has no stdin")
            for frame in job.frames:
                rendered = _render_samples(
                    premultiplied,
                    frame.matrices_3x3,
                    width=job.width,
                    height=job.height,
                    interpolation=job.interpolation,
                )
                process.stdin.write(rendered.tobytes(order="C"))
            process.stdin.close()
            stderr = (
                process.stderr.read().decode("utf-8", errors="replace")
                if process.stderr is not None
                else ""
            )
            return_code = process.wait(timeout=900)
            if return_code != 0:
                raise CompositorError(
                    stderr[-5000:] or "compositor FFmpeg encoder returned non-zero"
                )
            if not output.is_file() or output.stat().st_size < 1024:
                raise CompositorError("compositor intermediate is missing or too small")
            return output
        except Exception as error:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            output.unlink(missing_ok=True)
            if isinstance(error, CompositorError):
                raise
            raise CompositorError(str(error)) from error


def cleanup_compositor_jobs(jobs: list[CompositorJobV2]) -> None:
    """Remove only compiler-declared disposable intermediates."""

    parents: set[Path] = set()
    for job in jobs:
        job.output_path.unlink(missing_ok=True)
        parents.add(job.output_path.parent)
    for parent in sorted(parents, key=lambda item: len(item.parts), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            continue


def _premultiplied_bgra(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] == 4:
        bgra = image.copy()
    else:
        raise CompositorError(f"unsupported source channel count: {image.shape}")
    alpha = bgra[:, :, 3].astype(np.uint16)
    colors = bgra[:, :, :3].astype(np.uint16)
    bgra[:, :, :3] = np.asarray(
        (colors * alpha[:, :, None] + 127) // 255,
        dtype=np.uint8,
    )
    return cast(NDArray[np.uint8], bgra)


def _render_samples(
    source: NDArray[np.uint8],
    matrices: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
    *,
    width: int,
    height: int,
    interpolation: str,
) -> NDArray[np.uint8]:
    flags = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[interpolation]
    warped = [
        cast(
            NDArray[np.uint8],
            cv2.warpPerspective(
                source,
                np.asarray(matrix, dtype=np.float64),
                (width, height),
                flags=flags,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            ),
        )
        for matrix in matrices
    ]
    if len(warped) == 1:
        return _unpremultiply(warped[0])
    mean = np.mean(np.stack(warped, axis=0, dtype=np.float32), axis=0)
    return _unpremultiply(np.asarray(np.clip(np.rint(mean), 0, 255), dtype=np.uint8))


def _unpremultiply(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    output = image.copy()
    alpha = output[:, :, 3].astype(np.uint32)
    visible = alpha > 0
    colors = output[:, :, :3].astype(np.uint32)
    restored = np.zeros_like(colors)
    restored[visible] = np.minimum(
        255,
        (colors[visible] * 255 + alpha[visible, None] // 2) // alpha[visible, None],
    ).astype(np.uint32)
    output[:, :, :3] = np.asarray(restored, dtype=np.uint8)
    return output
