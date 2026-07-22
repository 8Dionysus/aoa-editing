"""Objective comparison of a normal render against a frozen reference spec."""

from __future__ import annotations

import json
import math
import re
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from aoa_editing.analysis.reference import measure_video_motion
from aoa_editing.domain.models import (
    CheckResult,
    Provenance,
    ReferenceComparisonReport,
    ReferenceReconstructionSpec,
)
from aoa_editing.infrastructure.media import ffprobe, sha256_file

VMAF_SCORE = re.compile(r"VMAF score:\s*([0-9.]+)")
SSIM_ALL = re.compile(r"All:([0-9.]+)")
PSNR_AVERAGE = re.compile(r"average:(inf|[0-9.]+)", re.IGNORECASE)
MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?(?:inf|[0-9.]+))\s*dB", re.IGNORECASE)


class ComparisonError(RuntimeError):
    """Comparison inputs or adapters violated the frozen evaluation contract."""


def run_comparison(
    spec: ReferenceReconstructionSpec,
    source_path: Path,
    candidate_path: Path,
    reference_path: Path,
    output_root: Path,
    *,
    lineage_path: Path,
    sample_step: int = 5,
    visual_review: str | None = None,
) -> ReferenceComparisonReport:
    """Measure a candidate while keeping reference media outside render lineage."""

    source_path = source_path.expanduser().resolve(strict=True)
    candidate_path = candidate_path.expanduser().resolve(strict=True)
    reference_path = reference_path.expanduser().resolve(strict=True)
    lineage_path = lineage_path.expanduser().resolve(strict=True)
    if sha256_file(source_path) != spec.source_sha256:
        raise ComparisonError("source hash does not match frozen comparison spec")
    if sha256_file(reference_path) != spec.reference_sha256:
        raise ComparisonError("reference hash does not match frozen comparison spec")
    if sample_step < 1:
        raise ComparisonError("sample_step must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "comparison.json"
    if report_path.exists():
        raise ComparisonError(f"comparison root already contains a report: {report_path}")

    _, candidate_metadata, _, candidate_probe = ffprobe(candidate_path)
    _, reference_metadata, _, reference_probe = ffprobe(reference_path)
    measured_motion = measure_video_motion(
        source_path, candidate_path, sample_step=sample_step
    )
    geometry, geometry_checks = _geometry(spec, measured_motion, candidate_metadata)
    motion, motion_checks = _motion(spec, measured_motion.measurements)

    metric_width, metric_height = _metric_geometry(
        reference_metadata.width or spec.camera_motion.width,
        reference_metadata.height or spec.camera_motion.height,
    )
    vmaf_path = output_root / "vmaf.json"
    ssim_path = output_root / "ssim.log"
    psnr_path = output_root / "psnr.log"
    vmaf, vmaf_command = _vmaf(
        candidate_path, reference_path, metric_width, metric_height, vmaf_path
    )
    ssim, ssim_command = _ssim(
        candidate_path, reference_path, metric_width, metric_height, ssim_path
    )
    psnr, psnr_command = _psnr(
        candidate_path, reference_path, metric_width, metric_height, psnr_path
    )
    perceptual, perceptual_checks = _perceptual(spec, vmaf, ssim, psnr)

    audio, audio_check, audio_command = _audio(spec, candidate_path)
    lineage = _load_lineage(spec, lineage_path)
    lineage_check = CheckResult(
        id="render-lineage",
        status="pass" if lineage["valid"] else "fail",
        summary=(
            "render lineage contains only the permitted visual source"
            if lineage["valid"]
            else "render lineage contains missing or forbidden visual inputs"
        ),
        measured=lineage,
    )

    side_by_side = output_root / "side-by-side.mp4"
    contact_sheet = output_root / "side-by-side-contact-sheet.jpg"
    side_command = _side_by_side(
        reference_path,
        candidate_path,
        side_by_side,
        metric_width,
        metric_height,
    )
    sheet_command = _contact_sheet(side_by_side, contact_sheet, spec.duration_seconds)
    visual_check = CheckResult(
        id="visual-review",
        status="pass" if visual_review else "warn",
        summary=(
            "a human-readable visual review is recorded"
            if visual_review
            else "contact sheet exists but visual review is not yet recorded"
        ),
        measured={"review": visual_review, "contact_sheet": str(contact_sheet)},
    )
    checks = [
        *geometry_checks,
        *motion_checks,
        *perceptual_checks,
        audio_check,
        lineage_check,
        visual_check,
    ]
    overall = _overall(checks)
    commands = [
        candidate_probe,
        reference_probe,
        vmaf_command,
        ssim_command,
        psnr_command,
        audio_command,
        side_command,
        sheet_command,
    ]
    report = ReferenceComparisonReport(
        spec_id=spec.id,
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        candidate_sha256=sha256_file(candidate_path),
        candidate_path=str(candidate_path),
        geometry=geometry,
        motion=motion,
        perceptual=perceptual,
        audio=audio,
        lineage=lineage,
        checks=checks,
        artifacts={
            "report": str(report_path),
            "side_by_side": str(side_by_side),
            "contact_sheet": str(contact_sheet),
            "vmaf_log": str(vmaf_path),
            "ssim_log": str(ssim_path),
            "psnr_log": str(psnr_path),
        },
        visual_review=visual_review,
        provenance=Provenance(
            tool="aoa-editing-reference-comparison",
            tool_version=version("aoa-editing"),
            command=[item for command in commands for item in command],
            parameters={
                "sample_step_frames": sample_step,
                "metric_width": metric_width,
                "metric_height": metric_height,
                "reference_role": "comparison-only",
            },
            deterministic=True,
        ),
        overall=overall,
    )
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(report.model_dump_json(indent=2))
        stream.write("\n")
    return report


def _geometry(
    spec: ReferenceReconstructionSpec, motion: Any, metadata: Any
) -> tuple[dict[str, Any], list[CheckResult]]:
    expected_rate = spec.camera_motion.frame_rate.fps
    actual_rate = motion.frame_rate.fps
    duration_error = abs(motion.frame_count - spec.duration_frames)
    tolerance = int(
        spec.comparison_criteria.get("geometry", {}).get(
            "duration_tolerance_frames", 1
        )
    )
    resolution_pass = (motion.width, motion.height) == (
        spec.camera_motion.width,
        spec.camera_motion.height,
    )
    rate_pass = abs(actual_rate - expected_rate) < 1e-6
    duration_pass = duration_error <= tolerance
    geometry = {
        "expected_resolution": [spec.camera_motion.width, spec.camera_motion.height],
        "actual_resolution": [motion.width, motion.height],
        "expected_frame_rate": expected_rate,
        "actual_frame_rate": actual_rate,
        "expected_frames": spec.duration_frames,
        "actual_frames": motion.frame_count,
        "duration_error_frames": duration_error,
        "codec": metadata.video_codec,
        "pixel_format": metadata.pixel_format,
    }
    checks = [
        CheckResult(
            id="comparison-resolution",
            status="pass" if resolution_pass else "fail",
            summary="candidate resolution matches the frozen spec",
            measured=geometry,
        ),
        CheckResult(
            id="comparison-frame-rate",
            status="pass" if rate_pass else "fail",
            summary="candidate frame rate matches the frozen spec",
            measured={"expected": expected_rate, "actual": actual_rate},
        ),
        CheckResult(
            id="comparison-duration",
            status="pass" if duration_pass else "fail",
            summary="candidate frame count is within the frozen tolerance",
            measured={"error_frames": duration_error, "tolerance_frames": tolerance},
        ),
    ]
    return geometry, checks


def _motion(
    spec: ReferenceReconstructionSpec, actual_items: tuple[Any, ...]
) -> tuple[dict[str, Any], list[CheckResult]]:
    expected = {item.frame: item for item in spec.camera_motion.measurements}
    pairs = [(expected[item.frame], item) for item in actual_items if item.frame in expected]
    if not pairs:
        raise ComparisonError("candidate motion samples do not overlap frozen measurements")
    scale_rmse = _rmse(
        left.scale_relative_to_contain - right.scale_relative_to_contain
        for left, right in pairs
    )
    rotation_rmse = _rmse(
        left.rotation_degrees - right.rotation_degrees for left, right in pairs
    )
    center_rmse = _rmse(
        math.hypot(left.center_x - right.center_x, left.center_y - right.center_y)
        for left, right in pairs
    )
    criteria = spec.comparison_criteria.get("motion", {})
    scale_max = float(criteria.get("scale_rmse_max", 0.03))
    rotation_max = float(criteria.get("rotation_rmse_degrees_max", 0.30))
    center_max = float(criteria.get("center_rmse_normalized_max", 0.015))
    motion = {
        "sample_count": len(pairs),
        "sample_frames": [left.frame for left, _ in pairs],
        "scale_rmse": scale_rmse,
        "rotation_rmse_degrees": rotation_rmse,
        "center_rmse_normalized": center_rmse,
        "thresholds": {
            "scale_rmse_max": scale_max,
            "rotation_rmse_degrees_max": rotation_max,
            "center_rmse_normalized_max": center_max,
        },
    }
    checks = [
        _threshold_check("motion-scale", scale_rmse, scale_max, "scale RMSE"),
        _threshold_check("motion-rotation", rotation_rmse, rotation_max, "rotation RMSE"),
        _threshold_check("motion-center", center_rmse, center_max, "center-path RMSE"),
    ]
    return motion, checks


def _perceptual(
    spec: ReferenceReconstructionSpec, vmaf: float, ssim: float, psnr: float
) -> tuple[dict[str, Any], list[CheckResult]]:
    criteria = spec.comparison_criteria.get("perceptual", {})
    vmaf_min = float(criteria.get("mean_vmaf_min", 80.0))
    ssim_min = float(criteria.get("mean_ssim_min", 0.90))
    measured = {
        "vmaf_mean": vmaf,
        "ssim_all": ssim,
        "psnr_average_db": psnr,
        "thresholds": {"vmaf_mean_min": vmaf_min, "ssim_all_min": ssim_min},
    }
    return measured, [
        _minimum_check("perceptual-vmaf", vmaf, vmaf_min, "mean VMAF"),
        _minimum_check("perceptual-ssim", ssim, ssim_min, "mean SSIM"),
    ]


def _audio(
    spec: ReferenceReconstructionSpec, candidate: Path
) -> tuple[dict[str, Any], CheckResult, list[str]]:
    _, metadata, _, _ = ffprobe(candidate)
    if not metadata.has_audio:
        silent = True
        mean_volume: float | None = None
        command: list[str] = []
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(candidate),
            "-af",
            "volumedetect",
            "-vn",
            "-f",
            "null",
            "-",
        ]
        stderr = _run(command).stderr
        match = MEAN_VOLUME.search(stderr)
        mean_volume = _db(match.group(1)) if match else None
        silent = mean_volume is not None and mean_volume <= -60.0
    required = spec.audio.silent_for_full_duration
    passed = not required or silent
    measured = {
        "has_audio_stream": metadata.has_audio,
        "mean_volume_db": mean_volume,
        "silent_below_minus_60_db": silent,
        "silence_required": required,
    }
    return measured, CheckResult(
        id="comparison-audio-structure",
        status="pass" if passed else "fail",
        summary="candidate audio obeys the frozen silence requirement",
        measured=measured,
    ), command


def _load_lineage(
    spec: ReferenceReconstructionSpec, lineage_path: Path
) -> dict[str, Any]:
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    inputs = payload.get("input_hashes")
    forbidden = payload.get("forbidden_hashes_present", [])
    valid = (
        inputs == [spec.source_sha256]
        and spec.reference_sha256 not in inputs
        and spec.reference_sha256 not in forbidden
        and forbidden == []
    )
    return {
        "path": str(lineage_path),
        "input_hashes": inputs,
        "forbidden_hashes_present": forbidden,
        "expected_only_visual_source_hash": spec.source_sha256,
        "forbidden_reference_hash": spec.reference_sha256,
        "valid": valid,
    }


def _vmaf(
    candidate: Path, reference: Path, width: int, height: int, output: Path
) -> tuple[float, list[str]]:
    graph = _metric_graph(width, height, "libvmaf=log_fmt=json:log_path=" + str(output))
    command = _metric_command(candidate, reference, graph)
    result = _run(command)
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        try:
            return float(payload["pooled_metrics"]["vmaf"]["mean"]), command
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonError("libvmaf JSON has no pooled mean") from error
    match = VMAF_SCORE.search(result.stderr)
    if match is None:
        raise ComparisonError("libvmaf produced no score")
    return float(match.group(1)), command


def _ssim(
    candidate: Path, reference: Path, width: int, height: int, output: Path
) -> tuple[float, list[str]]:
    graph = _metric_graph(width, height, "ssim=stats_file=" + str(output))
    command = _metric_command(candidate, reference, graph)
    result = _run(command)
    match = SSIM_ALL.search(result.stderr)
    if match is None:
        raise ComparisonError("SSIM produced no aggregate score")
    return float(match.group(1)), command


def _psnr(
    candidate: Path, reference: Path, width: int, height: int, output: Path
) -> tuple[float, list[str]]:
    graph = _metric_graph(width, height, "psnr=stats_file=" + str(output))
    command = _metric_command(candidate, reference, graph)
    result = _run(command)
    match = PSNR_AVERAGE.search(result.stderr)
    if match is None:
        raise ComparisonError("PSNR produced no aggregate score")
    value = match.group(1)
    return (999.0 if value.lower() == "inf" else float(value)), command


def _metric_graph(width: int, height: int, terminal: str) -> str:
    return (
        f"[0:v]scale={width}:{height}:flags=lanczos,settb=AVTB,setpts=PTS-STARTPTS[d];"
        f"[1:v]scale={width}:{height}:flags=lanczos,settb=AVTB,setpts=PTS-STARTPTS[r];"
        f"[d][r]{terminal}"
    )


def _metric_command(candidate: Path, reference: Path, graph: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(candidate),
        "-i",
        str(reference),
        "-lavfi",
        graph,
        "-an",
        "-f",
        "null",
        "-",
    ]


def _side_by_side(
    reference: Path,
    candidate: Path,
    output: Path,
    width: int,
    height: int,
) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(reference),
        "-i",
        str(candidate),
        "-filter_complex",
        (
            f"[0:v]scale={width}:{height}:flags=lanczos,setpts=PTS-STARTPTS[left];"
            f"[1:v]scale={width}:{height}:flags=lanczos,setpts=PTS-STARTPTS[right];"
            "[left][right]hstack=inputs=2[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(output),
    ]
    _run(command)
    return command


def _contact_sheet(side_by_side: Path, output: Path, duration: float) -> list[str]:
    interval = max(duration / 6.0, 0.04)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(side_by_side),
        "-vf",
        f"fps=1/{interval},scale=540:-2:flags=lanczos,tile=3x2",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    _run(command)
    return command


def _metric_geometry(width: int, height: int) -> tuple[int, int]:
    bounded_width = min(width, 540)
    bounded_height = round(height * bounded_width / width)
    bounded_height -= bounded_height % 2
    return bounded_width, max(2, bounded_height)


def _rmse(values: Any) -> float:
    items = list(values)
    return math.sqrt(sum(value * value for value in items) / len(items))


def _threshold_check(check_id: str, value: float, maximum: float, label: str) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if value <= maximum else "fail",
        summary=f"{label} is within the frozen maximum",
        measured={"actual": value, "maximum": maximum},
    )


def _minimum_check(check_id: str, value: float, minimum: float, label: str) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if value >= minimum else "fail",
        summary=f"{label} meets the frozen minimum",
        measured={"actual": value, "minimum": minimum},
    )


def _overall(checks: list[CheckResult]) -> Literal["pass", "fail", "warn"]:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status in {"warn", "skip"} for check in checks):
        return "warn"
    return "pass"


def _db(value: str) -> float:
    return -999.0 if value.lower() == "-inf" else float(value)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        raise ComparisonError(result.stderr[-4000:] or f"{command[0]} failed")
    return result
