"""Gated, immutable reference-analysis workflow."""

from __future__ import annotations

import re
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

from aoa_editing.analysis.reference import measure_video_motion
from aoa_editing.domain.models import (
    CameraFrameMeasurement,
    MediaKind,
    Provenance,
    ReferenceAudioStructure,
    ReferenceCameraMotion,
    ReferenceComposition,
    ReferenceReconstructionSpec,
)
from aoa_editing.infrastructure.media import ffprobe, sha256_file

MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?(?:inf|[0-9.]+))\s*dB", re.IGNORECASE)
MAX_VOLUME = re.compile(r"max_volume:\s*(-?(?:inf|[0-9.]+))\s*dB", re.IGNORECASE)
SILENCE_START = re.compile(r"silence_start:\s*([-0-9.]+)")
SILENCE_END = re.compile(
    r"silence_end:\s*([-0-9.]+)\s*\|\s*silence_duration:\s*([-0-9.]+)"
)
SCENE_TIME = re.compile(r"pts_time:([0-9.]+)")


class ReferenceWorkflowError(RuntimeError):
    """The reference workflow failed without authoring a mutable target."""


def analyze_reference(
    source_path: Path,
    reference_path: Path,
    output_root: Path,
    *,
    readiness_receipt: dict[str, Any],
    sample_step: int = 5,
) -> ReferenceReconstructionSpec:
    """Freeze observable requirements before any reconstruction render exists."""

    _validate_readiness(readiness_receipt)
    source_path = source_path.expanduser().resolve(strict=True)
    reference_path = reference_path.expanduser().resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "reference-spec.json"
    if spec_path.exists():
        return ReferenceReconstructionSpec.model_validate_json(
            spec_path.read_text(encoding="utf-8")
        )
    kind, metadata, probe_payload, probe_command = ffprobe(reference_path)
    if kind is not MediaKind.VIDEO or not metadata.has_video:
        raise ReferenceWorkflowError("reference must contain a video stream")
    motion = measure_video_motion(source_path, reference_path, sample_step=sample_step)
    cuts, scene_command = _scene_cuts(reference_path)
    audio, audio_commands = _audio_structure(reference_path, metadata.duration_seconds)
    duration_seconds = metadata.duration_seconds or motion.frame_count / motion.frame_rate.fps
    measurements = [
        CameraFrameMeasurement(
            frame=item.frame,
            time_seconds=item.time_seconds,
            scale_relative_to_contain=item.scale_relative_to_contain,
            rotation_degrees=item.rotation_degrees,
            center_x=item.center_x,
            center_y=item.center_y,
            match_count=item.match_count,
            inlier_count=item.inlier_count,
            inlier_ratio=item.inlier_ratio,
            reprojection_rmse=item.reprojection_rmse,
        )
        for item in motion.measurements
    ]
    source_match_confidence = min(item.inlier_ratio for item in measurements)
    spec = ReferenceReconstructionSpec(
        readiness_revision=str(readiness_receipt["git_revision"]),
        readiness_receipt_path=str(readiness_receipt.get("run_root", "unknown")),
        source_sha256=sha256_file(source_path),
        reference_sha256=sha256_file(reference_path),
        reference_metadata=metadata,
        duration_frames=motion.frame_count,
        duration_seconds=duration_seconds,
        camera_motion=ReferenceCameraMotion(
            model="single_source_similarity_transform",
            frame_count=motion.frame_count,
            frame_rate=motion.frame_rate,
            width=motion.width,
            height=motion.height,
            curve_hint=motion.curve_hint,
            measurements=measurements,
        ),
        composition=ReferenceComposition(
            model="single_source_similarity_transform",
            cut_count=len(cuts),
            independent_layer_motion_observed=False,
            masks_observed=False,
            background="black outside the transformed source bounds",
            notes=[
                "A single global similarity transform explains sampled source features.",
                "No sampled frame requires independently moving source layers.",
                f"Minimum robust feature inlier ratio is {source_match_confidence:.4f}.",
            ],
        ),
        audio=audio,
        observed_effects={
            "camera": {
                "operation": "continuous zoom, rotation, and center-position animation",
                "onset_frame": 0,
                "offset_frame": motion.frame_count - 1,
                "curve_hint": motion.curve_hint,
            },
            "transitions": {"cut_times_seconds": cuts, "count": len(cuts)},
            "spatial": {
                "fit_transition": "cropped close view to full-source contain view",
                "alpha_or_background_exposure": "black canvas becomes visible",
            },
            "color_and_light": {
                "classification": "no independent animated effect established",
                "comparison_required": True,
            },
            "masks_and_layers": {
                "classification": "not observed",
                "claim_limit": "global affine agreement does not prove semantic layer absence",
            },
        },
        uncertainties=[
            "Compression and color-range conversion may account for residual pixel differences.",
            "Measurements are sampled; intermediate frames are reconstructed by keyframe curves.",
            (
                "Feature agreement supports, but cannot mathematically prove, absence of "
                "hidden layers."
            ),
        ],
        alternative_explanations=[
            "Equivalent motion may have been authored as a transformed layer or virtual camera.",
            "A denser keyframe curve may approximate the same continuous source transform.",
        ],
        comparison_criteria={
            "geometry": {
                "resolution": [motion.width, motion.height],
                "frame_rate": motion.frame_rate.model_dump(mode="json"),
                "duration_tolerance_frames": 1,
            },
            "motion": {
                "scale_rmse_max": 0.03,
                "rotation_rmse_degrees_max": 0.30,
                "center_rmse_normalized_max": 0.015,
            },
            "perceptual": {
                "mean_ssim_min": 0.90,
                "mean_vmaf_min": 80.0,
                "visual_contact_sheet_review_required": True,
            },
            "audio": {
                "duration_tolerance_frames": 1,
                "silence_required_when_reference_is_silent": True,
            },
            "lineage": {
                "reference_hash_forbidden": sha256_file(reference_path),
                "only_visual_source_hash": sha256_file(source_path),
            },
        },
        provenance=Provenance(
            tool="aoa-editing-reference-analysis",
            tool_version=version("aoa-editing"),
            command=[*probe_command, *scene_command, *audio_commands],
            parameters={
                "motion_algorithm": "SIFT + RANSAC similarity transform",
                "opencv_version": version("opencv-python-headless"),
                "sample_step_frames": sample_step,
                "probe_stream_count": len(probe_payload.get("streams", [])),
            },
            deterministic=True,
        ),
    )
    rendered = spec.model_dump_json(indent=2) + "\n"
    try:
        with spec_path.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    except FileExistsError:
        return ReferenceReconstructionSpec.model_validate_json(
            spec_path.read_text(encoding="utf-8")
        )
    return spec


def _validate_readiness(receipt: dict[str, Any]) -> None:
    if receipt.get("overall") != "pass":
        raise ReferenceWorkflowError("readiness receipt is not passing")
    if receipt.get("reference_content_decoded") is not False:
        raise ReferenceWorkflowError("readiness receipt does not prove sealed pre-analysis")
    if not receipt.get("git_revision"):
        raise ReferenceWorkflowError("readiness receipt has no Git revision")


def _scene_cuts(path: Path) -> tuple[list[float], list[str]]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-filter:v",
        "select='gt(scene,0.32)',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = _run(command)
    return sorted({float(value) for value in SCENE_TIME.findall(result.stderr)}), command


def _audio_structure(
    path: Path, duration_seconds: float | None
) -> tuple[ReferenceAudioStructure, list[str]]:
    _, metadata, _, _ = ffprobe(path)
    if not metadata.has_audio:
        return (
            ReferenceAudioStructure(
                has_audio_stream=False,
                silent_for_full_duration=True,
            ),
            [],
        )
    volume_command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        "volumedetect",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    silence_command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        "silencedetect=noise=-60dB:d=0.05",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    volume = _run(volume_command).stderr
    silence = _run(silence_command).stderr
    mean = _volume(MEAN_VOLUME, volume)
    maximum = _volume(MAX_VOLUME, volume)
    starts = [float(value) for value in SILENCE_START.findall(silence)]
    ends = [(float(end), float(length)) for end, length in SILENCE_END.findall(silence)]
    intervals = [
        {
            "start_seconds": start,
            "end_seconds": ends[index][0] if index < len(ends) else duration_seconds or start,
            "duration_seconds": (
                ends[index][1]
                if index < len(ends)
                else max(0.0, (duration_seconds or start) - start)
            ),
        }
        for index, start in enumerate(starts)
    ]
    full_silence = bool(
        intervals
        and intervals[0]["start_seconds"] <= 0.05
        and duration_seconds is not None
        and intervals[-1]["end_seconds"] >= duration_seconds - 0.05
    )
    return (
        ReferenceAudioStructure(
            has_audio_stream=True,
            codec=metadata.audio_codec,
            sample_rate=metadata.sample_rate,
            channels=metadata.channels,
            mean_volume_db=mean,
            max_volume_db=maximum,
            silent_for_full_duration=full_silence,
            silence_intervals=intervals,
        ),
        [*volume_command, *silence_command],
    )


def _volume(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if match is None:
        return None
    if match.group(1).lower() == "-inf":
        return -999.0
    return float(match.group(1))


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        raise ReferenceWorkflowError(result.stderr[-3000:] or f"{command[0]} failed")
    return result
