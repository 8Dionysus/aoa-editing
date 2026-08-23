"""Independent, network-free quality corpus for Video Anatomy.

The fixtures are authored from explicit temporal truth and only then handed to
the production pipeline.  Detector output never participates in fixture
generation or truth construction.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from aoa_editing import __version__
from aoa_editing.application.video_jobs import VideoAnatomyJobService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    AICapabilityDeclaration,
    CheckResult,
    FrameRange,
    FrameRate,
    Intent,
    LocalProviderBinding,
    LocalProviderHealthBinding,
    Provenance,
    ProviderHealthEvidence,
    Scenario,
    VideoAnatomy,
    VideoAnatomyEvalCase,
    VideoAnatomyEvalCorpus,
    VideoAnatomyEvalReport,
    VideoAnatomyFixtureTruth,
    VideoAnatomyProfile,
    VideoAnatomyTruthBoundary,
    VideoMotionClassification,
    VideoTransitionType,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.service import (
    ProviderExecution,
    ProviderExecutor,
    ProviderService,
)

VISION_ALIAS = "local-ai://vision/describe/default"
SPEECH_ALIAS = "local-ai://speech/transcript/default"
_WIDTH = 320
_HEIGHT = 180


class VideoAnatomyEvalError(RuntimeError):
    """The synthetic quality gate cannot make a complete, truthful claim."""


@dataclass(frozen=True, slots=True)
class _RenderedFixture:
    truth: VideoAnatomyFixtureTruth
    truth_path: Path
    video_path: Path


@dataclass(slots=True)
class _FixtureProviderExecutor(ProviderExecutor):
    """Deterministic local adapter used only to prove provider-neutral plumbing."""

    def health(
        self,
        _binding: LocalProviderBinding,
        _declaration: AICapabilityDeclaration,
    ) -> ProviderHealthEvidence:
        return ProviderHealthEvidence(
            status="healthy",
            latency_milliseconds=0,
            source="injected-test",
            response_sha256="a" * 64,
            reported_status="ready",
            protocol_version="aoa-local-ai-v1",
            backend="fixture-vision",
            model_id="fixture-vision-model",
            model_revision="synthetic-v1",
            summary="deterministic network-free eval provider",
        )

    def invoke(
        self,
        binding: LocalProviderBinding,
        payload: dict[str, Any],
    ) -> ProviderExecution:
        if binding.alias == SPEECH_ALIAS:
            return ProviderExecution(
                output={
                    "segments": [
                        {
                            "start": 0.3,
                            "end": 1.1,
                            "text": "Measured speech.",
                            "speaker": "synthetic-speaker",
                            "words": [
                                {
                                    "text": "Measured",
                                    "start_seconds": 0.3,
                                    "end_seconds": 0.7,
                                    "confidence": 1.0,
                                },
                                {
                                    "text": "speech.",
                                    "start_seconds": 0.7,
                                    "end_seconds": 1.1,
                                    "confidence": 1.0,
                                },
                            ],
                        }
                    ],
                    "partial": False,
                }
            )
        times = payload.get("sample_times_seconds", [])
        return ProviderExecution(
            output={
                "descriptions": [
                    {
                        "time_seconds": float(value),
                        "text": f"Synthetic fixture observation at {float(value):.6f} seconds.",
                        "confidence": 1.0,
                    }
                    for value in times
                ],
                "partial": False,
            }
        )


def create_video_anatomy_corpus(root: Path) -> tuple[VideoAnatomyEvalCorpus, Path]:
    """Render all compact fixtures without reading user or reference media."""

    selected = root.expanduser().resolve()
    if selected.exists():
        raise FileExistsError(f"Video Anatomy corpus root already exists: {selected}")
    selected.mkdir(parents=True)
    rendered = [
        _render_transitions(selected / "fixtures" / "transitions"),
        _render_motion(selected / "fixtures" / "motion"),
        _render_audio(selected / "fixtures" / "audio"),
        _render_fractional_long(selected / "fixtures" / "fractional-long"),
        _render_vfr(selected / "fixtures" / "vfr"),
    ]
    coverage: dict[str, list[str]] = {}
    for item in rendered:
        for tag in item.truth.requirement_tags:
            coverage.setdefault(tag, []).append(item.truth.id)
    corpus = VideoAnatomyEvalCorpus(
        fixtures=[item.truth for item in rendered],
        requirement_coverage=dict(sorted(coverage.items())),
        provenance=Provenance(
            tool="aoa-editing-video-anatomy-synthetic-corpus",
            tool_version=__version__,
            parameters={
                "fixture_count": len(rendered),
                "network_used": False,
                "reference_media_used": False,
                "truth_derived_from_detector": False,
            },
            deterministic=True,
        ),
    )
    manifest = selected / "video-anatomy-corpus.json"
    _write_json_new(manifest, corpus.model_dump(mode="json"))
    return corpus, manifest


def run_video_anatomy_eval(
    output_root: Path,
    *,
    repository: Path,
    implementation_revision: str,
    git_clean: bool,
    require_clean: bool = True,
) -> VideoAnatomyEvalReport:
    """Run production services over independent truth and emit a fail-closed report."""

    root = output_root.expanduser().resolve()
    if root.exists():
        raise VideoAnatomyEvalError(f"Video Anatomy eval output root must be absent: {root}")
    if require_clean and not git_clean:
        raise VideoAnatomyEvalError("release Video Anatomy eval requires a clean revision")
    if len(implementation_revision) != 40:
        raise VideoAnatomyEvalError("implementation revision must be a full Git SHA")
    corpus, corpus_path = create_video_anatomy_corpus(root / "corpus")
    settings = Settings.for_home(root / "workspace")
    store = ProjectStore(settings)
    providers = _fixture_provider_service(settings)
    cases: list[VideoAnatomyEvalCase] = []
    truth_by_id = {item.id: item for item in corpus.fixtures}
    profiles = {
        "transitions": (VideoAnatomyProfile.STRUCTURAL, VideoAnatomyProfile.SEMANTIC),
        "motion": (VideoAnatomyProfile.MOTION,),
        "audio": (VideoAnatomyProfile.QUICK,),
        "fractional-long": (VideoAnatomyProfile.QUICK, VideoAnatomyProfile.STRUCTURAL),
        "vfr": (VideoAnatomyProfile.STRUCTURAL,),
    }
    for fixture_id, selected_profiles in profiles.items():
        truth = truth_by_id[fixture_id]
        video = root / "corpus" / truth.video_path
        project = store.create_project(
            f"Video Anatomy eval: {fixture_id}",
            Intent(
                text="Measure synthetic video structure without changing a timeline.",
                scenario=Scenario.MEMORY_MONTAGE,
            ),
        )
        asset, _probe = store.ingest_asset(project.id, video)
        if asset.sha256 != truth.video_sha256:
            raise VideoAnatomyEvalError(f"fixture hash changed before analysis: {fixture_id}")
        for profile in selected_profiles:
            started = time.monotonic()
            result = VideoAnatomyJobService(store, providers=providers).run(
                project.id,
                asset.id,
                profile=profile,
                explicit_provider_opt_in=profile is VideoAnatomyProfile.SEMANTIC,
            )
            elapsed = time.monotonic() - started
            artifact_root = store.video_anatomy_path(project.id, result.anatomy.plan.id)
            artifact_bytes = sum(
                path.stat().st_size for path in artifact_root.rglob("*") if path.is_file()
            )
            cases.append(
                _evaluate_case(
                    truth,
                    result.anatomy,
                    runtime_seconds=elapsed,
                    artifact_bytes=artifact_bytes,
                )
            )

    aggregate = _aggregate_metrics(cases)
    required_tags = _required_tags()
    missing_tags = sorted(required_tags - set(corpus.requirement_coverage))
    checks = [
        _check(
            "requirement-coverage",
            not missing_tags,
            "every required synthetic condition is independently represented",
            missing=missing_tags,
        ),
        _check(
            "boundary-recall",
            aggregate["boundary_recall"] >= 0.8,
            "shot-forming boundary recall is at least 0.80",
            value=aggregate["boundary_recall"],
        ),
        _check(
            "boundary-precision",
            aggregate["boundary_precision"] >= 0.7,
            "shot-forming boundary precision is at least 0.70",
            value=aggregate["boundary_precision"],
        ),
        _check(
            "transition-classification",
            aggregate["transition_classification_accuracy"] >= 0.65,
            "matched transition types are correct at least 65 percent of the time",
            value=aggregate["transition_classification_accuracy"],
        ),
        _check(
            "temporal-error",
            aggregate["boundary_temporal_error_frames"] <= 4.0,
            "mean matched boundary error is at most four frames",
            value=aggregate["boundary_temporal_error_frames"],
        ),
        _check(
            "shot-coverage",
            aggregate["shot_coverage"] == 1.0,
            "all structural cases cover the complete requested time axis",
            value=aggregate["shot_coverage"],
        ),
        _check(
            "short-event-recall",
            aggregate["short_event_frame_selection_recall"] == 1.0,
            "sampling preserves every synthetic short event",
            value=aggregate["short_event_frame_selection_recall"],
        ),
        _check(
            "semantic-coverage",
            aggregate["semantic_observation_coverage"] == 1.0,
            "the neutral fixture provider covers every detected shot",
            value=aggregate["semantic_observation_coverage"],
        ),
        _check(
            "motion-model-error",
            aggregate["motion_model_error"] <= 0.4,
            "synthetic motion label error is at most 0.40",
            value=aggregate["motion_model_error"],
        ),
        _check(
            "audio-event-coverage",
            aggregate["audio_event_coverage"] == 1.0,
            "transcript and deterministic audio analysis recover every truth event family",
            value=aggregate["audio_event_coverage"],
        ),
        _check(
            "audio-accent-alignment",
            aggregate["audio_accent_boundary_error_frames"] <= 5.0,
            "the nearest measured accent is within five frames of its truth boundary",
            value=aggregate["audio_accent_boundary_error_frames"],
        ),
        _check(
            "vfr-exact-timestamps",
            aggregate["vfr_timestamp_exactness"] == 1.0,
            "VFR evidence uses the hash-bound decoded-order PTS map",
            value=aggregate["vfr_timestamp_exactness"],
        ),
        _check(
            "fractional-rate-identity",
            aggregate["fractional_rate_identity"] == 1.0,
            "fractional source rate survives probe, plan, and evidence unchanged",
            value=aggregate["fractional_rate_identity"],
        ),
        _check(
            "unresolved-ratio",
            aggregate["unresolved_range_ratio"] <= 0.2,
            "unresolved structural range ratio remains bounded",
            value=aggregate["unresolved_range_ratio"],
        ),
        _check(
            "no-reference-media",
            True,
            "corpus construction and evaluation never access sealed reference media",
            reference_media_used=False,
        ),
        CheckResult(
            id="clean-revision",
            status="pass" if git_clean else ("fail" if require_clean else "warn"),
            summary="report is bound to a clean implementation revision",
            measured={"git_clean": git_clean, "required": require_clean},
        ),
    ]
    mandatory_skips = sum(item.status == "skip" for item in checks)
    overall: Literal["pass", "fail"] = (
        "pass" if all(item.status in {"pass", "warn"} for item in checks) else "fail"
    )
    report = VideoAnatomyEvalReport(
        git_revision=implementation_revision,
        git_clean=git_clean,
        corpus_path=str(corpus_path.relative_to(root)),
        corpus_sha256=sha256_file(corpus_path),
        cases=cases,
        aggregate_metrics=aggregate,
        requirement_coverage=corpus.requirement_coverage,
        checks=checks,
        mandatory_skips=mandatory_skips,
        provenance=Provenance(
            tool="aoa-editing-video-anatomy-eval",
            tool_version=__version__,
            parameters={
                "implementation_revision": implementation_revision,
                "repository": str(repository.resolve()),
                "network_used": False,
                "reference_media_used": False,
                "thresholds_predeclared": True,
            },
            deterministic=False,
        ),
        overall=overall,
    )
    _write_json_new(root / "video-anatomy-eval.json", report.model_dump(mode="json"))
    return report


def _render_transitions(root: Path) -> _RenderedFixture:
    fps = 24
    red = _pattern((20, 45, 220), "RED SHOT")
    green = _pattern((20, 150, 35), "GREEN SHOT")
    yellow = _pattern((20, 210, 220), "YELLOW SHOT")
    equal_luma_green = _pattern((0, 130, 0), "EQUAL LUMA B")
    equal_luma_red = _pattern((0, 0, 255), "EQUAL LUMA A")
    frames: list[NDArray[np.uint8]] = []
    title = _pattern((18, 18, 18), "VIDEO ANATOMY TITLE", overlay="SMALL TEXT OVERLAY")
    frames.extend(_copies(title, 24))
    frames.extend(_copies(red, 24))
    short_insert = _pattern((220, 190, 30), "SHORT INSERT", overlay="3 FRAME EVENT")
    frames.extend(_copies(short_insert, 5))
    frames.extend(_copies(red, 19))
    frames.extend(_blend(red, green, 12))
    frames.extend(_copies(green, 24))
    frames.append(np.full_like(green, 255))
    frames.extend(_copies(green, 11))
    frames.extend(_fade(green, 16, fade_in=False))
    frames.extend(_copies(np.zeros_like(green), 12))
    frames.extend(_fade(yellow, 16, fade_in=True))
    frames.extend(_copies(yellow, 16))
    panorama = np.concatenate([yellow, equal_luma_green, equal_luma_red], axis=1)
    for offset in range(16):
        start = round(offset * (_WIDTH * 2) / 15)
        frame = panorama[:, start : start + _WIDTH].copy()
        frame = cast(NDArray[np.uint8], cv2.GaussianBlur(frame, (11, 1), 0))
        frames.append(frame)
    frames.extend(_copies(equal_luma_red, 20))
    frames.extend(_copies(equal_luma_green, 24))
    if len(frames) != 240:
        raise AssertionError(f"transition fixture length drifted: {len(frames)}")
    video = root / "transitions.mp4"
    _write_video(video, frames, fps)
    boundaries = [
        _truth(24, "hard-cut", 1, True, "title to first content shot"),
        _truth(48, "hard-cut", 1, True, "short insert entry"),
        _truth(53, "hard-cut", 1, True, "short insert exit"),
        _truth(78, "cross-dissolve", 7, True, "twelve-frame dissolve midpoint"),
        _truth(108, "flash", 1, False, "single-frame flash inside one shot"),
        _truth(127, "fade-out", 8, True, "fade to black midpoint"),
        _truth(155, "fade-in", 8, True, "fade from black midpoint"),
        _truth(187, "wipe-or-directional", 8, True, "whip-like directional transition"),
        _truth(216, "hard-cut", 1, True, "equal-luminance different-colour cut"),
    ]
    return _persist_truth(
        root,
        fixture_id="transitions",
        video=video,
        fps=FrameRate(numerator=fps),
        frame_count=len(frames),
        boundaries=boundaries,
        short_events=[FrameRange(start=48, duration=5), FrameRange(start=108, duration=1)],
        tags=[
            "hard-cuts",
            "fade-in-out",
            "cross-dissolve",
            "flash-within-shot",
            "whip-like-motion",
            "short-insert",
            "equal-luma-different-color",
            "text-overlay",
            "titles",
        ],
    )


def _render_motion(root: Path) -> _RenderedFixture:
    fps = 24
    shot_length = 36
    static_base = _motion_canvas(1, (35, 35, 180))
    pan_base = _motion_canvas(2, (180, 45, 35))
    zoom_base = _motion_canvas(3, (35, 170, 50))
    rotation_base = _motion_canvas(4, (150, 45, 160))
    independent_base = _motion_canvas(5, (30, 175, 180))
    frames: list[NDArray[np.uint8]] = []
    frames.extend(_copies(static_base, shot_length))
    wide = cast(NDArray[np.uint8], np.concatenate([pan_base, np.fliplr(pan_base)], axis=1))
    for index in range(shot_length):
        x = round(index * (_WIDTH - 1) / (shot_length - 1))
        frames.append(wide[:, x : x + _WIDTH].copy())
    for index in range(shot_length):
        scale = 1 + 0.22 * index / (shot_length - 1)
        matrix = cv2.getRotationMatrix2D((_WIDTH / 2, _HEIGHT / 2), 0, scale)
        frames.append(
            cast(
                NDArray[np.uint8],
                cv2.warpAffine(
                    zoom_base,
                    matrix,
                    (_WIDTH, _HEIGHT),
                    borderMode=cv2.BORDER_REFLECT,
                ),
            )
        )
    for index in range(shot_length):
        degrees = 12 * index / (shot_length - 1)
        matrix = cv2.getRotationMatrix2D((_WIDTH / 2, _HEIGHT / 2), degrees, 1.0)
        frames.append(
            cast(
                NDArray[np.uint8],
                cv2.warpAffine(
                    rotation_base,
                    matrix,
                    (_WIDTH, _HEIGHT),
                    borderMode=cv2.BORDER_REFLECT,
                ),
            )
        )
    for index in range(shot_length):
        frame = independent_base.copy()
        center = (20 + round(index * 270 / (shot_length - 1)), 90)
        cv2.circle(frame, center, 14, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        frames.append(frame)
    video = root / "motion.mp4"
    _write_video(video, frames, fps)
    return _persist_truth(
        root,
        fixture_id="motion",
        video=video,
        fps=FrameRate(numerator=fps),
        frame_count=len(frames),
        boundaries=[
            _truth(frame, "hard-cut", 1, True, f"motion phase boundary {frame}")
            for frame in (36, 72, 108, 144)
        ],
        motion=["static", "pan", "zoom", "rotation", "independent"],
        tags=["static-shot", "slow-zoom", "pan", "rotation", "independent-object"],
    )


def _render_audio(root: Path) -> _RenderedFixture:
    fps = 24
    frames = _copies(_pattern((30, 35, 45), "SPEECH / SILENCE"), 72)
    frames.extend(_copies(_pattern((45, 25, 55), "MUSIC / BEATS"), 72))
    silent = root / "audio-silent.mp4"
    _write_video(silent, frames, fps)
    speech = root / "speech.wav"
    subprocess.run(
        [
            "espeak-ng",
            "-s",
            "240",
            "-w",
            str(speech),
            "Measured speech.",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    bed = root / "music-beats.wav"
    _write_audio_bed(bed, duration_seconds=6.0)
    video = root / "audio.mp4"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(silent),
        "-i",
        str(speech),
        "-i",
        str(bed),
        "-filter_complex",
        "[1:a]adelay=300|300,volume=0.75[s];[s][2:a]amix=inputs=2:duration=longest:normalize=0[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-t",
        "6",
        str(video),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    silent.unlink()
    speech.unlink()
    bed.unlink()
    return _persist_truth(
        root,
        fixture_id="audio",
        video=video,
        fps=FrameRate(numerator=fps),
        frame_count=len(frames),
        boundaries=[_truth(72, "hard-cut", 1, True, "visual and audio accent boundary")],
        audio=["speech", "silence", "accent", "music", "beat"],
        tags=["speech", "silence", "audio-accent-boundary", "music-beats"],
    )


def _render_fractional_long(root: Path) -> _RenderedFixture:
    fps = FrameRate(numerator=30_000, denominator=1_001)
    base = _pattern((40, 40, 40), "LONG TIMELINE")
    event = _pattern((230, 150, 20), "SHORT EVENT")
    frames = _copies(base, 300)
    event_ranges = [
        FrameRange(start=2, duration=3),
        FrameRange(start=149, duration=3),
        FrameRange(start=295, duration=3),
    ]
    for item in event_ranges:
        for index in range(item.start, item.end):
            frames[index] = event.copy()
    video = root / "fractional-long.mp4"
    _write_video(video, frames, fps.fps)
    actual_frame_count, actual_frame_rate = _probe_video_identity(video)
    boundaries = [
        _truth(frame, "hard-cut", 1, True, label)
        for frame, label in (
            (2, "beginning event entry"),
            (5, "beginning event exit"),
            (149, "middle event entry"),
            (152, "middle event exit"),
            (295, "ending event entry"),
            (298, "ending event exit"),
        )
    ]
    return _persist_truth(
        root,
        fixture_id="fractional-long",
        video=video,
        fps=actual_frame_rate,
        frame_count=actual_frame_count,
        boundaries=boundaries,
        short_events=event_ranges,
        tags=["fractional-fps", "long-video-events-start-middle-end", "no-audio"],
    )


def _render_vfr(root: Path) -> _RenderedFixture:
    root.mkdir(parents=True, exist_ok=False)
    frame_root = root / "source-frames"
    frame_root.mkdir()
    durations = [0.04 if index % 3 else 0.09 for index in range(24)]
    concat_lines: list[str] = []
    for index, duration in enumerate(durations):
        color = (25, 90, 180) if index < 12 else (180, 65, 25)
        path = frame_root / f"frame-{index:04d}.png"
        if not cv2.imwrite(str(path), _pattern(color, f"VFR {index:02d}")):
            raise VideoAnatomyEvalError(f"cannot write VFR source frame: {path}")
        concat_lines.extend([f"file '{path.as_posix()}'", f"duration {duration:.6f}"])
    concat_lines.append(f"file '{(frame_root / 'frame-0023.png').as_posix()}'")
    concat = root / "frames.ffconcat"
    concat.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    video = root / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    shutil.rmtree(frame_root)
    concat.unlink()
    frame_count, frame_rate = _probe_video_identity(video)
    return _persist_truth(
        root,
        fixture_id="vfr",
        video=video,
        fps=frame_rate,
        frame_count=frame_count,
        boundaries=[_truth(12, "hard-cut", 2, True, "VFR colour boundary")],
        variable_frame_rate=True,
        tags=["variable-frame-rate", "no-audio"],
    )


def _persist_truth(
    root: Path,
    *,
    fixture_id: str,
    video: Path,
    fps: FrameRate,
    frame_count: int,
    boundaries: list[VideoAnatomyTruthBoundary] | None = None,
    short_events: list[FrameRange] | None = None,
    motion: list[str] | None = None,
    audio: list[str] | None = None,
    variable_frame_rate: bool = False,
    tags: list[str],
) -> _RenderedFixture:
    truth = VideoAnatomyFixtureTruth(
        id=fixture_id,
        video_path=str(video.relative_to(root.parents[1])),
        video_sha256=sha256_file(video),
        frame_count=frame_count,
        frame_rate=fps,
        variable_frame_rate=variable_frame_rate,
        has_audio=bool(audio),
        boundaries=boundaries or [],
        short_event_ranges=short_events or [],
        expected_motion_labels=motion or [],
        expected_audio_labels=audio or [],
        requirement_tags=tags,
        provenance=Provenance(
            tool="aoa-editing-video-anatomy-fixture-generator",
            tool_version=__version__,
            parameters={
                "fixture_id": fixture_id,
                "truth_authored_before_analysis": True,
                "network_used": False,
                "reference_media_used": False,
            },
            deterministic=True,
        ),
    )
    truth_path = root / "ground-truth.json"
    _write_json_new(truth_path, truth.model_dump(mode="json"))
    return _RenderedFixture(truth=truth, truth_path=truth_path, video_path=video)


def _evaluate_case(
    truth: VideoAnatomyFixtureTruth,
    anatomy: VideoAnatomy,
    *,
    runtime_seconds: float,
    artifact_bytes: int,
) -> VideoAnatomyEvalCase:
    split_truth = [item for item in truth.boundaries if item.splits_shot]
    split_detected = [
        item
        for item in anatomy.structure.transitions
        if item.transition_type
        not in {VideoTransitionType.FLASH, VideoTransitionType.CONTINUOUS_MOTION}
        and item.confidence >= 0.55
    ]
    split_matches = _match_boundaries(split_truth, split_detected)
    all_matches = _match_boundaries(truth.boundaries, anatomy.structure.transitions)
    true_positive = len(split_matches)
    precision = (
        true_positive / len(split_detected) if split_detected else (1.0 if not split_truth else 0.0)
    )
    recall = true_positive / len(split_truth) if split_truth else 1.0
    temporal_error_sum = sum(
        abs(truth_item.frame - detected.frame) for truth_item, detected in split_matches
    )
    temporal_error = temporal_error_sum / len(split_matches) if split_matches else 0.0
    classification_correct = sum(
        truth_item.transition_type is detected.transition_type
        for truth_item, detected in all_matches
    )
    classification = classification_correct / len(truth.boundaries) if truth.boundaries else 1.0
    kept_frames = {item.source_frame_index for item in anatomy.frame_manifest.samples if item.kept}
    short_recall = (
        sum(
            any(frame in kept_frames for frame in range(item.start, item.end))
            for item in truth.short_event_ranges
        )
        / len(truth.short_event_ranges)
        if truth.short_event_ranges
        else 1.0
    )
    short_event_matches = sum(
        any(frame in kept_frames for frame in range(item.start, item.end))
        for item in truth.short_event_ranges
    )
    duplicate_reduction = anatomy.frame_manifest.duplicate_count / max(
        1, len(anatomy.frame_manifest.samples)
    )
    unresolved_frames = sum(item.duration for item in anatomy.unresolved_ranges)
    unresolved_ratio = unresolved_frames / anatomy.plan.analysis_range.duration
    semantic_coverage = anatomy.coverage_matrix.get("visual", 1.0)
    motion_error = _motion_error(truth.expected_motion_labels, anatomy)
    audio_kinds = (
        {item.kind for item in anatomy.audio_timeline.events}
        if anatomy.audio_timeline is not None
        else set()
    )
    expected_audio = set(truth.expected_audio_labels)
    audio_event_coverage = (
        len(expected_audio & audio_kinds) / len(expected_audio) if expected_audio else 1.0
    )
    accent_events = (
        [item for item in anatomy.audio_timeline.events if item.kind == "accent"]
        if anatomy.audio_timeline is not None
        else []
    )
    audio_accent_error = 0.0
    if truth.id == "audio":
        boundary = next(item for item in truth.boundaries if item.label.endswith("boundary"))
        audio_accent_error = (
            min(
                abs(item.start_seconds * truth.frame_rate.fps - boundary.frame)
                for item in accent_events
            )
            if accent_events
            else float(truth.frame_count)
        )
    vfr_timestamp_exactness = 1.0
    if truth.variable_frame_rate:
        exact_samples = all(
            item.provenance.parameters.get("exact_pts_seconds") == item.time_seconds
            and not item.seek_limitations
            for item in anatomy.frame_manifest.samples
        )
        vfr_timestamp_exactness = float(
            anatomy.plan.variable_frame_rate
            and anatomy.plan.frame_timestamp_count == truth.frame_count
            and anatomy.plan.frame_timestamps_sha256 is not None
            and exact_samples
        )
    fractional_rate_identity = 1.0
    if "fractional-fps" in truth.requirement_tags:
        fractional_rate_identity = float(
            truth.frame_rate.denominator > 1
            and anatomy.technical_metadata.frame_rate == truth.frame_rate
            and anatomy.plan.time_base == truth.frame_rate
        )
    metrics = {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_temporal_error_frames": temporal_error,
        "boundary_truth_count": float(len(split_truth)),
        "boundary_detected_count": float(len(split_detected)),
        "boundary_match_count": float(len(split_matches)),
        "boundary_temporal_error_sum": float(temporal_error_sum),
        "transition_classification_accuracy": classification,
        "transition_truth_count": float(len(truth.boundaries)),
        "transition_correct_count": float(classification_correct),
        "shot_coverage": anatomy.structure.coverage.coverage_ratio,
        "short_event_frame_selection_recall": short_recall,
        "short_event_count": float(len(truth.short_event_ranges)),
        "short_event_match_count": float(short_event_matches),
        "duplicate_reduction": duplicate_reduction,
        "semantic_observation_coverage": semantic_coverage,
        "motion_model_error": motion_error,
        "audio_event_coverage": audio_event_coverage,
        "audio_accent_boundary_error_frames": audio_accent_error,
        "vfr_timestamp_exactness": vfr_timestamp_exactness,
        "fractional_rate_identity": fractional_rate_identity,
        "unresolved_range_ratio": unresolved_ratio,
        "unresolved_frames": float(unresolved_frames),
        "requested_frames": float(anatomy.plan.analysis_range.duration),
    }
    checks = [
        _check(
            "complete-coverage",
            metrics["shot_coverage"] == 1.0,
            "shot partition covers the requested range",
        ),
        _check("first-frame", 0 in kept_frames, "first source frame is retained"),
        _check(
            "last-frame",
            anatomy.plan.analysis_range.end - 1 in kept_frames,
            "last requested frame is retained",
        ),
        _check(
            "short-events", short_recall == 1.0, "short events retain at least one selected frame"
        ),
        _check(
            "boundary-samples",
            all(
                any(
                    abs(sample.source_frame_index - item.frame) <= item.tolerance_frames + 1
                    for sample in anatomy.frame_manifest.samples
                    if sample.kept
                )
                for item in truth.boundaries
            ),
            "truth boundary windows retain review samples",
        ),
        _check(
            "timeline-inert",
            anatomy.canonical_edit_decision is False,
            "analysis produces evidence and never a canonical edit decision",
        ),
    ]
    return VideoAnatomyEvalCase(
        fixture_id=truth.id,
        profile=anatomy.profile,
        source_sha256=anatomy.source_sha256,
        plan_sha256=anatomy.plan.plan_sha256,
        anatomy_sha256=anatomy.anatomy_sha256,
        detected_boundaries=[
            {
                "frame": item.frame,
                "transition_type": item.transition_type.value,
                "confidence": item.confidence,
                "ambiguity_reasons": item.ambiguity_reasons,
            }
            for item in anatomy.structure.transitions
        ],
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        artifact_bytes=artifact_bytes,
        checks=checks,
        overall="pass" if all(item.status == "pass" for item in checks) else "fail",
    )


def _match_boundaries(
    truth: list[VideoAnatomyTruthBoundary],
    detected: list[Any],
) -> list[tuple[VideoAnatomyTruthBoundary, Any]]:
    matches: list[tuple[VideoAnatomyTruthBoundary, Any]] = []
    remaining = list(detected)
    for expected in truth:
        eligible = [
            item
            for item in remaining
            if abs(item.frame - expected.frame) <= expected.tolerance_frames
        ]
        if not eligible:
            continue
        selected = min(eligible, key=lambda item: abs(item.frame - expected.frame))
        matches.append((expected, selected))
        remaining.remove(selected)
    return matches


def _motion_error(labels: list[str], anatomy: VideoAnatomy) -> float:
    if not labels:
        return 0.0
    observed: set[str] = set()
    for item in anatomy.motion_evidence:
        if item.classification is VideoMotionClassification.STATIC:
            observed.add("static")
        if item.independent_motion_detected:
            observed.add("independent")
        observed.update(item.camera_hypotheses)
    return 1.0 - len(set(labels) & observed) / len(set(labels))


def _aggregate_metrics(cases: list[VideoAnatomyEvalCase]) -> dict[str, float]:
    primary_by_fixture: dict[str, VideoAnatomyEvalCase] = {}
    for item in cases:
        existing = primary_by_fixture.get(item.fixture_id)
        if existing is None or item.profile is VideoAnatomyProfile.STRUCTURAL:
            primary_by_fixture[item.fixture_id] = item
    primary = list(primary_by_fixture.values())
    truth_count = sum(item.metrics["boundary_truth_count"] for item in primary)
    detected_count = sum(item.metrics["boundary_detected_count"] for item in primary)
    match_count = sum(item.metrics["boundary_match_count"] for item in primary)
    transition_count = sum(item.metrics["transition_truth_count"] for item in primary)
    transition_correct = sum(item.metrics["transition_correct_count"] for item in primary)
    short_count = sum(item.metrics["short_event_count"] for item in primary)
    short_matches = sum(item.metrics["short_event_match_count"] for item in primary)
    unresolved_frames = sum(item.metrics["unresolved_frames"] for item in primary)
    requested_frames = sum(item.metrics["requested_frames"] for item in primary)
    semantic = next(
        item
        for item in cases
        if item.fixture_id == "transitions" and item.profile is VideoAnatomyProfile.SEMANTIC
    )
    motion = next(item for item in cases if item.profile is VideoAnatomyProfile.MOTION)
    audio = next(item for item in cases if item.fixture_id == "audio")
    vfr = next(item for item in cases if item.fixture_id == "vfr")
    fractional = next(
        item
        for item in cases
        if item.fixture_id == "fractional-long" and item.profile is VideoAnatomyProfile.STRUCTURAL
    )
    metrics = {
        "boundary_precision": match_count / detected_count if detected_count else 0.0,
        "boundary_recall": match_count / truth_count if truth_count else 1.0,
        "boundary_temporal_error_frames": (
            sum(item.metrics["boundary_temporal_error_sum"] for item in primary) / match_count
            if match_count
            else 0.0
        ),
        "transition_classification_accuracy": (
            transition_correct / transition_count if transition_count else 1.0
        ),
        "shot_coverage": min(item.metrics["shot_coverage"] for item in primary),
        "short_event_frame_selection_recall": (short_matches / short_count if short_count else 1.0),
        "duplicate_reduction": (
            sum(item.metrics["duplicate_reduction"] for item in primary) / len(primary)
        ),
        "semantic_observation_coverage": semantic.metrics["semantic_observation_coverage"],
        "motion_model_error": motion.metrics["motion_model_error"],
        "audio_event_coverage": audio.metrics["audio_event_coverage"],
        "audio_accent_boundary_error_frames": audio.metrics["audio_accent_boundary_error_frames"],
        "vfr_timestamp_exactness": vfr.metrics["vfr_timestamp_exactness"],
        "fractional_rate_identity": fractional.metrics["fractional_rate_identity"],
        "unresolved_range_ratio": (
            unresolved_frames / requested_frames if requested_frames else 0.0
        ),
    }
    metrics["runtime_seconds_total"] = sum(item.runtime_seconds for item in cases)
    metrics["artifact_bytes_total"] = float(sum(item.artifact_bytes for item in cases))
    return metrics


def _required_tags() -> set[str]:
    return {
        "hard-cuts",
        "fade-in-out",
        "cross-dissolve",
        "flash-within-shot",
        "whip-like-motion",
        "static-shot",
        "slow-zoom",
        "pan",
        "rotation",
        "independent-object",
        "short-insert",
        "equal-luma-different-color",
        "text-overlay",
        "titles",
        "variable-frame-rate",
        "fractional-fps",
        "no-audio",
        "speech",
        "silence",
        "audio-accent-boundary",
        "music-beats",
        "long-video-events-start-middle-end",
    }


def _fixture_provider_service(settings: Settings) -> ProviderService:
    vision_binding = LocalProviderBinding(
        alias=VISION_ALIAS,
        state="enabled",
        resolved_owner="aoa-editing",
        adapter_kind="localhost-http",
        backend="synthetic-fixture",
        model_id="fixture-vision-model",
        model_revision="synthetic-v1",
        model_hash_authority="generated fixture implementation",
        license_authority="repository test fixture",
        metadata_evidence="fixture://video-anatomy/vision",
        endpoint="http://127.0.0.1:9/video-anatomy-fixture",
        health=LocalProviderHealthBinding(endpoint="http://127.0.0.1:9/health"),
        data_boundary="local-host",
    )
    speech_binding = LocalProviderBinding(
        alias=SPEECH_ALIAS,
        state="enabled",
        resolved_owner="aoa-editing",
        adapter_kind="localhost-http",
        backend="synthetic-fixture",
        model_id="fixture-speech-model",
        model_revision="synthetic-v1",
        model_hash_authority="generated fixture implementation",
        license_authority="repository test fixture",
        metadata_evidence="fixture://video-anatomy/speech",
        endpoint="http://127.0.0.1:9/video-anatomy-speech-fixture",
        health=LocalProviderHealthBinding(endpoint="http://127.0.0.1:9/health"),
        data_boundary="local-host",
    )
    return ProviderService(
        settings,
        bindings={VISION_ALIAS: vision_binding, SPEECH_ALIAS: speech_binding},
        executor=_FixtureProviderExecutor(),
    )


def _truth(
    frame: int,
    transition_type: str,
    tolerance: int,
    splits_shot: bool,
    label: str,
) -> VideoAnatomyTruthBoundary:
    return VideoAnatomyTruthBoundary(
        frame=frame,
        transition_type=VideoTransitionType(transition_type),
        tolerance_frames=tolerance,
        splits_shot=splits_shot,
        label=label,
    )


def _pattern(
    color: tuple[int, int, int],
    label: str,
    *,
    overlay: str | None = None,
) -> NDArray[np.uint8]:
    frame = np.full((_HEIGHT, _WIDTH, 3), color, dtype=np.uint8)
    for x in range(0, _WIDTH, 32):
        cv2.line(frame, (x, 0), (x, _HEIGHT - 1), (255, 255, 255), 1)
    for y in range(0, _HEIGHT, 30):
        cv2.line(frame, (0, y), (_WIDTH - 1, y), (0, 0, 0), 1)
    cv2.putText(
        frame, label, (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    if overlay is not None:
        cv2.rectangle(frame, (10, 130), (310, 170), (0, 0, 0), -1)
        cv2.putText(
            frame,
            overlay,
            (18, 157),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def _motion_canvas(seed: int, color: tuple[int, int, int]) -> NDArray[np.uint8]:
    rng = np.random.default_rng(20260823 + seed)
    noise = rng.integers(-24, 25, size=(_HEIGHT, _WIDTH, 3), dtype=np.int16)
    frame = np.clip(
        np.full((_HEIGHT, _WIDTH, 3), color, dtype=np.int16) + noise,
        0,
        255,
    ).astype(np.uint8)
    frame = cast(NDArray[np.uint8], cv2.GaussianBlur(frame, (3, 3), 0))
    for index in range(24):
        x = 8 + (index * 53) % 304
        y = 8 + (index * 37) % 164
        cv2.circle(frame, (x, y), 5 + index % 7, (255, 255, 255), 2)
    return frame


def _copies(frame: NDArray[np.uint8], count: int) -> list[NDArray[np.uint8]]:
    return [frame.copy() for _ in range(count)]


def _blend(
    first: NDArray[np.uint8], second: NDArray[np.uint8], count: int
) -> list[NDArray[np.uint8]]:
    return [
        cast(
            NDArray[np.uint8],
            cv2.addWeighted(
                first,
                1.0 - index / (count - 1),
                second,
                index / (count - 1),
                0,
            ),
        )
        for index in range(count)
    ]


def _fade(frame: NDArray[np.uint8], count: int, *, fade_in: bool) -> list[NDArray[np.uint8]]:
    output: list[NDArray[np.uint8]] = []
    for index in range(count):
        progress = index / (count - 1)
        strength = progress if fade_in else 1 - progress
        output.append(np.clip(frame.astype(np.float32) * strength, 0, 255).astype(np.uint8))
    return output


def _write_video(path: Path, frames: list[NDArray[np.uint8]], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    writer = cv2.VideoWriter(
        str(path),
        cast(Any, cv2).VideoWriter_fourcc(*"mp4v"),
        fps,
        (_WIDTH, _HEIGHT),
    )
    if not writer.isOpened():
        raise VideoAnatomyEvalError(f"cannot open synthetic video writer: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoAnatomyEvalError(f"synthetic video writer produced no output: {path}")


def _write_audio_bed(path: Path, *, duration_seconds: float) -> None:
    sample_rate = 16_000
    times = np.arange(round(duration_seconds * sample_rate), dtype=np.float64) / sample_rate
    samples = np.zeros_like(times)
    music = times >= 3.0
    samples[music] = 0.06 * np.sin(2 * math.pi * 220 * times[music])
    for onset in np.arange(3.0, duration_seconds, 0.5):
        start = round(onset * sample_rate)
        duration = round(0.05 * sample_rate)
        envelope = np.linspace(1.0, 0.0, duration, endpoint=False)
        samples[start : start + duration] += 0.8 * envelope
    pcm = np.clip(samples, -1, 1)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes((pcm * 32767).astype("<i2").tobytes())


def _probe_video_identity(path: Path) -> tuple[int, FrameRate]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    return int(stream["nb_read_frames"]), FrameRate(numerator=numerator, denominator=denominator)


def _check(check_id: str, passed: bool, summary: str, **measured: Any) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if passed else "fail",
        summary=summary,
        measured=measured,
    )


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
