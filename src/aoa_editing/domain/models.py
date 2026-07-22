"""Versioned canonical contracts for projects, evidence, edits, and artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from math import gcd, isclose
from pathlib import Path
from typing import Annotated, Any, Final, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class TrackKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    CAPTION = "caption"


class Scenario(StrEnum):
    SPEECH_CLEAN = "speech.clean"
    MEMORY_MONTAGE = "memory.montage"
    STILL_MOTION = "still.motion"
    SCREEN_WORKFLOW = "screen.workflow"
    REFERENCE_RECONSTRUCT = "reference.reconstruct"


class ScreenWorkflowShotKind(StrEnum):
    ESTABLISH = "terminal.establish"
    FOCUS = "terminal.focus"
    FOLLOW = "terminal.follow"
    RESULT = "terminal.result"
    RESET = "terminal.reset"
    STEP_CUT = "workflow.step-cut"
    WAIT_COMPRESS = "workflow.wait-compress"


class ScreenWorkflowPacingRole(StrEnum):
    """Editorial meaning of one source-bound screen segment."""

    PROMPT_ENTRY = "prompt-entry"
    SEMANTIC_ACTION = "semantic-action"
    AGENT_PROGRESS = "agent-progress"
    READABLE_RESULT = "readable-result"
    MEASURED_WAIT = "measured-wait"


class ScreenWorkflowLiveSignal(StrEnum):
    """Visible terminal/UI motion that makes a nominally quiet frame live."""

    CURSOR_BLINK = "cursor-blink"
    PROGRESS_SHIMMER = "progress-shimmer"
    MONOTONIC_TIMER = "monotonic-timer"
    SPINNER = "spinner"
    MASCOT_ANIMATION = "mascot-animation"
    DIRECTIONAL_MOTION = "directional-motion"
    OTHER = "other"


class ScreenWorkflowTemporalBehavior(StrEnum):
    PERIODIC = "periodic"
    MONOTONIC = "monotonic"
    DIRECTIONAL = "directional"
    IRREGULAR = "irregular"


class ScreenWorkflowExperienceDisposition(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ScreenWorkflowExperienceRecurrence(StrEnum):
    SINGLE_DEMO = "single-demo"
    RECURRING_PROJECT_DEMOS = "recurring-project-demos"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DecisionOrigin(StrEnum):
    HUMAN = "human"
    RULE = "rule"
    MODEL = "model"
    IMPORT = "import"


class DecisionApproval(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class EvidenceAuthority(StrEnum):
    ANALYZER = "analyzer"
    HUMAN_CORRECTION = "human-correction"


class FrameRate(FrozenModel):
    numerator: int = Field(default=30, gt=0)
    denominator: int = Field(default=1, gt=0)

    @property
    def fps(self) -> float:
        return self.numerator / self.denominator


class FrameRange(FrozenModel):
    start: int = Field(ge=0)
    duration: int = Field(gt=0)

    @property
    def end(self) -> int:
        return self.start + self.duration


class MediaMetadata(FrozenModel):
    duration_seconds: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    frame_rate: FrameRate | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    format_name: str | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    pixel_format: str | None = None
    color_space: str | None = None
    has_video: bool = False
    has_audio: bool = False


class Asset(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("asset"))
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_name: str
    media_kind: MediaKind
    stored_path: str
    size_bytes: int = Field(ge=0)
    ingested_at: datetime = Field(default_factory=utc_now)
    immutable: Literal[True] = True
    metadata: MediaMetadata


class Provenance(FrozenModel):
    tool: str
    tool_version: str
    command: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    model_id: str | None = None
    model_revision: str | None = None
    deterministic: bool = True
    generated_at: datetime = Field(default_factory=utc_now)


class DerivedMediaManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("derivatives"))
    project_id: str
    asset_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: dict[str, str]
    output_hashes: dict[str, str]
    job_id: str
    provenance: Provenance
    generated_at: datetime = Field(default_factory=utc_now)


class EvidenceRecord(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("evidence"))
    project_id: str
    asset_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str
    authority: EvidenceAuthority = EvidenceAuthority.ANALYZER
    confidence: float = Field(default=1.0, ge=0, le=1)
    payload: dict[str, Any]
    artifacts: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    time_base: FrameRate | None = None
    ranges: list[FrameRange] = Field(default_factory=list)
    provenance: Provenance


class ScalarKeyframe(FrozenModel):
    frame: int = Field(ge=0)
    value: float
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out", "hold"] = "linear"


class Vec2Keyframe(FrozenModel):
    frame: int = Field(ge=0)
    x: float
    y: float
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out", "hold"] = "linear"


class TransformEffect(FrozenModel):
    type: Literal["transform"] = "transform"
    fit_mode: Literal["cover", "contain"] = "cover"
    position_mode: Literal["offset", "canvas_center"] = "offset"
    position: list[Vec2Keyframe] = Field(default_factory=list)
    scale: list[ScalarKeyframe] = Field(default_factory=list)
    rotation: list[ScalarKeyframe] = Field(default_factory=list)
    opacity: list[ScalarKeyframe] = Field(default_factory=list)
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    @model_validator(mode="after")
    def validate_keyframe_curves(self) -> TransformEffect:
        for name, keyframes in (
            ("position", self.position),
            ("scale", self.scale),
            ("rotation", self.rotation),
            ("opacity", self.opacity),
        ):
            frames = [keyframe.frame for keyframe in keyframes]
            if any(left >= right for left, right in pairwise(frames)):
                raise ValueError(f"{name} keyframe frames must be strictly increasing")
        if any(keyframe.value <= 0 for keyframe in self.scale):
            raise ValueError("scale keyframes must be positive")
        if any(not 0 <= keyframe.value <= 1 for keyframe in self.opacity):
            raise ValueError("opacity keyframes must be between 0 and 1")
        return self


class MotionTimeV2(FrozenModel):
    """Exact non-negative frame time with a canonical rational subframe."""

    frame: int = Field(ge=0)
    subframe_numerator: int = Field(default=0, ge=0)
    subframe_denominator: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_canonical_fraction(self) -> MotionTimeV2:
        if self.subframe_numerator >= self.subframe_denominator:
            raise ValueError("subframe numerator must be smaller than denominator")
        if gcd(self.subframe_numerator, self.subframe_denominator) != 1:
            raise ValueError("subframe fraction must be reduced")
        return self

    @property
    def as_fraction(self) -> Fraction:
        return Fraction(
            self.frame * self.subframe_denominator + self.subframe_numerator,
            self.subframe_denominator,
        )


class BezierHandleV2(FrozenModel):
    time_offset_frames: float
    value_offset: float


class Vec2ValueV2(FrozenModel):
    x: float
    y: float


class Vec2BezierHandleV2(FrozenModel):
    time_offset_frames: float
    x_offset: float
    y_offset: float


MotionInterpolationV2 = Literal[
    "linear",
    "hold",
    "cubic_bezier",
    "cubic_hermite",
    "monotone_cubic",
]


class ScalarMotionKeyframeV2(FrozenModel):
    time: MotionTimeV2
    value: float
    interpolation: MotionInterpolationV2 = "linear"
    incoming_tangent: float | None = None
    outgoing_tangent: float | None = None
    incoming_handle: BezierHandleV2 | None = None
    outgoing_handle: BezierHandleV2 | None = None


class Vec2MotionKeyframeV2(FrozenModel):
    time: MotionTimeV2
    value: Vec2ValueV2
    interpolation: MotionInterpolationV2 = "linear"
    incoming_tangent: Vec2ValueV2 | None = None
    outgoing_tangent: Vec2ValueV2 | None = None
    incoming_handle: Vec2BezierHandleV2 | None = None
    outgoing_handle: Vec2BezierHandleV2 | None = None


class ScalarMotionCurveV2(FrozenModel):
    keyframes: list[ScalarMotionKeyframeV2] = Field(min_length=1)
    continuity: Literal["c0", "c1", "c2"] = "c0"
    monotonicity: Literal["none", "increasing", "decreasing", "auto"] = "none"
    overshoot_policy: Literal["forbid", "bounded", "allow"] = "allow"
    overshoot_limit_fraction: float = Field(default=0.0, ge=0)
    angle_unwrap: bool = False

    @model_validator(mode="after")
    def validate_curve(self) -> ScalarMotionCurveV2:
        _validate_motion_times([item.time for item in self.keyframes], "scalar")
        _validate_scalar_segments(self.keyframes)
        values = [item.value for item in self.keyframes]
        if self.monotonicity == "increasing" and any(
            left > right for left, right in pairwise(values)
        ):
            raise ValueError("increasing curve keyframes must not decrease")
        if self.monotonicity == "decreasing" and any(
            left < right for left, right in pairwise(values)
        ):
            raise ValueError("decreasing curve keyframes must not increase")
        if self.monotonicity == "auto" and values:
            increasing = all(left <= right for left, right in pairwise(values))
            decreasing = all(left >= right for left, right in pairwise(values))
            if not increasing and not decreasing:
                raise ValueError("auto monotonic curve has non-monotonic keyframes")
        _validate_scalar_continuity(self)
        if self.overshoot_policy != "allow" and len(self.keyframes) > 1:
            from aoa_editing.domain.motion_v2 import scalar_curve_extrema

            observed_min, observed_max = scalar_curve_extrema(self)
            keyframe_min = min(values)
            keyframe_max = max(values)
            span = max(1e-12, keyframe_max - keyframe_min)
            allowance = (
                0.0 if self.overshoot_policy == "forbid" else span * self.overshoot_limit_fraction
            )
            tolerance = max(1e-10, span * 1e-9)
            if (
                observed_min < keyframe_min - allowance - tolerance
                or observed_max > keyframe_max + allowance + tolerance
            ):
                raise ValueError("curve overshoot exceeds the declared policy")
        return self


class Vec2MotionCurveV2(FrozenModel):
    keyframes: list[Vec2MotionKeyframeV2] = Field(min_length=1)
    continuity: Literal["c0", "c1", "c2"] = "c0"

    @model_validator(mode="after")
    def validate_curve(self) -> Vec2MotionCurveV2:
        _validate_motion_times([item.time for item in self.keyframes], "vector")
        _validate_vector_segments(self.keyframes)
        _validate_vector_continuity(self)
        return self


Matrix3x3V2 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class MatrixMotionKeyframeV2(FrozenModel):
    time: MotionTimeV2
    matrix_3x3: Matrix3x3V2
    interpolation: Literal["linear", "hold", "cubic_hermite"] = "linear"
    incoming_tangent_3x3: Matrix3x3V2 | None = None
    outgoing_tangent_3x3: Matrix3x3V2 | None = None

    @model_validator(mode="after")
    def validate_matrix(self) -> MatrixMotionKeyframeV2:
        if abs(_determinant_3x3(self.matrix_3x3)) <= 1e-12:
            raise ValueError("motion matrix must be invertible")
        return self


class MatrixMotionCurveV2(FrozenModel):
    keyframes: list[MatrixMotionKeyframeV2] = Field(min_length=1)
    continuity: Literal["c0", "c1"] = "c0"

    @model_validator(mode="after")
    def validate_curve(self) -> MatrixMotionCurveV2:
        _validate_motion_times([item.time for item in self.keyframes], "matrix")
        for left, right in pairwise(self.keyframes):
            if left.interpolation == "cubic_hermite" and (
                left.outgoing_tangent_3x3 is None or right.incoming_tangent_3x3 is None
            ):
                raise ValueError("matrix Hermite segment requires both endpoint tangents")
        return self


TransformOperationV2 = Literal[
    "fit",
    "scale",
    "rotate",
    "affine",
    "homography",
    "translate",
]


def _default_transform_order_v2() -> list[TransformOperationV2]:
    return [
        "fit",
        "scale",
        "rotate",
        "affine",
        "homography",
        "translate",
    ]


class DecomposedTransformMotionV2(FrozenModel):
    kind: Literal["decomposed"] = "decomposed"
    fit_mode: Literal["contain", "cover"] = "contain"
    position: Vec2MotionCurveV2
    scale: ScalarMotionCurveV2
    rotation: ScalarMotionCurveV2
    pivot: Vec2MotionCurveV2
    affine: MatrixMotionCurveV2 | None = None
    homography: MatrixMotionCurveV2 | None = None
    transform_order: list[TransformOperationV2] = Field(default_factory=_default_transform_order_v2)

    @model_validator(mode="after")
    def validate_motion(self) -> DecomposedTransformMotionV2:
        if len(self.transform_order) != len(set(self.transform_order)):
            raise ValueError("transform order operations must be unique")
        if not self.transform_order or self.transform_order[0] != "fit":
            raise ValueError("transform order must begin with fit")
        if self.transform_order[-1] != "translate":
            raise ValueError("transform order must end with translate")
        required = {"fit", "scale", "rotate", "translate"}
        if not required <= set(self.transform_order):
            raise ValueError("transform order omits a required decomposed operation")
        curve_bounds = {
            _curve_bounds(self.position.keyframes),
            _curve_bounds(self.scale.keyframes),
            _curve_bounds(self.rotation.keyframes),
            _curve_bounds(self.pivot.keyframes),
        }
        if self.affine is not None:
            curve_bounds.add(_curve_bounds(self.affine.keyframes))
        if self.homography is not None:
            curve_bounds.add(_curve_bounds(self.homography.keyframes))
        if len(curve_bounds) != 1:
            raise ValueError("all decomposed transform curves must share endpoints")
        if any(item.value <= 0 for item in self.scale.keyframes):
            raise ValueError("motion v2 scale keyframes must be positive")
        return self


class MatrixTransformMotionV2(FrozenModel):
    kind: Literal["matrix"] = "matrix"
    coordinate_space: Literal["source_pixels_to_output_pixels"] = "source_pixels_to_output_pixels"
    matrix: MatrixMotionCurveV2


TransformMotionV2 = Annotated[
    DecomposedTransformMotionV2 | MatrixTransformMotionV2,
    Field(discriminator="kind"),
]


class MotionPhaseV2(FrozenModel):
    id: str
    start: MotionTimeV2
    end: MotionTimeV2
    channels: list[Literal["position", "scale", "rotation", "pivot", "matrix"]] = Field(
        min_length=1
    )
    intent: Literal["onset", "accelerate", "twist", "decelerate", "settle", "custom"]

    @model_validator(mode="after")
    def validate_phase(self) -> MotionPhaseV2:
        if self.start.as_fraction >= self.end.as_fraction:
            raise ValueError("motion phase start must precede end")
        return self


class MotionCouplingV2(FrozenModel):
    driver_channel: Literal["position", "scale", "rotation", "pivot", "matrix"]
    follower_channels: list[Literal["position", "scale", "rotation", "pivot", "matrix"]] = Field(
        min_length=1
    )
    relation: Literal[
        "shared_progress",
        "phase_lag",
        "phase_lead",
        "explicit_matrix",
    ]
    tolerance: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_coupling(self) -> MotionCouplingV2:
        if self.driver_channel in self.follower_channels:
            raise ValueError("motion coupling driver cannot follow itself")
        if len(self.follower_channels) != len(set(self.follower_channels)):
            raise ValueError("motion coupling followers must be unique")
        return self


class MotionSamplingV2(FrozenModel):
    samples_per_frame: int = Field(default=1, ge=1, le=16)
    shutter_fraction: float = Field(default=0.0, ge=0, le=1)
    distribution: Literal["center", "uniform"] = "center"

    @model_validator(mode="after")
    def validate_sampling(self) -> MotionSamplingV2:
        if self.samples_per_frame == 1 and self.shutter_fraction != 0:
            raise ValueError("single-sample motion must have zero shutter fraction")
        if self.samples_per_frame > 1 and (
            self.shutter_fraction == 0 or self.distribution != "uniform"
        ):
            raise ValueError("multi-sample motion requires a non-zero uniform shutter interval")
        return self


class TransformEffectV2(FrozenModel):
    type: Literal["transform_v2"] = "transform_v2"
    motion_language_version: Literal["2.0.0"] = "2.0.0"
    motion: TransformMotionV2
    sampling: MotionSamplingV2 = Field(default_factory=MotionSamplingV2)
    phases: list[MotionPhaseV2] = Field(default_factory=list)
    couplings: list[MotionCouplingV2] = Field(default_factory=list)


class ColorEffect(FrozenModel):
    type: Literal["color"] = "color"
    brightness: float = Field(default=0, ge=-1, le=1)
    contrast: float = Field(default=1, ge=0, le=3)
    saturation: float = Field(default=1, ge=0, le=3)
    gamma: float = Field(default=1, gt=0, le=10)
    vignette: float = Field(default=0, ge=0, le=1)


class MaskEffect(FrozenModel):
    type: Literal["mask"] = "mask"
    mask_asset_path: str
    invert: bool = False
    feather: float = Field(default=0, ge=0, le=100)


class AudioEffect(FrozenModel):
    type: Literal["audio"] = "audio"
    gain_db: float = Field(default=0, ge=-96, le=24)
    gain_keyframes: list[ScalarKeyframe] = Field(default_factory=list)
    fade_in_frames: int = Field(default=0, ge=0)
    fade_out_frames: int = Field(default=0, ge=0)
    normalize_lufs: float | None = Field(default=None, ge=-40, le=-5)

    @model_validator(mode="after")
    def validate_gain_curve(self) -> AudioEffect:
        frames = [keyframe.frame for keyframe in self.gain_keyframes]
        if any(left >= right for left, right in pairwise(frames)):
            raise ValueError("gain keyframe frames must be strictly increasing")
        if any(not -96 <= item.value <= 24 for item in self.gain_keyframes):
            raise ValueError("gain keyframes must be between -96 and 24 dB")
        return self


class TextEffect(FrozenModel):
    type: Literal["text"] = "text"
    text: str
    x: float = Field(default=0.5, ge=0, le=1)
    y: float = Field(default=0.85, ge=0, le=1)
    font_size: int = Field(default=48, gt=0)
    color: str = "#ffffff"
    background: str | None = "#00000088"


MAX_SPEED_RATE = 64.0


class SpeedEffect(FrozenModel):
    type: Literal["speed"] = "speed"
    rate: float = Field(default=1.0, ge=0.5, le=MAX_SPEED_RATE)


class DissolveEffect(FrozenModel):
    type: Literal["dissolve"] = "dissolve"
    fade_in_frames: int = Field(default=0, ge=0)
    fade_out_frames: int = Field(default=0, ge=0)


class BlurEffect(FrozenModel):
    type: Literal["blur"] = "blur"
    sigma: float = Field(default=1.0, ge=0, le=50)


class GlowEffect(FrozenModel):
    type: Literal["glow"] = "glow"
    radius: float = Field(default=4.0, ge=0.1, le=50)
    intensity: float = Field(default=0.25, ge=0, le=1)


class BlendEffect(FrozenModel):
    type: Literal["blend"] = "blend"
    mode: Literal["normal", "multiply", "screen", "overlay", "addition"] = "normal"
    opacity: float = Field(default=1.0, ge=0, le=1)


Effect = Annotated[
    TransformEffect
    | TransformEffectV2
    | ColorEffect
    | MaskEffect
    | AudioEffect
    | TextEffect
    | SpeedEffect
    | DissolveEffect
    | BlurEffect
    | GlowEffect
    | BlendEffect,
    Field(discriminator="type"),
]


def _validate_motion_times(times: list[MotionTimeV2], label: str) -> None:
    fractions = [item.as_fraction for item in times]
    if any(left >= right for left, right in pairwise(fractions)):
        raise ValueError(f"{label} motion keyframe times must be strictly increasing")


def _curve_bounds(keyframes: list[Any]) -> tuple[Fraction, Fraction]:
    return keyframes[0].time.as_fraction, keyframes[-1].time.as_fraction


def _validate_scalar_segments(keyframes: list[ScalarMotionKeyframeV2]) -> None:
    for left, right in pairwise(keyframes):
        duration = float(right.time.as_fraction - left.time.as_fraction)
        if left.interpolation == "cubic_bezier":
            outgoing = left.outgoing_handle
            incoming = right.incoming_handle
            if outgoing is None or incoming is None:
                raise ValueError("cubic Bezier segment requires both endpoint handles")
            _validate_handle_times(
                outgoing.time_offset_frames,
                incoming.time_offset_frames,
                duration,
            )
        if left.interpolation == "cubic_hermite" and (
            left.outgoing_tangent is None or right.incoming_tangent is None
        ):
            raise ValueError("cubic Hermite segment requires both endpoint tangents")


def _validate_vector_segments(keyframes: list[Vec2MotionKeyframeV2]) -> None:
    for left, right in pairwise(keyframes):
        duration = float(right.time.as_fraction - left.time.as_fraction)
        if left.interpolation == "cubic_bezier":
            outgoing = left.outgoing_handle
            incoming = right.incoming_handle
            if outgoing is None or incoming is None:
                raise ValueError("vector cubic Bezier segment requires endpoint handles")
            _validate_handle_times(
                outgoing.time_offset_frames,
                incoming.time_offset_frames,
                duration,
            )
        if left.interpolation == "cubic_hermite" and (
            left.outgoing_tangent is None or right.incoming_tangent is None
        ):
            raise ValueError("vector cubic Hermite segment requires endpoint tangents")


def _validate_handle_times(outgoing: float, incoming: float, duration: float) -> None:
    if not 0 < outgoing < duration:
        raise ValueError("outgoing Bezier handle time must lie inside its segment")
    if not -duration < incoming < 0:
        raise ValueError("incoming Bezier handle time must lie inside its segment")
    if outgoing > duration + incoming:
        raise ValueError("Bezier handle times must not cross")


def _validate_scalar_continuity(curve: ScalarMotionCurveV2) -> None:
    if curve.continuity == "c0" or len(curve.keyframes) < 3:
        return
    for item in curve.keyframes[1:-1]:
        if (
            item.incoming_tangent is not None
            and item.outgoing_tangent is not None
            and not isclose(
                item.incoming_tangent,
                item.outgoing_tangent,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("C1 scalar curve has unequal tangents at a keyframe")
    if curve.continuity != "c2":
        return
    for index in range(1, len(curve.keyframes) - 1):
        previous = curve.keyframes[index - 1]
        current = curve.keyframes[index]
        following = curve.keyframes[index + 1]
        if (
            previous.interpolation != "cubic_hermite"
            or current.interpolation != "cubic_hermite"
            or previous.outgoing_tangent is None
            or current.incoming_tangent is None
            or current.outgoing_tangent is None
            or following.incoming_tangent is None
        ):
            raise ValueError("C2 scalar continuity requires explicit Hermite segments")
        left_duration = float(current.time.as_fraction - previous.time.as_fraction)
        right_duration = float(following.time.as_fraction - current.time.as_fraction)
        left_second = (
            -6.0 * (current.value - previous.value) / left_duration**2
            + (2.0 * previous.outgoing_tangent + 4.0 * current.incoming_tangent) / left_duration
        )
        right_second = (
            6.0 * (following.value - current.value) / right_duration**2
            - (4.0 * current.outgoing_tangent + 2.0 * following.incoming_tangent) / right_duration
        )
        if not isclose(left_second, right_second, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("C2 scalar curve has unequal second derivatives")


def _validate_vector_continuity(curve: Vec2MotionCurveV2) -> None:
    if curve.continuity == "c0" or len(curve.keyframes) < 3:
        return
    for item in curve.keyframes[1:-1]:
        incoming = item.incoming_tangent
        outgoing = item.outgoing_tangent
        if (
            incoming is not None
            and outgoing is not None
            and (
                not isclose(incoming.x, outgoing.x, rel_tol=1e-7, abs_tol=1e-9)
                or not isclose(incoming.y, outgoing.y, rel_tol=1e-7, abs_tol=1e-9)
            )
        ):
            raise ValueError("C1 vector curve has unequal tangents at a keyframe")
    if curve.continuity != "c2":
        return
    for index in range(1, len(curve.keyframes) - 1):
        previous = curve.keyframes[index - 1]
        current = curve.keyframes[index]
        following = curve.keyframes[index + 1]
        previous_tangent = previous.outgoing_tangent
        incoming = current.incoming_tangent
        outgoing = current.outgoing_tangent
        following_tangent = following.incoming_tangent
        if (
            previous.interpolation != "cubic_hermite"
            or current.interpolation != "cubic_hermite"
            or previous_tangent is None
            or incoming is None
            or outgoing is None
            or following_tangent is None
        ):
            raise ValueError("C2 vector continuity requires explicit Hermite segments")
        left_duration = float(current.time.as_fraction - previous.time.as_fraction)
        right_duration = float(following.time.as_fraction - current.time.as_fraction)
        for name in ("x", "y"):
            left_second = (
                -6.0
                * (getattr(current.value, name) - getattr(previous.value, name))
                / left_duration**2
                + (2.0 * getattr(previous_tangent, name) + 4.0 * getattr(incoming, name))
                / left_duration
            )
            right_second = (
                6.0
                * (getattr(following.value, name) - getattr(current.value, name))
                / right_duration**2
                - (4.0 * getattr(outgoing, name) + 2.0 * getattr(following_tangent, name))
                / right_duration
            )
            if not isclose(left_second, right_second, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("C2 vector curve has unequal second derivatives")


def _determinant_3x3(matrix: Matrix3x3V2) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


class Clip(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("clip"))
    asset_id: str
    timeline_range: FrameRange
    source_range: FrameRange | None = None
    role: str = "primary"
    enabled: bool = True
    pinned: bool = False
    effects: list[Effect] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    decision_refs: list[str] = Field(default_factory=list)


class Track(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("track"))
    kind: TrackKind
    name: str
    clips: list[Clip] = Field(default_factory=list)
    muted: bool = False
    locked: bool = False


class Timeline(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    frame_rate: FrameRate = Field(default_factory=FrameRate)
    duration_frames: int = Field(gt=0)
    background: str = "#000000"
    audio_sample_rate: int = Field(default=48000, gt=0)
    tracks: list[Track] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_clip_bounds(self) -> Timeline:
        for track in self.tracks:
            for clip in track.clips:
                if clip.timeline_range.end > self.duration_frames:
                    raise ValueError(
                        f"clip {clip.id} ends after timeline ({clip.timeline_range.end} > "
                        f"{self.duration_frames})"
                    )
        return self


class Intent(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    text: str = Field(min_length=1)
    scenario: Scenario
    target_duration_seconds: float | None = Field(default=None, gt=0)
    audience: str | None = None
    frame_format: str | None = None
    tempo: str | None = None
    mood: str | None = None
    required_elements: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    sound_requirements: list[str] = Field(default_factory=list)
    privacy_mode: Literal["local-only", "explicit-provider-opt-in"] = "local-only"
    automation_level: Literal["suggest", "review-before-apply", "approved-auto"] = (
        "review-before-apply"
    )
    preferences: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)


class ScreenWorkflowShotTemplate(FrozenModel):
    kind: ScreenWorkflowShotKind
    purpose: str = Field(min_length=1)
    default_scale: float = Field(ge=1.0, le=2.0)
    camera_transition_seconds: float = Field(ge=0, le=1.0)
    minimum_hold_seconds: float = Field(ge=0.5, le=10.0)
    camera_policy: str = Field(min_length=1)
    cut_policy: Literal["continuous-camera", "hard-cut"]


class ScreenWorkflowBeatInput(FrozenModel):
    narration: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    capture_action: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    shot_kind: ScreenWorkflowShotKind
    estimated_duration_seconds: float = Field(ge=0.5, le=30.0)
    operator_notes: list[str] = Field(default_factory=list)


class ScreenWorkflowBeat(ScreenWorkflowBeatInput):
    id: str = Field(default_factory=lambda: new_id("workflowbeat"))
    order: int = Field(ge=1)


class ScreenWorkflowPlan(FrozenModel):
    """Pre-capture contract; it deliberately has no source-media binding."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("workflowplan"))
    generated_at: datetime = Field(default_factory=utc_now)
    title: str = Field(min_length=1)
    script: str = Field(min_length=1)
    script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["editorial-draft", "reviewed-for-capture"]
    capture_ready: bool
    reviewed_by: str | None = None
    template_revision: Literal[
        "terminal-workflow-candidate-v1",
        "terminal-workflow-temporal-integrity-v2",
    ] = "terminal-workflow-temporal-integrity-v2"
    templates: list[ScreenWorkflowShotTemplate] = Field(min_length=7)
    beats: list[ScreenWorkflowBeat] = Field(min_length=1)
    capture_rules: list[str] = Field(min_length=1)
    framing_rules: list[str] = Field(min_length=1)
    forbidden_content: list[str] = Field(min_length=1)
    unresolved_questions: list[str] = Field(default_factory=list)
    reference_study_ids: list[str] = Field(default_factory=list)
    style_profile_written: Literal[False] = False
    source_media_bound: Literal[False] = False
    provenance: Provenance

    @model_validator(mode="after")
    def validate_plan(self) -> ScreenWorkflowPlan:
        digest = hashlib.sha256(self.script.encode("utf-8")).hexdigest()
        if digest != self.script_sha256:
            raise ValueError("script_sha256 does not match script text")
        if [item.order for item in self.beats] != list(range(1, len(self.beats) + 1)):
            raise ValueError("workflow beat order must be contiguous from one")
        if len({item.id for item in self.beats}) != len(self.beats):
            raise ValueError("workflow beat ids must be unique")
        template_kinds = [item.kind for item in self.templates]
        if set(template_kinds) != set(ScreenWorkflowShotKind):
            raise ValueError("workflow plan must carry every terminal template exactly once")
        if len(template_kinds) != len(set(template_kinds)):
            raise ValueError("workflow template kinds must be unique")
        if self.capture_ready != (self.status == "reviewed-for-capture"):
            raise ValueError("only a reviewed-for-capture plan can be capture-ready")
        if self.capture_ready and not self.reviewed_by:
            raise ValueError("capture-ready plan requires an attributable reviewer")
        return self


class ScreenWorkflowExperienceClaim(FrozenModel):
    """One owner-reviewed lesson with private evidence and a public-safe summary."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    disposition: ScreenWorkflowExperienceDisposition
    source_observation: str = Field(min_length=1)
    public_summary: str = Field(min_length=1)
    public_limitation: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    owner_review_refs: list[str] = Field(min_length=1)
    target_surfaces: list[str] = Field(min_length=1)
    public_case_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )

    @model_validator(mode="after")
    def validate_claim(self) -> ScreenWorkflowExperienceClaim:
        if self.disposition is ScreenWorkflowExperienceDisposition.ACCEPTED:
            if self.public_case_id is None:
                raise ValueError("an accepted experience claim requires a public case id")
        elif self.public_case_id is not None:
            raise ValueError("only accepted experience claims may create a public case")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("experience evidence refs must be unique")
        if len(self.owner_review_refs) != len(set(self.owner_review_refs)):
            raise ValueError("experience owner-review refs must be unique")
        if not set(self.owner_review_refs).issubset(self.evidence_refs):
            raise ValueError("experience owner-review refs must also be evidence refs")
        if len(self.target_surfaces) != len(set(self.target_surfaces)):
            raise ValueError("experience target surfaces must be unique")
        return self


class ScreenWorkflowExperienceAdmission(FrozenModel):
    """Private owner admission over one immutable external review snapshot."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    source_kind: Literal[
        "session-memory-manual-review",
        "operator-review-packet",
        "other-reviewed-evidence",
    ]
    source_record_id: str = Field(min_length=1)
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_wave_id: str = Field(min_length=1)
    recurrence: ScreenWorkflowExperienceRecurrence
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime
    review_note: str = Field(min_length=1)
    claims: list[ScreenWorkflowExperienceClaim] = Field(min_length=1)
    public_safe_projection_checked: Literal[True]
    provenance: Provenance

    @model_validator(mode="after")
    def validate_admission(self) -> ScreenWorkflowExperienceAdmission:
        claim_ids = [item.id for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("experience claim ids must be unique")
        if not any(
            item.disposition is ScreenWorkflowExperienceDisposition.ACCEPTED
            for item in self.claims
        ):
            raise ValueError("an experience admission requires at least one accepted claim")
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("experience owner review time must include a timezone")
        return self


class ScreenWorkflowExperiencePublicClaim(FrozenModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    disposition: ScreenWorkflowExperienceDisposition
    summary: str = Field(min_length=1)
    limitation: str = Field(min_length=1)
    target_surfaces: list[str] = Field(min_length=1)
    public_case_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )


class ScreenWorkflowExperiencePublicProjection(FrozenModel):
    """Source-neutral projection safe for a public candidate-knowledge packet."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    admission_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    source_kind: Literal[
        "session-memory-manual-review",
        "operator-review-packet",
        "other-reviewed-evidence",
    ]
    recurrence: ScreenWorkflowExperienceRecurrence
    truth_status: Literal["owner-reviewed-candidate"] = "owner-reviewed-candidate"
    private_evidence_retained: Literal[True] = True
    claims: list[ScreenWorkflowExperiencePublicClaim] = Field(min_length=1)


class ScreenWorkflowExperienceReceipt(FrozenModel):
    """Immutable local receipt binding private refs to a sanitized projection hash."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(pattern=r"^workflowexperience_[0-9a-f]{24}$")
    admission: ScreenWorkflowExperienceAdmission
    public_projection: ScreenWorkflowExperiencePublicProjection
    public_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stored_at: datetime
    immutable: Literal[True] = True

    @model_validator(mode="after")
    def validate_receipt(self) -> ScreenWorkflowExperienceReceipt:
        if self.public_projection.admission_id != self.admission.id:
            raise ValueError("experience projection does not identify its admission")
        expected_claims = [
            ScreenWorkflowExperiencePublicClaim(
                id=item.id,
                disposition=item.disposition,
                summary=item.public_summary,
                limitation=item.public_limitation,
                target_surfaces=item.target_surfaces,
                public_case_id=item.public_case_id,
            )
            for item in self.admission.claims
        ]
        if (
            self.public_projection.source_kind != self.admission.source_kind
            or self.public_projection.recurrence != self.admission.recurrence
            or self.public_projection.claims != expected_claims
        ):
            raise ValueError("experience projection does not match its private admission")
        payload = json.dumps(
            self.public_projection.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.public_projection_sha256:
            raise ValueError("experience public projection hash does not match")
        admission_payload = json.dumps(
            self.admission.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        admission_digest = hashlib.sha256(admission_payload).hexdigest()
        if self.id != f"workflowexperience_{admission_digest[:24]}":
            raise ValueError("experience receipt id does not match its private admission")
        if self.stored_at != self.admission.reviewed_at:
            raise ValueError("experience receipt time must equal the owner review time")
        return self


class ScreenWorkflowFocus(FrozenModel):
    center_x: float = Field(ge=0, le=1)
    center_y: float = Field(ge=0, le=1)
    scale: float = Field(ge=1.0, le=2.0)


class ScreenWorkflowRegion(FrozenModel):
    """Normalized final-frame region containing a live terminal/UI signal."""

    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> ScreenWorkflowRegion:
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("screen workflow live region must remain inside the frame")
        return self


class ScreenWorkflowLiveRegion(FrozenModel):
    """Reviewed evidence about motion that a hold must not visibly freeze."""

    id: str = Field(min_length=1)
    region: ScreenWorkflowRegion
    signal: ScreenWorkflowLiveSignal
    temporal_behavior: ScreenWorkflowTemporalBehavior
    loopable: bool = False
    loop_period_frames: int | None = Field(default=None, gt=1)
    seam_strategy: Literal["phase-match", "short-crossfade"] | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_loop_claim(self) -> ScreenWorkflowLiveRegion:
        if self.loopable and self.temporal_behavior is not ScreenWorkflowTemporalBehavior.PERIODIC:
            raise ValueError("only a reviewed periodic live region may be marked loopable")
        if self.loopable and (
            self.loop_period_frames is None or self.seam_strategy is None
        ):
            raise ValueError("a loopable live region requires a reviewed period and seam strategy")
        if not self.loopable and (
            self.loop_period_frames is not None or self.seam_strategy is not None
        ):
            raise ValueError("non-loopable live region cannot carry loop synthesis fields")
        return self


class ScreenWorkflowTemporalIntent(FrozenModel):
    """Reviewed pacing and live-state meaning for one bound source segment."""

    pacing_role: ScreenWorkflowPacingRole = ScreenWorkflowPacingRole.SEMANTIC_ACTION
    hold_policy: Literal[
        "none",
        "natural-source",
        "rebalance-adjacent-speeds",
    ] = "none"
    live_regions: list[ScreenWorkflowLiveRegion] = Field(default_factory=list)
    protected_static_regions: list[ScreenWorkflowRegion] = Field(default_factory=list)
    speed_override_reason: str | None = None

    @model_validator(mode="after")
    def validate_live_regions(self) -> ScreenWorkflowTemporalIntent:
        ids = [item.id for item in self.live_regions]
        if len(ids) != len(set(ids)):
            raise ValueError("screen workflow live region ids must be unique")
        return self


SCREEN_WORKFLOW_RECOMMENDED_MAX_SPEED = 24.0
SCREEN_WORKFLOW_MAX_CONTIGUOUS_TIER_RATIO = 4.0


class ScreenWorkflowSourceBinding(FrozenModel):
    beat_id: str
    segment_order: int = Field(default=1, ge=1)
    continuity_group: str | None = Field(default=None, min_length=1)
    source_range: FrameRange
    timeline_duration_frames: int = Field(gt=0)
    shot_kind: ScreenWorkflowShotKind
    focus_start: ScreenWorkflowFocus = Field(
        default_factory=lambda: ScreenWorkflowFocus(center_x=0.5, center_y=0.5, scale=1.0)
    )
    focus_end: ScreenWorkflowFocus = Field(
        default_factory=lambda: ScreenWorkflowFocus(center_x=0.5, center_y=0.5, scale=1.0)
    )
    speed: float = Field(default=1.0, ge=0.5, le=MAX_SPEED_RATE)
    temporal_intent: ScreenWorkflowTemporalIntent = Field(
        default_factory=ScreenWorkflowTemporalIntent
    )
    caption_text: str | None = None
    include_source_audio: bool = False
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_duration(self) -> ScreenWorkflowSourceBinding:
        expected = max(1, round(self.timeline_duration_frames * self.speed))
        if self.source_range.duration != expected:
            raise ValueError(
                "source range duration must equal timeline duration multiplied by speed"
            )
        if (
            self.temporal_intent.pacing_role is ScreenWorkflowPacingRole.PROMPT_ENTRY
            and not isclose(self.speed, 1.0, abs_tol=1e-9)
        ):
            raise ValueError("prompt entry must remain at real-time speed")
        if (
            self.temporal_intent.hold_policy == "natural-source"
            and not isclose(self.speed, 1.0, abs_tol=1e-9)
        ):
            raise ValueError("a natural-source hold must remain at real-time speed")
        if (
            self.speed > SCREEN_WORKFLOW_RECOMMENDED_MAX_SPEED
            and not self.temporal_intent.speed_override_reason
        ):
            raise ValueError(
                "screen workflow speed above 24x requires an explicit perceptual-risk override"
            )
        return self


class ScreenWorkflowVoiceoverCue(FrozenModel):
    order: int = Field(ge=1)
    beat_id: str = Field(min_length=1)
    timeline_range: FrameRange
    spoken_range: FrameRange | None = None
    transcript_text: str | None = None
    confidence: float = Field(ge=0, le=1)
    alignment_basis: Literal[
        "transcript-segments",
        "silence-weighted",
        "human-reviewed",
    ]
    reviewer_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_spoken_range(self) -> ScreenWorkflowVoiceoverCue:
        if self.spoken_range is not None and (
            self.spoken_range.start < self.timeline_range.start
            or self.spoken_range.end > self.timeline_range.end
        ):
            raise ValueError("voiceover spoken range must lie inside its timeline range")
        return self


class ScreenWorkflowVoiceoverTiming(FrozenModel):
    """Reviewed narration timing between a capture plan and the final edit binding."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("workflowvoiceover"))
    generated_at: datetime = Field(default_factory=utc_now)
    plan_id: str
    script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_duration_seconds: float = Field(gt=0)
    timeline_frame_rate: FrameRate
    duration_frames: int = Field(gt=0)
    status: Literal["alignment-draft", "reviewed-for-edit"]
    edit_ready: bool
    draft_alignment_method: Literal["transcript-segments", "silence-weighted"]
    cues: list[ScreenWorkflowVoiceoverCue] = Field(min_length=1)
    analysis_evidence_refs: list[str] = Field(min_length=1)
    normalization_target_lufs: float = Field(default=-16.0, ge=-40, le=-5)
    reviewed_by: str | None = None
    review_note: str | None = None
    supersedes_timing_id: str | None = None
    source_media_bound: Literal[True] = True
    provenance: Provenance

    @model_validator(mode="after")
    def validate_timing(self) -> ScreenWorkflowVoiceoverTiming:
        if not isclose(self.normalization_target_lufs, -16.0, abs_tol=1e-9):
            raise ValueError("screen workflow voiceover target must remain -16 LUFS")
        if self.edit_ready != (self.status == "reviewed-for-edit"):
            raise ValueError("only reviewed voiceover timing can be edit-ready")
        if self.edit_ready and (not self.reviewed_by or not self.review_note):
            raise ValueError("edit-ready voiceover timing requires an attributable review")
        if self.edit_ready and self.supersedes_timing_id is None:
            raise ValueError("reviewed voiceover timing must supersede its draft")
        if not self.edit_ready and any(
            item is not None
            for item in (self.reviewed_by, self.review_note, self.supersedes_timing_id)
        ):
            raise ValueError("voiceover timing draft cannot carry review fields")
        if [item.order for item in self.cues] != list(range(1, len(self.cues) + 1)):
            raise ValueError("voiceover cue order must be contiguous from one")
        beat_ids = [item.beat_id for item in self.cues]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("voiceover timing must name each beat at most once")
        if self.cues[0].timeline_range.start != 0:
            raise ValueError("voiceover timing must begin at frame zero")
        for left, right in pairwise(self.cues):
            if left.timeline_range.end != right.timeline_range.start:
                raise ValueError("voiceover cue ranges must be contiguous")
        if self.cues[-1].timeline_range.end != self.duration_frames:
            raise ValueError("voiceover cue ranges must cover the complete narration")
        if self.edit_ready and any(
            item.alignment_basis != "human-reviewed" for item in self.cues
        ):
            raise ValueError("reviewed voiceover timing requires human-reviewed cues")
        return self


class ScreenWorkflowEditSpec(FrozenModel):
    """Human-reviewed binding between a capture plan and one immutable recording."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("workfloweditspec"))
    generated_at: datetime = Field(default_factory=utc_now)
    plan_id: str
    script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_id: str
    bindings: list[ScreenWorkflowSourceBinding] = Field(min_length=1)
    voiceover_timing: ScreenWorkflowVoiceoverTiming | None = None
    human_reviewed: Literal[True]
    reviewed_by: str = Field(min_length=1)
    review_note: str = Field(min_length=1)
    provenance: Provenance

    @model_validator(mode="after")
    def validate_bindings(self) -> ScreenWorkflowEditSpec:
        seen_beats: set[str] = set()
        prior_beat: str | None = None
        expected_segment_order = 1
        closed_groups: set[str] = set()
        prior_group: str | None = None
        for index, item in enumerate(self.bindings):
            if item.beat_id != prior_beat:
                if item.beat_id in seen_beats:
                    raise ValueError("workflow beat segments must remain contiguous")
                seen_beats.add(item.beat_id)
                expected_segment_order = 1
            if item.segment_order != expected_segment_order:
                raise ValueError("workflow beat segment_order must be contiguous from one")
            expected_segment_order += 1
            prior_beat = item.beat_id

            group = item.continuity_group
            if group is None:
                if prior_group is not None:
                    closed_groups.add(prior_group)
                prior_group = None
                continue
            if group != prior_group:
                if group in closed_groups:
                    raise ValueError("screen workflow continuity groups must not reappear")
                if prior_group is not None:
                    closed_groups.add(prior_group)
            if index > 0 and group == prior_group:
                previous = self.bindings[index - 1]
                if previous.source_range.end != item.source_range.start:
                    raise ValueError("source-contiguous speed tiers must not omit source frames")
                if previous.focus_end != item.focus_start:
                    raise ValueError(
                        "source-contiguous speed tiers must preserve camera continuity"
                    )
                tier_ratio = max(previous.speed, item.speed) / min(previous.speed, item.speed)
                if tier_ratio > SCREEN_WORKFLOW_MAX_CONTIGUOUS_TIER_RATIO:
                    raise ValueError(
                        "source-contiguous speed tiers must ramp through intermediate rates"
                    )
            prior_group = group
        return self


ReferenceWorkflowContentClass = Literal[
    "terminal",
    "ide",
    "browser-workflow",
    "workflow-ui",
    "diagram",
    "human",
    "title-card",
    "other",
]


class ReferenceWorkflowRangePlan(FrozenModel):
    id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    content_class: ReferenceWorkflowContentClass
    decision: Literal["include", "exclude"]
    human_present: bool
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> ReferenceWorkflowRangePlan:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("reference workflow range must have positive duration")
        allowed = {"terminal", "ide", "browser-workflow", "workflow-ui", "diagram"}
        if self.decision == "include" and (
            self.human_present or self.content_class not in allowed
        ):
            raise ValueError("included workflow ranges must be human-free workflow content")
        return self


class ReferenceWorkflowSourcePlan(FrozenModel):
    alias: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    path: str = Field(min_length=1)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranges: list[ReferenceWorkflowRangePlan] = Field(min_length=1)
    reviewed_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ranges(self) -> ReferenceWorkflowSourcePlan:
        ordered = sorted(self.ranges, key=lambda item: (item.start_seconds, item.end_seconds))
        if any(left.end_seconds > right.start_seconds for left, right in pairwise(ordered)):
            raise ValueError("reference workflow ranges must not overlap")
        if not any(item.decision == "include" for item in ordered):
            raise ValueError("each workflow reference must contain an included range")
        if len({item.id for item in ordered}) != len(ordered):
            raise ValueError("reference workflow range ids must be unique per source")
        return self


class ReferenceWorkflowCandidateRule(FrozenModel):
    statement: str = Field(min_length=1)
    evidence_range_refs: list[str] = Field(min_length=1)
    authority: Literal["candidate-observation"] = "candidate-observation"


class ReferenceWorkflowStudyPlan(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("workflowstudyplan"))
    generated_at: datetime = Field(default_factory=utc_now)
    title: str = Field(min_length=1)
    sources: list[ReferenceWorkflowSourcePlan] = Field(min_length=1)
    candidate_rules: list[ReferenceWorkflowCandidateRule] = Field(default_factory=list)
    manual_review_complete: Literal[True]
    reference_media_role: Literal["analysis-and-evaluation-only"] = (
        "analysis-and-evaluation-only"
    )
    provenance: Provenance

    @model_validator(mode="after")
    def validate_sources(self) -> ReferenceWorkflowStudyPlan:
        aliases = [item.alias for item in self.sources]
        if len(aliases) != len(set(aliases)):
            raise ValueError("reference workflow source aliases must be unique")
        refs = {
            f"{source.alias}:{range_plan.id}"
            for source in self.sources
            for range_plan in source.ranges
            if range_plan.decision == "include"
        }
        for rule in self.candidate_rules:
            if not set(rule.evidence_range_refs) <= refs:
                raise ValueError("candidate rule cites an unknown workflow range")
        return self


class ReferenceWorkflowRangeResult(FrozenModel):
    range_id: str
    start_seconds: float
    end_seconds: float
    content_class: ReferenceWorkflowContentClass
    decision: Literal["include", "exclude"]
    human_present: bool
    sample_count: int = Field(ge=0)
    mean_frame_change: float = Field(ge=0)
    p90_frame_change: float = Field(ge=0)
    static_fraction: float = Field(ge=0, le=1)
    rationale: str


class ReferenceWorkflowSourceResult(FrozenModel):
    alias: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    metadata: MediaMetadata
    ranges: list[ReferenceWorkflowRangeResult]
    contact_sheet: str
    contact_sheet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path_persisted: Literal[False] = False


class ReferenceWorkflowStudyReceipt(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("workflowstudy"))
    generated_at: datetime = Field(default_factory=utc_now)
    plan_id: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readiness_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sources: list[ReferenceWorkflowSourceResult]
    candidate_rules: list[ReferenceWorkflowCandidateRule]
    reference_media_copied: Literal[False] = False
    eligible_as_project_assets: Literal[False] = False
    eligible_as_render_inputs: Literal[False] = False
    style_profile_written: Literal[False] = False
    manual_review_complete: Literal[True] = True
    overall: Literal["pass"] = "pass"
    provenance: Provenance


class EditorialBriefRevision(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("brief"))
    project_id: str
    revision: int = Field(ge=1)
    intent: Intent
    rationale: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)


class PatchOperation(FrozenModel):
    op: Literal["add", "remove", "replace", "test"]
    path: str = Field(pattern=r"^/")
    value: Any = None


class EditPatch(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("patch"))
    project_id: str
    base_version_id: str
    operations: list[PatchOperation]
    inverse_operations: list[PatchOperation] = Field(default_factory=list)
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class PatchPreview(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("preview"))
    project_id: str
    base_version_id: str
    natural_language_command: str
    patch: EditPatch
    summary: str
    warnings: list[str] = Field(default_factory=list)
    requires_confirmation: Literal[True] = True


class MotionCorrectionContextV2(FrozenModel):
    """Evidence-bounded phase context used to interpret correction language."""

    duration_frames: int = Field(gt=1)
    onset_frame: int = Field(ge=0)
    twist_start_frame: int = Field(ge=0)
    twist_peak_frame: int = Field(ge=0)
    twist_end_frame: int = Field(ge=0)
    settle_frame: int = Field(ge=0)
    pivot_identifiable: bool

    @model_validator(mode="after")
    def validate_boundaries(self) -> MotionCorrectionContextV2:
        ordered = [
            self.onset_frame,
            self.twist_start_frame,
            self.twist_peak_frame,
            self.twist_end_frame,
            self.settle_frame,
        ]
        if any(left >= right for left, right in pairwise(ordered)):
            raise ValueError("motion correction phase context must be strictly ordered")
        if self.settle_frame >= self.duration_frames:
            raise ValueError("motion correction settle frame is outside the timeline")
        return self


class AdjustTangentCorrectionV2(FrozenModel):
    kind: Literal["adjust_tangent"] = "adjust_tangent"
    channel: Literal["position", "scale", "rotation", "matrix"]
    anchor_frame: int = Field(ge=0)
    strength: float = Field(gt=0, le=1)
    affected_range: FrameRange

    @model_validator(mode="after")
    def validate_anchor(self) -> AdjustTangentCorrectionV2:
        if not self.affected_range.start <= self.anchor_frame < self.affected_range.end:
            raise ValueError("tangent anchor must lie in its affected range")
        return self


class MovePhaseBoundaryCorrectionV2(FrozenModel):
    kind: Literal["move_phase_boundary"] = "move_phase_boundary"
    boundary: Literal["onset", "twist_start", "twist_peak", "twist_end", "settle"]
    original_frame: int = Field(ge=0)
    delta_frames: int = Field(ge=-24, le=24)
    affected_range: FrameRange

    @model_validator(mode="after")
    def validate_boundary(self) -> MovePhaseBoundaryCorrectionV2:
        if self.delta_frames == 0:
            raise ValueError("phase-boundary correction must move at least one frame")
        if not self.affected_range.start < self.original_frame < self.affected_range.end - 1:
            raise ValueError("phase boundary must be internal to the affected range")
        target = self.original_frame + self.delta_frames
        if not self.affected_range.start < target < self.affected_range.end - 1:
            raise ValueError("moved phase boundary must remain internal to the affected range")
        return self


class RetimeSettleCorrectionV2(FrozenModel):
    kind: Literal["retime_settle"] = "retime_settle"
    direction: Literal["earlier", "later"]
    strength: float = Field(gt=0, le=0.75)
    affected_range: FrameRange


class AdjustPivotCurveCorrectionV2(FrozenModel):
    kind: Literal["adjust_pivot_curve"] = "adjust_pivot_curve"
    delta_x: float = Field(ge=-1, le=1)
    delta_y: float = Field(ge=-1, le=1)
    affected_range: FrameRange

    @model_validator(mode="after")
    def validate_delta(self) -> AdjustPivotCurveCorrectionV2:
        if self.delta_x == 0 and self.delta_y == 0:
            raise ValueError("pivot correction requires a non-zero delta")
        return self


class AdjustRotationCouplingCorrectionV2(FrozenModel):
    kind: Literal["adjust_rotation_coupling"] = "adjust_rotation_coupling"
    lag_delta_frames: int = Field(ge=-12, le=12)
    smoothing_strength: float = Field(gt=0, le=1)
    affected_range: FrameRange


class ChangeEasingCorrectionV2(FrozenModel):
    kind: Literal["change_easing"] = "change_easing"
    channel: Literal["all", "position", "scale", "rotation", "matrix"]
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"]
    strength: float = Field(gt=0, le=1)
    affected_range: FrameRange


MotionCorrectionOperationV2 = Annotated[
    AdjustTangentCorrectionV2
    | MovePhaseBoundaryCorrectionV2
    | RetimeSettleCorrectionV2
    | AdjustPivotCurveCorrectionV2
    | AdjustRotationCouplingCorrectionV2
    | ChangeEasingCorrectionV2,
    Field(discriminator="kind"),
]


class MotionCorrectionDiffV2(FrozenModel):
    """Compact typed diff; canonical patch value is generated only on approval."""

    target_path: str = Field(pattern=r"^/")
    before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    affected_range: FrameRange
    changed_frame_count: int = Field(ge=0)
    maximum_matrix_coefficient_delta: float = Field(ge=0)
    endpoint_preserved: bool
    operation_kinds: list[str] = Field(min_length=1)


class MotionCorrectionProposalItemV2(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("motioncorrection"))
    operation: MotionCorrectionOperationV2
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    executable: bool
    blockers: list[str] = Field(default_factory=list)
    diff: MotionCorrectionDiffV2 | None = None

    @model_validator(mode="after")
    def validate_executable_preview(self) -> MotionCorrectionProposalItemV2:
        if self.executable and (self.blockers or self.diff is None):
            raise ValueError("executable correction requires a diff and no blockers")
        if not self.executable and (not self.blockers or self.diff is not None):
            raise ValueError(
                "non-executable correction requires blockers and cannot carry a diff"
            )
        return self


class MotionCorrectionProposalSetV2(FrozenModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("motionproposal"))
    generated_at: datetime = Field(default_factory=utc_now)
    project_id: str
    workspace_id: str
    base_version_id: str
    target_effect_path: str = Field(pattern=r"^/")
    base_effect_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    natural_language_command: str | None = None
    items: list[MotionCorrectionProposalItemV2] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    provenance: Provenance
    requires_confirmation: Literal[True] = True

    @model_validator(mode="after")
    def validate_proposal_items(self) -> MotionCorrectionProposalSetV2:
        item_ids = [item.id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("motion correction proposal item ids must be unique")
        for item in self.items:
            if item.diff is not None and (
                item.diff.target_path != self.target_effect_path
                or item.diff.before_sha256 != self.base_effect_sha256
            ):
                raise ValueError(
                    "motion correction diff must join the proposal target and base"
                )
        return self


class MotionCorrectionReviewDecisionV2(FrozenModel):
    correction_id: str
    decision: Literal["approve", "reject"]
    rationale: str = Field(min_length=1)


class MotionCorrectionReviewV2(FrozenModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("motionreview"))
    generated_at: datetime = Field(default_factory=utc_now)
    project_id: str
    workspace_id: str
    proposal_id: str
    base_version_id: str
    decisions: list[MotionCorrectionReviewDecisionV2] = Field(min_length=1)
    actor: Literal["human-editor"] = "human-editor"
    applied_patch_id: str | None = None
    created_version_id: str | None = None
    human_evidence_id: str
    outcome: Literal["applied", "rejected"]
    provenance: Provenance

    @model_validator(mode="after")
    def validate_review(self) -> MotionCorrectionReviewV2:
        ids = [item.correction_id for item in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("motion correction review decisions must be unique")
        applied = self.outcome == "applied"
        has_patch = self.applied_patch_id is not None
        has_version = self.created_version_id is not None
        if applied and (not has_patch or not has_version):
            raise ValueError("applied correction review requires patch and version ids")
        if not applied and (has_patch or has_version):
            raise ValueError("rejected correction review cannot carry patch or version ids")
        has_approval = any(item.decision == "approve" for item in self.decisions)
        if applied != has_approval:
            raise ValueError(
                "review outcome must match whether at least one correction was approved"
            )
        return self


class ReferenceMotionHumanCorrectionV2(FrozenModel):
    """Explicit human interpretation evidence; it never mutates the frozen spec."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    workspace_id: str
    proposal_id: str
    phase_markers: dict[str, int]
    pivot_assessment: Literal["identified", "unidentifiable", "uncertain"]
    accepted_correction_ids: list[str]
    rejected_correction_ids: list[str]
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_human_interpretation(self) -> ReferenceMotionHumanCorrectionV2:
        marker_keys = (
            "onset",
            "twist_start",
            "twist_peak",
            "twist_end",
            "settle",
        )
        if set(self.phase_markers) != set(marker_keys):
            raise ValueError("human correction must preserve all canonical phase markers")
        markers = [self.phase_markers[key] for key in marker_keys]
        if any(left >= right for left, right in pairwise(markers)):
            raise ValueError("human-corrected phase markers must be strictly ordered")
        accepted = self.accepted_correction_ids
        rejected = self.rejected_correction_ids
        if (
            len(accepted) != len(set(accepted))
            or len(rejected) != len(set(rejected))
            or set(accepted) & set(rejected)
            or not accepted + rejected
        ):
            raise ValueError(
                "human correction decisions must be unique, disjoint, and non-empty"
            )
        return self


class RationaleItem(FrozenModel):
    claim: str
    evidence_refs: list[str]
    confidence: float = Field(ge=0, le=1)


class Treatment(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("treatment"))
    project_id: str
    base_version_id: str
    scenario: Scenario
    title: str
    summary: str
    structure: list[str] = Field(default_factory=list)
    duration_frames: int | None = Field(default=None, gt=0)
    used_asset_ids: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    expected_style: str | None = None
    rationale: list[RationaleItem]
    patch: EditPatch
    alternatives: list[str] = Field(default_factory=list)
    planner: Provenance
    decision_graph_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ProjectVersion(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("version"))
    project_id: str
    parent_version_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    message: str
    timeline: Timeline
    accepted_treatment_id: str | None = None
    applied_patch_id: str | None = None
    decision_graph_id: str | None = None


class DecisionAlternative(FrozenModel):
    id: str = Field(default_factory=lambda: new_id("alternative"))
    description: str
    tradeoffs: list[str] = Field(default_factory=list)
    rejected_reason: str | None = None


class EditorialDecision(FrozenModel):
    """One explainable and reversible semantic editing decision."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("decision"))
    operation_type: str
    source_asset_ids: list[str] = Field(default_factory=list)
    source_ranges: dict[str, list[FrameRange]] = Field(default_factory=dict)
    timeline_ranges: list[FrameRange] = Field(default_factory=list)
    intent: str
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    alternatives: list[DecisionAlternative] = Field(default_factory=list)
    origin: DecisionOrigin
    approval: DecisionApproval = DecisionApproval.PROPOSED
    patch_id: str
    reversible_operations: list[PatchOperation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class DecisionEdge(FrozenModel):
    source_decision_id: str
    target_decision_id: str
    relation: Literal["depends_on", "supports", "conflicts_with", "alternative_to"]


class EditorialDecisionGraph(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("decisiongraph"))
    project_id: str
    base_version_id: str | None = None
    version_id: str | None = None
    treatment_id: str | None = None
    parent_graph_id: str | None = None
    state: Literal["proposed", "approved", "rejected", "superseded"]
    decisions: list[EditorialDecision] = Field(default_factory=list)
    edges: list[DecisionEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_graph_links(self) -> EditorialDecisionGraph:
        ids = [decision.id for decision in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("decision ids must be unique within a graph")
        known = set(ids)
        for edge in self.edges:
            if edge.source_decision_id not in known or edge.target_decision_id not in known:
                raise ValueError("decision edge points outside the graph")
        if self.state == "approved":
            if self.decisions and self.version_id is None:
                raise ValueError("approved decision graph requires a version id")
            if any(
                decision.approval is not DecisionApproval.APPROVED
                or not decision.reversible_operations
                for decision in self.decisions
            ):
                raise ValueError("approved decisions require approval and reversible operations")
        return self


class DecisionLogEntry(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("decisionevent"))
    project_id: str
    graph_id: str
    decision_id: str | None = None
    action: Literal["proposed", "approved", "rejected", "superseded", "reverted"]
    actor: DecisionOrigin
    version_id: str | None = None
    rationale: str
    created_at: datetime = Field(default_factory=utc_now)


class DecisionLog(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    project_id: str
    entries: list[DecisionLogEntry] = Field(default_factory=list)


class StyleConfirmation(FrozenModel):
    preference: str
    value: Any
    confirmed_at: datetime = Field(default_factory=utc_now)
    source: Literal["explicit-user-confirmation"] = "explicit-user-confirmation"


class StyleProfile(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("style"))
    scope: str
    explicit_opt_in: Literal[True]
    confirmations: list[StyleConfirmation] = Field(min_length=1)
    updated_at: datetime = Field(default_factory=utc_now)


class CameraMotionRecipe(FrozenModel):
    duration_frames: int = Field(gt=0)
    frame_rate: FrameRate
    aspect_width: int = Field(gt=0)
    aspect_height: int = Field(gt=0)
    fit_mode: Literal["contain", "cover"]
    position_mode: Literal["canvas_center", "offset"]
    position: list[Vec2Keyframe]
    scale: list[ScalarKeyframe]
    rotation: list[ScalarKeyframe]

    @model_validator(mode="after")
    def validate_complete_curves(self) -> CameraMotionRecipe:
        for name, curve in (
            ("position", self.position),
            ("scale", self.scale),
            ("rotation", self.rotation),
        ):
            if not curve or curve[0].frame != 0:
                raise ValueError(f"{name} curve must begin at frame zero")
            if curve[-1].frame != self.duration_frames - 1:
                raise ValueError(f"{name} curve must include the final frame")
        return self


class TechniqueApplicationRecord(FrozenModel):
    case_id: str
    source_class: str
    outcome: Literal["pass", "fail", "warn"]
    evidence_path: str
    metrics: dict[str, float] = Field(default_factory=dict)


class TechniqueCurveModelV2(FrozenModel):
    """Source-neutral temporal meaning retained by a transferable technique."""

    kind: Literal["dense_phase_compensated_c1_similarity_v2"] = (
        "dense_phase_compensated_c1_similarity_v2"
    )
    temporal_sampling: Literal["one_normalized_sample_per_output_frame"] = (
        "one_normalized_sample_per_output_frame"
    )
    interpolation: Literal["cubic_hermite_c1"] = "cubic_hermite_c1"
    position_space: Literal["normalized_output_center"] = "normalized_output_center"
    scale_space: Literal["relative_to_contain_fit"] = "relative_to_contain_fit"
    rotation_space: Literal["unwrapped_degrees"] = "unwrapped_degrees"
    pivot_policy: Literal["fixed_normalized_source_center"] = "fixed_normalized_source_center"
    maximum_interpolation_overshoot_fraction: float = Field(default=0.01, ge=0)
    transform_order: list[TransformOperationV2] = Field(
        default_factory=_default_transform_order_v2
    )


class TechniquePhaseModelV2(FrozenModel):
    """Portable editorial phases without retaining reference identities."""

    onset_frame: int = Field(ge=0)
    twist_start_frame: int = Field(ge=0)
    twist_peak_frame: int = Field(ge=0)
    twist_end_frame: int = Field(ge=0)
    settle_frame: int = Field(ge=0)
    center_phase_relation: Literal["center_lags"]
    scale_rotation_relation: Literal["synchronized"]
    scale_rotation_progress_tolerance: float = Field(ge=0)
    maximum_center_phase_offset: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> TechniquePhaseModelV2:
        if not (
            self.onset_frame
            <= self.twist_start_frame
            <= self.twist_peak_frame
            <= self.twist_end_frame
            <= self.settle_frame
        ):
            raise ValueError("portable technique phases must be ordered")
        return self


class TechniqueResolutionConstraintV2(FrozenModel):
    """Fail-closed pixel-density guard evaluated on the canonical canvas."""

    measurement: Literal["maximum_output_pixels_per_source_pixel_after_contain"] = (
        "maximum_output_pixels_per_source_pixel_after_contain"
    )
    maximum_effective_upscale: float = Field(gt=0)
    violation_policy: Literal["refuse_without_render"] = "refuse_without_render"
    refusal_code: Literal["insufficient_source_resolution"] = (
        "insufficient_source_resolution"
    )


class PortableCameraMotionRecipeV2(FrozenModel):
    """Dense normalized similarity motion that is independent of source dimensions."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    duration_frames: int = Field(gt=1)
    frame_rate: FrameRate
    aspect_width: int = Field(gt=0)
    aspect_height: int = Field(gt=0)
    fit_mode: Literal["contain"] = "contain"
    aspect_adaptation: Literal["normalized_contain_source_aspect_neutral"] = (
        "normalized_contain_source_aspect_neutral"
    )
    centers: list[tuple[float, float]] = Field(min_length=2)
    scales_relative_to_contain: list[float] = Field(min_length=2)
    rotations_degrees: list[float] = Field(min_length=2)
    pivot: tuple[float, float] = (0.5, 0.5)
    sampling: MotionSamplingV2 = Field(default_factory=MotionSamplingV2)

    @model_validator(mode="after")
    def validate_dense_recipe(self) -> PortableCameraMotionRecipeV2:
        lengths = {
            len(self.centers),
            len(self.scales_relative_to_contain),
            len(self.rotations_degrees),
            self.duration_frames,
        }
        if len(lengths) != 1:
            raise ValueError("portable motion channels must contain one sample per frame")
        if any(value <= 0 for value in self.scales_relative_to_contain):
            raise ValueError("portable motion scales must be positive")
        if self.pivot != (0.5, 0.5):
            raise ValueError("portable v2 recipe requires the identifiable fixed-pivot gauge")
        return self


class TechniqueCompositionEvidenceV2(FrozenModel):
    """Explicit composition requirement used by the applicability boundary."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    source_class: str = Field(min_length=1)
    single_global_transform_sufficient: bool
    independent_layer_motion_required: bool
    authority: Literal["fixture_ground_truth", "human_assertion", "analyzer_proposal"]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_composition_claim(self) -> TechniqueCompositionEvidenceV2:
        if (
            self.single_global_transform_sufficient
            == self.independent_layer_motion_required
        ):
            raise ValueError(
                "composition evidence must choose either one global transform "
                "or independent layer motion"
            )
        return self


class TechniquePacket(FrozenModel):
    schema_version: Literal["1.0.0", "2.0.0"] = SCHEMA_VERSION
    id: str
    revision: int = Field(default=1, gt=0)
    title: str
    intent: str
    applicability: list[str]
    contraindications: list[str]
    required_evidence: list[str]
    allowed_operations: list[str]
    rules: list[str]
    heuristics: list[str]
    creative_variants: list[str]
    constraints: list[str]
    failure_modes: list[str]
    qc: list[str]
    positive_examples: list[str]
    negative_examples: list[str]
    application_history: list[TechniqueApplicationRecord]
    status: Literal["candidate", "evaluated", "canonical"] = "candidate"
    camera_motion: CameraMotionRecipe | None = None
    curve_model_v2: TechniqueCurveModelV2 | None = None
    phase_model_v2: TechniquePhaseModelV2 | None = None
    resolution_constraint_v2: TechniqueResolutionConstraintV2 | None = None
    portable_camera_motion_v2: PortableCameraMotionRecipeV2 | None = None
    promotion_questions: list[str]
    provenance: Provenance

    @model_validator(mode="after")
    def validate_generation_contract(self) -> TechniquePacket:
        v2_fields = (
            self.curve_model_v2,
            self.phase_model_v2,
            self.resolution_constraint_v2,
            self.portable_camera_motion_v2,
        )
        if self.schema_version == "1.0.0":
            if any(item is not None for item in v2_fields):
                raise ValueError("v1 technique packet cannot carry v2 motion fields")
            return self
        if self.revision < 3:
            raise ValueError("v2 technique packet requires revision three or later")
        if self.camera_motion is not None:
            raise ValueError("v2 technique packet must not retain the legacy camera recipe")
        if any(item is None for item in v2_fields):
            raise ValueError("v2 technique packet requires all portable motion contracts")
        recipe = self.portable_camera_motion_v2
        phase = self.phase_model_v2
        assert recipe is not None and phase is not None
        if phase.settle_frame >= recipe.duration_frames:
            raise ValueError("portable technique phase is outside the motion duration")
        return self


class ProjectManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("project"))
    name: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    intent: Intent
    brief_revision_ids: list[str] = Field(default_factory=list)
    assets: list[str] = Field(default_factory=list)
    versions: list[str] = Field(default_factory=list)
    current_version_id: str | None = None
    sealed_reference_hashes: list[str] = Field(default_factory=list)


class RenderProfile(FrozenModel):
    name: Literal["preview", "final"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p"
    crf: int = Field(ge=0, le=51)
    preset: str
    audio_bitrate: str = "192k"


class CompositorFrameV2(FrozenModel):
    output_frame: int = Field(ge=0)
    timeline_frame: int = Field(ge=0)
    matrices_3x3: list[Matrix3x3V2] = Field(min_length=1)


class CompositorJobV2(FrozenModel):
    motion_language_version: Literal["2.0.0"] = "2.0.0"
    clip_id: str
    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: FrameRate
    frames: list[CompositorFrameV2] = Field(min_length=1)
    interpolation: Literal["linear", "cubic", "lanczos4"] = "lanczos4"
    encoder_command: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_frames(self) -> CompositorJobV2:
        expected = list(range(len(self.frames)))
        actual = [item.output_frame for item in self.frames]
        if actual != expected:
            raise ValueError("compositor output frames must be contiguous from zero")
        sample_counts = {len(item.matrices_3x3) for item in self.frames}
        if len(sample_counts) != 1:
            raise ValueError("compositor frames must use one deterministic sample count")
        return self


class RenderPlan(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    project_id: str
    version_id: str
    command: list[str]
    filter_graph: str
    output_path: Path
    input_hashes: list[str]
    profile: RenderProfile
    frame_range: FrameRange | None = None
    compositor_jobs: list[CompositorJobV2] = Field(default_factory=list)
    compiler: Provenance


class JobReceipt(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("job"))
    kind: str
    project_id: str
    version_id: str | None = None
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    command: list[str] = Field(default_factory=list)
    input_hashes: list[str] = Field(default_factory=list)
    output_paths: list[str] = Field(default_factory=list)
    output_hashes: list[str] = Field(default_factory=list)
    error: str | None = None


class CheckResult(FrozenModel):
    id: str
    status: Literal["pass", "fail", "warn", "skip"]
    summary: str
    measured: dict[str, Any] = Field(default_factory=dict)


class AILatencyExpectation(FrozenModel):
    service_class: Literal["interactive", "nearline", "batch"]
    target_milliseconds: int = Field(gt=0)
    hard_timeout_seconds: float = Field(gt=0)


class AIFallbackContract(FrozenModel):
    strategy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    behavior: Literal["deterministic", "localized-failure", "disabled"]
    preserves_baseline: Literal[True] = True
    description: str = Field(min_length=1)


class AIFailureContract(FrozenModel):
    localized: Literal[True] = True
    sibling_evidence_survives: Literal[True] = True
    retryable_codes: list[str] = Field(default_factory=list)
    non_retryable_codes: list[str] = Field(default_factory=list)


class AIHealthContract(FrozenModel):
    probe_required: Literal[True] = True
    maximum_age_seconds: int = Field(gt=0)
    expected_protocol_version: str = Field(min_length=1)
    model_revision_required: bool
    fail_closed: Literal[True] = True


class AICapabilityDeclaration(FrozenModel):
    """Tracked, path-free product declaration for one optional AI capability."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    capability_id: str = Field(pattern=r"^(speech|vision|editorial)\.[a-z0-9][a-z0-9-]*$")
    alias: str = Field(
        pattern=r"^local-ai://(speech|vision|editorial)/[a-z0-9][a-z0-9-]*/default$"
    )
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required: bool
    privacy_class: Literal[
        "local-media-sensitive",
        "project-confidential",
        "non-sensitive",
    ]
    authority_class: Literal["evidence", "proposal"]
    latency: AILatencyExpectation
    temporal_precision: Literal[
        "none",
        "sample",
        "frame",
        "millisecond",
        "segment",
    ]
    fallback: AIFallbackContract
    failure_contract: AIFailureContract
    provenance_requirements: list[str] = Field(min_length=1)
    health_contract: AIHealthContract

    @model_validator(mode="after")
    def validate_alias_and_authority(self) -> AICapabilityDeclaration:
        expected_alias = f"local-ai://{self.capability_id.replace('.', '/')}/default"
        if self.alias != expected_alias:
            raise ValueError(
                f"capability alias must be canonical: expected {expected_alias}"
            )
        required_provenance = {
            "resolved_owner",
            "adapter_kind",
            "backend",
            "model_id",
            "model_revision",
            "model_hash_authority",
            "license_authority",
            "health_evidence",
            "privacy_decision",
            "fallback_status",
        }
        if not required_provenance.issubset(self.provenance_requirements):
            missing = sorted(required_provenance - set(self.provenance_requirements))
            raise ValueError(f"capability provenance requirements are incomplete: {missing}")
        return self


class AICapabilityCatalog(FrozenModel):
    """The complete tracked alias surface owned by AoA Editing."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: Literal["aoa-editing-local-ai-capabilities"] = (
        "aoa-editing-local-ai-capabilities"
    )
    baseline_requires_ai: Literal[False] = False
    declarations: list[AICapabilityDeclaration] = Field(min_length=9)
    physical_model_paths_allowed: Literal[False] = False
    filesystem_links_allowed: Literal[False] = False
    output_authority_contract: Literal["evidence-or-proposal-only"] = (
        "evidence-or-proposal-only"
    )

    @model_validator(mode="after")
    def validate_catalog(self) -> AICapabilityCatalog:
        required_capabilities = {
            "speech.transcript",
            "speech.vad",
            "speech.diarization",
            "vision.describe",
            "vision.regions",
            "vision.segment",
            "vision.embed",
            "editorial.plan",
            "editorial.critique",
        }
        capability_ids = [item.capability_id for item in self.declarations]
        aliases = [item.alias for item in self.declarations]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("AI capability IDs must be unique")
        if len(aliases) != len(set(aliases)):
            raise ValueError("AI aliases must be unique")
        missing = required_capabilities - set(capability_ids)
        if missing:
            raise ValueError(f"AI capability catalog is incomplete: {sorted(missing)}")
        if any(item.required for item in self.declarations):
            raise ValueError("the deterministic baseline cannot require an AI provider")
        return self


class LocalProviderHealthBinding(FrozenModel):
    """Untracked instructions for proving one local provider is live."""

    command: list[str] = Field(default_factory=list)
    endpoint: str | None = None
    selector: list[str] = Field(default_factory=list)
    status_field: str = "status"
    ready_values: list[str] = Field(
        default_factory=lambda: ["ready", "available", "healthy", "resident-running"]
    )
    observed_at_field: str | None = "observed_at"
    protocol_version_field: str | None = "protocol_version"
    model_id_field: str | None = "model_id"
    model_revision_field: str | None = "model_revision"
    backend_field: str | None = "backend"
    timeout_seconds: float = Field(default=8.0, gt=0)

    @model_validator(mode="after")
    def validate_probe(self) -> LocalProviderHealthBinding:
        if bool(self.command) == bool(self.endpoint):
            raise ValueError("provider health requires exactly one command or endpoint")
        _validate_local_endpoint(self.endpoint)
        _validate_command_template(self.command)
        return self


class LocalProviderBinding(FrozenModel):
    """Untracked alias resolution; never a tracked model or filesystem link."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    alias: str = Field(
        pattern=r"^local-ai://(speech|vision|editorial)/[a-z0-9][a-z0-9-]*/default$"
    )
    state: Literal["enabled", "disabled", "unavailable"]
    resolved_owner: Literal["abyss-stack", "abyss-machine", "aoa-editing"] | None = None
    adapter_kind: Literal[
        "abyss-stack-service",
        "abyss-machine-cli",
        "localhost-http",
        "disabled",
        "unavailable",
    ]
    backend: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    model_hash_authority: str | None = None
    license_authority: str | None = None
    metadata_evidence: str | None = None
    command: list[str] = Field(default_factory=list)
    endpoint: str | None = None
    output_selector: list[str] = Field(default_factory=list)
    partial_field: str | None = "partial"
    parameters: dict[str, Any] = Field(default_factory=dict)
    health: LocalProviderHealthBinding | None = None
    data_boundary: Literal["local-host", "external"] = "local-host"
    timeout_seconds: float = Field(default=120.0, gt=0)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> LocalProviderBinding:
        _validate_local_endpoint(self.endpoint)
        _validate_command_template(self.command)
        serialized = self.model_dump_json()
        if _contains_secret_material(serialized):
            raise ValueError("provider binding contains a secret-like field or value")
        if self.state == "enabled":
            if self.adapter_kind in {"disabled", "unavailable"}:
                raise ValueError("enabled provider cannot use a disabled adapter")
            required_metadata = (
                self.resolved_owner,
                self.backend,
                self.model_id,
                self.model_revision,
                self.model_hash_authority,
                self.license_authority,
                self.metadata_evidence,
                self.health,
            )
            if any(item is None or item == "" for item in required_metadata):
                raise ValueError(
                    "enabled provider requires owner, model metadata, authorities, "
                    "metadata evidence, and health"
                )
            if self.adapter_kind == "abyss-machine-cli":
                if self.resolved_owner != "abyss-machine" or not self.command:
                    raise ValueError(
                        "abyss-machine CLI binding requires its owner and command"
                    )
                if self.endpoint is not None:
                    raise ValueError("CLI binding cannot carry an endpoint")
            else:
                if self.endpoint is None or self.command:
                    raise ValueError("service binding requires one localhost endpoint")
                if (
                    self.adapter_kind == "abyss-stack-service"
                    and self.resolved_owner != "abyss-stack"
                ):
                    raise ValueError("abyss-stack service binding requires stack owner")
        else:
            expected_adapter = "disabled" if self.state == "disabled" else "unavailable"
            if self.adapter_kind != expected_adapter:
                raise ValueError("inactive binding adapter must match its state")
            if self.command or self.endpoint or self.health is not None:
                raise ValueError("inactive binding cannot carry execution instructions")
            if not self.reason:
                raise ValueError("inactive binding requires an explicit reason")
        return self


class ProviderHealthEvidence(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    status: Literal[
        "healthy",
        "unavailable",
        "stale",
        "version-mismatch",
        "malformed",
    ]
    checked_at: datetime = Field(default_factory=utc_now)
    observed_at: datetime | None = None
    latency_milliseconds: float = Field(ge=0)
    source: Literal["command", "localhost-http", "injected-test"]
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reported_status: str | None = None
    protocol_version: str | None = None
    backend: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    summary: str


class ProviderPrivacyDecision(FrozenModel):
    mode: Literal["local-only", "explicit-provider-opt-in"]
    data_boundary: Literal["local-host", "external", "deterministic-fallback"]
    decision: Literal["allowed", "denied"]
    reason: str
    explicit_opt_in: bool
    media_paths_persisted: Literal[False] = False
    secrets_persisted: Literal[False] = False


class ResolvedProviderReceipt(FrozenModel):
    """Immutable evidence for one alias resolution and invocation attempt."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("providerreceipt"))
    generated_at: datetime = Field(default_factory=utc_now)
    requested_alias: str
    capability_id: str
    resolved_owner: Literal[
        "abyss-stack",
        "abyss-machine",
        "aoa-editing",
    ] | None = None
    adapter_kind: Literal[
        "abyss-stack-service",
        "abyss-machine-cli",
        "localhost-http",
        "deterministic-fallback",
        "disabled",
        "unavailable",
        "unbound",
    ]
    backend: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    model_hash_authority: str | None = None
    license_authority: str | None = None
    metadata_evidence: str | None = None
    endpoint: str | None = None
    command: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    health_evidence: ProviderHealthEvidence | None = None
    execution_time_milliseconds: float = Field(ge=0)
    privacy_decision: ProviderPrivacyDecision
    output_authority: Literal["evidence", "proposal"]
    fallback_status: Literal["not-used", "used", "unavailable", "failed"]
    outcome: Literal["succeeded", "partial", "failed", "refused"]
    failure_code: Literal[
        "provider_missing",
        "provider_disabled",
        "provider_unavailable",
        "provider_stale",
        "version_mismatch",
        "provider_timeout",
        "malformed_response",
        "privacy_denied",
        "invalid_request",
        "fallback_failed",
    ] | None = None
    failure_summary: str | None = None
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_schema_valid: bool | None = None
    partial: bool = False
    project_id: str | None = None
    asset_id: str | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> ResolvedProviderReceipt:
        serialized = self.model_dump_json()
        if _contains_secret_material(serialized):
            raise ValueError("provider receipt contains secret material")
        if self.outcome in {"succeeded", "partial"}:
            if (
                self.response_sha256 is None
                or self.output_schema_valid is not True
                or self.failure_code is not None
            ):
                raise ValueError("successful provider receipt lacks validated output")
        elif self.failure_code is None:
            raise ValueError("failed provider receipt requires a typed failure code")
        if self.outcome == "partial" and not self.partial:
            raise ValueError("partial provider outcome must be declared partial")
        if self.privacy_decision.decision == "denied" and (
            self.failure_code != "privacy_denied" or self.outcome != "refused"
        ):
            raise ValueError("denied privacy decision must refuse the provider call")
        if self.adapter_kind == "deterministic-fallback" and (
            self.resolved_owner != "aoa-editing"
            or self.fallback_status not in {"used", "failed"}
        ):
            raise ValueError("deterministic fallback must remain AoA Editing-owned")
        return self


class ProviderInvocationResult(FrozenModel):
    receipt: ResolvedProviderReceipt
    receipt_path: str
    output: dict[str, Any] | None = None


class ProviderAliasEvalCase(FrozenModel):
    case_id: str
    expected_outcome: str
    actual_outcome: str
    receipt_path: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: list[CheckResult] = Field(min_length=1)
    overall: Literal["pass", "fail"]


class ProviderAliasEvalReport(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("provideraliaseval"))
    generated_at: datetime = Field(default_factory=utc_now)
    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_clean: bool
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    declaration_count: int = Field(ge=9)
    cases: list[ProviderAliasEvalCase] = Field(min_length=8)
    checks: list[CheckResult] = Field(min_length=1)
    artifacts: dict[str, str]
    mandatory_skips: Literal[0] = 0
    reference_media_used_as_input: Literal[False] = False
    provenance: Provenance
    overall: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_eval(self) -> ProviderAliasEvalReport:
        if self.overall == "pass" and (
            not self.git_clean
            or any(item.overall != "pass" for item in self.cases)
            or any(item.status != "pass" for item in self.checks)
        ):
            raise ValueError("passing provider alias eval contains unresolved evidence")
        return self


def _validate_local_endpoint(endpoint: str | None) -> None:
    if endpoint is None:
        return
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("provider endpoint must be a credential-free localhost URL")


def _validate_command_template(command: list[str]) -> None:
    if not command:
        return
    if command[0] in {"sh", "bash", "zsh", "fish"} or any(
        token in {"-c", "--command"} for token in command[:2]
    ):
        raise ValueError("provider commands must not use a shell")
    if any("\n" in token or "\x00" in token for token in command):
        raise ValueError("provider command contains an unsafe token")


def _contains_secret_material(serialized: str) -> bool:
    lowered = serialized.lower()
    secret_markers = (
        '"api_key":',
        '"apikey":',
        '"password":',
        '"secret":',
        '"token":',
        '"authorization":',
        "bearer ",
        "sk-proj-",
    )
    return any(marker in lowered for marker in secret_markers)


class TechniqueApplicabilityDecisionV2(FrozenModel):
    """Typed eligible/refused result emitted before any technique render."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("techniqueapplicability"))
    generated_at: datetime = Field(default_factory=utc_now)
    project_id: str
    asset_id: str
    technique_id: str
    technique_revision: int = Field(gt=0)
    outcome: Literal["eligible", "refused"]
    refusal_code: Literal[
        "insufficient_source_resolution",
        "inapplicable_composition",
        "missing_required_evidence",
        "output_aspect_mismatch",
    ] | None = None
    checks: list[CheckResult] = Field(min_length=1)
    evidence_refs: list[str]
    measured: dict[str, Any]
    provenance: Provenance

    @model_validator(mode="after")
    def validate_outcome(self) -> TechniqueApplicabilityDecisionV2:
        if self.outcome == "eligible" and self.refusal_code is not None:
            raise ValueError("eligible technique decision cannot carry a refusal code")
        if self.outcome == "refused" and self.refusal_code is None:
            raise ValueError("refused technique decision requires a refusal code")
        expected = "pass" if self.outcome == "eligible" else "fail"
        if self.outcome == "eligible" and any(
            item.status != expected for item in self.checks
        ):
            raise ValueError("eligible technique decision requires passing checks")
        if self.outcome == "refused" and not any(
            item.status == "fail" for item in self.checks
        ):
            raise ValueError("refused technique decision requires a failing check")
        if any(item.status == "skip" for item in self.checks):
            raise ValueError("technique applicability cannot skip a check")
        return self


class TechniqueProposalResultV2(FrozenModel):
    """Application-service result preserving an honest refusal without a Treatment."""

    decision: TechniqueApplicabilityDecisionV2
    treatment: Treatment | None = None

    @model_validator(mode="after")
    def validate_result(self) -> TechniqueProposalResultV2:
        if (self.decision.outcome == "eligible") != (self.treatment is not None):
            raise ValueError("eligible proposal result must contain exactly one Treatment")
        return self

class QCReport(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("qc"))
    project_id: str
    version_id: str
    render_path: str
    render_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime = Field(default_factory=utc_now)
    checks: list[CheckResult]
    overall: Literal["pass", "fail", "warn"]


class CompatibilityItem(FrozenModel):
    feature: str
    status: Literal["native", "approximated", "baked", "omitted", "unsupported"]
    detail: str


class InterchangeReport(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("interchange"))
    project_id: str
    version_id: str
    format: str
    output_path: str
    generated_at: datetime = Field(default_factory=utc_now)
    items: list[CompatibilityItem]
    validator: Provenance | None = None


class CameraFrameMeasurement(FrozenModel):
    frame: int = Field(ge=0)
    time_seconds: float = Field(ge=0)
    scale_relative_to_contain: float = Field(gt=0)
    rotation_degrees: float
    center_x: float
    center_y: float
    match_count: int = Field(ge=0)
    inlier_count: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    reprojection_rmse: float = Field(ge=0)


class ReferenceCameraMotion(FrozenModel):
    model: Literal["single_source_similarity_transform"]
    fit_mode: Literal["contain"] = "contain"
    frame_count: int = Field(gt=0)
    frame_rate: FrameRate
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    curve_hint: Literal["linear", "ease_in", "ease_out", "ease_in_out", "hold", "underdetermined"]
    measurements: list[CameraFrameMeasurement]


class ReferenceComposition(FrozenModel):
    model: Literal["single_source_similarity_transform"]
    visual_source_count: Literal[1] = 1
    cut_count: int = Field(ge=0)
    independent_layer_motion_observed: bool
    masks_observed: bool
    background: str
    notes: list[str]


class ReferenceAudioStructure(FrozenModel):
    has_audio_stream: bool
    codec: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    silent_for_full_duration: bool
    silence_intervals: list[dict[str, float]] = Field(default_factory=list)


class ReferenceReconstructionSpec(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("reference"))
    generated_at: datetime = Field(default_factory=utc_now)
    readiness_revision: str
    readiness_receipt_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_metadata: MediaMetadata
    duration_frames: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    camera_motion: ReferenceCameraMotion
    composition: ReferenceComposition
    audio: ReferenceAudioStructure
    observed_effects: dict[str, Any]
    uncertainties: list[str]
    alternative_explanations: list[str]
    comparison_criteria: dict[str, Any]
    provenance: Provenance
    frozen_before_first_render: Literal[True] = True
    reference_media_allowed_in_render: Literal[False] = False


MotionModel = Literal["similarity", "affine", "homography", "unknown"]
MotionCurve = Literal[
    "linear",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "cubic_bezier",
    "hermite",
    "overshoot",
    "settle",
    "hold",
    "underdetermined",
]


class ReferenceMotionFrameV2(FrozenModel):
    """All-frame observable motion evidence with explicit uncertainty."""

    frame: int = Field(ge=0)
    time_seconds: float = Field(ge=0)
    selected_model: MotionModel
    matrix_3x3: list[list[float]]
    visible_source_boundary: list[list[float]]
    center_x: float
    center_y: float
    scale_relative_to_contain: float = Field(gt=0)
    rotation_degrees: float
    pivot_x: None = None
    pivot_y: None = None
    pivot_identifiability: Literal["unidentifiable"] = "unidentifiable"
    confidence: float = Field(ge=0, le=1)
    confidence_interval: dict[str, float]
    match_count: int = Field(ge=0)
    inlier_count: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    reprojection_rmse: float = Field(ge=0)
    residual_flow_mean: float = Field(ge=0)
    residual_flow_p95: float = Field(ge=0)
    residual_valid_fraction: float = Field(ge=0, le=1)
    velocity: dict[str, float]
    acceleration: dict[str, float]
    jerk: dict[str, float]
    model_scores: dict[str, float]
    outlier: bool
    uncertainty: list[str]


class ReferenceMotionEvidenceV2(FrozenModel):
    """Immutable all-frame evidence produced after both generic gates pass."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("referencemotion"))
    generated_at: datetime = Field(default_factory=utc_now)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readiness_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    readiness_receipt_path: str
    readiness_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    motion_gate_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    motion_gate_path: str
    motion_gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_metadata: MediaMetadata
    audio: ReferenceAudioStructure
    frame_count: int = Field(gt=0)
    frame_rate: FrameRate
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    selected_model: MotionModel
    model_distribution: dict[str, int]
    model_mismatch: bool
    curve_hint: MotionCurve
    curve_confidence: float = Field(ge=0, le=1)
    phase_model: dict[str, Any]
    transform_order_assessment: dict[str, Any]
    frames: list[ReferenceMotionFrameV2]
    ambiguous_frames: list[int]
    uncertainties: list[str]
    alternative_explanations: list[str]
    artifacts: dict[str, str]
    provenance: Provenance
    all_frames_analyzed: Literal[True] = True
    reference_media_role: Literal["analysis_and_comparison_only"] = "analysis_and_comparison_only"


class ReferenceReconstructionSpecV2(FrozenModel):
    """Frozen reconstruction contract derived from Reference Motion Evidence v2."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("referencespec"))
    generated_at: datetime = Field(default_factory=utc_now)
    evidence_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readiness_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    motion_gate_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    duration_frames: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    frame_rate: FrameRate
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    audio: ReferenceAudioStructure
    motion_model: MotionModel
    motion_frames: list[ReferenceMotionFrameV2]
    curve_representation: dict[str, Any]
    phase_model: dict[str, Any]
    transform_order: dict[str, Any]
    uncertainties: list[str]
    alternative_explanations: list[str]
    comparison_criteria: dict[str, Any]
    artifacts: dict[str, str]
    provenance: Provenance
    frozen_before_first_v2_render: Literal[True] = True
    reference_media_allowed_in_render: Literal[False] = False

    @model_validator(mode="after")
    def validate_dense_motion_contract(self) -> ReferenceReconstructionSpecV2:
        if len(self.motion_frames) != self.duration_frames:
            raise ValueError("reference spec v2 must contain one motion sample per frame")
        if [item.frame for item in self.motion_frames] != list(range(self.duration_frames)):
            raise ValueError("reference spec v2 motion frames must be contiguous from zero")
        return self


class TimeWarpAnchorV2(FrozenModel):
    """One monotonic output-time to source-time mapping sample."""

    output_frame: int = Field(ge=0)
    source_frame: float = Field(ge=0)


class ReferencePhaseCorrectionV2(FrozenModel):
    """Objective, bounded retime proposal derived from a prior v2 comparison."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("phasecorrection"))
    generated_at: datetime = Field(default_factory=utc_now)
    spec_id: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_report_id: str
    comparison_report_path: str
    comparison_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_motion_report_id: str
    candidate_motion_path: str
    candidate_motion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_frames: int = Field(gt=1)
    anchors: list[TimeWarpAnchorV2] = Field(min_length=4)
    identity_ranges: list[FrameRange] = Field(min_length=1)
    protected_coupling_range: FrameRange
    previous_phase: dict[str, int]
    predicted_phase: dict[str, int]
    predicted_metrics: dict[str, Any]
    predicted_checks: list[CheckResult] = Field(min_length=1)
    failed_check_ids: list[
        Literal[
            "temporal-phase-onset",
            "temporal-phase-duration",
            "temporal-phase-settle",
        ]
    ] = Field(min_length=1)
    threshold_revision: Literal["reference-comparison-v2-frozen-1"] = (
        "reference-comparison-v2-frozen-1"
    )
    algorithm_revision: Literal["phase-correction-v2-localized-search-1"] = (
        "phase-correction-v2-localized-search-1"
    )
    selection: dict[str, Any]
    authority: Literal["objective_evidence_proposal"] = "objective_evidence_proposal"
    reference_media_used_as_render_input: Literal[False] = False
    candidate_media_used_as_render_input: Literal[False] = False
    provenance: Provenance

    @model_validator(mode="after")
    def validate_bounded_warp(self) -> ReferencePhaseCorrectionV2:
        output_frames = [item.output_frame for item in self.anchors]
        source_frames = [item.source_frame for item in self.anchors]
        if any(right <= left for left, right in pairwise(output_frames)):
            raise ValueError("time-warp output frames must be strictly increasing")
        if any(right <= left for left, right in pairwise(source_frames)):
            raise ValueError("time-warp source frames must be strictly increasing")
        final = self.duration_frames - 1
        if (
            output_frames[0] != 0
            or source_frames[0] != 0.0
            or output_frames[-1] != final
            or source_frames[-1] != float(final)
        ):
            raise ValueError("time warp must start and end at the final identity endpoint")
        if any(item.output_frame > final or item.source_frame > final for item in self.anchors):
            raise ValueError("time-warp anchor is outside the bounded duration")

        def mapped(frame: int) -> float:
            for left, right in pairwise(self.anchors):
                if left.output_frame <= frame <= right.output_frame:
                    progress = (frame - left.output_frame) / (
                        right.output_frame - left.output_frame
                    )
                    return left.source_frame + progress * (right.source_frame - left.source_frame)
            raise AssertionError("validated frame has no time-warp segment")

        ranges = [*self.identity_ranges, self.protected_coupling_range]
        for frame_range in ranges:
            if frame_range.end > self.duration_frames:
                raise ValueError("identity range is outside the bounded duration")
            for frame in range(frame_range.start, frame_range.end):
                if not isclose(mapped(frame), float(frame), abs_tol=1e-9):
                    raise ValueError("declared identity range contains a retimed frame")
        if any(item.status != "pass" for item in self.predicted_checks):
            raise ValueError("phase correction may retain only passing predicted checks")
        if len(self.failed_check_ids) != len(set(self.failed_check_ids)):
            raise ValueError("failed phase check ids must be unique")
        return self


ReferenceReconstructionPassKindV2 = Literal[
    "timing_easing",
    "continuous_tangents",
    "moving_pivot_hypothesis",
    "coupled_scale_rotation",
    "phase_compensated_matrix",
]
ReferenceReconstructionPassSequenceV2 = Literal[2, 3, 4, 5, "5r"]


class ReconstructionHypothesisDecisionV2(FrozenModel):
    """Evidence gate for one bounded reconstruction-complexity hypothesis."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    hypothesis: Literal["moving_pivot", "affine_homography", "local_warp"]
    status: Literal[
        "evidence_supported",
        "experimental_gauge_only",
        "rejected_by_evidence",
    ]
    rationale: str = Field(min_length=1)
    measurements: dict[str, Any]
    evidence_refs: list[str] = Field(min_length=1)
    permitted_execution: Literal[
        "render_as_bounded_hypothesis",
        "render_observable_matrix_only",
        "do_not_add_operation",
    ]


class ReferenceReconstructionPassReceiptV2(FrozenModel):
    """Normal application-path proof for one explainable reconstruction pass."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("reconstructionpass"))
    generated_at: datetime = Field(default_factory=utc_now)
    pass_kind: ReferenceReconstructionPassKindV2
    sequence: ReferenceReconstructionPassSequenceV2
    spec_id: str
    spec_path: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_media_used_as_input: Literal[False] = False
    project_id: str
    project_root: str
    project_manifest: str
    asset_id: str
    frozen_spec_evidence: str
    treatment_id: str
    treatment_path: str
    proposed_decision_graph: str
    version_id: str
    approved_edit_graph: str
    timeline_projection: str
    profiles: dict[str, dict[str, Any]]
    editable_exports: dict[str, dict[str, Any]]
    hypotheses: dict[str, ReconstructionHypothesisDecisionV2]
    phase_correction: ReferencePhaseCorrectionV2 | None = None
    phase_correction_path: str | None = None
    phase_correction_evidence: str | None = None
    hidden_manual_steps: list[Any] = Field(default_factory=list)
    provenance: Provenance
    overall: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_optional_phase_correction(self) -> ReferenceReconstructionPassReceiptV2:
        values = (
            self.phase_correction,
            self.phase_correction_path,
            self.phase_correction_evidence,
        )
        if self.pass_kind == "phase_compensated_matrix":
            if self.sequence != "5r" or any(value is None for value in values):
                raise ValueError("phase-compensated pass requires its 5r correction evidence")
        elif any(value is not None for value in values):
            raise ValueError("only the phase-compensated pass may carry correction evidence")
        return self


ReconstructionStudySequenceV2 = Literal["1", "2", "3", "4", "5", "5r"]
ReconstructionStudyPassKindV2 = Literal[
    "v1_baseline",
    "timing_easing",
    "continuous_tangents",
    "moving_pivot_hypothesis",
    "coupled_scale_rotation",
    "phase_compensated_matrix",
]
ReconstructionStudyDispositionV2 = Literal[
    "baseline",
    "rejected",
    "retained",
    "selected",
]


class ReconstructionStudyPassPlanV2(FrozenModel):
    """Human-authored study intent; objective fields are always re-derived."""

    sequence: ReconstructionStudySequenceV2
    label: str = Field(min_length=1)
    pass_kind: ReconstructionStudyPassKindV2
    comparison_path: str
    receipt_path: str | None = None
    motion_gate_path: str | None = None
    disposition: ReconstructionStudyDispositionV2
    rationale: str = Field(min_length=1)
    diagnostic_visual_observation: str = Field(min_length=1)
    diagnostic_artifacts: list[str] = Field(min_length=1)


class ReferenceReconstructionStudyPlanV2(FrozenModel):
    """Reproducible input joining immutable pass evidence and explicit judgments."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    spec_path: str
    protocol_path: str
    passes: list[ReconstructionStudyPassPlanV2] = Field(min_length=6, max_length=6)
    selected_sequence: ReconstructionStudySequenceV2
    hypothesis_decisions: dict[
        Literal["affine_homography", "local_warp"],
        Literal["evidence_rejected_unexecuted"],
    ]
    supplemental_artifacts: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance

    @model_validator(mode="after")
    def validate_study_plan(self) -> ReferenceReconstructionStudyPlanV2:
        sequences = [item.sequence for item in self.passes]
        expected_sequences = ["1", "2", "3", "4", "5", "5r"]
        if sequences != expected_sequences:
            raise ValueError(
                "reconstruction study must preserve the ordered 1, 2, 3, 4, 5, 5r sequence"
            )
        expected_kinds = {
            "1": "v1_baseline",
            "2": "timing_easing",
            "3": "continuous_tangents",
            "4": "moving_pivot_hypothesis",
            "5": "coupled_scale_rotation",
            "5r": "phase_compensated_matrix",
        }
        for item in self.passes:
            if item.pass_kind != expected_kinds[item.sequence]:
                raise ValueError(
                    f"reconstruction study sequence {item.sequence} has the wrong pass kind"
                )
        if self.selected_sequence not in sequences:
            raise ValueError("selected reconstruction sequence is absent from the plan")
        if set(self.hypothesis_decisions) != {"affine_homography", "local_warp"}:
            raise ValueError("study plan must decide both conditional spatial hypotheses")
        return self


class ReconstructionStudyPassV2(FrozenModel):
    """Verified facts and explicit diagnostic judgment for one pass."""

    sequence: ReconstructionStudySequenceV2
    label: str
    pass_kind: ReconstructionStudyPassKindV2
    comparison_id: str
    comparison_path: str
    comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_id: str | None = None
    receipt_path: str | None = None
    receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    motion_gate_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    motion_gate_path: str | None = None
    motion_gate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_same_as_previous: bool
    objective_overall: Literal["pass", "fail"]
    comparison_overall: Literal["pass", "fail", "warn"]
    failed_check_ids: list[str]
    metric_summary: dict[str, float]
    delta_from_previous: dict[str, float]
    disposition: ReconstructionStudyDispositionV2
    rationale: str
    diagnostic_visual_observation: str
    diagnostic_visual_authority: Literal["agent_diagnostic"] = "agent_diagnostic"
    artifacts: dict[str, str]
    diagnostic_artifact_sha256: dict[str, str]


class ReferenceReconstructionStudyV2(FrozenModel):
    """Machine-readable selection ledger across bounded reconstruction passes."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("reconstructionstudy"))
    generated_at: datetime = Field(default_factory=utc_now)
    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_clean: Literal[True]
    plan_path: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_id: str
    spec_path: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passes: list[ReconstructionStudyPassV2] = Field(min_length=6, max_length=6)
    selected_sequence: ReconstructionStudySequenceV2
    selected_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_comparison_id: str
    final_comparison_path: str
    final_comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_decisions: dict[
        Literal["affine_homography", "local_warp"],
        Literal["evidence_rejected_unexecuted"],
    ]
    supplemental_artifacts: dict[str, str]
    supplemental_artifact_sha256: dict[str, str]
    human_review_status: Literal["pending", "complete"]
    human_review_id: str | None = None
    human_review_verdict: str | None = None
    objective_overall: Literal["pass"]
    mandatory_skips: Literal[0] = 0
    provenance: Provenance
    overall: Literal["pass", "warn"]

    @model_validator(mode="after")
    def validate_study_result(self) -> ReferenceReconstructionStudyV2:
        if [item.sequence for item in self.passes] != ["1", "2", "3", "4", "5", "5r"]:
            raise ValueError("study result must preserve all six ordered passes")
        selected = [item for item in self.passes if item.sequence == self.selected_sequence]
        if len(selected) != 1:
            raise ValueError("study must resolve exactly one selected sequence")
        if selected[0].objective_overall != "pass":
            raise ValueError("selected study pass must pass objective comparison")
        if selected[0].candidate_sha256 != self.selected_candidate_sha256:
            raise ValueError("selected study candidate hash is inconsistent")
        if self.human_review_status == "pending":
            if (
                self.human_review_id is not None
                or self.human_review_verdict is not None
                or self.overall != "warn"
            ):
                raise ValueError("pending human review must keep the study at warn")
        elif (
            self.human_review_id is None
            or self.human_review_verdict != "perceptually_and_editorially_indistinguishable"
            or self.overall != "pass"
        ):
            raise ValueError("completed study requires the attributable passing verdict")
        return self


class ReferenceWorkspaceArtifactV2(FrozenModel):
    """Hash-addressed workbench artifact that is never a render input."""

    role: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    scope: Literal["editing_home", "project", "reference_binding"]
    relative_path: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)
    authority: Literal[
        "reference_evidence",
        "candidate_render",
        "comparison_diagnostic",
    ]
    reference_derived: bool
    evaluation_only: Literal[True] = True
    allowed_as_render_input: Literal[False] = False

    @model_validator(mode="after")
    def validate_location(self) -> ReferenceWorkspaceArtifactV2:
        if self.scope == "reference_binding":
            if self.relative_path is not None:
                raise ValueError("reference binding artifacts do not persist physical paths")
            if self.authority != "reference_evidence" or not self.reference_derived:
                raise ValueError("reference binding must remain reference evidence")
            return self
        if self.relative_path is None:
            raise ValueError("workspace artifact requires a root-relative path")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("workspace artifact path must be safe and root-relative")
        return self


class ReferenceWorkspaceRegistrationV2(FrozenModel):
    """Immutable project-local registration of one evidence-backed comparison."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("referenceworkspace"))
    generated_at: datetime = Field(default_factory=utc_now)
    project_id: str
    base_version_id: str
    asset_id: str
    target_effect_path: str = Field(pattern=r"^/")
    target_effect_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_evidence_id: str
    study_id: str
    study_path: str
    study_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_evidence_id: str
    reference_evidence_path: str
    reference_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_id: str
    spec_path: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_id: str
    comparison_path: str
    comparison_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_motion_id: str
    candidate_motion_path: str
    candidate_motion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readiness_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    frame_count: int = Field(gt=1)
    frame_rate: FrameRate
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    phase_markers: dict[str, int]
    pivot_identifiable: bool
    artifacts: dict[str, ReferenceWorkspaceArtifactV2]
    provenance: Provenance
    reference_media_role: Literal["analysis_and_comparison_only"] = (
        "analysis_and_comparison_only"
    )
    reference_media_allowed_in_render: Literal[False] = False
    candidate_media_allowed_as_source: Literal[False] = False

    @model_validator(mode="after")
    def validate_workspace(self) -> ReferenceWorkspaceRegistrationV2:
        relative_documents = (
            self.study_path,
            self.reference_evidence_path,
            self.spec_path,
            self.comparison_path,
            self.candidate_motion_path,
        )
        for value in relative_documents:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("workspace document paths must be editing-home-relative")
        required = {
            "reference_media",
            "candidate_media",
            "side_by_side",
            "aligned_difference",
            "transform_curves",
            "velocity",
            "acceleration",
            "jerk",
            "pivot",
            "phase_contact_sheet",
            "frame_overlay",
            "optical_flow_residual",
            "motion_error",
            "derivative_error",
        }
        missing = required - set(self.artifacts)
        if missing:
            raise ValueError(f"reference workspace artifacts are incomplete: {sorted(missing)}")
        if any(key != artifact.role for key, artifact in self.artifacts.items()):
            raise ValueError("workspace artifact keys must match artifact roles")
        expected_markers = {
            "onset",
            "twist_start",
            "twist_peak",
            "twist_end",
            "settle",
        }
        if set(self.phase_markers) != expected_markers:
            raise ValueError("workspace must carry every canonical phase marker exactly once")
        ordered_markers = [
            self.phase_markers[key]
            for key in (
                "onset",
                "twist_start",
                "twist_peak",
                "twist_end",
                "settle",
            )
        ]
        if any(left >= right for left, right in pairwise(ordered_markers)):
            raise ValueError("workspace phase markers must be strictly ordered")
        reference = self.artifacts["reference_media"]
        candidate = self.artifacts["candidate_media"]
        if (
            reference.scope != "reference_binding"
            or reference.sha256 != self.reference_sha256
        ):
            raise ValueError("reference media must resolve only through its sealed binding")
        if candidate.scope != "project" or candidate.sha256 != self.candidate_sha256:
            raise ValueError("candidate media must be a hash-matched project artifact")
        if any(
            frame < 0 or frame >= self.frame_count for frame in self.phase_markers.values()
        ):
            raise ValueError("workspace phase marker is outside the timeline")
        return self


class ReferenceCurveSampleV2(FrozenModel):
    center_x: float
    center_y: float
    scale: float = Field(gt=0)
    rotation_degrees: float
    velocity: dict[str, float]
    acceleration: dict[str, float]
    jerk: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    residual_flow_mean: float | None = Field(default=None, ge=0)
    uncertainty: list[str] = Field(default_factory=list)


class ReferenceWorkbenchSampleV2(FrozenModel):
    frame: int = Field(ge=0)
    time_seconds: float = Field(ge=0)
    reference: ReferenceCurveSampleV2
    candidate: ReferenceCurveSampleV2


class ReferenceMotionInterpretationV2(FrozenModel):
    id: str
    label: str = Field(min_length=1)
    status: Literal["supported", "ambiguous", "rejected"]
    explanation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    evidence_refs: list[str] = Field(min_length=1)
    editable_operation_kinds: list[str] = Field(default_factory=list)


class ReferenceWorkspaceVersionV2(FrozenModel):
    id: str
    message: str
    created_at: datetime
    current: bool
    preview_available: bool
    final_available: bool


class ReferenceWorkbenchBundleV2(FrozenModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    workspace: ReferenceWorkspaceRegistrationV2
    samples: list[ReferenceWorkbenchSampleV2]
    interpretations: list[ReferenceMotionInterpretationV2]
    versions: list[ReferenceWorkspaceVersionV2]
    proposals: list[MotionCorrectionProposalSetV2]
    reviews: list[MotionCorrectionReviewV2]
    human_correction_evidence_ids: list[str]
    objective_summary: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dense_samples(self) -> ReferenceWorkbenchBundleV2:
        if [item.frame for item in self.samples] != list(
            range(self.workspace.frame_count)
        ):
            raise ValueError("reference workbench requires contiguous all-frame samples")
        return self


class ReferenceMediaBindingV2(FrozenModel):
    """Untracked host-local resolution of a sealed reference hash."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_path: str
    size_bytes: int = Field(gt=0)
    bound_at: datetime = Field(default_factory=utc_now)
    provenance: Provenance


class TechnicalComparisonCriteriaV2(FrozenModel):
    resolution: tuple[int, int]
    frame_rate: FrameRate
    frame_count_tolerance: Literal[0]
    audio_structure_exact: Literal[True]


class GeometricComparisonCriteriaV2(FrozenModel):
    all_frame_center_rmse_normalized_max: float = Field(gt=0)
    all_frame_scale_rmse_max: float = Field(gt=0)
    all_frame_rotation_rmse_degrees_max: float = Field(gt=0)
    visible_boundary_rmse_normalized_max: float = Field(gt=0)
    homography_corner_rmse_normalized_max: float = Field(gt=0)


class TemporalComparisonCriteriaV2(FrozenModel):
    position_velocity_rmse_max: float = Field(gt=0)
    scale_velocity_rmse_max: float = Field(gt=0)
    angular_velocity_rmse_degrees_max: float = Field(gt=0)
    position_acceleration_rmse_max: float = Field(gt=0)
    scale_acceleration_rmse_max: float = Field(gt=0)
    angular_acceleration_rmse_degrees_max: float = Field(gt=0)
    jerk_rmse_max: float = Field(gt=0)
    phase_onset_tolerance_frames: int = Field(ge=0)
    phase_settle_tolerance_frames: int = Field(ge=0)
    temporal_flow_rmse_max: float = Field(gt=0)


class PerceptualComparisonCriteriaV2(FrozenModel):
    vmaf_mean_min: float = Field(ge=0, le=100)
    ssim_all_min: float = Field(ge=0, le=1)
    psnr_diagnostic_only: Literal[True]
    no_black_frames: Literal[True]
    no_edge_exposure: Literal[True]
    no_unwanted_blur: Literal[True]


BaseEditorialCriterionV2 = Literal[
    "smoothness",
    "weight",
    "natural_acceleration",
    "rotation_character",
    "twist_character",
    "settle_timing",
    "absence_of_mechanical_linearity",
    "compositional_match",
]

EditorialCriterionV2 = Literal[
    "smoothness",
    "weight",
    "natural_acceleration",
    "rotation_character",
    "twist_character",
    "settle_timing",
    "absence_of_mechanical_linearity",
    "compositional_match",
    "reference_overlay_absence",
    "encoding_defect_separation",
]


class EditorialComparisonCriteriaV2(FrozenModel):
    human_review_required: Literal[True]
    rubric: list[BaseEditorialCriterionV2]
    required_verdict: Literal["perceptually_and_editorially_indistinguishable"]

    @model_validator(mode="after")
    def validate_rubric(self) -> EditorialComparisonCriteriaV2:
        expected = {
            "smoothness",
            "weight",
            "natural_acceleration",
            "rotation_character",
            "twist_character",
            "settle_timing",
            "absence_of_mechanical_linearity",
            "compositional_match",
        }
        if set(self.rubric) != expected or len(self.rubric) != len(expected):
            raise ValueError("editorial comparison rubric must contain every frozen axis once")
        return self


class ComparisonLineageCriteriaV2(FrozenModel):
    only_visual_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_hash_forbidden: str = Field(pattern=r"^[0-9a-f]{64}$")


class ComparisonCriteriaV2(FrozenModel):
    threshold_revision: Literal["reference-comparison-v2-frozen-1"]
    frozen_before_first_v2_render: Literal[True]
    threshold_basis: dict[str, Any]
    technical: TechnicalComparisonCriteriaV2
    geometric: GeometricComparisonCriteriaV2
    temporal: TemporalComparisonCriteriaV2
    perceptual: PerceptualComparisonCriteriaV2
    editorial: EditorialComparisonCriteriaV2
    lineage: ComparisonLineageCriteriaV2


class EditorialRubricAssessmentV2(FrozenModel):
    criterion: EditorialCriterionV2
    status: Literal["pass", "fail"]
    notes: str = Field(min_length=1)


class EditorialReviewV2(FrozenModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("editorialreview"))
    generated_at: datetime = Field(default_factory=utc_now)
    protocol_id: str
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1)
    reviewer_role: Literal["human_operator"] = "human_operator"
    verdict: Literal[
        "perceptually_and_editorially_indistinguishable",
        "revision_required",
    ]
    rubric: list[EditorialRubricAssessmentV2] = Field(min_length=1)
    reviewed_artifacts: list[str] = Field(min_length=3)
    reference_media_role: Literal["comparison_only"] = "comparison_only"

    @model_validator(mode="after")
    def validate_review(self) -> EditorialReviewV2:
        criteria = [item.criterion for item in self.rubric]
        if len(criteria) != len(set(criteria)):
            raise ValueError("editorial review criteria must be unique")
        if self.verdict == "perceptually_and_editorially_indistinguishable" and any(
            item.status != "pass" for item in self.rubric
        ):
            raise ValueError("an indistinguishable verdict requires every rubric axis to pass")
        return self


class ReferenceComparisonProtocolV2(FrozenModel):
    """Revision-bound executable protocol frozen before the first v2 candidate."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("comparisonprotocol"))
    generated_at: datetime = Field(default_factory=utc_now)
    spec_id: str
    spec_path: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_clean: Literal[True]
    threshold_revision: Literal["reference-comparison-v2-frozen-1"]
    criteria: ComparisonCriteriaV2
    algorithm_revision: Literal["comparison-v2-all-frame-1"]
    algorithms: dict[str, Any]
    mandatory_axes: list[
        Literal[
            "technical",
            "geometric",
            "temporal",
            "perceptual",
            "editorial",
            "lineage",
        ]
    ]
    mandatory_checks: list[str] = Field(min_length=1)
    required_artifacts: list[str] = Field(min_length=1)
    human_rubric: list[EditorialCriterionV2] = Field(min_length=1)
    provenance: Provenance
    frozen_before_first_candidate_render: Literal[True] = True
    reference_media_allowed_in_render: Literal[False] = False

    @model_validator(mode="after")
    def validate_complete_protocol(self) -> ReferenceComparisonProtocolV2:
        expected_axes = {
            "technical",
            "geometric",
            "temporal",
            "perceptual",
            "editorial",
            "lineage",
        }
        if set(self.mandatory_axes) != expected_axes or len(self.mandatory_axes) != len(
            expected_axes
        ):
            raise ValueError("comparison v2 protocol must declare every mandatory axis once")
        if len(self.mandatory_checks) != len(set(self.mandatory_checks)):
            raise ValueError("comparison v2 mandatory checks must be unique")
        if len(self.required_artifacts) != len(set(self.required_artifacts)):
            raise ValueError("comparison v2 required artifacts must be unique")
        if len(self.human_rubric) != len(set(self.human_rubric)):
            raise ValueError("comparison v2 human rubric axes must be unique")
        return self


class ReferenceComparisonReportV2(FrozenModel):
    """All-frame objective result plus a separately attributable human verdict."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("comparison"))
    generated_at: datetime = Field(default_factory=utc_now)
    supersedes_report_id: str | None = None
    protocol_id: str
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_id: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_path: str
    technical: dict[str, Any]
    geometric: dict[str, Any]
    temporal: dict[str, Any]
    perceptual: dict[str, Any]
    editorial: dict[str, Any]
    lineage: dict[str, Any]
    checks: list[CheckResult]
    artifacts: dict[str, str]
    human_review: EditorialReviewV2 | None = None
    provenance: Provenance
    reference_media_used_only_for_comparison: Literal[True] = True
    all_frames_compared: Literal[True] = True
    mandatory_skips: Literal[0] = 0
    objective_overall: Literal["pass", "fail"]
    overall: Literal["pass", "fail", "warn"]


class ReferenceComparisonReport(FrozenModel):
    """Objective and reviewable comparison against a frozen reference spec."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("comparison"))
    generated_at: datetime = Field(default_factory=utc_now)
    spec_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_path: str
    geometry: dict[str, Any]
    motion: dict[str, Any]
    perceptual: dict[str, Any]
    audio: dict[str, Any]
    lineage: dict[str, Any]
    checks: list[CheckResult]
    artifacts: dict[str, str]
    visual_review: str | None = None
    provenance: Provenance
    reference_media_used_only_for_comparison: Literal[True] = True
    overall: Literal["pass", "fail", "warn"]


class CleanRerunReport(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("cleanrerun"))
    generated_at: datetime = Field(default_factory=utc_now)
    spec_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: dict[str, Any]
    replay: dict[str, Any]
    checks: list[CheckResult]
    artifacts: dict[str, str]
    provenance: Provenance
    old_render_used_as_input: Literal[False] = False
    overall: Literal["pass", "fail", "warn"]


class CleanRerunRenderProofV2(FrozenModel):
    """One independently recreated render joined to QC and source-only lineage."""

    profile: Literal["preview", "final"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inode: int = Field(gt=0)
    qc_path: str
    qc_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qc_overall: Literal["pass"]
    lineage_path: str
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_hashes: list[str] = Field(min_length=1)
    forbidden_hashes_present: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_render_lineage(self) -> CleanRerunRenderProofV2:
        if self.forbidden_hashes_present:
            raise ValueError("clean-rerun render lineage contains a forbidden hash")
        return self


class CleanRerunExportProofV2(FrozenModel):
    """Parser-backed editable projection recreated in the clean workspace."""

    format: Literal["kdenlive", "otio"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validator_tool: str = Field(min_length=1)
    validator_present: Literal[True] = True


class CleanRerunApplicationProofV2(FrozenModel):
    """Hash-complete normal application path for baseline or clean replay."""

    receipt_id: str
    receipt_path: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_id: str
    project_root: str
    project_manifest_path: str
    project_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_id: str
    asset_path: str
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_inode: int = Field(gt=0)
    derivatives_manifest_path: str
    derivatives_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivative_artifacts: list[str] = Field(min_length=1)
    treatment_id: str
    treatment_path: str
    treatment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch_id: str
    patch_path: str
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_id: str
    version_path: str
    version_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_graph_id: str
    decision_graph_path: str
    decision_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_log_path: str
    semantic_timeline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profiles: dict[str, CleanRerunRenderProofV2]
    editable_exports: dict[str, CleanRerunExportProofV2]
    hidden_manual_steps: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_complete_application_path(self) -> CleanRerunApplicationProofV2:
        if set(self.profiles) != {"preview", "final"}:
            raise ValueError("clean rerun requires preview and final render proofs")
        if any(key != proof.profile for key, proof in self.profiles.items()):
            raise ValueError("clean-rerun profile keys must match proof profiles")
        if set(self.editable_exports) != {"kdenlive", "otio"}:
            raise ValueError("clean rerun requires Kdenlive and OTIO export proofs")
        if any(key != proof.format for key, proof in self.editable_exports.items()):
            raise ValueError("clean-rerun export keys must match proof formats")
        if self.hidden_manual_steps:
            raise ValueError("clean rerun cannot contain hidden manual steps")
        return self


class CleanRerunReportV2(FrozenModel):
    """Revision-bound replay of approved v2 semantics in an empty editing home."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("cleanrerun"))
    generated_at: datetime = Field(default_factory=utc_now)
    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_clean: Literal[True]
    spec_id: str
    spec_path: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_phase_correction_id: str
    approved_phase_correction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: CleanRerunApplicationProofV2
    replay: CleanRerunApplicationProofV2
    checks: list[CheckResult] = Field(min_length=1)
    artifacts: dict[str, str]
    mandatory_skips: Literal[0] = 0
    old_render_used_as_input: Literal[False] = False
    reference_media_used_as_input: Literal[False] = False
    provenance: Provenance
    overall: Literal["pass", "fail"]


class TechniqueTransferReport(FrozenModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("transfer"))
    generated_at: datetime = Field(default_factory=utc_now)
    technique_id: str
    technique_revision: int
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_id: str
    version_id: str
    motion: dict[str, Any]
    profiles: dict[str, Any]
    checks: list[CheckResult]
    artifacts: dict[str, str]
    provenance: Provenance
    target_specific_media_used: Literal[False] = False
    overall: Literal["pass", "fail", "warn"]


class TechniqueTransferCaseV2(FrozenModel):
    """One applied or honestly refused source-class transfer case."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    case_id: str
    source_class: str
    expected_application_outcome: Literal["eligible", "refused"]
    application_outcome: Literal["eligible", "refused"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    source_provenance: dict[str, Any]
    applicability: TechniqueApplicabilityDecisionV2
    project_id: str
    version_id: str | None = None
    motion: dict[str, Any] = Field(default_factory=dict)
    profiles: dict[str, Any] = Field(default_factory=dict)
    editable_exports: dict[str, Any] = Field(default_factory=dict)
    deterministic_replay: dict[str, Any] = Field(default_factory=dict)
    checks: list[CheckResult] = Field(min_length=1)
    artifacts: dict[str, str] = Field(default_factory=dict)
    reference_media_used_as_input: Literal[False] = False
    target_specific_branch_used: Literal[False] = False
    overall: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_case(self) -> TechniqueTransferCaseV2:
        if (
            self.expected_application_outcome != self.application_outcome
            or self.applicability.outcome != self.application_outcome
        ):
            raise ValueError("transfer case application outcome differs from its contract")
        if self.application_outcome == "eligible":
            if self.version_id is None or not self.profiles or not self.editable_exports:
                raise ValueError("eligible transfer case lacks version, renders, or exports")
        elif self.version_id is not None or self.profiles or self.editable_exports:
            raise ValueError("refused transfer case cannot carry edit or render products")
        if self.overall == "pass" and any(item.status != "pass" for item in self.checks):
            raise ValueError("passing transfer case contains a non-passing check")
        return self


class TechniqueTransferCorpusReportV2(FrozenModel):
    """Revision-bound transfer gate over heterogeneous positive and negative cases."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    id: str = Field(default_factory=lambda: new_id("transfercorpus"))
    generated_at: datetime = Field(default_factory=utc_now)
    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_clean: bool
    technique_id: str
    input_technique_revision: int = Field(gt=0)
    output_technique_revision: int = Field(gt=0)
    cases: list[TechniqueTransferCaseV2] = Field(min_length=7)
    checks: list[CheckResult] = Field(min_length=1)
    artifacts: dict[str, str]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    forbidden_media_hashes: list[str] = Field(default_factory=list)
    mandatory_skips: Literal[0] = 0
    reference_media_used_as_input: Literal[False] = False
    target_specific_branch_used: Literal[False] = False
    provenance: Provenance
    overall: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_corpus(self) -> TechniqueTransferCorpusReportV2:
        case_ids = [item.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("transfer corpus case IDs must be unique")
        if self.output_technique_revision != self.input_technique_revision + 1:
            raise ValueError("transfer corpus must advance its packet exactly one revision")
        if self.overall == "pass" and (
            not self.git_clean
            or any(item.overall != "pass" for item in self.cases)
            or any(item.status != "pass" for item in self.checks)
        ):
            raise ValueError("passing transfer corpus contains unresolved evidence")
        return self


class CompletionAuditReport(FrozenModel):
    """Fail-closed evidence map for the complete prototype definition of done."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("completion"))
    generated_at: datetime = Field(default_factory=utc_now)
    git_revision: str
    evidence: dict[str, str]
    requirement_coverage: dict[str, list[str]]
    checks: list[CheckResult]
    artifacts: dict[str, str]
    mandatory_skips: int = Field(ge=0)
    provenance: Provenance
    overall: Literal["pass", "fail", "warn"]


class StorageRelocationEntry(FrozenModel):
    """One measured copy/rebuild unit in a product-home relocation."""

    id: str
    classification: str
    method: Literal["copy_then_verify", "rebuild_from_canonical"]
    source_path: str
    destination_path: str
    file_count: int = Field(ge=0)
    logical_bytes: int = Field(ge=0)
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    destination_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    excluded_paths: list[str] = Field(default_factory=list)
    status: Literal["planned", "verified", "rebuilt"]


class StorageRelocationReport(FrozenModel):
    """Fail-closed receipt for copying legacy product state into one home."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("relocation"))
    generated_at: datetime = Field(default_factory=utc_now)
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_data_root: str
    destination_home: str
    dry_run: bool
    entries: list[StorageRelocationEntry]
    checks: list[CheckResult]
    filesystem_links_created: Literal[False] = False
    legacy_cleanup_authorized: Literal[False] = False
    cutover_ready: bool
    overall: Literal["pass", "fail"]


class MotionGroundTruthFrame(FrozenModel):
    """One exact synthetic transform sample before rasterization and encoding."""

    frame: int = Field(ge=0)
    time_seconds: float = Field(ge=0)
    matrix_3x3: list[list[float]]
    center_x: float
    center_y: float
    scale_relative_to_contain: float = Field(gt=0)
    rotation_degrees: float
    pivot_x: float
    pivot_y: float
    velocity: dict[str, float]
    acceleration: dict[str, float]
    jerk: dict[str, float]
    phase: str
    transform_order: list[str]


class MotionGroundTruthFixture(FrozenModel):
    """Independent render truth for one motion family or degradation condition."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str
    content_origin: Literal["synthetic_ground_truth"] = "synthetic_ground_truth"
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_path: str
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    frame_rate: FrameRate
    expected_model: MotionModel
    expected_curve: MotionCurve
    requirement_tags: list[str]
    degradations: list[str] = Field(default_factory=list)
    expected_uncertainties: list[str] = Field(default_factory=list)
    frames: list[MotionGroundTruthFrame]
    provenance: Provenance


class MotionGroundTruthCorpus(FrozenModel):
    """Manifest joining all independently rendered motion fixtures."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("motioncorpus"))
    generated_at: datetime = Field(default_factory=utc_now)
    content_origin: Literal["synthetic_ground_truth"] = "synthetic_ground_truth"
    fixtures: list[MotionGroundTruthFixture]
    requirement_coverage: dict[str, list[str]]
    provenance: Provenance


class MotionRecoveryFrame(FrozenModel):
    """Observable transform estimate and uncertainty for one decoded frame."""

    frame: int = Field(ge=0)
    time_seconds: float = Field(ge=0)
    selected_model: MotionModel
    matrix_3x3: list[list[float]]
    scale_relative_to_contain: float | None = Field(default=None, gt=0)
    rotation_degrees: float | None = None
    center_x: float | None = None
    center_y: float | None = None
    affine_anisotropy: float | None = Field(default=None, ge=0)
    affine_shear: float | None = Field(default=None, ge=0)
    perspective_strength: float | None = Field(default=None, ge=0)
    match_count: int = Field(ge=0)
    inlier_count: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    reprojection_rmse: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    model_scores: dict[str, float]
    velocity: dict[str, float] = Field(default_factory=dict)
    acceleration: dict[str, float] = Field(default_factory=dict)
    jerk: dict[str, float] = Field(default_factory=dict)
    pivot_x: None = None
    pivot_y: None = None
    pivot_identifiability: Literal["unidentifiable"] = "unidentifiable"
    outlier: bool = False


class MotionRecoveryReport(FrozenModel):
    """All-frame recovery result that preserves failures and alternative models."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("motionrecovery"))
    generated_at: datetime = Field(default_factory=utc_now)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_count: int = Field(gt=0)
    analyzed_frame_count: int = Field(gt=0)
    frame_rate: FrameRate
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    selected_model: MotionModel
    model_distribution: dict[str, int]
    curve_hint: MotionCurve
    curve_confidence: float = Field(ge=0, le=1)
    mean_confidence: float = Field(ge=0, le=1)
    model_mismatch: bool
    outlier_frames: list[int]
    phase_boundaries: dict[str, int]
    uncertainties: list[str]
    alternative_explanations: list[str]
    frames: list[MotionRecoveryFrame]
    provenance: Provenance


class MotionRecoveryGateReport(FrozenModel):
    """Fail-closed generic recovery gate over the independent corpus."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("motiongate"))
    generated_at: datetime = Field(default_factory=utc_now)
    git_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_clean: bool
    corpus_path: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_reports: dict[str, str]
    metrics: dict[str, dict[str, Any]]
    checks: list[CheckResult]
    mandatory_skips: int = Field(ge=0)
    provenance: Provenance
    overall: Literal["pass", "fail"]


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "asset": Asset,
    "media-manifest": Asset,
    "derived-media": DerivedMediaManifest,
    "editorial-brief": EditorialBriefRevision,
    "screen-workflow-plan": ScreenWorkflowPlan,
    "screen-workflow-experience-admission": ScreenWorkflowExperienceAdmission,
    "screen-workflow-experience-public-projection": (
        ScreenWorkflowExperiencePublicProjection
    ),
    "screen-workflow-experience-receipt": ScreenWorkflowExperienceReceipt,
    "screen-workflow-voiceover-timing": ScreenWorkflowVoiceoverTiming,
    "screen-workflow-edit-spec": ScreenWorkflowEditSpec,
    "reference-workflow-study-plan": ReferenceWorkflowStudyPlan,
    "reference-workflow-study": ReferenceWorkflowStudyReceipt,
    "evidence": EvidenceRecord,
    "timeline": Timeline,
    "motion-language-v2": TransformEffectV2,
    "patch": EditPatch,
    "patch-preview": PatchPreview,
    "motion-correction-proposal-v2": MotionCorrectionProposalSetV2,
    "motion-correction-review-v2": MotionCorrectionReviewV2,
    "reference-motion-human-correction-v2": ReferenceMotionHumanCorrectionV2,
    "treatment": Treatment,
    "version": ProjectVersion,
    "editorial-decision-graph": EditorialDecisionGraph,
    "decision-log": DecisionLog,
    "style-profile": StyleProfile,
    "technique-packet": TechniquePacket,
    "technique-composition-evidence-v2": TechniqueCompositionEvidenceV2,
    "technique-applicability-v2": TechniqueApplicabilityDecisionV2,
    "project": ProjectManifest,
    "job": JobReceipt,
    "render-plan": RenderPlan,
    "qc": QCReport,
    "interchange": InterchangeReport,
    "reference-spec": ReferenceReconstructionSpec,
    "reference-motion-evidence-v2": ReferenceMotionEvidenceV2,
    "reference-spec-v2": ReferenceReconstructionSpecV2,
    "reference-phase-correction-v2": ReferencePhaseCorrectionV2,
    "reference-reconstruction-pass-v2": ReferenceReconstructionPassReceiptV2,
    "reference-reconstruction-study-plan-v2": ReferenceReconstructionStudyPlanV2,
    "reference-reconstruction-study-v2": ReferenceReconstructionStudyV2,
    "reference-workspace-v2": ReferenceWorkspaceRegistrationV2,
    "reference-workbench-v2": ReferenceWorkbenchBundleV2,
    "reference-media-binding-v2": ReferenceMediaBindingV2,
    "comparison-protocol-v2": ReferenceComparisonProtocolV2,
    "comparison-v2": ReferenceComparisonReportV2,
    "editorial-review-v2": EditorialReviewV2,
    "comparison": ReferenceComparisonReport,
    "clean-rerun": CleanRerunReport,
    "clean-rerun-v2": CleanRerunReportV2,
    "technique-transfer": TechniqueTransferReport,
    "technique-transfer-corpus-v2": TechniqueTransferCorpusReportV2,
    "completion-audit": CompletionAuditReport,
    "storage-relocation": StorageRelocationReport,
    "motion-ground-truth": MotionGroundTruthCorpus,
    "motion-recovery": MotionRecoveryReport,
    "motion-recovery-gate": MotionRecoveryGateReport,
    "ai-capability-catalog": AICapabilityCatalog,
    "local-provider-binding": LocalProviderBinding,
    "resolved-provider-receipt": ResolvedProviderReceipt,
    "provider-alias-eval": ProviderAliasEvalReport,
    "timeline-projection": Timeline,
}
