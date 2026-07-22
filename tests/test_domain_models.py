from __future__ import annotations

import pytest

from aoa_editing.domain.invariants import InvariantViolation, validate_timeline
from aoa_editing.domain.models import (
    Clip,
    FrameRange,
    ScalarKeyframe,
    Timeline,
    Track,
    TrackKind,
    TransformEffect,
    Vec2Keyframe,
)


def test_timeline_rejects_clip_beyond_duration() -> None:
    with pytest.raises(ValueError, match="ends after timeline"):
        Timeline(
            duration_frames=20,
            tracks=[
                Track(
                    kind=TrackKind.VIDEO,
                    name="picture",
                    clips=[
                        Clip(
                            asset_id="asset_a",
                            timeline_range=FrameRange(start=10, duration=20),
                        )
                    ],
                )
            ],
        )


def test_non_layered_video_track_rejects_overlap() -> None:
    timeline = Timeline(
        duration_frames=100,
        tracks=[
            Track(
                kind=TrackKind.VIDEO,
                name="picture",
                clips=[
                    Clip(asset_id="a", timeline_range=FrameRange(start=0, duration=60)),
                    Clip(asset_id="b", timeline_range=FrameRange(start=50, duration=50)),
                ],
            )
        ],
    )
    with pytest.raises(InvariantViolation, match="overlap"):
        validate_timeline(timeline)


def test_separate_video_tracks_allow_layering() -> None:
    timeline = Timeline(
        duration_frames=100,
        tracks=[
            Track(
                kind=TrackKind.VIDEO,
                name="background",
                clips=[Clip(asset_id="a", timeline_range=FrameRange(start=0, duration=100))],
            ),
            Track(
                kind=TrackKind.VIDEO,
                name="foreground",
                clips=[Clip(asset_id="b", timeline_range=FrameRange(start=0, duration=100))],
            ),
        ],
    )
    validate_timeline(timeline)


def test_virtual_camera_contract_is_explicit_and_keyframes_are_ordered() -> None:
    effect = TransformEffect(
        fit_mode="contain",
        position_mode="canvas_center",
        position=[
            Vec2Keyframe(frame=0, x=0.3, y=0.6),
            Vec2Keyframe(frame=20, x=0.5, y=0.5, easing="ease_in_out"),
        ],
        scale=[
            ScalarKeyframe(frame=0, value=2.8),
            ScalarKeyframe(frame=20, value=1.0, easing="ease_in_out"),
        ],
        rotation=[
            ScalarKeyframe(frame=0, value=-7.2),
            ScalarKeyframe(frame=20, value=0.0, easing="ease_in_out"),
        ],
    )

    assert effect.fit_mode == "contain"
    assert effect.position_mode == "canvas_center"

    with pytest.raises(ValueError, match="strictly increasing"):
        TransformEffect(
            scale=[
                ScalarKeyframe(frame=10, value=1.0),
                ScalarKeyframe(frame=5, value=2.0),
            ]
        )
