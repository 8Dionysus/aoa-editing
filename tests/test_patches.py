from __future__ import annotations

from aoa_editing.domain.models import (
    Clip,
    FrameRange,
    PatchOperation,
    Timeline,
    Track,
    TrackKind,
)
from aoa_editing.domain.patches import apply_timeline_patch


def test_patch_round_trip_restores_timeline() -> None:
    base = Timeline(duration_frames=90)
    track = Track(
        kind=TrackKind.VIDEO,
        name="picture",
        clips=[Clip(asset_id="asset_a", timeline_range=FrameRange(start=0, duration=90))],
    )
    changed, inverse = apply_timeline_patch(
        base,
        [PatchOperation(op="add", path="/tracks/-", value=track.model_dump(mode="json"))],
    )
    assert changed.tracks == [track]
    restored, _ = apply_timeline_patch(changed, inverse)
    assert restored == base


def test_patch_replace_is_reversible() -> None:
    base = Timeline(duration_frames=90, background="#000000")
    changed, inverse = apply_timeline_patch(
        base,
        [PatchOperation(op="replace", path="/background", value="#ffffff")],
    )
    assert changed.background == "#ffffff"
    restored, _ = apply_timeline_patch(changed, inverse)
    assert restored == base

