from __future__ import annotations

import pytest

from aoa_editing.domain.migrations import MigrationError, migrate_document
from aoa_editing.domain.models import FrameRate, Timeline


def test_legacy_timeline_migrates_additively() -> None:
    legacy = {
        "width": 1280,
        "height": 720,
        "frame_rate": FrameRate(numerator=24).model_dump(mode="json"),
        "duration_frames": 240,
    }
    migrated = migrate_document("timeline", legacy)
    assert migrated["schema_version"] == "1.0.0"
    assert migrated["audio_sample_rate"] == 48000
    assert Timeline.model_validate(migrated).duration_frames == 240


def test_unknown_schema_fails_without_guessing() -> None:
    with pytest.raises(MigrationError, match="no migration"):
        migrate_document("timeline", {"schema_version": "9.0.0"})

