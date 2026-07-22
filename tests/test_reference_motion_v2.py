from __future__ import annotations

import json
from pathlib import Path

import pytest

from aoa_editing.evals.motion_fixtures import fixture_definitions, render_motion_fixture
from aoa_editing.evals.reference_v2 import (
    ReferenceV2GateError,
    analyze_reference_v2,
)


@pytest.mark.integration
def test_reference_v2_analyzes_every_frame_and_freezes_artifact_backed_spec(
    tmp_path: Path,
) -> None:
    definition = next(item for item in fixture_definitions() if item.id == "ease-in-out")
    fixture = render_motion_fixture(definition, tmp_path / "fixture")
    readiness, motion_gate = _gate_receipts(tmp_path, revision="a" * 40)
    output = tmp_path / "reference-v2"

    evidence, spec = analyze_reference_v2(
        fixture.source_path,
        fixture.video_path,
        output,
        readiness_path=readiness,
        motion_gate_path=motion_gate,
    )

    assert evidence.schema_version == "2.0.0"
    assert evidence.all_frames_analyzed is True
    assert evidence.frame_count == definition.frame_count
    assert len(evidence.frames) == definition.frame_count
    assert evidence.selected_model == "similarity"
    assert evidence.curve_hint == "ease_in_out"
    assert evidence.motion_gate_revision == "a" * 40
    assert evidence.readiness_revision == "a" * 40
    assert len(evidence.ambiguous_frames) < evidence.frame_count
    assert all(frame.visible_source_boundary for frame in evidence.frames)
    assert all(frame.uncertainty for frame in evidence.frames)
    assert any("pivot" in item.lower() for item in evidence.uncertainties)
    coupling = evidence.phase_model["channel_coupling"]
    assert coupling["scale_rotation_progress_rmse"] < 0.02
    assert coupling["center_phase_relation"] in {
        "center_lags",
        "center_leads",
        "synchronized",
    }

    assert spec.schema_version == "2.0.0"
    assert spec.evidence_id == evidence.id
    assert spec.duration_frames == definition.frame_count
    assert len(spec.motion_frames) == definition.frame_count
    assert spec.frozen_before_first_v2_render is True
    assert spec.reference_media_allowed_in_render is False
    assert spec.comparison_criteria["temporal"]["jerk_rmse_max"] > 0
    assert spec.comparison_criteria["editorial"]["human_review_required"] is True
    assert spec.curve_representation["sampling"] == "every_frame"

    assert (output / "reference-motion-evidence-v2.json").is_file()
    assert (output / "reference-spec-v2.json").is_file()
    required_artifacts = {
        "position_scale_rotation_plot",
        "velocity_plot",
        "acceleration_plot",
        "jerk_plot",
        "pivot_identifiability_plot",
        "phase_contact_sheet",
        "residual_summary",
    }
    assert required_artifacts <= set(evidence.artifacts)
    assert all(Path(evidence.artifacts[item]).is_file() for item in required_artifacts)

    with pytest.raises(FileExistsError):
        analyze_reference_v2(
            fixture.source_path,
            fixture.video_path,
            output,
            readiness_path=readiness,
            motion_gate_path=motion_gate,
        )


def test_reference_v2_rejects_unmatched_or_failing_generic_gates(
    tmp_path: Path,
) -> None:
    definition = next(item for item in fixture_definitions() if item.id == "linear")
    fixture = render_motion_fixture(definition, tmp_path / "fixture")
    readiness, motion_gate = _gate_receipts(
        tmp_path,
        revision="b" * 40,
        motion_revision="c" * 40,
    )

    with pytest.raises(ReferenceV2GateError, match="same Git revision"):
        analyze_reference_v2(
            fixture.source_path,
            fixture.video_path,
            tmp_path / "rejected",
            readiness_path=readiness,
            motion_gate_path=motion_gate,
        )

    payload = json.loads(motion_gate.read_text(encoding="utf-8"))
    payload["git_revision"] = "b" * 40
    payload["overall"] = "fail"
    motion_gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceV2GateError, match="not passing"):
        analyze_reference_v2(
            fixture.source_path,
            fixture.video_path,
            tmp_path / "also-rejected",
            readiness_path=readiness,
            motion_gate_path=motion_gate,
        )


def _gate_receipts(
    root: Path,
    *,
    revision: str,
    motion_revision: str | None = None,
) -> tuple[Path, Path]:
    readiness = root / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "overall": "pass",
                "git_revision": revision,
                "reference_content_decoded": False,
                "run_root": str(root / "readiness"),
            }
        ),
        encoding="utf-8",
    )
    motion_gate = root / "motion-gate.json"
    motion_gate.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "id": "motiongate_test",
                "generated_at": "2026-07-19T00:00:00Z",
                "git_revision": motion_revision or revision,
                "git_clean": True,
                "corpus_path": str(root / "corpus.json"),
                "corpus_sha256": "0" * 64,
                "fixture_reports": {},
                "metrics": {},
                "checks": [],
                "mandatory_skips": 0,
                "provenance": {
                    "tool": "test",
                    "tool_version": "0",
                    "command": [],
                    "parameters": {},
                    "deterministic": True,
                    "generated_at": "2026-07-19T00:00:00Z",
                },
                "overall": "pass",
            }
        ),
        encoding="utf-8",
    )
    return readiness, motion_gate
