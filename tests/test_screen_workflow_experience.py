from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aoa_editing.application.screen_workflow_experience import (
    admit_screen_workflow_experience,
)
from aoa_editing.config import Settings
from aoa_editing.domain.models import ScreenWorkflowExperienceAdmission


def _payload() -> dict[str, object]:
    return {
        "id": "reviewed-demo-20260721",
        "source_kind": "session-memory-manual-review",
        "source_record_id": "private-session-id",
        "source_snapshot_sha256": "a" * 64,
        "review_packet_sha256": "b" * 64,
        "review_wave_id": "demo-review-wave",
        "recurrence": "recurring-project-demos",
        "reviewed_by": "fixture-owner",
        "reviewed_at": datetime(2026, 7, 21, tzinfo=UTC).isoformat(),
        "review_note": "The operator compared two revisions and admitted bounded lessons.",
        "public_safe_projection_checked": True,
        "claims": [
            {
                "id": "prompt-once-readable",
                "disposition": "accepted",
                "source_observation": "Private operator wording and local evidence context.",
                "public_summary": "Show prompt entry once and retain the completed text.",
                "public_limitation": "Observed in one terminal workflow.",
                "evidence_refs": ["raw:line:10", "raw:line:12", "segment:002"],
                "owner_review_refs": ["raw:line:12"],
                "target_surfaces": ["capture-plan", "qc"],
                "public_case_id": "prompt-once-readable",
            },
            {
                "id": "full-frame-loop",
                "disposition": "deferred",
                "source_observation": "A loop could keep periodic UI alive but repeat timers.",
                "public_summary": "Evaluate region-aware live holds before implementation.",
                "public_limitation": "No generic seam and monotonic-timer proof exists.",
                "evidence_refs": ["raw:line:20", "raw:line:21"],
                "owner_review_refs": ["raw:line:21"],
                "target_surfaces": ["eval"],
            },
        ],
        "provenance": {
            "tool": "fixture-session-review",
            "tool_version": "1",
            "deterministic": False,
        },
    }


def test_private_experience_admission_writes_content_addressed_receipt(tmp_path) -> None:
    admission = ScreenWorkflowExperienceAdmission.model_validate(_payload())
    settings = Settings.for_home(tmp_path / "editing-home")

    receipt, path = admit_screen_workflow_experience(admission, settings=settings)
    repeated, repeated_path = admit_screen_workflow_experience(admission, settings=settings)

    assert path == repeated_path
    assert receipt == repeated
    assert path.parent == settings.state_root / "workflow-experience-receipts"
    assert path.name == f"{receipt.id}.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["admission"]["claims"][0]["evidence_refs"] == [
        "raw:line:10",
        "raw:line:12",
        "segment:002",
    ]
    public = receipt.public_projection.model_dump_json()
    assert "private-session-id" not in public
    assert "raw:line" not in public
    assert "Private operator wording" not in public
    assert receipt.public_projection_sha256 in path.read_text(encoding="utf-8")


def test_accepted_experience_claim_requires_public_case_id() -> None:
    payload = _payload()
    claims = payload["claims"]
    assert isinstance(claims, list)
    accepted = claims[0]
    assert isinstance(accepted, dict)
    accepted.pop("public_case_id")

    with pytest.raises(ValidationError, match="requires a public case id"):
        ScreenWorkflowExperienceAdmission.model_validate(payload)
