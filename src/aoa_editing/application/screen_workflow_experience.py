"""Owner-reviewed experience admission for repeatable screen-workflow demos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    ScreenWorkflowExperienceAdmission,
    ScreenWorkflowExperiencePublicClaim,
    ScreenWorkflowExperiencePublicProjection,
    ScreenWorkflowExperienceReceipt,
)


class ScreenWorkflowExperienceError(ValueError):
    """The requested receipt would weaken immutability or provenance."""


def public_projection(
    admission: ScreenWorkflowExperienceAdmission,
) -> ScreenWorkflowExperiencePublicProjection:
    """Drop private observations and evidence refs while retaining dispositions."""

    return ScreenWorkflowExperiencePublicProjection(
        admission_id=admission.id,
        source_kind=admission.source_kind,
        recurrence=admission.recurrence,
        claims=[
            ScreenWorkflowExperiencePublicClaim(
                id=item.id,
                disposition=item.disposition,
                summary=item.public_summary,
                limitation=item.public_limitation,
                target_surfaces=item.target_surfaces,
                public_case_id=item.public_case_id,
            )
            for item in admission.claims
        ],
    )


def projection_sha256(projection: ScreenWorkflowExperiencePublicProjection) -> str:
    payload = json.dumps(
        projection.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def admission_sha256(admission: ScreenWorkflowExperienceAdmission) -> str:
    payload = json.dumps(
        admission.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def admit_screen_workflow_experience(
    admission: ScreenWorkflowExperienceAdmission,
    *,
    settings: Settings,
) -> tuple[ScreenWorkflowExperienceReceipt, Path]:
    """Persist one immutable private receipt under the product-owned state root."""

    projection = public_projection(admission)
    digest = projection_sha256(projection)
    admission_digest = admission_sha256(admission)
    receipt = ScreenWorkflowExperienceReceipt(
        id=f"workflowexperience_{admission_digest[:24]}",
        admission=admission,
        public_projection=projection,
        public_projection_sha256=digest,
        stored_at=admission.reviewed_at,
        immutable=True,
    )
    receipt_root = settings.state_root / "workflow-experience-receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    path = receipt_root / f"{receipt.id}.json"
    rendered = receipt.model_dump_json(indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ScreenWorkflowExperienceError(
                "an immutable workflow-experience receipt already exists with other bytes"
            )
        return receipt, path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
    return receipt, path
