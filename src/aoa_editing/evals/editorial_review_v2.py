"""Portable human-review attachment over the frozen objective comparator."""

from __future__ import annotations

from pathlib import Path

from aoa_editing.domain.models import (
    EditorialReviewV2,
    ReferenceComparisonProtocolV2,
    ReferenceComparisonReportV2,
    new_id,
    utc_now,
)
from aoa_editing.evals.comparison_v2 import (
    ComparisonV2Error,
    _combined_overall,
    _editorial,
    _write_new,
)
from aoa_editing.infrastructure.media import sha256_file

REQUIRED_REVIEW_ARTIFACT_ROLES = (
    "side_by_side",
    "phase_contact_sheet",
    "aligned_difference",
)


def record_editorial_review_v2(
    report_path: Path,
    protocol_path: Path,
    review_path: Path,
    output_path: Path,
) -> ReferenceComparisonReportV2:
    """Attach a role-portable human verdict without changing objective evidence."""

    selected_report = report_path.expanduser().resolve(strict=True)
    selected_protocol = protocol_path.expanduser().resolve(strict=True)
    selected_review = review_path.expanduser().resolve(strict=True)
    if output_path.exists():
        raise FileExistsError(f"reviewed comparison report already exists: {output_path}")

    report = ReferenceComparisonReportV2.model_validate_json(
        selected_report.read_text(encoding="utf-8")
    )
    protocol = ReferenceComparisonProtocolV2.model_validate_json(
        selected_protocol.read_text(encoding="utf-8")
    )
    review = EditorialReviewV2.model_validate_json(
        selected_review.read_text(encoding="utf-8")
    )
    if report.protocol_id != protocol.id:
        raise ComparisonV2Error("comparison report and protocol IDs differ")
    if report.protocol_sha256 != sha256_file(selected_protocol):
        raise ComparisonV2Error("comparison report protocol hash is stale")

    artifact_subset = {
        role: Path(report.artifacts[role])
        for role in REQUIRED_REVIEW_ARTIFACT_ROLES
    }
    reviewed_artifact_roles: set[str] = set()
    normalized_artifacts: list[str] = []
    for item in review.reviewed_artifacts:
        role = _artifact_role(item, artifact_subset)
        if role is None:
            normalized_artifacts.append(item)
            continue
        reviewed_artifact_roles.add(role)
        normalized_artifacts.append(str(artifact_subset[role]))

    normalized_review = review.model_copy(
        update={"reviewed_artifacts": normalized_artifacts}
    )
    editorial, editorial_check = _editorial(
        protocol,
        report.candidate_sha256,
        normalized_review,
        artifact_subset,
    )
    editorial = {
        **editorial,
        "reviewed_artifacts": review.reviewed_artifacts,
        "reviewed_artifact_roles": sorted(reviewed_artifact_roles),
        "required_artifacts": {
            role: str(path) for role, path in artifact_subset.items()
        },
    }
    editorial_check = editorial_check.model_copy(update={"measured": editorial})
    checks = [
        editorial_check if item.id == "editorial-human-review" else item
        for item in report.checks
    ]
    reviewed = report.model_copy(
        update={
            "id": new_id("comparison"),
            "generated_at": utc_now(),
            "supersedes_report_id": report.id,
            "editorial": editorial,
            "checks": checks,
            "human_review": review,
            "provenance": report.provenance.model_copy(
                update={
                    "parameters": {
                        **report.provenance.parameters,
                        "editorial_review_id": review.id,
                        "editorial_review_sha256": sha256_file(selected_review),
                    }
                }
            ),
            "overall": _combined_overall(report.objective_overall, editorial_check),
        }
    )
    _write_new(output_path, reviewed.model_dump_json(indent=2) + "\n")
    return reviewed


def _artifact_role(item: str, artifacts: dict[str, Path]) -> str | None:
    if item in artifacts:
        return item
    selected = Path(item).expanduser()
    resolved = selected.resolve(strict=False)
    for role, path in artifacts.items():
        if item == str(path) or resolved == path.expanduser().resolve(strict=False):
            return role
    return None
