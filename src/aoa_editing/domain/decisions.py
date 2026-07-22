"""Pure construction and transition logic for editorial decision graphs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from aoa_editing.domain.models import (
    DecisionAlternative,
    DecisionApproval,
    DecisionLogEntry,
    DecisionOrigin,
    EditorialDecision,
    EditorialDecisionGraph,
    EditPatch,
    FrameRange,
    PatchOperation,
    Treatment,
    new_id,
)


def proposed_graph(treatment: Treatment, *, graph_id: str | None = None) -> EditorialDecisionGraph:
    """Translate a treatment into an explicit evidence-linked decision proposal."""

    assets, source_ranges, timeline_ranges = _ranges(treatment.patch.operations)
    confidence = min(
        (item.confidence for item in treatment.rationale), default=0.5
    )
    rationale = " ".join(item.claim for item in treatment.rationale) or treatment.patch.rationale
    origin = DecisionOrigin.MODEL if treatment.planner.model_id else DecisionOrigin.RULE
    decision = EditorialDecision(
        operation_type="timeline.patch",
        source_asset_ids=assets,
        source_ranges=source_ranges,
        timeline_ranges=timeline_ranges,
        intent=treatment.summary,
        evidence_refs=treatment.patch.evidence_refs,
        rationale=rationale,
        confidence=confidence,
        uncertainties=[
            "Planner confidence is bounded by the weakest stated rationale."
        ],
        constraints=[treatment.patch.rationale],
        alternatives=[
            DecisionAlternative(description=description) for description in treatment.alternatives
        ],
        origin=origin,
        approval=DecisionApproval.PROPOSED,
        patch_id=treatment.patch.id,
    )
    return EditorialDecisionGraph(
        id=graph_id or new_id("decisiongraph"),
        project_id=treatment.project_id,
        base_version_id=treatment.base_version_id,
        treatment_id=treatment.id,
        state="proposed",
        decisions=[decision],
    )


def approve_graph(
    proposed: EditorialDecisionGraph,
    *,
    version_id: str,
    inverse_operations: list[PatchOperation],
    graph_id: str | None = None,
) -> EditorialDecisionGraph:
    """Create an immutable approved revision while preserving decision identities."""

    if proposed.state != "proposed":
        raise ValueError("only a proposed decision graph can be approved")
    if not inverse_operations and proposed.decisions:
        raise ValueError("approval requires an executable inverse")
    decisions = [
        decision.model_copy(
            update={
                "approval": DecisionApproval.APPROVED,
                "reversible_operations": inverse_operations,
            }
        )
        for decision in proposed.decisions
    ]
    return EditorialDecisionGraph(
        id=graph_id or new_id("decisiongraph"),
        project_id=proposed.project_id,
        base_version_id=proposed.base_version_id,
        version_id=version_id,
        treatment_id=proposed.treatment_id,
        parent_graph_id=proposed.id,
        state="approved",
        decisions=decisions,
        edges=proposed.edges,
    )


def approved_patch_graph(
    *,
    project_id: str,
    base_version_id: str,
    parent_graph_id: str | None,
    version_id: str,
    patch: EditPatch,
    origin: DecisionOrigin,
    intent: str,
) -> EditorialDecisionGraph:
    """Record an already-reviewed direct UI/CLI/agent patch as canonical decisions."""

    if not patch.inverse_operations:
        raise ValueError("approved direct patch requires inverse operations")
    assets, source_ranges, timeline_ranges = _ranges(patch.operations)
    decision = EditorialDecision(
        operation_type="timeline.patch",
        source_asset_ids=assets,
        source_ranges=source_ranges,
        timeline_ranges=timeline_ranges,
        intent=intent,
        evidence_refs=patch.evidence_refs,
        rationale=patch.rationale,
        confidence=1.0 if origin is DecisionOrigin.HUMAN else 0.8,
        origin=origin,
        approval=DecisionApproval.APPROVED,
        patch_id=patch.id,
        reversible_operations=patch.inverse_operations,
    )
    return EditorialDecisionGraph(
        project_id=project_id,
        base_version_id=base_version_id,
        version_id=version_id,
        parent_graph_id=parent_graph_id,
        state="approved",
        decisions=[decision],
    )


def initial_graph(project_id: str, version_id: str) -> EditorialDecisionGraph:
    """Represent an empty initial timeline without inventing editorial decisions."""

    return EditorialDecisionGraph(
        project_id=project_id,
        version_id=version_id,
        state="approved",
    )


def graph_events(
    graph: EditorialDecisionGraph, *, rationale: str
) -> list[DecisionLogEntry]:
    """Create append-only events for every graph transition."""

    action = graph.state
    actor = (
        graph.decisions[0].origin if graph.decisions else DecisionOrigin.RULE
    )
    decision_ids: Iterable[str | None] = (
        (item.id for item in graph.decisions) if graph.decisions else [None]
    )
    return [
        DecisionLogEntry(
            project_id=graph.project_id,
            graph_id=graph.id,
            decision_id=decision_id,
            action=action,
            actor=actor,
            version_id=graph.version_id,
            rationale=rationale,
        )
        for decision_id in decision_ids
    ]


def _ranges(
    operations: list[PatchOperation],
) -> tuple[list[str], dict[str, list[FrameRange]], list[FrameRange]]:
    assets: set[str] = set()
    source_ranges: dict[str, list[FrameRange]] = {}
    timeline_ranges: list[FrameRange] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            asset_id = value.get("asset_id")
            if isinstance(asset_id, str):
                assets.add(asset_id)
                source = value.get("source_range")
                if isinstance(source, dict):
                    source_ranges.setdefault(asset_id, []).append(FrameRange.model_validate(source))
                timeline = value.get("timeline_range")
                if isinstance(timeline, dict):
                    timeline_ranges.append(FrameRange.model_validate(timeline))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for operation in operations:
        visit(operation.value)
    return sorted(assets), source_ranges, timeline_ranges
