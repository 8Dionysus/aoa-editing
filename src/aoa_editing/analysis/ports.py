"""Replaceable analyzer boundary used by the analysis orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from aoa_editing.domain.models import Asset, EvidenceRecord


class AnalyzerPort(Protocol):
    """One evidence capability; adapters must not mutate projects or timelines."""

    kind: str

    def analyze(self, project_id: str, asset: Asset) -> EvidenceRecord:
        """Return one immutable evidence record or raise a localized error."""
        ...


@dataclass(frozen=True)
class FunctionAnalyzerPort:
    """Small adapter useful for host integrations, plugins, and tests."""

    kind: str
    function: Callable[[str, Asset], EvidenceRecord]

    def analyze(self, project_id: str, asset: Asset) -> EvidenceRecord:
        return self.function(project_id, asset)
