"""Reconstruct a frozen reference spec through normal product services."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import Intent, ReferenceReconstructionSpec, Scenario
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService


class ReconstructionError(RuntimeError):
    """A normal reconstruction stage failed a provenance or quality invariant."""


def run_reconstruction(
    spec: ReferenceReconstructionSpec,
    source_path: Path,
    output_root: Path,
    *,
    settings: Settings | None = None,
    project_name: str = "Reference reconstruction",
) -> dict[str, Any]:
    """Create, plan, render, QC, and export a new project from spec parameters."""

    selected = settings or Settings.from_env()
    source_path = source_path.expanduser().resolve(strict=True)
    source_hash = sha256_file(source_path)
    if source_hash != spec.source_sha256:
        raise ReconstructionError("source hash does not match the frozen spec")
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "reconstruction.json"
    if receipt_path.exists():
        raise ReconstructionError(f"evaluation root already contains a receipt: {receipt_path}")

    store = ProjectStore(selected)
    editing = EditingService(store)
    project = editing.create_project(
        project_name,
        Intent(
            text=(
                "Reconstruct the frozen observed camera motion from the permitted still "
                "without using reference media as source."
            ),
            scenario=Scenario.REFERENCE_RECONSTRUCT,
            target_duration_seconds=spec.duration_seconds,
            constraints=[
                "Reference video hash is forbidden in project and render inputs.",
                "Preserve the frozen comparison criteria.",
                "Use silence when the reference audio is silent.",
            ],
            preferences={
                "canvas": [spec.camera_motion.width, spec.camera_motion.height],
                "background": "#000000",
                "automation": "reviewable-treatment",
            },
        ),
    )
    editing.seal_reference(project.id, spec.reference_sha256)
    asset, probe = editing.ingest(project.id, source_path)
    derivatives = store.load_derivatives(project.id, asset.id)
    analyzed = AnalysisService(store).analyze(project.id, asset.id)
    treatment = editing.propose_reference(project.id, asset_id=asset.id, spec=spec)
    version = editing.accept_treatment(project.id, treatment.id)
    if version.decision_graph_id is None:
        raise ReconstructionError("accepted version has no editorial decision graph")

    profiles: dict[str, Any] = {}
    for profile in ("preview", "final"):
        job = RenderService(store).render(project.id, version.id, profile=profile)
        quality = QualityService(store).inspect(project.id, version.id, profile=profile)
        if quality.overall == "fail":
            raise ReconstructionError(f"{profile} render failed QC")
        project_root = store.project_path(project.id)
        output = project_root / job.output_paths[0]
        lineage_path = output.parent / "lineage.json"
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if lineage.get("input_hashes") != [source_hash]:
            raise ReconstructionError(f"unexpected {profile} render lineage")
        if spec.reference_sha256 in lineage.get("input_hashes", []):
            raise ReconstructionError("sealed reference reached render lineage")
        profiles[profile] = {
            "path": str(output),
            "sha256": sha256_file(output),
            "job_path": str(project_root / "jobs" / f"{job.id}.json"),
            "qc_path": str(project_root / "qc" / f"{quality.id}.json"),
            "qc": quality.model_dump(mode="json"),
            "lineage_path": str(lineage_path),
            "lineage": lineage,
        }

    kdenlive = KdenliveExporter(store).export(project.id, version.id)
    otio = OTIOExporter(store).export(project.id, version.id)
    project_root = store.project_path(project.id)
    receipt: dict[str, Any] = {
        "schema": "aoa_editing_reference_reconstruction_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "overall": "pass",
        "source_sha256": source_hash,
        "reference_sha256": spec.reference_sha256,
        "reference_media_used_as_input": False,
        "project_id": project.id,
        "project_root": str(project_root),
        "project_manifest": str(project_root / "project.json"),
        "asset_id": asset.id,
        "derived_media": derivatives.model_dump(mode="json"),
        "probe_evidence_id": probe.id,
        "analysis_evidence_ids": [item.id for item in analyzed],
        "frozen_spec_evidence": str(
            next(
                path
                for path in sorted((project_root / "evidence").glob("*.json"))
                if '"reference.reconstruction_spec"' in path.read_text(encoding="utf-8")
            )
        ),
        "treatment_id": treatment.id,
        "treatment_path": str(project_root / "treatments" / f"{treatment.id}.json"),
        "version_id": version.id,
        "approved_edit_graph": str(
            project_root / "decisions" / "graphs" / f"{version.decision_graph_id}.json"
        ),
        "decision_log": str(project_root / "decisions" / "log"),
        "timeline_projection": str(project_root / "versions" / f"{version.id}.json"),
        "profiles": profiles,
        "editable_exports": {
            "kdenlive": kdenlive.model_dump(mode="json"),
            "otio": otio.model_dump(mode="json"),
        },
        "hidden_manual_steps": [],
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return receipt
