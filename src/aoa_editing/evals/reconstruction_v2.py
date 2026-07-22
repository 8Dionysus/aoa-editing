"""Run one frozen Reference Spec v2 pass through normal product services."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from aoa_editing.analysis.phase_correction import derive_phase_correction_v2
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Intent,
    MotionRecoveryReport,
    Provenance,
    ReferenceComparisonReportV2,
    ReferencePhaseCorrectionV2,
    ReferenceReconstructionPassKindV2,
    ReferenceReconstructionPassReceiptV2,
    ReferenceReconstructionSpecV2,
    Scenario,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService
from aoa_editing.scenarios.reference_v2 import (
    PASS_SEQUENCE,
    classify_reconstruction_hypotheses_v2,
)


class ReconstructionV2Error(RuntimeError):
    """A v2 pass violated source isolation, application flow, or quality."""


def run_reconstruction_pass_v2(
    spec_path: Path,
    source_path: Path,
    output_root: Path,
    *,
    pass_kind: ReferenceReconstructionPassKindV2,
    comparison_path: Path | None = None,
    approved_phase_correction: ReferencePhaseCorrectionV2 | None = None,
    settings: Settings | None = None,
    project_name: str | None = None,
) -> ReferenceReconstructionPassReceiptV2:
    """Plan, approve, render, QC, and export one explainable reconstruction pass."""

    selected_spec_path = spec_path.expanduser().resolve(strict=True)
    selected_source = source_path.expanduser().resolve(strict=True)
    selected_output = output_root.expanduser().resolve()
    if selected_output.exists():
        raise FileExistsError(f"reconstruction v2 pass root already exists: {selected_output}")
    spec = ReferenceReconstructionSpecV2.model_validate_json(
        selected_spec_path.read_text(encoding="utf-8")
    )
    source_hash = sha256_file(selected_source)
    if source_hash != spec.source_sha256:
        raise ReconstructionV2Error("source hash does not match the frozen spec v2")

    phase_correction: ReferencePhaseCorrectionV2 | None = None
    phase_correction_source: str | None = None
    selected_comparison_path: Path | None = None
    selected_candidate_motion_path: Path | None = None
    if pass_kind == "phase_compensated_matrix":
        if comparison_path is not None and approved_phase_correction is not None:
            raise ReconstructionV2Error(
                "phase-compensated pass accepts either a prior comparison or "
                "approved correction semantics, not both"
            )
        if comparison_path is None and approved_phase_correction is None:
            raise ReconstructionV2Error(
                "phase-compensated pass requires a prior comparison or approved "
                "correction semantics"
            )
        if approved_phase_correction is not None:
            phase_correction = approved_phase_correction
            _validate_approved_phase_correction(
                phase_correction,
                spec=spec,
                spec_sha256=sha256_file(selected_spec_path),
            )
            phase_correction_source = "approved_semantics"
        else:
            if comparison_path is None:  # pragma: no cover - guarded above
                raise AssertionError("phase-compensated comparison path disappeared")
            selected_comparison_path = comparison_path.expanduser().resolve(strict=True)
            comparison = ReferenceComparisonReportV2.model_validate_json(
                selected_comparison_path.read_text(encoding="utf-8")
            )
            candidate_motion_ref = comparison.artifacts.get("candidate_motion")
            if not candidate_motion_ref:
                raise ReconstructionV2Error(
                    "Comparison v2 report has no candidate motion artifact"
                )
            selected_candidate_motion_path = _resolve_artifact(
                candidate_motion_ref,
                relative_to=selected_comparison_path.parent,
            )
            recovery = MotionRecoveryReport.model_validate_json(
                selected_candidate_motion_path.read_text(encoding="utf-8")
            )
            phase_correction = derive_phase_correction_v2(
                spec,
                comparison,
                recovery,
                spec_path=str(selected_spec_path),
                spec_sha256=sha256_file(selected_spec_path),
                comparison_path=str(selected_comparison_path),
                comparison_sha256=sha256_file(selected_comparison_path),
                candidate_motion_path=str(selected_candidate_motion_path),
                candidate_motion_sha256=sha256_file(selected_candidate_motion_path),
            )
            phase_correction_source = "derived_from_comparison"
    elif comparison_path is not None or approved_phase_correction is not None:
        raise ReconstructionV2Error(
            "comparison or approved correction semantics may only drive the "
            "phase-compensated pass"
        )

    selected_output.mkdir(parents=True, exist_ok=False)
    phase_correction_path: Path | None = None
    if phase_correction is not None:
        phase_correction_path = selected_output / "phase-correction-v2.json"
        with phase_correction_path.open("x", encoding="utf-8") as stream:
            stream.write(phase_correction.model_dump_json(indent=2))
            stream.write("\n")
    selected = settings or Settings.from_env()
    store = ProjectStore(selected)
    editing = EditingService(store)
    sequence = PASS_SEQUENCE[pass_kind]
    project = editing.create_project(
        project_name or f"Reference reconstruction v2 pass {sequence}: {pass_kind}",
        Intent(
            text=(
                f"Execute bounded reconstruction v2 pass {sequence} ({pass_kind}) from "
                "the frozen motion spec and the permitted still only."
            ),
            scenario=Scenario.REFERENCE_RECONSTRUCT,
            target_duration_seconds=spec.duration_seconds,
            constraints=[
                "Reference video hash is forbidden in project and render inputs.",
                "Do not alter the frozen comparison thresholds.",
                "Use the ordinary Treatment and Decision Graph approval workflow.",
                "Use silence when the frozen reference audio contract is silent.",
            ],
            preferences={
                "canvas": [spec.width, spec.height],
                "background": "#000000",
                "automation": "reviewable-treatment",
                "reconstruction_pass": pass_kind,
            },
        ),
    )
    editing.seal_reference(project.id, spec.reference_sha256)
    asset, _ = editing.ingest(project.id, selected_source)
    AnalysisService(store).analyze(project.id, asset.id)
    treatment = editing.propose_reference_v2(
        project.id,
        asset_id=asset.id,
        spec=spec,
        pass_kind=pass_kind,
        phase_correction=phase_correction,
    )
    if treatment.decision_graph_id is None:
        raise ReconstructionV2Error("saved treatment has no proposed decision graph")
    version_record = editing.accept_treatment(project.id, treatment.id)
    if version_record.decision_graph_id is None:
        raise ReconstructionV2Error("accepted version has no approved decision graph")
    project_root = store.project_path(project.id)
    approved = store.load_decision_graph(project.id, version_record.decision_graph_id)
    if approved.state != "approved" or approved.parent_graph_id != treatment.decision_graph_id:
        raise ReconstructionV2Error("decision graph transition is not proposed to approved")

    profiles: dict[str, dict[str, Any]] = {}
    for profile in ("preview", "final"):
        job = RenderService(store).render(
            project.id,
            version_record.id,
            profile=profile,
        )
        quality = QualityService(store).inspect(
            project.id,
            version_record.id,
            profile=profile,
        )
        if quality.overall == "fail":
            raise ReconstructionV2Error(f"{profile} render failed QC")
        output = project_root / job.output_paths[0]
        lineage_path = output.parent / "lineage.json"
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if lineage.get("input_hashes") != [source_hash]:
            raise ReconstructionV2Error(f"unexpected {profile} render lineage")
        if spec.reference_sha256 in lineage.get("input_hashes", []):
            raise ReconstructionV2Error("sealed reference reached render lineage")
        profiles[profile] = {
            "path": str(output),
            "sha256": sha256_file(output),
            "job_path": str(project_root / "jobs" / f"{job.id}.json"),
            "qc_path": str(project_root / "qc" / f"{quality.id}.json"),
            "qc": quality.model_dump(mode="json"),
            "lineage_path": str(lineage_path),
            "lineage": lineage,
        }

    kdenlive = KdenliveExporter(store).export(project.id, version_record.id)
    otio = OTIOExporter(store).export(project.id, version_record.id)
    evidence_path = next(
        path
        for path in sorted((project_root / "evidence").glob("*.json"))
        if '"reference.reconstruction_spec.v2"' in path.read_text(encoding="utf-8")
    )
    phase_correction_evidence: Path | None = None
    if phase_correction is not None:
        phase_correction_evidence = next(
            path
            for path in sorted((project_root / "evidence").glob("*.json"))
            if '"reference.phase_correction.v2"' in path.read_text(encoding="utf-8")
        )
    receipt = ReferenceReconstructionPassReceiptV2(
        pass_kind=pass_kind,
        sequence=sequence,
        spec_id=spec.id,
        spec_path=str(selected_spec_path),
        spec_sha256=sha256_file(selected_spec_path),
        source_sha256=source_hash,
        reference_sha256=spec.reference_sha256,
        project_id=project.id,
        project_root=str(project_root),
        project_manifest=str(project_root / "project.json"),
        asset_id=asset.id,
        frozen_spec_evidence=str(evidence_path),
        treatment_id=treatment.id,
        treatment_path=str(project_root / "treatments" / f"{treatment.id}.json"),
        proposed_decision_graph=str(
            project_root
            / "decisions"
            / "graphs"
            / f"{treatment.decision_graph_id}.json"
        ),
        version_id=version_record.id,
        approved_edit_graph=str(
            project_root
            / "decisions"
            / "graphs"
            / f"{version_record.decision_graph_id}.json"
        ),
        timeline_projection=str(project_root / "versions" / f"{version_record.id}.json"),
        profiles=profiles,
        editable_exports={
            "kdenlive": kdenlive.model_dump(mode="json"),
            "otio": otio.model_dump(mode="json"),
        },
        hypotheses=classify_reconstruction_hypotheses_v2(spec),
        phase_correction=phase_correction,
        phase_correction_path=(
            str(phase_correction_path)
            if phase_correction_path is not None
            else None
        ),
        phase_correction_evidence=(
            str(phase_correction_evidence)
            if phase_correction_evidence is not None
            else None
        ),
        hidden_manual_steps=[],
        provenance=Provenance(
            tool="aoa-editing-reference-reconstruction-pass-v2",
            tool_version=version("aoa-editing"),
            command=[],
            parameters={
                "pass_kind": pass_kind,
                "pass_sequence": sequence,
                "application_path": [
                    "create_project",
                    "seal_reference",
                    "ingest",
                    "analyze",
                    "propose_reference_v2",
                    "accept_treatment",
                    "render",
                    "quality",
                    "interchange",
                ],
                "reference_media_used_as_input": False,
                "comparison_report": (
                    str(selected_comparison_path)
                    if selected_comparison_path is not None
                    else None
                ),
                "candidate_motion_report": (
                    str(selected_candidate_motion_path)
                    if selected_candidate_motion_path is not None
                    else None
                ),
                "phase_correction_id": (
                    phase_correction.id if phase_correction is not None else None
                ),
                "phase_correction_source": phase_correction_source,
            },
            deterministic=True,
        ),
        overall="pass",
    )
    receipt_path = selected_output / "reconstruction-pass-v2.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(receipt.model_dump_json(indent=2))
        stream.write("\n")
    return receipt


def _validate_approved_phase_correction(
    correction: ReferencePhaseCorrectionV2,
    *,
    spec: ReferenceReconstructionSpecV2,
    spec_sha256: str,
) -> None:
    if (
        correction.spec_id != spec.id
        or correction.spec_sha256 != spec_sha256
        or correction.source_sha256 != spec.source_sha256
        or correction.reference_sha256 != spec.reference_sha256
        or correction.duration_frames != spec.duration_frames
        or correction.reference_media_used_as_render_input
        or correction.candidate_media_used_as_render_input
    ):
        raise ReconstructionV2Error(
            "approved phase correction does not match the frozen spec"
        )


def _resolve_artifact(reference: str, *, relative_to: Path) -> Path:
    selected = Path(reference).expanduser()
    if selected.is_absolute():
        return selected.resolve(strict=True)
    candidates = (
        (Path.cwd() / selected),
        (relative_to / selected),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise ReconstructionV2Error(f"comparison artifact does not exist: {reference}")
