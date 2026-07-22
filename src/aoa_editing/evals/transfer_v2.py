"""Heterogeneous positive/negative transfer gate for a portable motion technique."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from aoa_editing.analysis.reference import measure_video_motion
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Asset,
    CheckResult,
    DecomposedTransformMotionV2,
    EditorialDecisionGraph,
    Intent,
    ProjectVersion,
    Provenance,
    RenderPlan,
    Scenario,
    TechniqueApplicationRecord,
    TechniqueCompositionEvidenceV2,
    TechniquePacket,
    TechniqueProposalResultV2,
    TechniqueTransferCaseV2,
    TechniqueTransferCorpusReportV2,
    TransformEffectV2,
)
from aoa_editing.evals.fixtures import create_transfer_corpus_fixtures
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.knowledge.techniques import add_application_records
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService


class TransferCorpusV2Error(RuntimeError):
    """The v2 transfer corpus could not produce a fail-closed receipt."""


@dataclass(frozen=True, slots=True)
class TransferSourceV2:
    case_id: str
    source_class: str
    path: Path
    expected_outcome: Literal["eligible", "refused"]
    expected_refusal_code: Literal[
        "insufficient_source_resolution",
        "inapplicable_composition",
    ] | None
    single_global_transform_sufficient: bool
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PreparedApplication:
    store: ProjectStore
    editing: EditingService
    project_id: str
    asset: Asset
    proposal: TechniqueProposalResultV2


def run_technique_transfer_corpus_v2(
    packet: TechniquePacket,
    output_root: Path,
    *,
    photograph_path: Path,
    photograph_provenance: dict[str, Any],
    runtime_settings: Settings | None = None,
    canvas_width: int = 270,
    canvas_height: int = 480,
    forbidden_hashes: list[str] | None = None,
    implementation_revision: str | None = None,
    git_clean: bool | None = None,
) -> TechniqueTransferCorpusReportV2:
    """Apply one packet across five positive and two fail-closed negative classes."""

    if packet.schema_version != "2.0.0" or packet.revision != 3:
        raise TransferCorpusV2Error("transfer corpus v2 requires an upgraded revision-three packet")
    if packet.status != "candidate":
        raise TransferCorpusV2Error("transfer corpus cannot run on promoted knowledge")
    recipe = packet.portable_camera_motion_v2
    if recipe is None:
        raise TransferCorpusV2Error("packet lacks portable motion v2")
    if output_root.exists():
        raise TransferCorpusV2Error(f"transfer corpus root already exists: {output_root}")
    photograph_path = photograph_path.expanduser().resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=False)
    input_packet_path = output_root / "input-technique-v2-revision3.json"
    input_packet_path.write_text(
        packet.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    selected_settings = runtime_settings or Settings.from_env()
    forbidden = sorted(set(forbidden_hashes or []))
    repo = Path(__file__).resolve().parents[3]
    revision, clean = _revision_state(
        repo,
        implementation_revision=implementation_revision,
        git_clean=git_clean,
    )

    generated = create_transfer_corpus_fixtures(output_root / "fixtures")
    sources = _sources(generated, photograph_path, photograph_provenance)
    fixture_manifest_path = output_root / "fixture-manifest-v2.json"
    fixture_manifest = {
        "schema": "aoa_editing_transfer_fixture_manifest_v2",
        "sources": [
            {
                "case_id": item.case_id,
                "source_class": item.source_class,
                "path": str(item.path),
                "sha256": sha256_file(item.path),
                "expected_outcome": item.expected_outcome,
                "expected_refusal_code": item.expected_refusal_code,
                "single_global_transform_sufficient": (
                    item.single_global_transform_sufficient
                ),
                "provenance": item.provenance,
            }
            for item in sources
        ],
    }
    fixture_manifest_path.write_text(
        json.dumps(fixture_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cases: list[TechniqueTransferCaseV2] = []
    for source in sources:
        case_root = output_root / "cases" / source.case_id
        case_root.mkdir(parents=True, exist_ok=False)
        cases.append(
            _run_case(
                packet,
                source,
                case_root,
                runtime_settings=selected_settings,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                forbidden_hashes=forbidden,
            )
        )

    records = [
        TechniqueApplicationRecord(
            case_id=case.case_id,
            source_class=case.source_class,
            outcome=case.overall,
            evidence_path=f"runtime-eval:transfer-corpus-v2/cases/{case.case_id}/case-v2.json",
            metrics={
                "eligible": 1.0 if case.application_outcome == "eligible" else 0.0,
                "honest_refusal": 1.0 if case.application_outcome == "refused" else 0.0,
                "effective_upscale": float(
                    case.applicability.measured["effective_upscale"]
                ),
                "deterministic_replay": float(
                    bool(case.deterministic_replay.get("equivalent"))
                ),
                "maximum_matrix_error": float(
                    case.motion.get("maximum_matrix_absolute_error", 0.0)
                ),
            },
        )
        for case in cases
    ]
    updated_packet = add_application_records(packet, records)
    packet_path = output_root / "technique-candidate-with-transfer-v2.json"
    packet_path.write_text(
        updated_packet.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    packet_sha256 = sha256_file(packet_path)

    checks = _corpus_checks(
        packet,
        updated_packet,
        cases,
        sources,
        packet_path,
        forbidden,
        revision=revision,
        git_clean=clean,
        core_paths=[
            repo / "src" / "aoa_editing" / "application" / "service.py",
            repo / "src" / "aoa_editing" / "scenarios" / "planners.py",
            repo / "src" / "aoa_editing" / "knowledge" / "techniques.py",
        ],
    )
    overall: Literal["pass", "fail"] = (
        "pass"
        if all(item.status == "pass" for item in checks)
        and all(item.overall == "pass" for item in cases)
        else "fail"
    )
    report_path = output_root / "transfer-corpus-v2.json"
    report = TechniqueTransferCorpusReportV2(
        implementation_revision=revision,
        git_clean=clean,
        technique_id=packet.id,
        input_technique_revision=packet.revision,
        output_technique_revision=updated_packet.revision,
        cases=cases,
        checks=checks,
        artifacts={
            "report": str(report_path),
            "fixture_manifest": str(fixture_manifest_path),
            "input_candidate_packet": str(input_packet_path),
            "updated_candidate_packet": str(packet_path),
        },
        packet_sha256=packet_sha256,
        forbidden_media_hashes=forbidden,
        provenance=Provenance(
            tool="aoa-editing-technique-transfer-corpus-v2",
            tool_version=version("aoa-editing"),
            parameters={
                "canvas": [canvas_width, canvas_height],
                "positive_case_count": 5,
                "negative_case_count": 2,
                "normal_application_service": True,
                "reference_media_used_as_input": False,
                "candidate_status_after_gate": updated_packet.status,
            },
            deterministic=True,
        ),
        overall=overall,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def _sources(
    generated: dict[str, Path],
    photograph_path: Path,
    photograph_provenance: dict[str, Any],
) -> list[TransferSourceV2]:
    synthetic = {
        "owner": "aoa-editing",
        "authority": "deterministic_fixture_generator",
        "license": "project-generated-evaluation-fixture",
        "generator": "aoa-editing-transfer-fixtures-v2",
    }
    return [
        TransferSourceV2(
            case_id="square_illustration",
            source_class="square illustration",
            path=generated["square_illustration"],
            expected_outcome="eligible",
            expected_refusal_code=None,
            single_global_transform_sufficient=True,
            provenance={**synthetic, "seed": "0xA0A13"},
        ),
        TransferSourceV2(
            case_id="photograph",
            source_class="licensed operating-system photograph",
            path=photograph_path,
            expected_outcome="eligible",
            expected_refusal_code=None,
            single_global_transform_sufficient=True,
            provenance=photograph_provenance,
        ),
        TransferSourceV2(
            case_id="portrait_source",
            source_class="portrait source",
            path=generated["portrait_source"],
            expected_outcome="eligible",
            expected_refusal_code=None,
            single_global_transform_sufficient=True,
            provenance={**synthetic, "seed": "0xA0A14"},
        ),
        TransferSourceV2(
            case_id="alpha_source",
            source_class="alpha source",
            path=generated["alpha_source"],
            expected_outcome="eligible",
            expected_refusal_code=None,
            single_global_transform_sufficient=True,
            provenance={**synthetic, "seed": "0xA0A15", "has_alpha": True},
        ),
        TransferSourceV2(
            case_id="low_resolution",
            source_class="low-resolution source",
            path=generated["low_resolution"],
            expected_outcome="refused",
            expected_refusal_code="insufficient_source_resolution",
            single_global_transform_sufficient=True,
            provenance={**synthetic, "generator_case": "low-resolution-48px"},
        ),
        TransferSourceV2(
            case_id="complex_texture",
            source_class="complex texture",
            path=generated["complex_texture"],
            expected_outcome="eligible",
            expected_refusal_code=None,
            single_global_transform_sufficient=True,
            provenance={**synthetic, "seed": "0xA0A16"},
        ),
        TransferSourceV2(
            case_id="inapplicable_composition",
            source_class="composition requiring independent layer motion",
            path=generated["inapplicable_composition"],
            expected_outcome="refused",
            expected_refusal_code="inapplicable_composition",
            single_global_transform_sufficient=False,
            provenance={
                **synthetic,
                "generator_case": "two-independent-panels",
                "composition_ground_truth": "independent_layer_motion_required",
            },
        ),
    ]


def _run_case(
    packet: TechniquePacket,
    source: TransferSourceV2,
    case_root: Path,
    *,
    runtime_settings: Settings,
    canvas_width: int,
    canvas_height: int,
    forbidden_hashes: list[str],
) -> TechniqueTransferCaseV2:
    source_before = sha256_file(source.path)
    prepared = _prepare_application(
        packet,
        source,
        case_root / "workspace",
        runtime_settings=runtime_settings,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        forbidden_hashes=forbidden_hashes,
    )
    decision_path = case_root / "applicability-decision-v2.json"
    decision_path.write_text(
        prepared.proposal.decision.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if source.expected_outcome == "refused":
        case = _refused_case(
            packet,
            source,
            prepared,
            case_root,
            decision_path,
            runtime_settings=runtime_settings,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            forbidden_hashes=forbidden_hashes,
            source_before=source_before,
        )
    else:
        case = _eligible_case(
            packet,
            source,
            prepared,
            case_root,
            decision_path,
            runtime_settings=runtime_settings,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            forbidden_hashes=forbidden_hashes,
            source_before=source_before,
        )
    case_path = case_root / "case-v2.json"
    completed = case.model_copy(
        update={"artifacts": {**case.artifacts, "report": str(case_path)}}
    )
    case_path.write_text(completed.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return completed


def _prepare_application(
    packet: TechniquePacket,
    source: TransferSourceV2,
    workspace: Path,
    *,
    runtime_settings: Settings,
    canvas_width: int,
    canvas_height: int,
    forbidden_hashes: list[str],
) -> _PreparedApplication:
    recipe = packet.portable_camera_motion_v2
    if recipe is None:  # pragma: no cover - top-level invariant
        raise TransferCorpusV2Error("packet lost its portable recipe")
    settings = Settings.for_home(
        workspace,
        runtime_root=runtime_settings.runtime_root,
        bind_port=runtime_settings.bind_port,
    )
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        f"Technique transfer v2: {source.source_class}",
        Intent(
            text="Evaluate a source-neutral camera technique on an independent still.",
            scenario=Scenario.STILL_MOTION,
            target_duration_seconds=recipe.duration_frames / recipe.frame_rate.fps,
            frame_format=f"{recipe.aspect_width}:{recipe.aspect_height}",
            privacy_mode="local-only",
            automation_level="review-before-apply",
            constraints=[
                "Use only the ingested independent source.",
                "Refuse before Treatment when applicability constraints fail.",
            ],
        ),
    )
    for digest in forbidden_hashes:
        editing.seal_reference(project.id, digest)
    asset, _ = editing.ingest(project.id, source.path)
    AnalysisService(store).analyze(project.id, asset.id)
    editing.record_technique_composition_v2(
        project.id,
        asset_id=asset.id,
        assessment=TechniqueCompositionEvidenceV2(
            source_class=source.source_class,
            single_global_transform_sufficient=(
                source.single_global_transform_sufficient
            ),
            independent_layer_motion_required=(
                not source.single_global_transform_sufficient
            ),
            authority="fixture_ground_truth",
            rationale=(
                "The transfer manifest defines whether one global similarity "
                "transform can express the intended composition."
            ),
        ),
    )
    proposal = editing.propose_technique_v2(
        project.id,
        asset_id=asset.id,
        technique=packet,
        width=canvas_width,
        height=canvas_height,
    )
    return _PreparedApplication(
        store=store,
        editing=editing,
        project_id=project.id,
        asset=asset,
        proposal=proposal,
    )


def _refused_case(
    packet: TechniquePacket,
    source: TransferSourceV2,
    prepared: _PreparedApplication,
    case_root: Path,
    decision_path: Path,
    *,
    runtime_settings: Settings,
    canvas_width: int,
    canvas_height: int,
    forbidden_hashes: list[str],
    source_before: str,
) -> TechniqueTransferCaseV2:
    decision = prepared.proposal.decision
    replay = _prepare_application(
        packet,
        source,
        case_root / "replay-workspace",
        runtime_settings=runtime_settings,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        forbidden_hashes=forbidden_hashes,
    )
    project = prepared.store.load_project(prepared.project_id)
    render_root = prepared.store.project_path(prepared.project_id) / "renders"
    treatment_root = prepared.store.project_path(prepared.project_id) / "treatments"
    rendered_files = (
        [path for path in render_root.rglob("*") if path.is_file()]
        if render_root.exists()
        else []
    )
    treatment_files = (
        [path for path in treatment_root.rglob("*") if path.is_file()]
        if treatment_root.exists()
        else []
    )
    equivalent = (
        replay.proposal.decision.outcome == decision.outcome
        and replay.proposal.decision.refusal_code == decision.refusal_code
        and _decision_semantics(replay.proposal.decision.model_dump(mode="json"))
        == _decision_semantics(decision.model_dump(mode="json"))
    )
    checks = [
        _check(
            "expected-refusal",
            decision.outcome == "refused"
            and decision.refusal_code == source.expected_refusal_code
            and prepared.proposal.treatment is None,
            "typed application result matches the expected contraindication",
            {
                "actual_outcome": decision.outcome,
                "actual_code": decision.refusal_code,
                "expected_code": source.expected_refusal_code,
            },
        ),
        _check(
            "refusal-before-treatment",
            project.current_version_id is None
            and not rendered_files
            and not treatment_files,
            "refusal created neither canonical version, Treatment, nor render",
            {
                "current_version_id": project.current_version_id,
                "rendered_files": [str(path) for path in rendered_files],
                "treatment_files": [str(path) for path in treatment_files],
            },
        ),
        _check(
            "deterministic-refusal-replay",
            equivalent
            and replay.project_id != prepared.project_id
            and replay.proposal.treatment is None,
            "a distinct workspace produced equivalent refusal semantics",
            {
                "equivalent": equivalent,
                "project_ids": [prepared.project_id, replay.project_id],
            },
        ),
        _check(
            "independent-source",
            source_before not in forbidden_hashes,
            "refused case remains independent of sealed media",
            {"source_sha256": source_before},
        ),
        _check(
            "source-immutable",
            sha256_file(source.path) == source_before,
            "source bytes remain unchanged after both applicability passes",
            {"source_sha256": source_before},
        ),
    ]
    overall = _overall(checks)
    return TechniqueTransferCaseV2(
        case_id=source.case_id,
        source_class=source.source_class,
        expected_application_outcome=source.expected_outcome,
        application_outcome=decision.outcome,
        source_sha256=source_before,
        source_width=cast(int, prepared.asset.metadata.width),
        source_height=cast(int, prepared.asset.metadata.height),
        source_provenance=source.provenance,
        applicability=decision,
        project_id=prepared.project_id,
        deterministic_replay={
            "equivalent": equivalent,
            "project_id": replay.project_id,
            "outcome": replay.proposal.decision.outcome,
            "refusal_code": replay.proposal.decision.refusal_code,
        },
        checks=checks,
        artifacts={"applicability_decision": str(decision_path)},
        overall=overall,
    )


def _eligible_case(
    packet: TechniquePacket,
    source: TransferSourceV2,
    prepared: _PreparedApplication,
    case_root: Path,
    decision_path: Path,
    *,
    runtime_settings: Settings,
    canvas_width: int,
    canvas_height: int,
    forbidden_hashes: list[str],
    source_before: str,
) -> TechniqueTransferCaseV2:
    treatment = prepared.proposal.treatment
    if treatment is None:
        raise TransferCorpusV2Error(
            f"positive case {source.case_id} was refused: "
            f"{prepared.proposal.decision.refusal_code}"
        )
    edit_version = prepared.editing.accept_treatment(prepared.project_id, treatment.id)
    profiles: dict[str, Any] = {}
    plans: dict[str, RenderPlan] = {}
    profile_paths: dict[str, Path] = {}
    for profile_name in ("preview", "final"):
        job = RenderService(prepared.store).render(
            prepared.project_id,
            edit_version.id,
            profile=profile_name,
        )
        qc = QualityService(prepared.store).inspect(
            prepared.project_id,
            edit_version.id,
            profile=profile_name,
        )
        path = prepared.store.project_path(prepared.project_id) / job.output_paths[0]
        profile_paths[profile_name] = path
        lineage_path = path.parent / "lineage.json"
        plan_path = path.parent / "render-plan.json"
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        plan = RenderPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        plans[profile_name] = plan
        profiles[profile_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "qc": qc.model_dump(mode="json"),
            "lineage": lineage,
            "lineage_path": str(lineage_path),
            "render_plan": str(plan_path),
        }

    kdenlive = KdenliveExporter(prepared.store).export(
        prepared.project_id, edit_version.id
    )
    otio = OTIOExporter(prepared.store).export(prepared.project_id, edit_version.id)
    exports = {
        "kdenlive": {
            **kdenlive.model_dump(mode="json"),
            "resolved_path": str(
                prepared.store.project_path(prepared.project_id)
                / kdenlive.output_path
            ),
        },
        "otio": {
            **otio.model_dump(mode="json"),
            "resolved_path": str(
                prepared.store.project_path(prepared.project_id) / otio.output_path
            ),
        },
    }
    graph_path = (
        prepared.store.project_path(prepared.project_id)
        / "decisions"
        / "graphs"
        / f"{edit_version.decision_graph_id}.json"
    )
    graph = EditorialDecisionGraph.model_validate_json(
        graph_path.read_text(encoding="utf-8")
    )
    effect = _single_motion_effect(edit_version)
    motion = _motion_contract_metrics(
        packet,
        effect,
        plans["final"],
        source_width=cast(int, prepared.asset.metadata.width),
        source_height=cast(int, prepared.asset.metadata.height),
        output_width=canvas_width,
        output_height=canvas_height,
    )
    if source.case_id == "complex_texture":
        motion["decoded_media_remeasurement"] = _decoded_motion_metrics(
            packet,
            source.path,
            profile_paths["final"],
        )

    replay_prepared = _prepare_application(
        packet,
        source,
        case_root / "replay-workspace",
        runtime_settings=runtime_settings,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        forbidden_hashes=forbidden_hashes,
    )
    replay_treatment = replay_prepared.proposal.treatment
    if replay_treatment is None:
        raise TransferCorpusV2Error("eligible replay was refused")
    replay_version = replay_prepared.editing.accept_treatment(
        replay_prepared.project_id, replay_treatment.id
    )
    replay_job = RenderService(replay_prepared.store).render(
        replay_prepared.project_id,
        replay_version.id,
        profile="final",
    )
    replay_qc = QualityService(replay_prepared.store).inspect(
        replay_prepared.project_id,
        replay_version.id,
        profile="final",
    )
    replay_path = (
        replay_prepared.store.project_path(replay_prepared.project_id)
        / replay_job.output_paths[0]
    )
    replay_lineage = json.loads(
        (replay_path.parent / "lineage.json").read_text(encoding="utf-8")
    )
    deterministic = {
        "equivalent": sha256_file(replay_path) == profiles["final"]["sha256"],
        "project_id": replay_prepared.project_id,
        "version_id": replay_version.id,
        "path": str(replay_path),
        "sha256": sha256_file(replay_path),
        "qc": replay_qc.model_dump(mode="json"),
        "lineage": replay_lineage,
        "distinct_project": replay_prepared.project_id != prepared.project_id,
        "distinct_path": replay_path != profile_paths["final"],
        "distinct_inode": replay_path.stat().st_ino != profile_paths["final"].stat().st_ino,
    }
    contact_sheet = case_root / "contact-sheet.jpg"
    recipe = packet.portable_camera_motion_v2
    if recipe is None:  # pragma: no cover - caller invariant
        raise TransferCorpusV2Error("packet lost its portable recipe")
    _contact_sheet(
        profile_paths["final"],
        contact_sheet,
        duration_frames=recipe.duration_frames,
    )

    all_lineages = [
        profiles["preview"]["lineage"],
        profiles["final"]["lineage"],
        replay_lineage,
    ]
    decoded = cast(dict[str, Any], motion.get("decoded_media_remeasurement", {}))
    checks = [
        _check(
            "expected-eligibility",
            prepared.proposal.decision.outcome == "eligible",
            "typed applicability permits this positive case",
            {"outcome": prepared.proposal.decision.outcome},
        ),
        _check(
            "motion-continuity",
            motion["c1_declared"] is True
            and motion["maximum_center_step"] <= 0.02
            and motion["maximum_scale_step"] <= 0.05
            and motion["maximum_rotation_step_degrees"] <= 0.25,
            "dense normalized channels remain bounded and C1-authored",
            motion,
        ),
        _check(
            "compiled-matrix-equivalence",
            motion["frame_count"] == recipe.duration_frames
            and motion["maximum_matrix_absolute_error"] <= 1e-9,
            "every compiled compositor matrix matches the portable normalized recipe",
            {
                "frame_count": motion["frame_count"],
                "maximum_absolute_error": motion[
                    "maximum_matrix_absolute_error"
                ],
            },
        ),
        _check(
            "decoded-media-motion",
            source.case_id != "complex_texture"
            or (
                decoded.get("scale_rmse", math.inf) <= 0.03
                and decoded.get("rotation_rmse_degrees", math.inf) <= 0.30
                and decoded.get("center_rmse_normalized", math.inf) <= 0.02
            ),
            "feature-rich decoded output follows the authored motion",
            decoded if decoded else {"covered_by": "complex_texture_case"},
        ),
        _check(
            "resolution-and-aspect-adaptation",
            _decision_check_status(
                prepared.proposal.decision.model_dump(mode="json"),
                "resolution-guard",
            )
            == "pass"
            and recipe.aspect_adaptation
            == "normalized_contain_source_aspect_neutral",
            "source aspect is adapted by contain geometry within the pixel-density cap",
            {
                "source_aspect": cast(int, prepared.asset.metadata.width)
                / cast(int, prepared.asset.metadata.height),
                "effective_upscale": prepared.proposal.decision.measured[
                    "effective_upscale"
                ],
            },
        ),
        _check(
            "pivot-stability",
            motion["maximum_pivot_deviation"] <= 1e-12,
            "portable authoring keeps the normalized source-center pivot fixed",
            {"maximum_deviation": motion["maximum_pivot_deviation"]},
        ),
        _check(
            "source-only-lineage",
            all(
                item.get("input_hashes") == [source_before]
                and not set(forbidden_hashes).intersection(
                    item.get("input_hashes", [])
                )
                for item in all_lineages
            ),
            "preview, final, and replay lineage contain only this independent source",
            {"input_hashes": [item.get("input_hashes") for item in all_lineages]},
        ),
        _check(
            "approved-reversible-decision",
            graph.state == "approved"
            and bool(graph.decisions)
            and graph.decisions[0].patch_id == treatment.patch.id,
            "normal approval created a reversible editorial decision",
            {"graph_path": str(graph_path)},
        ),
        _check(
            "preview-final-qc",
            all(
                profiles[name]["qc"]["overall"] != "fail"
                for name in ("preview", "final")
            )
            and replay_qc.overall != "fail",
            "preview, final, and replay technical QC do not fail",
            {
                "preview": profiles["preview"]["qc"]["overall"],
                "final": profiles["final"]["qc"]["overall"],
                "replay": replay_qc.overall,
            },
        ),
        _check(
            "editable-exports",
            kdenlive.validator is not None and otio.validator is not None,
            "Kdenlive/MLT and OTIO projections validate",
            {
                "kdenlive": exports["kdenlive"]["resolved_path"],
                "otio": exports["otio"]["resolved_path"],
            },
        ),
        _check(
            "deterministic-replay",
            all(
                (
                    deterministic["equivalent"],
                    deterministic["distinct_project"],
                    deterministic["distinct_path"],
                    deterministic["distinct_inode"],
                )
            ),
            "a distinct project reproduced byte-equivalent final media",
            deterministic,
        ),
        _check(
            "source-immutable",
            sha256_file(source.path) == source_before,
            "independent source remains byte-identical",
            {"source_sha256": source_before},
        ),
    ]
    overall = _overall(checks)
    return TechniqueTransferCaseV2(
        case_id=source.case_id,
        source_class=source.source_class,
        expected_application_outcome=source.expected_outcome,
        application_outcome=prepared.proposal.decision.outcome,
        source_sha256=source_before,
        source_width=cast(int, prepared.asset.metadata.width),
        source_height=cast(int, prepared.asset.metadata.height),
        source_provenance=source.provenance,
        applicability=prepared.proposal.decision,
        project_id=prepared.project_id,
        version_id=edit_version.id,
        motion=motion,
        profiles=profiles,
        editable_exports=exports,
        deterministic_replay=deterministic,
        checks=checks,
        artifacts={
            "applicability_decision": str(decision_path),
            "approved_edit_graph": str(graph_path),
            "contact_sheet": str(contact_sheet),
        },
        overall=overall,
    )


def _single_motion_effect(version_value: ProjectVersion) -> TransformEffectV2:
    effects = [
        effect
        for track in version_value.timeline.tracks
        for clip in track.clips
        for effect in clip.effects
        if isinstance(effect, TransformEffectV2)
    ]
    if len(effects) != 1:
        raise TransferCorpusV2Error("transfer version must contain one Motion Language v2 effect")
    return effects[0]


def _motion_contract_metrics(
    packet: TechniquePacket,
    effect: TransformEffectV2,
    plan: RenderPlan,
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> dict[str, Any]:
    recipe = packet.portable_camera_motion_v2
    if recipe is None or not isinstance(effect.motion, DecomposedTransformMotionV2):
        raise TransferCorpusV2Error("portable effect is not decomposed motion")
    if len(plan.compositor_jobs) != 1:
        raise TransferCorpusV2Error("transfer render plan must contain one compositor job")
    job = plan.compositor_jobs[0]
    if len(job.frames) != recipe.duration_frames:
        raise TransferCorpusV2Error("compiled transfer frame count differs from the packet")
    errors: list[float] = []
    for frame, compiled in enumerate(job.frames):
        expected = _expected_similarity_matrix(
            center=recipe.centers[frame],
            scale=recipe.scales_relative_to_contain[frame],
            rotation_degrees=recipe.rotations_degrees[frame],
            pivot=recipe.pivot,
            source_width=source_width,
            source_height=source_height,
            output_width=output_width,
            output_height=output_height,
        )
        actual = np.asarray(compiled.matrices_3x3[0], dtype=np.float64)
        errors.append(float(np.max(np.abs(actual - expected))))
    center_steps = [
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(recipe.centers, recipe.centers[1:], strict=False)
    ]
    scale_steps = [
        abs(right - left)
        for left, right in zip(
            recipe.scales_relative_to_contain,
            recipe.scales_relative_to_contain[1:],
            strict=False,
        )
    ]
    rotation_steps = [
        abs(right - left)
        for left, right in zip(
            recipe.rotations_degrees,
            recipe.rotations_degrees[1:],
            strict=False,
        )
    ]
    pivot_deviations = [
        math.hypot(item.value.x - recipe.pivot[0], item.value.y - recipe.pivot[1])
        for item in effect.motion.pivot.keyframes
    ]
    return {
        "frame_count": len(job.frames),
        "curve_model": packet.curve_model_v2.kind if packet.curve_model_v2 else None,
        "c1_declared": (
            effect.motion.position.continuity == "c1"
            and effect.motion.scale.continuity == "c1"
            and effect.motion.rotation.continuity == "c1"
        ),
        "maximum_center_step": max(center_steps),
        "maximum_scale_step": max(scale_steps),
        "maximum_rotation_step_degrees": max(rotation_steps),
        "maximum_pivot_deviation": max(pivot_deviations),
        "maximum_matrix_absolute_error": max(errors),
    }


def _expected_similarity_matrix(
    *,
    center: tuple[float, float],
    scale: float,
    rotation_degrees: float,
    pivot: tuple[float, float],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    contain = min(output_width / source_width, output_height / source_height)
    radians = math.radians(rotation_degrees)
    cosine = math.cos(radians) * contain * scale
    sine = math.sin(radians) * contain * scale
    linear = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    source_pivot = np.asarray(
        (pivot[0] * source_width, pivot[1] * source_height),
        dtype=np.float64,
    )
    target = np.asarray(
        (center[0] * output_width, center[1] * output_height),
        dtype=np.float64,
    )
    translation = target - linear @ source_pivot
    return np.asarray(
        (
            (linear[0, 0], linear[0, 1], translation[0]),
            (linear[1, 0], linear[1, 1], translation[1]),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _decoded_motion_metrics(
    packet: TechniquePacket,
    source_path: Path,
    final_path: Path,
) -> dict[str, Any]:
    recipe = packet.portable_camera_motion_v2
    if recipe is None:  # pragma: no cover - caller invariant
        raise TransferCorpusV2Error("packet lost its portable motion")
    measured = measure_video_motion(source_path, final_path, sample_step=40)
    return {
        "sample_count": len(measured.measurements),
        "sample_frames": [item.frame for item in measured.measurements],
        "scale_rmse": _rmse(
            item.scale_relative_to_contain
            - recipe.scales_relative_to_contain[item.frame]
            for item in measured.measurements
        ),
        "rotation_rmse_degrees": _rmse(
            item.rotation_degrees - recipe.rotations_degrees[item.frame]
            for item in measured.measurements
        ),
        "center_rmse_normalized": _rmse(
            math.hypot(
                item.center_x - recipe.centers[item.frame][0],
                item.center_y - recipe.centers[item.frame][1],
            )
            for item in measured.measurements
        ),
    }


def _corpus_checks(
    input_packet: TechniquePacket,
    output_packet: TechniquePacket,
    cases: list[TechniqueTransferCaseV2],
    sources: list[TransferSourceV2],
    packet_path: Path,
    forbidden_hashes: list[str],
    *,
    revision: str,
    git_clean: bool,
    core_paths: list[Path],
) -> list[CheckResult]:
    by_id = {item.case_id: item for item in cases}
    source_hashes = [item.source_sha256 for item in cases]
    aspects = [item.source_width / item.source_height for item in cases]
    serialized_packet = packet_path.read_text(encoding="utf-8")
    core_text = "\n".join(path.read_text(encoding="utf-8") for path in core_paths)
    case_branch_tokens = [
        "square_illustration",
        "portrait_source",
        "alpha_source",
        "complex_texture",
    ]
    photograph = by_id["photograph"]
    alpha = by_id["alpha_source"]
    return [
        _check(
            "revision-bound-clean-run",
            git_clean and len(revision) == 40,
            "transfer corpus is bound to a clean implementation revision",
            {"implementation_revision": revision, "git_clean": git_clean},
        ),
        _check(
            "required-source-classes",
            len(cases) == 7
            and set(by_id)
            == {
                "square_illustration",
                "photograph",
                "portrait_source",
                "alpha_source",
                "low_resolution",
                "complex_texture",
                "inapplicable_composition",
            },
            "all five positive and two negative source classes are present",
            {"case_ids": sorted(by_id)},
        ),
        _check(
            "positive-transfer-cases",
            all(
                by_id[item].application_outcome == "eligible"
                and by_id[item].overall == "pass"
                for item in (
                    "square_illustration",
                    "photograph",
                    "portrait_source",
                    "alpha_source",
                    "complex_texture",
                )
            ),
            "illustration, photograph, portrait, alpha, and texture all transfer",
            {
                item.case_id: item.application_outcome
                for item in cases
                if item.expected_application_outcome == "eligible"
            },
        ),
        _check(
            "honest-negative-cases",
            by_id["low_resolution"].applicability.refusal_code
            == "insufficient_source_resolution"
            and by_id["inapplicable_composition"].applicability.refusal_code
            == "inapplicable_composition"
            and by_id["low_resolution"].overall == "pass"
            and by_id["inapplicable_composition"].overall == "pass",
            "resolution and composition contraindications refuse before rendering",
            {
                "low_resolution": by_id[
                    "low_resolution"
                ].applicability.refusal_code,
                "inapplicable_composition": by_id[
                    "inapplicable_composition"
                ].applicability.refusal_code,
            },
        ),
        _check(
            "source-diversity-and-provenance",
            len(source_hashes) == len(set(source_hashes))
            and min(aspects) < 0.8
            and max(aspects) > 1.2
            and alpha.source_provenance.get("has_alpha") is True
            and bool(photograph.source_provenance.get("license_authority"))
            and bool(photograph.source_provenance.get("owner")),
            "sources span portrait, square, landscape, alpha, and licensed photography",
            {
                "unique_hashes": len(set(source_hashes)),
                "aspects": aspects,
                "photograph_provenance": photograph.source_provenance,
            },
        ),
        _check(
            "independent-source-lineage",
            not set(source_hashes).intersection(forbidden_hashes)
            and all(
                case.reference_media_used_as_input is False for case in cases
            ),
            "no sealed source or reference hash entered any transfer case",
            {
                "source_hashes": source_hashes,
                "forbidden_hashes": forbidden_hashes,
            },
        ),
        _check(
            "no-target-specific-branches",
            not any(token in core_text for token in case_branch_tokens)
            and not any(digest in core_text for digest in forbidden_hashes),
            "application, planner, and technique core contain no per-case or target branch",
            {"core_paths": [str(path) for path in core_paths]},
        ),
        _check(
            "deterministic-replay",
            all(
                case.deterministic_replay.get("equivalent") is True for case in cases
            ),
            "every eligible render and every refusal repeats in a distinct workspace",
            {
                case.case_id: case.deterministic_replay.get("equivalent")
                for case in cases
            },
        ),
        _check(
            "candidate-packet-revision",
            output_packet.schema_version == "2.0.0"
            and output_packet.revision == input_packet.revision + 1
            and output_packet.status == "candidate"
            and len(output_packet.application_history)
            == len(input_packet.application_history) + len(cases)
            and all(record.outcome == "pass" for record in output_packet.application_history),
            "one atomic corpus revision records all cases without promotion",
            {
                "input_revision": input_packet.revision,
                "output_revision": output_packet.revision,
                "status": output_packet.status,
                "application_count": len(output_packet.application_history),
            },
        ),
        _check(
            "candidate-source-neutrality",
            not any(digest in serialized_packet for digest in forbidden_hashes)
            and "/home/" not in serialized_packet
            and "/srv/" not in serialized_packet,
            "candidate packet contains neither media hashes nor physical owner paths",
            {"packet_path": str(packet_path)},
        ),
        _check(
            "mandatory-skips",
            all(
                all(check.status != "skip" for check in case.checks)
                for case in cases
            ),
            "no transfer requirement was skipped",
            {"mandatory_skips": 0},
        ),
    ]


def system_photograph_provenance(path: Path) -> dict[str, Any]:
    """Read OS package authority for a photograph without guessing its license."""

    selected = path.expanduser().resolve(strict=True)
    owner = _run(["rpm", "-qf", str(selected)]).strip()
    metadata = _run(
        [
            "rpm",
            "-q",
            "--qf",
            "%{NAME}\\n%{VERSION}-%{RELEASE}\\n%{LICENSE}\\n%{VENDOR}\\n%{URL}\\n",
            owner,
        ]
    ).splitlines()
    if len(metadata) < 5:
        raise TransferCorpusV2Error("RPM did not return complete photograph provenance")
    with Image.open(selected) as image:
        dimensions = [image.width, image.height]
        format_name = image.format
    return {
        "owner": "operating_system_upstream",
        "authority": "installed_rpm_metadata",
        "package": metadata[0],
        "package_revision": metadata[1],
        "license_authority": metadata[2],
        "vendor": metadata[3],
        "upstream": metadata[4],
        "media_type": "photograph",
        "format": format_name,
        "dimensions": dimensions,
        "installed_path": str(selected),
        "sha256": sha256_file(selected),
    }


def _decision_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": payload.get("outcome"),
        "refusal_code": payload.get("refusal_code"),
        "checks": [
            {
                "id": item.get("id"),
                "status": item.get("status"),
                "measured": item.get("measured"),
            }
            for item in cast(list[dict[str, Any]], payload.get("checks", []))
        ],
        "measured": {
            key: value
            for key, value in cast(dict[str, Any], payload.get("measured", {})).items()
            if key != "source_sha256"
        },
    }


def _decision_check_status(payload: dict[str, Any], check_id: str) -> str | None:
    for item in cast(list[dict[str, Any]], payload.get("checks", [])):
        if item.get("id") == check_id:
            return cast(str | None, item.get("status"))
    return None


def _contact_sheet(video: Path, output: Path, *, duration_frames: int) -> None:
    indices = sorted(
        {
            round((duration_frames - 1) * index / 5)
            for index in range(6)
        }
    )
    expression = "+".join(f"eq(n,{index})" for index in indices)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select='{expression}',scale=180:-2:flags=lanczos,tile=3x2",
        "-vsync",
        "vfr",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise TransferCorpusV2Error(
            result.stderr[-3000:] or "contact sheet generation failed"
        )


def _revision_state(
    repo: Path,
    *,
    implementation_revision: str | None,
    git_clean: bool | None,
) -> tuple[str, bool]:
    if implementation_revision is not None and git_clean is not None:
        return implementation_revision, git_clean
    revision = _run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip()
    clean = not _run(["git", "-C", str(repo), "status", "--porcelain"]).strip()
    return implementation_revision or revision, clean if git_clean is None else git_clean


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise TransferCorpusV2Error(result.stderr.strip() or f"{command[0]} failed")
    return result.stdout


def _rmse(values: Any) -> float:
    items = list(values)
    return math.sqrt(sum(value * value for value in items) / len(items))


def _check(
    check_id: str,
    passed: bool,
    summary: str,
    measured: dict[str, Any],
) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if passed else "fail",
        summary=summary,
        measured=measured,
    )


def _overall(checks: list[CheckResult]) -> Literal["pass", "fail"]:
    return "pass" if all(item.status == "pass" for item in checks) else "fail"
