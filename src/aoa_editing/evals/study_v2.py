"""Join bounded reconstruction passes into one verifiable selection ledger."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Literal

from aoa_editing.domain.models import (
    MotionRecoveryGateReport,
    Provenance,
    ReconstructionStudyPassPlanV2,
    ReconstructionStudyPassV2,
    ReferenceComparisonReportV2,
    ReferenceReconstructionPassReceiptV2,
    ReferenceReconstructionStudyPlanV2,
    ReferenceReconstructionStudyV2,
)
from aoa_editing.infrastructure.media import sha256_file


class ReconstructionStudyV2Error(RuntimeError):
    """Study inputs are inconsistent, incomplete, or select a failed pass."""


def build_reconstruction_study_v2(
    plan_path: Path,
    output_path: Path,
    *,
    implementation_revision: str,
    git_clean: bool,
) -> ReferenceReconstructionStudyV2:
    """Verify immutable pass evidence and write one selection study."""

    if not git_clean:
        raise ReconstructionStudyV2Error(
            "reconstruction study requires a clean implementation revision"
        )
    selected_plan_path = plan_path.expanduser().resolve(strict=True)
    selected_output = output_path.expanduser().resolve()
    if selected_output.exists():
        raise FileExistsError(f"reconstruction study already exists: {selected_output}")
    plan = ReferenceReconstructionStudyPlanV2.model_validate_json(
        selected_plan_path.read_text(encoding="utf-8")
    )
    spec_path = _resolve(plan.spec_path, relative_to=selected_plan_path.parent)
    protocol_path = _resolve(plan.protocol_path, relative_to=selected_plan_path.parent)
    spec_sha256 = sha256_file(spec_path)
    protocol_sha256 = sha256_file(protocol_path)

    passes: list[ReconstructionStudyPassV2] = []
    previous_metrics: dict[str, float] | None = None
    previous_candidate: str | None = None
    comparisons: dict[str, ReferenceComparisonReportV2] = {}
    expected: dict[str, str] | None = None
    for pass_plan in plan.passes:
        comparison_path = _resolve(
            pass_plan.comparison_path,
            relative_to=selected_plan_path.parent,
        )
        comparison = ReferenceComparisonReportV2.model_validate_json(
            comparison_path.read_text(encoding="utf-8")
        )
        if comparison.spec_sha256 != spec_sha256:
            raise ReconstructionStudyV2Error(
                f"pass {pass_plan.sequence} comparison has a different spec hash"
            )
        if comparison.protocol_sha256 != protocol_sha256:
            raise ReconstructionStudyV2Error(
                f"pass {pass_plan.sequence} comparison has a different protocol hash"
            )
        binding = {
            "spec_id": comparison.spec_id,
            "protocol_id": comparison.protocol_id,
            "source_sha256": comparison.source_sha256,
            "reference_sha256": comparison.reference_sha256,
        }
        if expected is None:
            expected = binding
        elif binding != expected:
            raise ReconstructionStudyV2Error(
                f"pass {pass_plan.sequence} comparison binding drifted"
            )
        receipt = _load_receipt(
            pass_plan,
            plan_path=selected_plan_path,
            comparison=comparison,
            spec_sha256=spec_sha256,
        )
        motion_gate = _load_motion_gate(
            pass_plan,
            plan_path=selected_plan_path,
        )
        metrics = _metric_summary(comparison)
        delta = (
            {}
            if previous_metrics is None
            else {key: metrics[key] - previous_metrics[key] for key in sorted(metrics)}
        )
        diagnostic_artifacts = {
            f"diagnostic_{index}": str(_resolve(value, relative_to=selected_plan_path.parent))
            for index, value in enumerate(pass_plan.diagnostic_artifacts)
        }
        diagnostic_artifact_sha256 = {
            key: sha256_file(Path(value)) for key, value in diagnostic_artifacts.items()
        }
        passes.append(
            ReconstructionStudyPassV2(
                sequence=pass_plan.sequence,
                label=pass_plan.label,
                pass_kind=pass_plan.pass_kind,
                comparison_id=comparison.id,
                comparison_path=str(comparison_path),
                comparison_sha256=sha256_file(comparison_path),
                receipt_id=receipt.id if receipt is not None else None,
                receipt_path=(
                    str(
                        _resolve(
                            pass_plan.receipt_path,
                            relative_to=selected_plan_path.parent,
                        )
                    )
                    if pass_plan.receipt_path is not None
                    else None
                ),
                receipt_sha256=(
                    sha256_file(
                        _resolve(
                            pass_plan.receipt_path,
                            relative_to=selected_plan_path.parent,
                        )
                    )
                    if pass_plan.receipt_path is not None
                    else None
                ),
                motion_gate_revision=(
                    motion_gate.git_revision if motion_gate is not None else None
                ),
                motion_gate_path=(
                    str(
                        _resolve(
                            pass_plan.motion_gate_path,
                            relative_to=selected_plan_path.parent,
                        )
                    )
                    if pass_plan.motion_gate_path is not None
                    else None
                ),
                motion_gate_sha256=(
                    sha256_file(
                        _resolve(
                            pass_plan.motion_gate_path,
                            relative_to=selected_plan_path.parent,
                        )
                    )
                    if pass_plan.motion_gate_path is not None
                    else None
                ),
                candidate_sha256=comparison.candidate_sha256,
                candidate_same_as_previous=(
                    previous_candidate == comparison.candidate_sha256
                    if previous_candidate is not None
                    else False
                ),
                objective_overall=comparison.objective_overall,
                comparison_overall=comparison.overall,
                failed_check_ids=[item.id for item in comparison.checks if item.status == "fail"],
                metric_summary=metrics,
                delta_from_previous=delta,
                disposition=pass_plan.disposition,
                rationale=pass_plan.rationale,
                diagnostic_visual_observation=(pass_plan.diagnostic_visual_observation),
                artifacts={
                    **comparison.artifacts,
                    **diagnostic_artifacts,
                },
                diagnostic_artifact_sha256=diagnostic_artifact_sha256,
            )
        )
        comparisons[pass_plan.sequence] = comparison
        previous_metrics = metrics
        previous_candidate = comparison.candidate_sha256

    if expected is None:
        raise ReconstructionStudyV2Error("study plan has no pass evidence")
    selected_pass = next(item for item in passes if item.sequence == plan.selected_sequence)
    selected_comparison = comparisons[plan.selected_sequence]
    if selected_pass.objective_overall != "pass":
        raise ReconstructionStudyV2Error("selected pass is not objective Comparison v2 pass")
    selected_dispositions = [item for item in passes if item.disposition == "selected"]
    if (
        len(selected_dispositions) != 1
        or selected_dispositions[0].sequence != plan.selected_sequence
    ):
        raise ReconstructionStudyV2Error(
            "study plan disposition must identify the selected sequence exactly once"
        )
    if selected_comparison.human_review is None:
        human_status: Literal["pending", "complete"] = "pending"
        human_id = None
        human_verdict = None
        overall: Literal["pass", "warn"] = "warn"
    else:
        human_status = "complete"
        human_id = selected_comparison.human_review.id
        human_verdict = selected_comparison.human_review.verdict
        if human_verdict != "perceptually_and_editorially_indistinguishable":
            raise ReconstructionStudyV2Error("selected pass has a human revision-required verdict")
        overall = "pass"
    supplemental = {
        key: str(_resolve(value, relative_to=selected_plan_path.parent))
        for key, value in plan.supplemental_artifacts.items()
    }
    supplemental_sha256 = {key: sha256_file(Path(value)) for key, value in supplemental.items()}
    study = ReferenceReconstructionStudyV2(
        implementation_revision=implementation_revision,
        git_clean=True,
        plan_path=str(selected_plan_path),
        plan_sha256=sha256_file(selected_plan_path),
        spec_id=expected["spec_id"],
        spec_path=str(spec_path),
        spec_sha256=spec_sha256,
        protocol_id=expected["protocol_id"],
        protocol_path=str(protocol_path),
        protocol_sha256=protocol_sha256,
        source_sha256=expected["source_sha256"],
        reference_sha256=expected["reference_sha256"],
        passes=passes,
        selected_sequence=plan.selected_sequence,
        selected_candidate_sha256=selected_pass.candidate_sha256,
        final_comparison_id=selected_comparison.id,
        final_comparison_path=selected_pass.comparison_path,
        final_comparison_sha256=selected_pass.comparison_sha256,
        hypothesis_decisions=plan.hypothesis_decisions,
        supplemental_artifacts=supplemental,
        supplemental_artifact_sha256=supplemental_sha256,
        human_review_status=human_status,
        human_review_id=human_id,
        human_review_verdict=human_verdict,
        objective_overall="pass",
        mandatory_skips=0,
        provenance=Provenance(
            tool="aoa-editing-reference-reconstruction-study-v2",
            tool_version=version("aoa-editing"),
            parameters={
                "plan_path": str(selected_plan_path),
                "pass_count": len(passes),
                "selected_sequence": plan.selected_sequence,
                "implementation_revision": implementation_revision,
                "objective_fields_rederived": True,
                "human_verdict_inferred": False,
            },
            deterministic=True,
        ),
        overall=overall,
    )
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    with selected_output.open("x", encoding="utf-8") as stream:
        stream.write(study.model_dump_json(indent=2))
        stream.write("\n")
    return study


def _load_receipt(
    pass_plan: ReconstructionStudyPassPlanV2,
    *,
    plan_path: Path,
    comparison: ReferenceComparisonReportV2,
    spec_sha256: str,
) -> ReferenceReconstructionPassReceiptV2 | None:
    if pass_plan.receipt_path is None:
        return None
    path = _resolve(pass_plan.receipt_path, relative_to=plan_path.parent)
    receipt = ReferenceReconstructionPassReceiptV2.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if receipt.pass_kind != pass_plan.pass_kind:
        raise ReconstructionStudyV2Error(
            f"pass {pass_plan.sequence} receipt kind does not match the plan"
        )
    expected_sequence = "5r" if pass_plan.sequence == "5r" else int(pass_plan.sequence)
    if receipt.sequence != expected_sequence:
        raise ReconstructionStudyV2Error(
            f"pass {pass_plan.sequence} receipt sequence does not match the plan"
        )
    if receipt.spec_sha256 != spec_sha256:
        raise ReconstructionStudyV2Error(
            f"pass {pass_plan.sequence} receipt has a different spec hash"
        )
    if (
        receipt.spec_id != comparison.spec_id
        or receipt.source_sha256 != comparison.source_sha256
        or receipt.reference_sha256 != comparison.reference_sha256
    ):
        raise ReconstructionStudyV2Error(
            f"pass {pass_plan.sequence} receipt binding does not match comparison"
        )
    if receipt.overall != "pass" or receipt.hidden_manual_steps:
        raise ReconstructionStudyV2Error(
            f"pass {pass_plan.sequence} application receipt is not a clean pass"
        )
    final = receipt.profiles.get("final")
    if final is None or final.get("sha256") != comparison.candidate_sha256:
        raise ReconstructionStudyV2Error(
            f"pass {pass_plan.sequence} receipt final does not match comparison"
        )
    return receipt


def _load_motion_gate(
    pass_plan: ReconstructionStudyPassPlanV2,
    *,
    plan_path: Path,
) -> MotionRecoveryGateReport | None:
    if pass_plan.motion_gate_path is None:
        return None
    path = _resolve(pass_plan.motion_gate_path, relative_to=plan_path.parent)
    gate = MotionRecoveryGateReport.model_validate_json(path.read_text(encoding="utf-8"))
    if gate.overall != "pass" or gate.mandatory_skips != 0 or not gate.git_clean:
        raise ReconstructionStudyV2Error(
            f"pass {pass_plan.sequence} generic motion gate is not clean pass"
        )
    return gate


def _metric_summary(comparison: ReferenceComparisonReportV2) -> dict[str, float]:
    geometric = comparison.geometric
    temporal = comparison.temporal
    phase = temporal["phase"]
    coupling = temporal["coupling"]["errors"]
    perceptual = comparison.perceptual
    return {
        "center_rmse_normalized": float(geometric["center_rmse_normalized"]),
        "scale_rmse": float(geometric["scale_rmse"]),
        "rotation_rmse_degrees": float(geometric["rotation_rmse_degrees"]),
        "phase_onset_error_frames": float(phase["onset_error_frames"]),
        "phase_duration_error_frames": float(phase["duration_error_frames"]),
        "phase_settle_error_frames": float(phase["settle_error_frames"]),
        "coupling_peak_lag_error_frames": float(coupling["peak_velocity_lag_frames"]),
        "temporal_flow_rmse": float(temporal["temporal_flow_rmse"]),
        "vmaf_mean": float(perceptual["vmaf_mean"]),
        "ssim_all": float(perceptual["ssim_all"]),
        "psnr_average_db": float(perceptual["psnr_average_db"]),
    }


def _resolve(reference: str | Path, *, relative_to: Path) -> Path:
    selected = Path(reference).expanduser()
    if selected.is_absolute():
        return selected.resolve(strict=True)
    candidate = relative_to / selected
    if candidate.exists():
        return candidate.resolve(strict=True)
    raise ReconstructionStudyV2Error(
        f"study artifact relative to {relative_to} does not exist: {reference}"
    )
