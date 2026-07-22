from __future__ import annotations

from pathlib import Path

import pytest

from aoa_editing.domain.models import (
    CheckResult,
    Provenance,
    ReconstructionStudyPassPlanV2,
    ReferenceComparisonReportV2,
    ReferenceReconstructionStudyPlanV2,
)
from aoa_editing.evals.study_v2 import (
    ReconstructionStudyV2Error,
    build_reconstruction_study_v2,
)
from aoa_editing.infrastructure.media import sha256_file

SOURCE_HASH = "1" * 64
REFERENCE_HASH = "2" * 64
IMPLEMENTATION_REVISION = "a" * 40


def _provenance() -> Provenance:
    return Provenance(tool="synthetic-study", tool_version="1")


def _comparison(
    *,
    spec_sha256: str,
    protocol_sha256: str,
    candidate_sha256: str,
    objective: str,
    vmaf: float,
    center: float,
) -> ReferenceComparisonReportV2:
    failed = objective == "fail"
    return ReferenceComparisonReportV2(
        protocol_id="comparisonprotocol_" + "a" * 32,
        protocol_path="comparison-protocol-v2.json",
        protocol_sha256=protocol_sha256,
        spec_id="referencespec_" + "b" * 32,
        spec_sha256=spec_sha256,
        source_sha256=SOURCE_HASH,
        reference_sha256=REFERENCE_HASH,
        candidate_sha256=candidate_sha256,
        candidate_path="candidate.mp4",
        technical={"frame_count": 20},
        geometric={
            "center_rmse_normalized": center,
            "scale_rmse": center / 2,
            "rotation_rmse_degrees": center * 2,
            "visible_boundary_rmse_normalized": center,
            "homography_corner_rmse_normalized": center,
        },
        temporal={
            "derivative_rmse": {
                "velocity_position": center,
                "velocity_scale": center,
                "velocity_rotation": center,
                "acceleration_position": center,
                "acceleration_scale": center,
                "acceleration_rotation": center,
                "jerk_position": center,
                "jerk_scale": center,
                "jerk_rotation": center,
            },
            "phase": {
                "onset_error_frames": 3 if failed else 1,
                "duration_error_frames": 4 if failed else 1,
                "settle_error_frames": 3 if failed else 0,
            },
            "coupling": {"errors": {"peak_velocity_lag_frames": 2 if failed else 0}},
            "temporal_flow_rmse": center,
        },
        perceptual={
            "vmaf_mean": vmaf,
            "ssim_all": vmaf / 100,
            "psnr_average_db": 40.0,
        },
        editorial={},
        lineage={
            "input_hashes": [SOURCE_HASH],
            "forbidden_hashes_present": [],
            "valid": True,
        },
        checks=[
            CheckResult(
                id="temporal-phase",
                status="fail" if failed else "pass",
                summary="synthetic phase result",
            ),
            CheckResult(
                id="editorial-human-review",
                status="warn",
                summary="human review remains separate",
            ),
        ],
        artifacts={"phase_contact_sheet": "phase-contact-sheet.jpg"},
        provenance=_provenance(),
        objective_overall=objective,
        overall="fail" if failed else "warn",
    )


def _plan(tmp_path: Path) -> Path:
    spec_path = tmp_path / "reference-spec-v2.json"
    protocol_path = tmp_path / "comparison-protocol-v2.json"
    spec_path.write_text('{"fixture":"spec"}\n', encoding="utf-8")
    protocol_path.write_text('{"fixture":"protocol"}\n', encoding="utf-8")
    tail_contact_sheet = tmp_path / "tail-contact-sheet.jpg"
    tail_contact_sheet.write_bytes(b"synthetic supplemental diagnostic")
    spec_sha256 = sha256_file(spec_path)
    protocol_sha256 = sha256_file(protocol_path)
    definitions = (
        ("1", "v1-baseline", "v1_baseline", "baseline", 88.0, 0.01, "3"),
        ("2", "timing-easing", "timing_easing", "rejected", 20.0, 0.03, "4"),
        (
            "3",
            "continuous-tangents",
            "continuous_tangents",
            "retained",
            99.0,
            0.002,
            "5",
        ),
        (
            "4",
            "moving-pivot",
            "moving_pivot_hypothesis",
            "rejected",
            99.0,
            0.002,
            "5",
        ),
        (
            "5",
            "coupled-scale-rotation",
            "coupled_scale_rotation",
            "retained",
            99.4,
            0.001,
            "6",
        ),
        (
            "5r",
            "phase-compensated",
            "phase_compensated_matrix",
            "selected",
            98.9,
            0.0014,
            "7",
        ),
    )
    passes = []
    for sequence, label, kind, disposition, vmaf, center, candidate_digit in definitions:
        comparison = _comparison(
            spec_sha256=spec_sha256,
            protocol_sha256=protocol_sha256,
            candidate_sha256=candidate_digit * 64,
            objective="pass" if sequence == "5r" else "fail",
            vmaf=vmaf,
            center=center,
        )
        comparison_path = tmp_path / f"{sequence}-comparison-v2.json"
        comparison_path.write_text(
            comparison.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        passes.append(
            ReconstructionStudyPassPlanV2(
                sequence=sequence,
                label=label,
                pass_kind=kind,
                comparison_path=str(comparison_path),
                disposition=disposition,
                rationale=f"Synthetic disposition for pass {sequence}.",
                diagnostic_visual_observation=(
                    f"Synthetic diagnostic observation for pass {sequence}."
                ),
                diagnostic_artifacts=[str(comparison_path)],
            )
        )
    plan = ReferenceReconstructionStudyPlanV2(
        spec_path=str(spec_path),
        protocol_path=str(protocol_path),
        passes=passes,
        selected_sequence="5r",
        hypothesis_decisions={
            "affine_homography": "evidence_rejected_unexecuted",
            "local_warp": "evidence_rejected_unexecuted",
        },
        supplemental_artifacts={"tail_contact_sheet": str(tail_contact_sheet)},
        provenance=_provenance(),
    )
    plan_path = tmp_path / "study-plan-v2.json"
    plan_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return plan_path


def test_reconstruction_study_joins_passes_without_inventing_human_review(
    tmp_path: Path,
) -> None:
    plan_path = _plan(tmp_path)
    output = tmp_path / "reconstruction-study-v2.json"
    study = build_reconstruction_study_v2(
        plan_path,
        output,
        implementation_revision=IMPLEMENTATION_REVISION,
        git_clean=True,
    )

    assert study.objective_overall == "pass"
    assert study.overall == "warn"
    assert study.human_review_status == "pending"
    assert study.selected_sequence == "5r"
    assert study.selected_candidate_sha256 == "7" * 64
    assert study.mandatory_skips == 0
    assert len(study.passes) == 6
    assert study.passes[3].candidate_same_as_previous is True
    assert study.passes[-1].diagnostic_artifact_sha256
    assert study.supplemental_artifact_sha256
    assert study.passes[-1].delta_from_previous["vmaf_mean"] == pytest.approx(-0.5)
    assert output.is_file()


def test_reconstruction_study_refuses_to_select_failed_objective(
    tmp_path: Path,
) -> None:
    plan_path = _plan(tmp_path)
    payload = ReferenceReconstructionStudyPlanV2.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    ).model_dump(mode="json")
    payload["selected_sequence"] = "5"
    plan_path.write_text(
        ReferenceReconstructionStudyPlanV2.model_validate(payload).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReconstructionStudyV2Error, match="selected pass is not objective"):
        build_reconstruction_study_v2(
            plan_path,
            tmp_path / "reconstruction-study-v2.json",
            implementation_revision=IMPLEMENTATION_REVISION,
            git_clean=True,
        )


def test_reconstruction_study_refuses_dirty_implementation(tmp_path: Path) -> None:
    plan_path = _plan(tmp_path)
    with pytest.raises(ReconstructionStudyV2Error, match="clean implementation"):
        build_reconstruction_study_v2(
            plan_path,
            tmp_path / "reconstruction-study-v2.json",
            implementation_revision=IMPLEMENTATION_REVISION,
            git_clean=False,
        )
