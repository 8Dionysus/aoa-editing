from __future__ import annotations

import json
from pathlib import Path

import pytest

from aoa_editing.domain.models import (
    EditorialReviewV2,
    EditorialRubricAssessmentV2,
)
from aoa_editing.evals.comparison_v2 import (
    ComparisonV2Error,
    freeze_comparison_protocol_v2,
    record_editorial_review_v2,
    run_comparison_v2,
)
from aoa_editing.evals.motion_fixtures import fixture_definitions, render_motion_fixture
from aoa_editing.evals.reference_v2 import analyze_reference_v2
from aoa_editing.infrastructure.media import sha256_file


@pytest.mark.integration
def test_comparison_protocol_v2_is_frozen_before_an_objective_identity_run(
    tmp_path: Path,
) -> None:
    definition = next(item for item in fixture_definitions() if item.id == "ease-in-out")
    fixture = render_motion_fixture(definition, tmp_path / "fixture")
    readiness, motion_gate = _gate_receipts(tmp_path, revision="a" * 40)
    analysis_root = tmp_path / "analysis"
    _, spec = analyze_reference_v2(
        fixture.source_path,
        fixture.video_path,
        analysis_root,
        readiness_path=readiness,
        motion_gate_path=motion_gate,
    )
    spec_path = analysis_root / "reference-spec-v2.json"
    protocol_root = tmp_path / "protocol"
    protocol = freeze_comparison_protocol_v2(
        spec_path,
        protocol_root,
        implementation_revision="b" * 40,
        git_clean=True,
    )

    assert protocol.spec_id == spec.id
    assert protocol.spec_sha256 == sha256_file(spec_path)
    assert protocol.threshold_revision == "reference-comparison-v2-frozen-1"
    assert protocol.frozen_before_first_candidate_render is True
    assert {
        "technical",
        "geometric",
        "temporal",
        "perceptual",
        "editorial",
        "lineage",
    } == set(protocol.mandatory_axes)
    assert "temporal-flow" in protocol.mandatory_checks
    assert "editorial-human-review" in protocol.mandatory_checks
    assert (protocol_root / "comparison-protocol-v2.json").is_file()
    template = json.loads(
        (protocol_root / "human-review-template-v2.json").read_text(encoding="utf-8")
    )
    assert [item["criterion"] for item in template["rubric"]] == protocol.human_rubric
    with pytest.raises(FileExistsError):
        freeze_comparison_protocol_v2(
            spec_path,
            protocol_root,
            implementation_revision="b" * 40,
            git_clean=True,
        )

    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(
        json.dumps(
            {
                "input_hashes": [sha256_file(fixture.source_path)],
                "forbidden_hashes_present": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ComparisonV2Error, match="must not be the reference"):
        run_comparison_v2(
            protocol_root / "comparison-protocol-v2.json",
            spec_path,
            fixture.source_path,
            fixture.video_path,
            fixture.video_path,
            tmp_path / "production-identity-is-forbidden",
            lineage_path=lineage_path,
        )
    report = run_comparison_v2(
        protocol_root / "comparison-protocol-v2.json",
        spec_path,
        fixture.source_path,
        fixture.video_path,
        fixture.video_path,
        tmp_path / "comparison",
        lineage_path=lineage_path,
        allow_synthetic_identity_fixture=True,
    )

    assert report.objective_overall == "pass"
    assert report.overall == "warn"
    assert report.all_frames_compared is True
    assert report.mandatory_skips == 0
    assert report.geometric["sample_count"] == definition.frame_count
    assert report.temporal["temporal_flow_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert report.perceptual["vmaf_mean"] >= 99.7
    assert report.perceptual["ssim_all"] == pytest.approx(1.0, abs=0.0001)
    assert next(item for item in report.checks if item.id == "editorial-human-review").status == (
        "warn"
    )
    assert all(Path(path).is_file() for path in report.artifacts.values())

    review = EditorialReviewV2(
        protocol_id=protocol.id,
        candidate_sha256=report.candidate_sha256,
        reviewer="synthetic-fixture-reviewer",
        verdict="perceptually_and_editorially_indistinguishable",
        rubric=[
            EditorialRubricAssessmentV2(
                criterion=criterion,
                status="pass",
                notes="Synthetic identity fixture has exact decoded motion on this axis.",
            )
            for criterion in protocol.human_rubric
        ],
        reviewed_artifacts=[
            report.artifacts["side_by_side"],
            report.artifacts["phase_contact_sheet"],
            report.artifacts["aligned_difference"],
        ],
    )
    review_path = tmp_path / "human-review.json"
    review_path.write_text(review.model_dump_json(indent=2) + "\n", encoding="utf-8")
    reviewed = record_editorial_review_v2(
        tmp_path / "comparison" / "comparison-v2.json",
        protocol_root / "comparison-protocol-v2.json",
        review_path,
        tmp_path / "comparison" / "comparison-v2-reviewed.json",
    )
    assert reviewed.supersedes_report_id == report.id
    assert reviewed.objective_overall == "pass"
    assert reviewed.overall == "pass"
    assert (
        next(item for item in reviewed.checks if item.id == "editorial-human-review").status
        == "pass"
    )

    failed_review = review.model_copy(
        update={
            "id": "editorialreview_" + "f" * 32,
            "verdict": "revision_required",
            "rubric": [
                review.rubric[0].model_copy(
                    update={
                        "status": "fail",
                        "notes": "Synthetic negative case requires a revision.",
                    }
                ),
                *review.rubric[1:],
            ],
        }
    )
    failed_review_path = tmp_path / "failed-human-review.json"
    failed_review_path.write_text(
        failed_review.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    failed = record_editorial_review_v2(
        tmp_path / "comparison" / "comparison-v2.json",
        protocol_root / "comparison-protocol-v2.json",
        failed_review_path,
        tmp_path / "comparison" / "comparison-v2-rejected.json",
    )
    assert failed.objective_overall == "pass"
    assert failed.overall == "fail"


def _gate_receipts(root: Path, *, revision: str) -> tuple[Path, Path]:
    readiness = root / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "overall": "pass",
                "reference_content_decoded": False,
                "git_revision": revision,
            }
        ),
        encoding="utf-8",
    )
    motion = root / "motion-gate.json"
    motion.write_text(
        json.dumps(
            {
                "overall": "pass",
                "git_clean": True,
                "mandatory_skips": 0,
                "git_revision": revision,
            }
        ),
        encoding="utf-8",
    )
    return readiness, motion
