from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Provenance,
    ReferenceWorkflowCandidateRule,
    ReferenceWorkflowRangePlan,
    ReferenceWorkflowSourcePlan,
    ReferenceWorkflowStudyPlan,
)
from aoa_editing.evals.workflow_reference import (
    ReferenceWorkflowStudyError,
    run_reference_workflow_study,
)
from aoa_editing.infrastructure.media import sha256_file

REVISION = "a" * 40


def _settings(tmp_path: Path) -> Settings:
    return Settings.for_home(tmp_path / "editing-home")


def _workflow_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _readiness(_settings: Settings) -> dict[str, object]:
    return {"overall": "pass", "git_revision": REVISION}


def _plan(path: Path) -> ReferenceWorkflowStudyPlan:
    return ReferenceWorkflowStudyPlan(
        title="Human-free workflow reference corpus",
        sources=[
            ReferenceWorkflowSourcePlan(
                alias="workflow-a",
                path=str(path),
                expected_sha256=sha256_file(path),
                reviewed_by="fixture-reviewer",
                ranges=[
                    ReferenceWorkflowRangePlan(
                        id="terminal-wide",
                        start_seconds=0.0,
                        end_seconds=1.25,
                        content_class="terminal",
                        decision="include",
                        human_present=False,
                        rationale="Synthetic terminal-like workflow fixture.",
                    ),
                    ReferenceWorkflowRangePlan(
                        id="title-tail",
                        start_seconds=1.25,
                        end_seconds=1.9,
                        content_class="title-card",
                        decision="exclude",
                        human_present=False,
                        rationale="Excluded non-workflow tail.",
                    ),
                ],
            )
        ],
        candidate_rules=[
            ReferenceWorkflowCandidateRule(
                statement="Hold the wide state before moving to detail.",
                evidence_range_refs=["workflow-a:terminal-wide"],
            )
        ],
        manual_review_complete=True,
        provenance=Provenance(
            tool="fixture-review",
            tool_version="1",
            deterministic=False,
        ),
    )


def test_gated_workflow_study_is_batch_hashed_path_free_and_eval_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reference.mp4"
    _workflow_video(source)
    plan = _plan(source)
    plan_path = tmp_path / "workflow-plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    settings = _settings(tmp_path)
    output = settings.evals_root / "reference-workflow" / "fixture"

    receipt = run_reference_workflow_study(
        plan_path,
        output,
        settings=settings,
        readiness_guard=_readiness,
    )

    assert receipt.overall == "pass"
    assert receipt.reference_media_copied is False
    assert receipt.eligible_as_project_assets is False
    assert receipt.eligible_as_render_inputs is False
    assert receipt.sources[0].sha256 == sha256_file(source)
    assert receipt.sources[0].ranges[0].sample_count >= 2
    assert receipt.sources[0].ranges[1].sample_count == 0
    assert (output / receipt.sources[0].contact_sheet).is_file()
    rendered = (output / "reference-workflow-study.json").read_text(encoding="utf-8")
    assert str(source) not in rendered
    assert not list(output.rglob("*.mp4"))


def test_included_reference_range_must_be_human_free() -> None:
    with pytest.raises(ValidationError, match="human-free workflow content"):
        ReferenceWorkflowRangePlan(
            id="bad",
            start_seconds=0,
            end_seconds=1,
            content_class="human",
            decision="include",
            human_present=True,
            rationale="This must fail closed.",
        )


def test_candidate_rule_cannot_cite_an_excluded_range(tmp_path: Path) -> None:
    source = tmp_path / "reference.mp4"
    _workflow_video(source)
    valid = _plan(source)
    payload = valid.model_dump(mode="json")
    payload["candidate_rules"] = [
        ReferenceWorkflowCandidateRule(
            statement="An exclusion cannot become positive style evidence.",
            evidence_range_refs=["workflow-a:title-tail"],
        ).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match="unknown workflow range"):
        ReferenceWorkflowStudyPlan.model_validate(payload)


def test_workflow_study_refuses_project_asset_output(tmp_path: Path) -> None:
    source = tmp_path / "reference.mp4"
    _workflow_video(source)
    plan_path = tmp_path / "workflow-plan.json"
    plan_path.write_text(_plan(source).model_dump_json(indent=2), encoding="utf-8")
    settings = _settings(tmp_path)

    with pytest.raises(ReferenceWorkflowStudyError, match="evals_root"):
        run_reference_workflow_study(
            plan_path,
            settings.projects_root / "forbidden-study",
            settings=settings,
            readiness_guard=_readiness,
        )


def test_workflow_study_fails_before_decode_without_readiness(tmp_path: Path) -> None:
    source = tmp_path / "reference.mp4"
    _workflow_video(source)
    plan_path = tmp_path / "workflow-plan.json"
    plan_path.write_text(_plan(source).model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ReferenceWorkflowStudyError, match="passing revision-bound"):
        run_reference_workflow_study(
            plan_path,
            _settings(tmp_path).evals_root / "blocked",
            settings=_settings(tmp_path),
            readiness_guard=lambda _settings: {
                "overall": "fail",
                "git_revision": REVISION,
            },
        )
