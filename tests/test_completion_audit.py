from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aoa_editing.evals.completion import run_completion_audit


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completion_audit_is_fail_closed_and_covers_the_done_ledger(
    tmp_path: Path, monkeypatch: object
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    required_files = [
        "README.md",
        "DESIGN.md",
        "docs/architecture.md",
        "docs/definition-of-done.md",
        "docs/evaluation.md",
        "docs/roadmap.md",
        "manifests/environment.example.json",
        "manifests/models.json",
        "manifests/analyzers.json",
        "scripts/bootstrap",
        "scripts/launch",
        "scripts/test",
        "scripts/eval",
        "scripts/clean",
        "scripts/reproduce-reference",
        "src/aoa_editing/api/app.py",
        "src/aoa_editing/agent.py",
        "src/aoa_editing/cli.py",
        "src/aoa_editing/domain/models.py",
        "src/aoa_editing/web/index.html",
        "editing-knowledge/scenarios/speech.clean.json",
        "editing-knowledge/scenarios/memory.montage.json",
        "editing-knowledge/scenarios/still.motion.json",
    ]
    for relative in required_files:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    (repo / "schemas" / "v1").mkdir(parents=True)
    (repo / "schemas" / "v1" / "project.schema.json").write_text("{}\n")

    source = tmp_path / "source.png"
    reference = tmp_path / "reference.mp4"
    source.write_bytes(b"independent-source")
    reference.write_bytes(b"sealed-reference")
    source_hash = _digest(source)
    reference_hash = _digest(reference)
    revision = "a" * 40

    lock = _write_json(
        repo / "evals" / "reference.lock.json",
        {
            "source_image": {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
            },
            "reference_video": {
                "path": str(reference),
                "size_bytes": reference.stat().st_size,
                "sha256": reference_hash,
            },
        },
    )
    readiness_checks = [
        "git-clean",
        "schemas-current",
        "ruff",
        "mypy",
        "pytest",
        "ui-browser-smoke",
        "doctor",
        "sealed-input-metadata",
        "desktop-launcher",
        "clean-bootstrap",
        "generic-three-scenario-e2e",
    ]
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "overall": "pass",
            "git_revision": revision,
            "reference_content_decoded": False,
            "checks": [
                {"id": check_id, "status": "pass"} for check_id in readiness_checks
            ],
        },
    )

    final_video = tmp_path / "candidate.mp4"
    preview_video = tmp_path / "preview.mp4"
    final_video.write_bytes(b"candidate")
    preview_video.write_bytes(b"preview")
    candidate_hash = _digest(final_video)
    graph = _write_json(
        tmp_path / "decision-graph.json",
        {"state": "approved", "decisions": [{"reversible_operations": [{}]}]},
    )
    export = tmp_path / "timeline.kdenlive"
    export.write_text("editable", encoding="utf-8")
    reconstruction = _write_json(
        tmp_path / "reconstruction.json",
        {
            "overall": "pass",
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "reference_media_used_as_input": False,
            "hidden_manual_steps": [],
            "approved_edit_graph": str(graph),
            "profiles": {
                "preview": {
                    "path": str(preview_video),
                    "sha256": _digest(preview_video),
                    "qc": {"overall": "pass"},
                    "lineage": {
                        "input_hashes": [source_hash],
                        "forbidden_hashes_present": [],
                    },
                },
                "final": {
                    "path": str(final_video),
                    "sha256": candidate_hash,
                    "qc": {"overall": "pass"},
                    "lineage": {
                        "input_hashes": [source_hash],
                        "forbidden_hashes_present": [],
                    },
                },
            },
            "editable_exports": {
                "kdenlive": {
                    "output_path": str(export),
                    "validator": {"tool": "melt"},
                }
            },
        },
    )

    comparison_artifact = tmp_path / "side-by-side.mp4"
    comparison_artifact.write_bytes(b"comparison")
    comparison = _write_json(
        tmp_path / "comparison.json",
        {
            "overall": "pass",
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "candidate_sha256": candidate_hash,
            "reference_media_used_only_for_comparison": True,
            "visual_review": "Six aligned samples reviewed without visible defects.",
            "checks": [{"id": "similarity", "status": "pass"}],
            "artifacts": {"side_by_side": str(comparison_artifact)},
        },
    )
    replay_video = tmp_path / "replay.mp4"
    replay_video.write_bytes(b"candidate")
    clean = _write_json(
        tmp_path / "clean-rerun.json",
        {
            "overall": "pass",
            "old_render_used_as_input": False,
            "checks": [{"id": "deterministic-final-render", "status": "pass"}],
            "baseline": {"final_sha256": candidate_hash, "final_path": str(final_video)},
            "replay": {"final_sha256": candidate_hash, "final_path": str(replay_video)},
            "artifacts": {"replay": str(replay_video)},
        },
    )
    transfer_artifact = tmp_path / "transfer.mp4"
    transfer_artifact.write_bytes(b"transfer")
    transfer_source_hash = "d" * 64
    candidate = _write_json(
        tmp_path / "technique.json",
        {
            "id": "technique_continuous_contain_reveal",
            "revision": 2,
            "title": "Source-neutral reveal",
            "intent": "Reveal a whole still from a close detail.",
            "applicability": ["single still"],
            "contraindications": ["independent layers"],
            "required_evidence": ["image.statistics"],
            "allowed_operations": ["virtual-camera"],
            "rules": ["preserve source lineage"],
            "heuristics": ["use normalized coordinates"],
            "creative_variants": ["reverse the curve"],
            "constraints": ["no reference media"],
            "failure_modes": ["insufficient source detail"],
            "qc": ["remeasure motion"],
            "positive_examples": ["textured illustration"],
            "negative_examples": ["layered scene"],
            "status": "candidate",
            "application_history": [
                {
                    "case_id": "reference",
                    "source_class": "textured still",
                    "outcome": "pass",
                    "evidence_path": "runtime-eval:reference/comparison.json",
                    "metrics": {},
                },
                {
                    "case_id": "transfer",
                    "source_class": "procedural still",
                    "outcome": "pass",
                    "evidence_path": "runtime-eval:transfer/transfer.json",
                    "metrics": {},
                },
            ],
            "promotion_questions": ["Does it transfer across source classes?"],
            "provenance": {
                "tool": "fixture",
                "tool_version": "1",
                "deterministic": True,
            },
        },
    )
    transfer = _write_json(
        tmp_path / "transfer.json",
        {
            "overall": "pass",
            "source_sha256": transfer_source_hash,
            "target_specific_media_used": False,
            "checks": [{"id": "materially-different-source", "status": "pass"}],
            "artifacts": {
                "final": str(transfer_artifact),
                "updated_candidate_packet": str(candidate),
            },
        },
    )

    import aoa_editing.evals.completion as completion

    def fake_git(_repo: Path, arguments: list[str]) -> str:
        values = {
            ("rev-parse", "HEAD"): revision + "\n",
            ("status", "--porcelain"): "",
            ("remote",): "",
            ("ls-files", "-z"): "",
        }
        return values[tuple(arguments)]

    monkeypatch.setattr(completion, "_git", fake_git)  # type: ignore[attr-defined]
    report = run_completion_audit(
        repo=repo,
        lock_path=lock,
        readiness_path=readiness,
        reconstruction_path=reconstruction,
        comparison_path=comparison,
        clean_rerun_path=clean,
        transfer_path=transfer,
        candidate_path=candidate,
        output_root=tmp_path / "completion",
    )

    assert report.overall == "pass"
    assert report.mandatory_skips == 0
    assert all(check.status == "pass" for check in report.checks)
    assert set(report.requirement_coverage) == set(completion.DONE_REQUIREMENTS)
    assert Path(report.artifacts["report"]).is_file()

    failed_comparison = json.loads(comparison.read_text(encoding="utf-8"))
    failed_comparison["visual_review"] = None
    comparison.write_text(json.dumps(failed_comparison), encoding="utf-8")
    failed = run_completion_audit(
        repo=repo,
        lock_path=lock,
        readiness_path=readiness,
        reconstruction_path=reconstruction,
        comparison_path=comparison,
        clean_rerun_path=clean,
        transfer_path=transfer,
        candidate_path=candidate,
        output_root=tmp_path / "completion-failed",
    )
    assert failed.overall == "fail"
    assert next(check for check in failed.checks if check.id == "comparison").status == "fail"
