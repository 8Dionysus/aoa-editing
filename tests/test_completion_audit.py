from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from aoa_editing.domain.models import CompletionAuditReport
from aoa_editing.evals.completion import DONE_REQUIREMENTS, run_completion_audit
from aoa_editing.evals.motion_recovery import REQUIRED_CORPUS_TAGS


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(identifier: str = "fixture") -> dict[str, Any]:
    return {
        "id": identifier,
        "status": "pass",
        "summary": "fixture passed",
        "measured": {},
    }


def _candidate_packet() -> dict[str, Any]:
    return {
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
                "case_id": f"case-{index}",
                "source_class": f"fixture-{index}",
                "outcome": "pass",
                "evidence_path": f"runtime-eval:case-{index}",
                "metrics": {},
            }
            for index in range(9)
        ],
        "promotion_questions": ["Does it transfer across source classes?"],
        "provenance": {
            "tool": "fixture",
            "tool_version": "1",
            "deterministic": True,
        },
    }


def test_completion_audit_v2_is_fail_closed_and_covers_every_done_row(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    required_files = [
        "README.md",
        "DESIGN.md",
        "docs/architecture.md",
        "docs/definition-of-done.md",
        "docs/evaluation.md",
        "docs/operations.md",
        "docs/roadmap.md",
        "manifests/ai-capabilities.json",
        "scripts/bootstrap",
        "scripts/launch",
        "scripts/test",
        "scripts/eval",
        "scripts/clean",
        "scripts/reproduce-reference",
        "src/aoa_editing/agent.py",
        "src/aoa_editing/api/app.py",
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
        content = (
            "\n".join(f"| `{item}` | fixture | fixture | fixture |" for item in DONE_REQUIREMENTS)
            + "\n"
            if relative == "docs/definition-of-done.md"
            else "fixture\n"
        )
        path.write_text(content, encoding="utf-8")
    for relative in (
        ".venv",
        "var/projects",
        "var/evals",
        "var/artifacts",
        "var/cache",
        "var/tmp",
        "var/state",
    ):
        (repo / relative).mkdir(parents=True)
    (repo / "schemas" / "v1").mkdir(parents=True)
    (repo / "schemas" / "v1" / "project.schema.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _write_json(
        repo / "manifests" / "storage-layout.json",
        {"default_home": "repo-local", "implementation_state": "cutover_verified"},
    )

    revision = "a" * 40
    goal_start = "b" * 40
    private_revision = "c" * 40
    public_root = "d" * 40
    tree = "e" * 40
    remote_enabled = False
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.mp4"
    source.write_bytes(b"independent-source")
    reference.write_bytes(b"sealed-reference")
    source_hash = _digest(source)
    reference_hash = _digest(reference)
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
            "checks": [_check(item) for item in readiness_checks],
        },
    )

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    storage = _write_json(
        tmp_path / "storage.json",
        {
            "overall": "pass",
            "dry_run": False,
            "cutover_ready": True,
            "filesystem_links_created": False,
            "legacy_cleanup_authorized": False,
            "source_data_root": str(legacy),
            "destination_home": str(repo),
            "entries": [
                {
                    "method": "copy_then_verify",
                    "status": "verified",
                    "source_tree_sha256": "1" * 64,
                    "destination_tree_sha256": "1" * 64,
                }
            ],
            "checks": [_check()],
        },
    )

    motion_source = tmp_path / "motion-source.png"
    motion_video = tmp_path / "motion-video.avi"
    motion_source.write_bytes(b"synthetic-motion-source")
    motion_video.write_bytes(b"synthetic-motion-video")
    motion_source_hash = _digest(motion_source)
    motion_video_hash = _digest(motion_video)
    identity_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    provenance = {
        "tool": "fixture",
        "tool_version": "1",
        "deterministic": True,
    }
    truth_frame = {
        "frame": 0,
        "time_seconds": 0.0,
        "matrix_3x3": identity_matrix,
        "center_x": 0.5,
        "center_y": 0.5,
        "scale_relative_to_contain": 1.0,
        "rotation_degrees": 0.0,
        "pivot_x": 0.5,
        "pivot_y": 0.5,
        "velocity": {},
        "acceleration": {},
        "jerk": {},
        "phase": "onset",
        "transform_order": ["translate", "rotate", "scale"],
    }
    recovery_frame = {
        "frame": 0,
        "time_seconds": 0.0,
        "selected_model": "similarity",
        "matrix_3x3": identity_matrix,
        "scale_relative_to_contain": 1.0,
        "rotation_degrees": 0.0,
        "center_x": 0.5,
        "center_y": 0.5,
        "match_count": 1,
        "inlier_count": 1,
        "inlier_ratio": 1.0,
        "reprojection_rmse": 0.0,
        "confidence": 1.0,
        "model_scores": {"similarity": 0.0},
        "velocity": {},
        "acceleration": {},
        "jerk": {},
        "pivot_x": None,
        "pivot_y": None,
        "pivot_identifiability": "unidentifiable",
        "outlier": False,
    }
    fixture_reports: dict[str, str] = {}
    truth_fixtures: list[dict[str, Any]] = []
    for index in range(23):
        fixture_id = f"fixture-{index:02d}"
        truth_fixtures.append(
            {
                "id": fixture_id,
                "content_origin": "synthetic_ground_truth",
                "source_path": str(motion_source),
                "source_sha256": motion_source_hash,
                "video_path": str(motion_video),
                "video_sha256": motion_video_hash,
                "source_width": 64,
                "source_height": 64,
                "output_width": 64,
                "output_height": 64,
                "frame_count": 1,
                "frame_rate": {"numerator": 1, "denominator": 1},
                "expected_model": "similarity",
                "expected_curve": "linear",
                "requirement_tags": sorted(REQUIRED_CORPUS_TAGS),
                "frames": [truth_frame],
                "provenance": provenance,
            }
        )
        fixture_report = _write_json(
            tmp_path / "motion-recoveries" / fixture_id / "motion-recovery.json",
            {
                "source_sha256": motion_source_hash,
                "video_sha256": motion_video_hash,
                "frame_count": 1,
                "analyzed_frame_count": 1,
                "frame_rate": {"numerator": 1, "denominator": 1},
                "width": 64,
                "height": 64,
                "selected_model": "similarity",
                "model_distribution": {"similarity": 1},
                "curve_hint": "linear",
                "curve_confidence": 1.0,
                "mean_confidence": 1.0,
                "model_mismatch": False,
                "outlier_frames": [],
                "phase_boundaries": {"onset": 0, "settle": 0},
                "uncertainties": ["fixture"],
                "alternative_explanations": ["fixture"],
                "frames": [recovery_frame],
                "provenance": provenance,
            },
        )
        fixture_reports[fixture_id] = str(fixture_report)
    corpus = _write_json(
        tmp_path / "motion-corpus.json",
        {
            "content_origin": "synthetic_ground_truth",
            "fixtures": truth_fixtures,
            "requirement_coverage": {
                item: ["fixture-00"] for item in sorted(REQUIRED_CORPUS_TAGS)
            },
            "provenance": {
                **provenance,
                "parameters": {"selection": "complete"},
            },
        },
    )
    motion_gate = _write_json(
        tmp_path / "motion-gate.json",
        {
            "overall": "pass",
            "git_clean": True,
            "git_revision": revision,
            "mandatory_skips": 0,
            "corpus_path": str(corpus),
            "corpus_sha256": _digest(corpus),
            "fixture_reports": fixture_reports,
            "metrics": {item: {} for item in fixture_reports},
            "checks": [_check()],
            "provenance": {
                **provenance,
                "parameters": {"fixture_selection": "complete"},
            },
        },
    )
    motion_plot = tmp_path / "motion.png"
    motion_plot.write_bytes(b"plot")
    frames = [
        {
            "frame": index,
            "velocity": {"x": 0.1},
            "acceleration": {"x": 0.1},
            "jerk": {"x": 0.1},
            "pivot_identifiability": "unidentifiable",
        }
        for index in range(2)
    ]
    phase_model = {
        "twist_candidate": {
            "leading_explanation": "staggered_center_path_with_synchronized_scale_rotation"
        },
        "channel_coupling": {
            "center_phase_relation": "center_lags",
            "peak_velocity_lag_frames": 2,
        },
    }
    motion_evidence = _write_json(
        tmp_path / "motion-evidence.json",
        {
            "schema_version": "2.0.0",
            "id": "motion-evidence",
            "readiness_revision": private_revision,
            "motion_gate_revision": private_revision,
            "motion_gate_path": str(motion_gate),
            "motion_gate_sha256": _digest(motion_gate),
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "all_frames_analyzed": True,
            "frame_count": 2,
            "frames": frames,
            "selected_model": "similarity",
            "phase_model": phase_model,
            "artifacts": {"plot": str(motion_plot)},
        },
    )
    spec = _write_json(
        tmp_path / "spec.json",
        {
            "schema_version": "2.0.0",
            "id": "spec",
            "evidence_id": "motion-evidence",
            "readiness_revision": private_revision,
            "motion_gate_revision": private_revision,
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "frozen_before_first_v2_render": True,
            "reference_media_allowed_in_render": False,
            "duration_frames": 2,
            "motion_frames": frames,
            "phase_model": phase_model,
        },
    )

    final_video = tmp_path / "candidate.mp4"
    preview_video = tmp_path / "preview.mp4"
    replay_video = tmp_path / "replay.mp4"
    replay_preview = tmp_path / "replay-preview.mp4"
    final_video.write_bytes(b"candidate")
    replay_video.write_bytes(b"candidate")
    preview_video.write_bytes(b"preview")
    replay_preview.write_bytes(b"preview")
    candidate_hash = _digest(final_video)
    preview_hash = _digest(preview_video)
    lineage = {
        "input_hashes": [source_hash],
        "forbidden_hashes_present": [],
    }
    reconstruction = _write_json(
        tmp_path / "reconstruction.json",
        {
            "schema_version": "2.0.0",
            "id": "reconstruction",
            "overall": "pass",
            "sequence": "5r",
            "pass_kind": "phase_compensated_matrix",
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "spec_id": "spec",
            "spec_sha256": _digest(spec),
            "reference_media_used_as_input": False,
            "hidden_manual_steps": [],
            "phase_correction_evidence": {"approved": True},
            "profiles": {
                "preview": {
                    "sha256": preview_hash,
                    "qc": {"overall": "pass"},
                    "lineage": lineage,
                },
                "final": {
                    "sha256": candidate_hash,
                    "qc": {"overall": "pass"},
                    "lineage": lineage,
                },
            },
            "editable_exports": {
                "kdenlive": {"validator": {"tool": "melt"}},
                "otio": {"validator": {"tool": "OpenTimelineIO"}},
            },
        },
    )

    protocol = _write_json(
        tmp_path / "protocol.json",
        {
            "schema_version": "2.0.0",
            "id": "protocol",
            "implementation_revision": private_revision,
            "spec_id": "spec",
            "spec_sha256": _digest(spec),
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
        },
    )
    comparison_artifacts: dict[str, str] = {}
    for name in (
        "candidate_motion",
        "frame_diagnostics",
        "side_by_side",
        "aligned_difference",
        "phase_contact_sheet",
        "motion_error_plot",
        "derivative_error_plot",
        "vmaf_log",
        "ssim_log",
        "psnr_log",
    ):
        artifact = tmp_path / f"{name}.bin"
        artifact.write_bytes(name.encode())
        comparison_artifacts[name] = str(artifact)
    human_review = {
        "reviewer": "fixture-human",
        "reviewer_role": "human_operator",
        "candidate_sha256": candidate_hash,
        "verdict": "perceptually_and_editorially_indistinguishable",
        "rubric": [
            {"criterion": criterion, "status": "pass", "notes": "reviewed"}
            for criterion in (
                "smoothness",
                "weight",
                "natural_acceleration",
                "rotation_character",
                "twist_character",
                "settle_timing",
                "absence_of_mechanical_linearity",
                "compositional_match",
                "reference_overlay_absence",
                "encoding_defect_separation",
            )
        ],
        "reviewed_artifacts": [
            "side_by_side",
            "phase_contact_sheet",
            "aligned_difference",
        ],
    }
    comparison = _write_json(
        tmp_path / "comparison.json",
        {
            "schema_version": "2.0.0",
            "overall": "pass",
            "objective_overall": "pass",
            "all_frames_compared": True,
            "mandatory_skips": 0,
            "protocol_path": str(protocol),
            "protocol_id": "protocol",
            "protocol_sha256": _digest(protocol),
            "spec_id": "spec",
            "spec_sha256": _digest(spec),
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "candidate_sha256": candidate_hash,
            "reference_media_used_only_for_comparison": True,
            "lineage": {"reference_hash_present_anywhere": False},
            "checks": [
                _check("technical-resolution"),
                _check("editorial-human-review"),
            ],
            "human_review": human_review,
            "artifacts": comparison_artifacts,
        },
    )

    screenshot = tmp_path / "reference-ui.png"
    screenshot.write_bytes(b"screenshot")
    reference_ui = _write_json(
        tmp_path / "reference-ui.json",
        {
            "overall": "pass",
            "browser": "Chromium fixture",
            "mutated_project_state": False,
            "checks": {
                "synchronized_player_surfaces": True,
                "transform_and_confidence_charts": True,
                "editable_tangent_controls": True,
                "per_item_review_surface": True,
            },
            "screenshot": str(screenshot),
        },
    )

    kdenlive = tmp_path / "timeline.kdenlive"
    otio = tmp_path / "timeline.otio"
    kdenlive.write_text("editable", encoding="utf-8")
    otio.write_text("editable", encoding="utf-8")
    replay_receipt = _write_json(
        tmp_path / "replay-reconstruction.json",
        {"id": "replay-reconstruction"},
    )
    proof = {
        "receipt_id": "reconstruction",
        "receipt_path": str(reconstruction),
        "receipt_sha256": _digest(reconstruction),
        "semantic_timeline_sha256": "9" * 64,
        "profiles": {
            "preview": {
                "path": str(preview_video),
                "sha256": preview_hash,
                **lineage,
            },
            "final": {
                "path": str(final_video),
                "sha256": candidate_hash,
                **lineage,
            },
        },
        "editable_exports": {
            "kdenlive": {"path": str(kdenlive)},
            "otio": {"path": str(otio)},
        },
    }
    replay_proof = {
        **proof,
        "receipt_id": "replay-reconstruction",
        "receipt_path": str(replay_receipt),
        "receipt_sha256": _digest(replay_receipt),
        "project_id": "replay-project",
        "profiles": {
            "preview": {
                "path": str(replay_preview),
                "sha256": preview_hash,
                **lineage,
            },
            "final": {
                "path": str(replay_video),
                "sha256": candidate_hash,
                **lineage,
            },
        },
    }
    baseline_proof = {**proof, "project_id": "baseline-project"}
    clean_rerun = _write_json(
        tmp_path / "clean-rerun.json",
        {
            "schema_version": "2.0.0",
            "implementation_revision": private_revision,
            "git_clean": True,
            "overall": "pass",
            "mandatory_skips": 0,
            "old_render_used_as_input": False,
            "reference_media_used_as_input": False,
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "spec_id": "spec",
            "spec_sha256": _digest(spec),
            "baseline": baseline_proof,
            "replay": replay_proof,
            "checks": [_check()],
        },
    )

    candidate = _write_json(tmp_path / "candidate.json", _candidate_packet())
    transfer_cases: list[dict[str, Any]] = []
    for index in range(5):
        transfer_cases.append(
            {
                "case_id": f"eligible-{index}",
                "application_outcome": "eligible",
                "overall": "pass",
                "checks": [_check()],
                "applicability": {"refusal_code": None},
            }
        )
    for case_id, refusal_code in (
        ("low-resolution", "insufficient_source_resolution"),
        ("composition", "inapplicable_composition"),
    ):
        transfer_cases.append(
            {
                "case_id": case_id,
                "application_outcome": "refused",
                "overall": "pass",
                "checks": [_check()],
                "applicability": {"refusal_code": refusal_code},
            }
        )
    transfer = _write_json(
        tmp_path / "transfer.json",
        {
            "schema_version": "2.0.0",
            "implementation_revision": private_revision,
            "git_clean": True,
            "overall": "pass",
            "mandatory_skips": 0,
            "reference_media_used_as_input": False,
            "target_specific_branch_used": False,
            "forbidden_media_hashes": [source_hash, reference_hash],
            "input_technique_revision": 1,
            "output_technique_revision": 2,
            "packet_sha256": _digest(candidate),
            "cases": transfer_cases,
            "checks": [_check()],
            "artifacts": {"updated_candidate_packet": str(candidate)},
        },
    )

    provider_artifact = tmp_path / "provider-artifact.json"
    provider_artifact.write_text("{}", encoding="utf-8")
    provider_receipts = tmp_path / "provider-receipts"
    provider_receipts.mkdir()
    provider_alias = _write_json(
        tmp_path / "provider-alias.json",
        {
            "implementation_revision": revision,
            "git_clean": True,
            "overall": "pass",
            "declaration_count": 9,
            "mandatory_skips": 0,
            "reference_media_used_as_input": False,
            "cases": [
                {"case_id": case_id, "overall": "pass"}
                for case_id in (
                    "provider_available",
                    "provider_missing",
                    "provider_stale",
                    "version_mismatch",
                    "provider_timeout",
                    "malformed_response",
                    "privacy_denied",
                    "deterministic_fallback",
                    "partial_evidence",
                )
            ],
            "checks": [_check()],
            "artifacts": {
                "doctor": str(provider_artifact),
                "provider_receipts": str(provider_receipts),
            },
        },
    )

    local_artifact = tmp_path / "local-ai-artifact.json"
    local_artifact.write_text("{}", encoding="utf-8")
    capability_ids = (
        "speech.transcript",
        "speech.vad",
        "speech.diarization",
        "vision.describe",
        "vision.regions",
        "vision.segment",
        "vision.embed",
        "editorial.plan",
        "editorial.critique",
    )
    local_ai = _write_json(
        tmp_path / "local-ai.json",
        {
            "implementation_revision": revision,
            "git_clean": True,
            "overall": "pass",
            "selected_alias": "local-ai://speech/transcript/default",
            "capability_decisions": [
                {
                    "capability_id": item,
                    "action": "bind-existing" if item == "speech.transcript" else "defer",
                    "alias_state": "enabled" if item == "speech.transcript" else "unbound",
                }
                for item in capability_ids
            ],
            "model": {
                "model_id": "openai/whisper-large-v3-turbo",
                "model_owner": "abyss-stack",
                "runtime_owner": "abyss-machine",
                "adapter_owner": "aoa-editing",
                "license_id": "mit",
                "stack_registration_parity": True,
                "upstream_revision": "1" * 40,
                "inventory_revision": "2" * 64,
                "export_tree_sha256": "3" * 64,
            },
            "resources": {
                "resource_class": "medium",
                "resource_kind": "ai",
                "resource_gate_admitted": True,
                "evaluation_unit_memory_peak_bytes": 1024,
                "resource_gate_unit": "fixture.service",
            },
            "new_models_installed": False,
            "downloaded_model_bytes": 0,
            "baseline_requires_ai": False,
            "ai_output_authority": "evidence",
            "mandatory_skips": 0,
            "reference_media_used_as_input": False,
            "cases": [
                {
                    "case_id": "synthetic-russian",
                    "overall": "pass",
                    "word_error_rate": 0.0,
                    "character_error_rate": 0.0,
                }
            ],
            "checks": [_check()],
            "artifacts": {"evidence": str(local_artifact)},
        },
    )

    import aoa_editing.evals.completion as completion

    def fake_git(_repo: Path, arguments: list[str]) -> str:
        if arguments == ["rev-parse", "HEAD"]:
            return revision + "\n"
        if arguments == ["rev-parse", goal_start]:
            return goal_start + "\n"
        if arguments == ["status", "--porcelain"]:
            return ""
        if arguments == ["ls-files", "-z"]:
            return ""
        if arguments == ["merge-base", "--is-ancestor", revision, public_root]:
            raise subprocess.CalledProcessError(1, ["git", *arguments])
        if arguments[:2] == ["merge-base", "--is-ancestor"]:
            return ""
        if arguments == ["rev-list", "--max-parents=0", "HEAD"]:
            return public_root + "\n"
        if arguments == [
            "rev-parse",
            "local/private-history-pre-publication-20260721^{tree}",
        ]:
            return tree + "\n"
        if arguments == ["rev-parse", f"{public_root}^{{tree}}"]:
            return tree + "\n"
        if arguments[:3] == ["diff", "--name-only", f"{public_root}..HEAD"]:
            return ""
        if arguments == ["rev-parse", "--verify", "refs/tags/prototype-v0.3.0^{}"]:
            return revision + "\n"
        if arguments == ["remote"]:
            return "origin\n" if remote_enabled else ""
        if arguments == ["ls-remote", "--heads", "--tags", "origin"]:
            return f"{public_root}\trefs/heads/main\n"
        if arguments == ["cat-file", "-e", f"{public_root}^{{commit}}"]:
            return ""
        if arguments == ["branch", "--show-current"]:
            return "goal/completion\n"
        if arguments == [
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ]:
            raise subprocess.CalledProcessError(128, ["git", *arguments])
        if arguments == ["rev-list", f"{goal_start}..{revision}"]:
            return revision + "\n"
        raise AssertionError(f"unexpected git call: {arguments}")

    monkeypatch.setattr(completion, "_git", fake_git)
    def audit(
        output_root: Path,
        *,
        allow_preexisting_remote: bool = False,
    ) -> CompletionAuditReport:
        return run_completion_audit(
            repo=repo,
            lock_path=lock,
            storage_path=storage,
            readiness_path=readiness,
            motion_gate_path=motion_gate,
            motion_evidence_path=motion_evidence,
            spec_path=spec,
            reconstruction_path=reconstruction,
            comparison_path=comparison,
            reference_ui_path=reference_ui,
            clean_rerun_path=clean_rerun,
            transfer_path=transfer,
            candidate_path=candidate,
            provider_alias_path=provider_alias,
            local_ai_path=local_ai,
            output_root=output_root,
            private_history_ref="local/private-history-pre-publication-20260721",
            goal_start_revision=goal_start,
            local_tag="prototype-v0.3.0",
            allow_preexisting_remote=allow_preexisting_remote,
        )

    report = audit(tmp_path / "completion")

    assert report.overall == "pass"
    assert report.mandatory_skips == 0
    assert report.unresolved_requirements == []
    assert report.unresolved_checks == []
    assert all(check.status == "pass" for check in report.checks)
    assert set(report.requirement_coverage) == set(completion.DONE_REQUIREMENTS)
    assert Path(report.artifacts["report"]).is_file()

    remote_enabled = True
    remote_report = audit(
        tmp_path / "completion-with-preexisting-remote",
        allow_preexisting_remote=True,
    )
    assert remote_report.overall == "pass"
    publication = next(
        check for check in remote_report.checks if check.id == "publication-boundary"
    )
    assert publication.measured["published_goal_commits"] == []
    assert publication.measured["revision_reachable_remotely"] is False

    failed_comparison = json.loads(comparison.read_text(encoding="utf-8"))
    failed_comparison["overall"] = "warn"
    failed_comparison["human_review"] = None
    failed_comparison["checks"][-1]["status"] = "warn"
    comparison.write_text(json.dumps(failed_comparison), encoding="utf-8")
    failed = audit(
        tmp_path / "completion-failed",
        allow_preexisting_remote=True,
    )
    assert failed.overall == "fail"
    assert "independent-recreation" in failed.unresolved_requirements
    assert "human-editorial-verdict" in failed.unresolved_checks
    assert (
        next(check for check in failed.checks if check.id == "human-editorial-verdict").status
        == "fail"
    )
