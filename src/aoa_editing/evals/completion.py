"""Fail-closed v2 completion join over every independent prototype proof lane."""

from __future__ import annotations

import json
import os
import re
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from aoa_editing.domain.models import (
    CheckResult,
    CompletionAuditReport,
    Provenance,
    TechniquePacket,
)
from aoa_editing.infrastructure.media import sha256_file

DONE_REQUIREMENTS = (
    "clean-reproducible-repository",
    "product-owned-storage",
    "no-terminal-editing-workflow",
    "cli-agent-parity",
    "immutable-source-provenance",
    "versioned-schemas-migrations",
    "evidence-linked-reversible-decisions",
    "preview-final-render",
    "qc",
    "editable-export",
    "three-scenarios",
    "restart-resume",
    "partial-evidence-corrections",
    "brief-style-memory",
    "editing-language",
    "motion-ground-truth-recovery",
    "reference-motion-evidence-v2",
    "motion-language-execution-v2",
    "human-correctable-motion-ui",
    "reference-isolation",
    "independent-recreation",
    "clean-rerun-v2",
    "transfer-corpus-v2",
    "knowledge-candidate",
    "ai-provider-boundary",
    "live-local-ai-admission",
    "publication-boundary",
    "honest-closeout",
)

REQUIRED_READINESS_CHECKS = {
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
}

REQUIRED_REPOSITORY_PATHS = (
    "README.md",
    "DESIGN.md",
    "docs/architecture.md",
    "docs/definition-of-done.md",
    "docs/evaluation.md",
    "docs/operations.md",
    "docs/roadmap.md",
    "manifests/ai-capabilities.json",
    "manifests/storage-layout.json",
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
)

CRITICAL_HISTORICAL_PATHS = (
    "src/aoa_editing/analysis/motion.py",
    "src/aoa_editing/application/reference_workspace.py",
    "src/aoa_editing/domain/motion_corrections.py",
    "src/aoa_editing/evals/clean_rerun_v2.py",
    "src/aoa_editing/evals/comparison_v2.py",
    "src/aoa_editing/evals/motion_recovery.py",
    "src/aoa_editing/evals/reconstruction_v2.py",
    "src/aoa_editing/evals/reference_v2.py",
    "src/aoa_editing/evals/transfer_v2.py",
    "src/aoa_editing/rendering",
)

REQUIRED_COMPARISON_ARTIFACTS = {
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
}

REQUIRED_HUMAN_RUBRIC = {
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
}

REQUIRED_PROVIDER_CASES = {
    "provider_available",
    "provider_missing",
    "provider_stale",
    "version_mismatch",
    "provider_timeout",
    "malformed_response",
    "privacy_denied",
    "deterministic_fallback",
    "partial_evidence",
}


class CompletionAuditError(RuntimeError):
    """Completion evidence is malformed, unavailable, or would be overwritten."""


def run_completion_audit(
    *,
    repo: Path,
    lock_path: Path,
    storage_path: Path,
    readiness_path: Path,
    motion_gate_path: Path,
    motion_evidence_path: Path,
    spec_path: Path,
    reconstruction_path: Path,
    comparison_path: Path,
    reference_ui_path: Path,
    clean_rerun_path: Path,
    transfer_path: Path,
    candidate_path: Path,
    provider_alias_path: Path,
    local_ai_path: Path,
    output_root: Path,
    private_history_ref: str,
    goal_start_revision: str,
    local_tag: str | None,
    allow_preexisting_remote: bool,
) -> CompletionAuditReport:
    """Join every v2 proof lane without turning pending human work into a pass."""

    selected_repo = repo.expanduser().resolve(strict=True)
    selected_output = output_root.expanduser().resolve()
    if selected_output.exists():
        raise CompletionAuditError(f"completion audit root already exists: {selected_output}")

    input_paths = {
        "lock": lock_path,
        "storage": storage_path,
        "readiness": readiness_path,
        "motion_gate": motion_gate_path,
        "motion_evidence": motion_evidence_path,
        "spec": spec_path,
        "reconstruction": reconstruction_path,
        "comparison": comparison_path,
        "reference_ui": reference_ui_path,
        "clean_rerun": clean_rerun_path,
        "transfer": transfer_path,
        "candidate": candidate_path,
        "provider_alias": provider_alias_path,
        "local_ai": local_ai_path,
    }
    selected_paths = {
        name: path.expanduser().resolve(strict=True) for name, path in input_paths.items()
    }
    payloads = {name: _object(path) for name, path in selected_paths.items()}
    lock = payloads["lock"]
    storage = payloads["storage"]
    readiness = payloads["readiness"]
    motion_gate = payloads["motion_gate"]
    motion_evidence = payloads["motion_evidence"]
    spec = payloads["spec"]
    reconstruction = payloads["reconstruction"]
    comparison = payloads["comparison"]
    reference_ui = payloads["reference_ui"]
    clean_rerun = payloads["clean_rerun"]
    transfer = payloads["transfer"]
    candidate_payload = payloads["candidate"]
    provider_alias = payloads["provider_alias"]
    local_ai = payloads["local_ai"]

    protocol_path = _resolve_required(
        str(comparison.get("protocol_path", "")),
        (selected_repo, selected_paths["comparison"].parent),
    )
    protocol = _object(protocol_path)
    evidence_paths = {**selected_paths, "comparison_protocol": protocol_path}
    evidence_sha256 = {name: sha256_file(path) for name, path in evidence_paths.items()}
    spec_sha256 = evidence_sha256["spec"]
    reconstruction_sha256 = evidence_sha256["reconstruction"]
    protocol_sha256 = evidence_sha256["comparison_protocol"]

    revision = _git(selected_repo, ["rev-parse", "HEAD"]).strip()
    goal_start = _git(selected_repo, ["rev-parse", goal_start_revision]).strip()
    status = _git(selected_repo, ["status", "--porcelain"]).splitlines()
    tracked = [item for item in _git(selected_repo, ["ls-files", "-z"]).split("\0") if item]
    tracked_hashes = {
        sha256_file(selected_repo / item)
        for item in tracked
        if (selected_repo / item).is_file()
    }

    source_entry = _mapping(lock.get("source_image"))
    reference_entry = _mapping(lock.get("reference_video"))
    source_path = Path(str(source_entry.get("path", ""))).expanduser()
    reference_path = Path(str(reference_entry.get("path", ""))).expanduser()
    source_hash = str(source_entry.get("sha256", ""))
    reference_hash = str(reference_entry.get("sha256", ""))

    checks: list[CheckResult] = []
    checks.append(
        _check(
            "repository-state",
            (
                not status
                and readiness.get("git_revision") == revision
                and _is_ancestor(selected_repo, goal_start, revision)
            ),
            "worktree is clean, readiness is current, and the goal boundary joins HEAD",
            git_revision=revision,
            readiness_revision=readiness.get("git_revision"),
            goal_start_revision=goal_start,
            status=status,
        )
    )

    tag_target = _git_optional(
        selected_repo,
        ["rev-parse", "--verify", f"refs/tags/{local_tag}^{{}}"],
    )
    checks.append(
        _check(
            "final-local-tag",
            local_tag is not None and tag_target is not None and tag_target.strip() == revision,
            "the requested local-only completion tag points at the audited revision",
            tag=local_tag,
            tag_target=tag_target.strip() if tag_target else None,
            expected_revision=revision,
        )
    )

    readiness_statuses = _check_statuses(readiness)
    checks.append(
        _check(
            "generic-readiness",
            (
                readiness.get("overall") == "pass"
                and readiness.get("reference_content_decoded") is False
                and REQUIRED_READINESS_CHECKS.issubset(readiness_statuses)
                and all(readiness_statuses[item] == "pass" for item in REQUIRED_READINESS_CHECKS)
                and all(item == "pass" for item in readiness_statuses.values())
            ),
            "bootstrap, schemas, lint, types, tests, doctor, scenarios, and Chromium pass",
            receipt=str(selected_paths["readiness"]),
            statuses=readiness_statuses,
            required=sorted(REQUIRED_READINESS_CHECKS),
        )
    )

    public_roots = _git(selected_repo, ["rev-list", "--max-parents=0", "HEAD"]).splitlines()
    public_root = public_roots[0] if len(public_roots) == 1 else ""
    private_tree = _git_optional(
        selected_repo,
        ["rev-parse", f"{private_history_ref}^{{tree}}"],
    )
    public_tree = (
        _git_optional(selected_repo, ["rev-parse", f"{public_root}^{{tree}}"])
        if public_root
        else None
    )
    critical_drift = (
        _git(
            selected_repo,
            [
                "diff",
                "--name-only",
                f"{public_root}..HEAD",
                "--",
                *CRITICAL_HISTORICAL_PATHS,
            ],
        ).splitlines()
        if public_root
        else ["missing-public-root"]
    )
    historical_revisions = {
        "motion-evidence-readiness": str(motion_evidence.get("readiness_revision", "")),
        "motion-evidence-gate": str(motion_evidence.get("motion_gate_revision", "")),
        "spec-readiness": str(spec.get("readiness_revision", "")),
        "spec-motion-gate": str(spec.get("motion_gate_revision", "")),
        "comparison-protocol": str(protocol.get("implementation_revision", "")),
        "clean-rerun": str(clean_rerun.get("implementation_revision", "")),
        "transfer": str(transfer.get("implementation_revision", "")),
        "provider-alias": str(provider_alias.get("implementation_revision", "")),
        "local-ai": str(local_ai.get("implementation_revision", "")),
    }
    joins = {
        name: _revision_join(selected_repo, item, revision, private_history_ref)
        for name, item in historical_revisions.items()
    }
    bridge_ok = (
        bool(public_root)
        and private_tree is not None
        and public_tree is not None
        and private_tree.strip() == public_tree.strip()
        and not critical_drift
        and all(value in {"current-history", "private-history-bridge"} for value in joins.values())
    )
    checks.append(
        _check(
            "history-bridge",
            bridge_ok,
            "private proof revisions join the identical public-root tree without owner drift",
            public_root=public_root,
            public_tree=public_tree.strip() if public_tree else None,
            private_history_ref=private_history_ref,
            private_tree=private_tree.strip() if private_tree else None,
            critical_drift=critical_drift,
            revision_joins=joins,
        )
    )

    storage_layout_path = selected_repo / "manifests" / "storage-layout.json"
    storage_layout = _object(storage_layout_path)
    storage_entries = [
        item for item in storage.get("entries", []) if isinstance(item, dict)
    ]
    owner_paths = [
        selected_repo / ".venv",
        selected_repo / "var" / "projects",
        selected_repo / "var" / "evals",
        selected_repo / "var" / "artifacts",
        selected_repo / "var" / "cache",
        selected_repo / "var" / "tmp",
        selected_repo / "var" / "state",
    ]
    legacy_root = Path(str(storage.get("source_data_root", ""))).expanduser()
    storage_ok = (
        storage.get("overall") == "pass"
        and storage.get("dry_run") is False
        and storage.get("cutover_ready") is True
        and storage.get("filesystem_links_created") is False
        and storage.get("legacy_cleanup_authorized") is False
        and Path(str(storage.get("destination_home", ""))).resolve() == selected_repo
        and legacy_root.is_dir()
        and all(path.exists() for path in owner_paths)
        and storage_layout.get("default_home") == "repo-local"
        and storage_layout.get("implementation_state") == "cutover_verified"
        and bool(storage_entries)
        and all(item.get("status") in {"verified", "rebuilt"} for item in storage_entries)
        and all(
            item.get("method") != "copy_then_verify"
            or item.get("source_tree_sha256") == item.get("destination_tree_sha256")
            for item in storage_entries
        )
        and not _legacy_symlinks(selected_repo, legacy_root)
        and _all_checks_pass(storage)
    )
    checks.append(
        _check(
            "product-owned-storage",
            storage_ok,
            "runtime and product state are repo-local while legacy remains rollback evidence",
            receipt=str(selected_paths["storage"]),
            destination_home=storage.get("destination_home"),
            legacy_root=str(legacy_root),
            legacy_root_exists=legacy_root.is_dir(),
            owner_paths=[str(path) for path in owner_paths],
            relocation_entries=len(storage_entries),
            legacy_symlinks=_legacy_symlinks(selected_repo, legacy_root),
        )
    )

    immutable_ok = (
        _locked_file_matches(source_path, source_entry)
        and _locked_file_matches(reference_path, reference_entry)
        and source_hash not in tracked_hashes
        and reference_hash not in tracked_hashes
        and not _inside(source_path, selected_repo)
        and not _inside(reference_path, selected_repo)
    )
    checks.append(
        _check(
            "immutable-sealed-inputs",
            immutable_ok,
            "both user-owned inputs still match their locks and remain outside Git",
            source=_file_measurement(source_path, source_entry),
            reference=_file_measurement(reference_path, reference_entry),
            source_inside_repo=_inside(source_path, selected_repo),
            reference_inside_repo=_inside(reference_path, selected_repo),
        )
    )

    missing_repo_paths = [
        relative
        for relative in REQUIRED_REPOSITORY_PATHS
        if not (selected_repo / relative).is_file()
    ]
    schemas = sorted((selected_repo / "schemas" / "v1").glob("*.schema.json"))
    checks.append(
        _check(
            "application-and-contract-surfaces",
            not missing_repo_paths and bool(schemas),
            "application, operator, scenarios, storage law, and generated schemas exist",
            missing=missing_repo_paths,
            schema_count=len(schemas),
        )
    )

    corpus_path = _resolve_required(
        str(motion_gate.get("corpus_path", "")),
        (selected_repo, selected_paths["motion_gate"].parent),
    )
    fixture_reports = _mapping(motion_gate.get("fixture_reports"))
    fixture_paths = [
        _resolve_required(str(item), (selected_repo, selected_paths["motion_gate"].parent))
        for item in fixture_reports.values()
    ]
    fixture_payloads = [_object(path) for path in fixture_paths]
    motion_gate_ok = (
        motion_gate.get("overall") == "pass"
        and motion_gate.get("git_clean") is True
        and motion_gate.get("git_revision") == revision
        and motion_gate.get("mandatory_skips") == 0
        and sha256_file(corpus_path) == motion_gate.get("corpus_sha256")
        and bool(fixture_payloads)
        and all(item.get("overall") == "pass" for item in fixture_payloads)
        and _all_checks_pass(motion_gate)
    )
    checks.append(
        _check(
            "motion-ground-truth-recovery",
            motion_gate_ok,
            "current analyzer recovers the complete independent ground-truth corpus",
            receipt=str(selected_paths["motion_gate"]),
            git_revision=motion_gate.get("git_revision"),
            fixture_count=len(fixture_payloads),
            corpus_sha256=motion_gate.get("corpus_sha256"),
        )
    )

    motion_frames = [
        item for item in motion_evidence.get("frames", []) if isinstance(item, dict)
    ]
    frame_count = int(motion_evidence.get("frame_count") or 0)
    phase_model = _mapping(motion_evidence.get("phase_model"))
    twist = _mapping(phase_model.get("twist_candidate"))
    coupling = _mapping(phase_model.get("channel_coupling"))
    linked_motion_gate = _resolve_required(
        str(motion_evidence.get("motion_gate_path", "")),
        (selected_repo, selected_paths["motion_evidence"].parent),
    )
    motion_artifacts = _artifact_paths(
        motion_evidence,
        (selected_repo, selected_paths["motion_evidence"].parent),
    )
    motion_evidence_ok = (
        motion_evidence.get("schema_version") == "2.0.0"
        and motion_evidence.get("source_sha256") == source_hash
        and motion_evidence.get("reference_sha256") == reference_hash
        and motion_evidence.get("all_frames_analyzed") is True
        and len(motion_frames) == frame_count
        and [item.get("frame") for item in motion_frames] == list(range(frame_count))
        and all(
            isinstance(item.get("velocity"), dict)
            and isinstance(item.get("acceleration"), dict)
            and isinstance(item.get("jerk"), dict)
            and item.get("pivot_identifiability") == "unidentifiable"
            for item in motion_frames
        )
        and sha256_file(linked_motion_gate) == motion_evidence.get("motion_gate_sha256")
        and twist.get("leading_explanation")
        == "staggered_center_path_with_synchronized_scale_rotation"
        and coupling.get("center_phase_relation") == "center_lags"
        and float(coupling.get("peak_velocity_lag_frames") or 0) > 0
        and bool(motion_artifacts)
        and all(path.exists() for path in motion_artifacts)
    )
    checks.append(
        _check(
            "reference-motion-evidence-v2",
            motion_evidence_ok,
            "all-frame kinematics preserve uncertainty and explain the coupled twist",
            receipt=str(selected_paths["motion_evidence"]),
            frame_count=frame_count,
            selected_model=motion_evidence.get("selected_model"),
            twist=twist,
            coupling=coupling,
            artifacts=[str(path) for path in motion_artifacts],
        )
    )

    spec_frames = [item for item in spec.get("motion_frames", []) if isinstance(item, dict)]
    spec_ok = (
        spec.get("schema_version") == "2.0.0"
        and spec.get("evidence_id") == motion_evidence.get("id")
        and spec.get("source_sha256") == source_hash
        and spec.get("reference_sha256") == reference_hash
        and spec.get("frozen_before_first_v2_render") is True
        and spec.get("reference_media_allowed_in_render") is False
        and len(spec_frames) == int(spec.get("duration_frames") or 0) == frame_count
        and [item.get("frame") for item in spec_frames] == list(range(frame_count))
        and spec.get("phase_model") == motion_evidence.get("phase_model")
        and spec.get("id") == reconstruction.get("spec_id")
        and spec_sha256 == reconstruction.get("spec_sha256")
    )
    checks.append(
        _check(
            "frozen-reference-spec-v2",
            spec_ok,
            "the dense source-only reconstruction contract was frozen before rendering",
            receipt=str(selected_paths["spec"]),
            spec_id=spec.get("id"),
            spec_sha256=spec_sha256,
            duration_frames=spec.get("duration_frames"),
        )
    )

    comparison_binding_ok = (
        protocol.get("schema_version") == "2.0.0"
        and protocol.get("id") == comparison.get("protocol_id")
        and protocol.get("spec_id") == spec.get("id")
        and protocol.get("spec_sha256") == spec_sha256
        and protocol.get("source_sha256") == source_hash
        and protocol.get("reference_sha256") == reference_hash
        and comparison.get("spec_id") == spec.get("id")
        and comparison.get("spec_sha256") == spec_sha256
        and comparison.get("protocol_sha256") == protocol_sha256
    )
    checks.append(
        _check(
            "comparison-proof-binding",
            comparison_binding_ok,
            "the comparator is bound to the exact frozen spec and protocol bytes",
            spec_id=spec.get("id"),
            spec_sha256=spec_sha256,
            protocol_id=protocol.get("id"),
            protocol_sha256=protocol_sha256,
            comparison_spec_id=comparison.get("spec_id"),
            comparison_spec_sha256=comparison.get("spec_sha256"),
            comparison_protocol_id=comparison.get("protocol_id"),
            comparison_protocol_sha256=comparison.get("protocol_sha256"),
        )
    )

    profiles = _mapping(reconstruction.get("profiles"))
    reconstruction_final = _mapping(profiles.get("final"))
    reconstruction_preview = _mapping(profiles.get("preview"))
    editable_exports = _mapping(reconstruction.get("editable_exports"))
    reconstruction_ok = (
        reconstruction.get("schema_version") == "2.0.0"
        and reconstruction.get("overall") == "pass"
        and reconstruction.get("sequence") == "5r"
        and reconstruction.get("pass_kind") == "phase_compensated_matrix"
        and reconstruction.get("source_sha256") == source_hash
        and reconstruction.get("reference_sha256") == reference_hash
        and reconstruction.get("reference_media_used_as_input") is False
        and reconstruction.get("hidden_manual_steps") in (False, [])
        and set(profiles) == {"preview", "final"}
        and all(_mapping(item.get("qc")).get("overall") == "pass" for item in profiles.values())
        and set(editable_exports) == {"kdenlive", "otio"}
        and all(_mapping(item).get("validator") for item in editable_exports.values())
        and bool(reconstruction.get("phase_correction_evidence"))
    )
    checks.append(
        _check(
            "motion-v2-execution",
            reconstruction_ok,
            "normal services rendered the bounded phase-compensated Motion v2 edit",
            receipt=str(selected_paths["reconstruction"]),
            sequence=reconstruction.get("sequence"),
            pass_kind=reconstruction.get("pass_kind"),
            final_sha256=reconstruction_final.get("sha256"),
            preview_sha256=reconstruction_preview.get("sha256"),
            editable_exports=sorted(editable_exports),
        )
    )

    ui_checks = _mapping(reference_ui.get("checks"))
    screenshot_path = _resolve_required(
        str(reference_ui.get("screenshot", "")),
        (selected_repo, selected_paths["reference_ui"].parent),
    )
    reference_ui_ok = (
        reference_ui.get("overall") == "pass"
        and "Chromium" in str(reference_ui.get("browser", ""))
        and reference_ui.get("mutated_project_state") is False
        and bool(ui_checks)
        and all(value is True for value in ui_checks.values())
        and screenshot_path.is_file()
    )
    checks.append(
        _check(
            "human-correctable-motion-ui",
            reference_ui_ok,
            "real Chromium exposed synchronized motion, derivatives, variants, and corrections",
            receipt=str(selected_paths["reference_ui"]),
            browser=reference_ui.get("browser"),
            checks=ui_checks,
            screenshot=str(screenshot_path),
            screenshot_sha256=sha256_file(screenshot_path),
        )
    )

    reconstruction_lineage = [
        _mapping(_mapping(item).get("lineage")) for item in profiles.values()
    ]
    clean_baseline = _mapping(clean_rerun.get("baseline"))
    clean_replay = _mapping(clean_rerun.get("replay"))
    clean_profile_sets = [
        _mapping(_mapping(proof).get("profiles"))
        for proof in (clean_baseline, clean_replay)
    ]
    all_lineages = [
        *reconstruction_lineage,
        *[
            _mapping(profile)
            for profile_set in clean_profile_sets
            for profile in profile_set.values()
        ],
    ]
    source_lineage_ok = bool(all_lineages) and all(
        lineage.get("input_hashes") == [source_hash]
        and lineage.get("forbidden_hashes_present") == []
        for lineage in all_lineages
    )
    checks.append(
        _check(
            "source-only-lineage",
            (
                source_lineage_ok
                and comparison.get("reference_media_used_only_for_comparison") is True
                and _mapping(comparison.get("lineage")).get("reference_hash_present_anywhere")
                is False
            ),
            "render and replay lineage contains only the permitted PNG",
            lineage_count=len(all_lineages),
            expected_source_sha256=source_hash,
            forbidden_reference_sha256=reference_hash,
        )
    )

    comparison_checks = _check_statuses(comparison)
    objective_statuses = {
        name: status
        for name, status in comparison_checks.items()
        if name != "editorial-human-review"
    }
    comparison_artifacts = {
        name: _resolve_required(
            str(_mapping(comparison.get("artifacts")).get(name, "")),
            (selected_repo, selected_paths["comparison"].parent),
        )
        for name in REQUIRED_COMPARISON_ARTIFACTS
    }
    candidate_hash = str(comparison.get("candidate_sha256", ""))
    objective_ok = (
        comparison.get("schema_version") == "2.0.0"
        and comparison.get("objective_overall") == "pass"
        and comparison.get("all_frames_compared") is True
        and comparison.get("mandatory_skips") == 0
        and comparison.get("source_sha256") == source_hash
        and comparison.get("reference_sha256") == reference_hash
        and candidate_hash == reconstruction_final.get("sha256")
        and bool(objective_statuses)
        and all(status == "pass" for status in objective_statuses.values())
        and all(path.is_file() for path in comparison_artifacts.values())
    )
    checks.append(
        _check(
            "objective-comparison-v2",
            objective_ok,
            "all-frame technical, geometric, temporal, perceptual, and lineage gates pass",
            receipt=str(selected_paths["comparison"]),
            candidate_sha256=candidate_hash,
            statuses=objective_statuses,
            artifacts={name: str(path) for name, path in comparison_artifacts.items()},
        )
    )

    human_review = _mapping(comparison.get("human_review"))
    rubric = [
        item for item in human_review.get("rubric", []) if isinstance(item, dict)
    ]
    rubric_statuses = {str(item.get("criterion")): item.get("status") for item in rubric}
    human_ok = (
        comparison.get("overall") == "pass"
        and comparison_checks.get("editorial-human-review") == "pass"
        and human_review.get("reviewer_role") == "human_operator"
        and bool(str(human_review.get("reviewer", "")).strip())
        and human_review.get("candidate_sha256") == candidate_hash
        and human_review.get("verdict")
        == "perceptually_and_editorially_indistinguishable"
        and set(rubric_statuses) == REQUIRED_HUMAN_RUBRIC
        and all(status == "pass" for status in rubric_statuses.values())
        and set(human_review.get("reviewed_artifacts", []))
        >= {"side_by_side", "phase_contact_sheet", "aligned_difference"}
    )
    checks.append(
        _check(
            "human-editorial-verdict",
            human_ok,
            "an attributable human accepts smoothness, weight, twist, settle, and parity",
            reviewer=human_review.get("reviewer"),
            verdict=human_review.get("verdict"),
            rubric=rubric_statuses,
            candidate_sha256=human_review.get("candidate_sha256"),
        )
    )

    baseline_profiles = _mapping(clean_baseline.get("profiles"))
    replay_profiles = _mapping(clean_replay.get("profiles"))
    replay_final = _mapping(replay_profiles.get("final"))
    replay_preview = _mapping(replay_profiles.get("preview"))
    replay_final_path = _resolve_required(
        str(replay_final.get("path", "")),
        (selected_repo, selected_paths["clean_rerun"].parent),
    )
    replay_preview_path = _resolve_required(
        str(replay_preview.get("path", "")),
        (selected_repo, selected_paths["clean_rerun"].parent),
    )
    replay_exports = _mapping(clean_replay.get("editable_exports"))
    replay_export_paths = {
        name: _resolve_required(
            str(_mapping(item).get("path", "")),
            (selected_repo, selected_paths["clean_rerun"].parent),
        )
        for name, item in replay_exports.items()
    }
    baseline_receipt_path = _resolve_required(
        str(clean_baseline.get("receipt_path", "")),
        (selected_repo, selected_paths["clean_rerun"].parent),
    )
    replay_receipt_path = _resolve_required(
        str(clean_replay.get("receipt_path", "")),
        (selected_repo, selected_paths["clean_rerun"].parent),
    )
    clean_ok = (
        clean_rerun.get("schema_version") == "2.0.0"
        and clean_rerun.get("overall") == "pass"
        and clean_rerun.get("git_clean") is True
        and clean_rerun.get("mandatory_skips") == 0
        and clean_rerun.get("old_render_used_as_input") is False
        and clean_rerun.get("reference_media_used_as_input") is False
        and clean_rerun.get("source_sha256") == source_hash
        and clean_rerun.get("reference_sha256") == reference_hash
        and clean_rerun.get("spec_id") == spec.get("id")
        and clean_rerun.get("spec_sha256") == spec_sha256
        and clean_baseline.get("receipt_id") == reconstruction.get("id")
        and baseline_receipt_path == selected_paths["reconstruction"]
        and clean_baseline.get("receipt_sha256") == reconstruction_sha256
        and sha256_file(baseline_receipt_path) == reconstruction_sha256
        and bool(clean_replay.get("receipt_id"))
        and clean_replay.get("receipt_id") != clean_baseline.get("receipt_id")
        and clean_replay.get("receipt_sha256") == sha256_file(replay_receipt_path)
        and clean_baseline.get("semantic_timeline_sha256")
        == clean_replay.get("semantic_timeline_sha256")
        and _mapping(baseline_profiles.get("final")).get("sha256")
        == replay_final.get("sha256")
        == candidate_hash
        and _mapping(baseline_profiles.get("preview")).get("sha256")
        == replay_preview.get("sha256")
        and clean_baseline.get("project_id") != clean_replay.get("project_id")
        and replay_final_path.is_file()
        and sha256_file(replay_final_path) == replay_final.get("sha256")
        and replay_preview_path.is_file()
        and sha256_file(replay_preview_path) == replay_preview.get("sha256")
        and set(replay_exports) == {"kdenlive", "otio"}
        and all(path.is_file() for path in replay_export_paths.values())
        and _all_checks_pass(clean_rerun)
    )
    checks.append(
        _check(
            "clean-rerun-v2",
            clean_ok,
            "an empty product home recreated semantic identity, media, QC, and interchange",
            receipt=str(selected_paths["clean_rerun"]),
            baseline_project=clean_baseline.get("project_id"),
            replay_project=clean_replay.get("project_id"),
            baseline_receipt=str(baseline_receipt_path),
            baseline_receipt_sha256=reconstruction_sha256,
            replay_receipt=str(replay_receipt_path),
            replay_receipt_sha256=clean_replay.get("receipt_sha256"),
            semantic_timeline_sha256=clean_replay.get("semantic_timeline_sha256"),
            final_path=str(replay_final_path),
            final_sha256=replay_final.get("sha256"),
            preview_path=str(replay_preview_path),
            exports={name: str(path) for name, path in replay_export_paths.items()},
        )
    )

    transfer_cases = [
        item for item in transfer.get("cases", []) if isinstance(item, dict)
    ]
    eligible = [
        item for item in transfer_cases if item.get("application_outcome") == "eligible"
    ]
    refused = [
        item for item in transfer_cases if item.get("application_outcome") == "refused"
    ]
    refusal_codes = {
        _mapping(item.get("applicability")).get("refusal_code") for item in refused
    }
    transfer_artifacts = _artifact_paths(
        transfer,
        (selected_repo, selected_paths["transfer"].parent),
    )
    transfer_ok = (
        transfer.get("schema_version") == "2.0.0"
        and transfer.get("overall") == "pass"
        and transfer.get("git_clean") is True
        and transfer.get("mandatory_skips") == 0
        and transfer.get("reference_media_used_as_input") is False
        and transfer.get("target_specific_branch_used") is False
        and {source_hash, reference_hash}.issubset(
            set(transfer.get("forbidden_media_hashes", []))
        )
        and len(transfer_cases) == 7
        and len(eligible) == 5
        and len(refused) == 2
        and refusal_codes
        == {"insufficient_source_resolution", "inapplicable_composition"}
        and all(item.get("overall") == "pass" and _all_checks_pass(item) for item in transfer_cases)
        and _all_checks_pass(transfer)
        and all(path.exists() for path in transfer_artifacts)
    )
    checks.append(
        _check(
            "transfer-corpus-v2",
            transfer_ok,
            "five heterogeneous applications and two typed refusals replay independently",
            receipt=str(selected_paths["transfer"]),
            eligible_cases=[item.get("case_id") for item in eligible],
            refused_cases={
                str(item.get("case_id")): _mapping(item.get("applicability")).get("refusal_code")
                for item in refused
            },
            artifacts=[str(path) for path in transfer_artifacts],
        )
    )

    candidate: TechniquePacket | None
    candidate_error: str | None = None
    try:
        candidate = TechniquePacket.model_validate(candidate_payload)
    except ValueError as error:
        candidate = None
        candidate_error = str(error)
    candidate_serialized = json.dumps(candidate_payload, sort_keys=True)
    transferred_packet = _resolve_required(
        str(_mapping(transfer.get("artifacts")).get("updated_candidate_packet", "")),
        (selected_repo, selected_paths["transfer"].parent),
    )
    candidate_ok = (
        candidate is not None
        and candidate.status == "candidate"
        and candidate.revision == transfer.get("output_technique_revision")
        and len(candidate.application_history) == 9
        and all(item.outcome == "pass" for item in candidate.application_history)
        and source_hash not in candidate_serialized
        and reference_hash not in candidate_serialized
        and "/srv/" not in candidate_serialized
        and sha256_file(selected_paths["candidate"]) == transfer.get("packet_sha256")
        and sha256_file(transferred_packet) == sha256_file(selected_paths["candidate"])
    )
    checks.append(
        _check(
            "knowledge-candidate",
            candidate_ok,
            "portable motion knowledge has nine applications and remains human-governed candidate",
            path=str(selected_paths["candidate"]),
            validation_error=candidate_error,
            revision=candidate.revision if candidate else None,
            status=candidate.status if candidate else None,
            application_count=len(candidate.application_history) if candidate else 0,
            packet_sha256=sha256_file(selected_paths["candidate"]),
        )
    )

    provider_cases = [
        item for item in provider_alias.get("cases", []) if isinstance(item, dict)
    ]
    provider_case_ids = {str(item.get("case_id")) for item in provider_cases}
    provider_artifacts = _artifact_paths(
        provider_alias,
        (selected_repo, selected_paths["provider_alias"].parent),
    )
    provider_ok = (
        provider_alias.get("overall") == "pass"
        and provider_alias.get("git_clean") is True
        and provider_alias.get("implementation_revision") == revision
        and provider_alias.get("declaration_count") == 9
        and provider_alias.get("mandatory_skips") == 0
        and provider_alias.get("reference_media_used_as_input") is False
        and provider_case_ids == REQUIRED_PROVIDER_CASES
        and all(item.get("overall") == "pass" for item in provider_cases)
        and _all_checks_pass(provider_alias)
        and all(path.exists() for path in provider_artifacts)
    )
    checks.append(
        _check(
            "ai-provider-boundary",
            provider_ok,
            "all alias, failure, privacy, fallback, supersession, and no-AI cases pass",
            receipt=str(selected_paths["provider_alias"]),
            implementation_revision=provider_alias.get("implementation_revision"),
            cases=sorted(provider_case_ids),
            artifacts=[str(path) for path in provider_artifacts],
        )
    )

    local_cases = [item for item in local_ai.get("cases", []) if isinstance(item, dict)]
    local_decisions = [
        item for item in local_ai.get("capability_decisions", []) if isinstance(item, dict)
    ]
    local_artifacts = _artifact_paths(
        local_ai,
        (selected_repo, selected_paths["local_ai"].parent),
    )
    selected_decision = next(
        (
            item
            for item in local_decisions
            if item.get("capability_id") == "speech.transcript"
        ),
        {},
    )
    model = _mapping(local_ai.get("model"))
    resources = _mapping(local_ai.get("resources"))
    local_ai_ok = (
        local_ai.get("overall") == "pass"
        and local_ai.get("git_clean") is True
        and local_ai.get("implementation_revision") == revision
        and local_ai.get("selected_alias") == "local-ai://speech/transcript/default"
        and len(local_decisions) == 9
        and selected_decision.get("action") == "bind-existing"
        and selected_decision.get("alias_state") == "enabled"
        and all(
            item.get("alias_state") != "enabled"
            for item in local_decisions
            if item.get("capability_id") != "speech.transcript"
        )
        and model.get("model_owner") == "abyss-stack"
        and model.get("runtime_owner") == "abyss-machine"
        and model.get("adapter_owner") == "aoa-editing"
        and model.get("license_id") == "mit"
        and model.get("stack_registration_parity") is True
        and resources.get("resource_class") == "medium"
        and resources.get("resource_kind") == "ai"
        and resources.get("resource_gate_admitted") is True
        and int(resources.get("evaluation_unit_memory_peak_bytes") or 0) > 0
        and local_ai.get("new_models_installed") is False
        and local_ai.get("downloaded_model_bytes") == 0
        and local_ai.get("baseline_requires_ai") is False
        and local_ai.get("ai_output_authority") == "evidence"
        and local_ai.get("mandatory_skips") == 0
        and local_ai.get("reference_media_used_as_input") is False
        and bool(local_cases)
        and all(item.get("overall") == "pass" for item in local_cases)
        and _all_checks_pass(local_ai)
        and all(path.exists() for path in local_artifacts)
    )
    checks.append(
        _check(
            "live-local-ai-admission",
            local_ai_ok,
            "the existing owner-registered ASR passes the product alias against no-AI baseline",
            receipt=str(selected_paths["local_ai"]),
            implementation_revision=local_ai.get("implementation_revision"),
            model_id=model.get("model_id"),
            upstream_revision=model.get("upstream_revision"),
            inventory_revision=model.get("inventory_revision"),
            license=model.get("license_id"),
            export_tree_sha256=model.get("export_tree_sha256"),
            resource_gate_unit=resources.get("resource_gate_unit"),
            cases=[
                {
                    "case_id": item.get("case_id"),
                    "word_error_rate": item.get("word_error_rate"),
                    "character_error_rate": item.get("character_error_rate"),
                }
                for item in local_cases
            ],
        )
    )

    remotes = _git(selected_repo, ["remote"]).splitlines()
    branch = _git(selected_repo, ["branch", "--show-current"]).strip()
    upstream = _git_optional(
        selected_repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
    )
    goal_commits = set(
        _git(selected_repo, ["rev-list", f"{goal_start}..{revision}"]).splitlines()
    )
    remote_hashes: set[str] = set()
    remote_refs: set[str] = set()
    remote_errors: dict[str, str] = {}
    for remote in remotes:
        try:
            rows = _git(selected_repo, ["ls-remote", "--heads", "--tags", remote])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            remote_errors[remote] = str(error)
            continue
        for row in rows.splitlines():
            fields = row.split()
            if len(fields) == 2:
                remote_hashes.add(fields[0])
                remote_refs.add(fields[1])
    unresolved_remote_tips: list[str] = []
    published_goal_commits: set[str] = set()
    revision_reachable_remotely = False
    for remote_hash in sorted(remote_hashes):
        if (
            _git_optional(
                selected_repo,
                ["cat-file", "-e", f"{remote_hash}^{{commit}}"],
            )
            is None
        ):
            unresolved_remote_tips.append(remote_hash)
            continue
        if _is_ancestor(selected_repo, revision, remote_hash):
            revision_reachable_remotely = True
        published_goal_commits.update(
            commit
            for commit in goal_commits
            if _is_ancestor(selected_repo, commit, remote_hash)
        )
    tag_remote_ref = f"refs/tags/{local_tag}" if local_tag else None
    publication_ok = (
        (
            not remotes
            or (
                allow_preexisting_remote
                and not remote_errors
                and not unresolved_remote_tips
                and not published_goal_commits
                and not revision_reachable_remotely
                and (tag_remote_ref is None or tag_remote_ref not in remote_refs)
            )
        )
        and upstream is None
    )
    checks.append(
        _check(
            "publication-boundary",
            publication_ok,
            "pre-existing remotes are read-only evidence; goal commits and tag remain local",
            allow_preexisting_remote=allow_preexisting_remote,
            remotes=remotes,
            branch=branch,
            upstream=upstream.strip() if upstream else None,
            goal_start_revision=goal_start,
            goal_commit_count=len(goal_commits),
            remote_commit_tip_count=len(remote_hashes),
            unresolved_remote_tips=unresolved_remote_tips,
            published_goal_commits=sorted(published_goal_commits),
            revision_reachable_remotely=revision_reachable_remotely,
            local_tag=local_tag,
            remote_tag_present=tag_remote_ref in remote_refs if tag_remote_ref else False,
            remote_errors=remote_errors,
        )
    )

    immutable_after = _locked_file_matches(source_path, source_entry) and _locked_file_matches(
        reference_path, reference_entry
    )
    checks.append(
        _check(
            "post-evaluation-immutability",
            immutable_after,
            "both operator inputs remain byte-identical after every evaluation",
            source_sha256=sha256_file(source_path) if source_path.is_file() else None,
            reference_sha256=sha256_file(reference_path) if reference_path.is_file() else None,
        )
    )

    requirement_coverage = _coverage()
    ledger_text = (selected_repo / "docs" / "definition-of-done.md").read_text(
        encoding="utf-8"
    )
    ledger_requirements = set(
        re.findall(r"^\| `([^`]+)` \|", ledger_text, flags=re.MULTILINE)
    )
    check_by_id = {check.id: check for check in checks}
    non_closeout = [item for item in DONE_REQUIREMENTS if item != "honest-closeout"]
    expected_requirements = set(DONE_REQUIREMENTS)
    coverage_ok = (
        set(requirement_coverage) == expected_requirements
        and ledger_requirements == expected_requirements
        and all(
            requirement_coverage[requirement]
            and all(
                check_id in check_by_id and check_by_id[check_id].status == "pass"
                for check_id in requirement_coverage[requirement]
            )
            for requirement in non_closeout
        )
    )
    checks.append(
        _check(
            "completion-coverage",
            coverage_ok,
            "every mandatory Definition of Done row links to passing machine evidence",
            requirements=requirement_coverage,
            documented_requirement_ids=sorted(ledger_requirements),
            missing_documented_ids=sorted(expected_requirements - ledger_requirements),
            unexpected_documented_ids=sorted(ledger_requirements - expected_requirements),
        )
    )

    check_by_id = {check.id: check for check in checks}
    unresolved_checks = [item.id for item in checks if item.status != "pass"]
    unresolved_requirements = [
        requirement
        for requirement, check_ids in requirement_coverage.items()
        if any(
            check_id not in check_by_id or check_by_id[check_id].status != "pass"
            for check_id in check_ids
        )
    ]
    mandatory_skips = sum(
        value
        for payload in payloads.values()
        if isinstance((value := payload.get("mandatory_skips")), int)
        and not isinstance(value, bool)
        and value > 0
    )
    overall: Literal["pass", "fail"] = (
        "pass"
        if not unresolved_checks and not unresolved_requirements and mandatory_skips == 0
        else "fail"
    )
    report_path = selected_output / "completion-audit-v2.json"
    artifacts = {
        "report": str(report_path),
        "final_video": str(replay_final_path),
        "preview_video": str(replay_preview_path),
        "side_by_side": str(comparison_artifacts["side_by_side"]),
        "aligned_difference": str(comparison_artifacts["aligned_difference"]),
        "phase_contact_sheet": str(comparison_artifacts["phase_contact_sheet"]),
        "motion_error_plot": str(comparison_artifacts["motion_error_plot"]),
        "derivative_error_plot": str(comparison_artifacts["derivative_error_plot"]),
        "reference_motion_evidence": str(selected_paths["motion_evidence"]),
        "reference_spec": str(selected_paths["spec"]),
        "comparison": str(selected_paths["comparison"]),
        "clean_rerun": str(selected_paths["clean_rerun"]),
        "transfer": str(selected_paths["transfer"]),
        "candidate_packet": str(selected_paths["candidate"]),
        "provider_alias": str(selected_paths["provider_alias"]),
        "local_ai": str(selected_paths["local_ai"]),
    }
    report = CompletionAuditReport(
        git_revision=revision,
        git_clean=not status,
        goal_start_revision=goal_start,
        local_tag=local_tag,
        evidence={name: str(path) for name, path in evidence_paths.items()},
        evidence_sha256=evidence_sha256,
        requirement_coverage=requirement_coverage,
        checks=checks,
        artifacts=artifacts,
        unresolved_requirements=unresolved_requirements,
        unresolved_checks=unresolved_checks,
        mandatory_skips=mandatory_skips,
        provenance=Provenance(
            tool="aoa-editing-completion-audit-v2",
            tool_version=version("aoa-editing"),
            parameters={
                "fail_closed": True,
                "requirement_count": len(DONE_REQUIREMENTS),
                "private_history_ref": private_history_ref,
                "allow_preexisting_remote": allow_preexisting_remote,
            },
            deterministic=False,
        ),
        overall=overall,
    )
    selected_output.mkdir(parents=True, exist_ok=False)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def _coverage() -> dict[str, list[str]]:
    return {
        "clean-reproducible-repository": ["repository-state", "generic-readiness"],
        "product-owned-storage": ["product-owned-storage", "generic-readiness"],
        "no-terminal-editing-workflow": [
            "generic-readiness",
            "application-and-contract-surfaces",
        ],
        "cli-agent-parity": ["generic-readiness", "application-and-contract-surfaces"],
        "immutable-source-provenance": [
            "immutable-sealed-inputs",
            "source-only-lineage",
            "post-evaluation-immutability",
        ],
        "versioned-schemas-migrations": [
            "generic-readiness",
            "application-and-contract-surfaces",
            "product-owned-storage",
        ],
        "evidence-linked-reversible-decisions": [
            "motion-v2-execution",
            "clean-rerun-v2",
        ],
        "preview-final-render": ["motion-v2-execution", "clean-rerun-v2"],
        "qc": ["motion-v2-execution", "clean-rerun-v2", "transfer-corpus-v2"],
        "editable-export": ["motion-v2-execution", "clean-rerun-v2", "transfer-corpus-v2"],
        "three-scenarios": ["generic-readiness"],
        "restart-resume": ["generic-readiness", "clean-rerun-v2"],
        "partial-evidence-corrections": [
            "generic-readiness",
            "ai-provider-boundary",
        ],
        "brief-style-memory": ["generic-readiness", "application-and-contract-surfaces"],
        "editing-language": ["generic-readiness", "motion-v2-execution"],
        "motion-ground-truth-recovery": ["motion-ground-truth-recovery"],
        "reference-motion-evidence-v2": [
            "reference-motion-evidence-v2",
            "frozen-reference-spec-v2",
        ],
        "motion-language-execution-v2": [
            "motion-v2-execution",
            "comparison-proof-binding",
            "objective-comparison-v2",
        ],
        "human-correctable-motion-ui": [
            "human-correctable-motion-ui",
            "generic-readiness",
        ],
        "reference-isolation": [
            "immutable-sealed-inputs",
            "source-only-lineage",
            "post-evaluation-immutability",
        ],
        "independent-recreation": [
            "comparison-proof-binding",
            "objective-comparison-v2",
            "human-editorial-verdict",
        ],
        "clean-rerun-v2": ["clean-rerun-v2"],
        "transfer-corpus-v2": ["transfer-corpus-v2"],
        "knowledge-candidate": ["knowledge-candidate"],
        "ai-provider-boundary": ["ai-provider-boundary"],
        "live-local-ai-admission": ["live-local-ai-admission"],
        "publication-boundary": ["final-local-tag", "publication-boundary"],
        "honest-closeout": ["completion-coverage"],
    }


def _object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompletionAuditError(f"cannot load JSON evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CompletionAuditError(f"expected JSON object: {path}")
    return payload


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _check_statuses(payload: dict[str, Any]) -> dict[str, str]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return {}
    return {
        str(item.get("id")): str(item.get("status"))
        for item in checks
        if isinstance(item, dict) and item.get("id") is not None
    }


def _all_checks_pass(payload: dict[str, Any]) -> bool:
    statuses = _check_statuses(payload)
    return bool(statuses) and all(status == "pass" for status in statuses.values())


def _locked_file_matches(path: Path, expected: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected.get("size_bytes")
        and sha256_file(path) == expected.get("sha256")
    )


def _file_measurement(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "expected_size_bytes": expected.get("size_bytes"),
        "expected_sha256": expected.get("sha256"),
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _artifact_paths(
    payload: dict[str, Any],
    roots: tuple[Path, ...],
) -> list[Path]:
    artifacts = _mapping(payload.get("artifacts"))
    paths: list[Path] = []
    for value in artifacts.values():
        if not isinstance(value, str) or not value or value.startswith("runtime-eval:"):
            continue
        paths.append(_resolve_required(value, roots))
    return paths


def _resolve_required(value: str, roots: tuple[Path, ...]) -> Path:
    if not value:
        raise CompletionAuditError("required evidence path is empty")
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise CompletionAuditError(
        f"required evidence artifact is absent: {value} (roots: {[str(item) for item in roots]})"
    )


def _legacy_symlinks(repo: Path, legacy_root: Path) -> list[str]:
    matches: list[str] = []
    for current, directories, files in os.walk(repo, followlinks=False):
        directories[:] = [item for item in directories if item != ".git"]
        for name in [*directories, *files]:
            candidate = Path(current) / name
            if not candidate.is_symlink():
                continue
            try:
                target = candidate.resolve(strict=False)
                target.relative_to(legacy_root.resolve())
            except ValueError:
                continue
            matches.append(str(candidate))
    return sorted(matches)


def _revision_join(repo: Path, revision: str, head: str, private_ref: str) -> str:
    if not revision:
        return "missing"
    if _is_ancestor(repo, revision, head):
        return "current-history"
    if _is_ancestor(repo, revision, private_ref):
        return "private-history-bridge"
    return "unjoined"


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    try:
        _git(repo, ["merge-base", "--is-ancestor", ancestor, descendant])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _check(
    check_id: str,
    passed: bool,
    summary: str,
    **measured: Any,
) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if passed else "fail",
        summary=summary,
        measured=measured,
    )


def _git_optional(repo: Path, arguments: list[str]) -> str | None:
    try:
        return _git(repo, arguments)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _git(repo: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout
