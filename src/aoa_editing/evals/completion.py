"""Fail-closed completion audit over all independent prototype proof lanes."""

from __future__ import annotations

import json
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
    "reference-isolation",
    "independent-recreation",
    "clean-rerun",
    "transfer",
    "knowledge-candidate",
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
)


class CompletionAuditError(RuntimeError):
    """Completion evidence could not be loaded or the output would be overwritten."""


def run_completion_audit(
    *,
    repo: Path,
    lock_path: Path,
    readiness_path: Path,
    reconstruction_path: Path,
    comparison_path: Path,
    clean_rerun_path: Path,
    transfer_path: Path,
    candidate_path: Path,
    output_root: Path,
) -> CompletionAuditReport:
    """Join all proof receipts and verify every mandatory completion requirement."""

    repo = repo.expanduser().resolve(strict=True)
    if output_root.exists():
        raise CompletionAuditError(f"completion audit root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    lock = _object(lock_path)
    readiness = _object(readiness_path)
    reconstruction = _object(reconstruction_path)
    comparison = _object(comparison_path)
    clean_rerun = _object(clean_rerun_path)
    transfer = _object(transfer_path)
    candidate_payload = _object(candidate_path)
    candidate: TechniquePacket | None
    candidate_error: str | None = None
    try:
        candidate = TechniquePacket.model_validate(candidate_payload)
    except ValueError as error:
        candidate = None
        candidate_error = str(error)

    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    status = _git(repo, ["status", "--porcelain"]).splitlines()
    remotes = _git(repo, ["remote"]).splitlines()
    tracked = [item for item in _git(repo, ["ls-files", "-z"]).split("\0") if item]
    tracked_hashes = {
        sha256_file(repo / item)
        for item in tracked
        if (repo / item).is_file()
    }

    source_entry = _mapping(lock.get("source_image"))
    reference_entry = _mapping(lock.get("reference_video"))
    source_path = Path(str(source_entry.get("path", "")))
    reference_path = Path(str(reference_entry.get("path", "")))
    source_hash = str(source_entry.get("sha256", ""))
    reference_hash = str(reference_entry.get("sha256", ""))

    checks: list[CheckResult] = []
    checks.append(
        _check(
            "repository-state",
            not status and readiness.get("git_revision") == revision,
            "worktree is clean and readiness belongs to the current revision",
            {
                "git_revision": revision,
                "readiness_revision": readiness.get("git_revision"),
                "status": status,
            },
        )
    )

    readiness_statuses = _check_statuses(readiness)
    checks.append(
        _check(
            "generic-readiness",
            readiness.get("overall") == "pass"
            and readiness.get("reference_content_decoded") is False
            and REQUIRED_READINESS_CHECKS.issubset(readiness_statuses)
            and all(readiness_statuses[item] == "pass" for item in REQUIRED_READINESS_CHECKS),
            "all mandatory generic readiness lanes passed before reference decoding",
            {
                "receipt": str(readiness_path),
                "statuses": readiness_statuses,
                "required": sorted(REQUIRED_READINESS_CHECKS),
            },
        )
    )

    immutable_ok = _locked_file_matches(source_path, source_entry) and _locked_file_matches(
        reference_path, reference_entry
    )
    checks.append(
        _check(
            "immutable-sealed-inputs",
            immutable_ok,
            "source and sealed reference still match their original size and hash locks",
            {
                "source": _file_measurement(source_path, source_entry),
                "reference": _file_measurement(reference_path, reference_entry),
            },
        )
    )

    checks.append(
        _check(
            "publication-and-media-boundary",
            not remotes
            and source_hash not in tracked_hashes
            and reference_hash not in tracked_hashes
            and not _inside(source_path, repo)
            and not _inside(reference_path, repo),
            "repository has no remote and neither user media object is tracked",
            {
                "remotes": remotes,
                "tracked_file_count": len(tracked),
                "source_inside_repo": _inside(source_path, repo),
                "reference_inside_repo": _inside(reference_path, repo),
            },
        )
    )

    missing_repo_paths = [
        relative for relative in REQUIRED_REPOSITORY_PATHS if not (repo / relative).is_file()
    ]
    schemas = sorted((repo / "schemas" / "v1").glob("*.schema.json"))
    checks.append(
        _check(
            "application-and-contract-surfaces",
            not missing_repo_paths and bool(schemas),
            "documented application, operator, scenario, and schema surfaces exist",
            {
                "missing": missing_repo_paths,
                "schema_count": len(schemas),
                "required_paths": list(REQUIRED_REPOSITORY_PATHS),
            },
        )
    )

    profiles = _mapping(reconstruction.get("profiles"))
    preview = _mapping(profiles.get("preview"))
    final = _mapping(profiles.get("final"))
    final_path = Path(str(final.get("path", "")))
    preview_path = Path(str(preview.get("path", "")))
    final_hash = str(final.get("sha256", ""))
    preview_hash = str(preview.get("sha256", ""))
    reconstruction_ok = (
        reconstruction.get("overall") == "pass"
        and reconstruction.get("source_sha256") == source_hash
        and reconstruction.get("reference_sha256") == reference_hash
        and reconstruction.get("reference_media_used_as_input") is False
        and reconstruction.get("hidden_manual_steps") in (False, [])
        and final_path.is_file()
        and preview_path.is_file()
        and sha256_file(final_path) == final_hash
        and sha256_file(preview_path) == preview_hash
        and final_hash != reference_hash
    )
    checks.append(
        _check(
            "independent-reconstruction",
            reconstruction_ok,
            "normal workflow produced intact preview and final artifacts from the permitted source",
            {
                "receipt": str(reconstruction_path),
                "preview_path": str(preview_path),
                "preview_sha256": preview_hash,
                "final_path": str(final_path),
                "final_sha256": final_hash,
                "reference_sha256": reference_hash,
            },
        )
    )

    lineage_ok = bool(profiles) and all(
        _mapping(profile).get("lineage", {}).get("input_hashes") == [source_hash]
        and _mapping(profile).get("lineage", {}).get("forbidden_hashes_present") == []
        for profile in profiles.values()
        if isinstance(profile, dict)
    )
    checks.append(
        _check(
            "source-only-lineage",
            lineage_ok,
            "every reconstruction render lineage contains only the immutable PNG",
            {
                name: _mapping(profile).get("lineage")
                for name, profile in profiles.items()
                if isinstance(profile, dict)
            },
        )
    )

    graph_path = Path(str(reconstruction.get("approved_edit_graph", "")))
    graph = _object_or_empty(graph_path)
    decisions = graph.get("decisions")
    decisions_ok = (
        graph.get("state") == "approved"
        and isinstance(decisions, list)
        and bool(decisions)
        and all(
            isinstance(item, dict) and bool(item.get("reversible_operations"))
            for item in decisions
        )
    )
    checks.append(
        _check(
            "approved-reversible-decisions",
            decisions_ok,
            "the reconstructed edit is owned by an approved graph with inverse operations",
            {"path": str(graph_path), "decision_count": len(decisions or [])},
        )
    )

    exports = _mapping(reconstruction.get("editable_exports"))
    project_root = Path(str(reconstruction.get("project_root", "")))
    export_details: dict[str, Any] = {}
    exports_ok = bool(exports)
    for name, value in exports.items():
        item = _mapping(value)
        output = _resolve_artifact(str(item.get("output_path", "")), project_root)
        valid = output.is_file() and item.get("validator") is not None
        exports_ok = exports_ok and valid
        export_details[name] = {"path": str(output), "exists": output.is_file(), "valid": valid}
    qc_ok = bool(profiles) and all(
        _mapping(_mapping(profile).get("qc")).get("overall") in {"pass", "warn"}
        for profile in profiles.values()
        if isinstance(profile, dict)
    )
    checks.append(
        _check(
            "qc-and-editable-exports",
            qc_ok and exports_ok,
            "preview/final QC and editable interchange validators passed",
            {
                "qc": {
                    name: _mapping(_mapping(item).get("qc")).get("overall")
                    for name, item in profiles.items()
                },
                "exports": export_details,
            },
        )
    )

    comparison_statuses = _check_statuses(comparison)
    comparison_artifacts = _artifact_paths(comparison, Path())
    comparison_ok = (
        comparison.get("overall") == "pass"
        and comparison.get("source_sha256") == source_hash
        and comparison.get("reference_sha256") == reference_hash
        and comparison.get("candidate_sha256") == final_hash
        and comparison.get("reference_media_used_only_for_comparison") is True
        and bool(str(comparison.get("visual_review") or "").strip())
        and bool(comparison_statuses)
        and all(status == "pass" for status in comparison_statuses.values())
        and all(path.exists() for path in comparison_artifacts)
    )
    checks.append(
        _check(
            "comparison",
            comparison_ok,
            "objective thresholds and a recorded visual review passed",
            {
                "receipt": str(comparison_path),
                "statuses": comparison_statuses,
                "visual_review": comparison.get("visual_review"),
                "artifacts": [str(path) for path in comparison_artifacts],
            },
        )
    )

    clean_statuses = _check_statuses(clean_rerun)
    clean_baseline = _mapping(clean_rerun.get("baseline"))
    clean_replay = _mapping(clean_rerun.get("replay"))
    clean_ok = (
        clean_rerun.get("overall") == "pass"
        and clean_rerun.get("old_render_used_as_input") is False
        and bool(clean_statuses)
        and all(status == "pass" for status in clean_statuses.values())
        and clean_baseline.get("final_sha256") == clean_replay.get("final_sha256") == final_hash
        and clean_baseline.get("final_path") != clean_replay.get("final_path")
        and all(path.exists() for path in _artifact_paths(clean_rerun, Path()))
    )
    checks.append(
        _check(
            "clean-rerun",
            clean_ok,
            "a fresh project store reproduced identical approved edit semantics and media",
            {
                "receipt": str(clean_rerun_path),
                "statuses": clean_statuses,
                "baseline": clean_baseline,
                "replay": clean_replay,
            },
        )
    )

    transfer_statuses = _check_statuses(transfer)
    transfer_hash = str(transfer.get("source_sha256", ""))
    transfer_artifacts = _artifact_paths(transfer, Path())
    transfer_ok = (
        transfer.get("overall") == "pass"
        and transfer.get("target_specific_media_used") is False
        and transfer_hash not in {"", source_hash, reference_hash}
        and bool(transfer_statuses)
        and all(status == "pass" for status in transfer_statuses.values())
        and all(path.exists() for path in transfer_artifacts)
    )
    checks.append(
        _check(
            "independent-transfer",
            transfer_ok,
            "the same technique passed on materially different procedural media",
            {
                "receipt": str(transfer_path),
                "source_sha256": transfer_hash,
                "statuses": transfer_statuses,
                "artifacts": [str(path) for path in transfer_artifacts],
            },
        )
    )

    candidate_serialized = json.dumps(candidate_payload, sort_keys=True)
    transferred_candidate_value = _mapping(transfer.get("artifacts")).get(
        "updated_candidate_packet"
    )
    transferred_candidate = (
        Path(str(transferred_candidate_value)) if transferred_candidate_value else Path()
    )
    candidate_ok = (
        candidate is not None
        and candidate.status == "candidate"
        and len(candidate.application_history) >= 2
        and all(item.outcome == "pass" for item in candidate.application_history)
        and source_hash not in candidate_serialized
        and reference_hash not in candidate_serialized
        and transferred_candidate.is_file()
        and sha256_file(transferred_candidate) == sha256_file(candidate_path)
    )
    checks.append(
        _check(
            "knowledge-candidate",
            candidate_ok,
            "source-neutral knowledge remains candidate and carries two passing applications",
            {
                "path": str(candidate_path),
                "validation_error": candidate_error,
                "status": candidate.status if candidate else None,
                "revision": candidate.revision if candidate else None,
                "application_count": len(candidate.application_history) if candidate else 0,
            },
        )
    )

    immutable_after = _locked_file_matches(source_path, source_entry) and _locked_file_matches(
        reference_path, reference_entry
    )
    checks.append(
        _check(
            "post-evaluation-immutability",
            immutable_after,
            "both user-owned inputs remain byte-identical after all evaluations",
            {
                "source_sha256": sha256_file(source_path) if source_path.is_file() else None,
                "reference_sha256": (
                    sha256_file(reference_path) if reference_path.is_file() else None
                ),
            },
        )
    )

    requirement_coverage = _coverage()
    check_by_id = {check.id: check for check in checks}
    non_closeout = [item for item in DONE_REQUIREMENTS if item != "honest-closeout"]
    coverage_ok = set(requirement_coverage) == set(DONE_REQUIREMENTS) and all(
        requirement_coverage[requirement]
        and all(
            check_id in check_by_id and check_by_id[check_id].status == "pass"
            for check_id in requirement_coverage[requirement]
        )
        for requirement in non_closeout
    )
    checks.append(
        _check(
            "completion-coverage",
            coverage_ok,
            "every Definition of Done row links to passing machine evidence",
            {"requirements": requirement_coverage},
        )
    )

    mandatory_skips = sum(check.status == "skip" for check in checks)
    overall = _overall(checks)
    report_path = output_root / "completion-audit.json"
    report = CompletionAuditReport(
        git_revision=revision,
        evidence={
            "lock": str(lock_path),
            "readiness": str(readiness_path),
            "reconstruction": str(reconstruction_path),
            "comparison": str(comparison_path),
            "clean_rerun": str(clean_rerun_path),
            "transfer": str(transfer_path),
            "candidate": str(candidate_path),
        },
        requirement_coverage=requirement_coverage,
        checks=checks,
        artifacts={"report": str(report_path)},
        mandatory_skips=mandatory_skips,
        provenance=Provenance(
            tool="aoa-editing-completion-audit",
            tool_version=version("aoa-editing"),
            parameters={"fail_closed": True, "requirement_count": len(DONE_REQUIREMENTS)},
            deterministic=True,
        ),
        overall=overall,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def _coverage() -> dict[str, list[str]]:
    return {
        "clean-reproducible-repository": ["repository-state", "generic-readiness"],
        "no-terminal-editing-workflow": ["generic-readiness", "application-and-contract-surfaces"],
        "cli-agent-parity": ["generic-readiness", "application-and-contract-surfaces"],
        "immutable-source-provenance": [
            "immutable-sealed-inputs",
            "source-only-lineage",
            "post-evaluation-immutability",
        ],
        "versioned-schemas-migrations": ["generic-readiness", "application-and-contract-surfaces"],
        "evidence-linked-reversible-decisions": ["approved-reversible-decisions"],
        "preview-final-render": ["independent-reconstruction"],
        "qc": ["qc-and-editable-exports"],
        "editable-export": ["qc-and-editable-exports"],
        "three-scenarios": ["generic-readiness"],
        "restart-resume": ["generic-readiness"],
        "partial-evidence-corrections": ["generic-readiness"],
        "brief-style-memory": ["generic-readiness", "application-and-contract-surfaces"],
        "editing-language": ["generic-readiness"],
        "reference-isolation": [
            "generic-readiness",
            "publication-and-media-boundary",
            "source-only-lineage",
        ],
        "independent-recreation": ["independent-reconstruction", "comparison"],
        "clean-rerun": ["clean-rerun"],
        "transfer": ["independent-transfer"],
        "knowledge-candidate": ["knowledge-candidate"],
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


def _object_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _object(path)
    except CompletionAuditError:
        return {}


def _mapping(value: Any) -> dict[str, Any]:
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


def _artifact_paths(payload: dict[str, Any], relative_root: Path) -> list[Path]:
    artifacts = _mapping(payload.get("artifacts"))
    paths: list[Path] = []
    for value in artifacts.values():
        if not isinstance(value, str) or not value or value.startswith("runtime-eval:"):
            continue
        paths.append(_resolve_artifact(value, relative_root))
    return paths


def _resolve_artifact(value: str, relative_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else relative_root / path


def _check(
    check_id: str, passed: bool, summary: str, measured: dict[str, Any]
) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if passed else "fail",
        summary=summary,
        measured=measured,
    )


def _overall(checks: list[CheckResult]) -> Literal["pass", "fail", "warn"]:
    if any(item.status == "fail" for item in checks):
        return "fail"
    if any(item.status in {"warn", "skip"} for item in checks):
        return "warn"
    return "pass"


def _git(repo: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout
