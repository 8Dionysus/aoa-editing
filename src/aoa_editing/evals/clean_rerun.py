"""Clean-room replay proof from frozen intent and approved edit semantics."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    CheckResult,
    CleanRerunReport,
    EditorialDecisionGraph,
    Provenance,
    ReferenceReconstructionSpec,
)
from aoa_editing.evals.reconstruction import run_reconstruction
from aoa_editing.infrastructure.media import sha256_file


class CleanRerunError(RuntimeError):
    """Clean replay could not establish independence from prior derivatives."""


def run_clean_rerun(
    spec: ReferenceReconstructionSpec,
    source_path: Path,
    baseline_receipt_path: Path,
    output_root: Path,
    *,
    runtime_settings: Settings | None = None,
) -> CleanRerunReport:
    """Recreate a new project in an empty store and compare semantic/output proofs."""

    source_path = source_path.expanduser().resolve(strict=True)
    baseline_receipt_path = baseline_receipt_path.expanduser().resolve(strict=True)
    if output_root.exists():
        raise CleanRerunError(f"clean rerun root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    baseline = _object(baseline_receipt_path)
    if baseline.get("overall") != "pass":
        raise CleanRerunError("baseline reconstruction is not passing")
    if baseline.get("source_sha256") != spec.source_sha256:
        raise CleanRerunError("baseline source hash does not match the frozen spec")
    if baseline.get("reference_sha256") != spec.reference_sha256:
        raise CleanRerunError("baseline reference seal does not match the frozen spec")

    source_before = sha256_file(source_path)
    selected = runtime_settings or Settings.from_env()
    workspace = output_root / "workspace"
    clean_settings = Settings.for_home(
        workspace,
        runtime_root=selected.runtime_root,
        bind_port=selected.bind_port,
    )
    replay_root = output_root / "reconstruction"
    replay = run_reconstruction(
        spec,
        source_path,
        replay_root,
        settings=clean_settings,
        project_name="Clean-room reference reconstruction",
    )
    source_after = sha256_file(source_path)

    baseline_final = Path(baseline["profiles"]["final"]["path"]).resolve(strict=True)
    replay_final = Path(replay["profiles"]["final"]["path"]).resolve(strict=True)
    baseline_timeline = _timeline_document(baseline)
    replay_timeline = _timeline_document(replay)
    baseline_fingerprint = _semantic_fingerprint(baseline_timeline)
    replay_fingerprint = _semantic_fingerprint(replay_timeline)
    replay_lineage = replay["profiles"]["final"]["lineage"]
    command_text = "\n".join(str(item) for item in replay_lineage.get("command", []))
    replay_graph_path = Path(replay["approved_edit_graph"])
    replay_graph = EditorialDecisionGraph.model_validate_json(
        replay_graph_path.read_text(encoding="utf-8")
    )
    derivatives = replay["derived_media"]
    derivative_paths = [
        Path(replay["project_root"]) / relative
        for relative in derivatives.get("artifacts", {}).values()
    ]

    checks = [
        _check(
            "fresh-project-identity",
            baseline["project_id"] != replay["project_id"],
            "replay uses a new project identity",
            {"baseline": baseline["project_id"], "replay": replay["project_id"]},
        ),
        _check(
            "semantic-plan-fingerprint",
            baseline_fingerprint == replay_fingerprint,
            "saved edit semantics reproduce exactly after volatile ids are removed",
            {"baseline": baseline_fingerprint, "replay": replay_fingerprint},
        ),
        _check(
            "deterministic-final-render",
            baseline["profiles"]["final"]["sha256"]
            == replay["profiles"]["final"]["sha256"],
            "clean replay produces the same final media hash",
            {
                "baseline": baseline["profiles"]["final"]["sha256"],
                "replay": replay["profiles"]["final"]["sha256"],
            },
        ),
        _check(
            "source-immutability",
            source_before == source_after == spec.source_sha256,
            "source hash is unchanged across clean replay",
            {"before": source_before, "after": source_after},
        ),
        _check(
            "source-only-lineage",
            replay_lineage.get("input_hashes") == [spec.source_sha256]
            and spec.reference_sha256 not in replay_lineage.get("input_hashes", []),
            "replay lineage contains only the permitted still",
            {
                "input_hashes": replay_lineage.get("input_hashes"),
                "forbidden_hash": spec.reference_sha256,
            },
        ),
        _check(
            "old-render-not-reused",
            str(baseline_final) not in command_text
            and replay_final != baseline_final
            and replay_final.stat().st_ino != baseline_final.stat().st_ino,
            "old render path and inode are absent from replay execution",
            {
                "baseline_path": str(baseline_final),
                "replay_path": str(replay_final),
                "baseline_inode": baseline_final.stat().st_ino,
                "replay_inode": replay_final.stat().st_ino,
            },
        ),
        _check(
            "derived-media-recreated",
            bool(derivative_paths) and all(path.is_file() for path in derivative_paths),
            "proxy, thumbnail, and contact sheet exist in the clean workspace",
            {"paths": [str(path) for path in derivative_paths]},
        ),
        _check(
            "decision-graph-recreated",
            replay_graph.state == "approved"
            and bool(replay_graph.decisions)
            and all(item.reversible_operations for item in replay_graph.decisions),
            "approved evidence-linked decisions and inverses exist in the clean project",
            {"path": str(replay_graph_path), "graph_id": replay_graph.id},
        ),
        _check(
            "all-profile-qc",
            all(profile["qc"]["overall"] != "fail" for profile in replay["profiles"].values()),
            "preview and final QC pass in the clean workspace",
            {
                name: profile["qc"]["overall"]
                for name, profile in replay["profiles"].items()
            },
        ),
    ]
    overall = _overall(checks)
    report_path = output_root / "clean-rerun.json"
    report = CleanRerunReport(
        spec_id=spec.id,
        source_sha256=spec.source_sha256,
        baseline={
            "receipt": str(baseline_receipt_path),
            "project_id": baseline["project_id"],
            "version_id": baseline["version_id"],
            "final_path": str(baseline_final),
            "final_sha256": baseline["profiles"]["final"]["sha256"],
            "semantic_plan_sha256": baseline_fingerprint,
        },
        replay={
            "receipt": str(replay_root / "reconstruction.json"),
            "workspace": str(workspace),
            "project_id": replay["project_id"],
            "version_id": replay["version_id"],
            "final_path": str(replay_final),
            "final_sha256": replay["profiles"]["final"]["sha256"],
            "semantic_plan_sha256": replay_fingerprint,
            "lineage_path": replay["profiles"]["final"]["lineage_path"],
        },
        checks=checks,
        artifacts={
            "report": str(report_path),
            "reconstruction_receipt": str(replay_root / "reconstruction.json"),
            "approved_edit_graph": str(replay_graph_path),
            "decision_log": replay["decision_log"],
            "render_plan": str(replay_final.parent / "render-plan.json"),
            "final_qc": replay["profiles"]["final"]["qc_path"],
        },
        provenance=Provenance(
            tool="aoa-editing-clean-rerun",
            tool_version=version("aoa-editing"),
            parameters={
                "fresh_editing_home": str(clean_settings.editing_home),
                "baseline_media_admitted_as_input": False,
                "volatile_fields_removed_from_plan_fingerprint": True,
            },
            deterministic=True,
        ),
        overall=overall,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def _timeline_document(receipt: dict[str, Any]) -> dict[str, Any]:
    path_value = receipt.get("timeline_projection") or receipt.get("approved_edit_graph")
    if not isinstance(path_value, str):
        raise CleanRerunError("reconstruction receipt has no timeline projection path")
    payload = _object(Path(path_value))
    timeline = payload.get("timeline")
    if not isinstance(timeline, dict):
        raise CleanRerunError(f"timeline projection has no timeline object: {path_value}")
    return timeline


def _semantic_fingerprint(timeline: dict[str, Any]) -> str:
    volatile = {
        "id",
        "project_id",
        "created_at",
        "updated_at",
        "decision_graph_id",
        "decision_refs",
        "evidence_refs",
    }

    def normalize(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {
                item_key: normalize(item_value, item_key)
                for item_key, item_value in sorted(value.items())
                if item_key not in volatile
            }
        if isinstance(value, list):
            return [normalize(item, key) for item in value]
        if key == "asset_id":
            return "SOURCE"
        return value

    encoded = json.dumps(normalize(timeline), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CleanRerunError(f"expected JSON object: {path}")
    return payload


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
