"""Clean-room replay of the selected Reference Reconstruction v2 semantics."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    Asset,
    CheckResult,
    CleanRerunApplicationProofV2,
    CleanRerunExportProofV2,
    CleanRerunRenderProofV2,
    CleanRerunReportV2,
    DerivedMediaManifest,
    EditorialDecisionGraph,
    ProjectManifest,
    ProjectVersion,
    Provenance,
    QCReport,
    ReferenceReconstructionPassReceiptV2,
    ReferenceReconstructionSpecV2,
    Treatment,
)
from aoa_editing.evals.reconstruction_v2 import run_reconstruction_pass_v2
from aoa_editing.infrastructure.media import sha256_file


class CleanRerunV2Error(RuntimeError):
    """A v2 replay could not prove source, semantic, or storage independence."""


def run_clean_rerun_v2(
    spec_path: Path,
    source_path: Path,
    baseline_receipt_path: Path,
    output_root: Path,
    *,
    readiness_receipt: dict[str, Any],
    runtime_settings: Settings | None = None,
) -> CleanRerunReportV2:
    """Recreate pass 5r in an empty home from the still and approved semantics only."""

    revision = _passing_readiness_revision(readiness_receipt)
    selected_spec_path = spec_path.expanduser().resolve(strict=True)
    selected_source = source_path.expanduser().resolve(strict=True)
    selected_baseline_path = baseline_receipt_path.expanduser().resolve(strict=True)
    selected_output = output_root.expanduser().resolve()
    if selected_output.exists():
        raise CleanRerunV2Error(f"clean rerun v2 root already exists: {selected_output}")

    spec = _read_model(selected_spec_path, ReferenceReconstructionSpecV2)
    baseline_receipt = _read_model(
        selected_baseline_path,
        ReferenceReconstructionPassReceiptV2,
    )
    _validate_baseline(
        spec,
        selected_spec_path,
        baseline_receipt,
        selected_source,
    )
    phase_correction = baseline_receipt.phase_correction
    if phase_correction is None:  # model/pass invariant, repeated at trust boundary
        raise CleanRerunV2Error("selected baseline has no approved phase correction")
    phase_correction_sha256 = _canonical_model_sha256(phase_correction)
    source_before = sha256_file(selected_source)
    baseline = _application_proof(baseline_receipt, selected_baseline_path)

    selected_output.mkdir(parents=True, exist_ok=False)
    selected_settings = runtime_settings or Settings.from_env()
    clean_home = selected_output / "workspace"
    clean_settings = Settings.for_home(
        clean_home,
        runtime_root=selected_settings.runtime_root,
        bind_port=selected_settings.bind_port,
    )
    replay_root = selected_output / "reconstruction"
    replay_receipt = run_reconstruction_pass_v2(
        selected_spec_path,
        selected_source,
        replay_root,
        pass_kind="phase_compensated_matrix",
        approved_phase_correction=phase_correction,
        settings=clean_settings,
        project_name="Clean-room Reference Reconstruction v2 pass 5r",
    )
    replay_receipt_path = replay_root / "reconstruction-pass-v2.json"
    source_after = sha256_file(selected_source)
    replay = _application_proof(replay_receipt, replay_receipt_path)

    replay_commands = _render_command_text(replay_receipt)
    baseline_media_paths = [
        proof.path for proof in baseline.profiles.values()
    ]
    copied_reference_files = _matching_digest_files(
        clean_home,
        spec.reference_sha256,
    )
    expected_inputs = [spec.source_sha256]
    canonical_paths = [
        replay.project_manifest_path,
        replay.treatment_path,
        replay.patch_path,
        replay.version_path,
        replay.decision_graph_path,
    ]
    checks = [
        _check(
            "fresh-empty-editing-home",
            clean_home.is_dir()
            and clean_home != selected_settings.editing_home.resolve()
            and len(list((clean_home / "var" / "projects").glob("project_*"))) == 1,
            "one new project was created beneath a distinct empty editing home",
            {
                "editing_home": str(clean_home),
                "normal_editing_home": str(selected_settings.editing_home),
            },
        ),
        _check(
            "distinct-project-identity",
            baseline.project_id != replay.project_id
            and baseline.version_id != replay.version_id
            and baseline.asset_id != replay.asset_id,
            "project, version, and asset identities were regenerated",
            {
                "baseline": {
                    "project_id": baseline.project_id,
                    "version_id": baseline.version_id,
                    "asset_id": baseline.asset_id,
                },
                "replay": {
                    "project_id": replay.project_id,
                    "version_id": replay.version_id,
                    "asset_id": replay.asset_id,
                },
            },
        ),
        _check(
            "distinct-paths-and-inodes",
            baseline.project_root != replay.project_root
            and baseline.asset_path != replay.asset_path
            and baseline.asset_inode != replay.asset_inode
            and all(
                baseline.profiles[name].path != replay.profiles[name].path
                and baseline.profiles[name].inode != replay.profiles[name].inode
                for name in ("preview", "final")
            ),
            "replay assets and outputs are separate filesystem objects",
            {
                "baseline_asset_inode": baseline.asset_inode,
                "replay_asset_inode": replay.asset_inode,
                "baseline_final_inode": baseline.profiles["final"].inode,
                "replay_final_inode": replay.profiles["final"].inode,
            },
        ),
        _check(
            "approved-semantics-only",
            replay_receipt.phase_correction == phase_correction
            and _canonical_model_sha256(replay_receipt.phase_correction)
            == phase_correction_sha256
            and replay_receipt.provenance.parameters.get("phase_correction_source")
            == "approved_semantics"
            and replay_receipt.provenance.parameters.get("comparison_report") is None
            and replay_receipt.provenance.parameters.get("candidate_motion_report")
            is None,
            "replay consumed the approved phase correction without re-reading comparison media",
            {
                "phase_correction_id": phase_correction.id,
                "phase_correction_sha256": phase_correction_sha256,
                "source": replay_receipt.provenance.parameters.get(
                    "phase_correction_source"
                ),
            },
        ),
        _check(
            "semantic-timeline-fingerprint",
            baseline.semantic_timeline_sha256 == replay.semantic_timeline_sha256,
            "volatile identities removed from both timelines leave identical semantics",
            {
                "baseline": baseline.semantic_timeline_sha256,
                "replay": replay.semantic_timeline_sha256,
            },
        ),
        _check(
            "deterministic-preview-equivalence",
            baseline.profiles["preview"].sha256
            == replay.profiles["preview"].sha256,
            "clean replay reproduces the preview bytes",
            {
                "baseline": baseline.profiles["preview"].sha256,
                "replay": replay.profiles["preview"].sha256,
            },
        ),
        _check(
            "deterministic-final-equivalence",
            baseline.profiles["final"].sha256 == replay.profiles["final"].sha256,
            "clean replay reproduces the selected final bytes",
            {
                "baseline": baseline.profiles["final"].sha256,
                "replay": replay.profiles["final"].sha256,
            },
        ),
        _check(
            "source-immutability",
            source_before == source_after == spec.source_sha256,
            "the permitted PNG is unchanged across replay",
            {"before": source_before, "after": source_after},
        ),
        _check(
            "source-only-render-lineage",
            all(
                proof.input_hashes == expected_inputs
                and not proof.forbidden_hashes_present
                for proof in replay.profiles.values()
            ),
            "both render profiles contain only the permitted PNG hash",
            {
                name: proof.input_hashes for name, proof in replay.profiles.items()
            },
        ),
        _check(
            "reference-media-not-copied",
            not copied_reference_files,
            "no file in the clean editing home has the sealed reference byte hash",
            {"matching_files": copied_reference_files},
        ),
        _check(
            "old-render-not-reused",
            all(path not in replay_commands for path in baseline_media_paths)
            and baseline.project_root not in replay_commands,
            "no baseline render or project path appears in replay renderer commands",
            {
                "forbidden_media_paths": baseline_media_paths,
                "forbidden_project_root": baseline.project_root,
            },
        ),
        _check(
            "canonical-records-recreated",
            all(Path(path).is_file() for path in canonical_paths)
            and not replay.hidden_manual_steps,
            "treatment, patch, version, decision graph, and manifest were recreated",
            {"paths": canonical_paths},
        ),
        _check(
            "derived-media-recreated",
            bool(replay.derivative_artifacts)
            and all(Path(path).is_file() for path in replay.derivative_artifacts),
            "proxy, thumbnail, and contact-sheet derivatives were recreated",
            {"paths": replay.derivative_artifacts},
        ),
        _check(
            "qc-and-editable-exports",
            all(proof.qc_overall == "pass" for proof in replay.profiles.values())
            and all(
                proof.validator_present and Path(proof.path).is_file()
                for proof in replay.editable_exports.values()
            ),
            "preview/final QC and Kdenlive/OTIO validation pass",
            {
                "qc": {
                    name: proof.qc_overall
                    for name, proof in replay.profiles.items()
                },
                "exports": {
                    name: proof.validator_tool
                    for name, proof in replay.editable_exports.items()
                },
            },
        ),
        _check(
            "no-hidden-manual-steps",
            not baseline.hidden_manual_steps and not replay.hidden_manual_steps,
            "both receipts explicitly contain no hidden manual steps",
            {},
        ),
    ]
    overall: Literal["pass", "fail"] = (
        "fail" if any(item.status != "pass" for item in checks) else "pass"
    )
    report_path = selected_output / "clean-rerun-v2.json"
    report = CleanRerunReportV2(
        implementation_revision=revision,
        git_clean=True,
        spec_id=spec.id,
        spec_path=str(selected_spec_path),
        spec_sha256=sha256_file(selected_spec_path),
        source_sha256=spec.source_sha256,
        reference_sha256=spec.reference_sha256,
        approved_phase_correction_id=phase_correction.id,
        approved_phase_correction_sha256=phase_correction_sha256,
        baseline=baseline,
        replay=replay,
        checks=checks,
        artifacts={
            "report": str(report_path),
            "spec": str(selected_spec_path),
            "baseline_receipt": str(selected_baseline_path),
            "replay_receipt": str(replay_receipt_path),
            "replay_workspace": str(clean_home),
            "replay_final": replay.profiles["final"].path,
            "replay_preview": replay.profiles["preview"].path,
            "replay_kdenlive": replay.editable_exports["kdenlive"].path,
            "replay_otio": replay.editable_exports["otio"].path,
        },
        provenance=Provenance(
            tool="aoa-editing-clean-rerun-v2",
            tool_version=version("aoa-editing"),
            parameters={
                "fresh_editing_home": str(clean_home),
                "application_path": [
                    "create_project",
                    "seal_reference_hash",
                    "ingest_permitted_source",
                    "analyze",
                    "propose_reference_v2",
                    "accept_treatment",
                    "render_preview_and_final",
                    "quality",
                    "interchange",
                ],
                "baseline_media_admitted_as_input": False,
                "reference_media_admitted_as_input": False,
                "approved_semantics_only": True,
            },
            deterministic=True,
        ),
        overall=overall,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def semantic_timeline_sha256(version_record: ProjectVersion) -> str:
    """Stable timeline identity after volatile project/application IDs are removed."""

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

    normalized = normalize(version_record.timeline.model_dump(mode="json"))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_baseline(
    spec: ReferenceReconstructionSpecV2,
    spec_path: Path,
    receipt: ReferenceReconstructionPassReceiptV2,
    source_path: Path,
) -> None:
    if (
        receipt.overall != "pass"
        or receipt.pass_kind != "phase_compensated_matrix"
        or receipt.sequence != "5r"
        or receipt.spec_id != spec.id
        or receipt.spec_sha256 != sha256_file(spec_path)
        or receipt.source_sha256 != spec.source_sha256
        or receipt.reference_sha256 != spec.reference_sha256
        or receipt.reference_media_used_as_input
        or receipt.hidden_manual_steps
        or receipt.phase_correction is None
        or receipt.phase_correction.spec_id != spec.id
        or receipt.phase_correction.spec_sha256 != sha256_file(spec_path)
        or sha256_file(source_path) != spec.source_sha256
    ):
        raise CleanRerunV2Error(
            "baseline receipt, frozen spec, and permitted source do not form "
            "one passing phase-compensated reconstruction"
        )


def _application_proof(
    receipt: ReferenceReconstructionPassReceiptV2,
    receipt_path: Path,
) -> CleanRerunApplicationProofV2:
    root = Path(receipt.project_root).resolve(strict=True)
    manifest_path = Path(receipt.project_manifest).resolve(strict=True)
    manifest = _read_model(manifest_path, ProjectManifest)
    asset_record_path = root / "assets" / receipt.asset_id / "asset.json"
    asset = _read_model(asset_record_path, Asset)
    asset_path = (root / asset.stored_path).resolve(strict=True)
    derivatives_path = root / "assets" / asset.id / "derivatives.json"
    derivatives = _read_model(derivatives_path, DerivedMediaManifest)
    derivative_artifacts = [
        str((root / relative).resolve(strict=True))
        for relative in derivatives.artifacts.values()
    ]
    treatment_path = Path(receipt.treatment_path).resolve(strict=True)
    treatment = _read_model(treatment_path, Treatment)
    patch_path = root / "patches" / f"{treatment.patch.id}.json"
    version_path = Path(receipt.timeline_projection).resolve(strict=True)
    version_record = _read_model(version_path, ProjectVersion)
    graph_path = Path(receipt.approved_edit_graph).resolve(strict=True)
    graph = _read_model(graph_path, EditorialDecisionGraph)
    decision_log = root / "decisions" / "log"
    if (
        manifest.id != receipt.project_id
        or asset.id != receipt.asset_id
        or treatment.id != receipt.treatment_id
        or version_record.id != receipt.version_id
        or version_record.applied_patch_id != treatment.patch.id
        or graph.id != version_record.decision_graph_id
        or graph.version_id != version_record.id
        or graph.treatment_id != treatment.id
        or not decision_log.is_dir()
        or not patch_path.is_file()
    ):
        raise CleanRerunV2Error("reconstruction receipt application records do not join")

    profiles: dict[str, CleanRerunRenderProofV2] = {}
    for profile_name in ("preview", "final"):
        profile = receipt.profiles.get(profile_name)
        if not isinstance(profile, dict):
            raise CleanRerunV2Error(f"receipt lacks {profile_name} profile")
        media_path = Path(str(profile["path"])).resolve(strict=True)
        qc_path = Path(str(profile["qc_path"])).resolve(strict=True)
        qc = _read_model(qc_path, QCReport)
        if qc.overall != "pass":
            raise CleanRerunV2Error(f"{profile_name} QC is not passing")
        lineage_path = Path(str(profile["lineage_path"])).resolve(strict=True)
        lineage = _object(lineage_path)
        input_hashes = lineage.get("input_hashes")
        forbidden_hashes = lineage.get("forbidden_hashes_present")
        if not isinstance(input_hashes, list) or not all(
            isinstance(item, str) for item in input_hashes
        ):
            raise CleanRerunV2Error(f"{profile_name} lineage input hashes are invalid")
        if not isinstance(forbidden_hashes, list) or not all(
            isinstance(item, str) for item in forbidden_hashes
        ):
            raise CleanRerunV2Error(
                f"{profile_name} lineage forbidden hashes are invalid"
            )
        profiles[profile_name] = CleanRerunRenderProofV2(
            profile=profile_name,
            path=str(media_path),
            sha256=sha256_file(media_path),
            inode=media_path.stat().st_ino,
            qc_path=str(qc_path),
            qc_sha256=sha256_file(qc_path),
            qc_overall="pass",
            lineage_path=str(lineage_path),
            lineage_sha256=sha256_file(lineage_path),
            input_hashes=input_hashes,
            forbidden_hashes_present=forbidden_hashes,
        )

    exports: dict[str, CleanRerunExportProofV2] = {}
    for format_name in ("kdenlive", "otio"):
        export = receipt.editable_exports.get(format_name)
        if not isinstance(export, dict):
            raise CleanRerunV2Error(f"receipt lacks {format_name} export")
        relative_path = export.get("output_path")
        validator = export.get("validator")
        if not isinstance(relative_path, str) or not isinstance(validator, dict):
            raise CleanRerunV2Error(f"{format_name} export is not validator-backed")
        output_path = (root / relative_path).resolve(strict=True)
        tool = validator.get("tool")
        if not isinstance(tool, str) or not tool:
            raise CleanRerunV2Error(f"{format_name} validator tool is absent")
        exports[format_name] = CleanRerunExportProofV2(
            format=format_name,
            path=str(output_path),
            sha256=sha256_file(output_path),
            validator_tool=tool,
        )

    return CleanRerunApplicationProofV2(
        receipt_id=receipt.id,
        receipt_path=str(receipt_path.resolve(strict=True)),
        receipt_sha256=sha256_file(receipt_path),
        project_id=receipt.project_id,
        project_root=str(root),
        project_manifest_path=str(manifest_path),
        project_manifest_sha256=sha256_file(manifest_path),
        asset_id=asset.id,
        asset_path=str(asset_path),
        asset_sha256=sha256_file(asset_path),
        asset_inode=asset_path.stat().st_ino,
        derivatives_manifest_path=str(derivatives_path),
        derivatives_manifest_sha256=sha256_file(derivatives_path),
        derivative_artifacts=derivative_artifacts,
        treatment_id=treatment.id,
        treatment_path=str(treatment_path),
        treatment_sha256=sha256_file(treatment_path),
        patch_id=treatment.patch.id,
        patch_path=str(patch_path),
        patch_sha256=sha256_file(patch_path),
        version_id=version_record.id,
        version_path=str(version_path),
        version_sha256=sha256_file(version_path),
        decision_graph_id=graph.id,
        decision_graph_path=str(graph_path),
        decision_graph_sha256=sha256_file(graph_path),
        decision_log_path=str(decision_log),
        semantic_timeline_sha256=semantic_timeline_sha256(version_record),
        profiles=profiles,
        editable_exports=exports,
        hidden_manual_steps=receipt.hidden_manual_steps,
    )


def _passing_readiness_revision(receipt: dict[str, Any]) -> str:
    revision = receipt.get("git_revision")
    checks = receipt.get("checks")
    clean = (
        isinstance(checks, list)
        and any(
            isinstance(item, dict)
            and item.get("id") == "git-clean"
            and item.get("status") == "pass"
            for item in checks
        )
    )
    if (
        receipt.get("overall") != "pass"
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not clean
    ):
        raise CleanRerunV2Error("clean rerun v2 requires clean passing readiness")
    return revision


def _render_command_text(receipt: ReferenceReconstructionPassReceiptV2) -> str:
    values = [
        profile.get("lineage", {}).get("command", [])
        for profile in receipt.profiles.values()
        if isinstance(profile, dict)
    ]
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def _matching_digest_files(root: Path, digest: str) -> list[str]:
    matches: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if sha256_file(path) == digest:
            matches.append(str(path))
    return matches


def _canonical_model_sha256(model: BaseModel | None) -> str:
    if model is None:
        raise CleanRerunV2Error("approved semantics are absent")
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CleanRerunV2Error(f"cannot load {path}: {error}") from error


def _object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CleanRerunV2Error(f"cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CleanRerunV2Error(f"expected JSON object: {path}")
    return payload


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
