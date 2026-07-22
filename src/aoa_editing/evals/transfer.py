"""Independent-source transfer evaluation for an operational technique packet."""

from __future__ import annotations

import json
import math
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from aoa_editing.analysis.reference import measure_video_motion
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    CheckResult,
    EditorialDecisionGraph,
    Intent,
    Provenance,
    Scenario,
    TechniqueApplicationRecord,
    TechniquePacket,
    TechniqueTransferReport,
    new_id,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.knowledge.techniques import add_application_record
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService


class TransferError(RuntimeError):
    """Technique transfer failed a source-neutrality or rendering invariant."""


def run_technique_transfer(
    packet: TechniquePacket,
    source_path: Path,
    output_root: Path,
    *,
    runtime_settings: Settings | None = None,
    canvas_width: int = 720,
    canvas_height: int = 1280,
    forbidden_hashes: list[str] | None = None,
) -> TechniqueTransferReport:
    """Apply a packet to unrelated media through the normal project workflow."""

    recipe = packet.camera_motion
    if recipe is None:
        raise TransferError("packet has no camera motion recipe")
    source_path = source_path.expanduser().resolve(strict=True)
    if output_root.exists():
        raise TransferError(f"transfer root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    selected = runtime_settings or Settings.from_env()
    workspace = output_root / "workspace"
    settings = Settings.for_home(
        workspace,
        runtime_root=selected.runtime_root,
        bind_port=selected.bind_port,
    )
    source_before = sha256_file(source_path)
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "Independent camera-technique transfer",
        Intent(
            text="Apply a reviewed source-neutral camera technique to unrelated procedural art.",
            scenario=Scenario.STILL_MOTION,
            target_duration_seconds=recipe.duration_frames / recipe.frame_rate.fps,
            frame_format=f"{recipe.aspect_width}:{recipe.aspect_height}",
            privacy_mode="local-only",
            automation_level="review-before-apply",
            constraints=["Do not use any reference-case media."],
        ),
    )
    for digest in forbidden_hashes or []:
        editing.seal_reference(project.id, digest)
    asset, _ = editing.ingest(project.id, source_path)
    evidence = AnalysisService(store).analyze(project.id, asset.id)
    treatment = editing.propose_technique(
        project.id,
        asset_id=asset.id,
        technique=packet,
        width=canvas_width,
        height=canvas_height,
    )
    edit_version = editing.accept_treatment(project.id, treatment.id)
    if edit_version.decision_graph_id is None:
        raise TransferError("accepted transfer has no decision graph")

    profiles: dict[str, Any] = {}
    for profile_name in ("preview", "final"):
        job = RenderService(store).render(
            project.id, edit_version.id, profile=profile_name
        )
        qc = QualityService(store).inspect(
            project.id, edit_version.id, profile=profile_name
        )
        path = store.project_path(project.id) / job.output_paths[0]
        lineage_path = path.parent / "lineage.json"
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        profiles[profile_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "qc_path": str(store.project_path(project.id) / "qc" / f"{qc.id}.json"),
            "qc": qc.model_dump(mode="json"),
            "lineage_path": str(lineage_path),
            "lineage": lineage,
            "render_plan": str(path.parent / "render-plan.json"),
        }
    final_path = Path(profiles["final"]["path"])
    measured = measure_video_motion(source_path, final_path, sample_step=5)
    motion = _motion_metrics(packet, measured.measurements)

    contact_sheet = output_root / "transfer-contact-sheet.jpg"
    sheet_command = _contact_sheet(
        final_path,
        contact_sheet,
        recipe.duration_frames / recipe.frame_rate.fps,
    )
    kdenlive = KdenliveExporter(store).export(project.id, edit_version.id)
    otio = OTIOExporter(store).export(project.id, edit_version.id)
    graph_path = (
        store.project_path(project.id)
        / "decisions"
        / "graphs"
        / f"{edit_version.decision_graph_id}.json"
    )
    graph = EditorialDecisionGraph.model_validate_json(
        graph_path.read_text(encoding="utf-8")
    )
    derivatives = store.load_derivatives(project.id, asset.id)
    final_lineage = profiles["final"]["lineage"]
    forbidden = set(forbidden_hashes or [])
    checks = [
        _check(
            "materially-different-source",
            source_before not in forbidden,
            "transfer source hash differs from all sealed reference-case media",
            {"source_sha256": source_before, "forbidden_hashes": sorted(forbidden)},
        ),
        _check(
            "source-only-lineage",
            final_lineage.get("input_hashes") == [source_before]
            and not forbidden.intersection(final_lineage.get("input_hashes", [])),
            "final lineage contains only the independent source",
            {"input_hashes": final_lineage.get("input_hashes")},
        ),
        _check(
            "camera-scale-transfer",
            motion["scale_rmse"] <= 0.01,
            "rendered scale curve follows the packet",
            {"actual": motion["scale_rmse"], "maximum": 0.01},
        ),
        _check(
            "camera-rotation-transfer",
            motion["rotation_rmse_degrees"] <= 0.10,
            "rendered rotation curve follows the packet",
            {"actual": motion["rotation_rmse_degrees"], "maximum": 0.10},
        ),
        _check(
            "camera-center-transfer",
            motion["center_rmse_normalized"] <= 0.01,
            "rendered center curve follows the packet",
            {"actual": motion["center_rmse_normalized"], "maximum": 0.01},
        ),
        _check(
            "decision-contract",
            graph.state == "approved"
            and bool(graph.decisions)
            and graph.decisions[0].patch_id == treatment.patch.id,
            "transfer is represented by an approved reversible decision",
            {"graph_path": str(graph_path)},
        ),
        _check(
            "derivatives-created",
            {"proxy", "thumbnail", "contact_sheet"}.issubset(derivatives.artifacts),
            "independent source passed normal proxy and gallery ingest",
            {"artifacts": derivatives.artifacts},
        ),
        _check(
            "preview-final-qc",
            all(item["qc"]["overall"] != "fail" for item in profiles.values()),
            "preview and final transfer renders pass QC",
            {name: item["qc"]["overall"] for name, item in profiles.items()},
        ),
        _check(
            "editable-exports",
            kdenlive.validator is not None and otio.validator is not None,
            "Kdenlive and OTIO transfer projections validate",
            {
                "kdenlive": kdenlive.output_path,
                "otio": otio.output_path,
            },
        ),
        _check(
            "source-immutable",
            sha256_file(source_path) == source_before,
            "independent source remains unchanged",
            {"sha256": source_before},
        ),
    ]
    overall = _overall(checks)
    report_id = new_id("transfer")
    report_path = output_root / "transfer.json"
    updated_packet_path = output_root / "technique-candidate-with-transfer.json"
    updated_packet = add_application_record(
        packet,
        TechniqueApplicationRecord(
            case_id=report_id,
            source_class="procedural textured landscape illustration",
            outcome=overall,
            evidence_path="runtime-eval:transfer/transfer.json",
            metrics={
                "scale_rmse": float(motion["scale_rmse"]),
                "rotation_rmse_degrees": float(motion["rotation_rmse_degrees"]),
                "center_rmse_normalized": float(motion["center_rmse_normalized"]),
            },
        ),
    )
    updated_packet_path.write_text(
        updated_packet.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    report = TechniqueTransferReport(
        id=report_id,
        technique_id=packet.id,
        technique_revision=packet.revision,
        source_sha256=source_before,
        project_id=project.id,
        version_id=edit_version.id,
        motion=motion,
        profiles=profiles,
        checks=checks,
        artifacts={
            "report": str(report_path),
            "contact_sheet": str(contact_sheet),
            "approved_edit_graph": str(graph_path),
            "decision_log": str(store.project_path(project.id) / "decisions" / "log"),
            "updated_candidate_packet": str(updated_packet_path),
            "kdenlive": str(store.project_path(project.id) / kdenlive.output_path),
            "otio": str(store.project_path(project.id) / otio.output_path),
        },
        provenance=Provenance(
            tool="aoa-editing-technique-transfer",
            tool_version=version("aoa-editing"),
            command=sheet_command,
            parameters={
                "canvas": [canvas_width, canvas_height],
                "evidence_ids": [item.id for item in evidence],
                "normal_product_workflow": True,
            },
            deterministic=True,
        ),
        overall=overall,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def _motion_metrics(packet: TechniquePacket, actual_items: tuple[Any, ...]) -> dict[str, Any]:
    recipe = packet.camera_motion
    if recipe is None:  # pragma: no cover - caller guards
        raise TransferError("packet has no camera recipe")
    positions = {item.frame: item for item in recipe.position}
    scales = {item.frame: item for item in recipe.scale}
    rotations = {item.frame: item for item in recipe.rotation}
    pairs = [
        (item, positions[item.frame], scales[item.frame], rotations[item.frame])
        for item in actual_items
        if item.frame in positions and item.frame in scales and item.frame in rotations
    ]
    if not pairs:
        raise TransferError("render motion samples do not overlap technique keyframes")
    return {
        "sample_count": len(pairs),
        "sample_frames": [item.frame for item, *_ in pairs],
        "scale_rmse": _rmse(
            item.scale_relative_to_contain - scale.value for item, _, scale, _ in pairs
        ),
        "rotation_rmse_degrees": _rmse(
            item.rotation_degrees - rotation.value for item, _, _, rotation in pairs
        ),
        "center_rmse_normalized": _rmse(
            math.hypot(item.center_x - position.x, item.center_y - position.y)
            for item, position, _, _ in pairs
        ),
    }


def _contact_sheet(video: Path, output: Path, duration: float) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps=6/{max(duration, 0.04)},scale=320:-2:flags=lanczos,tile=3x2",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise TransferError(result.stderr[-3000:] or "contact sheet generation failed")
    return command


def _rmse(values: Any) -> float:
    items = list(values)
    return math.sqrt(sum(value * value for value in items) / len(items))


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
