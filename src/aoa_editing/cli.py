"""Human CLI over the same application services used by other adapters."""

from __future__ import annotations

import json
import subprocess
import threading
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import uvicorn
from pydantic import TypeAdapter
from rich.console import Console

from aoa_editing import __version__
from aoa_editing.agent import run_stdio
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.api.app import create_app
from aoa_editing.application.language import NaturalLanguagePatchService
from aoa_editing.application.reference_workspace import (
    MotionCorrectionService,
    ReferenceWorkspaceService,
)
from aoa_editing.application.screen_workflow_experience import (
    admit_screen_workflow_experience,
)
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    FrameRange,
    FrameRate,
    Intent,
    MotionCorrectionOperationV2,
    MotionCorrectionReviewDecisionV2,
    ReferenceComparisonReport,
    ReferenceReconstructionPassKindV2,
    ReferenceReconstructionPassReceiptV2,
    ReferenceReconstructionSpec,
    ReferenceReconstructionSpecV2,
    Scenario,
    ScreenWorkflowBeatInput,
    ScreenWorkflowEditSpec,
    ScreenWorkflowExperienceAdmission,
    ScreenWorkflowPlan,
    ScreenWorkflowVoiceoverCue,
    ScreenWorkflowVoiceoverTiming,
)
from aoa_editing.evals.clean_rerun import run_clean_rerun
from aoa_editing.evals.clean_rerun_v2 import run_clean_rerun_v2
from aoa_editing.evals.comparison import run_comparison
from aoa_editing.evals.comparison_v2 import (
    freeze_comparison_protocol_v2,
    run_comparison_v2,
)
from aoa_editing.evals.completion import run_completion_audit
from aoa_editing.evals.editorial_review_v2 import record_editorial_review_v2
from aoa_editing.evals.fixtures import create_transfer_fixture
from aoa_editing.evals.gate import require_readiness, run_readiness_gate
from aoa_editing.evals.generic import run_generic_suite
from aoa_editing.evals.local_ai_live import (
    DEFAULT_FIXTURE_TEXT,
    run_local_ai_live_eval,
)
from aoa_editing.evals.motion_recovery import run_motion_recovery_gate
from aoa_editing.evals.provider_aliases import run_provider_alias_eval
from aoa_editing.evals.reconstruction import run_reconstruction
from aoa_editing.evals.reconstruction_v2 import run_reconstruction_pass_v2
from aoa_editing.evals.reference import analyze_reference
from aoa_editing.evals.reference_v2 import analyze_reference_v2
from aoa_editing.evals.study_v2 import build_reconstruction_study_v2
from aoa_editing.evals.transfer import run_technique_transfer
from aoa_editing.evals.transfer_v2 import (
    run_technique_transfer_corpus_v2,
    system_photograph_provenance,
)
from aoa_editing.evals.workflow_reference import run_reference_workflow_study
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.probes import doctor_report
from aoa_editing.infrastructure.relocation import migrate_legacy_storage
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.knowledge.techniques import (
    extract_contain_reveal_technique,
    load_packet,
    upgrade_contain_reveal_technique_v2,
)
from aoa_editing.providers.catalog import load_capability_catalog
from aoa_editing.providers.service import ProviderService
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService

app = typer.Typer(no_args_is_help=True, help="Local-first AI video co-editor")
console = Console()
project_app = typer.Typer(no_args_is_help=True, help="Create and inspect projects")
asset_app = typer.Typer(no_args_is_help=True, help="Ingest and analyze immutable assets")
treatment_app = typer.Typer(no_args_is_help=True, help="Propose and accept editing treatments")
render_app = typer.Typer(no_args_is_help=True, help="Render and inspect outputs")
export_app = typer.Typer(no_args_is_help=True, help="Create editable interchange projections")
eval_app = typer.Typer(no_args_is_help=True, help="Run synthetic and gated evaluations")
gate_app = typer.Typer(no_args_is_help=True, help="Control sealed reference access")
migration_app = typer.Typer(no_args_is_help=True, help="Plan and verify storage relocation")
reference_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect reference understanding and review typed motion corrections",
)
provider_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and invoke typed local-AI provider aliases",
)
workflow_app = typer.Typer(
    no_args_is_help=True,
    help="Plan terminal-first screen capture before media exists",
)
app.add_typer(project_app, name="project")
app.add_typer(asset_app, name="asset")
app.add_typer(treatment_app, name="treatment")
app.add_typer(render_app, name="render")
app.add_typer(export_app, name="export")
app.add_typer(eval_app, name="eval")
app.add_typer(gate_app, name="gate")
app.add_typer(migration_app, name="migrate")
app.add_typer(reference_app, name="reference")
app.add_typer(provider_app, name="provider")
app.add_typer(workflow_app, name="workflow")


def _services() -> tuple[ProjectStore, EditingService]:
    store = ProjectStore(Settings.from_env())
    return store, EditingService(store)


def _emit(model: Any) -> None:
    if hasattr(model, "model_dump_json"):
        typer.echo(model.model_dump_json(indent=2))
    else:
        typer.echo(json.dumps(model, indent=2, ensure_ascii=False, default=str))


def _reference_services() -> tuple[
    ProjectStore,
    ReferenceWorkspaceService,
    MotionCorrectionService,
]:
    store = ProjectStore(Settings.from_env())
    return store, ReferenceWorkspaceService(store), MotionCorrectionService(store)


def _git(repo: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise typer.BadParameter(result.stderr.strip() or "git command failed")
    return result.stdout


@app.command()
def version() -> None:
    """Print the product version."""

    console.print(__version__)


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    create_roots: bool = typer.Option(
        False, "--create-roots", help="Create configured data/cache/tmp roots before probing"
    ),
) -> None:
    """Probe mandatory tools, optional adapters, and storage routes."""

    report = doctor_report(Settings.from_env(), create_roots=create_roots)
    if json_output:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        state = "READY" if report["ok"] else "NOT READY"
        console.print(f"AoA Editing doctor: {state}")
        console.print(f"Python: {report['python']['version']}")
        console.print(f"FFmpeg: {report['ffmpeg']['version'] or 'missing'}")
        console.print(f"ffprobe: {report['ffprobe']['version'] or 'missing'}")
        console.print(f"Kdenlive: {report['kdenlive']['summary'] or 'optional / missing'}")
        console.print(f"MLT: {report['melt']['version'] or 'optional / missing'}")
        providers = report["providers"]
        console.print(
            "AI aliases: "
            f"{len(providers.get('bindings', []))} declared "
            f"(contract_ok={providers['contract_ok']})"
        )
        console.print(
            "Editing home: "
            f"{report['editing_home']['path']} "
            f"({report['editing_home']['resolution_source']})"
        )
        for name, status in report["paths"].items():
            console.print(
                f"{name}: {status['path']} (owner={status['owner']}, writable={status['writable']})"
            )
    if not report["ok"]:
        raise typer.Exit(1)


@provider_app.command("catalog")
def provider_catalog() -> None:
    """Print tracked path-free capability declarations."""

    _emit(load_capability_catalog())


@provider_app.command("bindings")
def provider_bindings(
    no_probe: bool = typer.Option(
        False,
        "--no-probe",
        help="Resolve declarations and overlay without executing health probes",
    ),
) -> None:
    """Show untracked local bindings and current health without exposing secrets."""

    settings = Settings.from_env()
    _emit(ProviderService(settings).inspect_bindings(probe=not no_probe))


@provider_app.command("invoke")
def provider_invoke(
    alias: str,
    request: Annotated[
        Path,
        typer.Option("--request", help="Path to a JSON request matching the declaration"),
    ],
    privacy_mode: Annotated[
        str,
        typer.Option("--privacy-mode"),
    ] = "local-only",
    explicit_opt_in: bool = typer.Option(False, "--explicit-opt-in"),
    no_fallback: bool = typer.Option(False, "--no-fallback"),
) -> None:
    """Invoke one alias and emit its immutable resolved-provider receipt."""

    try:
        payload = json.loads(request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"cannot read provider request: {error}") from error
    if not isinstance(payload, dict):
        raise typer.BadParameter("provider request JSON must be an object")
    result = ProviderService(Settings.from_env()).invoke(
        alias,
        payload,
        privacy_mode=privacy_mode,
        explicit_opt_in=explicit_opt_in,
        allow_fallback=not no_fallback,
    )
    _emit(result)
    if result.receipt.outcome not in {"succeeded", "partial"}:
        raise typer.Exit(2)


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Option("--name", prompt=True)],
    scenario: Annotated[Scenario, typer.Option("--scenario")],
    intent: Annotated[str, typer.Option("--intent", prompt=True)],
    target_duration: Annotated[float | None, typer.Option("--target-duration", min=0.1)] = None,
) -> None:
    """Create a project manifest without touching source media."""

    _, service = _services()
    project = service.create_project(
        name,
        Intent(
            text=intent,
            scenario=scenario,
            target_duration_seconds=target_duration,
        ),
    )
    _emit(project)


@project_app.command("list")
def project_list() -> None:
    """List project manifests under the configured data root."""

    store, _ = _services()
    _emit([project.model_dump(mode="json") for project in store.list_projects()])


@project_app.command("show")
def project_show(project_id: str) -> None:
    store, _ = _services()
    project = store.load_project(project_id)
    payload = project.model_dump(mode="json")
    if project.current_version_id:
        payload["current_version"] = store.load_version(project_id).model_dump(mode="json")
    payload["asset_objects"] = [
        asset.model_dump(mode="json") for asset in store.list_assets(project_id)
    ]
    _emit(payload)


@asset_app.command("ingest")
def asset_ingest(project_id: str, source: Path) -> None:
    """Copy a source immutably, probe it, and create resumable derivatives."""

    store, service = _services()
    asset, evidence = service.ingest(project_id, source)
    _emit(
        {
            "asset": asset.model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json"),
            "derivatives": store.load_derivatives(project_id, asset.id).model_dump(mode="json"),
        }
    )


@asset_app.command("analyze")
def asset_analyze(
    project_id: str,
    asset_id: str,
    transcribe: bool = typer.Option(False, "--transcribe"),
) -> None:
    """Produce evidence without changing the edit graph."""

    store, _ = _services()
    records = AnalysisService(store).analyze(project_id, asset_id, transcribe=transcribe)
    _emit([record.model_dump(mode="json") for record in records])


@workflow_app.command("plan-script")
def workflow_plan_script(
    title: Annotated[str, typer.Option("--title")],
    script: Annotated[Path, typer.Option("--script", help="UTF-8 script or narration file")],
    output: Annotated[Path, typer.Option("--output", help="New capture-plan JSON path")],
    beats: Annotated[
        Path | None,
        typer.Option("--beats", help="Optional authored ScreenWorkflowBeatInput JSON list"),
    ] = None,
    reviewed_by: Annotated[str | None, typer.Option("--reviewed-by")] = None,
    reference_study: Annotated[
        list[str] | None,
        typer.Option("--reference-study", help="Candidate workflow-study receipt id"),
    ] = None,
) -> None:
    """Turn a script into a media-free capture plan; authored beats can make it capture-ready."""

    script_text = script.read_text(encoding="utf-8")
    authored = None
    if beats is not None:
        authored = TypeAdapter(list[ScreenWorkflowBeatInput]).validate_json(
            beats.read_text(encoding="utf-8")
        )
    if reviewed_by is not None and authored is None:
        raise typer.BadParameter("--reviewed-by requires --beats")
    _, service = _services()
    plan = service.plan_screen_workflow(
        title=title,
        script=script_text,
        beats=authored,
        reviewed_by=reviewed_by,
        reference_study_ids=reference_study or [],
    )
    selected_output = output.expanduser().resolve()
    if selected_output.exists():
        raise typer.BadParameter("capture-plan output already exists")
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _emit(plan)


@workflow_app.command("admit-experience")
def workflow_admit_experience(
    admission_path: Annotated[
        Path,
        typer.Option(
            "--admission",
            help="Owner-reviewed private experience-admission JSON",
        ),
    ],
) -> None:
    """Bind private reviewed evidence to an immutable public-safe projection hash."""

    admission = ScreenWorkflowExperienceAdmission.model_validate_json(
        admission_path.read_text(encoding="utf-8")
    )
    settings = Settings.from_env()
    receipt, path = admit_screen_workflow_experience(admission, settings=settings)
    _emit(
        {
            "schema_version": "1.0.0",
            "receipt_id": receipt.id,
            "receipt_path": str(path),
            "public_projection_sha256": receipt.public_projection_sha256,
            "public_projection": receipt.public_projection.model_dump(mode="json"),
            "private_evidence_retained": True,
        }
    )


@workflow_app.command("time-voiceover")
def workflow_time_voiceover(
    project_id: str,
    plan_path: Annotated[Path, typer.Option("--plan", help="Reviewed capture plan")],
    asset_id: Annotated[str, typer.Option("--asset", help="Ingested voiceover asset id")],
    output: Annotated[Path, typer.Option("--output", help="New timing-draft JSON path")],
    transcribe: Annotated[
        bool,
        typer.Option("--transcribe/--no-transcribe", help="Use typed local STT when available"),
    ] = True,
    frame_rate_numerator: Annotated[
        int,
        typer.Option("--frame-rate-numerator", min=1),
    ] = 30,
    frame_rate_denominator: Annotated[
        int,
        typer.Option("--frame-rate-denominator", min=1),
    ] = 1,
) -> None:
    """Measure one ingested narration and draft script-beat timing for human review."""

    plan = ScreenWorkflowPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    _, service = _services()
    timing = service.time_screen_workflow_voiceover(
        project_id,
        plan=plan,
        asset_id=asset_id,
        timeline_frame_rate=FrameRate(
            numerator=frame_rate_numerator,
            denominator=frame_rate_denominator,
        ),
        transcribe=transcribe,
    )
    selected_output = output.expanduser().resolve()
    if selected_output.exists():
        raise typer.BadParameter("voiceover timing output already exists")
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(timing.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _emit(timing)


@workflow_app.command("review-voiceover")
def workflow_review_voiceover(
    plan_path: Annotated[Path, typer.Option("--plan", help="Reviewed capture plan")],
    draft_path: Annotated[Path, typer.Option("--draft", help="Voiceover timing draft")],
    cues_path: Annotated[
        Path,
        typer.Option("--cues", help="Reviewed ScreenWorkflowVoiceoverCue JSON list"),
    ],
    reviewed_by: Annotated[str, typer.Option("--reviewed-by")],
    review_note: Annotated[str, typer.Option("--review-note")],
    output: Annotated[Path, typer.Option("--output", help="New reviewed timing JSON path")],
) -> None:
    """Approve or correct every narration boundary before it can enter an edit."""

    plan = ScreenWorkflowPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    draft = ScreenWorkflowVoiceoverTiming.model_validate_json(
        draft_path.read_text(encoding="utf-8")
    )
    cues = TypeAdapter(list[ScreenWorkflowVoiceoverCue]).validate_json(
        cues_path.read_text(encoding="utf-8")
    )
    _, service = _services()
    timing = service.review_screen_workflow_voiceover(
        plan=plan,
        draft=draft,
        cues=cues,
        reviewed_by=reviewed_by,
        review_note=review_note,
    )
    selected_output = output.expanduser().resolve()
    if selected_output.exists():
        raise typer.BadParameter("reviewed voiceover timing output already exists")
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(timing.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _emit(timing)


@treatment_app.command("propose")
def treatment_propose(
    project_id: str,
    scenario: Annotated[Scenario | None, typer.Option("--scenario")] = None,
    asset_id: Annotated[str | None, typer.Option("--asset")] = None,
    duration: Annotated[float | None, typer.Option("--duration", min=0.1)] = None,
) -> None:
    """Generate and persist a reversible evidence-backed treatment."""

    _, service = _services()
    _emit(
        service.propose(
            project_id,
            scenario=scenario,
            asset_id=asset_id,
            duration_seconds=duration,
        )
    )


@treatment_app.command("accept")
def treatment_accept(project_id: str, treatment_id: str) -> None:
    """Accept one treatment and create an immutable version."""

    _, service = _services()
    _emit(service.accept_treatment(project_id, treatment_id))


@treatment_app.command("reconstruct")
def treatment_reconstruct(project_id: str, asset_id: str, spec_path: Path) -> None:
    """Propose a normal reversible treatment from a frozen reference spec."""

    _, service = _services()
    spec = ReferenceReconstructionSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    _emit(service.propose_reference(project_id, asset_id=asset_id, spec=spec))


@treatment_app.command("workflow")
def treatment_workflow(
    project_id: str,
    plan_path: Annotated[Path, typer.Option("--plan", help="Reviewed capture plan")],
    edit_spec_path: Annotated[
        Path,
        typer.Option("--edit-spec", help="Human-reviewed raw-recording bindings"),
    ],
) -> None:
    """Propose a reversible screen.workflow Treatment after raw capture is ingested."""

    _, service = _services()
    plan = ScreenWorkflowPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    edit_spec = ScreenWorkflowEditSpec.model_validate_json(
        edit_spec_path.read_text(encoding="utf-8")
    )
    _emit(service.propose_screen_workflow(project_id, plan=plan, edit_spec=edit_spec))


@treatment_app.command("revert")
def treatment_revert(project_id: str, version_id: str) -> None:
    """Apply the stored inverse patch as a new version."""

    _, service = _services()
    _emit(service.revert_version(project_id, version_id))


@treatment_app.command("language-preview")
def treatment_language_preview(
    project_id: str,
    command: Annotated[str, typer.Option("--command", prompt=True)],
    version_id: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    """Preview an allowlisted natural-language edit as a typed patch; never apply it."""

    store, _ = _services()
    _emit(NaturalLanguagePatchService(store).preview(project_id, command, version_id=version_id))


@reference_app.command("attach")
def reference_attach(
    project_id: str,
    study: Annotated[Path, typer.Option("--study", help="Reconstruction study v2")],
    reference_media: Annotated[
        Path | None,
        typer.Option(
            "--reference-media",
            help="Host-local sealed reference binding; its path is not stored canonically",
        ),
    ] = None,
) -> None:
    """Register frozen comparison evidence as a project-local understanding workspace."""

    _store, workspaces, _corrections = _reference_services()
    _emit(
        workspaces.attach(
            project_id,
            study_path=study,
            reference_media_path=reference_media,
        )
    )


@reference_app.command("show")
def reference_show(project_id: str, workspace_id: str) -> None:
    """Show all-frame curves, interpretations, versions, proposals, and reviews."""

    _store, workspaces, _corrections = _reference_services()
    _emit(workspaces.bundle(project_id, workspace_id))


@reference_app.command("preview-language")
def reference_preview_language(
    project_id: str,
    workspace_id: str,
    command: Annotated[str, typer.Option("--command", prompt=True)],
    version_id: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    """Translate editorial wording to typed, non-mutating motion diffs."""

    _store, _workspaces, corrections = _reference_services()
    _emit(
        corrections.preview_language(
            project_id,
            workspace_id,
            command,
            version_id=version_id,
        )
    )


@reference_app.command("preview-operations")
def reference_preview_operations(
    project_id: str,
    workspace_id: str,
    operations: Annotated[Path, typer.Option("--operations", help="JSON operation list")],
    version_id: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    """Preview explicitly authored semantic operations without applying them."""

    payload = json.loads(operations.read_text(encoding="utf-8"))
    parsed = TypeAdapter(list[MotionCorrectionOperationV2]).validate_python(payload)
    _store, _workspaces, corrections = _reference_services()
    _emit(
        corrections.preview_operations(
            project_id,
            workspace_id,
            parsed,
            version_id=version_id,
        )
    )


@reference_app.command("review")
def reference_review(
    project_id: str,
    proposal_id: str,
    decisions: Annotated[
        Path,
        typer.Option("--decisions", help="JSON approve/reject decision list"),
    ],
) -> None:
    """Apply only explicitly approved corrections through the canonical patch service."""

    payload = json.loads(decisions.read_text(encoding="utf-8"))
    parsed = TypeAdapter(list[MotionCorrectionReviewDecisionV2]).validate_python(
        payload
    )
    _store, _workspaces, corrections = _reference_services()
    _emit(corrections.review(project_id, proposal_id, parsed))


@render_app.command("run")
def render_run(
    project_id: str,
    version_id: str | None = typer.Option(None, "--version"),
    profile: str = typer.Option("preview", "--profile"),
    start_frame: int | None = typer.Option(None, "--start-frame", min=0),
    duration_frames: int | None = typer.Option(None, "--duration-frames", min=1),
) -> None:
    """Compile and atomically render preview or final media."""

    store, _ = _services()
    if (start_frame is None) != (duration_frames is None):
        raise typer.BadParameter("--start-frame and --duration-frames must be supplied together")
    _emit(
        RenderService(store).render(
            project_id,
            version_id,
            profile=profile,
            frame_range=(
                FrameRange(start=start_frame, duration=duration_frames)
                if start_frame is not None and duration_frames is not None
                else None
            ),
        )
    )


@render_app.command("qc")
def render_qc(
    project_id: str,
    version_id: str | None = typer.Option(None, "--version"),
    profile: str = typer.Option("preview", "--profile"),
) -> None:
    """Run technical QC against a completed render."""

    store, _ = _services()
    _emit(QualityService(store).inspect(project_id, version_id, profile=profile))


@export_app.command("kdenlive")
def export_kdenlive(
    project_id: str, version_id: str | None = typer.Option(None, "--version")
) -> None:
    """Export and live-validate an editable Kdenlive/MLT project."""

    store, _ = _services()
    _emit(KdenliveExporter(store).export(project_id, version_id))


@export_app.command("otio")
def export_otio(project_id: str, version_id: str | None = typer.Option(None, "--version")) -> None:
    """Export and round-trip an OpenTimelineIO projection."""

    store, _ = _services()
    _emit(OTIOExporter(store).export(project_id, version_id))


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port", min=1, max=65535),
    open_browser: bool = typer.Option(False, "--open", help="Open the workbench in a browser"),
) -> None:
    """Run the local workbench on a loopback address by default."""

    settings = Settings.from_env()
    selected_host = host or settings.bind_host
    selected_port = port or settings.bind_port
    if open_browser:
        url_host = "127.0.0.1" if selected_host in {"0.0.0.0", "::"} else selected_host
        timer = threading.Timer(1.0, webbrowser.open, args=(f"http://{url_host}:{selected_port}",))
        timer.daemon = True
        timer.start()
    uvicorn.run(
        create_app(settings),
        host=selected_host,
        port=selected_port,
        log_level="info",
    )


@app.command("agent")
def agent_stdio() -> None:
    """Serve the allowlisted line-delimited JSON agent protocol on stdin/stdout."""

    run_stdio()


@migration_app.command("storage-v1")
def migrate_storage_v1(
    legacy_data_root: Annotated[
        Path, typer.Option("--legacy-data-root", help="Explicit completed-v1 data root")
    ],
    inventory: Annotated[
        Path, typer.Option("--inventory", help="Checked-in pre-migration inventory")
    ],
    apply: Annotated[
        bool, typer.Option("--apply", help="Copy, verify, and promote into the resolved home")
    ] = False,
) -> None:
    """Dry-run by default; never cleans the explicitly supplied legacy root."""

    selected_inventory = inventory.expanduser().resolve(strict=True)
    report = migrate_legacy_storage(
        legacy_data_root=legacy_data_root,
        destination=Settings.from_env(),
        inventory_sha256=sha256_file(selected_inventory),
        dry_run=not apply,
    )
    _emit(report)


@eval_app.command("generic")
def eval_generic(
    output: Annotated[Path | None, typer.Option("--output", help="Evaluation run root")] = None,
) -> None:
    """Run all three scenarios on synthetic fixtures only."""

    settings = Settings.from_env()
    root = output or settings.evals_root / "generic" / "manual"
    _emit(run_generic_suite(root))


@eval_app.command("reference-analyze")
def eval_reference_analyze(
    output: Annotated[
        Path | None, typer.Option("--output", help="Immutable reference-analysis root")
    ] = None,
    sample_step: Annotated[
        int, typer.Option("--sample-step", min=1, help="Frames between affine samples")
    ] = 5,
) -> None:
    """Analyze sealed reference media only after a revision-matched readiness gate."""

    settings = Settings.from_env()
    receipt = require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    root = output or settings.evals_root / "reference" / "manual" / "analysis"
    spec = analyze_reference(
        Path(lock["source_image"]["path"]),
        Path(lock["reference_video"]["path"]),
        root,
        readiness_receipt=receipt,
        sample_step=sample_step,
    )
    _emit(spec)


@eval_app.command("reference-workflow-study")
def eval_reference_workflow_study(
    plan: Annotated[
        Path,
        typer.Option("--plan", help="Reviewed batch plan with external reference bindings"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="New immutable root beneath the configured evals root"),
    ] = None,
) -> None:
    """Analyze only explicitly reviewed human-free workflow ranges after readiness."""

    settings = Settings.from_env()
    root = output or settings.evals_root / "reference-workflow" / "manual"
    _emit(run_reference_workflow_study(plan, root, settings=settings))


@eval_app.command("reference-analyze-v2")
def eval_reference_analyze_v2(
    motion_gate: Annotated[
        Path,
        typer.Option("--motion-gate", help="Passing revision-bound motion recovery gate"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New immutable Reference Motion Evidence v2 root"),
    ],
    baseline_comparison: Annotated[
        Path | None,
        typer.Option("--baseline-comparison", help="Historical v1 comparison observation"),
    ] = None,
) -> None:
    """Analyze every sealed reference frame after readiness and motion recovery pass."""

    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    readiness_path = settings.evals_root / "readiness" / "latest.json"
    evidence, spec = analyze_reference_v2(
        Path(lock["source_image"]["path"]),
        Path(lock["reference_video"]["path"]),
        output,
        readiness_path=readiness_path,
        motion_gate_path=motion_gate,
        baseline_comparison_path=baseline_comparison,
    )
    _emit(
        {
            "evidence": evidence.model_dump(mode="json"),
            "spec": spec.model_dump(mode="json"),
        }
    )


@eval_app.command("reference-freeze-comparison-v2")
def eval_reference_freeze_comparison_v2(
    spec_path: Annotated[
        Path, typer.Option("--spec", help="Frozen Reference Reconstruction Spec v2")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="New immutable Comparison Protocol v2 root")
    ],
) -> None:
    """Freeze all objective algorithms and the human rubric before a v2 candidate."""

    repo = Path(__file__).resolve().parents[2]
    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    status = _git(repo, ["status", "--porcelain"]).strip()
    _emit(
        freeze_comparison_protocol_v2(
            spec_path,
            output,
            implementation_revision=revision,
            git_clean=not bool(status),
        )
    )


@eval_app.command("reference-compare-v2")
def eval_reference_compare_v2(
    protocol_path: Annotated[
        Path, typer.Option("--protocol", help="Frozen Comparison Protocol v2")
    ],
    spec_path: Annotated[
        Path, typer.Option("--spec", help="Frozen Reference Reconstruction Spec v2")
    ],
    candidate_path: Annotated[
        Path, typer.Option("--candidate", help="Normally rendered v2 candidate")
    ],
    lineage_path: Annotated[Path, typer.Option("--lineage", help="Candidate render lineage JSON")],
    output: Annotated[Path, typer.Option("--output", help="New immutable Comparison v2 root")],
    source: Annotated[
        Path | None, typer.Option("--source", help="Permitted visual source override")
    ] = None,
    human_review: Annotated[
        Path | None,
        typer.Option("--human-review", help="Optional completed Editorial Review v2"),
    ] = None,
) -> None:
    """Run all-frame objective comparison; never infer the human editorial verdict."""

    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    _emit(
        run_comparison_v2(
            protocol_path,
            spec_path,
            source or Path(lock["source_image"]["path"]),
            candidate_path,
            Path(lock["reference_video"]["path"]),
            output,
            lineage_path=lineage_path,
            human_review_path=human_review,
        )
    )


@eval_app.command("reference-review-v2")
def eval_reference_review_v2(
    report_path: Annotated[
        Path, typer.Option("--comparison", help="Objective Comparison v2 report")
    ],
    protocol_path: Annotated[
        Path, typer.Option("--protocol", help="Frozen Comparison Protocol v2")
    ],
    review_path: Annotated[
        Path, typer.Option("--review", help="Completed human Editorial Review v2")
    ],
    output: Annotated[Path, typer.Option("--output", help="New reviewed Comparison v2 report")],
) -> None:
    """Attach an attributable human verdict as a new immutable report revision."""

    _emit(
        record_editorial_review_v2(
            report_path,
            protocol_path,
            review_path,
            output,
        )
    )


@eval_app.command("reference-reconstruct")
def eval_reference_reconstruct(
    spec_path: Annotated[Path, typer.Option("--spec", help="Frozen reference spec")],
    output: Annotated[Path, typer.Option("--output", help="New reconstruction receipt root")],
    source: Annotated[
        Path | None, typer.Option("--source", help="Permitted visual source override")
    ] = None,
) -> None:
    """Reconstruct a frozen spec through ingest, treatment, render, QC, and export."""

    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    spec = ReferenceReconstructionSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    _emit(
        run_reconstruction(
            spec,
            source or Path(lock["source_image"]["path"]),
            output,
            settings=settings,
        )
    )


@eval_app.command("reference-reconstruct-pass-v2")
def eval_reference_reconstruct_pass_v2(
    spec_path: Annotated[
        Path, typer.Option("--spec", help="Frozen Reference Reconstruction Spec v2")
    ],
    output: Annotated[Path, typer.Option("--output", help="New reconstruction pass receipt root")],
    pass_kind: Annotated[
        str,
        typer.Option(
            "--pass",
            help=(
                "timing_easing, continuous_tangents, moving_pivot_hypothesis, "
                "coupled_scale_rotation, or phase_compensated_matrix"
            ),
        ),
    ],
    comparison: Annotated[
        Path | None,
        typer.Option(
            "--comparison",
            help="Prior Comparison v2 report required by phase_compensated_matrix",
        ),
    ] = None,
    source: Annotated[
        Path | None, typer.Option("--source", help="Permitted visual source override")
    ] = None,
) -> None:
    """Execute one bounded v2 pass through Treatment, approval, render, QC, and export."""

    permitted = {
        "timing_easing",
        "continuous_tangents",
        "moving_pivot_hypothesis",
        "coupled_scale_rotation",
        "phase_compensated_matrix",
    }
    if pass_kind not in permitted:
        raise typer.BadParameter(f"unknown reconstruction pass: {pass_kind}")
    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    _emit(
        run_reconstruction_pass_v2(
            spec_path,
            source or Path(lock["source_image"]["path"]),
            output,
            pass_kind=cast(ReferenceReconstructionPassKindV2, pass_kind),
            comparison_path=comparison,
            settings=settings,
        )
    )


@eval_app.command("reference-summarize-study-v2")
def eval_reference_summarize_study_v2(
    plan_path: Annotated[
        Path,
        typer.Option(
            "--plan",
            help="Human-authored study plan joining immutable pass evidence",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New immutable Reconstruction Study v2 report"),
    ],
) -> None:
    """Verify bounded pass evidence and record one explicit selection ledger."""

    repo = Path(__file__).resolve().parents[2]
    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    status = _git(repo, ["status", "--porcelain"]).strip()
    _emit(
        build_reconstruction_study_v2(
            plan_path,
            output,
            implementation_revision=revision,
            git_clean=not bool(status),
        )
    )


@eval_app.command("reference-compare")
def eval_reference_compare(
    spec_path: Annotated[Path, typer.Option("--spec", help="Frozen reference spec")],
    reconstruction_path: Annotated[
        Path, typer.Option("--reconstruction", help="Passing reconstruction receipt")
    ],
    output: Annotated[Path, typer.Option("--output", help="New comparison report root")],
    sample_step: Annotated[
        int, typer.Option("--sample-step", min=1, help="Frames between motion samples")
    ] = 5,
    visual_review: Annotated[
        str | None, typer.Option("--visual-review", help="Recorded contact-sheet review")
    ] = None,
) -> None:
    """Compare a normal render with sealed media without admitting it to lineage."""

    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    spec = ReferenceReconstructionSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if reconstruction.get("overall") != "pass":
        raise typer.BadParameter("reconstruction receipt is not passing")
    if reconstruction.get("source_sha256") != spec.source_sha256:
        raise typer.BadParameter("reconstruction source does not match the spec")
    if reconstruction.get("reference_sha256") != spec.reference_sha256:
        raise typer.BadParameter("reconstruction reference seal does not match the spec")
    final = reconstruction.get("profiles", {}).get("final", {})
    _emit(
        run_comparison(
            spec,
            Path(lock["source_image"]["path"]),
            Path(final["path"]),
            Path(lock["reference_video"]["path"]),
            output,
            lineage_path=Path(final["lineage_path"]),
            sample_step=sample_step,
            visual_review=visual_review,
        )
    )


@eval_app.command("technique-extract")
def eval_technique_extract(
    spec_path: Annotated[Path, typer.Option("--spec", help="Frozen reference spec")],
    comparison_path: Annotated[
        Path, typer.Option("--comparison", help="Passing comparison report")
    ],
    output: Annotated[Path, typer.Option("--output", help="New candidate packet path")],
) -> None:
    """Extract a source-neutral candidate packet from passing evaluation evidence."""

    require_readiness(Settings.from_env())
    spec = ReferenceReconstructionSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    comparison = ReferenceComparisonReport.model_validate_json(
        comparison_path.read_text(encoding="utf-8")
    )
    packet = extract_contain_reveal_technique(spec, comparison)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(packet.model_dump_json(indent=2))
        stream.write("\n")
    _emit(packet)


@eval_app.command("clean-rerun")
def eval_clean_rerun(
    spec_path: Annotated[Path, typer.Option("--spec", help="Frozen reference spec")],
    baseline_path: Annotated[
        Path, typer.Option("--baseline", help="Passing reconstruction receipt")
    ],
    output: Annotated[Path, typer.Option("--output", help="New empty clean-rerun root")],
) -> None:
    """Replay the approved semantics in an empty project store without old renders."""

    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    spec = ReferenceReconstructionSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    report = run_clean_rerun(
        spec,
        Path(lock["source_image"]["path"]),
        baseline_path,
        output,
        runtime_settings=settings,
    )
    _emit(report)
    if report.overall != "pass":
        raise typer.Exit(1)


@eval_app.command("clean-rerun-v2")
def eval_clean_rerun_v2(
    spec_path: Annotated[
        Path,
        typer.Option("--spec", help="Frozen Reference Reconstruction Spec v2"),
    ],
    baseline_path: Annotated[
        Path,
        typer.Option(
            "--baseline",
            help="Passing selected phase-compensated reconstruction receipt",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New absent clean-rerun v2 root"),
    ],
) -> None:
    """Replay approved pass-5r semantics in a new home without old media."""

    settings = Settings.from_env()
    readiness = require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads(
        (repo / "evals" / "reference.lock.json").read_text(encoding="utf-8")
    )
    ReferenceReconstructionSpecV2.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    report = run_clean_rerun_v2(
        spec_path,
        Path(lock["source_image"]["path"]),
        baseline_path,
        output,
        readiness_receipt=readiness,
        runtime_settings=settings,
    )
    _emit(report)
    if report.overall != "pass":
        raise typer.Exit(1)


@eval_app.command("technique-transfer")
def eval_technique_transfer(
    packet_path: Annotated[Path, typer.Option("--packet", help="Candidate technique packet")],
    output: Annotated[Path, typer.Option("--output", help="New empty transfer root")],
    source: Annotated[
        Path | None, typer.Option("--source", help="Independent source; generated if omitted")
    ] = None,
    width: Annotated[int, typer.Option("--width", min=2)] = 720,
    height: Annotated[int, typer.Option("--height", min=2)] = 1280,
) -> None:
    """Apply a candidate packet to unrelated procedural media through normal services."""

    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    selected_source = source or output.with_name(f"{output.name}-procedural-source.png")
    if source is None:
        create_transfer_fixture(selected_source)
    report = run_technique_transfer(
        load_packet(packet_path),
        selected_source,
        output,
        runtime_settings=settings,
        canvas_width=width,
        canvas_height=height,
        forbidden_hashes=[
            lock["source_image"]["sha256"],
            lock["reference_video"]["sha256"],
        ],
    )
    _emit(report)
    if report.overall != "pass":
        raise typer.Exit(1)


@eval_app.command("technique-transfer-corpus-v2")
def eval_technique_transfer_corpus_v2(
    packet_path: Annotated[
        Path,
        typer.Option("--packet", help="Checked-in revision-two candidate packet"),
    ],
    spec_path: Annotated[
        Path,
        typer.Option("--spec", help="Frozen Reference Reconstruction Spec v2"),
    ],
    selected_pass_path: Annotated[
        Path,
        typer.Option("--selected-pass", help="Passing selected 5r reconstruction receipt"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New empty heterogeneous transfer root"),
    ],
    photograph: Annotated[
        Path,
        typer.Option("--photograph", help="Licensed independent photograph"),
    ] = Path("/usr/share/pixmaps/faces/mountain.jpg"),
    width: Annotated[int, typer.Option("--width", min=2)] = 270,
    height: Annotated[int, typer.Option("--height", min=2)] = 480,
) -> None:
    """Upgrade approved semantics and gate them across seven independent classes."""

    settings = Settings.from_env()
    require_readiness(settings)
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "evals" / "reference.lock.json").read_text(encoding="utf-8"))
    spec = ReferenceReconstructionSpecV2.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    selected_pass = ReferenceReconstructionPassReceiptV2.model_validate_json(
        selected_pass_path.read_text(encoding="utf-8")
    )
    upgraded = upgrade_contain_reveal_technique_v2(
        load_packet(packet_path),
        spec,
        selected_pass,
    )
    report = run_technique_transfer_corpus_v2(
        upgraded,
        output,
        photograph_path=photograph,
        photograph_provenance=system_photograph_provenance(photograph),
        runtime_settings=settings,
        canvas_width=width,
        canvas_height=height,
        forbidden_hashes=[
            lock["source_image"]["sha256"],
            lock["reference_video"]["sha256"],
        ],
    )
    _emit(report)
    if report.overall != "pass":
        raise typer.Exit(1)


@eval_app.command("provider-aliases")
def eval_provider_aliases(
    output: Annotated[
        Path,
        typer.Option("--output", help="New immutable provider-alias evaluation root"),
    ],
) -> None:
    """Gate declarations, bindings, failures, fallback, privacy, and authority."""

    repo = Path(__file__).resolve().parents[2]
    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    status = _git(repo, ["status", "--porcelain"]).strip()
    _emit(
        run_provider_alias_eval(
            output,
            repository=repo,
            implementation_revision=revision,
            git_clean=not bool(status),
        )
    )


@eval_app.command("local-ai-live")
def eval_local_ai_live(
    output: Annotated[
        Path,
        typer.Option("--output", help="New immutable live local-AI evaluation root"),
    ],
    model_owner_root: Annotated[
        Path,
        typer.Option(
            "--model-owner-root",
            help="Owner root that must contain the selected live model",
        ),
    ],
    upstream_revision_file: Annotated[
        Path,
        typer.Option(
            "--upstream-revision-file",
            help="Local owner ref containing the exact upstream model revision",
        ),
    ],
    upstream_metadata_file: Annotated[
        Path,
        typer.Option(
            "--upstream-metadata-file",
            help="Captured official model metadata JSON with revision and license",
        ),
    ],
    stack_source_registration: Annotated[
        Path,
        typer.Option(
            "--stack-source-registration",
            help="Source-owned stack service registration",
        ),
    ],
    stack_deployed_registration: Annotated[
        Path,
        typer.Option(
            "--stack-deployed-registration",
            help="Deployed stack service registration",
        ),
    ],
    stack_managed_units: Annotated[
        Path,
        typer.Option(
            "--stack-managed-units",
            help="Stack-managed user-unit allowlist",
        ),
    ],
    fixture_text: Annotated[
        str,
        typer.Option(
            "--fixture-text",
            help="Known synthetic Russian text; never reference or user media",
        ),
    ] = DEFAULT_FIXTURE_TEXT,
) -> None:
    """Prove one existing host model through the alias and no-AI baseline."""

    repo = Path(__file__).resolve().parents[2]
    revision = _git(repo, ["rev-parse", "HEAD"]).strip()
    status = _git(repo, ["status", "--porcelain"]).strip()
    _emit(
        run_local_ai_live_eval(
            output,
            settings=Settings.from_env(),
            repository=repo,
            implementation_revision=revision,
            git_clean=not bool(status),
            expected_model_owner_root=model_owner_root,
            upstream_revision_file=upstream_revision_file,
            upstream_metadata_file=upstream_metadata_file,
            stack_source_registration=stack_source_registration,
            stack_deployed_registration=stack_deployed_registration,
            stack_managed_units=stack_managed_units,
            fixture_text=fixture_text,
        )
    )


@eval_app.command("completion-audit")
def eval_completion_audit(
    readiness: Annotated[Path, typer.Option("--readiness", help="Passing readiness receipt")],
    storage: Annotated[
        Path,
        typer.Option("--storage", help="Passing product-home relocation receipt"),
    ],
    motion_gate: Annotated[
        Path,
        typer.Option("--motion-gate", help="Current passing motion-recovery gate"),
    ],
    motion_evidence: Annotated[
        Path,
        typer.Option("--motion-evidence", help="All-frame Reference Motion Evidence v2"),
    ],
    spec: Annotated[
        Path,
        typer.Option("--spec", help="Frozen Reference Reconstruction Spec v2"),
    ],
    reconstruction: Annotated[
        Path,
        typer.Option("--reconstruction", help="Selected passing reconstruction v2 receipt"),
    ],
    comparison: Annotated[
        Path,
        typer.Option("--comparison", help="Reviewed passing Comparison v2 report"),
    ],
    reference_ui: Annotated[
        Path,
        typer.Option("--reference-ui", help="Real Chromium reference-workspace receipt"),
    ],
    clean_rerun: Annotated[
        Path,
        typer.Option("--clean-rerun", help="Passing clean-rerun v2 report"),
    ],
    transfer: Annotated[
        Path,
        typer.Option("--transfer", help="Passing seven-case transfer corpus v2"),
    ],
    candidate: Annotated[
        Path, typer.Option("--candidate", help="Transferred candidate technique packet")
    ],
    provider_alias: Annotated[
        Path,
        typer.Option("--provider-alias", help="Current passing provider-alias eval"),
    ],
    local_ai: Annotated[
        Path,
        typer.Option("--local-ai", help="Current passing live local-AI admission"),
    ],
    goal_start_revision: Annotated[
        str,
        typer.Option(
            "--goal-start-revision",
            help="Pre-existing revision after which publication is forbidden",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", help="New completion-audit root")],
    private_history_ref: Annotated[
        str,
        typer.Option(
            "--private-history-ref",
            help="Local pre-publication history whose tip tree equals the public root",
        ),
    ] = "local/private-history-pre-publication-20260721",
    local_tag: Annotated[
        str | None,
        typer.Option("--local-tag", help="Local-only final tag that must point at HEAD"),
    ] = None,
    allow_preexisting_remote: Annotated[
        bool,
        typer.Option(
            "--allow-preexisting-remote",
            help="Accept the evolved repo only if no goal commit or tag is published",
        ),
    ] = False,
) -> None:
    """Join every v2 proof lane and fail closed on human, tag, or publication debt."""

    repo = Path(__file__).resolve().parents[2]
    report = run_completion_audit(
        repo=repo,
        lock_path=repo / "evals" / "reference.lock.json",
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
        output_root=output,
        private_history_ref=private_history_ref,
        goal_start_revision=goal_start_revision,
        local_tag=local_tag,
        allow_preexisting_remote=allow_preexisting_remote,
    )
    _emit(report)
    if report.overall != "pass":
        raise typer.Exit(1)


@gate_app.command("readiness")
def gate_readiness() -> None:
    """Prove generic readiness without decoding the sealed reference."""

    receipt = run_readiness_gate()
    _emit(receipt)
    if receipt["overall"] != "pass":
        raise typer.Exit(1)


@gate_app.command("motion-recovery")
def gate_motion_recovery(
    output: Annotated[
        Path | None,
        typer.Option("--output", help="New generic motion-recovery run root"),
    ] = None,
) -> None:
    """Recover every independent ground-truth fixture before reference analysis v2."""

    settings = Settings.from_env()
    selected = output or (
        settings.evals_root / "motion-recovery" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    report = run_motion_recovery_gate(selected, require_clean_revision=True)
    _emit(report)
    if report.overall != "pass":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
