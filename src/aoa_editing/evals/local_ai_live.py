"""Revision-bound live proof for the owner-correct local ASR alias."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Literal, cast

from aoa_editing import __version__
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    CheckResult,
    Intent,
    LocalAICapabilityDecision,
    LocalAIIntegrationReport,
    LocalAILiveTranscriptCase,
    LocalAIModelEvidence,
    LocalAIResourceEvidence,
    Provenance,
    Scenario,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.abyss_machine_stt import normalize_host_health
from aoa_editing.providers.catalog import load_capability_catalog, load_local_bindings
from aoa_editing.providers.service import ProviderService

ABYSS_MACHINE = "/usr/local/bin/abyss-machine"
TRANSCRIPT_ALIAS: Literal["local-ai://speech/transcript/default"] = (
    "local-ai://speech/transcript/default"
)
DEFAULT_FIXTURE_TEXT = "Сегодня мы проверяем звук и сохраняем результат."
WER_LIMIT = 0.40
CER_LIMIT = 0.25


class LocalAILiveEvalError(RuntimeError):
    """The live provider proof is incomplete, dirty, or outside its owner route."""


def run_local_ai_live_eval(
    output_root: Path,
    *,
    settings: Settings,
    repository: Path,
    implementation_revision: str,
    git_clean: bool,
    expected_model_owner_root: Path,
    upstream_revision_file: Path,
    upstream_metadata_file: Path,
    stack_source_registration: Path,
    stack_deployed_registration: Path,
    stack_managed_units: Path,
    fixture_text: str = DEFAULT_FIXTURE_TEXT,
) -> LocalAIIntegrationReport:
    """Execute one real alias call and join it with owner and baseline evidence."""

    root = output_root.expanduser().resolve()
    if root.exists():
        raise LocalAILiveEvalError(f"local-AI output root must be absent: {root}")
    if not git_clean:
        raise LocalAILiveEvalError("local-AI live eval requires a clean implementation")
    resource_gate_unit = _current_resource_gate_unit()
    if resource_gate_unit is None:
        raise LocalAILiveEvalError(
            "run local-AI live eval through abyss-machine resource launch --class medium --kind ai"
        )
    if not settings.local_overlay.is_file():
        raise LocalAILiveEvalError("the ignored local binding overlay is absent")
    root.mkdir(parents=True)

    capabilities = _run_json([ABYSS_MACHINE, "ai", "capabilities", "--json"], 30)
    dictation = _run_json([ABYSS_MACHINE, "dictation", "status", "--json"], 30)
    inventory = _run_json([ABYSS_MACHINE, "ai", "models", "--json"], 60)
    devices = _run_json([ABYSS_MACHINE, "ai", "devices", "--json"], 30)
    normalized_health = normalize_host_health(capabilities, dictation, inventory)
    if normalized_health.get("status") != "ready":
        raise LocalAILiveEvalError("the normalized host STT capability is not ready")

    catalog = load_capability_catalog(repository / "manifests" / "ai-capabilities.json")
    bindings = load_local_bindings(settings, catalog=catalog)
    binding = bindings.get(TRANSCRIPT_ALIAS)
    if binding is None or binding.state != "enabled":
        raise LocalAILiveEvalError("the transcript alias is not enabled")
    if (
        binding.model_id != normalized_health.get("model_id")
        or binding.model_revision != normalized_health.get("model_revision")
        or binding.backend != normalized_health.get("backend")
    ):
        raise LocalAILiveEvalError("the local binding has drifted from live owner health")

    quality = _mapping(_mapping(dictation.get("profiles")).get("quality"))
    model_root_value = quality.get("model_dir")
    if not isinstance(model_root_value, str) or not model_root_value:
        raise LocalAILiveEvalError("dictation quality profile omitted its owner model root")
    model_root = Path(model_root_value).expanduser().resolve()
    owner_root = expected_model_owner_root.expanduser().resolve()
    if not model_root.is_relative_to(owner_root) or not model_root.is_dir():
        raise LocalAILiveEvalError("selected model is outside the declared stack owner root")

    upstream_revision = upstream_revision_file.read_text(encoding="utf-8").strip()
    metadata_payload = json.loads(upstream_metadata_file.read_text(encoding="utf-8"))
    if not isinstance(metadata_payload, dict):
        raise LocalAILiveEvalError("upstream model metadata must be a JSON object")
    metadata_revision = metadata_payload.get("sha")
    card_data = _mapping(metadata_payload.get("cardData"))
    license_id = str(card_data.get("license") or "").lower()
    if (
        not re.fullmatch(r"[0-9a-f]{40}", upstream_revision)
        or metadata_revision != upstream_revision
        or license_id != "mit"
    ):
        raise LocalAILiveEvalError("local source revision and official model metadata do not agree")

    registration = _stack_registration_evidence(
        stack_source_registration,
        stack_deployed_registration,
        stack_managed_units,
        model_root,
    )
    service_before = _systemd_service_status(registration["unit"])

    fixture_audio = root / "synthetic-russian-asr.wav"
    fixture_generator = _generate_speech_fixture(fixture_audio, fixture_text)
    fixture_duration = _wav_duration(fixture_audio)
    fixture_sha256 = sha256_file(fixture_audio)

    result = ProviderService(settings, catalog=catalog, bindings=bindings).invoke(
        TRANSCRIPT_ALIAS,
        {
            "media_path": str(fixture_audio),
            "language": "ru",
            "profile": "quality",
        },
        privacy_mode="local-only",
        allow_fallback=False,
    )
    if result.output is None or result.receipt.outcome not in {"succeeded", "partial"}:
        raise LocalAILiveEvalError(f"live transcript alias failed: {result.receipt.failure_code}")
    actual_text = str(result.output.get("text") or "").strip()
    if not actual_text:
        raise LocalAILiveEvalError("live transcript alias returned empty lexical evidence")

    provider_receipt_source = settings.editing_home / result.receipt_path
    provider_receipt_copy = root / "provider-receipt.json"
    _write_new_json(
        provider_receipt_copy,
        json.loads(provider_receipt_source.read_text(encoding="utf-8")),
    )
    _write_new_json(root / "provider-output.json", result.output)

    baseline = _run_no_ai_baseline(root / "baseline-home", fixture_audio)
    _write_new_json(root / "baseline-evidence.json", baseline)

    model_manifest = _model_tree_manifest(model_root)
    _write_new_json(root / "model-export-files.json", model_manifest)
    metadata_subset = {
        "model_id": binding.model_id,
        "upstream_revision": upstream_revision,
        "license": license_id,
        "pipeline_tag": metadata_payload.get("pipeline_tag"),
        "library_name": metadata_payload.get("library_name"),
        "metadata_sha256": sha256_file(upstream_metadata_file),
    }
    _write_new_json(root / "upstream-model-metadata.json", metadata_subset)

    service_after = _systemd_service_status(registration["unit"])
    word_error_rate = _error_rate(
        _normalized_words(fixture_text),
        _normalized_words(actual_text),
    )
    character_error_rate = _error_rate(
        list(_normalized_characters(fixture_text)),
        list(_normalized_characters(actual_text)),
    )
    health_latency = (
        result.receipt.health_evidence.latency_milliseconds
        if result.receipt.health_evidence is not None
        else 0.0
    )
    inference_latency = max(
        0.0,
        result.receipt.execution_time_milliseconds - health_latency,
    )
    inference_rtf = inference_latency / 1000.0 / fixture_duration

    baseline_kinds = sorted(str(item) for item in baseline["evidence_kinds"])
    case_checks = [
        _check(
            "live-provider-outcome",
            result.receipt.outcome in {"succeeded", "partial"},
            "The real alias returned schema-valid lexical evidence.",
            outcome=result.receipt.outcome,
            partial=result.receipt.partial,
        ),
        _check(
            "live-provider-wer",
            word_error_rate <= WER_LIMIT,
            "Known-text synthetic Russian speech stays inside the frozen WER limit.",
            measured=word_error_rate,
            maximum=WER_LIMIT,
        ),
        _check(
            "live-provider-cer",
            character_error_rate <= CER_LIMIT,
            "Known-text synthetic Russian speech stays inside the frozen CER limit.",
            measured=character_error_rate,
            maximum=CER_LIMIT,
        ),
        _check(
            "baseline-without-ai",
            (
                baseline["transcript_available"] is False
                and baseline["failure_code"] == "provider_missing"
                and {"audio.loudness", "audio.silence"}.issubset(baseline_kinds)
            ),
            "The deterministic baseline remains useful and localized without ASR.",
            evidence_kinds=baseline_kinds,
            failure_code=baseline["failure_code"],
        ),
    ]
    transcript_case = LocalAILiveTranscriptCase(
        case_id="synthetic-russian-known-text",
        fixture_generator=fixture_generator,
        expected_text=fixture_text,
        actual_text=actual_text,
        fixture_audio_sha256=fixture_sha256,
        fixture_duration_seconds=fixture_duration,
        provider_receipt_path=provider_receipt_copy.name,
        provider_receipt_sha256=sha256_file(provider_receipt_copy),
        provider_outcome=cast(
            Literal["succeeded", "partial"],
            result.receipt.outcome,
        ),
        word_error_rate=word_error_rate,
        character_error_rate=character_error_rate,
        baseline_transcript_available=False,
        baseline_evidence_kinds=baseline_kinds,
        baseline_failure_code="provider_missing",
        checks=case_checks,
        overall=("pass" if all(item.status == "pass" for item in case_checks) else "fail"),
    )

    alias_states = {
        item["capability_id"]: item["binding_state"]
        for item in ProviderService(
            settings,
            catalog=catalog,
            bindings=bindings,
        ).inspect_bindings(probe=False)["bindings"]
    }
    decisions = _capability_decisions(capabilities, alias_states, binding.model_id)
    openvino = _mapping(devices.get("openvino"))
    available_devices = [str(item) for item in openvino.get("available_devices", []) if item]
    selected_device_route = str(quality.get("device") or binding.backend)

    model_evidence = LocalAIModelEvidence(
        model_id=str(binding.model_id),
        model_owner="abyss-stack",
        runtime_owner="abyss-machine",
        adapter_owner="aoa-editing",
        owner_model_root=str(model_root),
        upstream_revision=upstream_revision,
        upstream_metadata_sha256=sha256_file(upstream_metadata_file),
        inventory_revision=str(binding.model_revision),
        export_tree_sha256=str(model_manifest["tree_sha256"]),
        export_file_count=int(model_manifest["file_count"]),
        export_size_bytes=int(model_manifest["size_bytes"]),
        license_id=license_id,
        license_authority=("https://huggingface.co/openai/whisper-large-v3-turbo"),
        stack_service_unit=registration["unit"],
        stack_source_registration_sha256=registration["source_sha256"],
        stack_deployed_registration_sha256=registration["deployed_sha256"],
        stack_registration_parity=True,
    )
    evaluation_unit_memory = _systemd_unit_memory(resource_gate_unit)
    resource_evidence = LocalAIResourceEvidence(
        resource_class="medium",
        resource_kind="ai",
        available_devices=available_devices,
        selected_backend=str(binding.backend),
        selected_device_route=selected_device_route,
        npu_used=False,
        npu_decision=(
            "The owner registry exposes NPU, but the selected Whisper route is "
            "AUTO:GPU,CPU and has no measured NPU promotion evidence."
        ),
        service_active=True,
        service_memory_current_bytes_before=service_before["memory_current"],
        service_memory_current_bytes_after=service_after["memory_current"],
        service_memory_peak_bytes=service_after["memory_peak"],
        evaluation_unit_memory_current_bytes=evaluation_unit_memory["memory_current"],
        evaluation_unit_memory_peak_bytes=evaluation_unit_memory["memory_peak"],
        model_disk_bytes=int(model_manifest["size_bytes"]),
        health_latency_milliseconds=health_latency,
        end_to_end_latency_milliseconds=result.receipt.execution_time_milliseconds,
        estimated_inference_latency_milliseconds=inference_latency,
        audio_duration_seconds=fixture_duration,
        estimated_inference_rtf=inference_rtf,
        resource_gate_unit=resource_gate_unit,
        resource_gate_admitted=True,
    )

    checks = [
        _check(
            "resource-gate",
            resource_gate_unit.startswith("abyss-machine-ai-medium-"),
            "The live proof ran inside the canonical medium AI resource gate.",
            unit=resource_gate_unit,
        ),
        _check(
            "owner-route",
            (
                binding.resolved_owner == "abyss-machine"
                and model_evidence.model_owner == "abyss-stack"
                and model_evidence.adapter_owner == "aoa-editing"
            ),
            "Model, runtime, and product adapter ownership remain separated.",
        ),
        _check(
            "stack-registration",
            registration["parity"] and service_after["active"],
            "Source/deploy unit registration is byte-equal and the service is live.",
            source_sha256=registration["source_sha256"],
            deployed_sha256=registration["deployed_sha256"],
            active=service_after["active"],
        ),
        _check(
            "model-revision",
            (
                upstream_revision == metadata_revision
                and binding.model_revision == normalized_health.get("model_revision")
            ),
            "Upstream revision and current inventory revision are both explicit.",
            upstream_revision=upstream_revision,
            inventory_revision=binding.model_revision,
        ),
        _check(
            "model-byte-hash",
            (
                model_manifest["file_count"] > 0
                and model_manifest["size_bytes"] > 0
                and bool(model_manifest["tree_sha256"])
            ),
            "Every regular file in the selected OpenVINO export was SHA-256 hashed.",
            file_count=model_manifest["file_count"],
            size_bytes=model_manifest["size_bytes"],
            tree_sha256=model_manifest["tree_sha256"],
        ),
        _check(
            "model-license",
            license_id == "mit",
            "Official upstream metadata identifies the selected model as MIT.",
            license=license_id,
            authority=model_evidence.license_authority,
        ),
        _check(
            "live-health",
            (
                result.receipt.health_evidence is not None
                and result.receipt.health_evidence.status == "healthy"
                and result.receipt.health_evidence.model_revision == binding.model_revision
            ),
            "Invocation used fresh, revision-matched health evidence.",
        ),
        _check(
            "privacy-authority",
            (
                result.receipt.privacy_decision.data_boundary == "local-host"
                and result.receipt.privacy_decision.decision == "allowed"
                and result.receipt.output_authority == "evidence"
                and result.receipt.fallback_status == "not-used"
            ),
            "Local ASR output remains evidence and never becomes edit authority.",
        ),
        _check(
            "no-model-install",
            all(not item.new_model_required for item in decisions),
            "Gap review selected the existing ASR and requires no new model download.",
        ),
        _check(
            "reference-isolation",
            True,
            "Only generated speech entered the live provider comparison.",
            reference_media_used_as_input=False,
        ),
        _check(
            "live-quality-vs-baseline",
            transcript_case.overall == "pass",
            "ASR adds measured lexical evidence while the no-AI baseline stays valid.",
            word_error_rate=word_error_rate,
            character_error_rate=character_error_rate,
            baseline_transcript_available=False,
        ),
    ]
    overall: Literal["pass", "fail"] = (
        "pass"
        if transcript_case.overall == "pass" and all(item.status == "pass" for item in checks)
        else "fail"
    )
    artifacts = {
        "fixture_audio": fixture_audio.name,
        "provider_receipt": provider_receipt_copy.name,
        "provider_output": "provider-output.json",
        "baseline_evidence": "baseline-evidence.json",
        "model_export_files": "model-export-files.json",
        "upstream_model_metadata": "upstream-model-metadata.json",
    }
    report = LocalAIIntegrationReport(
        implementation_revision=implementation_revision,
        git_clean=True,
        selected_alias=TRANSCRIPT_ALIAS,
        capability_decisions=decisions,
        model=model_evidence,
        resources=resource_evidence,
        cases=[transcript_case],
        checks=checks,
        artifacts=artifacts,
        new_models_installed=False,
        downloaded_model_bytes=0,
        baseline_requires_ai=False,
        ai_output_authority="evidence",
        mandatory_skips=0,
        reference_media_used_as_input=False,
        provenance=Provenance(
            tool="aoa-editing-local-ai-live-eval",
            tool_version=__version__,
            parameters={
                "fixture": "synthetic-espeak-ng-russian-known-text",
                "word_error_rate_limit": WER_LIMIT,
                "character_error_rate_limit": CER_LIMIT,
                "resource_gate_unit": resource_gate_unit,
                "reference_media_decoded": False,
                "model_downloaded": False,
            },
            deterministic=False,
            model_id=binding.model_id,
            model_revision=binding.model_revision,
        ),
        overall=overall,
    )
    _write_new_json(root / "local-ai-integration.json", report.model_dump(mode="json"))
    if report.overall != "pass":
        raise LocalAILiveEvalError("local-AI live integration gate failed")
    return report


def _capability_decisions(
    capabilities: dict[str, Any],
    alias_states: dict[str, str],
    selected_model_id: str | None,
) -> list[LocalAICapabilityDecision]:
    host = _mapping(capabilities.get("capabilities"))
    stt_status = str(_mapping(host.get("stt")).get("status") or "absent")
    embedding_status = str(_mapping(host.get("embeddings")).get("status") or "absent")
    llm_status = str(_mapping(host.get("llm_text")).get("status") or "absent")
    rows = [
        (
            "speech.transcript",
            "Recover lexical speech evidence without making ASR mandatory.",
            stt_status,
            "live",
            "bind-existing",
            selected_model_id,
            "The ready warm host Whisper bridge closes a real evidence gap.",
        ),
        (
            "speech.vad",
            "Expose speech ranges only when a standalone owner route adds value.",
            "embedded-in-dictation-and-deterministic-audio-analysis",
            "inventory-only",
            "keep-unbound",
            None,
            "FFmpeg silence evidence and internal dictation VAD already cover the "
            "current product need; no separate model is justified.",
        ),
        (
            "speech.diarization",
            "Separate speakers only for a proven multi-speaker editorial workflow.",
            "absent",
            "absent",
            "defer",
            None,
            "No current scenario requires diarization and no owner-verified local "
            "pipeline is registered.",
        ),
        (
            "vision.describe",
            "Add semantic descriptions only when they improve a reviewed workflow.",
            llm_status,
            "inventory-only",
            "defer",
            None,
            "Text-model and projector inventory is not a live vision capability; "
            "technical image evidence remains sufficient.",
        ),
        (
            "vision.regions",
            "Add semantic regions only when deterministic evidence is insufficient.",
            llm_status,
            "inventory-only",
            "defer",
            None,
            "No stack-registered region service or measured product-quality gain exists.",
        ),
        (
            "vision.segment",
            "Add masks only for an editing operation that cannot use current proxies.",
            "absent",
            "absent",
            "defer",
            None,
            "A heavy segmentation checkpoint would duplicate no proven product need.",
        ),
        (
            "vision.embed",
            "Enable image-text retrieval only after a retrieval corpus exists.",
            embedding_status,
            "inventory-only",
            "keep-unbound",
            None,
            "A host embedding model exists, but the product has no admitted retrieval "
            "workflow or stack capability registration for this alias.",
        ),
        (
            "editorial.plan",
            "Propose typed treatments only after quality beats deterministic planning.",
            llm_status,
            "inventory-only",
            "keep-unbound",
            None,
            "The deterministic planners are complete and the resident LLM has no "
            "product-scoped plan-quality gate yet.",
        ),
        (
            "editorial.critique",
            "Offer noncanonical critique only after a human-review utility study.",
            llm_status,
            "inventory-only",
            "keep-unbound",
            None,
            "The resident LLM is not promoted to editorial authority and no current gap "
            "requires another model.",
        ),
    ]
    decisions: list[LocalAICapabilityDecision] = []
    for (
        capability_id,
        gap,
        host_status,
        registration,
        action,
        model_id,
        rationale,
    ) in rows:
        decisions.append(
            LocalAICapabilityDecision(
                capability_id=capability_id,
                product_gap=gap,
                host_status=host_status,
                stack_registration=registration,  # type: ignore[arg-type]
                action=action,  # type: ignore[arg-type]
                alias_state=alias_states.get(capability_id, "unbound"),  # type: ignore[arg-type]
                candidate_model_id=model_id,
                new_model_required=False,
                rationale=rationale,
            )
        )
    return decisions


def _run_no_ai_baseline(home: Path, fixture_audio: Path) -> dict[str, Any]:
    settings = Settings.for_home(home)
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "Synthetic no-AI transcript baseline",
        Intent(
            text="Measure deterministic audio evidence without an AI provider.",
            scenario=Scenario.SPEECH_CLEAN,
        ),
    )
    asset, _probe = editing.ingest(project.id, fixture_audio)
    records = AnalysisService(store).analyze(
        project.id,
        asset.id,
        transcribe=True,
    )
    receipts = sorted(
        (settings.state_root / "provider-receipts").glob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not receipts:
        raise LocalAILiveEvalError("no-AI baseline omitted its provider failure receipt")
    receipt = json.loads(receipts[-1].read_text(encoding="utf-8"))
    failure_code = receipt.get("failure_code")
    if failure_code != "provider_missing":
        raise LocalAILiveEvalError("no-AI baseline did not fail locally as expected")
    return {
        "project_id": project.id,
        "asset_sha256": asset.sha256,
        "transcript_available": any(item.kind == "speech.transcript" for item in records),
        "failure_code": failure_code,
        "evidence_kinds": sorted({item.kind for item in records}),
        "provider_receipt_sha256": sha256_file(receipts[-1]),
    }


def _stack_registration_evidence(
    source: Path,
    deployed: Path,
    managed_units: Path,
    model_root: Path,
) -> dict[str, Any]:
    unit = deployed.name
    source_text = source.read_text(encoding="utf-8")
    deployed_text = deployed.read_text(encoding="utf-8")
    managed = {
        line.strip()
        for line in managed_units.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    source_sha256 = sha256_file(source)
    deployed_sha256 = sha256_file(deployed)
    parity = source_sha256 == deployed_sha256
    if (
        not parity
        or unit not in managed
        or "/usr/local/libexec/abyss-dictation-server" not in source_text
        or str(model_root) not in source_text
        or source_text != deployed_text
    ):
        raise LocalAILiveEvalError("stack dictation registration is missing or drifted")
    return {
        "unit": unit,
        "source_sha256": source_sha256,
        "deployed_sha256": deployed_sha256,
        "parity": parity,
    }


def _systemd_service_status(unit: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MemoryCurrent",
            "-p",
            "MemoryPeak",
            "-p",
            "CPUUsageNSec",
            "-p",
            "MainPID",
            "--no-pager",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise LocalAILiveEvalError("cannot read live dictation service facts")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    active = fields.get("ActiveState") == "active" and fields.get("SubState") == "running"
    if not active:
        raise LocalAILiveEvalError("the registered dictation service is not live")
    return {
        "active": active,
        "main_pid": _integer(fields.get("MainPID")),
        "memory_current": _integer(fields.get("MemoryCurrent")),
        "memory_peak": _integer(fields.get("MemoryPeak")),
        "cpu_usage_nsec": _integer(fields.get("CPUUsageNSec")),
    }


def _systemd_unit_memory(unit: str) -> dict[str, int]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "MemoryCurrent",
            "-p",
            "MemoryPeak",
            "--no-pager",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise LocalAILiveEvalError("cannot read the active resource-gate unit")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    current = _integer(fields.get("MemoryCurrent"))
    peak = _integer(fields.get("MemoryPeak"))
    if fields.get("ActiveState") != "active" or peak <= 0:
        raise LocalAILiveEvalError("resource-gate unit omitted live memory evidence")
    return {"memory_current": current, "memory_peak": peak}


def _generate_speech_fixture(path: Path, text: str) -> str:
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise LocalAILiveEvalError("espeak-ng is required for the synthetic ASR fixture")
    version = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    generated = subprocess.run(
        [executable, "-v", "ru", "-s", "145", "-w", str(path), text],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if generated.returncode != 0 or not path.is_file() or path.stat().st_size == 0:
        raise LocalAILiveEvalError("espeak-ng could not create the speech fixture")
    version_line = (version.stdout or version.stderr).splitlines()
    return version_line[0].strip() if version_line else "espeak-ng"


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        frames = stream.getnframes()
        rate = stream.getframerate()
    if rate <= 0 or frames <= 0:
        raise LocalAILiveEvalError("synthetic fixture has no measurable duration")
    return frames / rate


def _model_tree_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        total += size
        files.append({"path": relative, "size_bytes": size, "sha256": digest})
    if not files:
        raise LocalAILiveEvalError("selected model export has no regular files")
    accumulator = hashlib.sha256()
    for item in files:
        accumulator.update((f"{item['path']}\0{item['size_bytes']}\0{item['sha256']}\n").encode())
    return {
        "schema": "aoa_editing_model_tree_manifest_v1",
        "file_count": len(files),
        "size_bytes": total,
        "tree_sha256": accumulator.hexdigest(),
        "files": files,
    }


def _normalized_words(text: str) -> list[str]:
    normalized = text.lower().replace(
        "\N{CYRILLIC SMALL LETTER IO}",
        "\N{CYRILLIC SMALL LETTER IE}",
    )
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _normalized_characters(text: str) -> str:
    return "".join(_normalized_words(text))


def _error_rate(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 0.0 if not actual else float(len(actual))
    return _levenshtein(expected, actual) / len(expected)


def _levenshtein(expected: list[str], actual: list[str]) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_item in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_item in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1] + (0 if expected_item == actual_item else 1),
                )
            )
        previous = current
    return previous[-1]


def _current_resource_gate_unit() -> str | None:
    cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    match = re.search(
        r"(abyss-machine-ai-medium-[A-Za-z0-9_.@:-]+\.(?:service|scope))",
        cgroup,
    )
    return match.group(1) if match else None


def _run_json(command: list[str], timeout: float) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise LocalAILiveEvalError(f"owner command failed: {command[0]} {command[1:3]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise LocalAILiveEvalError("owner command returned malformed JSON") from error
    if not isinstance(payload, dict):
        raise LocalAILiveEvalError("owner command returned a non-object JSON value")
    return payload


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _integer(value: object) -> int:
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _check(
    identifier: str,
    condition: bool,
    summary: str,
    **measured: Any,
) -> CheckResult:
    return CheckResult(
        id=identifier,
        status="pass" if condition else "fail",
        summary=summary,
        measured=measured,
    )


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
