from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aoa_editing.agent import AgentProtocol, AgentRequest
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    AICapabilityDeclaration,
    EvidenceAuthority,
    EvidenceRecord,
    Intent,
    LocalProviderBinding,
    LocalProviderHealthBinding,
    Provenance,
    ProviderHealthEvidence,
    Scenario,
)
from aoa_editing.evals.provider_aliases import run_provider_alias_eval
from aoa_editing.infrastructure.probes import doctor_report
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.abyss_machine_stt import (
    normalize_host_health,
    normalize_host_transcript,
)
from aoa_editing.providers.catalog import (
    ProviderConfigurationError,
    load_capability_catalog,
    load_local_bindings,
)
from aoa_editing.providers.service import (
    ProviderExecution,
    ProviderExecutor,
    ProviderService,
    ProviderTimeout,
)

ROOT = Path(__file__).resolve().parents[1]
DESCRIBE_ALIAS = "local-ai://vision/describe/default"
REGIONS_ALIAS = "local-ai://vision/regions/default"


@dataclass
class ScriptedExecutor(ProviderExecutor):
    health_result: ProviderHealthEvidence
    execution: ProviderExecution | Exception
    health_calls: int = 0
    invocation_calls: int = 0

    def health(
        self,
        _binding: LocalProviderBinding,
        _declaration: AICapabilityDeclaration,
    ) -> ProviderHealthEvidence:
        self.health_calls += 1
        return self.health_result

    def invoke(
        self,
        _binding: LocalProviderBinding,
        _payload: dict[str, Any],
    ) -> ProviderExecution:
        self.invocation_calls += 1
        if isinstance(self.execution, Exception):
            raise self.execution
        return self.execution


def _health(
    status: str = "healthy",
    *,
    observed_at: datetime | None = None,
) -> ProviderHealthEvidence:
    return ProviderHealthEvidence(
        status=status,  # type: ignore[arg-type]
        checked_at=datetime.now(UTC),
        observed_at=observed_at,
        latency_milliseconds=1.0,
        source="injected-test",
        response_sha256="a" * 64,
        reported_status="ready",
        protocol_version="aoa-local-ai-v1",
        backend="fixture-backend",
        model_id="fixture-model",
        model_revision="fixture-r1",
        summary=f"fixture health is {status}",
    )


def _binding(
    alias: str = DESCRIBE_ALIAS,
    *,
    data_boundary: str = "local-host",
) -> LocalProviderBinding:
    return LocalProviderBinding(
        alias=alias,
        state="enabled",
        resolved_owner="abyss-machine",
        adapter_kind="abyss-machine-cli",
        backend="fixture-backend",
        model_id="fixture-model",
        model_revision="fixture-r1",
        model_hash_authority="fixture inventory",
        license_authority="fixture model card",
        metadata_evidence="fixture://provider-inventory",
        command=["fixture-provider", "{request_json}"],
        health=LocalProviderHealthBinding(command=["fixture-health"]),
        data_boundary=data_boundary,  # type: ignore[arg-type]
    )


def _service(
    tmp_path: Path,
    executor: ScriptedExecutor,
    *,
    binding: LocalProviderBinding | None = None,
    fallbacks: dict[str, Any] | None = None,
) -> ProviderService:
    selected = binding or _binding()
    return ProviderService(
        Settings.for_home(tmp_path / "editing-home"),
        bindings={selected.alias: selected},
        executor=executor,
        fallbacks=fallbacks,
    )


def _describe_request() -> dict[str, Any]:
    return {
        "media_path": "/private/runtime/source.png",
        "sample_times_seconds": [0.0],
        "prompt": "describe the composition",
    }


def _describe_output(*, partial: bool = False) -> dict[str, Any]:
    return {
        "descriptions": [
            {
                "time_seconds": 0.0,
                "text": "A geometric fixture.",
                "confidence": 0.9,
            }
        ],
        "partial": partial,
    }


def test_tracked_catalog_declares_complete_path_free_optional_surface() -> None:
    catalog = load_capability_catalog()
    capabilities = {item.capability_id for item in catalog.declarations}
    serialized = (ROOT / "manifests" / "ai-capabilities.json").read_text(
        encoding="utf-8"
    )

    assert capabilities == {
        "speech.transcript",
        "speech.vad",
        "speech.diarization",
        "vision.describe",
        "vision.regions",
        "vision.segment",
        "vision.embed",
        "editorial.plan",
        "editorial.critique",
    }
    assert all(item.alias.startswith("local-ai://") for item in catalog.declarations)
    assert all(item.required is False for item in catalog.declarations)
    assert catalog.baseline_requires_ai is False
    assert catalog.filesystem_links_allowed is False
    assert catalog.physical_model_paths_allowed is False
    assert "/srv/" not in serialized
    assert "/home/" not in serialized


def test_untracked_overlay_loads_disabled_binding_and_rejects_unknown_alias(
    tmp_path: Path,
) -> None:
    settings = Settings.for_home(tmp_path / "home")
    settings.editing_home.mkdir(parents=True)
    settings.local_overlay.write_text(
        "\n".join(
            [
                f'[ai.bindings."{DESCRIBE_ALIAS}"]',
                'state = "disabled"',
                'adapter_kind = "disabled"',
                'reason = "operator disabled semantic vision"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    bindings = load_local_bindings(settings)

    assert bindings[DESCRIBE_ALIAS].state == "disabled"
    assert bindings[DESCRIBE_ALIAS].command == []
    settings.local_overlay.write_text(
        "\n".join(
            [
                '[ai.bindings."local-ai://vision/unknown/default"]',
                'state = "unavailable"',
                'adapter_kind = "unavailable"',
                'reason = "not declared"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProviderConfigurationError, match="undeclared alias"):
        load_local_bindings(settings)


def test_checked_in_overlay_example_is_typed_but_not_a_live_claim(
    tmp_path: Path,
) -> None:
    settings = Settings.for_home(tmp_path / "home")
    settings.editing_home.mkdir(parents=True)
    example = (ROOT / ".aoa-editing.local.example.toml").read_text(encoding="utf-8")
    settings.local_overlay.write_text(example, encoding="utf-8")

    bindings = load_local_bindings(settings)

    assert bindings["local-ai://speech/transcript/default"].adapter_kind == (
        "abyss-machine-cli"
    )
    assert bindings["local-ai://vision/describe/default"].resolved_owner == "abyss-stack"
    assert bindings["local-ai://editorial/plan/default"].adapter_kind == "localhost-http"
    assert bindings["local-ai://speech/diarization/default"].state == "unavailable"
    assert bindings["local-ai://vision/segment/default"].state == "disabled"
    assert all(binding.model_id != "/srv/model" for binding in bindings.values())


def test_available_provider_writes_valid_redacted_evidence_receipt(
    tmp_path: Path,
) -> None:
    executor = ScriptedExecutor(_health(), ProviderExecution(_describe_output()))
    service = _service(tmp_path, executor)
    request = _describe_request()
    request["prompt"] = "do not persist Bearer fixture-secret-value"

    result = service.invoke(DESCRIBE_ALIAS, request, allow_fallback=False)
    receipt_path = service.settings.editing_home / result.receipt_path
    persisted = receipt_path.read_text(encoding="utf-8")

    assert result.receipt.outcome == "succeeded"
    assert result.receipt.output_authority == "evidence"
    assert result.receipt.resolved_owner == "abyss-machine"
    assert result.receipt.health_evidence is not None
    assert result.receipt.health_evidence.status == "healthy"
    assert result.output == _describe_output()
    assert executor.health_calls == 1
    assert executor.invocation_calls == 1
    assert "/private/runtime/source.png" not in persisted
    assert "Bearer fixture-secret-value" not in persisted
    assert "<local-media>" in persisted
    assert "<redacted>" in persisted


def test_missing_stale_version_timeout_and_malformed_are_typed(
    tmp_path: Path,
) -> None:
    settings = Settings.for_home(tmp_path / "home")
    missing = ProviderService(settings, bindings={}).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )
    assert missing.receipt.failure_code == "provider_missing"

    stale_executor = ScriptedExecutor(
        _health("stale", observed_at=datetime.now(UTC) - timedelta(hours=1)),
        ProviderExecution(_describe_output()),
    )
    stale = _service(tmp_path / "stale", stale_executor).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )
    assert stale.receipt.failure_code == "provider_stale"
    assert stale_executor.invocation_calls == 0

    mismatch_executor = ScriptedExecutor(
        _health("version-mismatch"),
        ProviderExecution(_describe_output()),
    )
    mismatch = _service(tmp_path / "mismatch", mismatch_executor).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )
    assert mismatch.receipt.failure_code == "version_mismatch"

    timeout_executor = ScriptedExecutor(
        _health(),
        ProviderTimeout("fixture timeout"),
    )
    timeout = _service(tmp_path / "timeout", timeout_executor).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )
    assert timeout.receipt.failure_code == "provider_timeout"

    malformed_executor = ScriptedExecutor(
        _health(),
        ProviderExecution({"unexpected": True}),
    )
    malformed = _service(tmp_path / "malformed", malformed_executor).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )
    assert malformed.receipt.failure_code == "malformed_response"


def test_privacy_denial_prevents_health_and_invocation(tmp_path: Path) -> None:
    executor = ScriptedExecutor(_health(), ProviderExecution(_describe_output()))
    service = _service(
        tmp_path,
        executor,
        binding=_binding(data_boundary="external"),
    )

    result = service.invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        privacy_mode="local-only",
        explicit_opt_in=False,
        allow_fallback=False,
    )

    assert result.receipt.outcome == "refused"
    assert result.receipt.failure_code == "privacy_denied"
    assert result.receipt.privacy_decision.decision == "denied"
    assert executor.health_calls == 0
    assert executor.invocation_calls == 0


def test_deterministic_fallback_and_partial_evidence_remain_noncanonical(
    tmp_path: Path,
) -> None:
    catalog = load_capability_catalog()
    regions = next(
        item for item in catalog.declarations if item.alias == REGIONS_ALIAS
    )
    fallback = ProviderService(
        Settings.for_home(tmp_path / "fallback"),
        catalog=catalog,
        bindings={},
        fallbacks={
            regions.fallback.strategy_id: lambda _payload: {
                "regions": [
                    {
                        "label": "nonsemantic-saliency",
                        "box_normalized": [0.1, 0.1, 0.8, 0.8],
                        "confidence": 0.5,
                    }
                ],
                "partial": False,
            }
        },
    ).invoke(
        REGIONS_ALIAS,
        {"media_path": "/private/source.png", "time_seconds": None, "labels": []},
    )
    assert fallback.receipt.outcome == "succeeded"
    assert fallback.receipt.adapter_kind == "deterministic-fallback"
    assert fallback.receipt.fallback_status == "used"
    assert fallback.receipt.output_authority == "evidence"

    partial_executor = ScriptedExecutor(
        _health(),
        ProviderExecution(_describe_output(partial=True), partial=True),
    )
    partial = _service(tmp_path / "partial", partial_executor).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )
    assert partial.receipt.outcome == "partial"
    assert partial.receipt.partial is True
    assert partial.receipt.output_authority == "evidence"


def test_provider_evidence_is_superseded_by_explicit_human_correction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (96, 96), "#224466").save(source)
    settings = Settings.for_home(tmp_path / "editing-home")
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "AI evidence boundary",
        Intent(text="Describe without granting authority", scenario=Scenario.STILL_MOTION),
    )
    asset, _probe = editing.ingest(project.id, source)
    executor = ScriptedExecutor(_health(), ProviderExecution(_describe_output()))
    result = ProviderService(
        settings,
        bindings={DESCRIBE_ALIAS: _binding()},
        executor=executor,
    ).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
        project_id=project.id,
        asset_id=asset.id,
    )
    evidence = EvidenceRecord(
        project_id=project.id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="vision.describe",
        payload=result.output or {},
        provenance=Provenance(
            tool="fixture-provider-alias",
            tool_version="1",
            parameters={
                "provider_receipt_id": result.receipt.id,
                "provider_receipt_path": result.receipt_path,
                "output_authority": result.receipt.output_authority,
            },
            model_id=result.receipt.model_id,
            model_revision=result.receipt.model_revision,
            deterministic=False,
        ),
    )
    store.save_evidence(evidence)
    correction = editing.correct_evidence(
        project.id,
        asset_id=asset.id,
        kind="vision.describe",
        payload={"descriptions": [{"text": "Human-reviewed description."}]},
        supersedes=[evidence.id],
        rationale="A human reviewed the actual source.",
    )

    assert evidence.authority is EvidenceAuthority.ANALYZER
    assert correction.authority is EvidenceAuthority.HUMAN_CORRECTION
    effective_ids = {item.id for item in store.list_effective_evidence(project.id)}
    assert correction.id in effective_ids
    assert evidence.id not in effective_ids
    assert store.load_project(project.id).current_version_id is None


def test_doctor_exposes_declarations_and_untracked_binding_state(
    tmp_path: Path,
) -> None:
    settings = Settings.for_home(tmp_path / "home")
    settings.editing_home.mkdir(parents=True)
    settings.local_overlay.write_text(
        "\n".join(
            [
                f'[ai.bindings."{DESCRIBE_ALIAS}"]',
                'state = "disabled"',
                'adapter_kind = "disabled"',
                'reason = "no semantic vision provider installed"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = doctor_report(settings, create_roots=True)
    by_alias = {
        item["alias"]: item
        for item in report["providers"]["bindings"]
    }

    assert report["providers"]["contract_ok"] is True
    assert report["providers"]["declaration_count"] == 9
    assert by_alias[DESCRIBE_ALIAS]["binding_state"] == "disabled"
    assert by_alias[DESCRIBE_ALIAS]["health"] is None
    assert report["providers"]["baseline_requires_ai"] is False


def test_agent_uses_the_shared_provider_service_without_expanding_authority(
    tmp_path: Path,
) -> None:
    agent = AgentProtocol(Settings.for_home(tmp_path / "home"))

    status = agent.execute(
        AgentRequest(id=1, method="provider.status", params={"probe": False})
    )
    missing = agent.execute(
        AgentRequest(
            id=2,
            method="provider.invoke",
            params={
                "alias": DESCRIBE_ALIAS,
                "payload": _describe_request(),
                "allow_fallback": False,
            },
        )
    )

    assert status["ok"] is True
    assert status["result"]["declaration_count"] == 9
    assert missing["ok"] is True
    assert missing["result"]["receipt"]["failure_code"] == "provider_missing"
    assert missing["result"]["receipt"]["output_authority"] == "evidence"


def test_default_transcript_analysis_resolves_alias_and_localizes_missing_provider(
    tmp_path: Path,
) -> None:
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(
            b"".join(
                int(8000 * math.sin(2 * math.pi * 440 * index / 8000)).to_bytes(
                    2,
                    byteorder="little",
                    signed=True,
                )
                for index in range(8000)
            )
        )
    settings = Settings.for_home(tmp_path / "home")
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "Missing transcript alias",
        Intent(text="Keep baseline evidence", scenario=Scenario.SPEECH_CLEAN),
    )
    asset, _probe = editing.ingest(project.id, source)

    records = AnalysisService(store).analyze(
        project.id,
        asset.id,
        transcribe=True,
    )
    failure = next(
        item
        for item in records
        if item.kind == "analysis.failure"
        and item.payload["failed_kind"] == "speech.transcript"
    )
    receipt_paths = sorted(
        (settings.state_root / "provider-receipts").glob("providerreceipt_*.json")
    )
    receipt = json.loads(receipt_paths[-1].read_text(encoding="utf-8"))

    assert failure.payload["localized"] is True
    assert receipt["requested_alias"] == "local-ai://speech/transcript/default"
    assert receipt["failure_code"] == "provider_missing"
    assert any(item.kind == "audio.silence" for item in records)
    assert any(item.kind == "audio.loudness" for item in records)


def test_abyss_machine_stt_adapter_normalizes_path_free_health_and_revision() -> None:
    health = normalize_host_health(
        {
            "schema": "abyss_machine_ai_capabilities_v1",
            "ok": True,
            "version": "fixture-host",
            "generated_at": "2026-07-21T12:00:00Z",
            "capabilities": {
                "stt": {
                    "status": "ready",
                    "host_recommended_backend": "OpenVINO Whisper AUTO:GPU,CPU",
                }
            },
        },
        {
            "schema": "abyss_machine_dictation_status_v1",
            "server_socket_exists": True,
            "profiles": {
                "quality": {
                    "model_id": "openai/whisper-large-v3-turbo",
                    "enabled": True,
                    "model_dir_exists": True,
                }
            },
        },
        {
            "schema": "abyss_machine_ai_models_v1",
            "entries": [
                {
                    "kind": "directory",
                    "category": "stt_whisper_openvino",
                    "name": "whisper-large-v3-turbo",
                    "path": "/private/owner/path/must-not-leak",
                    "relative_path": "stt/whisper/openvino/whisper-large-v3-turbo",
                    "artifacts": {"xml": ["encoder.xml", "decoder.xml"]},
                    "file_summary": {"immediate_files": 14},
                    "read_only_source": True,
                }
            ],
        },
    )

    assert health["status"] == "ready"
    assert health["protocol_version"] == "aoa-local-ai-v1"
    assert health["model_id"] == "openai/whisper-large-v3-turbo"
    assert len(health["model_revision"]) == 64
    assert "/private/owner/path" not in json.dumps(health)


def test_abyss_machine_stt_adapter_marks_file_level_transcript_partial() -> None:
    transcript = normalize_host_transcript(
        {
            "schema": "abyss_machine_dictation_transcript_v1",
            "audio": "/private/source/voiceover.wav",
            "model_dir": "/private/model/root",
            "language": "ru",
            "profile_name": "quality",
            "via": "server",
            "text": "Показываем задачу и затем результат.",
            "raw_audio_duration_sec": 4.25,
            "segments": [
                {"start_sec": 0.2, "end_sec": 1.5},
                {"start_sec": 2.0, "end_sec": 4.0},
            ],
        }
    )

    assert transcript["partial"] is True
    assert transcript["temporal_precision"] == "file"
    assert transcript["segments"] == [
        {
            "start": 0.0,
            "end": 4.25,
            "text": "Показываем задачу и затем результат.",
            "speaker": None,
        }
    ]
    assert transcript["host_segment_timing_available"] is True
    assert transcript["host_segment_text_available"] is False
    assert "/private/" not in json.dumps(transcript)


def test_domain_core_has_no_physical_model_path_contract() -> None:
    domain = (ROOT / "src" / "aoa_editing" / "domain" / "models.py").read_text(
        encoding="utf-8"
    )

    assert "model_path:" not in domain
    assert "/legacy-host-root" not in domain
    assert "/home/" not in domain


def test_provider_alias_eval_covers_failures_fallback_and_authority(
    tmp_path: Path,
) -> None:
    report = run_provider_alias_eval(
        tmp_path / "provider-alias-eval",
        repository=ROOT,
        implementation_revision="a" * 40,
        git_clean=True,
    )

    assert report.overall == "pass"
    assert report.mandatory_skips == 0
    assert report.reference_media_used_as_input is False
    assert report.declaration_count == 9
    assert len(report.cases) == 9
    assert all(item.overall == "pass" for item in report.cases)
    assert all(item.status == "pass" for item in report.checks)
