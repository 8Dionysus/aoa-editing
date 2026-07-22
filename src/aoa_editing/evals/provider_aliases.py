"""Revision-bound contract corpus for the local-AI alias boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from aoa_editing import __version__
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    AICapabilityCatalog,
    AICapabilityDeclaration,
    CheckResult,
    EvidenceRecord,
    Intent,
    LocalProviderBinding,
    LocalProviderHealthBinding,
    Provenance,
    ProviderAliasEvalCase,
    ProviderAliasEvalReport,
    ProviderHealthEvidence,
    ProviderInvocationResult,
    Scenario,
)
from aoa_editing.evals.generic import run_generic_suite
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.probes import doctor_report
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.catalog import load_capability_catalog
from aoa_editing.providers.service import (
    LocalProviderExecutor,
    ProviderExecution,
    ProviderExecutor,
    ProviderService,
    ProviderTimeout,
)

DESCRIBE_ALIAS = "local-ai://vision/describe/default"
REGIONS_ALIAS = "local-ai://vision/regions/default"
SECRET_SENTINEL = "Bearer provider-alias-eval-secret"


class ProviderAliasEvalError(RuntimeError):
    """The alias corpus cannot claim a partial or dirty pass."""


@dataclass
class _ScriptedExecutor(ProviderExecutor):
    health_result: ProviderHealthEvidence
    execution: ProviderExecution | Exception

    def health(
        self,
        _binding: LocalProviderBinding,
        _declaration: AICapabilityDeclaration,
    ) -> ProviderHealthEvidence:
        return self.health_result

    def invoke(
        self,
        _binding: LocalProviderBinding,
        _payload: dict[str, Any],
    ) -> ProviderExecution:
        if isinstance(self.execution, Exception):
            raise self.execution
        return self.execution


def run_provider_alias_eval(
    output_root: Path,
    *,
    repository: Path,
    implementation_revision: str,
    git_clean: bool,
) -> ProviderAliasEvalReport:
    root = output_root.expanduser().resolve()
    if root.exists():
        raise ProviderAliasEvalError(f"provider alias output root must be absent: {root}")
    if not git_clean:
        raise ProviderAliasEvalError("provider alias eval requires a clean implementation")
    root.mkdir(parents=True)
    catalog_path = repository / "manifests" / "ai-capabilities.json"
    catalog = load_capability_catalog(catalog_path)
    settings = Settings.for_home(root / "workspace")
    fixture_script = _write_fixture_provider(root / "fixture-provider.py")
    live_binding = _binding(
        command=[
            sys.executable,
            str(fixture_script),
            "invoke",
            "{request_json}",
        ],
        health_command=[sys.executable, str(fixture_script), "health"],
    )
    cases: list[ProviderAliasEvalCase] = []
    results: dict[str, ProviderInvocationResult] = {}

    available = ProviderService(
        settings,
        catalog=catalog,
        bindings={DESCRIBE_ALIAS: live_binding},
        executor=LocalProviderExecutor(),
    ).invoke(
        DESCRIBE_ALIAS,
        _describe_request(prompt=SECRET_SENTINEL),
        allow_fallback=False,
    )
    results["provider_available"] = available
    cases.append(
        _case(
            root,
            settings,
            "provider_available",
            available,
            expected="succeeded",
        )
    )

    missing = ProviderService(
        settings,
        catalog=catalog,
        bindings={},
    ).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )
    results["provider_missing"] = missing
    cases.append(
        _case(
            root,
            settings,
            "provider_missing",
            missing,
            expected="failed:provider_missing",
        )
    )

    stale = _invoke_scripted(
        settings,
        catalog,
        health=_health("stale", observed_at=datetime.now(UTC) - timedelta(hours=1)),
        execution=ProviderExecution(_describe_output()),
    )
    results["provider_stale"] = stale
    cases.append(
        _case(
            root,
            settings,
            "provider_stale",
            stale,
            expected="failed:provider_stale",
        )
    )

    mismatch = _invoke_scripted(
        settings,
        catalog,
        health=_health("version-mismatch"),
        execution=ProviderExecution(_describe_output()),
    )
    results["version_mismatch"] = mismatch
    cases.append(
        _case(
            root,
            settings,
            "version_mismatch",
            mismatch,
            expected="failed:version_mismatch",
        )
    )

    timeout = _invoke_scripted(
        settings,
        catalog,
        health=_health(),
        execution=ProviderTimeout("scripted timeout"),
    )
    results["provider_timeout"] = timeout
    cases.append(
        _case(
            root,
            settings,
            "provider_timeout",
            timeout,
            expected="failed:provider_timeout",
        )
    )

    malformed = _invoke_scripted(
        settings,
        catalog,
        health=_health(),
        execution=ProviderExecution({"unexpected": True}),
    )
    results["malformed_response"] = malformed
    cases.append(
        _case(
            root,
            settings,
            "malformed_response",
            malformed,
            expected="failed:malformed_response",
        )
    )

    external = _binding(data_boundary="external")
    privacy = ProviderService(
        settings,
        catalog=catalog,
        bindings={DESCRIBE_ALIAS: external},
        executor=_ScriptedExecutor(_health(), ProviderExecution(_describe_output())),
    ).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        privacy_mode="local-only",
        explicit_opt_in=False,
        allow_fallback=False,
    )
    results["privacy_denied"] = privacy
    cases.append(
        _case(
            root,
            settings,
            "privacy_denied",
            privacy,
            expected="refused:privacy_denied",
        )
    )

    regions = next(item for item in catalog.declarations if item.alias == REGIONS_ALIAS)
    fallback = ProviderService(
        settings,
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
        {
            "media_path": str(root / "unopened-source.png"),
            "time_seconds": None,
            "labels": [],
        },
    )
    results["deterministic_fallback"] = fallback
    cases.append(
        _case(
            root,
            settings,
            "deterministic_fallback",
            fallback,
            expected="succeeded:fallback-used",
        )
    )

    partial = _invoke_scripted(
        settings,
        catalog,
        health=_health(),
        execution=ProviderExecution(_describe_output(partial=True), partial=True),
    )
    results["partial_evidence"] = partial
    cases.append(
        _case(
            root,
            settings,
            "partial_evidence",
            partial,
            expected="partial",
        )
    )

    human_supersession = _prove_human_supersession(settings, available, root)
    baseline = run_generic_suite(root / "baseline-without-ai")
    doctor_settings = Settings.for_home(root / "doctor-home")
    doctor_settings.editing_home.mkdir(parents=True)
    _write_doctor_overlay(doctor_settings.local_overlay, live_binding)
    doctor = doctor_report(doctor_settings, create_roots=True)
    doctor_path = root / "doctor-provider-status.json"
    _write_new_json(doctor_path, doctor)
    doctor_binding = next(
        item
        for item in doctor["providers"]["bindings"]
        if item["alias"] == DESCRIBE_ALIAS
    )
    available_receipt = (
        settings.editing_home / available.receipt_path
    ).read_text(encoding="utf-8")
    domain_text = (
        repository / "src" / "aoa_editing" / "domain" / "models.py"
    ).read_text(encoding="utf-8")
    tracked_secret_hits = _tracked_secret_hits(repository)
    checks = [
        _check(
            "typed-capability-declarations",
            len(catalog.declarations) >= 9,
            "the tracked catalog contains every required typed capability",
            {"count": len(catalog.declarations)},
        ),
        _check(
            "path-free-alias-contract",
            catalog.physical_model_paths_allowed is False
            and catalog.filesystem_links_allowed is False
            and all(item.alias.startswith("local-ai://") for item in catalog.declarations),
            "aliases are URI contracts rather than model paths or symlinks",
        ),
        _check(
            "available-missing-stale-version-timeout-malformed",
            all(item.overall == "pass" for item in cases[:6]),
            "all provider resolution and transport branches are typed",
        ),
        _check(
            "privacy-and-fallback",
            privacy.receipt.failure_code == "privacy_denied"
            and fallback.receipt.adapter_kind == "deterministic-fallback"
            and fallback.receipt.output_authority == "evidence",
            "privacy denial and deterministic fallback preserve authority",
        ),
        _check(
            "partial-evidence",
            partial.receipt.outcome == "partial"
            and partial.receipt.output_authority == "evidence",
            "partial AI output remains explicitly partial evidence",
        ),
        _check(
            "human-correction-supersession",
            human_supersession["passed"] is True,
            "a human correction supersedes AI evidence without deleting history",
            human_supersession,
        ),
        _check(
            "baseline-without-ai",
            baseline.get("ok") is True and len(baseline.get("scenarios", {})) == 3,
            "all deterministic scenarios run without an AI binding",
            {"scenarios": sorted(baseline.get("scenarios", {}))},
        ),
        _check(
            "no-model-path-in-domain-core",
            "model_path:" not in domain_text
            and "/legacy-host-root" not in domain_text
            and "/home/" not in domain_text,
            "domain core contains no physical model path contract",
        ),
        _check(
            "doctor-resolved-binding",
            doctor["providers"]["contract_ok"] is True
            and doctor_binding["binding_state"] == "enabled"
            and doctor_binding["health"]["status"] == "healthy",
            "doctor resolves the untracked binding and live health",
            {
                "binding_state": doctor_binding["binding_state"],
                "health": doctor_binding["health"]["status"],
            },
        ),
        _check(
            "no-secrets-in-receipts-or-git",
            SECRET_SENTINEL not in available_receipt and not tracked_secret_hits,
            "provider parameters are redacted and tracked files contain no credential literal",
            {"tracked_hits": tracked_secret_hits},
        ),
        _check(
            "ai-output-noncanonical",
            all(
                item.receipt.output_authority in {"evidence", "proposal"}
                for item in results.values()
            ),
            "every provider receipt limits output to evidence or proposal authority",
        ),
    ]
    overall: Literal["pass", "fail"] = (
        "pass"
        if all(item.overall == "pass" for item in cases)
        and all(item.status == "pass" for item in checks)
        else "fail"
    )
    artifacts = {
        "catalog": str(catalog_path.relative_to(repository)),
        "doctor": str(doctor_path.relative_to(root)),
        "baseline": "baseline-without-ai/generic-eval.json",
        "provider_receipts": str(
            (settings.state_root / "provider-receipts").relative_to(root)
        ),
    }
    report = ProviderAliasEvalReport(
        implementation_revision=implementation_revision,
        git_clean=git_clean,
        catalog_sha256=sha256_file(catalog_path),
        declaration_count=len(catalog.declarations),
        cases=cases,
        checks=checks,
        artifacts=artifacts,
        provenance=Provenance(
            tool="aoa-editing-provider-alias-eval",
            tool_version=__version__,
            parameters={
                "transport": "shell-free-local-cli-and-injected-failure-corpus",
                "reference_media_decoded": False,
                "mandatory_skips": 0,
            },
            deterministic=True,
        ),
        overall=overall,
    )
    _write_new_json(root / "provider-alias-eval.json", report.model_dump(mode="json"))
    if report.overall != "pass":
        raise ProviderAliasEvalError("provider alias corpus failed")
    return report


def _invoke_scripted(
    settings: Settings,
    catalog: AICapabilityCatalog,
    *,
    health: ProviderHealthEvidence,
    execution: ProviderExecution | Exception,
) -> ProviderInvocationResult:
    return ProviderService(
        settings,
        catalog=catalog,
        bindings={DESCRIBE_ALIAS: _binding()},
        executor=_ScriptedExecutor(health, execution),
    ).invoke(
        DESCRIBE_ALIAS,
        _describe_request(),
        allow_fallback=False,
    )


def _binding(
    *,
    command: list[str] | None = None,
    health_command: list[str] | None = None,
    data_boundary: str = "local-host",
) -> LocalProviderBinding:
    return LocalProviderBinding(
        alias=DESCRIBE_ALIAS,
        state="enabled",
        resolved_owner="abyss-machine",
        adapter_kind="abyss-machine-cli",
        backend="fixture-backend",
        model_id="fixture-model",
        model_revision="fixture-r1",
        model_hash_authority="fixture inventory",
        license_authority="fixture model card",
        metadata_evidence="runtime-eval:fixture-provider-inventory",
        command=command or ["fixture-provider", "{request_json}"],
        health=LocalProviderHealthBinding(
            command=health_command or ["fixture-health"],
        ),
        data_boundary=data_boundary,  # type: ignore[arg-type]
    )


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
        summary=f"scripted provider health is {status}",
    )


def _describe_request(*, prompt: str = "describe the composition") -> dict[str, Any]:
    return {
        "media_path": "/runtime/private/source.png",
        "sample_times_seconds": [0.0],
        "prompt": prompt,
    }


def _describe_output(*, partial: bool = False) -> dict[str, Any]:
    return {
        "descriptions": [
            {
                "time_seconds": 0.0,
                "text": "A geometric evaluation fixture.",
                "confidence": 0.9,
            }
        ],
        "partial": partial,
    }


def _case(
    root: Path,
    settings: Settings,
    case_id: str,
    result: ProviderInvocationResult,
    *,
    expected: str,
) -> ProviderAliasEvalCase:
    actual = str(result.receipt.outcome)
    if result.receipt.failure_code is not None:
        actual = f"{actual}:{result.receipt.failure_code}"
    elif result.receipt.fallback_status == "used":
        actual = f"{actual}:fallback-used"
    receipt_path = settings.editing_home / result.receipt_path
    passed = actual == expected and receipt_path.is_file()
    return ProviderAliasEvalCase(
        case_id=case_id,
        expected_outcome=expected,
        actual_outcome=actual,
        receipt_path=str(receipt_path.relative_to(root)),
        receipt_sha256=sha256_file(receipt_path),
        checks=[
            _check(
                "typed-outcome",
                actual == expected,
                "provider outcome and failure code match the case contract",
                {"expected": expected, "actual": actual},
            ),
            _check(
                "immutable-receipt",
                receipt_path.is_file(),
                "the provider attempt produced a durable receipt",
                {"path": str(receipt_path.relative_to(root))},
            ),
            _check(
                "noncanonical-authority",
                result.receipt.output_authority in {"evidence", "proposal"},
                "provider output authority cannot become a canonical edit",
                {"authority": result.receipt.output_authority},
            ),
        ],
        overall="pass" if passed else "fail",
    )


def _prove_human_supersession(
    settings: Settings,
    invocation: ProviderInvocationResult,
    root: Path,
) -> dict[str, Any]:
    source = root / "human-correction-source.png"
    Image.new("RGB", (96, 96), "#335577").save(source)
    store = ProjectStore(settings)
    editing = EditingService(store)
    project = editing.create_project(
        "Provider authority boundary",
        Intent(text="Keep AI as evidence", scenario=Scenario.STILL_MOTION),
    )
    asset, _probe = editing.ingest(project.id, source)
    evidence = EvidenceRecord(
        project_id=project.id,
        asset_id=asset.id,
        source_sha256=asset.sha256,
        kind="vision.describe",
        payload=invocation.output or {},
        provenance=Provenance(
            tool="aoa-editing-provider-alias-eval",
            tool_version=__version__,
            parameters={
                "provider_receipt_id": invocation.receipt.id,
                "provider_receipt_path": invocation.receipt_path,
                "output_authority": invocation.receipt.output_authority,
            },
            model_id=invocation.receipt.model_id,
            model_revision=invocation.receipt.model_revision,
            deterministic=False,
        ),
    )
    store.save_evidence(evidence)
    correction = editing.correct_evidence(
        project.id,
        asset_id=asset.id,
        kind="vision.describe",
        payload={"descriptions": [{"text": "Human-reviewed fixture."}]},
        supersedes=[evidence.id],
        rationale="The operator reviewed the actual source.",
    )
    all_ids = {item.id for item in store.list_evidence(project.id)}
    effective_ids = {item.id for item in store.list_effective_evidence(project.id)}
    passed = (
        {evidence.id, correction.id}.issubset(all_ids)
        and correction.id in effective_ids
        and evidence.id not in effective_ids
        and store.load_project(project.id).current_version_id is None
    )
    return {
        "passed": passed,
        "provider_evidence_id": evidence.id,
        "human_correction_id": correction.id,
        "history_preserved": {evidence.id, correction.id}.issubset(all_ids),
        "canonical_version_created": False,
    }


def _write_fixture_provider(path: Path) -> Path:
    script = """#!/usr/bin/env python3
import json
import sys
from datetime import UTC, datetime

if sys.argv[1] == "health":
    print(json.dumps({
        "status": "ready",
        "observed_at": datetime.now(UTC).isoformat(),
        "protocol_version": "aoa-local-ai-v1",
        "backend": "fixture-backend",
        "model_id": "fixture-model",
        "model_revision": "fixture-r1"
    }))
elif sys.argv[1] == "invoke":
    request = json.loads(sys.argv[2])
    assert request["media_path"]
    print(json.dumps({
        "descriptions": [{
            "time_seconds": 0.0,
            "text": "A geometric evaluation fixture.",
            "confidence": 0.9
        }],
        "partial": False
    }))
else:
    raise SystemExit(2)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o700)
    return path


def _write_doctor_overlay(
    path: Path,
    binding: LocalProviderBinding,
) -> None:
    health = binding.health
    if health is None:
        raise ProviderAliasEvalError("doctor fixture binding has no health")
    lines = [
        f'[ai.bindings."{binding.alias}"]',
        f'state = "{binding.state}"',
        f'resolved_owner = "{binding.resolved_owner}"',
        f'adapter_kind = "{binding.adapter_kind}"',
        f'backend = "{binding.backend}"',
        f'model_id = "{binding.model_id}"',
        f'model_revision = "{binding.model_revision}"',
        f'model_hash_authority = "{binding.model_hash_authority}"',
        f'license_authority = "{binding.license_authority}"',
        f'metadata_evidence = "{binding.metadata_evidence}"',
        f"command = {json.dumps(binding.command)}",
        f'data_boundary = "{binding.data_boundary}"',
        "",
        f'[ai.bindings."{binding.alias}".health]',
        f"command = {json.dumps(health.command)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _tracked_secret_hits(repository: Path) -> list[str]:
    result = subprocess.run(
        ["git", "grep", "-I", "-n", "-E", r"sk-proj-[A-Za-z0-9_-]{8,}"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode not in {0, 1}:
        raise ProviderAliasEvalError("cannot scan tracked files for credential literals")
    return result.stdout.splitlines()


def _check(
    check_id: str,
    passed: bool,
    summary: str,
    measured: dict[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        id=check_id,
        status="pass" if passed else "fail",
        summary=summary,
        measured=measured or {},
    )


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ProviderAliasEvalError(f"refusing to replace eval artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
