"""Fail-closed provider resolution with local-only transports and durable receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx2 as httpx
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import (  # type: ignore[import-untyped]
    ValidationError as JSONSchemaValidationError,
)

from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    AICapabilityCatalog,
    AICapabilityDeclaration,
    LocalProviderBinding,
    ProviderHealthEvidence,
    ProviderInvocationResult,
    ProviderPrivacyDecision,
    ResolvedProviderReceipt,
)
from aoa_editing.providers.catalog import (
    ProviderConfigurationError,
    load_capability_catalog,
    load_local_bindings,
)

Fallback = Callable[[dict[str, Any]], dict[str, Any]]
PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.-]+)\}")
SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}
PATH_KEYS = {
    "audio",
    "file",
    "image",
    "media",
    "media_path",
    "path",
    "source",
    "video",
}


class ProviderExecutionError(RuntimeError):
    """A local provider transport did not satisfy its typed contract."""


class ProviderUnavailable(ProviderExecutionError):
    pass


class ProviderTimeout(ProviderExecutionError):
    pass


class ProviderMalformedResponse(ProviderExecutionError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderExecution:
    output: dict[str, Any]
    partial: bool = False


class ProviderExecutor(Protocol):
    def health(
        self,
        binding: LocalProviderBinding,
        declaration: AICapabilityDeclaration,
    ) -> ProviderHealthEvidence:
        """Probe current health without invoking the capability."""

    def invoke(
        self,
        binding: LocalProviderBinding,
        payload: dict[str, Any],
    ) -> ProviderExecution:
        """Invoke one already health-checked binding."""


class LocalProviderExecutor:
    """Shell-free CLI or credential-free loopback HTTP transport."""

    def health(
        self,
        binding: LocalProviderBinding,
        declaration: AICapabilityDeclaration,
    ) -> ProviderHealthEvidence:
        if binding.health is None:
            raise ProviderUnavailable("enabled binding has no health probe")
        started = time.monotonic()
        source: str
        try:
            if binding.health.command:
                source = "command"
                payload = _run_json_command(
                    binding.health.command,
                    timeout=binding.health.timeout_seconds,
                )
            else:
                source = "localhost-http"
                if binding.health.endpoint is None:
                    raise ProviderUnavailable("health endpoint is absent")
                payload = _get_json(
                    binding.health.endpoint,
                    timeout=binding.health.timeout_seconds,
                )
        except ProviderTimeout:
            return _health_failure(
                status="unavailable",
                started=started,
                source="command" if binding.health.command else "localhost-http",
                summary="provider health probe timed out",
            )
        except ProviderMalformedResponse:
            return _health_failure(
                status="malformed",
                started=started,
                source="command" if binding.health.command else "localhost-http",
                summary="provider health response is malformed",
            )
        except ProviderExecutionError:
            return _health_failure(
                status="unavailable",
                started=started,
                source="command" if binding.health.command else "localhost-http",
                summary="provider health probe failed",
            )
        selected = _select(payload, binding.health.selector)
        if not isinstance(selected, dict):
            return _health_failure(
                status="malformed",
                started=started,
                source=source,
                summary="provider health selector did not resolve to an object",
                response=payload,
            )
        reported_status = _optional_string(selected, binding.health.status_field)
        observed_at = _optional_datetime(selected, binding.health.observed_at_field)
        protocol = _optional_string(selected, binding.health.protocol_version_field)
        model_id = _optional_string(selected, binding.health.model_id_field)
        model_revision = _optional_string(selected, binding.health.model_revision_field)
        backend = _optional_string(selected, binding.health.backend_field)
        status: str = "healthy"
        summary = "provider health contract passed"
        if reported_status not in binding.health.ready_values:
            status = "unavailable"
            summary = "provider did not report a ready status"
        elif observed_at is not None and (
            datetime.now(UTC) - observed_at
        ).total_seconds() > declaration.health_contract.maximum_age_seconds:
            status = "stale"
            summary = "provider health evidence is stale"
        elif protocol is None or (
            protocol != declaration.health_contract.expected_protocol_version
        ):
            status = "version-mismatch"
            summary = "provider protocol version is absent or differs from the declaration"
        elif declaration.health_contract.model_revision_required and model_revision is None:
            status = "version-mismatch"
            summary = "provider health omitted the required model revision"
        elif model_id is not None and binding.model_id is not None and model_id != binding.model_id:
            status = "version-mismatch"
            summary = "live model ID differs from the local binding"
        elif (
            model_revision is not None
            and binding.model_revision is not None
            and model_revision != binding.model_revision
        ):
            status = "version-mismatch"
            summary = "live model revision differs from the local binding"
        elif backend is not None and binding.backend is not None and backend != binding.backend:
            status = "version-mismatch"
            summary = "live backend differs from the local binding"
        return ProviderHealthEvidence(
            status=status,  # type: ignore[arg-type]
            checked_at=datetime.now(UTC),
            observed_at=observed_at,
            latency_milliseconds=(time.monotonic() - started) * 1000,
            source=source,  # type: ignore[arg-type]
            response_sha256=_json_sha256(payload),
            reported_status=reported_status,
            protocol_version=protocol,
            backend=backend,
            model_id=model_id,
            model_revision=model_revision,
            summary=summary,
        )

    def invoke(
        self,
        binding: LocalProviderBinding,
        payload: dict[str, Any],
    ) -> ProviderExecution:
        if binding.command:
            response = _run_json_command(
                _render_command(binding.command, binding.alias, payload),
                timeout=binding.timeout_seconds,
            )
        elif binding.endpoint is not None:
            response = _post_json(
                binding.endpoint,
                payload,
                timeout=binding.timeout_seconds,
            )
        else:
            raise ProviderUnavailable("enabled binding has no invocation route")
        selected = _select(response, binding.output_selector)
        if not isinstance(selected, dict):
            raise ProviderMalformedResponse(
                "provider output selector did not resolve to an object"
            )
        partial = (
            bool(selected.get(binding.partial_field, False))
            if binding.partial_field is not None
            else False
        )
        return ProviderExecution(output=selected, partial=partial)


class ProviderService:
    """Application-facing resolver; every attempted call produces one receipt."""

    def __init__(
        self,
        settings: Settings,
        *,
        catalog: AICapabilityCatalog | None = None,
        bindings: Mapping[str, LocalProviderBinding] | None = None,
        executor: ProviderExecutor | None = None,
        fallbacks: Mapping[str, Fallback] | None = None,
    ):
        self.settings = settings
        self.catalog = catalog or load_capability_catalog()
        self.declarations = {
            declaration.alias: declaration
            for declaration in self.catalog.declarations
        }
        self.bindings = dict(
            bindings
            if bindings is not None
            else load_local_bindings(settings, catalog=self.catalog)
        )
        unknown = set(self.bindings) - set(self.declarations)
        if unknown:
            raise ProviderConfigurationError(
                f"bindings reference undeclared aliases: {sorted(unknown)}"
            )
        self.executor = executor or LocalProviderExecutor()
        self.fallbacks = dict(fallbacks or {})
        self.receipt_root = settings.state_root / "provider-receipts"

    @classmethod
    def for_settings(cls, settings: Settings) -> ProviderService:
        return cls(settings)

    def invoke(
        self,
        alias: str,
        payload: dict[str, Any],
        *,
        privacy_mode: str = "local-only",
        explicit_opt_in: bool = False,
        allow_fallback: bool = True,
        project_id: str | None = None,
        asset_id: str | None = None,
    ) -> ProviderInvocationResult:
        started = time.monotonic()
        declaration = self.declarations.get(alias)
        if declaration is None:
            raise ProviderConfigurationError(f"undeclared provider alias: {alias}")
        privacy = ProviderPrivacyDecision(
            mode=privacy_mode,  # type: ignore[arg-type]
            data_boundary="local-host",
            decision="allowed",
            reason="no enabled external binding has been selected",
            explicit_opt_in=explicit_opt_in,
        )
        try:
            Draft202012Validator(declaration.input_schema).validate(payload)
        except JSONSchemaValidationError:
            return self._failure(
                declaration,
                binding=None,
                code="invalid_request",
                summary="provider request does not match the tracked input schema",
                started=started,
                privacy=privacy,
                allow_fallback=False,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
            )
        binding = self.bindings.get(alias)
        if binding is None:
            return self._failure(
                declaration,
                binding=None,
                code="provider_missing",
                summary="no local binding exists for the requested alias",
                started=started,
                privacy=privacy,
                allow_fallback=allow_fallback,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
            )
        if binding.state != "enabled":
            code = (
                "provider_disabled"
                if binding.state == "disabled"
                else "provider_unavailable"
            )
            return self._failure(
                declaration,
                binding=binding,
                code=code,
                summary=binding.reason or "local provider binding is inactive",
                started=started,
                privacy=privacy,
                allow_fallback=allow_fallback,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
            )
        privacy = _privacy_decision(
            privacy_mode=privacy_mode,
            explicit_opt_in=explicit_opt_in,
            data_boundary=binding.data_boundary,
        )
        if privacy.decision == "denied":
            return self._failure(
                declaration,
                binding=binding,
                code="privacy_denied",
                summary="privacy policy denied provider execution",
                started=started,
                privacy=privacy,
                allow_fallback=False,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
                refused=True,
            )
        health = self.executor.health(binding, declaration)
        if health.status != "healthy":
            code_by_health = {
                "stale": "provider_stale",
                "version-mismatch": "version_mismatch",
                "unavailable": "provider_unavailable",
                "malformed": "provider_unavailable",
            }
            return self._failure(
                declaration,
                binding=binding,
                code=code_by_health[health.status],
                summary=health.summary,
                started=started,
                privacy=privacy,
                allow_fallback=allow_fallback,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
                health=health,
            )
        try:
            execution = self.executor.invoke(binding, payload)
            Draft202012Validator(declaration.output_schema).validate(execution.output)
        except ProviderTimeout:
            return self._failure(
                declaration,
                binding=binding,
                code="provider_timeout",
                summary="provider invocation exceeded its timeout",
                started=started,
                privacy=privacy,
                allow_fallback=allow_fallback,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
                health=health,
            )
        except ProviderUnavailable:
            return self._failure(
                declaration,
                binding=binding,
                code="provider_unavailable",
                summary="provider invocation route is unavailable",
                started=started,
                privacy=privacy,
                allow_fallback=allow_fallback,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
                health=health,
            )
        except (ProviderMalformedResponse, JSONSchemaValidationError):
            return self._failure(
                declaration,
                binding=binding,
                code="malformed_response",
                summary="provider response did not satisfy the tracked output contract",
                started=started,
                privacy=privacy,
                allow_fallback=allow_fallback,
                payload=payload,
                project_id=project_id,
                asset_id=asset_id,
                health=health,
            )
        receipt = ResolvedProviderReceipt(
            requested_alias=alias,
            capability_id=declaration.capability_id,
            **_binding_receipt_fields(binding),
            parameters=_receipt_parameters(binding, payload),
            health_evidence=health,
            execution_time_milliseconds=(time.monotonic() - started) * 1000,
            privacy_decision=privacy,
            output_authority=declaration.authority_class,
            fallback_status="not-used",
            outcome="partial" if execution.partial else "succeeded",
            response_sha256=_json_sha256(execution.output),
            output_schema_valid=True,
            partial=execution.partial,
            project_id=project_id,
            asset_id=asset_id,
        )
        return self._persist(receipt, execution.output)

    def inspect_bindings(self, *, probe: bool = True) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for declaration in self.catalog.declarations:
            binding = self.bindings.get(declaration.alias)
            health: ProviderHealthEvidence | None = None
            if binding is not None and binding.state == "enabled" and probe:
                health = self.executor.health(binding, declaration)
            entries.append(
                {
                    "capability_id": declaration.capability_id,
                    "alias": declaration.alias,
                    "required": declaration.required,
                    "authority_class": declaration.authority_class,
                    "binding_state": binding.state if binding else "unbound",
                    "resolved_owner": binding.resolved_owner if binding else None,
                    "adapter_kind": binding.adapter_kind if binding else "unbound",
                    "backend": binding.backend if binding else None,
                    "model_id": binding.model_id if binding else None,
                    "model_revision": binding.model_revision if binding else None,
                    "health": health.model_dump(mode="json") if health else None,
                }
            )
        return {
            "schema": "aoa_editing_provider_status_v1",
            "catalog_id": self.catalog.id,
            "declaration_count": len(self.catalog.declarations),
            "baseline_requires_ai": self.catalog.baseline_requires_ai,
            "local_overlay": str(self.settings.local_overlay),
            "local_overlay_present": self.settings.local_overlay.is_file(),
            "bindings": entries,
        }

    def _failure(
        self,
        declaration: AICapabilityDeclaration,
        *,
        binding: LocalProviderBinding | None,
        code: str,
        summary: str,
        started: float,
        privacy: ProviderPrivacyDecision,
        allow_fallback: bool,
        payload: dict[str, Any],
        project_id: str | None,
        asset_id: str | None,
        health: ProviderHealthEvidence | None = None,
        refused: bool = False,
    ) -> ProviderInvocationResult:
        fallback = self.fallbacks.get(declaration.fallback.strategy_id)
        if (
            allow_fallback
            and declaration.fallback.behavior == "deterministic"
            and fallback is not None
        ):
            try:
                output = fallback(payload)
                Draft202012Validator(declaration.output_schema).validate(output)
            except Exception:
                return self._persist(
                    ResolvedProviderReceipt(
                        requested_alias=declaration.alias,
                        capability_id=declaration.capability_id,
                        resolved_owner="aoa-editing",
                        adapter_kind="deterministic-fallback",
                        backend="builtin-deterministic",
                        parameters={"request": _sanitize(payload)},
                        execution_time_milliseconds=(time.monotonic() - started) * 1000,
                        privacy_decision=_fallback_privacy(privacy),
                        output_authority=declaration.authority_class,
                        fallback_status="failed",
                        outcome="failed",
                        failure_code="fallback_failed",
                        failure_summary="deterministic fallback failed its output contract",
                        project_id=project_id,
                        asset_id=asset_id,
                    ),
                    None,
                )
            return self._persist(
                ResolvedProviderReceipt(
                    requested_alias=declaration.alias,
                    capability_id=declaration.capability_id,
                    resolved_owner="aoa-editing",
                    adapter_kind="deterministic-fallback",
                    backend="builtin-deterministic",
                    model_id=None,
                    model_revision=None,
                    model_hash_authority=None,
                    license_authority="AoA Editing source license",
                    metadata_evidence="tracked deterministic fallback implementation",
                    parameters={"request": _sanitize(payload), "provider_failure": code},
                    health_evidence=health,
                    execution_time_milliseconds=(time.monotonic() - started) * 1000,
                    privacy_decision=_fallback_privacy(privacy),
                    output_authority=declaration.authority_class,
                    fallback_status="used",
                    outcome="succeeded",
                    response_sha256=_json_sha256(output),
                    output_schema_valid=True,
                    project_id=project_id,
                    asset_id=asset_id,
                ),
                output,
            )
        receipt_fields = (
            _binding_receipt_fields(binding)
            if binding is not None
            else {
                "resolved_owner": None,
                "adapter_kind": "unbound",
                "backend": None,
                "model_id": None,
                "model_revision": None,
                "model_hash_authority": None,
                "license_authority": None,
                "metadata_evidence": None,
                "endpoint": None,
                "command": [],
            }
        )
        receipt = ResolvedProviderReceipt(
            requested_alias=declaration.alias,
            capability_id=declaration.capability_id,
            **receipt_fields,
            parameters=_receipt_parameters(binding, payload),
            health_evidence=health,
            execution_time_milliseconds=(time.monotonic() - started) * 1000,
            privacy_decision=privacy,
            output_authority=declaration.authority_class,
            fallback_status=(
                "unavailable"
                if allow_fallback and declaration.fallback.behavior == "deterministic"
                else "not-used"
            ),
            outcome="refused" if refused else "failed",
            failure_code=code,  # type: ignore[arg-type]
            failure_summary=summary[:500],
            project_id=project_id,
            asset_id=asset_id,
        )
        return self._persist(receipt, None)

    def _persist(
        self,
        receipt: ResolvedProviderReceipt,
        output: dict[str, Any] | None,
    ) -> ProviderInvocationResult:
        self.receipt_root.mkdir(parents=True, exist_ok=True)
        path = self.receipt_root / f"{receipt.id}.json"
        _write_new_json(path, receipt.model_dump(mode="json"))
        relative = path.relative_to(self.settings.editing_home)
        return ProviderInvocationResult(
            receipt=receipt,
            receipt_path=str(relative),
            output=output,
        )


def _binding_receipt_fields(binding: LocalProviderBinding) -> dict[str, Any]:
    return {
        "resolved_owner": binding.resolved_owner,
        "adapter_kind": binding.adapter_kind,
        "backend": binding.backend,
        "model_id": binding.model_id,
        "model_revision": binding.model_revision,
        "model_hash_authority": binding.model_hash_authority,
        "license_authority": binding.license_authority,
        "metadata_evidence": binding.metadata_evidence,
        "endpoint": binding.endpoint,
        "command": binding.command,
    }


def _privacy_decision(
    *,
    privacy_mode: str,
    explicit_opt_in: bool,
    data_boundary: str,
) -> ProviderPrivacyDecision:
    allowed = data_boundary == "local-host" or (
        privacy_mode == "explicit-provider-opt-in" and explicit_opt_in
    )
    return ProviderPrivacyDecision(
        mode=privacy_mode,  # type: ignore[arg-type]
        data_boundary=data_boundary,  # type: ignore[arg-type]
        decision="allowed" if allowed else "denied",
        reason=(
            "execution remains on the local host"
            if data_boundary == "local-host"
            else (
                "explicit provider opt-in permits the declared external boundary"
                if allowed
                else "external execution requires explicit provider opt-in"
            )
        ),
        explicit_opt_in=explicit_opt_in,
    )


def _fallback_privacy(
    original: ProviderPrivacyDecision,
) -> ProviderPrivacyDecision:
    return ProviderPrivacyDecision(
        mode=original.mode,
        data_boundary="deterministic-fallback",
        decision="allowed",
        reason="fallback remains inside the deterministic AoA Editing baseline",
        explicit_opt_in=original.explicit_opt_in,
    )


def _receipt_parameters(
    binding: LocalProviderBinding | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "binding": _sanitize(binding.parameters) if binding is not None else {},
        "request": _sanitize(payload),
    }


def _sanitize(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        redacted_count = 0
        for raw_key, item in value.items():
            normalized = str(raw_key).lower().replace("-", "_")
            if normalized in SECRET_KEYS or any(
                marker in normalized for marker in ("password", "secret", "token")
            ):
                redacted_count += 1
                continue
            sanitized[str(raw_key)] = _sanitize(item, key=normalized)
        if redacted_count:
            sanitized["_redacted_field_count"] = redacted_count
        return sanitized
    if isinstance(value, list):
        return [_sanitize(item, key=key) for item in value]
    if isinstance(value, str):
        normalized_key = (key or "").lower()
        if normalized_key in PATH_KEYS or any(
            marker in normalized_key for marker in ("_path", "_file", "_media")
        ):
            return "<local-media>"
        if value.startswith("/") or value.startswith("file://"):
            return "<local-path>"
        if "bearer " in value.lower() or "sk-proj-" in value.lower():
            return "<redacted>"
    return value


def _run_json_command(command: list[str], *, timeout: float) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ProviderTimeout("provider command timed out") from error
    except OSError as error:
        raise ProviderUnavailable("provider command is unavailable") from error
    if result.returncode != 0:
        raise ProviderUnavailable("provider command returned a non-zero status")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProviderMalformedResponse("provider command returned malformed JSON") from error
    if not isinstance(payload, dict):
        raise ProviderMalformedResponse("provider command JSON must be an object")
    return payload


def _get_json(endpoint: str, *, timeout: float) -> dict[str, Any]:
    try:
        response = httpx.get(endpoint, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as error:
        raise ProviderTimeout("provider health endpoint timed out") from error
    except (httpx.HTTPError, ValueError) as error:
        raise ProviderUnavailable("provider health endpoint failed") from error
    if not isinstance(payload, dict):
        raise ProviderMalformedResponse("provider health JSON must be an object")
    return payload


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    try:
        response = httpx.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        output = response.json()
    except httpx.TimeoutException as error:
        raise ProviderTimeout("provider endpoint timed out") from error
    except (httpx.HTTPError, ValueError) as error:
        raise ProviderUnavailable("provider endpoint failed") from error
    if not isinstance(output, dict):
        raise ProviderMalformedResponse("provider output JSON must be an object")
    return output


def _render_command(
    template: list[str],
    alias: str,
    payload: dict[str, Any],
) -> list[str]:
    context: dict[str, str] = {
        "alias": alias,
        "request_json": json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    }
    for key, value in payload.items():
        context[str(key)] = (
            value
            if isinstance(value, str)
            else json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        )

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in context:
            raise ProviderMalformedResponse(
                f"provider command references unknown request field {name}"
            )
        return context[name]

    return [PLACEHOLDER.sub(replace, token) for token in template]


def _select(payload: Any, selector: list[str]) -> Any:
    current = payload
    for key in selector:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _optional_string(payload: dict[str, Any], field: str | None) -> str | None:
    if field is None:
        return None
    value = payload.get(field)
    return str(value) if value is not None else None


def _optional_datetime(
    payload: dict[str, Any],
    field: str | None,
) -> datetime | None:
    value = _optional_string(payload, field)
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _health_failure(
    *,
    status: str,
    started: float,
    source: str,
    summary: str,
    response: dict[str, Any] | None = None,
) -> ProviderHealthEvidence:
    return ProviderHealthEvidence(
        status=status,  # type: ignore[arg-type]
        latency_milliseconds=(time.monotonic() - started) * 1000,
        source=source,  # type: ignore[arg-type]
        response_sha256=_json_sha256(response) if response is not None else None,
        summary=summary,
    )


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ProviderExecutionError(f"refusing to replace provider receipt {path.name}")
    partial = path.with_suffix(path.suffix + ".partial")
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
