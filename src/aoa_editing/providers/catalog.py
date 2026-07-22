"""Load the tracked alias catalog and the untracked local binding overlay."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from pydantic import ValidationError

from aoa_editing.config import Settings
from aoa_editing.domain.models import AICapabilityCatalog, LocalProviderBinding


class ProviderConfigurationError(ValueError):
    """A declaration or local binding cannot be trusted without guessing."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_catalog_path() -> Path:
    return repository_root() / "manifests" / "ai-capabilities.json"


def load_capability_catalog(path: Path | None = None) -> AICapabilityCatalog:
    selected = path or default_catalog_path()
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
        catalog = AICapabilityCatalog.model_validate(payload)
        for declaration in catalog.declarations:
            Draft202012Validator.check_schema(declaration.input_schema)
            Draft202012Validator.check_schema(declaration.output_schema)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as error:
        raise ProviderConfigurationError(
            f"cannot load AI capability catalog {selected}: {error}"
        ) from error
    return catalog


def load_local_bindings(
    settings: Settings,
    *,
    catalog: AICapabilityCatalog | None = None,
) -> dict[str, LocalProviderBinding]:
    selected_catalog = catalog or load_capability_catalog()
    declared = {item.alias for item in selected_catalog.declarations}
    overlay = settings.local_overlay
    if not overlay.is_file():
        return {}
    try:
        with overlay.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ProviderConfigurationError(
            f"cannot parse local provider overlay {overlay}: {error}"
        ) from error
    ai = payload.get("ai", {})
    if not isinstance(ai, dict):
        raise ProviderConfigurationError(f"[ai] must be a table in {overlay}")
    raw_bindings = ai.get("bindings", {})
    if not isinstance(raw_bindings, dict):
        raise ProviderConfigurationError(f"[ai.bindings] must be a table in {overlay}")
    bindings: dict[str, LocalProviderBinding] = {}
    for alias, raw in raw_bindings.items():
        if not isinstance(alias, str) or not isinstance(raw, dict):
            raise ProviderConfigurationError(
                f"each [ai.bindings] entry must be a named table in {overlay}"
            )
        if alias not in declared:
            raise ProviderConfigurationError(f"local binding uses undeclared alias {alias}")
        try:
            binding = LocalProviderBinding.model_validate(
                {"alias": alias, **_plain_mapping(raw)}
            )
        except ValidationError as error:
            raise ProviderConfigurationError(
                f"invalid local binding {alias}: {error}"
            ) from error
        bindings[alias] = binding
    return bindings


def _plain_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _plain_value(item)
        for key, item in value.items()
    }


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _plain_mapping(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value
