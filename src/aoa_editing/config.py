"""Resolve one owner-correct AoA Editing operational home."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

LEGACY_ROOT_VARIABLES = (
    "AOA_EDITING_DATA_ROOT",
    "AOA_EDITING_CACHE_ROOT",
    "AOA_EDITING_TMP_ROOT",
    "AOA_EDITING_RUNTIME_ROOT",
)


class ConfigurationError(ValueError):
    """The operational home cannot be resolved without guessing."""


def _resolved_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.resolve()


def _source_checkout(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "aoa_editing"
        ).is_dir():
            return candidate.resolve()
    return None


def _read_local_overlay(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"cannot parse repo-local config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigurationError(f"repo-local config is not a TOML table: {path}")
    return payload


@dataclass(frozen=True, slots=True)
class StorageLayout:
    """Every project-owned runtime path derived from one editing home."""

    home: Path
    var_root: Path
    runtime_root: Path
    projects_root: Path
    evals_root: Path
    artifacts_root: Path
    cache_root: Path
    tmp_root: Path
    state_root: Path
    data_root: Path
    local_overlay: Path
    resolution_source: str
    compatibility_overrides: tuple[str, ...] = ()

    @classmethod
    def for_home(
        cls,
        home: Path,
        *,
        resolution_source: str,
        local_overlay: Path | None = None,
    ) -> StorageLayout:
        selected = home.expanduser().resolve()
        var_root = selected / "var"
        return cls(
            home=selected,
            var_root=var_root,
            runtime_root=selected / ".venv",
            projects_root=var_root / "projects",
            evals_root=var_root / "evals",
            artifacts_root=var_root / "artifacts",
            cache_root=var_root / "cache",
            tmp_root=var_root / "tmp",
            state_root=var_root / "state",
            data_root=var_root,
            local_overlay=local_overlay or selected / ".aoa-editing.local.toml",
            resolution_source=resolution_source,
        )

    @classmethod
    def discover(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        source_home: Path | None = None,
    ) -> StorageLayout:
        environment = os.environ if environ is None else environ
        explicit = environment.get("AOA_EDITING_HOME")
        if explicit:
            selected = _resolved_path(explicit)
            return cls.for_home(
                selected,
                resolution_source="environment",
                local_overlay=selected / ".aoa-editing.local.toml",
            )

        checkout = (
            source_home.expanduser().resolve()
            if source_home is not None
            else _source_checkout(Path(__file__).resolve())
        )
        if checkout is not None:
            overlay = checkout / ".aoa-editing.local.toml"
            if overlay.is_file():
                payload = _read_local_overlay(overlay)
                editing = payload.get("editing", {})
                if not isinstance(editing, dict):
                    raise ConfigurationError(
                        f"[editing] must be a TOML table in repo-local config {overlay}"
                    )
                configured = editing.get("home")
                if configured is not None:
                    if not isinstance(configured, str) or not configured.strip():
                        raise ConfigurationError(
                            f"editing.home must be a non-empty path string in {overlay}"
                        )
                    return cls.for_home(
                        _resolved_path(configured, relative_to=checkout),
                        resolution_source="repo_local_config",
                        local_overlay=overlay,
                    )
            return cls.for_home(
                checkout,
                resolution_source="source_checkout",
                local_overlay=overlay,
            )

        xdg_data = environment.get("XDG_DATA_HOME")
        packaged_home = (
            _resolved_path(xdg_data) / "aoa-editing"
            if xdg_data
            else Path.home().resolve() / ".local" / "share" / "aoa-editing"
        )
        return cls.for_home(packaged_home, resolution_source="packaged_profile")

    def with_legacy_overrides(self, environ: Mapping[str, str]) -> StorageLayout:
        data_root = self.data_root
        projects_root = self.projects_root
        evals_root = self.evals_root
        artifacts_root = self.artifacts_root
        state_root = self.state_root
        cache_root = self.cache_root
        tmp_root = self.tmp_root
        runtime_root = self.runtime_root
        used: list[str] = []
        legacy_data = environ.get("AOA_EDITING_DATA_ROOT")
        if legacy_data:
            root = _resolved_path(legacy_data)
            data_root = root
            projects_root = root / "projects"
            evals_root = root / "evals"
            artifacts_root = root / "artifacts"
            state_root = root
            used.append("AOA_EDITING_DATA_ROOT")
        legacy_cache = environ.get("AOA_EDITING_CACHE_ROOT")
        if legacy_cache:
            cache_root = _resolved_path(legacy_cache)
            used.append("AOA_EDITING_CACHE_ROOT")
        legacy_tmp = environ.get("AOA_EDITING_TMP_ROOT")
        if legacy_tmp:
            tmp_root = _resolved_path(legacy_tmp)
            used.append("AOA_EDITING_TMP_ROOT")
        legacy_runtime = environ.get("AOA_EDITING_RUNTIME_ROOT")
        if legacy_runtime:
            root = _resolved_path(legacy_runtime)
            runtime_root = root if root.name in {"venv", ".venv"} else root / "venv"
            used.append("AOA_EDITING_RUNTIME_ROOT")
        if not used:
            return self
        return replace(
            self,
            data_root=data_root,
            projects_root=projects_root,
            evals_root=evals_root,
            artifacts_root=artifacts_root,
            state_root=state_root,
            cache_root=cache_root,
            tmp_root=tmp_root,
            runtime_root=runtime_root,
            compatibility_overrides=tuple(used),
        )

    @property
    def owned_mutable_roots(self) -> tuple[Path, ...]:
        return (
            self.projects_root,
            self.evals_root,
            self.artifacts_root,
            self.cache_root,
            self.tmp_root,
            self.state_root,
        )

    def ensure_roots(self) -> None:
        for path in self.owned_mutable_roots:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by UI, CLI, agent, tests, and evaluators."""

    layout: StorageLayout
    bind_host: str
    bind_port: int

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        source_home: Path | None = None,
    ) -> Settings:
        environment = os.environ if environ is None else environ
        layout = StorageLayout.discover(
            environ=environment,
            source_home=source_home,
        )
        if layout.resolution_source not in {"environment", "repo_local_config"}:
            layout = layout.with_legacy_overrides(environment)
        return cls(
            layout=layout,
            bind_host=environment.get("AOA_EDITING_BIND", "127.0.0.1"),
            bind_port=int(environment.get("AOA_EDITING_PORT", "8787")),
        )

    @classmethod
    def for_home(
        cls,
        home: Path,
        *,
        runtime_root: Path | None = None,
        bind_host: str = "127.0.0.1",
        bind_port: int = 8787,
    ) -> Settings:
        layout = StorageLayout.for_home(home, resolution_source="test")
        if runtime_root is not None:
            layout = replace(layout, runtime_root=runtime_root.expanduser().resolve())
        return cls(
            layout=layout,
            bind_host=bind_host,
            bind_port=bind_port,
        )

    @property
    def editing_home(self) -> Path:
        return self.layout.home

    @property
    def local_overlay(self) -> Path:
        return self.layout.local_overlay

    @property
    def data_root(self) -> Path:
        """Compatibility view of the formerly combined data root."""

        return self.layout.data_root

    @property
    def runtime_root(self) -> Path:
        return self.layout.runtime_root

    @property
    def projects_root(self) -> Path:
        return self.layout.projects_root

    @property
    def evals_root(self) -> Path:
        return self.layout.evals_root

    @property
    def artifacts_root(self) -> Path:
        return self.layout.artifacts_root

    @property
    def cache_root(self) -> Path:
        return self.layout.cache_root

    @property
    def tmp_root(self) -> Path:
        return self.layout.tmp_root

    @property
    def state_root(self) -> Path:
        return self.layout.state_root

    def ensure_roots(self) -> None:
        self.layout.ensure_roots()
