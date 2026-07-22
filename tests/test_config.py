from __future__ import annotations

from pathlib import Path

import pytest

from aoa_editing.config import ConfigurationError, Settings, StorageLayout
from aoa_editing.infrastructure.probes import doctor_report

REPO = Path(__file__).resolve().parents[1]
LEGACY_ENV = (
    "AOA_EDITING_DATA_ROOT",
    "AOA_EDITING_CACHE_ROOT",
    "AOA_EDITING_TMP_ROOT",
    "AOA_EDITING_RUNTIME_ROOT",
)


def _clear_home_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AOA_EDITING_HOME", raising=False)
    for name in LEGACY_ENV:
        monkeypatch.delenv(name, raising=False)


def test_default_home_is_the_checkout_and_does_not_follow_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_home_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_env()

    assert settings.editing_home == REPO
    assert settings.runtime_root == REPO / ".venv"
    assert settings.projects_root == REPO / "var" / "projects"
    assert settings.evals_root == REPO / "var" / "evals"
    assert settings.artifacts_root == REPO / "var" / "artifacts"
    assert settings.cache_root == REPO / "var" / "cache"
    assert settings.tmp_root == REPO / "var" / "tmp"
    assert settings.state_root == REPO / "var" / "state"
    assert settings.layout.resolution_source == "source_checkout"


def test_explicit_home_is_a_complete_isolated_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_home_environment(monkeypatch)
    home = tmp_path / "portable-editing"
    monkeypatch.setenv("AOA_EDITING_HOME", str(home))

    settings = Settings.from_env()
    settings.ensure_roots()

    assert settings.editing_home == home.resolve()
    assert settings.layout.resolution_source == "environment"
    assert settings.layout.compatibility_overrides == ()
    for path in (
        settings.projects_root,
        settings.evals_root,
        settings.artifacts_root,
        settings.cache_root,
        settings.tmp_root,
        settings.state_root,
    ):
        assert path.is_dir()
        assert path.is_relative_to(home)


def test_repo_local_overlay_can_select_home(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    selected = tmp_path / "selected-home"
    (checkout / ".aoa-editing.local.toml").write_text(
        f'[editing]\nhome = "{selected}"\n', encoding="utf-8"
    )

    layout = StorageLayout.discover(environ={}, source_home=checkout)

    assert layout.home == selected.resolve()
    assert layout.resolution_source == "repo_local_config"
    assert layout.local_overlay == checkout / ".aoa-editing.local.toml"


def test_invalid_repo_local_overlay_fails_closed(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".aoa-editing.local.toml").write_text("[editing\nhome = false\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="cannot parse"):
        StorageLayout.discover(environ={}, source_home=checkout)


def test_legacy_root_overrides_are_explicit_compatibility_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_home_environment(monkeypatch)
    monkeypatch.setenv("AOA_EDITING_DATA_ROOT", str(tmp_path / "legacy-data"))
    monkeypatch.setenv("AOA_EDITING_CACHE_ROOT", str(tmp_path / "legacy-cache"))
    monkeypatch.setenv("AOA_EDITING_TMP_ROOT", str(tmp_path / "legacy-tmp"))
    monkeypatch.setenv("AOA_EDITING_RUNTIME_ROOT", str(tmp_path / "legacy-runtime"))

    settings = Settings.from_env()

    assert settings.projects_root == (tmp_path / "legacy-data" / "projects").resolve()
    assert settings.evals_root == (tmp_path / "legacy-data" / "evals").resolve()
    assert settings.state_root == (tmp_path / "legacy-data").resolve()
    assert settings.runtime_root == (tmp_path / "legacy-runtime" / "venv").resolve()
    assert set(settings.layout.compatibility_overrides) == set(LEGACY_ENV)


def test_explicit_home_has_priority_over_legacy_root_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_home_environment(monkeypatch)
    home = tmp_path / "explicit-home"
    monkeypatch.setenv("AOA_EDITING_HOME", str(home))
    monkeypatch.setenv("AOA_EDITING_DATA_ROOT", str(tmp_path / "stale-data"))
    monkeypatch.setenv("AOA_EDITING_RUNTIME_ROOT", str(tmp_path / "stale-runtime"))

    settings = Settings.from_env()

    assert settings.editing_home == home.resolve()
    assert settings.projects_root == (home / "var" / "projects").resolve()
    assert settings.runtime_root == (home / ".venv").resolve()
    assert settings.layout.compatibility_overrides == ()


def test_doctor_reports_one_owner_typed_layout(tmp_path: Path) -> None:
    settings = Settings.for_home(tmp_path / "home")
    report = doctor_report(settings, create_roots=True)

    assert report["editing_home"]["path"] == str(settings.editing_home)
    assert report["editing_home"]["resolution_source"] == "test"
    assert set(report["paths"]) == {
        "runtime",
        "projects",
        "evals",
        "artifacts",
        "cache",
        "tmp",
        "state",
    }
    assert {entry["owner"] for entry in report["paths"].values()} == {"aoa-editing"}
