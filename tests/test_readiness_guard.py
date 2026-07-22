from __future__ import annotations

import json

import pytest

from aoa_editing.config import Settings
from aoa_editing.evals import gate


def test_readiness_guard_rejects_dirty_worktree_after_passing_receipt(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    settings = Settings.for_home(tmp_path / "editing-home")
    latest = settings.evals_root / "readiness" / "latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps({"overall": "pass", "git_revision": "a" * 40}),
        encoding="utf-8",
    )

    def fake_git(_repo, arguments):  # type: ignore[no-untyped-def]
        return "a" * 40 if arguments == ["rev-parse", "HEAD"] else " M changed.py\n"

    monkeypatch.setattr(gate, "_git", fake_git)

    with pytest.raises(gate.ReadinessError, match="worktree changed after readiness"):
        gate.require_readiness(settings)


def test_readiness_guard_accepts_same_clean_revision(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    settings = Settings.for_home(tmp_path / "editing-home")
    latest = settings.evals_root / "readiness" / "latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps({"overall": "pass", "git_revision": "a" * 40}),
        encoding="utf-8",
    )

    def fake_git(_repo, arguments):  # type: ignore[no-untyped-def]
        return "a" * 40 if arguments == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(gate, "_git", fake_git)

    assert gate.require_readiness(settings)["git_revision"] == "a" * 40
