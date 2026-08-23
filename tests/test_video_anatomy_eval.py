from __future__ import annotations

import json
from pathlib import Path

import pytest

from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import Intent, Scenario
from aoa_editing.evals.gate import ReadinessError
from aoa_editing.evals.video_anatomy import _render_transitions, run_video_anatomy_eval
from aoa_editing.infrastructure.store import ProjectStore


def test_network_free_video_anatomy_gate_covers_declared_truth(tmp_path: Path) -> None:
    report = run_video_anatomy_eval(
        tmp_path / "video-anatomy-eval",
        repository=Path(__file__).resolve().parents[1],
        implementation_revision="a" * 40,
        git_clean=False,
        require_clean=False,
    )

    assert report.overall == "pass"
    assert report.mandatory_skips == 0
    assert report.reference_media_used is False
    assert report.aggregate_metrics["boundary_precision"] >= 0.7
    assert report.aggregate_metrics["boundary_recall"] >= 0.8
    assert report.aggregate_metrics["short_event_frame_selection_recall"] == 1
    assert report.aggregate_metrics["semantic_observation_coverage"] == 1
    assert report.aggregate_metrics["motion_model_error"] <= 0.4
    assert report.aggregate_metrics["audio_event_coverage"] == 1
    assert report.aggregate_metrics["vfr_timestamp_exactness"] == 1
    assert report.aggregate_metrics["fractional_rate_identity"] == 1
    payload = json.loads(
        (tmp_path / "video-anatomy-eval" / "video-anatomy-eval.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["reference_media_used"] is False


def test_reference_registration_fails_before_gate_and_never_copies_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _render_transitions(tmp_path / "reference-fixture")
    video = fixture.video_path
    digest = fixture.truth.video_sha256

    settings = Settings.for_home(tmp_path / "reference-home")
    store = ProjectStore(settings)
    project = store.create_project(
        "Sealed evaluation",
        Intent(text="Analyze only after readiness", scenario=Scenario.MEMORY_MONTAGE),
    )
    EditingService(store).seal_reference(project.id, digest)
    store.bind_reference_media(digest, video)

    def denied(_settings: Settings) -> dict[str, object]:
        raise ReadinessError("fixture gate remains sealed")

    monkeypatch.setattr("aoa_editing.evals.gate.require_readiness", denied)
    with pytest.raises(ReadinessError, match="remains sealed"):
        store.register_evaluation_reference_asset(
            project.id,
            digest,
            readiness_revision="b" * 40,
        )

    monkeypatch.setattr(
        "aoa_editing.evals.gate.require_readiness",
        lambda _settings: {"overall": "pass", "git_revision": "b" * 40},
    )
    asset, evidence = store.register_evaluation_reference_asset(
        project.id,
        digest,
        readiness_revision="b" * 40,
    )

    asset_root = store.project_path(project.id) / "assets" / asset.id
    assert [item.name for item in asset_root.iterdir()] == ["asset.json"]
    assert store.asset_source_path(project.id, asset.id) == video.resolve()
    assert evidence.payload["evaluation_reference"]["media_copied_into_project"] is False
    assert asset.sha256 in store.load_project(project.id).sealed_reference_hashes
