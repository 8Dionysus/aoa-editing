"""End-to-end generic suite across all product scenarios."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aoa_editing.analysis.service import AnalysisService
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import Intent, Scenario
from aoa_editing.evals.fixtures import create_fixtures
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.interchange.kdenlive import KdenliveExporter
from aoa_editing.interchange.otio import OTIOExporter
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService


def run_generic_suite(root: Path) -> dict[str, Any]:
    fixtures = create_fixtures(root / "fixtures")
    selected = Settings.from_env()
    settings = Settings.for_home(
        root / "workspace",
        runtime_root=selected.runtime_root,
    )
    store = ProjectStore(settings)
    editing = EditingService(store)
    results: dict[str, Any] = {}
    for scenario_text, fixture in fixtures.items():
        scenario = Scenario(scenario_text)
        project = editing.create_project(
            f"Generic {scenario.value}",
            Intent(
                text=_intent(scenario),
                scenario=scenario,
                target_duration_seconds=1.4 if scenario is Scenario.STILL_MOTION else None,
            ),
        )
        asset, probe = editing.ingest(project.id, fixture)
        derivatives = store.load_derivatives(project.id, asset.id)
        first_analysis = AnalysisService(store).analyze(project.id, asset.id)
        second_analysis = AnalysisService(store).analyze(project.id, asset.id)
        if [item.id for item in first_analysis] != [item.id for item in second_analysis]:
            raise RuntimeError(f"analysis is not idempotent for {scenario.value}")
        treatment = editing.propose(project.id)
        if not treatment.patch.evidence_refs:
            raise RuntimeError(f"treatment lacks evidence for {scenario.value}")
        version = editing.accept_treatment(project.id, treatment.id)
        patch_path = store.project_path(project.id) / "patches" / f"{version.applied_patch_id}.json"
        patch_payload = json.loads(patch_path.read_text(encoding="utf-8"))
        if not patch_payload.get("inverse_operations"):
            raise RuntimeError(f"accepted patch lacks inverse for {scenario.value}")
        profiles: dict[str, Any] = {}
        for profile in ("preview", "final"):
            render_receipt = RenderService(store).render(
                project.id, version.id, profile=profile
            )
            quality = QualityService(store).inspect(project.id, version.id, profile=profile)
            if quality.overall == "fail":
                raise RuntimeError(f"{scenario.value} {profile} failed QC")
            output = store.project_path(project.id) / render_receipt.output_paths[0]
            profiles[profile] = {
                "path": str(output),
                "sha256": sha256_file(output),
                "qc": quality.model_dump(mode="json"),
            }
        kdenlive = KdenliveExporter(store).export(project.id, version.id)
        otio = OTIOExporter(store).export(project.id, version.id)
        results[scenario.value] = {
            "project_id": project.id,
            "asset_id": asset.id,
            "source_sha256": asset.sha256,
            "probe_id": probe.id,
            "derived_media": derivatives.model_dump(mode="json"),
            "evidence_ids": [item.id for item in first_analysis],
            "treatment_id": treatment.id,
            "version_id": version.id,
            "profiles": profiles,
            "kdenlive": kdenlive.model_dump(mode="json"),
            "otio": otio.model_dump(mode="json"),
        }
    suite_receipt: dict[str, Any] = {
        "schema": "aoa_editing_generic_eval_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "ok": True,
        "fixture_hashes": {name: sha256_file(path) for name, path in fixtures.items()},
        "scenarios": results,
    }
    output = root / "generic-eval.json"
    output.write_text(
        json.dumps(suite_receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return suite_receipt


def _intent(scenario: Scenario) -> str:
    if scenario is Scenario.STILL_MOTION:
        return "Create a restrained layered parallax study from an abstract geometric still."
    if scenario is Scenario.SPEECH_CLEAN:
        return "Remove the synthetic dead-air interval while preserving audible phrase boundaries."
    return "Build a concise chronological memory rhythm from measured color-scene changes."
