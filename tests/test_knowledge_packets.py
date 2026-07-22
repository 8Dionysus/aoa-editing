from __future__ import annotations

from pathlib import Path

from aoa_editing.domain.models import TechniquePacket


def test_checked_in_scenario_packets_are_versioned_candidates_and_source_neutral() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "editing-knowledge" / "scenarios").glob("*.json"))
    packets = [
        TechniquePacket.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    ]

    assert {packet.id for packet in packets} == {
        "scenario_speech_clean",
        "scenario_memory_montage",
        "scenario_still_motion",
        "scenario_screen_workflow",
    }
    assert all(packet.status == "candidate" for packet in packets)
    assert all(packet.application_history for packet in packets)
    serialized = "\n".join(packet.model_dump_json() for packet in packets)
    assert "18936de745a585ea5156d6db5fea17079eebdc25a61503564757b9ab4b79c57d" not in serialized
    assert "3345b5be11f03e63a489c6ede858542c6d39182a92622ae6cae3703ab7cb1e90" not in serialized


def test_screen_workflow_candidate_retains_reviewed_temporal_failures() -> None:
    root = Path(__file__).resolve().parents[1]
    packet = TechniquePacket.model_validate_json(
        (root / "editing-knowledge" / "scenarios" / "screen.workflow.json").read_text(
            encoding="utf-8"
        )
    )

    assert packet.revision == 3
    assert "source-contiguous-speed-tier" in packet.allowed_operations
    assert any("Working shimmer" in item for item in packet.failure_modes)
    history = {item.case_id: item for item in packet.application_history}
    assert history["builder-week-hard-cut-compression-review"].outcome == "fail"
    assert history["builder-week-continuous-speed-review"].outcome == "warn"
    assert history["builder-week-continuous-speed-review"].metrics[
        "operator_score_fraction"
    ] == 0.87
