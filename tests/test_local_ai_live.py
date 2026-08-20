from __future__ import annotations

import json
from pathlib import Path

from aoa_editing.evals.local_ai_live import (
    _capability_decisions,
    _error_rate,
    _model_tree_manifest,
    _normalized_characters,
    _normalized_words,
    _stack_registration_evidence,
)


def test_known_text_metrics_normalize_russian_punctuation() -> None:
    expected = "Сегодня мы проверяем звук и сохраняем результат."
    exact = "сегодня, мы проверяем звук и сохраняем результат"
    one_substitution = "Сегодня мы проверяем кадр и сохраняем результат."

    assert _normalized_words(expected) == _normalized_words(exact)
    assert _normalized_characters(expected) == _normalized_characters(exact)
    assert _error_rate(_normalized_words(expected), _normalized_words(exact)) == 0
    assert (
        _error_rate(
            _normalized_words(expected),
            _normalized_words(one_substitution),
        )
        == 1 / 7
    )


def test_model_tree_manifest_hashes_every_relative_file(tmp_path: Path) -> None:
    model = tmp_path / "model"
    (model / "nested").mkdir(parents=True)
    (model / "config.json").write_text('{"model":"fixture"}\n', encoding="utf-8")
    (model / "nested" / "weights.bin").write_bytes(b"fixture-weights")

    first = _model_tree_manifest(model)
    second = _model_tree_manifest(model)

    assert first == second
    assert first["file_count"] == 2
    assert first["size_bytes"] == sum(item["size_bytes"] for item in first["files"])
    assert [item["path"] for item in first["files"]] == [
        "config.json",
        "nested/weights.bin",
    ]
    assert len(first["tree_sha256"]) == 64
    assert all(len(item["sha256"]) == 64 for item in first["files"])


def test_stack_registration_requires_source_deploy_parity_and_allowlist(
    tmp_path: Path,
) -> None:
    model = tmp_path / "owner" / "model"
    model.mkdir(parents=True)
    unit_text = "\n".join(
        [
            "[Service]",
            f"ExecStart=/usr/local/libexec/abyss-dictation-server --preload-model-dir {model}",
            "",
        ]
    )
    source = tmp_path / "source" / "abyss-dictation-server.service"
    deployed = tmp_path / "deployed" / "abyss-dictation-server.service"
    source.parent.mkdir()
    deployed.parent.mkdir()
    source.write_text(unit_text, encoding="utf-8")
    deployed.write_text(unit_text, encoding="utf-8")
    allowlist = tmp_path / "managed-units.txt"
    allowlist.write_text("abyss-dictation-server.service\n", encoding="utf-8")

    evidence = _stack_registration_evidence(
        source,
        deployed,
        allowlist,
        model,
    )

    assert evidence["parity"] is True
    assert evidence["source_sha256"] == evidence["deployed_sha256"]
    assert evidence["unit"] == "abyss-dictation-server.service"


def test_phase15_decisions_cover_every_alias_without_new_model() -> None:
    decisions = _capability_decisions(
        {
            "capabilities": {
                "stt": {"status": "ready"},
                "embeddings": {"status": "ready"},
                "llm_text": {"status": "resident-running"},
            }
        },
        {
            "speech.transcript": "enabled",
            "speech.vad": "unbound",
            "speech.diarization": "unbound",
            "vision.describe": "unbound",
            "vision.regions": "unbound",
            "vision.segment": "unbound",
            "vision.embed": "unbound",
            "editorial.plan": "unbound",
            "editorial.critique": "unbound",
        },
        "openai/whisper-large-v3-turbo",
    )
    serialized = json.dumps(
        [item.model_dump(mode="json") for item in decisions],
        ensure_ascii=False,
    )

    assert len(decisions) == 9
    assert {item.capability_id for item in decisions} == {
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
    transcript = next(item for item in decisions if item.capability_id == "speech.transcript")
    assert transcript.action == "bind-existing"
    assert transcript.stack_registration == "live"
    assert transcript.alias_state == "enabled"
    assert all(item.new_model_required is False for item in decisions)
    assert "/srv/" not in serialized
