from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aoa_editing.agent import AgentProtocol, AgentRequest
from aoa_editing.api.app import create_app
from aoa_editing.application.service import EditingService
from aoa_editing.config import Settings
from aoa_editing.domain.models import (
    EvidenceRecord,
    Intent,
    Provenance,
    Scenario,
    ScreenWorkflowBeatInput,
    ScreenWorkflowEditSpec,
    ScreenWorkflowFocus,
    ScreenWorkflowLiveRegion,
    ScreenWorkflowLiveSignal,
    ScreenWorkflowPacingRole,
    ScreenWorkflowShotKind,
    ScreenWorkflowSourceBinding,
    ScreenWorkflowTemporalBehavior,
    ScreenWorkflowTemporalIntent,
    ScreenWorkflowVoiceoverCue,
    ScreenWorkflowVoiceoverTiming,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.quality.service import QualityService
from aoa_editing.render.service import RenderService


def _settings(tmp_path: Path) -> Settings:
    return Settings.for_home(tmp_path / "editing-home")


def _screen_recording(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _voiceover(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            (
                "aevalsrc=if(between(t\\,0.3\\,1.4)+between(t\\,2.1\\,3.5)\\,"
                "0.2*sin(2*PI*440*t)\\,0):s=48000:d=4"
            ),
            str(path),
        ],
        check=True,
    )


def _authored_beats() -> list[ScreenWorkflowBeatInput]:
    return [
        ScreenWorkflowBeatInput(
            narration="Сначала открываем сессию Codex и показываем задачу.",
            purpose="Establish the session and the task.",
            capture_action="Open the clean Codex terminal session with the task visible.",
            expected_result="The task and session context are readable.",
            shot_kind=ScreenWorkflowShotKind.ESTABLISH,
            estimated_duration_seconds=2.0,
        ),
        ScreenWorkflowBeatInput(
            narration="Затем показываем готовый diff и зелёный тест.",
            purpose="Prove the completed workflow.",
            capture_action="Open the final diff and run the bounded test command.",
            expected_result="The changed lines and passing test are visible.",
            shot_kind=ScreenWorkflowShotKind.RESULT,
            estimated_duration_seconds=2.0,
        ),
    ]


def test_script_only_produces_media_free_editorial_draft(tmp_path: Path) -> None:
    service = EditingService(ProjectStore(_settings(tmp_path)))
    plan = service.plan_screen_workflow(
        title="Codex workflow",
        script="Сначала показываем задачу. Затем запускаем тест и показываем результат.",
    )

    assert plan.status == "editorial-draft"
    assert plan.capture_ready is False
    assert plan.source_media_bound is False
    assert plan.template_revision == "terminal-workflow-temporal-integrity-v2"
    assert len(plan.templates) == len(ScreenWorkflowShotKind)
    assert "presenter or talking-head footage" in plan.forbidden_content
    assert any("Type each prompt once" in item for item in plan.capture_rules)
    assert any("continuous acceleration" in item for item in plan.framing_rules)
    assert plan.unresolved_questions


def test_reviewed_plan_and_raw_binding_create_renderable_reversible_treatment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "screen.mp4"
    _screen_recording(source)
    store = ProjectStore(_settings(tmp_path))
    service = EditingService(store)
    plan = service.plan_screen_workflow(
        title="Codex workflow",
        script="Показываем задачу, затем подтверждаем результат.",
        beats=_authored_beats(),
        reviewed_by="fixture-editor",
        reference_study_ids=["workflowstudy_fixture"],
    )
    project = service.create_project(
        "Screen workflow",
        Intent(text=plan.title, scenario=Scenario.SCREEN_WORKFLOW),
    )
    asset, _probe = service.ingest(project.id, source)
    spec = ScreenWorkflowEditSpec(
        plan_id=plan.id,
        script_sha256=plan.script_sha256,
        asset_id=asset.id,
        bindings=[
            ScreenWorkflowSourceBinding(
                beat_id=plan.beats[0].id,
                source_range={"start": 0, "duration": 45},
                timeline_duration_frames=45,
                shot_kind=ScreenWorkflowShotKind.ESTABLISH,
                focus_end=ScreenWorkflowFocus(center_x=0.5, center_y=0.5, scale=1.0),
                rationale="Use the clean opening wide state.",
            ),
            ScreenWorkflowSourceBinding(
                beat_id=plan.beats[1].id,
                source_range={"start": 45, "duration": 45},
                timeline_duration_frames=45,
                shot_kind=ScreenWorkflowShotKind.RESULT,
                focus_start=ScreenWorkflowFocus(center_x=0.5, center_y=0.5, scale=1.0),
                focus_end=ScreenWorkflowFocus(center_x=0.68, center_y=0.62, scale=1.8),
                caption_text="Тест пройден",
                rationale="Land on the lower-right result region and hold.",
            ),
        ],
        human_reviewed=True,
        reviewed_by="fixture-editor",
        review_note="Every source range and focus target was inspected.",
        provenance=Provenance(
            tool="fixture-editor",
            tool_version="1",
            deterministic=False,
        ),
    )

    treatment = service.propose_screen_workflow(project.id, plan=plan, edit_spec=spec)
    assert treatment.scenario is Scenario.SCREEN_WORKFLOW
    assert treatment.used_asset_ids == [asset.id]
    assert treatment.patch.evidence_refs
    tracks = treatment.patch.operations[2].value
    assert [item["kind"] for item in tracks] == ["video", "caption"]
    assert tracks[0]["clips"][1]["effects"][0]["type"] == "transform"
    version = service.accept_treatment(project.id, treatment.id)
    patch_path = store.project_path(project.id) / "patches" / f"{version.applied_patch_id}.json"
    assert json.loads(patch_path.read_text(encoding="utf-8"))["inverse_operations"]
    render = RenderService(store).render(project.id, version.id, profile="preview")
    assert (store.project_path(project.id) / render.output_paths[0]).is_file()
    quality = QualityService(store).inspect(project.id, version.id, profile="preview")
    assert quality.overall != "fail"


def test_reviewed_beat_can_use_source_contiguous_speed_ramp_segments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "screen.mp4"
    _screen_recording(source)
    store = ProjectStore(_settings(tmp_path))
    service = EditingService(store)
    plan = service.plan_screen_workflow(
        title="Continuous Codex work",
        script="Показываем ввод и непрерывную работу, затем результат.",
        beats=_authored_beats(),
        reviewed_by="fixture-editor",
    )
    project = service.create_project(
        "Continuous workflow",
        Intent(text=plan.title, scenario=Scenario.SCREEN_WORKFLOW),
    )
    asset, _ = service.ingest(project.id, source)
    wide = ScreenWorkflowFocus(center_x=0.5, center_y=0.5, scale=1.0)
    focused = ScreenWorkflowFocus(center_x=0.62, center_y=0.58, scale=1.5)
    result_focus = ScreenWorkflowFocus(center_x=0.66, center_y=0.62, scale=1.6)
    spec = ScreenWorkflowEditSpec(
        plan_id=plan.id,
        script_sha256=plan.script_sha256,
        asset_id=asset.id,
        bindings=[
            ScreenWorkflowSourceBinding(
                beat_id=plan.beats[0].id,
                segment_order=1,
                continuity_group="first-agent-process",
                source_range={"start": 0, "duration": 30},
                timeline_duration_frames=30,
                shot_kind=ScreenWorkflowShotKind.FOCUS,
                focus_start=wide,
                focus_end=focused,
                temporal_intent=ScreenWorkflowTemporalIntent(
                    pacing_role=ScreenWorkflowPacingRole.PROMPT_ENTRY,
                ),
                rationale="Show the prompt once at real-time speed.",
            ),
            ScreenWorkflowSourceBinding(
                beat_id=plan.beats[0].id,
                segment_order=2,
                continuity_group="first-agent-process",
                source_range={"start": 30, "duration": 60},
                timeline_duration_frames=30,
                speed=2.0,
                shot_kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
                focus_start=focused,
                focus_end=result_focus,
                temporal_intent=ScreenWorkflowTemporalIntent(
                    pacing_role=ScreenWorkflowPacingRole.AGENT_PROGRESS,
                    live_regions=[
                        ScreenWorkflowLiveRegion(
                            id="working-shimmer",
                            region={"x": 0.0, "y": 0.78, "width": 0.45, "height": 0.08},
                            signal=ScreenWorkflowLiveSignal.PROGRESS_SHIMMER,
                            temporal_behavior=ScreenWorkflowTemporalBehavior.PERIODIC,
                            loopable=True,
                            loop_period_frames=30,
                            seam_strategy="phase-match",
                            rationale="Working text visibly shimmers while the agent runs.",
                        )
                    ],
                ),
                rationale="Accelerate continuously without omitting source frames.",
            ),
            ScreenWorkflowSourceBinding(
                beat_id=plan.beats[1].id,
                source_range={"start": 90, "duration": 30},
                timeline_duration_frames=30,
                shot_kind=ScreenWorkflowShotKind.RESULT,
                focus_start=result_focus,
                focus_end=result_focus,
                temporal_intent=ScreenWorkflowTemporalIntent(
                    pacing_role=ScreenWorkflowPacingRole.READABLE_RESULT,
                    hold_policy="natural-source",
                ),
                rationale="Keep the proof live and readable.",
            ),
        ],
        human_reviewed=True,
        reviewed_by="fixture-editor",
        review_note="Reviewed every source frame and temporal signal.",
        provenance=Provenance(tool="fixture-editor", tool_version="1", deterministic=False),
    )

    treatment = service.propose_screen_workflow(project.id, plan=plan, edit_spec=spec)
    tracks = treatment.patch.operations[2].value
    picture = tracks[0]["clips"]
    assert len(picture) == 3
    assert picture[1]["role"].endswith(":2:workflow.wait-compress")
    assert len(picture[1]["effects"][0]["scale"]) == 2
    evidence = next(
        item for item in store.list_evidence(project.id) if item.kind == "screen.workflow.binding"
    )
    checks = {item["id"]: item for item in evidence.payload["temporal_integrity_checks"]}
    assert checks["source-contiguous-pacing"]["measured"]["group_count"] == 1
    assert checks["prompt-entry-readable"]["measured"]["prompt_segment_count"] == 1
    assert checks["live-ui-hold-safety"]["measured"]["live_region_count"] == 1


def test_screen_workflow_rejects_unreviewed_capture_plan(tmp_path: Path) -> None:
    source = tmp_path / "screen.mp4"
    _screen_recording(source)
    store = ProjectStore(_settings(tmp_path))
    service = EditingService(store)
    plan = service.plan_screen_workflow(title="Draft", script="Показать задачу.")
    project = service.create_project(
        "Draft", Intent(text="Draft", scenario=Scenario.SCREEN_WORKFLOW)
    )
    asset, _ = service.ingest(project.id, source)
    spec = ScreenWorkflowEditSpec(
        plan_id=plan.id,
        script_sha256=plan.script_sha256,
        asset_id=asset.id,
        bindings=[
            ScreenWorkflowSourceBinding(
                beat_id=plan.beats[0].id,
                source_range={"start": 0, "duration": 30},
                timeline_duration_frames=30,
                shot_kind=ScreenWorkflowShotKind.ESTABLISH,
                rationale="Fixture binding.",
            )
        ],
        human_reviewed=True,
        reviewed_by="fixture-editor",
        review_note="Binding reviewed, plan not reviewed.",
        provenance=Provenance(tool="fixture-editor", tool_version="1", deterministic=False),
    )

    with pytest.raises(ValueError, match="reviewed for capture"):
        service.propose_screen_workflow(project.id, plan=plan, edit_spec=spec)


def test_reviewed_voiceover_drives_beat_timing_and_renders_as_narration(
    tmp_path: Path,
) -> None:
    screen_source = tmp_path / "screen.mp4"
    voice_source = tmp_path / "voiceover.wav"
    _screen_recording(screen_source)
    _voiceover(voice_source)
    store = ProjectStore(_settings(tmp_path))
    service = EditingService(store)
    plan = service.plan_screen_workflow(
        title="Voice-led Codex workflow",
        script="Показываем задачу, затем подтверждаем результат.",
        beats=_authored_beats(),
        reviewed_by="fixture-editor",
    )
    project = service.create_project(
        "Voice-led workflow",
        Intent(text=plan.title, scenario=Scenario.SCREEN_WORKFLOW),
    )
    voice_asset, _ = service.ingest(project.id, voice_source)
    draft = service.time_screen_workflow_voiceover(
        project.id,
        plan=plan,
        asset_id=voice_asset.id,
        transcribe=False,
    )
    assert draft.status == "alignment-draft"
    assert draft.draft_alignment_method == "silence-weighted"
    assert draft.cues[0].timeline_range.start == 0
    assert draft.cues[-1].timeline_range.end == draft.duration_frames
    assert all(
        left.timeline_range.end == right.timeline_range.start
        for left, right in zip(draft.cues, draft.cues[1:], strict=False)
    )

    reviewed = service.review_screen_workflow_voiceover(
        plan=plan,
        draft=draft,
        cues=draft.cues,
        reviewed_by="fixture-editor",
        review_note="Listened through and approved both narration boundaries.",
    )
    assert reviewed.status == "reviewed-for-edit"
    assert reviewed.edit_ready is True
    assert reviewed.supersedes_timing_id == draft.id
    assert all(item.alignment_basis == "human-reviewed" for item in reviewed.cues)

    screen_asset, _ = service.ingest(project.id, screen_source)
    source_cursor = 0
    bindings: list[ScreenWorkflowSourceBinding] = []
    for beat, cue in zip(plan.beats, reviewed.cues, strict=True):
        focus = ScreenWorkflowFocus(
            center_x=0.5 if beat.order == 1 else 0.68,
            center_y=0.5 if beat.order == 1 else 0.62,
            scale=1.0 if beat.order == 1 else 1.8,
        )
        if beat.order == 1:
            first_duration = cue.timeline_range.duration // 2
            second_duration = cue.timeline_range.duration - first_duration
            bindings.extend(
                [
                    ScreenWorkflowSourceBinding(
                        beat_id=beat.id,
                        segment_order=1,
                        continuity_group="voice-led-opening",
                        source_range={"start": source_cursor, "duration": first_duration},
                        timeline_duration_frames=first_duration,
                        shot_kind=beat.shot_kind,
                        focus_end=focus,
                        temporal_intent=ScreenWorkflowTemporalIntent(
                            pacing_role=ScreenWorkflowPacingRole.PROMPT_ENTRY,
                        ),
                        rationale="Keep prompt entry at real-time speed.",
                    ),
                    ScreenWorkflowSourceBinding(
                        beat_id=beat.id,
                        segment_order=2,
                        continuity_group="voice-led-opening",
                        source_range={
                            "start": source_cursor + first_duration,
                            "duration": second_duration,
                        },
                        timeline_duration_frames=second_duration,
                        shot_kind=beat.shot_kind,
                        focus_start=focus,
                        focus_end=focus,
                        temporal_intent=ScreenWorkflowTemporalIntent(
                            pacing_role=ScreenWorkflowPacingRole.READABLE_RESULT,
                            hold_policy="natural-source",
                        ),
                        rationale="Hold the completed prompt with live source.",
                    ),
                ]
            )
            source_cursor += cue.timeline_range.duration
            continue
        bindings.append(
            ScreenWorkflowSourceBinding(
                beat_id=beat.id,
                source_range={
                    "start": source_cursor,
                    "duration": cue.timeline_range.duration,
                },
                timeline_duration_frames=cue.timeline_range.duration,
                shot_kind=beat.shot_kind,
                focus_end=focus,
                rationale="Align the reviewed screen state to the narration cue.",
            )
        )
        source_cursor += cue.timeline_range.duration
    spec = ScreenWorkflowEditSpec(
        plan_id=plan.id,
        script_sha256=plan.script_sha256,
        asset_id=screen_asset.id,
        bindings=bindings,
        voiceover_timing=reviewed,
        human_reviewed=True,
        reviewed_by="fixture-editor",
        review_note="Screen states and narration timing were reviewed together.",
        provenance=Provenance(
            tool="fixture-editor",
            tool_version="1",
            deterministic=False,
        ),
    )
    treatment = service.propose_screen_workflow(project.id, plan=plan, edit_spec=spec)
    assert treatment.used_asset_ids == [screen_asset.id, voice_asset.id]
    tracks = treatment.patch.operations[2].value
    assert [item["name"] for item in tracks] == ["Screen workflow", "Reviewed voiceover"]
    assert len(tracks[0]["clips"]) == 3
    narration = tracks[1]["clips"][0]
    assert narration["role"].startswith("screen-workflow-voiceover:")
    assert narration["effects"][0]["normalize_lufs"] == -16.0

    version = service.accept_treatment(project.id, treatment.id)
    render = RenderService(store).render(project.id, version.id, profile="preview")
    output = store.project_path(project.id) / render.output_paths[0]
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(probe.stdout)["streams"][0]["codec_name"] == "aac"


def test_voiceover_timing_must_be_reviewed_before_treatment(tmp_path: Path) -> None:
    source = tmp_path / "voiceover.wav"
    _voiceover(source)
    store = ProjectStore(_settings(tmp_path))
    service = EditingService(store)
    plan = service.plan_screen_workflow(
        title="Voice review gate",
        script="Показать задачу. Показать результат.",
        beats=_authored_beats(),
        reviewed_by="fixture-editor",
    )
    project = service.create_project(
        "Voice review gate",
        Intent(text=plan.title, scenario=Scenario.SCREEN_WORKFLOW),
    )
    asset, _ = service.ingest(project.id, source)
    draft = service.time_screen_workflow_voiceover(
        project.id,
        plan=plan,
        asset_id=asset.id,
        transcribe=False,
    )
    payload = draft.model_dump(mode="json")
    payload["status"] = "reviewed-for-edit"
    payload["edit_ready"] = True
    payload["reviewed_by"] = "fixture-editor"
    payload["review_note"] = "Attempted review without an immutable draft link."
    with pytest.raises(ValueError, match="supersede"):
        ScreenWorkflowVoiceoverTiming.model_validate(payload)


def test_voiceover_timing_prefers_typed_transcript_segments(tmp_path: Path) -> None:
    source = tmp_path / "voiceover.wav"
    _voiceover(source)
    store = ProjectStore(_settings(tmp_path))
    service = EditingService(store)
    plan = service.plan_screen_workflow(
        title="Transcript timing",
        script="Показываем задачу, затем подтверждаем результат.",
        beats=_authored_beats(),
        reviewed_by="fixture-editor",
    )
    project = service.create_project(
        "Transcript timing",
        Intent(text=plan.title, scenario=Scenario.SCREEN_WORKFLOW),
    )
    asset, _ = service.ingest(project.id, source)
    store.save_evidence(
        EvidenceRecord(
            project_id=project.id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            kind="speech.transcript",
            payload={
                "text": "Сначала открываем сессию Codex. Затем показываем зелёный тест.",
                "language": "ru",
                "segments": [
                    {"start": 0.3, "end": 0.9, "text": "Сначала открываем сессию"},
                    {"start": 0.9, "end": 1.4, "text": "Codex и показываем задачу"},
                    {"start": 2.1, "end": 2.8, "text": "Затем показываем готовый diff"},
                    {"start": 2.8, "end": 3.5, "text": "и зелёный тест"},
                ],
            },
            provenance=Provenance(
                tool="fixture-transcript-provider",
                tool_version="1",
                deterministic=True,
            ),
        )
    )

    draft = service.time_screen_workflow_voiceover(
        project.id,
        plan=plan,
        asset_id=asset.id,
        transcribe=False,
    )

    assert draft.draft_alignment_method == "transcript-segments"
    assert all(item.spoken_range is not None for item in draft.cues)
    assert "Codex" in (draft.cues[0].transcript_text or "")
    assert "зелёный тест" in (draft.cues[1].transcript_text or "")


def test_api_and_agent_share_script_planning_service(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = TestClient(create_app(settings))
    api = client.post(
        "/api/workflows/screen/plan",
        json={
            "title": "API workflow",
            "script": "Показать задачу. Показать результат.",
        },
    )
    assert api.status_code == 201
    agent = AgentProtocol(settings).execute(
        AgentRequest(
            id="workflow",
            method="workflow.plan_script",
            params={
                "title": "API workflow",
                "script": "Показать задачу. Показать результат.",
            },
        )
    )
    assert agent["ok"] is True
    assert agent["result"]["script_sha256"] == api.json()["script_sha256"]
    assert agent["result"]["capture_ready"] is False


def test_api_and_agent_share_voiceover_review_service(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = EditingService(ProjectStore(settings))
    plan = service.plan_screen_workflow(
        title="Voiceover transport parity",
        script="Показать задачу. Показать результат.",
        beats=_authored_beats(),
        reviewed_by="fixture-editor",
    )
    cues = [
        ScreenWorkflowVoiceoverCue(
            order=1,
            beat_id=plan.beats[0].id,
            timeline_range={"start": 0, "duration": 30},
            confidence=0.4,
            alignment_basis="silence-weighted",
        ),
        ScreenWorkflowVoiceoverCue(
            order=2,
            beat_id=plan.beats[1].id,
            timeline_range={"start": 30, "duration": 30},
            confidence=0.4,
            alignment_basis="silence-weighted",
        ),
    ]
    draft = ScreenWorkflowVoiceoverTiming(
        plan_id=plan.id,
        script_sha256=plan.script_sha256,
        asset_id="asset_voiceover",
        source_sha256="a" * 64,
        source_duration_seconds=2.0,
        timeline_frame_rate={"numerator": 30, "denominator": 1},
        duration_frames=60,
        status="alignment-draft",
        edit_ready=False,
        draft_alignment_method="silence-weighted",
        cues=cues,
        analysis_evidence_refs=["evidence_silence"],
        provenance=Provenance(
            tool="fixture-timing",
            tool_version="1",
            deterministic=True,
        ),
    )
    request = {
        "plan": plan.model_dump(mode="json"),
        "draft": draft.model_dump(mode="json"),
        "cues": [item.model_dump(mode="json") for item in cues],
        "reviewed_by": "fixture-editor",
        "review_note": "Approved every narration cue through both transports.",
    }
    api = TestClient(create_app(settings)).post(
        "/api/workflows/screen/voiceover-review",
        json=request,
    )
    assert api.status_code == 201
    agent = AgentProtocol(settings).execute(
        AgentRequest(
            id="voiceover-review",
            method="workflow.review_voiceover",
            params=request,
        )
    )
    assert agent["ok"] is True
    assert api.json()["status"] == "reviewed-for-edit"
    assert agent["result"]["status"] == "reviewed-for-edit"
    assert api.json()["supersedes_timing_id"] == draft.id
    assert agent["result"]["supersedes_timing_id"] == draft.id


def test_edit_spec_source_duration_must_match_speed() -> None:
    with pytest.raises(ValueError, match="multiplied by speed"):
        ScreenWorkflowSourceBinding(
            beat_id="beat",
            source_range={"start": 0, "duration": 60},
            timeline_duration_frames=60,
            speed=2.0,
            shot_kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
            rationale="Compress a measured wait.",
        )


def test_edit_spec_supports_source_contiguous_agent_workflow_timelapse() -> None:
    binding = ScreenWorkflowSourceBinding(
        beat_id="wait",
        source_range={"start": 300, "duration": 1920},
        timeline_duration_frames=60,
        speed=32.0,
        shot_kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
        temporal_intent=ScreenWorkflowTemporalIntent(
            pacing_role=ScreenWorkflowPacingRole.MEASURED_WAIT,
            speed_override_reason="Reviewed duration target requires a brief 32x tier.",
        ),
        rationale="Retain every source frame while compressing a measured wait.",
    )

    assert binding.source_range.end == 2220
    assert binding.speed == 32.0


def test_prompt_entry_must_remain_real_time() -> None:
    with pytest.raises(ValueError, match="prompt entry must remain at real-time speed"):
        ScreenWorkflowSourceBinding(
            beat_id="prompt",
            source_range={"start": 0, "duration": 60},
            timeline_duration_frames=30,
            speed=2.0,
            shot_kind=ScreenWorkflowShotKind.FOCUS,
            temporal_intent=ScreenWorkflowTemporalIntent(
                pacing_role=ScreenWorkflowPacingRole.PROMPT_ENTRY,
            ),
            rationale="Invalid prompt compression fixture.",
        )


def test_speed_above_reviewed_workflow_threshold_requires_override() -> None:
    with pytest.raises(ValueError, match="above 24x"):
        ScreenWorkflowSourceBinding(
            beat_id="wait",
            source_range={"start": 0, "duration": 750},
            timeline_duration_frames=30,
            speed=25.0,
            shot_kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
            temporal_intent=ScreenWorkflowTemporalIntent(
                pacing_role=ScreenWorkflowPacingRole.MEASURED_WAIT,
            ),
            rationale="Unreviewed perceptual cut risk.",
        )


def test_monotonic_terminal_timer_cannot_be_claimed_as_loopable() -> None:
    with pytest.raises(ValueError, match="only a reviewed periodic live region"):
        ScreenWorkflowLiveRegion(
            id="working-timer",
            region={"x": 0.0, "y": 0.80, "width": 0.45, "height": 0.08},
            signal=ScreenWorkflowLiveSignal.MONOTONIC_TIMER,
            temporal_behavior=ScreenWorkflowTemporalBehavior.MONOTONIC,
            loopable=True,
            rationale="The visible seconds counter must continue increasing.",
        )


def test_periodic_live_region_requires_reviewed_loop_period_and_seam() -> None:
    with pytest.raises(ValueError, match="requires a reviewed period and seam strategy"):
        ScreenWorkflowLiveRegion(
            id="working-shimmer",
            region={"x": 0.0, "y": 0.80, "width": 0.45, "height": 0.08},
            signal=ScreenWorkflowLiveSignal.PROGRESS_SHIMMER,
            temporal_behavior=ScreenWorkflowTemporalBehavior.PERIODIC,
            loopable=True,
            rationale="A periodic claim without an inspected loop cycle is insufficient.",
        )


def test_source_contiguous_speed_tiers_require_no_gaps_and_a_gradual_ramp() -> None:
    focus = ScreenWorkflowFocus(center_x=0.5, center_y=0.5, scale=1.4)
    first = ScreenWorkflowSourceBinding(
        beat_id="work",
        segment_order=1,
        continuity_group="agent-work",
        source_range={"start": 0, "duration": 60},
        timeline_duration_frames=60,
        speed=1.0,
        shot_kind=ScreenWorkflowShotKind.FOLLOW,
        focus_start=focus,
        focus_end=focus,
        rationale="Readable action anchor.",
    )
    gap = ScreenWorkflowSourceBinding(
        beat_id="work",
        segment_order=2,
        continuity_group="agent-work",
        source_range={"start": 61, "duration": 180},
        timeline_duration_frames=60,
        speed=3.0,
        shot_kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
        focus_start=focus,
        focus_end=focus,
        rationale="Invalid one-frame omission.",
    )
    with pytest.raises(ValueError, match="must not omit source frames"):
        ScreenWorkflowEditSpec(
            plan_id="plan",
            script_sha256="0" * 64,
            asset_id="asset",
            bindings=[first, gap],
            human_reviewed=True,
            reviewed_by="fixture-editor",
            review_note="Continuity fixture.",
            provenance=Provenance(tool="fixture", tool_version="1", deterministic=True),
        )

    camera_jump = ScreenWorkflowSourceBinding(
        beat_id="work",
        segment_order=2,
        continuity_group="agent-work",
        source_range={"start": 60, "duration": 180},
        timeline_duration_frames=60,
        speed=3.0,
        shot_kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
        focus_start=ScreenWorkflowFocus(center_x=0.6, center_y=0.5, scale=1.4),
        focus_end=focus,
        rationale="Invalid camera jump inside one source-contiguous process.",
    )
    with pytest.raises(ValueError, match="preserve camera continuity"):
        ScreenWorkflowEditSpec(
            plan_id="plan",
            script_sha256="0" * 64,
            asset_id="asset",
            bindings=[first, camera_jump],
            human_reviewed=True,
            reviewed_by="fixture-editor",
            review_note="Camera continuity fixture.",
            provenance=Provenance(tool="fixture", tool_version="1", deterministic=True),
        )

    abrupt = ScreenWorkflowSourceBinding(
        beat_id="work",
        segment_order=2,
        continuity_group="agent-work",
        source_range={"start": 60, "duration": 360},
        timeline_duration_frames=60,
        speed=6.0,
        shot_kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
        focus_start=focus,
        focus_end=focus,
        rationale="Invalid abrupt 1x-to-6x tier.",
    )
    with pytest.raises(ValueError, match="ramp through intermediate rates"):
        ScreenWorkflowEditSpec(
            plan_id="plan",
            script_sha256="0" * 64,
            asset_id="asset",
            bindings=[first, abrupt],
            human_reviewed=True,
            reviewed_by="fixture-editor",
            review_note="Ramp fixture.",
            provenance=Provenance(tool="fixture", tool_version="1", deterministic=True),
        )


def test_cli_contract_can_round_trip_plan_json(tmp_path: Path) -> None:
    service = EditingService(ProjectStore(_settings(tmp_path)))
    plan = service.plan_screen_workflow(title="Round trip", script="Показать терминал.")
    path = tmp_path / "plan.json"
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["script_sha256"] == plan.script_sha256
    assert payload["source_media_bound"] is False
