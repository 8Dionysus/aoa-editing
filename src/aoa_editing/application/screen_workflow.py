"""Script-first planning for terminal and agent-workflow screen recordings."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from aoa_editing.domain.models import (
    CheckResult,
    Provenance,
    ScreenWorkflowBeat,
    ScreenWorkflowBeatInput,
    ScreenWorkflowPlan,
    ScreenWorkflowShotKind,
    ScreenWorkflowShotTemplate,
    ScreenWorkflowSourceBinding,
)


def terminal_workflow_templates() -> list[ScreenWorkflowShotTemplate]:
    """Return the complete candidate catalog extracted from the reviewed corpus."""

    return [
        ScreenWorkflowShotTemplate(
            kind=ScreenWorkflowShotKind.ESTABLISH,
            purpose="Orient the viewer in the full terminal or workspace before detail.",
            default_scale=1.0,
            camera_transition_seconds=0.0,
            minimum_hold_seconds=2.0,
            camera_policy="Hold the complete screen; do not chase cursor motion.",
            cut_policy="hard-cut",
        ),
        ScreenWorkflowShotTemplate(
            kind=ScreenWorkflowShotKind.FOCUS,
            purpose="Move once to the prompt, command, diff, or region that carries the beat.",
            default_scale=1.65,
            camera_transition_seconds=0.45,
            minimum_hold_seconds=2.0,
            camera_policy="Complete the move, then hold while the viewer reads or typing occurs.",
            cut_policy="continuous-camera",
        ),
        ScreenWorkflowShotTemplate(
            kind=ScreenWorkflowShotKind.FOLLOW,
            purpose="Reframe between two semantically connected regions on the same screen.",
            default_scale=1.65,
            camera_transition_seconds=0.50,
            minimum_hold_seconds=2.0,
            camera_policy="Follow a state transition, never every cursor displacement.",
            cut_policy="continuous-camera",
        ),
        ScreenWorkflowShotTemplate(
            kind=ScreenWorkflowShotKind.RESULT,
            purpose="Give the produced answer, diff, test, or artifact an explicit proof beat.",
            default_scale=1.80,
            camera_transition_seconds=0.40,
            minimum_hold_seconds=2.5,
            camera_policy="Land on the proof and hold long enough to verify it.",
            cut_policy="continuous-camera",
        ),
        ScreenWorkflowShotTemplate(
            kind=ScreenWorkflowShotKind.RESET,
            purpose="Return from detail to a wide view before a new workflow section.",
            default_scale=1.0,
            camera_transition_seconds=0.45,
            minimum_hold_seconds=1.5,
            camera_policy="Use one deliberate release to the full workspace.",
            cut_policy="continuous-camera",
        ),
        ScreenWorkflowShotTemplate(
            kind=ScreenWorkflowShotKind.STEP_CUT,
            purpose="Separate distinct workflow states without simulating spatial continuity.",
            default_scale=1.55,
            camera_transition_seconds=0.0,
            minimum_hold_seconds=1.5,
            camera_policy="Hard-cut only after preserving action and result anchors.",
            cut_policy="hard-cut",
        ),
        ScreenWorkflowShotTemplate(
            kind=ScreenWorkflowShotKind.WAIT_COMPRESS,
            purpose="Continuously accelerate low-density waiting between visible anchors.",
            default_scale=1.55,
            camera_transition_seconds=0.40,
            minimum_hold_seconds=1.0,
            camera_policy=(
                "Preserve prompt, action, and result; use source-contiguous speed tiers "
                "instead of hiding the wait behind cuts."
            ),
            cut_policy="continuous-camera",
        ),
    ]


def build_screen_workflow_plan(
    *,
    title: str,
    script: str,
    beats: Sequence[ScreenWorkflowBeatInput] | None = None,
    reviewed_by: str | None = None,
    reference_study_ids: Sequence[str] = (),
) -> ScreenWorkflowPlan:
    """Turn a script into a capture plan, with optional authored semantic beats."""

    normalized_script = script.strip()
    if not normalized_script:
        raise ValueError("screen workflow script is empty")
    authored = list(beats) if beats is not None else _draft_beats(normalized_script)
    if not authored:
        raise ValueError("screen workflow script produced no capture beats")
    reviewed = beats is not None and reviewed_by is not None
    unresolved = [] if reviewed else [
        "Replace generic capture actions with the exact terminal, Codex, IDE, or browser states.",
        "Confirm every camera target after the real workspace layout is known.",
        "Approve the beat order and capture checklist before recording.",
    ]
    return ScreenWorkflowPlan(
        title=title.strip(),
        script=normalized_script,
        script_sha256=hashlib.sha256(normalized_script.encode("utf-8")).hexdigest(),
        status="reviewed-for-capture" if reviewed else "editorial-draft",
        capture_ready=reviewed,
        reviewed_by=reviewed_by if reviewed else None,
        templates=terminal_workflow_templates(),
        beats=[
            ScreenWorkflowBeat(**item.model_dump(), order=index)
            for index, item in enumerate(authored, start=1)
        ],
        capture_rules=[
            "Begin from the already-open clean terminal; exclude desktop or workspace switching.",
            "Capture the terminal fullscreen or borderless at the delivery aspect ratio.",
            "Record a clean wide state before and after each risky interaction.",
            "Capture prompt, action, and result anchors even when waiting will be removed.",
            (
                "Type each prompt once at human speed and keep the completed prompt "
                "visible before Enter."
            ),
            "Hold readable terminal or UI states for at least two seconds.",
            (
                "Keep the cursor still during read holds, but record live indicators such as "
                "Working shimmer, timers, spinners, and mascot animation."
            ),
            "Record terminal text large and sharp enough for a 1.5x to 2.0x crop.",
            (
                "Hide unrelated notifications and services without disabling the workflow "
                "dependency being demonstrated."
            ),
        ],
        framing_rules=[
            "Use full-bleed screen recording as the primary visual plane.",
            "Move the virtual camera only when the semantic state changes.",
            "Prefer smooth 0.3s to 0.7s camera moves within one screen context.",
            "Prefer hard cuts between distinct application or workflow states.",
            "Prefer continuous acceleration to cuts while one agent process remains in progress.",
            "Do not use a still hold when any visible terminal or UI region remains live.",
            "Apply subtitles after camera transforms inside the final-frame safe area.",
        ],
        forbidden_content=[
            "presenter or talking-head footage",
            "human reflected in the screen or visible through a webcam tile",
            "reference media used as a project asset or render input",
            "unreviewed secrets, tokens, personal paths, notifications, or private data",
        ],
        unresolved_questions=unresolved,
        reference_study_ids=list(reference_study_ids),
        provenance=Provenance(
            tool="aoa-editing-screen-workflow-planner",
            tool_version="0.1.0",
            parameters={
                "planner": "script-to-capture-plan-v1",
                "beat_authority": "editor-authored" if beats is not None else "deterministic-draft",
                "reviewed": reviewed,
                "reference_style_authority": "candidate-only",
            },
            deterministic=beats is None,
        ),
    )


def screen_workflow_temporal_checks(
    bindings: Sequence[ScreenWorkflowSourceBinding],
) -> list[CheckResult]:
    """Summarize model-enforced temporal integrity for persisted human evidence."""

    prompt_segments = [
        item
        for item in bindings
        if item.temporal_intent.pacing_role.value == "prompt-entry"
    ]
    continuity_groups = {item.continuity_group for item in bindings if item.continuity_group}
    live_regions = [
        region for item in bindings for region in item.temporal_intent.live_regions
    ]
    protected_static_regions = [
        region
        for item in bindings
        for region in item.temporal_intent.protected_static_regions
    ]
    high_speed = [item for item in bindings if item.speed > 24.0]
    natural_holds = [
        item for item in bindings if item.temporal_intent.hold_policy == "natural-source"
    ]
    rebalanced_holds = [
        item
        for item in bindings
        if item.temporal_intent.hold_policy == "rebalance-adjacent-speeds"
    ]
    return [
        CheckResult(
            id="source-contiguous-pacing",
            status="pass",
            summary="continuity groups preserve every source frame and camera endpoint",
            measured={"group_count": len(continuity_groups)},
        ),
        CheckResult(
            id="prompt-entry-readable",
            status="pass",
            summary="declared prompt entry remains at real-time speed",
            measured={"prompt_segment_count": len(prompt_segments)},
        ),
        CheckResult(
            id="live-ui-hold-safety",
            status="pass",
            summary="holds use live source or adjacent-speed rebalancing, never a frozen frame",
            measured={
                "live_region_count": len(live_regions),
                "protected_static_region_count": len(protected_static_regions),
                "natural_source_hold_count": len(natural_holds),
                "rebalanced_hold_count": len(rebalanced_holds),
            },
        ),
        CheckResult(
            id="perceptual-speed-risk",
            status="warn" if high_speed else "pass",
            summary=(
                "reviewed speed override exceeds the 24x workflow recommendation"
                if high_speed
                else "screen-workflow speed remains within the reviewed 24x recommendation"
            ),
            measured={
                "maximum_speed": max((item.speed for item in bindings), default=1.0),
                "override_count": len(high_speed),
            },
        ),
    ]


def _draft_beats(script: str) -> list[ScreenWorkflowBeatInput]:
    paragraphs = [
        re.sub(r"\s+", " ", item).strip()
        for item in re.split(
            r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9\u0410-\u042f\u0401])", script
        )
        if item.strip() and not item.lstrip().startswith("#")
    ]
    if not paragraphs:
        paragraphs = [re.sub(r"\s+", " ", script).strip()]
    return [
        ScreenWorkflowBeatInput(
            narration=paragraph,
            purpose=_draft_purpose(index, len(paragraphs)),
            capture_action=(
                "Record the exact on-screen action that proves this narration beat; "
                "replace this draft instruction during editorial review."
            ),
            expected_result=(
                "A stable, readable terminal or workflow state that visibly confirms the beat."
            ),
            shot_kind=_draft_shot_kind(paragraph, index, len(paragraphs)),
            estimated_duration_seconds=min(10.0, max(2.0, len(paragraph.split()) / 2.5)),
            operator_notes=["Deterministic draft: exact command and camera target are unresolved."],
        )
        for index, paragraph in enumerate(paragraphs)
    ]


def _draft_purpose(index: int, count: int) -> str:
    if index == 0:
        return "Establish the workspace and the problem being solved."
    if index == count - 1:
        return "Show the final proof or result of the workflow."
    return "Advance one visible, verifiable step of the workflow."


def _draft_shot_kind(text: str, index: int, count: int) -> ScreenWorkflowShotKind:
    lowered = text.casefold()
    if index == 0:
        return ScreenWorkflowShotKind.ESTABLISH
    if index == count - 1 or any(
        token in lowered
        for token in ("result", "done", "passed", "готов", "результат", "получил")
    ):
        return ScreenWorkflowShotKind.RESULT
    if any(
        token in lowered
        for token in ("wait", "build", "install", "test", "жд", "сборк", "установ", "тест")
    ):
        return ScreenWorkflowShotKind.WAIT_COMPRESS
    return ScreenWorkflowShotKind.FOCUS if index % 2 else ScreenWorkflowShotKind.STEP_CUT
