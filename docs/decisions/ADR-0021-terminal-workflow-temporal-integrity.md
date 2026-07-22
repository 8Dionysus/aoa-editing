# ADR-0021: Temporal integrity for terminal and agent workflows

- Status: accepted
- Date: 2026-07-21
- Extends: ADR-0018 and ADR-0019

## Context

An operator-reviewed Builder Week edit exposed a gap between technical timeline
continuity and perceived workflow continuity. Early variants hid work behind
cuts, clipped focused text, repeated or flashed prompt entry, and used frozen
frames to satisfy narration timing. A later source-contiguous version was rated
substantially stronger, but a frozen terminal hold remained visible because the
`Working` label shimmer, elapsed timer, cursor, and mascot were expected to
move. Very high constant speeds also remained perceptually cut-like even when
no source frame was omitted.

These are reviewed observations from one real human-free Codex workflow, not a
universal style law. The durable product lesson is narrower: a screen-workflow
edit must be able to state and validate its temporal intent instead of treating
all quiet pixels, waits, holds, and speed changes as interchangeable.

## Decision

`ScreenWorkflowSourceBinding` may contain multiple ordered segments for one
semantic beat. A named continuity group asserts that adjacent segments preserve
every source frame and the camera endpoint. Speed tiers in that group may change
by at most 4x at one boundary, forcing intermediate rates for a perceptually
gradual ramp. The generic edit graph still supports up to 64x, but a
`screen.workflow` binding above the reviewed 24x recommendation requires an
explicit risk override.

Each segment carries a typed temporal intent:

- prompt entry, semantic action, agent progress, readable result, or measured
  wait;
- no hold, a natural-source hold, or an adjacent-speed rebalance;
- reviewed normalized live regions such as cursor blink, progress shimmer,
  monotonic timer, spinner, mascot animation, or directional motion.

Prompt entry and natural-source holds must remain at 1x. A live region may be
marked loopable only when its temporal behavior was reviewed as periodic. A
monotonic timer, directional motion, or irregular animation therefore cannot be
made loopable by assertion. Still-frame and full-frame-loop hold modes are not
part of the executable contract.

The capture plan now starts from an already-open clean terminal, records every
prompt once, retains the completed prompt before the response, records text with
enough pixel density for the intended crop, and calls out live terminal
indicators. `workflow.wait-compress` uses continuous camera behavior and
source-contiguous acceleration; hard cuts remain for genuinely distinct
workflow states.

The human binding evidence persists temporal-integrity checks alongside the
source ranges. Voiceover timing remains beat-based: when one beat has several
pacing segments, their output durations must sum to the one reviewed narration
cue.

## Candidate findings, not yet executable

A correct synthetic hold probably needs region-aware temporal composition: keep
completed prompt text stable, preserve periodic live regions with phase-matched
seams, and keep elapsed timers monotonic or outside the loop. Full-frame
ping-pong is not acceptable for spinners or directional motion. This remains a
candidate until a generic compositor and evaluator can detect frozen live UI,
timer reversal/repetition, and loop seams.

The same reviewed run also suggests a resource-aware final-render lane: render
bounded source-contiguous video ranges when the full graph exceeds admission,
join the video without visual gaps, and apply/normalize the continuous narration
once. Existing segment rendering is evidence that the decomposition is viable;
automatic admission planning and final recombination remain future work.

## Consequences

- The edit spec can express speed ramps without inventing extra script beats.
- Missing source frames and discontinuous camera targets fail before Treatment
  creation.
- Prompt visibility and live terminal motion become reviewable evidence rather
  than informal editor memory.
- Existing one-binding-per-beat specs remain valid through additive defaults.
- A high-speed override records risk; it does not claim the result will read
  smoothly.
- No current product path claims to synthesize a correct temporal micro-loop.

## Verification

Contract tests cover real-time prompt entry, high-speed override, periodic versus
monotonic loop claims, contiguous segment order, source and camera continuity,
gradual speed-tier ratios, multi-segment Treatment construction, persisted
temporal checks, voiceover duration aggregation, schema drift, and the reviewed
candidate application history. The ordinary screen-workflow integration test
continues through a real FFmpeg preview and QC.
