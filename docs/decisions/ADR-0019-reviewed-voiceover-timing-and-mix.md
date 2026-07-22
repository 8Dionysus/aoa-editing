# ADR-0019: Reviewed voiceover timing and canonical narration mix

- Status: accepted
- Date: 2026-07-21
- Extends: ADR-0017 and ADR-0018

## Context

The `screen.workflow` path could plan a terminal recording before capture and
later bind script beats to reviewed screen ranges, but narration remained an
unresolved timing dependency. Treating a delivered voice recording as a final
post-render overlay would hide its provenance, make beat timing external to the
edit graph, and bypass ordinary patch, render, QC, and interchange behavior.
Conversely, transcript segments and silence detection are analyzer evidence;
neither may silently decide the final pacing.

## Decision

An operator first ingests the delivered voiceover as an immutable project
asset. `ScreenWorkflowVoiceoverTiming` then binds its asset id and hash to the
reviewed capture plan, delivery frame rate, complete narration duration, and
one contiguous cue for every stable script beat.

The timing command prefers the typed local
`local-ai://speech/transcript/default` evidence when usable. If transcription
is unavailable or cannot give every beat a segment, it falls back to measured
silence and authored beat-duration weights. Both routes produce only an
`alignment-draft`; provider failure stays localized and the draft names its
evidence refs and confidence.

An attributable editor must submit the complete cue list through a separate
review action. That action creates a new immutable `reviewed-for-edit` timing
revision linked to the draft. Cue ranges must start at frame zero, remain
contiguous, follow every plan beat in order, and cover the complete narration.

The post-capture `ScreenWorkflowEditSpec` may embed this reviewed timing. Every
screen binding duration must equal its corresponding narration cue. The
application verifies plan, script hash, voiceover asset hash, frame rate, beat
order, and total duration before writing human timing evidence. The scenario
then adds the immutable voice asset as a dedicated `Reviewed voiceover` audio
track with a `-16 LUFS` target. FFmpeg compilation, source lineage, reversible
Treatment approval, QC, and editable export remain the ordinary downstream
path.

Source audio is still an explicit per-screen-binding choice. Automatic music,
sidechain ducking, word-level surgery, and acceptance of machine alignment are
not implied by this decision.

## Consequences

- The delivered narration becomes the temporal spine of the screen edit rather
  than an opaque final mux step.
- Transcript and pause analysis accelerate timing without acquiring editorial
  authority.
- A silent or unavailable STT provider does not block manual timing review.
- Replacing the voice performance produces a new asset and timing revision;
  prior edit evidence remains inspectable.
- Capture frame rate must be chosen consistently with the reviewed timing
  contract until an explicit cross-rate retiming contract is added.

## Verification

Schema drift tests validate the new contract and its embedding in the edit
spec. Screen-workflow tests cover silence-weighted draft generation, contiguous
coverage, the immutable review gate, beat-duration validation, audio lineage,
Treatment creation, a real FFmpeg preview, and an AAC output stream. CLI, HTTP,
and allowlisted agent routes call the same application services.
