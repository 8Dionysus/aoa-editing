# User guide

## Mental model

AoA Editing separates five things that ordinary generator demos often blur:

1. **Source** is an immutable local copy identified by SHA-256.
2. **Evidence** is what tools measured. It cannot edit a timeline.
3. **Treatment** is a proposed editorial plan with rationale and alternatives.
4. **Patch/version** is the accepted, reversible change to the canonical graph.
5. **Render/export** is derived output that can be regenerated.

## Workbench path

1. Open AoA Editing from the desktop launcher or run
   `./scripts/aoa-editing serve --open`.
2. Create a project and choose one scenario.
3. Add local media. The source is copied, hashed, re-hashed, and made read-only.
4. Run analysis for every source. Inspect payload and provenance in **Evidence**.
5. Ask for treatments. Creative scenarios return three executable options; read
   structure, sources, strengths, compromises, uncertainty, risks, style,
   rationale, evidence links, and alternatives.
6. Accept it, or edit the JSON patch in the advanced section and apply that as a
   separate version.
7. Inspect timeline tracks and use **Versions / A·B** to compare rendered
   versions. Revert creates another version; it never destroys history.
8. Render a preview, run QC, then render final.
9. Export Kdenlive/MLT or OTIO. Read the compatibility report before assuming
   exact effect parity.

Brief edits create immutable revisions. Style Memory is a separate form and is
written only after the user explicitly confirms one preference/value. In the
timeline, hover a clip to pin/unpin, include/exclude, or render just its frame
range. Natural-language commands show a typed diff first; confirming that diff
creates a new version. Evidence search covers analyzer records and timecoded
transcript segments. Source proxies provide local video/audio review.

## Scenarios

### `speech.clean`

Measures silence and loudness, keeps conservative boundary padding, cuts video
and audio together, adds micro-fades, and normalizes output speech. Optional host
Whisper transcription is evidence, not the cut authority. Reviewed segments
compile as a caption track; optional speaker labels are retained. Gain
keyframes provide a deterministic ducking-envelope primitive.

### `memory.montage`

Measures visual scene changes and builds concise chronological beats with
synchronized live sound and three alternative structures. The first
prototype deliberately avoids pretending that scene detection understands
meaning; future semantic ranking belongs behind a planner port.

### `still.motion`

Measures image statistics and creates background/midground/foreground depth
proxies. The masks are explicitly labeled non-semantic. Different keyframed
motion per layer creates parallax while the source remains unchanged.

### `screen.workflow`

This scenario starts before media exists. Run `workflow plan-script` with the
script to create a media-free editorial draft. For a capture-ready plan, supply
an authored `ScreenWorkflowBeatInput` list that states the narration, purpose,
exact capture action, expected proof, shot kind, and estimated duration for
every beat, plus `--reviewed-by`.

Record the terminal/Codex/IDE/browser workflow from that checklist, ingest the
raw video normally, then author a `ScreenWorkflowEditSpec`. It must bind every
beat id to one or more reviewed source ranges, timeline durations, semantic
focus targets, optional 0.5x–64.0x speed changes, captions, and source-audio
decisions. Use ordered segments in one continuity group when the same agent
process needs a speed ramp. Adjacent tiers must preserve every source frame,
the camera endpoint, and intermediate rates. Keep prompt entry at 1x. As a
candidate starting point, use roughly 3x–6x for meaningful actions, 10x–12x for
low-motion settling, and 18x–24x for measured low-information waits; a reviewed
reason is required above 24x because higher rates often read as cuts.

Mark visible cursor blink, `Working` shimmer, elapsed timers, spinners, and
mascot animation as live regions. A readable hold must use real source at 1x or
gain time by rebalancing neighboring speed tiers. The executable path does not
admit a still-frame or full-frame loop for a live terminal. Run
`treatment workflow --plan ... --edit-spec ...`; the resulting virtual-camera
moves, cuts, captions, patch, version, render, QC, and exports are ordinary
project data. Presenter and talking-head footage is explicitly outside this
scenario.

When a finished voice recording is supplied, ingest it before the screen
recording and run `workflow time-voiceover`. The command uses the typed local
transcript provider when available and otherwise falls back to measured pauses;
either result is only a draft. Listen through it, correct the complete cue JSON,
and run `workflow review-voiceover`. Embed that reviewed timing in the later
`ScreenWorkflowEditSpec`; every visual binding must use the corresponding cue
duration, or several segments for the same beat must sum to it. The accepted
Treatment then contains a dedicated `Reviewed
voiceover` track normalized toward -16 LUFS. Machine alignment never approves
its own timing.

### `reference.reconstruct`

This gated evaluation scenario is intentionally absent from the normal
one-click proposal list because it requires a previously frozen observation
spec. The CLI or evaluation runner submits that spec through the same treatment,
patch, version, render, QC, and export services as the three ordinary scenarios.
Its camera uses explicit contain/cover fit, normalized placement, rotation, and
multi-keyframe easing; reference pixels and audio are never project assets.

## Reference understanding and correction

After a passing Reconstruction Study v2 is attached, open **Референс**. The
candidate and sealed reference players share time, seeking, play/pause, and
speed. Adjust overlay opacity or inspect the aligned-difference view. The frame
scrubber joins all-frame transform values, velocity, acceleration, jerk,
confidence, uncertainty, the nearest diagnostic overlay, and optical-flow
residual. Phase markers and alternative explanations show what the analyzer can
and cannot identify; an unidentifiable pivot is shown as such rather than given
a fabricated coordinate.

Enter an editorial correction or edit a tangent/phase control. The first action
creates an immutable proposal only. Read each typed operation, affected range,
matrix delta, endpoint result, and blocker. Explicitly approve or reject every
item and write a rationale. Only then can approved items create a new reversible
version and separate human-correction evidence. Rejected items remain in the
review ledger. Use the affected-range point render and synchronized version A/B
view before final QC or export.

## Recovery

Canonical JSON is written atomically. Incomplete render files use a `.partial`
name and are removed on failure. Restart the application and re-run the action;
the disposable index is rebuilt, ingest and analysis reuse matching
hashes/evidence, missing derivatives are regenerated, and immutable versions are
never overwritten with different content. See `docs/troubleshooting.md`.
