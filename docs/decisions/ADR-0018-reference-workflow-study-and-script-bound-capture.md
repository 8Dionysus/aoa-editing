# ADR-0018: Gated workflow study and script-bound screen capture

- Status: accepted
- Date: 2026-07-20
- Extends: ADR-0004 and ADR-0006

## Context

Modern terminal and agent-workflow videos are often planned before the final
screen recording exists. The supplied references can inform pacing, framing,
and camera grammar, but they also contain presenter sections that are outside
the requested visual language. Treating whole reference files as sources or
turning observed preferences directly into Style Memory would violate the
reference-isolation and explicit-consent boundaries.

A script is not yet source evidence, while a raw screen recording does not by
itself identify the semantic action, proof, or camera target for each narrated
beat. Combining those stages into one automatic planner would hide unresolved
editorial decisions.

## Decision

Reference Workflow Study v1 is a readiness-gated evaluation family. Its
untracked plan binds external reference paths to expected hashes and exact,
non-overlapping, manually reviewed ranges. Only terminal, IDE,
browser-workflow, workflow-UI, and diagram ranges with
`human_present=false` may be included. The evaluator produces path-free source
facts, deterministic frame-activity measurements, and contact sheets from
included ranges only beneath `var/evals`. It never copies source videos and
marks every result ineligible as a project asset, render input, Style Profile,
or canon. Generalized observations remain candidate rules tied to range ids.

`screen.workflow` then uses two separate product contracts:

1. `ScreenWorkflowPlan` binds a script hash to stable semantic beats, capture
   actions, expected proofs, terminal camera templates, privacy checks, and
   framing rules. It has no source media. A deterministic script split is only
   an editorial draft; capture-ready status requires authored beats and an
   attributable reviewer.
2. After recording and immutable ingest, `ScreenWorkflowEditSpec` binds every
   beat to one human-reviewed source range, output duration, semantic focus
   target, speed, caption, and source-audio decision. The binding is persisted
   as human evidence and produces an ordinary reversible Treatment.

The executable baseline uses legacy `TransformEffect` because its FFmpeg path
is already video-safe. Motion Language v2 remains excluded from screen video
until its compositor supports moving-image sources through a dedicated generic
evaluation. Smooth moves are used inside one screen context; hard cuts separate
distinct states; the camera holds during reading and typing; wait compression
must retain prompt, action, and result anchors. Presenter and talking-head
footage is forbidden.

CLI, HTTP, and allowlisted agent routes call the same application service.
Neither a script draft, a reference observation, nor an AI/provider result can
approve an edit.

## Consequences

- A scenario can arrive before footage without inventing source timings.
- Capture can be repeated from an inspectable checklist, and later raw media can
  be replaced by authoring a new reviewed binding rather than rewriting intent.
- Reference-derived preferences stay local, attributable, and noncanonical.
- Automatic human detection, transcript alignment, music beats, floating screen
  stages, and video-safe Motion v2 remain explicit future eval slices.
- The accepted Treatment retains the existing patch/version/QC/interchange and
  source-lineage laws.

## Verification

Schema validation covers both workflow contracts and the reference study plan
and receipt. Tests exercise script-only drafting, authored capture readiness,
full beat binding, refusal of unreviewed plans, transform/caption compilation,
accept/revert data, a real preview render and QC, CLI/API/agent application
parity, revision gating, hash checks, human-free inclusion, eval-only output,
path redaction, derived contact sheets, and absence of copied reference video.
