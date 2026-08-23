# Video Anatomy

## Contract

Video Anatomy turns one immutable video asset into measured, hash-bound temporal
evidence. It is not an edit and it is not a free-form AI summary:

```text
asset -> sampling plan -> structure/frame/audio/visual/motion evidence
      -> VideoAnatomy -> editorial/reconstruction proposal
      -> human review -> reversible canonical patch
```

The analyzer never changes a timeline. `EditorialStructureProposal` and
`ReferenceReconstructionProposal` remain non-canonical until an attributable
review approves the exact proposal hash. An approved reconstruction uses the
ordinary patch/version service and retains an inverse patch. Human corrections
append superseding evidence; measured history is never rewritten.

The implementation is owned entirely by this repository. It has no Claude,
Codex, skill-host, Abyss, or Tree of Sophia runtime dependency. The
`claude-video` project was studied only as related work for progressive
sampling; no implementation or package is copied or installed.

## Profiles

| Profile | Decode strategy | Added evidence |
| --- | --- | --- |
| `quick` | bounded keyframe, uniform, endpoint and pinned random access | technical map, approximate shots, deterministic audio; sparse transition shapes remain `unknown` |
| `structural` | all requested frames | contiguous shots, dense boundary windows, representative/boundary frames, coverage, deduplication, contact sheet, silence/loudness/onsets |
| `semantic` | structural plus retained-frame batch | neutral `vision.describe` observations; missing or failed provider leaves a useful partial result |
| `motion` | structural plus all-frame per-shot motion | global/independent/mixed hypotheses, competing models, magnitude, direction and phases |
| `reconstruct` | maximum structural, audio, semantic and motion coverage | `VideoAnatomy` plus idempotently persisted editorial and source-neutral reconstruction proposals |

Every plan binds the source SHA-256, exact rational time base, VFR timestamp
map when needed, selected range, budgets, pins, profile, provider revision,
deduplication thresholds and deterministic plan hash. `quick` does not pretend
that sparse probes are a full decode. Focused plans are derived from ambiguous
ranges and can deepen only one shot or range.

The structural detector combines luma, colour histogram, perceptual hash,
SSIM/layout edges, motion activity, keyframes, FFmpeg scene candidates and
boundary-window progression. It distinguishes hard cut, fade in/out,
cross-dissolve, flash, directional/wipe-like change and continuous motion where
the evidence supports it. Conflicts remain competing candidates; unsupported
certainty is represented as `unknown`.

## Use

```bash
# Estimate before work; this never starts a job.
./scripts/aoa-editing anatomy estimate PROJECT ASSET --profile structural \
  --pin 0 --pin 120

# Analyze a whole video or a bounded frame range.
./scripts/aoa-editing anatomy run PROJECT ASSET --profile reconstruct
./scripts/aoa-editing anatomy run PROJECT ASSET --profile motion \
  --start-frame 240 --duration-frames 180 --pin 240 --pin 419

./scripts/aoa-editing anatomy show PROJECT PLAN
./scripts/aoa-editing anatomy contact-sheet PROJECT PLAN
./scripts/aoa-editing anatomy focused-plans PROJECT PLAN
./scripts/aoa-editing anatomy deepen-shot PROJECT PLAN shot-0001
```

Long work uses durable parent and phase receipts. `job`, `cancel`, and `retry`
operate at phase boundaries; retry verifies the checkpoint and reuses completed
sibling evidence. Cache keys include source, plan/profile, pins, provider
binding revision and pipeline revision.

The dedicated provider-result cache can be inspected and removed without
touching project evidence:

```bash
./scripts/aoa-editing anatomy prune-cache          # dry run
./scripts/aoa-editing anatomy prune-cache --apply  # rebuildable cache only
```

Both commands write a receipt under `var/state/video-anatomy/cache-prune`.
Frame manifests, anatomy packets, proposals, reviews, versions, renders and
evaluation data are outside the deletion scope.

## Surfaces

The web **Video Anatomy** tab, CLI, HTTP API and line-delimited agent protocol
call the same `VideoAnatomyJobService`, pipeline and proposal service. The web
view shows an overview timeline, shot/transition and audio lanes, coverage,
uncertainty, contact sheet, retained frames, transcript, visual and motion
observations, focused plans, provenance, human correction and proposal
review/acceptance. Its authority legend distinguishes measurement, AI
observation, inference, proposal, human correction and canonical version.

HTTP operations live under `/api/.../video-anatomy`; the agent methods use the
`video_anatomy.*` namespace. Agent reads return compact anatomy/shot projections
unless the caller explicitly requests children, so a large packet need not be
placed in one prompt.

## Providers

`structural` has no AI dependency. `semantic` and `reconstruct` may call the
tracked, provider-neutral `local-ai://vision/describe/default` alias. Requests
contain only the explicitly selected project media and timestamps. Resolution,
privacy, health, model/revision, timeout, raw-response hash, normalization
version and schema result are retained in receipts. Successful results are
cached by source, plan, prompt/config and provider revision. Missing, timeout,
malformed or privacy-denied providers become typed partial evidence and never
erase structure.

Future OCR, regions, tracking, segmentation, depth and embeddings belong behind
new declared capabilities and normalized models. To add one:

1. Add a versioned request/response contract and catalog declaration.
2. Implement an adapter that returns evidence or a typed failure, never an edit.
3. Persist raw and normalized hashes plus privacy and provider receipts.
4. Add network-free fixtures for success, timeout, malformed and absent cases.
5. Add the capability to a profile only after the structural no-provider path
   remains green.

To add a deterministic detector, emit named normalized signals and limitations,
calibrate it against independent fixture truth, and combine it without deleting
competing interpretations. Thresholds must be visible in provenance.

## Storage and reference isolation

Canonical packets and retained samples live under
`var/projects/<project>/analysis/video-anatomy/<plan>`. Temporary decode products
use `var/tmp`; provider-result caches use `var/cache/video-anatomy`; raw provider
responses remain project evidence. Not every decoded frame is persisted.
Resource preflight predicts decoded frames, runtime, artifact bytes and peak
temporary space, warns on long/deep work, and refuses a deep profile when free
space is insufficient.

A sealed evaluation reference is exceptional: it can be registered only after
a clean revision-matched readiness receipt, resolves through an untracked
hash-keyed binding, and is not copied into project assets. Its frames, audio,
masks and pixels are forbidden from render input. A reconstruction proposal may
carry durations, normalized center/scale/rotation curves, composition phases
and transition logic, but never reference pixels, source-specific matrices or
copied sound. Render lineage must contain only the permitted source hash.

## Evaluation

The network-free gate creates independently authored videos covering cuts,
fades, dissolves, flashes, rapid motion, static/zoom/pan/rotation/object motion,
short inserts, equal-luma colour changes, text/titles, VFR, fractional FPS,
audio absence, silence, accents, beats and events distributed through a longer
timeline:

```bash
./scripts/aoa-editing eval video-anatomy --output /new/eval/root
```

It reports boundary precision/recall and temporal error, transition accuracy,
shot coverage, short-event frame recall, duplicate reduction, semantic and
audio coverage, motion error, VFR/fractional-rate identity, unresolved ratio,
runtime and artifact size. Fixture truth is authored before and independently
of detector output, and sealed reference media is never accessed.

The named golden case remains a separate post-readiness evaluation. Run
`quick`, `structural`, and `reconstruct`, freeze all-frame Reference Spec v2 and
Comparison Protocol v2, create the source-neutral proposal, review its exact
hash, accept through the normal reversible path, render from the permitted
image, compare every frame, and attach a human rubric to the exact candidate
hash. Objective pass without that final human review remains `warn`.

## Known limits

- Sparse `quick` transition labels are deliberately conservative.
- Deterministic audio does not fabricate speech, speakers or music semantics
  when no provider is present.
- Camera and independent layer/object motion can be observationally ambiguous;
  motion evidence retains competing models.
- OCR, tracking, segmentation and depth are extension ports, not baseline
  claims.
- The workbench is an anatomy/review surface, not a full drag-and-drop NLE.
