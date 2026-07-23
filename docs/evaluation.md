# Evaluation and completion contract

## Generic readiness gate

Reference content stays sealed until all of the following are green on generic
fixtures:

- schema validation and migration checks;
- domain invariants and patch/version round trips;
- ingest, evidence, all three scenarios, preview, final, QC, and interchange;
- web/API, CLI, and agent contract smoke tests;
- real Chromium form, Brief, Style consent, project selection, and tab smoke;
- restart/resume and idempotency checks;
- a real FFmpeg render and real MLT/Kdenlive load smoke;
- source-lineage and forbidden-input tests;
- clean bootstrap from documented prerequisites.

The gate emits a machine-readable receipt with check IDs, commands, timestamps,
Git revision, tool versions, and artifact hashes. A skipped mandatory check is a
failure.

For `screen.workflow`, generic contract coverage includes multi-segment beats,
source and camera continuity, gradual speed tiers, real-time prompt entry,
explicit risk above 24x, live-region loopability, natural holds, and aggregation
of several visual segments into one reviewed voiceover cue. A monotonic timer
must fail any loopable claim. Region-aware loop synthesis is not considered
implemented until a separate media-path evaluator can detect frozen live UI,
timer repetition/reversal, and loop seams.

Reviewed demo experience is evaluated separately from render correctness. An
accepted experience claim requires private evidence refs, distinct owner-review
refs, a public-safe summary and limitation, and named target surfaces. Receipt
tests prove deterministic projection hashing, idempotent immutable storage, and
the absence of session ids, raw refs, and private source wording from the public
projection. Rejected and deferred claims remain in the projection so a proposed
technique cannot silently appear as implemented. Each later project demo should
be admitted as a new application or counterexample; cross-project defaults need
multiple independently reviewed cases.

## Human-free workflow reference study

After readiness, a batch plan may analyze multiple external terminal/workflow
references without ingesting them:

```bash
./scripts/aoa-editing eval reference-workflow-study \
  --plan var/evals/reference-workflow/plans/build-week.json \
  --output var/evals/reference-workflow/20260720-build-week
```

The plan supplies exact hashes and non-overlapping, manually reviewed ranges.
An included range must be classified as terminal, IDE, browser-workflow,
workflow-UI, or diagram and must declare `human_present=false`. The evaluator
verifies hashes and media duration, measures sampled frame activity, and emits
contact sheets from included ranges only. Excluded ranges retain the manual
decision but are not decoded. Output is restricted to the resolved eval root;
no source video is copied, no physical source path is written to the receipt,
and the receipt explicitly denies project-asset, render-input, Style Memory,
and canon authority. Candidate rules must cite exact included range ids.

## Ground-truth motion recovery gate

Reference Motion Evidence v2 is forbidden until a source-independent recovery
laboratory passes:

```bash
./scripts/aoa-editing gate motion-recovery
```

The gate deterministically renders synthetic curves and known similarity,
affine, homography, moving-pivot, and local-warp cases, including compression,
blur, texture loss, crop, aspect-ratio, alpha, and low-resolution stressors.
Each frame has exact matrices, normalized parameters, derivatives, phases, and
transform order. The analyzer sees only each synthetic source and encoded video,
then reports confidence, competing global models, temporal outliers, continuous
velocity/acceleration/jerk, and uncertainty. Pivot remains explicitly
unidentifiable when an equivalent translation decomposition exists. A local
warp must produce model mismatch rather than confident global precision.

Runtime corpus videos, truths, recovery reports, and the fail-closed gate receipt
live under `var/evals/motion-recovery`; only schemas, generator, analyzer,
thresholds, and tests are tracked.

## Reference Motion Evidence v2

After readiness and the motion-recovery gate pass on the same clean revision,
the all-frame analyzer may decode the sealed reference:

```bash
./scripts/aoa-editing eval reference-analyze-v2 \
  --motion-gate var/evals/motion-recovery/<run>/motion-recovery-gate.json \
  --baseline-comparison var/evals/reference/<v1>/comparison.json \
  --output var/evals/reference-v2/<new-run>/analysis
```

This creates new `reference-motion-evidence-v2.json` and
`reference-spec-v2.json` objects without changing v1. Every declared frame has
source correspondences, competing transform models, normalized geometry,
visible source boundaries, confidence intervals, derivatives, residual flow,
and uncertainty. Diagnostic artifacts include transform and derivative plots,
frame overlays, residual maps, a phase contact sheet, and explicit pivot
non-identifiability. Kinematics use deterministic local-polynomial estimates so
sub-pixel registration noise is not promoted into false jerk. The phase model
also compares normalized scale/rotation progress with the projected center-path
progress: it records channel lag, peak-velocity lag, path curvature, and the
strongest coupled interval without claiming an unobservable moving pivot.
Nested model fits with no material gain remain uncertainty, but do not make
every otherwise well-measured frame ambiguous. The v2 comparison criteria are
frozen in the spec before the first v2 reconstruction render.

Before rendering the first v2 candidate, freeze the executable interpretation
of those criteria from a clean revision:

```bash
./scripts/aoa-editing eval reference-freeze-comparison-v2 \
  --spec var/evals/reference-v2/<run>/analysis/reference-spec-v2.json \
  --output var/evals/reference-v2/<run>/comparison-protocol
```

The resulting protocol fixes every all-frame metric, flow/blur/black-frame
algorithm, required artifact, and editorial axis. It rejects a production
candidate identical to the reference. After an ordinary application render,
run:

```bash
./scripts/aoa-editing eval reference-compare-v2 \
  --protocol /path/to/comparison-protocol-v2.json \
  --spec /path/to/reference-spec-v2.json \
  --candidate /path/to/final/video.mp4 \
  --lineage /path/to/final/lineage.json \
  --output /new/comparison-pass
```

This re-analyzes every candidate frame and writes the candidate motion receipt,
frame diagnostics, VMAF/SSIM/PSNR logs, side-by-side and amplified difference
videos, phase contact sheet, and normalized geometric/derivative plots. An
objective pass remains `warn` until the operator completes the generated rubric
and attaches it without changing the objective report. The generated
`reviewed_artifacts` role names are portable identifiers and are resolved
against the exact comparison report; literal relative or absolute artifact
paths remain accepted for compatibility:

```bash
./scripts/aoa-editing eval reference-review-v2 \
  --comparison /path/to/comparison-v2.json \
  --protocol /path/to/comparison-protocol-v2.json \
  --review /path/to/human-review.json \
  --output /path/to/comparison-v2-reviewed.json
```

## Reference reconstruction

Only after readiness may the evaluator decode and analyze the reference. It may
measure duration, cadence, motion, framing, color, audio, and other observable
properties. It must write an evidence packet before authoring a treatment.

The executable analysis surface is:

```bash
./scripts/aoa-editing eval reference-analyze --output /path/to/eval/analysis
```

It requires a passing readiness receipt for the current Git revision, estimates
global source motion through a general feature-registration adapter, and creates
an immutable `reference-spec.json`. Running it again returns the frozen spec
rather than changing comparison criteria after a render.

The reconstruction then uses the ordinary project/application contracts:

```bash
./scripts/aoa-editing eval reference-reconstruct \
  --spec /path/to/reference-spec.json \
  --output /new/reconstruction/run
```

That route creates a sealed project, ingests and analyzes only the permitted
source, persists the frozen parameter spec as evidence, proposes and accepts a
reversible `reference.reconstruct` treatment, produces preview/final renders,
executes QC, and exports Kdenlive plus OTIO projections. Its receipt audits each
render lineage against the forbidden reference hash.

The reconstruction render may use only the supplied still image plus explicitly
declared generated or licensed auxiliary assets. Reference frames, pixels,
masks, optical flow, thumbnails, or audio must never enter the render graph.

### Bounded Reconstruction v2 passes

Spec v2 is executed as four primary named passes after the preserved v1
baseline, with one comparison-bound `5r` refinement permitted only for a
phase-only miss. Run each pass in a new output root:

```bash
./scripts/aoa-editing eval reference-reconstruct-pass-v2 \
  --spec var/evals/reference-v2/<run>/analysis/reference-spec-v2.json \
  --pass timing_easing \
  --output var/evals/reference-v2/<run>/passes/02-timing-easing
```

Valid primary pass names, in order, are `timing_easing`,
`continuous_tangents`, `moving_pivot_hypothesis`, and
`coupled_scale_rotation`. Each command creates a new project and emits
`reconstruction-pass-v2.json` with the proposed and approved graph paths,
preview/final QC and lineage, editable exports, and frozen hypothesis
decisions. The moving-pivot pass is explicitly gauge-equivalent; it does not
claim an observed pivot. Affine/homography and local warp remain unexecuted
when the pass receipt records `rejected_by_evidence`.

When pass 5 fails only frozen phase checks, derive and execute the bounded `5r`
refinement from that exact Comparison v2 report:

```bash
./scripts/aoa-editing eval reference-reconstruct-pass-v2 \
  --spec var/evals/reference-v2/<run>/analysis/reference-spec-v2.json \
  --pass phase_compensated_matrix \
  --comparison var/evals/reference-v2/<run>/passes/05/comparison/comparison-v2.json \
  --output var/evals/reference-v2/<run>/passes/05r-phase-compensated
```

The command refuses a prior non-phase failure. Its
`phase-correction-v2.json` binds the prior report and candidate-motion hashes,
records the monotonic anchors and protected identity range, and carries only
passing predictions against the unchanged thresholds. A new full
`reference-compare-v2` run remains mandatory; the prediction is not the result.

Run `reference-compare-v2` against every pass final and the same frozen
protocol. Comparison artifacts and objective deltas decide whether added
complexity is retained. They cannot revise the protocol or manufacture the
separate human verdict.

After all six comparisons exist, author a
`ReferenceReconstructionStudyPlanV2` whose relative paths resolve from the
plan file itself, then produce the machine-derived study:

```bash
./scripts/aoa-editing eval reference-summarize-study-v2 \
  --plan /path/to/reconstruction-study-plan-v2.json \
  --output /new/reconstruction-study-v2.json
```

The command requires the exact ordered `1, 2, 3, 4, 5, 5r` sequence, re-hashes
the frozen spec, protocol, comparisons, supplied run receipts and motion gates,
re-derives objective metrics and deltas, verifies the selected candidate
actually passes Comparison v2, and records rejected unexecuted spatial
hypotheses. It refuses a dirty implementation revision and binds the resulting
report to the exact clean Git commit. It never promotes an agent diagnostic
observation into the human rubric. Until `reference-review-v2` attaches an
attributable passing editorial review to the selected comparison, the study
remains objective `pass` but combined `warn`.

## Proof passes

1. **Reconstruction:** create the closest justified result through product
   interfaces, not hand-edited target scripts.
2. **Clean rerun:** delete only regenerable workspace products, bootstrap from
   the committed repository, and reproduce the result from saved intent and
   decisions.
3. **Transfer:** run a materially different generic source through the same
   scenario without target-specific branches.

`scripts/reproduce-reference` is the documented selected-result replay after a
Spec and passing `5r` receipt have been frozen:

```bash
./scripts/reproduce-reference \
  --spec /path/to/reference-spec-v2.json \
  --baseline /path/to/selected/reconstruction-pass-v2.json \
  --output /new/absent/clean-rerun-v2-root
```

It delegates to the product-owned Clean Rerun v2 evaluator. The replay uses a
fresh data/cache/tmp root and compares semantic fingerprints, output hashes,
inodes, lineages, derivatives, decisions, and QC. The separate transfer command
generates unrelated sources, applies the extracted packet through normal
services, remeasures motion, renders both profiles, checks editable exports,
and appends application history without promoting status.

The selected v2 result has a stricter replay lane:

```bash
./scripts/aoa-editing eval clean-rerun-v2 \
  --spec /path/to/reference-spec-v2.json \
  --baseline /path/to/selected/reconstruction-pass-v2.json \
  --output /new/absent/clean-rerun-v2-root
```

It creates a new Editing Home and new project identity. The source PNG is the
only media input. The exact approved `ReferencePhaseCorrectionV2` embedded in
the selected receipt is replayed without reading the old comparison, candidate
motion, reference video, or render. Fifteen mandatory checks cover separate
paths/inodes, canonical records, semantic fingerprint, byte-equivalent
preview/final outputs, source-only lineage, absence of copied reference bytes,
derivatives, QC, Kdenlive/OTIO, and hidden manual steps. This deterministic
proof does not fabricate the separately pending human editorial verdict.

## Heterogeneous technique transfer v2

The selected typed semantics can be upgraded and transferred without opening
either user media file:

```bash
./scripts/aoa-editing eval technique-transfer-corpus-v2 \
  --packet editing-knowledge/history/continuous-contain-reveal-v1-revision2.json \
  --spec /path/to/reference-spec-v2.json \
  --selected-pass /path/to/selected/reconstruction-pass-v2.json \
  --photograph /usr/share/pixmaps/faces/mountain.jpg \
  --output /new/absent/transfer-corpus-v2-root
```

The archived input packet is revision two; the evaluator derives a
revision-three portable recipe only from the frozen spec and embedded phase
correction. The checked-in current candidate is the resulting revision-four
packet after the complete corpus, so it is an output and not the reproducible
upgrade input. The portable recipe
retains dense normalized center/scale/rotation, C1 tangents, a fixed-pivot
gauge, phase/coupling meaning, and a two-times pixel-density cap. The output
packet advances once more after the complete corpus and remains `candidate`.
Neither packet contains media hashes or physical paths.

Five positive cases cover square illustration, an OS-owned licensed
photograph, portrait, alpha, and complex texture. Each follows the normal
application approval path, renders preview/final, passes QC, validates
Kdenlive/MLT and OTIO, matches every compiled matrix to an independent
normalized calculation, and reproduces byte-equivalent final media in a
distinct workspace. A feature-rich case also remeasures decoded video motion.

The 48-pixel source must return `insufficient_source_resolution`; the
independent-panel composition must return `inapplicable_composition`. Both
decisions are replayed in separate workspaces and pass only when no Treatment,
version, or render file exists. The system photograph is never tracked; its
runtime receipt records installed package, revision, vendor, upstream, hash,
and RPM license authority. Mandatory skips are forbidden.

## Local-AI provider alias gate

The provider boundary is evaluated independently of any sealed reference and
without downloading a model:

```bash
./scripts/aoa-editing eval provider-aliases \
  --output /new/absent/provider-alias-eval-root
```

The evaluator requires a clean revision and a new output root. It validates all
nine declarations and exercises available, missing, stale, version-mismatched,
timed-out, malformed, privacy-denied, deterministic-fallback, and partial
responses. Every attempt must write a redacted resolved-provider receipt.
Additional checks prove human correction supersession, all three deterministic
scenarios without AI, absence of model paths in the domain core, live doctor
resolution of an untracked fixture binding, and evidence/proposal-only output
authority. This gate proves the adapter boundary; live host model quality and
ownership require a separate owner inventory and product invocation receipt.

## Live local-AI admission gate

`aoa-editing eval local-ai-live` closes the separate Phase-15 claim without
downloading a model or opening the sealed reference. It must run inside
`abyss-machine resource launch --class medium --kind ai` and against a clean
revision. Required caller inputs bind the current stack source/deploy unit,
managed-unit allowlist, model-owner root, local upstream revision ref, and a
captured official model metadata response.

The gate joins:

- fresh machine capability, dictation, model, and device packets;
- the ignored alias binding and its live normalized health;
- byte-equal source/deploy stack registration and the active warm service;
- upstream revision and MIT license metadata;
- SHA-256 for every regular OpenVINO export file plus an aggregate tree hash;
- current/peak service and evaluator-unit memory, device route, health latency,
  invocation latency, and estimated inference RTF;
- a real provider receipt for unrelated known-text synthetic Russian speech;
- frozen WER `<= 0.40` and CER `<= 0.25`;
- an isolated no-binding product run that retains silence/loudness evidence and
  localizes the missing transcript.

The report decides all nine declared capability gaps. Only the already-present
ASR is bound. Inventory-only embeddings, resident text models, and multimodal
projectors do not become product capabilities; VAD, diarization, vision,
retrieval, editorial models, and a video-language model remain unbound until
separate evidence justifies them. `new_models_installed=false`,
`downloaded_model_bytes=0`, evidence-only authority, no reference input, and
zero mandatory skips are schema invariants.

## Completion claim

The goal is complete only when the full definition of done has an evidence row,
the worktree is clean, and all mandatory checks are green. Similarity alone is
not completion.

The final join is a fail-closed, versioned evaluation rather than a prose-only
claim:

```bash
./scripts/aoa-editing eval completion-audit \
  --storage /path/to/relocation-receipt.json \
  --readiness /path/to/readiness.json \
  --motion-gate /path/to/current/motion-recovery-gate.json \
  --motion-evidence /path/to/reference-motion-evidence-v2.json \
  --spec /path/to/reference-spec-v2.json \
  --reconstruction /path/to/selected/reconstruction-pass-v2.json \
  --comparison /path/to/reviewed-comparison-v2.json \
  --reference-ui /path/to/reference-ui-smoke.json \
  --clean-rerun /path/to/clean-rerun-v2.json \
  --transfer /path/to/transfer-corpus-v2.json \
  --candidate editing-knowledge/candidates/continuous-contain-reveal.json \
  --provider-alias /path/to/current/provider-alias-eval.json \
  --local-ai /path/to/current/local-ai-integration.json \
  --goal-start-revision FULL_40_HEX_REVISION \
  --private-history-ref local/private-history-pre-publication-20260721 \
  --local-tag prototype-v0.3.0 \
  --output /new/completion-audit/root
```

For an already evolved repository that has a deliberate remote, add
`--allow-preexisting-remote`. This does not waive non-publication: the audit
must successfully inspect all remote heads/tags, find no goal commit or final
tag there, and confirm that the working branch has no upstream. Do not use the
flag merely to ignore a remote error.

The audit re-hashes both immutable user inputs and every receipt, requires the
readiness, motion gate, provider gate, and live-AI gate to match the clean
current Git revision, verifies the exact Spec/Protocol/Reconstruction/Clean
Rerun chain, rejects a missing human review or any nested skip/failure, proves
that the reference hash is absent from render lineage and tracked files, and
maps all 28 rows in `docs/definition-of-done.md` to passing checks. Until the
human verdict and local tag exist, the report is expected to fail and names
those unresolved checks; it never upgrades an objective-only comparison.
