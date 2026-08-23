# Schemas and migrations

Canonical models live in `src/aoa_editing/domain/models.py`. Checked-in Draft
2020-12 JSON Schemas live in `schemas/v1/` and have stable `$id` values.

```bash
./scripts/aoa-editing --help
./.venv/bin/python scripts/generate_schemas.py --check
```

`scripts/generate_schemas.py --check` fails on drift. Schema compatibility is
not inferred: `domain/migrations.py` has an explicit registry and fails unknown
versions rather than guessing. The baseline migration supports the documented
additive `0.1.0` timeline/project shape into `1.0.0`; other legacy kinds are
rejected until a reviewed migration exists.

Video Anatomy adds checked schemas for the deterministic sampling plan, frame
sample manifest, contiguous structure/coverage and transition evidence, visual
observations, motion, synchronized audio timeline, aggregate anatomy, resource
estimate, rebuildable-cache prune receipt, editorial/reconstruction proposals,
human review/acceptance, and synthetic corpus/report. Each child is independently
versioned and hash-bound. The aggregate is evidence only; proposals explicitly
assert `timeline_mutated=false`, and acceptance is a separate receipt binding
the reviewed proposal hash to a reversible project version.

`reference-spec.schema.json` is the immutable post-gate observation contract.
It records technical media properties, sampled camera transforms, audio
structure, uncertainties, fixed comparison thresholds, and the rule that the
reference can never enter render lineage. It is evaluation evidence, not the
canonical edit graph.

`motion-ground-truth.schema.json`, `motion-recovery.schema.json`, and
`motion-recovery-gate.schema.json` define the independent synthetic corpus,
all-frame observable estimates, confidence and uncertainty, derivatives, and
the fail-closed gate. Pivot values in render truth do not imply that a recovered
global matrix can uniquely identify pivot separately from translation.

`reference-motion-evidence-v2.schema.json` and
`reference-spec-v2.schema.json` are additive v2 evaluation contracts. They do
not reinterpret or overwrite `reference-spec.schema.json`: v1 remains
historical evidence, while v2 stores every-frame matrices, temporal
derivatives, residual-flow diagnostics, ambiguity, phase structure, and the
criteria frozen before v2 reconstruction.

`reference-reconstruction-pass-v2.schema.json` proves one ordinary application
pass from the exact Spec v2 hash through a proposed and approved Decision Graph,
preview/final render and QC, editable exports, source-only lineage, and frozen
complexity-hypothesis decisions. It is a run receipt, not an editorial verdict.

`reference-phase-correction-v2.schema.json` types the optional `5r` temporal
refinement: exact prior Comparison v2 and candidate-motion hashes, monotonic
time-warp anchors, identity-protected coupling range, unchanged threshold
revision, predicted metrics/checks, and explicit exclusion of reference and
candidate media from render inputs. A pass receipt may carry this contract only
for `phase_compensated_matrix`.

`reference-reconstruction-study-plan-v2.schema.json` types the explicit
diagnostic judgments over the exact ordered `1, 2, 3, 4, 5, 5r` experiment.
`reference-reconstruction-study-v2.schema.json` then re-hashes and joins the
frozen spec, protocol, pass comparisons, optional run receipts, motion gates,
metric deltas, complexity decisions, and selected candidate. Agent visual
observations remain labelled diagnostic; a pending human review keeps the
study at `warn`, and only an attributable passing Editorial Review v2 can make
the combined study pass.

Reference Workspace v2 adds schemas for the immutable workspace registration,
the derived workbench bundle, local reference binding, motion-correction
proposal/review, and explicit human correction. The registration and
proposal/review documents are canonical project records; the workbench bundle
is a derived read model; the physical reference binding is untracked local
state.

`clean-rerun-v2.schema.json` is the revision-bound replay contract for the
selected phase-compensated result. It types both baseline and replay application
proofs down to project/asset/treatment/patch/version/Decision Graph hashes,
render inodes, QC, lineages, derivatives, semantic timeline fingerprints, and
validated Kdenlive/OTIO exports. The report forbids old/reference media inputs,
hidden manual steps, and mandatory skips. It remains distinct from the
historical `clean-rerun.schema.json`.

`comparison-protocol-v2.schema.json` types the revision-bound algorithms,
unchanged frozen criteria, mandatory checks/artifacts, and human rubric before
the first candidate. `comparison-v2.schema.json` separates objective all-frame
metrics from the combined verdict. `editorial-review-v2.schema.json` binds a
named human operator, candidate hash, reviewed artifacts, and every rubric axis;
review is attached as a new immutable comparison revision.

`motion-language-v2.schema.json` is the standalone schema for the additive
`transform_v2` effect also embedded in timeline/version schemas. It defines
canonical rational subframes, path/scalar/matrix curves, handles and tangents,
continuity and overshoot policies, phase/coupling metadata, transform order,
and deterministic sampling. Existing `transform` objects remain valid.
`migrate_transform_effect_v1` converts the old `canvas_center` virtual camera
into the new pivot-target semantics with mathematically equivalent Bezier
segments; ambiguous legacy offset semantics fail closed instead of being
reinterpreted.

The baseline also checks explicit schemas for Brief revisions, evidence
authority/time ranges/supersession, treatment comparison facts, decision graphs,
patch previews, render frame ranges, derived media, style confirmations,
clean-rerun reports, transfer reports, and candidate technique packets. A
`completion-audit.schema.json` v2 joins every current proof lane into a
fail-closed DoD map. It records the exact clean Git revision and goal boundary,
hashes every evidence file, binds Spec/Protocol/Reconstruction/Clean Rerun
bytes, requires an exact local tag, preserves a separate human editorial
verdict, and records unresolved checks instead of emitting a partial pass. Its
publication check accepts a pre-existing remote only after explicit opt-in and
successful read-only proof that no goal commit or completion tag is published.
A schema-compatible additive field has a default; an incompatible future
change requires a new schema version and explicit migration rather than silent
parsing.

Technique packet schema `2.0.0` is an additive generation within the same
packet family: v1 scenario and historical technique packets remain valid, while
a v2 packet must carry all curve, phase, resolution, and dense normalized
camera contracts and must not retain the legacy recipe. Separate schemas type
composition evidence, the pre-render applicability decision, and the
seven-class transfer-corpus receipt. The latter preserves both successful
applications and expected refusals, their independent replays, QC/export
evidence, and the still-candidate output packet.

`ai-capability-catalog.schema.json` types the tracked, path-free declaration
surface. `local-provider-binding.schema.json` types the ignored host overlay
without making it canonical. `resolved-provider-receipt.schema.json` records
one redacted resolution/invocation attempt and restricts output authority to
evidence or proposal. `provider-alias-eval.schema.json` joins every failure,
privacy, fallback, partial-evidence, human-supersession, doctor, and no-AI
baseline proof into one revision-bound receipt.
`local-ai-integration.schema.json` is the separate Phase-15 live proof. It
requires a decision for every declared capability, one owner-registered
existing ASR, upstream revision/license evidence, a complete OpenVINO export
tree digest, measured host fit and latency, a real alias receipt, known-text
WER/CER, and the still-functional no-AI baseline. It forbids model downloads,
reference input, canonical AI authority, and mandatory skips.

`screen-workflow-plan.schema.json` is the pre-capture boundary: script hash,
complete terminal template catalog, stable authored beats, capture/framing
rules, forbidden content, review state, and an explicit assertion that no media
is bound. `screen-workflow-voiceover-timing.schema.json` binds one immutable
narration asset and analysis evidence to contiguous script-beat cues. A draft
cannot enter an edit until a separate attributable review creates a superseding
revision. `screen-workflow-edit-spec.schema.json` is the later post-capture
boundary: an attributable human review binds every beat to one or more immutable
screen ranges, output durations, 1.0x-2.0x focus targets, optional 0.5x-64.0x
constant speeds/source-audio/captions, optional reviewed voiceover timing, and
rationale before an executable Treatment may exist. Named continuity groups
require adjacent source ranges, continuous camera endpoints, and gradual speed
tiers. Temporal intent types prompt entry, agent progress, results, waits,
natural holds, and reviewed live UI regions. Prompt entry and natural holds
remain at 1x; speeds above 24x require an explicit risk override. Transitions do
not justify omitting unreviewed source intervals, and no still-frame or
full-frame-loop hold is admitted.

`reference-workflow-study-plan.schema.json` types an untracked batch evaluation
plan with external paths, expected hashes, and manually classified ranges. An
included range fails validation if a human is present or its class is not a
terminal/IDE/browser/workflow/diagram surface.
`reference-workflow-study.schema.json` is path-free derived evidence containing
source aliases/hashes, media facts, range activity metrics, human-free contact
sheets, candidate observations, the exact readiness revision, and literal
non-copy/non-asset/non-render/non-style assertions.
