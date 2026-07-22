# ADR-0013: Typed reconstruction study and selection ledger

- Status: accepted
- Date: 2026-07-19

## Context

The reference reconstruction is an ordered experiment, not one lucky render.
Its baseline, timing/easing attempt, continuous-curve recovery, pivot gauge
test, coupled scale/rotation pass, and bounded phase refinement each answer a
different hypothesis. Separate Comparison v2 reports preserve the raw evidence
but do not by themselves prove that the sequence was complete, that every
comparison used the same frozen spec and protocol, or why one candidate was
selected.

A prose summary can silently omit a regression, transpose passes, copy stale
metrics, or blur an agent's visual diagnostic with the operator's required
editorial verdict.

## Decision

AoA Editing records the experiment in two additive v2 contracts:

- a human-authored plan carrying only explicit rationales, diagnostic visual
  observations, artifact references, dispositions, and conditional hypothesis
  decisions;
- a machine-derived study that re-opens and hashes every immutable evidence
  input, re-derives metric summaries and deltas, verifies common spec/protocol
bindings, and proves that the selected candidate passes objective
Comparison v2.

The plan must contain exactly the ordered sequence `1, 2, 3, 4, 5, 5r`, with
the corresponding bounded pass kind at each position. Relative artifact paths
are resolved from the plan's directory so the result cannot depend on the
launching shell's current directory. Supplied application receipts and generic
motion gates are validated rather than trusted as opaque files.

Agent-authored frame and contact-sheet observations are explicitly labelled
`agent_diagnostic`. They may explain a disposition but never satisfy
`editorial-human-review`. The derived study remains combined `warn` while the
selected comparison has no attributable Editorial Review v2, even when every
objective check passes. Study generation also requires and records one exact
clean implementation revision.

## Consequences

- The selected render is traceable to all retained and rejected attempts, not
  only to the winning file.
- Metric deltas and byte-identical candidates expose whether added complexity
  helped, regressed, or was merely gauge-equivalent.
- Conditional affine/homography and local-warp branches can be closed as
  evidence-rejected without pretending they were executed.
- Human judgment remains a real external acceptance boundary and cannot be
  manufactured by the evaluation runner.
- A later reviewed comparison can produce a new study report without mutating
  the frozen protocol, earlier comparisons, or original plan.
