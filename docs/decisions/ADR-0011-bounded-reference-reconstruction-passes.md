# ADR-0011: Bounded, evidence-gated Reference Reconstruction v2 passes

- Status: accepted
- Date: 2026-07-19

## Context

The v1 reconstruction is technically close, but the operator identifies a
meaningful difference: its transition feels more linear and lacks the smoother
twist and settle of the reference. Reference Motion Evidence v2 explains that
character as a curved center path lagging synchronized scale and rotation. It
also establishes three important negative facts:

- a global source-to-output matrix cannot identify pivot separately from
  compensated translation;
- every frame selects the similarity family rather than affine or homography;
- residual optical flow never reaches the generic materiality gate for a local
  warp hypothesis.

An iterative reconstruction can easily overfit if each disappointing candidate
silently adds another transform family. Conversely, jumping directly to the
dense measured matrix would hide which part of the improved result came from
timing, path continuity, or added spatial complexity.

## Decision

Reference Reconstruction v2 uses four primary executable, ordered passes after
the preserved v1 baseline:

1. `timing_easing` uses endpoint-only similarity motion with one shared
   nonlinear progress curve. It changes timing without adding a spatial
   operation.
2. `continuous_tangents` adds evidence-selected phase knots, the observed
   curved center path, and explicit C1 Hermite tangents.
3. `moving_pivot_hypothesis` renders a moving pivot only with exactly
   compensating position. Its observable integer-frame matrices equal the
   preceding pass; this tests and normally rejects authoring complexity that
   cannot improve output.
4. `coupled_scale_rotation` executes the frozen all-frame source-to-output
   matrix curve. Scale/rotation synchronization and center lag remain explicit
   interpretation metadata, but the canonical edit does not pretend that a
   unique pivot or authoring order was observed.

If pass 5 clears every non-phase objective check but its recovered onset,
duration, or settle remains outside the frozen tolerance, one refinement
`phase_compensated_matrix` may follow as sequence `5r`. ADR-0012 defines its
comparison binding, protected identity interval, and fail-closed search. It is
a temporal correction to the pass-5 matrix curve, not a fifth spatial
hypothesis.

Every executable pass is an ordinary `Treatment`, proposed Decision Graph,
approved reversible version, Motion Language v2 render, QC result, and editable
Kdenlive/OTIO export. The pass receipt records source-only lineage and the
frozen hypothesis gates.

Affine/homography and local warp are conditional hypotheses, not mandatory
effects. They are not rendered while their frozen evidence gates say
`rejected_by_evidence`. Candidate comparison scores may choose between already
justified passes, but may not retroactively change those gates or the frozen
Comparison v2 thresholds.

The local-warp hypothesis gate uses a generic analysis-space rule: more than
five percent of frames, and at least three frames, must have valid residual-flow
p95 above 0.5 analysis pixels. This is a complexity-admission rule, not a
replacement for any frozen candidate-comparison threshold.

## Consequences

- Improvements and rejected complexity remain attributable to a named pass.
- Moving pivot is tested without promoting a gauge choice into observed truth.
- The final dense matrix is a portable canonical edit operation, not a hidden
  renderer script.
- Unsupported affine, perspective, and local deformation do not enter the
  domain graph merely because the language can express them.
- Human judgment of smoothness, weight, twist, and settle remains separate from
  the automated pass receipt and Comparison v2 report.
