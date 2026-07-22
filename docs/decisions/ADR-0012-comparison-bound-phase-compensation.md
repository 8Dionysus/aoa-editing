# ADR-0012: Comparison-bound, localized phase compensation

- Status: accepted
- Date: 2026-07-19

## Context

The dense pass-5 matrix curve can reproduce the frozen geometry, coupling, and
perceptual structure while the independently recovered candidate still crosses
the phase detector a few frames late at onset or early at settle. This is a
measured renderer/recovery timing bias. Adding affine, homography, local warp,
or an identifiable pivot claim would not address it and would contradict the
frozen evidence gates.

Changing Comparison v2 thresholds after seeing the candidate would also erase
the meaning of the evaluation.

## Decision

AoA Editing permits one `phase_compensated_matrix` refinement, identified as
sequence `5r`, only when the prior Comparison v2 report:

- is bound to the exact frozen Spec v2 hash, source hash, reference hash, and
  candidate motion-recovery artifact;
- fails at least one of onset, active duration, or settle;
- has no other failed objective check.

The deterministic search constructs a strictly monotonic piecewise-linear map
from output frame to frozen matrix-curve time. It searches only before and
after the frozen scale/rotation–center coupling interval. That whole interval
must remain identity-mapped. Every candidate is predicted against the original
geometric, derivative, phase, and coupling thresholds; thresholds are never
rewritten. When available, selection requires a one-frame margin inside the
phase tolerance, then minimizes maximum segment-slope deviation, localized
warp regularization, and frozen-threshold utilization.

The resulting `ReferencePhaseCorrectionV2` records the exact report and
candidate-motion hashes, anchors, protected ranges, predicted metrics/checks,
algorithm revision, and the fact that neither reference nor prior candidate
media is a render input. The application service persists it as
`reference.phase_correction.v2` evidence. The normal Treatment, proposed and
approved Decision Graph, Motion Language v2 matrix curve, render, QC, and
interchange path remains authoritative.

## Consequences

- A phase-only miss can be corrected without admitting a new spatial
  explanation.
- A VMAF, SSIM, geometry, coupling, lineage, or other objective regression
  blocks the correction instead of being hidden by retiming.
- The central “twist” stays exactly the pass-5 observable trajectory.
- Comparison-derived scalar evidence may influence a proposal, but comparison
  pixels and the candidate video never enter render lineage.
- The correction remains a proposal until a new full Comparison v2 run proves
  the rendered candidate; human editorial review remains separately
  attributable.
