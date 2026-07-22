# ADR-0010: All-frame Comparison Protocol v2 and the human verdict boundary

- Status: accepted
- Date: 2026-07-19

## Context

The v1 comparison was useful proof for the first prototype, but sampled
position/scale/rotation RMSE and aggregate VMAF/SSIM could not explain the
user-observed difference in motion character. A technically close sequence can
still feel mechanically linear, mistime its settle, or lose the coupled
translation/scale/rotation relationship.

Reference Reconstruction Spec v2 already freezes strict criteria before the
first v2 render. Those criteria still need one executable interpretation.
Without a separate protocol, implementation details such as flow
normalization, blur detection, phase scoring, or the required review artifacts
could drift between passes. Conversely, an automated score must not impersonate
the human judgment that the transition has the intended weight and twist.

## Decision

Before the first candidate render, freeze a revision-bound
`ReferenceComparisonProtocolV2` against the exact Spec v2 hash. It contains:

- the unchanged `reference-comparison-v2-frozen-1` criteria;
- exact algorithm revisions and operational parameters;
- every mandatory technical, geometric, temporal, perceptual, editorial, and
  lineage check;
- every required diagnostic artifact;
- the complete human-review rubric;
- the clean implementation Git revision.

Each comparison independently recovers motion from every candidate frame using
the permitted still source. It scores:

- exact geometry, cadence, frame count, duration, codec decodability, and audio
  structure;
- center, scale, rotation, visible boundaries, projected matrix grids, model
  family, pivot identifiability, and observable transform order;
- velocity, acceleration, jerk, onset, duration, settle, overshoot,
  scale/rotation/center coupling, optical flow, and decoded temporal
  consistency;
- VMAF, SSIM, diagnostic PSNR, aligned frame difference, black frames,
  unintended edge exposure, and relative sharpness;
- source-only render lineage and absence of the reference hash anywhere in
  that lineage.

Pivot receives no invented numeric error when both evidence sets establish the
pivot/translation gauge as unidentifiable. Transform order is scored through
the observable source-to-output matrix rather than a fabricated unique
decomposition.

The production comparator rejects a candidate that is the reference file or
has its hash. A narrow opt-in exists only for synthetic identity fixtures.
Reference pixels may appear in comparison-only diagnostics; they remain
forbidden in project assets, canonical edit state, and render lineage.

## Human review

Objective checks produce `objective_overall=pass|fail`. They cannot produce the
final indistinguishability verdict. Without an attributable
`EditorialReviewV2`, an objectively passing report remains `overall=warn`.

The human review must cover smoothness, weight, natural acceleration, rotation
character, twist character, settle timing, absence of mechanical linearity,
composition, absence of a reference overlay, and separation of encoding
defects from motion mismatch. It names the reviewer and the exact candidate
hash and records the side-by-side, phase contact sheet, and aligned-difference
artifacts reviewed.

Review does not mutate the objective receipt. Attaching it creates a new
immutable report revision that supersedes the objective-only report. Any failed
rubric axis makes the combined report fail.

## Operational thresholds

Boolean frozen requirements need deterministic implementations. Protocol
revision `comparison-v2-all-frame-1` therefore freezes the Farneback flow
geometry and normalization, luma definition of an unexpected black frame,
Laplacian sharpness ratio, metric resolution, and diagnostic difference gain.
These parameters are frozen before a candidate and do not replace or relax the
Spec v2 thresholds.

PSNR remains diagnostic. No aggregate perceptual metric can override a failed
geometric, temporal, lineage, or human-review check.

## Consequences

- Each reconstruction pass is comparable on the same axes and thresholds.
- Motion character is measured through derivatives, phase/coupling, and flow,
  not inferred from a few aligned frames.
- Objective success and editorial acceptance remain distinct authorities.
- Failed or absent human review is visible and fail-closed for completion.
- V1 reports remain valid historical evidence and are not reinterpreted.
