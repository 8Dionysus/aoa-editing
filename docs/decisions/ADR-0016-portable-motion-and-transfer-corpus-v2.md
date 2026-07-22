# ADR-0016: Portable motion and transfer corpus v2

- Status: accepted
- Date: 2026-07-19

## Context

The revision-two reveal packet retained sampled v1 camera curves and had only
one independent procedural transfer. That was enough to show source neutrality,
but not enough to preserve the selected v2 timing character, bind resolution
risk, distinguish an inapplicable composition, or justify growth toward shared
editing knowledge.

The selected `5r` result is expressed as a dense observed matrix curve plus a
bounded phase correction. Copying those source-pixel matrices into a technique
would make the packet depend on the original source dimensions. Re-reading the
reference during transfer would also collapse evaluation evidence into product
execution.

## Decision

Upgrade the packet from the frozen Spec v2 and the selected passing `5r`
receipt, not from either media file. The portable recipe retains one sample per
output frame for normalized output center, scale relative to contain fit, and
unwrapped rotation. It derives C1 Hermite tangents, uses a fixed normalized
source-center pivot, and records the onset, lagging-center twist, synchronized
scale/rotation, and settle phases. It contains no media hash, reference ID,
physical path, or source-pixel matrix.

Before creating a Treatment, the application service emits a typed
`TechniqueApplicabilityDecisionV2`. It requires:

- image statistics and depth-proxy evidence;
- explicit composition evidence that one global transform is sufficient;
- the authored output aspect, pending a future typed reframe operation;
- peak output pixels per source pixel at or below the packet's declared cap.

A contraindication is a successful, typed refusal only when no Treatment,
version, or render file is created. Analyzer-only semantic claims cannot
authorize a positive application; the current accepted authorities are a human
assertion or deterministic evaluation ground truth.

The transfer gate contains five positive classes (square illustration,
licensed photograph, portrait, alpha, and complex texture) plus low-resolution
and independent-layer negative classes. Positive cases use the ordinary
application, approval, Motion v2 compositor, preview/final QC, Kdenlive/MLT,
and OTIO paths. Every positive case is replayed in a distinct workspace and
must produce byte-equivalent final media. Every compiled frame matrix is
checked against an independent normalized-similarity calculation. Negative
decisions are independently replayed and must retain their refusal semantics.

The complete corpus advances the packet revision once. It appends all case
records atomically and leaves status `candidate`. Only a separate
human-governed Tree of Editing review may promote it.

## Consequences

- The characteristic phase relationship transfers without importing reference
  pixels or original source geometry.
- Resolution and composition failures become explicit product behavior rather
  than post-render warnings.
- Source aspect changes are handled by normalized contain geometry; output
  aspect changes still fail closed until an authored reframe exists.
- A passing corpus proves deterministic applicability and execution across the
  declared classes, not universal editorial quality or canonical knowledge.
- The system photograph remains an OS-owned runtime input with package and
  license provenance; it is not copied into Git.
