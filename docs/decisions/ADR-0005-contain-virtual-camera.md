# ADR-0005: Make fit and camera placement explicit

- Status: accepted
- Date: 2026-07-15

## Context

- Problem: a cover-first render pipeline discards source bounds before motion,
  so a virtual camera cannot finish on a complete square image inside a portrait
  canvas.
- Pressure: post-gate reference measurement exposed a continuous global source
  transform that moves from a cropped close view to a contained full view.
- Constraint: the capability must transfer to unrelated images; measured target
  values belong to an eval treatment, never to compiler branches.

## Decision boundary

- Selected lenses: representation, workflow/process, and downstream impact.
- Owner: canonical `TransformEffect` semantics and the FFmpeg compiler adapter.
- Does not decide: the creative trajectory for a particular image, semantic
  layer segmentation, or promotion of a technique packet.

## Options

- Keep cover preprocessing and special-case the reconstruction.
  - Rejected because it loses source pixels and couples the renderer to one eval.
- Bake the animation to frames before normal rendering.
  - Rejected because it weakens editability, provenance, and clean reproduction.
- Define fit and placement in the canonical transform, then compile them.
  - Selected because the edit graph remains authoritative and reusable.

## Decision

`TransformEffect` explicitly declares `fit_mode` (`cover` or `contain`) and
`position_mode` (`offset` or normalized `canvas_center`). Scale is relative to
the selected fit. Rotation is expressed in degrees. Ordered keyframes use the
easing declared on the destination keyframe and compile into deterministic
piecewise FFmpeg expressions.

For `contain`, the compiler preserves up to a two-times output-resolution
virtual canvas, rotates the fitted source before transparent padding, and only
then samples the camera crop. This avoids destroying source detail before a
close-up while bounding memory and render cost for very large images.

The reference analyzer may estimate scale, rotation, and normalized source
center with OpenCV SIFT plus RANSAC after readiness. It emits parameters and
confidence only into an isolated evaluation spec. Reference pixels and derived
frames remain forbidden as project or render inputs.

## Consequences

- Benefit: the same camera can animate any suitable still from close crop to a
  complete contained view, including rotation and non-linear motion.
- Accepted cost: dynamic rotation and zoom add render work, and very dense
  measured curves can produce larger filter expressions.
- Compatibility: existing transforms retain `cover` plus `offset` defaults.
- Guardrails: keyframe frames are strictly increasing; scale stays positive;
  eval-specific coordinates are data in a treatment, not source code.

## Verification

- Contract tests reject unordered curves and round-trip generated schemas.
- A real render test proves a different square fixture becomes fully visible on
  a portrait canvas.
- A synthetic affine-video test proves the measurement adapter recovers scale,
  rotation, placement, and an ease-in-out curve within declared tolerances.
- The generic three-scenario gate must pass again before live reference analysis.

## Placement verification

- Canonical decision surface: `docs/decisions/`.
- Related owners: `domain.models`, `render.compiler`, `analysis.reference`, and
  the isolated reference evaluator.
- Reachability: decision index, architecture, evaluation, and schema docs.
