# ADR-0009: Motion Language v2 and the derived matrix compositor

- Status: accepted
- Date: 2026-07-19

## Context

Motion Language v2 can express editable decomposed motion as well as a measured
source-to-output matrix curve when pivot or transform order is not identifiable.
FFmpeg's stock transform filters cannot preserve all of that meaning:

- `zoompan` splits path and scale semantics and has frame-local expression
  constraints;
- arbitrary affine and homography curves need one common transform convention;
- exact subframe sampling and motion-blur accumulation must not be hidden in a
  renderer-specific expression;
- a reconstruction must be reproducible from the approved edit graph without
  reading reference pixels or a prior render.

Treating per-frame matrices or a compositor intermediate as canonical state
would invert the authority boundary. Treating an approximate FFmpeg expression
as equivalent would silently weaken the motion contract.

## Decision boundary

Motion Language v2 remains canonical domain data inside an approved timeline
projection and Editorial Decision Graph. A render plan may derive matrices and
disposable media from it, but those artifacts never become edit authority.

This decision adds a renderer adapter. It does not make OpenCV, FFV1, or a
particular sampling count part of project meaning, and it does not permit
reference media to enter render lineage.

## Decision

The compiler deterministically lowers each active `transform_v2` effect into a
typed `CompositorJobV2`:

1. validate that every motion curve covers local frame zero through the final
   clip frame;
2. evaluate exact rational sample times for every requested output frame;
3. compile decomposed motion in the selected output profile, or scale a direct
   matrix from canonical output pixels into that profile;
4. record every source-to-output 3x3 matrix, timeline frame, interpolation
   policy, immutable source hash, and encoder command in the render plan;
5. let the matrix compositor verify the source hash, warp premultiplied-alpha
   pixels, accumulate declared samples, and emit a lossless FFV1 intermediate;
6. let the ordinary FFmpeg pipeline apply the remaining supported effects,
   composition, audio, and final encoding;
7. validate and atomically promote the result, persist lineage, then delete only
   compiler-declared disposable intermediates.

Single-sample evaluation is centered on the frame. Multi-sample evaluation
requires an explicit non-zero uniform shutter interval. Boundary samples clamp
to the authored curve endpoints. Sample order and arithmetic are deterministic.

A segment render derives only the compositor frames intersecting its requested
canonical frame range. Their `output_frame` indices restart at zero while their
absolute `timeline_frame` values and curve sample times remain unchanged.

The baseline compositor accepts immutable still images. Canonical speed retiming
must happen before compilation; a Motion v2 clip with a legacy speed effect
fails closed. Masks also fail closed until a typed compositor mask stage exists.
A clip cannot carry both legacy and v2 transform generations.

## Interchange

OTIO and MLT/Kdenlive are derived projections. An adapter must preserve the
canonical Motion v2 payload in namespaced metadata when its format allows that
without claiming native editability. Any native approximation or omission is
listed in the compatibility report. Importing an interchange file cannot
silently replace the approved canonical effect.

## Verification

- Domain and schema tests cover sampling invariants and typed render plans.
- Compiler tests cover decomposed and direct-matrix paths, profile scaling,
  deterministic subframe matrices, and bounded segment jobs.
- A real FFmpeg test exercises premultiplied-alpha warping, multi-sample
  accumulation, final encoding, cleanup, deterministic rerender, restart, and
  inverse patch behavior.
- Render lineage records source hashes and compositor parameters; sealed
  reference hashes remain forbidden.

## Consequences

- Motion semantics are renderer-independent and can grow toward another
  compositor without changing canonical projects.
- Measured matrices can be reproduced without inventing an unidentifiable
  pivot decomposition.
- Render plans are larger, but auditable and sufficient to explain every
  sampled transform.
- OpenCV and FFV1 are implementation dependencies of the current adapter, not
  hidden authorities.
- Unsupported masks, implicit retiming, and interchange loss remain explicit
  instead of degrading silently.
