# ADR-0003: FFmpeg rendering and Kdenlive/MLT first interchange

- Status: accepted
- Date: 2026-07-14

## Context

- Problem: the prototype needs a reliable render path and at least one genuinely
  editable external handoff without collapsing the core into an NLE format.
- Constraint: FFmpeg 8.1.2 and Kdenlive 26.04.3 with MLT 7.40.0 are live on the
  host. OpenTimelineIO adapters explicitly vary in effects and transition
  coverage.
- Evidence/context: upstream FFmpeg filter docs, MLT XML docs, Kdenlive project
  format docs, OTIO adapter docs, and live version probes.

## Decision boundary

- Selected lenses: owner/source, evidence state, handoff/fan-out.
- Owner: command compilation and export compatibility reporting here.
- Does not decide: upstream filter semantics or complete third-party NLE
  compatibility.

## Options

- Considered: Blender-only rendering, Remotion-only rendering, OTIO as the sole
  interchange, or direct automation of a proprietary NLE.
- Rejected: each adds an unnecessary hard dependency, licensing/runtime
  constraint, or insufficient effect fidelity for the baseline.

## Decision

Use FFmpeg/ffprobe as the baseline render and media-evidence backend. Export a
Kdenlive-compatible MLT XML project as the first editable interchange target and
validate it with the installed MLT runtime. Offer OTIO as a secondary timeline
projection with an explicit compatibility report.

## Rationale

The selected tools are local, scriptable, mature, and already available. MLT can
represent multitrack compositions and Kdenlive can reopen them, while FFmpeg
provides deterministic production and QC primitives.

## Consequences

- Accepted tradeoff: some AoA effects need baking, approximation, or omission in
  interchange.
- Guardrail: every export emits feature-by-feature compatibility status and the
  canonical graph remains alongside it.

## Placement verification

- Canonical decision surface: `docs/decisions/`.
- Related change surface: render compiler and interchange adapters.
- Reachability: linked from the decision index and architecture docs.
- Handoff: external files remain derived and are validated independently.

