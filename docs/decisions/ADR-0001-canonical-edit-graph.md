# ADR-0001: Canonical edit graph with immutable versions

- Status: accepted
- Date: 2026-07-14

## Context

- Problem: UI state, an NLE project, a renderer command, or an AI response could
  each become an accidental source of truth and make edits irreproducible.
- Constraint: edits must be inspectable, reversible, portable, and usable from
  web, CLI, and agent interfaces.
- Evidence/context: OpenTimelineIO documents that only its native representation
  is lossless for OTIO features and that adapters vary by feature; Kdenlive
  project files are MLT XML references interpreted by an external application.

## Decision boundary

- Selected lenses: owner/source, evidence state, handoff/fan-out.
- Owner: this repository's domain models and schemas.
- Does not decide: external NLE semantics, codec behavior, or whether an AI
  proposal is editorially good.

## Options

- Considered: make OTIO, MLT XML, or SQLite the project source of truth.
- Rejected: each either loses product-specific decision/evidence meaning, binds
  the core to one consumer, or hides reviewable state in a database.

## Decision

Use a versioned AoA Edit Graph JSON contract as canonical project meaning.
Asset sources and versions are immutable. Treatments remain proposals; accepted
patches produce new version snapshots. SQLite and all external formats are
rebuildable indexes or projections.

## Rationale

This keeps stable editorial semantics independent from rendering and external
tools while preserving human-readable artifacts for testing, diffing, and
recovery.

## Consequences

- Accepted tradeoff: adapters must explicitly report unsupported or lossy
  features.
- Guardrail: schema changes require compatibility/migration tests; projections
  never write back silently.

## Placement verification

- Canonical decision surface: `docs/decisions/`.
- Related change surface: `src/aoa_editing/domain/`, `schemas/`, project store.
- Reachability: linked from this directory index and root route card.
- Handoff: render, interchange, UI, and agent adapters consume the graph.

