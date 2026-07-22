# ADR-0002: Local-first runtime with optional provider ports

- Status: accepted; storage placement partially superseded by ADR-0008
- Date: 2026-07-14

> Partially superseded by ADR-0008. The local-first, optional-provider, and
> provenance decisions remain accepted. The final sentence assigning
> product-owned data and the project-specific Python runtime to
> The legacy host root is historical and no longer governs current placement.

## Context

- Problem: a prototype tied directly to one model, cloud API, GPU, or host path
  will be brittle and difficult to transfer.
- Constraint: the current host has FFmpeg, OpenVINO Whisper profiles, and a local
  text-model endpoint, while heavy artifacts must live under
  legacy host storage and stack-owned model roots are read-only.
- Evidence/context: live `abyss-machine ai capabilities` and storage policy are
  runtime evidence, not repository authority.

## Decision boundary

- Selected lenses: owner/source, portability/overlay, runtime/body.
- Owner: provider interfaces and discovery/fallback behavior in this repository.
- Does not decide: host runtime lifecycle, model inventory, or upstream model
  quality.

## Options

- Considered: bundle a fixed AI runtime/model or require remote APIs.
- Rejected: both increase installation and privacy risk and make generic tests
  depend on an unstable external capability.

## Decision

Make the product local-first. Deterministic analyzers and planners provide a
complete baseline. Host STT and local LLM capabilities are optional adapters
discovered at runtime. Large project data, caches, temporary work, and Python
runtime were routed to a host-owned root for the v1 prototype. ADR-0008
supersedes that placement clause with a product-owned repo-local home.

## Rationale

The core remains reproducible and testable while stronger local AI can improve
evidence or proposals without gaining authority over project state.

## Consequences

- Accepted tradeoff: output quality may differ by available provider and must be
  recorded in provenance.
- Guardrail: `doctor` reports capability status and never claims an unavailable
  model; network providers are opt-in.

## Placement verification

- Canonical decision surface: `docs/decisions/`.
- Related change surface: provider ports, configuration, bootstrap, doctor.
- Reachability: linked from the decision index and architecture docs.
- Handoff: host policy remains stronger for live resource/storage facts.
- Supersession: ADR-0008 owns product-home placement and provider-alias
  binding; this ADR continues to own the local-first and provider-port choice.
