# ADR-0017: Typed local-AI alias resolution and noncanonical provider receipts

- Status: accepted
- Date: 2026-07-19
- Extends: ADR-0002 and ADR-0008

## Context

ADR-0002 established optional provider ports and a deterministic local-first
baseline. ADR-0008 assigned capability declarations and bindings to
`aoa-editing` while keeping shared services/models with `abyss-stack` and host
runtime truth with `abyss-machine`.

Those decisions did not yet provide an executable contract. The analyzer still
called the host transcription command directly, `doctor` exposed only a raw
host inventory, and there was no typed distinction between:

- a product capability declaration;
- an untracked host-specific binding;
- current live health;
- one invocation and its provider/model provenance;
- AI evidence or proposal versus canonical edit authority.

Presence of a model file or a remembered command is not capability evidence.

## Decision

`manifests/ai-capabilities.json` is the tracked product declaration catalog. It
contains exactly the initial speech, vision, and editorial capability surface.
Every declaration has a canonical `local-ai://.../default` alias, JSON input and
output schemas, required/optional status, privacy and authority classes, latency
and temporal precision, fallback/failure behavior, provenance requirements, and
a fail-closed health contract.

The deterministic baseline requires none of these providers.

`.aoa-editing.local.toml` is the only default local binding overlay and remains
untracked. A binding may select:

- an `abyss-stack` service;
- an `abyss-machine` CLI;
- a credential-free localhost endpoint;
- an explicit disabled or unavailable state.

Bindings contain owner-backed model metadata and evidence references, but no
physical model path or secret. Commands are argument arrays executed without a
shell. HTTP endpoints must be loopback and credential-free. A filesystem
symlink is never a capability or binding contract.

Before invocation, the resolver validates the request and privacy decision,
then requires fresh provider health. Missing, disabled, unavailable, stale,
protocol/model mismatch, timeout, and malformed-response states are typed and
produce receipts. Deterministic fallback is separately labelled and remains
owned by `aoa-editing`.

Every attempt atomically writes a `ResolvedProviderReceipt` beneath
`var/state/provider-receipts`. It records the requested alias, owner, adapter,
backend, model identity/revision and hash/license authorities, owner evidence,
endpoint or command template, redacted parameters, live health, execution time,
privacy decision, output authority, fallback state, response digest, and typed
outcome. Input paths and credential-like values are not persisted.

Provider output authority is restricted to `evidence` or `proposal`. It can
enter a canonical project only through the existing human correction,
Treatment, patch, and approval services. The default transcript analyzer now
uses this alias service; it no longer calls the host command directly.

## Failure and privacy posture

- An absent optional provider cannot make bootstrap, deterministic analysis, or
  deterministic planning unavailable.
- A provider failure is localized and cannot erase sibling evidence.
- `local-only` requests cannot cross an external data boundary.
- External execution requires both the explicit privacy mode and an explicit
  invocation opt-in.
- Partial output remains explicitly partial evidence.
- Human correction supersedes provider evidence without deleting history.
- Receipt/log surfaces redact secrets and runtime media paths.

## Verification

`aoa-editing eval provider-aliases` is the revision-bound contract gate. It
exercises available, missing, stale, version-mismatch, timeout, malformed,
privacy-denied, fallback, and partial cases; proves human supersession; runs all
three deterministic product scenarios without AI; checks the domain for model
paths; and makes `doctor` resolve and health-check a real untracked fixture
binding.

The evaluator proves the product boundary, not the existence or quality of a
particular host model. Phase-specific live provider and model evidence remains
owned by the relevant stack/machine inventory and a separate product invocation
receipt.

## Consequences

- UI-adjacent doctor, CLI, agent, and analyzers use one resolver service.
- Host deployment may change without changing project declarations or domain
  models.
- Provider quality can be compared against the deterministic baseline without
  turning AI output into editorial authority.
- A new capability requires a declaration/schema/eval change; a new host
  binding does not require a tracked source change.
