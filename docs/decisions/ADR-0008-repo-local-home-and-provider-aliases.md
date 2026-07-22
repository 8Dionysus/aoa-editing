# ADR-0008: Repo-local operational home and typed provider aliases

- Status: accepted
- Date: 2026-07-19
- Supersedes: storage-placement clause of ADR-0002 only

## Context

ADR-0002 correctly made AoA Editing local-first and kept optional AI behind
provider ports. Its final placement clause also routed project data, editor
cache/tmp, and the project-specific Python runtime into a host-owned root.
That conflated physical placement with semantic ownership:

- the source checkout already lives on the capacious `/srv` filesystem;
- projects, evals, editing artifacts, and the product runtime belong to
  `aoa-editing`;
- `abyss-machine` owns host facts, resource gates, host-managed runtimes, shared
  models, and physical paths for shared AI;
- `abyss-stack` owns shared AI service composition and capability
  registrations.

The completed `prototype-v0.1.0` state and its absolute-path receipts are valid
historical evidence. They must remain readable without making their old paths
the continuing architecture.

## Decision boundary

This decision partially supersedes ADR-0002. It preserves ADR-0002's
local-first baseline, deterministic fallbacks, optional provider ports,
network-opt-in posture, and provenance requirements. It supersedes only the
claim that product-owned state and the project-specific Python runtime belong
under `abyss-machine` roots.

Semantic ownership and physical placement are separate facts. A path located on
`/srv` does not thereby belong to `abyss-machine`.

## Decision

The default operational home is the repository home:

```text
<repository-root>/
  .venv/
  var/
    projects/
    evals/
    artifacts/
    cache/
    tmp/
    state/
  .aoa-editing.local.toml
```

`.venv`, `var`, and the local overlay are untracked. `aoa-editing` owns every
path in this topology. Canonical documents use IDs, content hashes, and paths
relative to project or workspace roots. An absolute origin path is ingest
provenance, not artifact identity.

Shared AI remains outside this home. The project declares typed capability
aliases and adapter contracts; stack and machine owner layers resolve those
aliases to live providers and record a receipt. A filesystem symlink, copied
model path, or physical model path in the domain core is not a provider
binding.

Old `AOA_EDITING_DATA_ROOT`, `AOA_EDITING_CACHE_ROOT`,
`AOA_EDITING_TMP_ROOT`, and `AOA_EDITING_RUNTIME_ROOT` inputs may be accepted
temporarily as compatibility overrides. They are not the primary design and
must never silently restore legacy defaults.

## Migration and rollback

Cutover follows:

```text
inventory
  -> dry run
  -> copy or rebuild by classification
  -> hash and semantic verification
  -> application validation
  -> default cutover
  -> completion proof
  -> separately authorized legacy cleanup
```

Canonical documents, immutable ingest copies, and required proof artifacts are
copied and verified. `.venv`, SQLite indexes, derivatives, cache, and tmp are
rebuilt where safe. The versioned relocation manifest records old and new
locations without rewriting historical observations.

Legacy operator roots remain available for rollback.
No compatibility filesystem symlink is created. No legacy cleanup is implied by
successful cutover; deletion requires a fresh, explicit operator decision.

## Consequences

- A checkout is a self-contained product home while shared AI remains
  owner-correct.
- Desktop, CLI, UI, agent, tests, and evals resolve one home independent of the
  current working directory.
- Host resource and storage gates still govern medium/heavy work and shared
  runtime artifacts.
- Historical receipts may contain old absolute paths; new canonical identity
  must not.
- Relocation and compatibility behavior become tested product contracts.

## Placement verification

- Owner contract: `manifests/storage-layout.json`.
- Pre-migration evidence: operator-local receipts under
  `var/artifacts/migrations` (excluded from the public seed).
- Implementation surfaces: configuration, bootstrap/launch scripts, store,
  doctor, cleanup, evaluation gates, desktop projection, and tests.
- Historical anchor: annotated tag `prototype-v0.1.0`.
