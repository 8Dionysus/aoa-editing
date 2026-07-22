# Prototype completion report

## Outcome

AoA Editing is a working, local-first, production-shaped co-editor prototype.
The web workbench, CLI, and line-delimited JSON agent surface share one typed
application core. Media ingest is immutable; evidence, human decisions, edit
graphs, patches, versions, renders, QC, and editable exports are explicit and
inspectable.

Completion is not inferred from this document. The executable completion audit
fails closed if the worktree or readiness revision differs, an input hash
changes, reference media leaks into render lineage, a mandatory check is
skipped, or a definition-of-done row lacks passing evidence.

## Public evidence boundary

This repository is a portable public seed. It contains the implementation,
schemas, synthetic fixtures, evaluation definitions, safe configuration
templates, and tests. It intentionally does not contain:

- operator media or the operator-specific `evals/reference.lock.json`;
- reference-derived frames, audio, transcripts, or render artifacts;
- machine snapshots, physical storage topology, or personal paths;
- runtime project identifiers, migration receipts, or completion receipts.

Those records remain operator-owned runtime evidence under ignored `var/`
paths. The public `evals/reference.lock.example.json` and
`manifests/environment.example.json` document the required shapes without
publishing a host snapshot.

## Reproducible proof chain

The executable proof order is:

1. bootstrap the pinned Python runtime;
2. run schemas, Ruff, Mypy, unit/integration tests, UI smoke, and doctor;
3. create an ignored reference lock from the public template;
4. pass the revision-bound generic readiness gate;
5. run independent reference reconstruction and comparison;
6. replay the selected semantics in a clean home;
7. transfer the technique to an unrelated synthetic source;
8. join every receipt with the fail-closed completion audit.

The reference remains evaluation evidence and is forbidden from render lineage.
The checked-in reveal technique remains a reviewable candidate rather than
durable canon.

## Verify the public seed

```bash
./scripts/bootstrap
./scripts/test
./scripts/eval
./scripts/aoa-editing doctor --json
```

Reference evaluation additionally requires:

```bash
cp evals/reference.lock.example.json evals/reference.lock.json
# Replace every placeholder path, size, and hash with operator-owned values.
./scripts/aoa-editing gate readiness
```

## Honest growth boundary

This is a complete prototype, not a claim of feature parity with a mature NLE.
The remaining layers are listed in `docs/roadmap.md`: process-isolated jobs and
checkpoints, semantic vision/audio adapters, richer direct timeline and
keyframe manipulation, higher-fidelity interchange, compositor ports, broader
transfer evaluation, and eventual human-governed Tree of Editing promotion.
Those additions should preserve the canonical edit graph, immutable history,
evidence/decision boundary, common application core, and source-lineage law.
