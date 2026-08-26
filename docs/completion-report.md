# Prototype completion report

## Product outcome and claim boundary

AoA Editing is a working, local-first, production-shaped co-editor prototype.
The web workbench, CLI, and line-delimited JSON agent surface share one typed
application core. Media ingest is immutable; evidence, human decisions, edit
graphs, patches, versions, renders, QC, and editable exports are explicit and
inspectable.

This document describes the available product and proof architecture; it does
not certify a particular operator run. Completion is emitted only by
Completion Audit v2. The audit fails closed if current revision receipts drift,
an input or evidence hash changes, Spec/Protocol/Reconstruction/Clean Rerun
bytes do not join, reference media leaks into render lineage, a mandatory check
is skipped, the attributable human verdict is absent, the final local tag is
missing, publication cannot be disproved, or a Definition of Done row lacks
passing evidence.

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
4. pass the independent network-free Video Anatomy corpus;
5. pass the revision-bound generic readiness gate;
6. pass the independent ground-truth motion-recovery gate;
7. run the sealed reference through `quick`, `structural`, and `reconstruct`
   Video Anatomy profiles and review its source-neutral proposal;
8. freeze all-frame Motion Evidence v2, Reconstruction Spec v2, and Comparison
   Protocol v2;
9. accept through the normal reversible version path and render independently;
10. compare the selected candidate on all objective axes;
11. inspect and correct motion through the real Reference Workspace UI;
12. attach an attributable human review for the exact candidate hash;
13. replay the selected semantics in a clean home;
14. transfer the technique across seven heterogeneous source classes;
15. pass provider-alias and live existing-model admission gates;
16. create the final local-only tag;
17. join every receipt, including the exact Video Anatomy eval, with Completion
    Audit v2.

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

## Current honest boundary

Machine receipts can prove objective similarity, replay, transfer, ownership,
and provider behavior, but they cannot author the human editorial verdict. An
objective-only Comparison v2 remains `warn`, and Completion Audit v2 remains
`fail`, until the operator reviews the final side-by-side, contact sheet, and
aligned difference and decides every rubric axis. Static documentation must not
be read as that verdict.

## Honest growth boundary

This is a complete prototype, not a claim of feature parity with a mature NLE.
The remaining layers are listed in `docs/roadmap.md`: a process-isolated queue
built on the Video Anatomy checkpoint spine, additional semantic vision/audio
adapters, richer direct timeline and
keyframe manipulation, higher-fidelity interchange, compositor ports, broader
transfer evaluation, and eventual human-governed Tree of Editing promotion.
Those additions should preserve the canonical edit graph, immutable history,
evidence/decision boundary, common application core, and source-lineage law.
