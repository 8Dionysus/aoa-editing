# Boundaries

## Authority order

1. authored domain models, invariants, and checked-in schemas;
2. immutable project version documents and asset hashes;
3. accepted patch and decision receipts;
4. evidence records with tool/model provenance;
5. derived indexes, renders, reports, and interchange projections.

## Adapter boundaries

- FFmpeg owns codec/filter behavior; AoA Editing owns the command plan and its
  relationship to the edit graph.
- Kdenlive/MLT owns external project interpretation; AoA Editing owns its export
  adapter and compatibility report.
- OpenTimelineIO owns its schema; an OTIO file is a derived projection and may be
  lossy for effects.
- `aoa-editing` owns its repo-local Python runtime, projects, ingest copies,
  evals, renders, QC, exports, and editor-specific cache/tmp/state.
- `abyss-stack` owns shared AI services, runtime/model recipes, capability
  registrations, and shared inference contracts.
- `abyss-machine` owns host runtime and resource truth, host-managed shared
  runtimes, and physical shared-model paths. This repository consumes its public
  CLI and never edits host inventory or stack-owned models.
- Shared AI is bound through tracked typed declarations, an untracked local
  overlay, live health, and immutable invocation receipts. A filesystem link,
  model-file presence, or embedded physical model path is not a capability
  contract. AI output authority is restricted to evidence or proposal.
- Tree of Sophia owns durable canon. This project may only emit candidate
  packets with evidence and promotion questions.
- Transfer ground truth may prove product behavior and append candidate
  application history. It cannot promote a packet or substitute for a
  human-governed Tree of Editing review.

## Placement boundary

Semantic ownership does not follow a mount-point prefix. The repository's
untracked `.venv` and `var` children remain product-owned wherever the checkout
lives. Legacy operator paths are rollback evidence, not current defaults.
Historical receipts may retain observed absolute paths outside the public seed,
while new canonical artifact identity is relative to the project or workspace
root.

## Evaluation boundary

The supplied reference video is a sealed evaluation target until the generic
readiness gate passes. After opening, it remains evidence only. Source-lineage
checks must prove every rendered stream derives from permitted reconstruction
inputs.

A batch `ReferenceWorkflowStudy` additionally requires manually reviewed,
non-overlapping ranges. Only terminal, IDE, browser-workflow, workflow-UI, or
diagram ranges explicitly marked human-free may be included. Its contact
sheets and activity measurements live under repo-local `var/evals`; the receipt
retains aliases and hashes but no source paths, source videos are never copied,
and candidate observations do not become Style Memory or Tree of Editing
canon.

Portable technique extraction may read approved typed Spec/correction semantics
after the gate, but not either media file. The checked-in packet cannot retain
the sealed hashes, physical owner paths, source-pixel matrices, or runtime
project identities. Transfer reports may retain runtime hashes and paths as
evidence; that placement does not make them knowledge authority.
