# Definition of done ledger

This ledger separates implemented surfaces from evaluation proof. Runtime
receipt paths are populated in closeout reports rather than committed here.

| Requirement ID | Requirement | Owner surface | Mandatory proof |
| --- | --- | --- | --- |
| `clean-reproducible-repository` | Clean reproducible repository | Git, bootstrap, lock | clean bootstrap + clean status |
| `product-owned-storage` | Product-owned storage | `EditingHome`, relocation manifest, legacy boundary | repo-local runtime/state roots, hash-verified migration, no legacy symlink or default write |
| `no-terminal-editing-workflow` | No-terminal editing workflow | web workbench + desktop launcher | real Chromium interaction and API E2E |
| `cli-agent-parity` | CLI and agent parity | application services, CLI, JSON agent | contract tests |
| `immutable-source-provenance` | Immutable source/provenance | project store | copy/hash/read-only/idempotency tests |
| `versioned-schemas-migrations` | Versioned schemas/migrations | models, `schemas/v1`, migrations | drift and compatibility tests |
| `evidence-linked-reversible-decisions` | Evidence-linked reversible decisions | treatment/patch/version | inverse round-trip and E2E |
| `preview-final-render` | Preview/final render | FFmpeg compiler/service | all three synthetic scenarios |
| `qc` | QC | quality service | duration/stream/black checks on real renders |
| `editable-export` | Editable export | MLT + OTIO | real `melt` load and OTIO round-trip |
| `three-scenarios` | Three scenarios | scenario planners | generic E2E receipt |
| `restart-resume` | Restart/resume | atomic store, idempotent ingest/analysis, partial render | interruption/idempotency tests |
| `partial-evidence-corrections` | Partial evidence and corrections | analyzer port, authority/supersession | injected failure and human-correction tests |
| `brief-style-memory` | Brief and Style Memory | immutable Brief revisions, explicit Style profiles | API, agent, and Chromium consent path |
| `editing-language` | Editing language | typed effects, captions, gain curves, pin/exclude, segment render | compile and real FFmpeg tests |
| `motion-ground-truth-recovery` | Motion ground-truth recovery | synthetic corpus and recovery gate | every known curve/model recovered with uncertainty and zero mandatory skips on current revision |
| `reference-motion-evidence-v2` | Reference Motion Evidence v2 | all-frame analyzer and phase model | every frame, derivatives, ambiguity, residual diagnostics, and evidence-backed twist explanation |
| `motion-language-execution-v2` | Motion Language execution v2 | Decision Graph, compiler, compositor | deterministic phase-compensated pass, preview/final QC, editable exports, and exact Spec binding |
| `human-correctable-motion-ui` | Human-correctable motion UI | Reference Workspace v2 | real Chromium synchronized players, diagnostics, typed correction preview/review, and screenshot |
| `reference-isolation` | Reference isolation | lock, gate, forbidden lineage | pre-gate stat/hash receipt and negative tests |
| `independent-recreation` | Independent recreation | reference eval | exact Spec/Protocol hash join, all-frame objective receipt, and attributable human verdict |
| `clean-rerun-v2` | Clean rerun v2 | clean-room evaluator | exact selected receipt and approved semantics reproduce semantic and byte-equivalent outputs in a distinct home |
| `transfer-corpus-v2` | Transfer corpus v2 | transfer corpus v2 | five heterogeneous applications, two typed refusals, source-only lineage, matrix equivalence, QC/exports, distinct-workspace replay |
| `knowledge-candidate` | Knowledge candidate | Tree of Editing packet | curve/phase model, evidence and resolution constraints, positive/negative history, non-claims, promotion questions |
| `ai-provider-boundary` | AI provider boundary | capability catalog, local overlay, resolver, receipts | all failure/privacy/fallback cases, live doctor binding, human supersession, baseline without AI |
| `live-local-ai-admission` | Live local AI admission | stack registration, machine health, product alias eval | existing ASR revision/license/byte/resource proof, real alias receipt, synthetic WER/CER versus no-AI baseline, zero new model downloads |
| `publication-boundary` | Publication boundary | Git history and read-only remote inspection | final local tag on HEAD; no goal commit/tag remotely; no upstream; remote absence or explicit verified pre-existing-remote mode |
| `honest-closeout` | Honest closeout | completion audit | every row linked, no mandatory skip |
