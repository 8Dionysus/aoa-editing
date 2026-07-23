# Definition of done ledger

This ledger separates implemented surfaces from evaluation proof. Runtime
receipt paths are populated in closeout reports rather than committed here.

| Requirement | Owner surface | Mandatory proof |
| --- | --- | --- |
| Clean reproducible repository | Git, bootstrap, lock | clean bootstrap + clean status |
| No-terminal editing workflow | web workbench + desktop launcher | real Chromium interaction and API E2E |
| CLI and agent parity | application services, CLI, JSON agent | contract tests |
| Immutable source/provenance | project store | copy/hash/read-only/idempotency tests |
| Versioned schemas/migrations | models, `schemas/v1`, migrations | drift and compatibility tests |
| Evidence-linked reversible decisions | treatment/patch/version | inverse round-trip and E2E |
| Preview/final render | FFmpeg compiler/service | all three synthetic scenarios |
| QC | quality service | duration/stream/black checks on real renders |
| Editable export | MLT + OTIO | real `melt` load and OTIO round-trip |
| Three scenarios | scenario planners | generic E2E receipt |
| Restart/resume | atomic store, idempotent ingest/analysis, partial render | interruption/idempotency tests |
| Partial evidence and corrections | analyzer port, authority/supersession | injected failure and human-correction tests |
| Brief and Style Memory | immutable Brief revisions, explicit Style profiles | API, agent, and Chromium consent path |
| Editing language | typed effects, captions, gain curves, pin/exclude, segment render | compile and real FFmpeg tests |
| Reference isolation | lock, gate, forbidden lineage | pre-gate stat/hash receipt and negative tests |
| Independent recreation | reference eval | frozen Comparison v2 protocol, all-frame objective receipt, and attributable human verdict |
| Clean rerun | clean-room evaluator | saved intent/decisions reproduce artifact |
| Transfer | transfer corpus v2 | five heterogeneous applications, two typed refusals, source-only lineage, matrix equivalence, QC/exports, distinct-workspace replay |
| Knowledge candidate | Tree of Editing packet | curve/phase model, evidence and resolution constraints, positive/negative history, non-claims, promotion questions |
| AI provider boundary | capability catalog, local overlay, resolver, receipts | all failure/privacy/fallback cases, live doctor binding, human supersession, baseline without AI |
| Live local AI admission | stack registration, machine health, product alias eval | existing ASR revision/license/byte/resource proof, real alias receipt, synthetic WER/CER versus no-AI baseline, zero new model downloads |
| Honest closeout | completion audit | every row linked, no mandatory skip |
