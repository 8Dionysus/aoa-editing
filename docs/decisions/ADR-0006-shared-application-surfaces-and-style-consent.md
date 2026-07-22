# ADR-0006: Shared application core and explicit style consent

- Status: accepted
- Date: 2026-07-15

## Context

The no-terminal workbench, automation CLI, and agent integration must not create
different project meanings. Natural-language output is untrusted until it is a
typed, validated diff. Style learning can become a hidden authority or privacy
leak if inferred from ordinary editing behavior.

## Decision

FastAPI/web, Typer CLI, and the allowlisted JSON agent protocol call the same
`EditingService`, store, analyzers, compiler, QC, and export services. The agent
has no shell method. Natural-language commands may only create `PatchPreview`;
application requires a separate confirmation through the normal patch contract.

Style Memory consists only of immutable `StyleProfile` records submitted with
`explicit_opt_in=true`. Each value is an `explicit-user-confirmation`. The
system does not infer a profile from timelines, prompts, accept/reject actions,
or media. Briefs are independently versioned and remain project-scoped intent.

## Consequences

- Direct UI edits and agent edits have identical validation/version behavior.
- Richer LLM providers can replace the deterministic language adapter but may
  emit only the same typed preview contract.
- Automatic preference learning is intentionally absent; future aggregation
  must preserve consent, provenance, scope, inspection, and deletion policy.

## Verification

API/agent contract tests, rejected `shell.exec`, natural-language preview tests,
explicit-opt-in validation, immutable Brief revisions, and the real Chromium
smoke constrain this decision.
