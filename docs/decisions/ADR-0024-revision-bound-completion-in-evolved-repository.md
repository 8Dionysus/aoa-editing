# ADR-0024: Revision-bound completion in an evolved repository

- Status: accepted
- Date: 2026-07-23
- Extends: ADR-0010, ADR-0015, ADR-0017, and ADR-0023

## Context

The original private prototype goal assumed a repository with no remote. The
repository later acquired a deliberate public root and a pre-existing `origin`
while the private reconstruction evidence remained on a preserved local
history. Removing that remote would erase useful repository configuration but
would not prove that the goal's commits or final tag were never published.

The proof lanes also matured independently. A readiness receipt, frozen motion
spec, comparison protocol, selected reconstruction, clean replay, transfer
corpus, provider gate, and live-model gate can each be valid while still
referring to different bytes or implementation revisions. The v1 completion
join cannot safely certify that expanded state.

## Decision

Completion uses a fail-closed `CompletionAuditReport` v2. It maps every current
Definition of Done row to explicit checks and hashes every selected receipt.
The audit additionally binds:

- Reference Motion Evidence v2 to its recovery gate;
- Reference Reconstruction Spec v2 to the selected reconstruction;
- Comparison v2 to the exact Spec and Comparison Protocol bytes;
- Clean Rerun v2 to the exact selected reconstruction receipt and Spec;
- the replayed final to the objectively compared candidate hash;
- current readiness, motion recovery, provider aliases, and live local-AI
  admission to the audited `HEAD`.

Historical private evidence may join a public-root history only when a named
local private-history ref has the exact same tree as the single public root,
every historical implementation revision is an ancestor of either current
history or that private ref, and no critical Motion v2 owner path drifted after
the public root. This is a proof bridge, not a history rewrite.

An existing remote is admissible only through an explicit
`--allow-preexisting-remote` opt-in. The audit must query every configured
remote and prove that:

- no commit after the recorded goal boundary is reachable by a remote head or
  tag;
- the audited `HEAD` is not present on a remote;
- the requested completion tag is absent remotely;
- the working branch has no upstream;
- all remote queries succeed.

A repository with no remote still passes without the opt-in. The invariant is
non-publication of this goal's work, not deletion of repository configuration.
The final local tag must point exactly to the clean audited revision.

Objective comparison never substitutes for editorial authority. A passing
completion receipt still requires a named human operator to accept every
frozen rubric axis for the exact candidate hash.

## Alternatives considered

- Delete `origin` before closeout. Rejected because configuration absence is
  not publication evidence and would discard newer repository state.
- Trust the local branch name or lack of upstream. Rejected because remote
  commits and tags must be inspected directly.
- Treat preserved private evidence as unrelated. Rejected when an exact
  public-root tree bridge proves byte-equivalent owner state.
- Accept receipt paths without cross-hashes. Rejected because individually
  valid but mutually incompatible proof lanes could otherwise be joined.

## Consequences

- Completion is intentionally unavailable while the human verdict or final
  local tag is missing.
- Immutable historical evidence remains reusable without pretending it was
  created on the public-root commit graph.
- Current runtime/provider claims must be rerun on the eventual final commit.
- Remote inspection is read-only; this decision does not authorize push, PR,
  release, tag publication, or mutation of remote state.
- Network or authentication failure during remote inspection fails the
  publication check rather than being converted into a skip.

## Verification

The hermetic completion test proves both a fully joined pass and a missing
human-review failure. Final runtime verification must run
`aoa-editing eval completion-audit` with the exact evidence paths, goal-start
revision, local tag, preserved private-history ref, and—only for an already
evolved repository—`--allow-preexisting-remote`.
