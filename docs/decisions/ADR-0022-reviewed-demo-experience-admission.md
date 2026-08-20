# ADR-0022: Reviewed experience admission for recurring project demos

- Status: accepted
- Date: 2026-07-21
- Extends: ADR-0007, ADR-0018, and ADR-0021

## Context

The first reviewed project-demo workflow produced useful editing rules, but its
initial application-history entries cited hand-authored `operator-review`
labels. Those labels described the source semantically without resolving to the
actual corrections, compared revisions, or owner dispositions. That was enough
to remember a lesson, but not enough to audit how it entered the product.

Project demonstrations are expected to recur. Their recordings, narration,
editorial iterations, and operator feedback may contain private paths and other
local context, while this repository is a portable public seed. The product
therefore needs a repeatable evidence boundary rather than either publishing
raw sessions or copying informal conclusions directly into candidate knowledge.

## Decision

`ScreenWorkflowExperienceAdmission` is the private owner-review contract for
one immutable external review snapshot. It records the source and review-packet
hashes, resolvable private evidence refs, separate owner-review refs, accepted,
rejected, and deferred claims, the target product surfaces, the review note,
and whether the experience is a single demo or part of a recurring demo lane.

Each claim also contains a separately authored public summary and limitation.
`workflow admit-experience` removes source observations, session identity, and
all evidence refs from that projection, hashes the canonical source-neutral
bytes, and writes an immutable full receipt under
`var/state/workflow-experience-receipts/`. Repeating the same admission is
idempotent; attempting to replace the same content-addressed receipt with other
bytes fails.

Tracked candidate knowledge may cite
`private-review-receipt:sha256:<projection-hash>#<case-id>`. It may not cite a
fabricated operator label as if that label were source evidence, and it may not
publish the private receipt, session id, raw refs, physical paths, or source
wording. The hash proves which sanitized claim set was admitted; the ignored
local receipt is required to resolve that set back to private evidence.

External session-memory tooling remains an optional evidence producer, not a
runtime dependency or knowledge authority for `aoa-editing`. Automatic
distillation and manual-review packetization remain provisional. Only explicit
owner dispositions can enter an admission, and the resulting public projection
remains candidate knowledge until broader evaluation supports promotion.

## Recurring workflow

For every substantial project demo:

1. preserve the recording, narration, edit versions, and operator review;
2. checkpoint and distill the working session through its evidence owner;
3. resolve the bounded raw/segment refs behind material claims;
4. record explicit owner dispositions and limitations in an admission;
5. create the immutable local receipt and source-neutral projection hash;
6. update only the affected contracts, candidate packets, tests, and evals;
7. use later demos as independent applications, counterexamples, or
   superseding evidence rather than silently strengthening the first case.

## Consequences

- Private evidence remains resolvable locally without entering the public seed.
- Public application history is content-addressed instead of relying on invented
  semantic labels.
- Rejected and deferred ideas remain visible, preventing a suggested temporal
  loop from being misreported as implemented behavior.
- One highly rated edit can improve candidate defaults without becoming a
  universal quality law.
- Recurring demos form an explicit improvement and regression corpus.

## Verification

Contract tests require evidence-bearing accepted claims, keep private wording
and refs out of the public projection, verify deterministic receipt identity and
idempotent storage, validate the checked-in schemas, reject legacy
`operator-review` evidence labels in the reviewed screen-workflow packet, and
retain the public-seed exclusion of `var/`.
