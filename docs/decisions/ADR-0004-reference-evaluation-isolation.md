# ADR-0004: Gate and isolate reference reconstruction

- Status: accepted
- Date: 2026-07-14

## Context

- Problem: using the supplied reference too early can turn evaluation into
  target-specific implementation or leak reference media into the output.
- Constraint: the product must first prove generic capability, then recreate the
  observed result from the supplied still, repeat it cleanly, and transfer the
  method to another source.
- Evidence/context: the operator-provided goal and sealed file hashes identify
  the inputs; they do not authorize pre-gate content inspection.

## Decision boundary

- Selected lenses: evidence state, workflow/process, risk/approval.
- Owner: this repository's evaluation workflow and lineage checks.
- Does not decide: artistic authorship, subjective equality, or Tree of Sophia
  promotion.

## Options

- Considered: inspect the target during implementation or use its frames as
  fixtures.
- Rejected: either can create hidden target coupling and invalidates an honest
  transfer claim.

## Decision

Seal reference content until a machine-readable generic readiness gate passes.
Afterward treat it only as measurement/comparison evidence. Enforce forbidden
source hashes in ingest and render lineage, then require reconstruction,
clean-rerun, and transfer receipts.

## Rationale

The gate separates general product construction from target evaluation, while
lineage checks make the non-leakage claim testable rather than rhetorical.

## Consequences

- Accepted tradeoff: target-specific gaps may be discovered late and must be
  solved through general features rather than one-off branches.
- Guardrail: a skipped mandatory readiness check fails closed; reference hashes
  are allowed in evaluation metadata but forbidden in render ancestry.

## Placement verification

- Canonical decision surface: `docs/decisions/`.
- Related change surface: eval gate, sealed-input policy, lineage audit.
- Reachability: linked from the decision index, root route, and evaluation doc.
- Handoff: candidate knowledge goes to a review packet, never directly to canon.

