# ADR-0014: Reference workspace and human-gated motion corrections

- Status: accepted
- Date: 2026-07-19

## Context

Reference Motion Evidence v2 and Comparison v2 explain the selected
reconstruction, but JSON reports and static plots alone do not give an editor a
safe way to inspect that explanation, challenge it, or turn a phrase such as
“мягче подкручиваться и чуть позже оседать” into a reviewable change.

Letting a language model emit FFmpeg, rewrite a render plan, or mutate a version
would bypass the canonical edit graph. Copying the reference into project assets
would violate evaluation isolation. Treating a moving pivot as measured would
also fabricate a decomposition where pivot and translation are
gauge-equivalent.

## Decision

AoA Editing registers one immutable `ReferenceWorkspaceRegistrationV2` beneath
the project `comparison/` tree. The registration joins the frozen reconstruction
study, spec, all-frame reference evidence, selected comparison, candidate motion
recovery, project version, source asset, and review artifacts by IDs and
SHA-256. Paths are relative to the editing home or project root. The reference
video itself is not copied into canonical project state: an untracked
hash-keyed binding beneath `var/state/reference-media-bindings/` resolves its
physical host path.

Every workbench read and media response requires a currently passing
revision-bound readiness gate and revalidates the registered hashes. Artifact
roles are allowlisted; no arbitrary evaluation path is served. Reference and
reference-derived artifacts remain evaluation-only and are explicitly forbidden
as render inputs.

The workspace exposes synchronized reference/candidate playback, overlays,
differences, all-frame transform and derivative curves, phase markers,
confidence, uncertainty, residual flow, variant interpretations, and version
A/B state. An analyzer-owned interpretation evidence record is preserved as the
baseline.

Natural-language and manual controls produce only allowlisted
`MotionCorrectionOperationV2` values. Each operation receives an independent
typed diff and an executable/blocker result. A human must approve or reject
every item and provide a rationale. A pivot correction fails closed when pivot
is not identifiable.

An accepted review compiles only approved operations against the exact
base-effect hash, records a separate `human-correction` evidence revision, and
submits one ordinary reversible `EditPatch` through `EditingService`. Rejected
items never enter the version. Proposals and reviews are immutable. The web
workbench, CLI, and allowlisted agent protocol call the same two application
services.

## Consequences

- Reference understanding becomes inspectable and correctable without making
  the reference a project source.
- Editorial language is preserved as semantic intent rather than renderer
  syntax.
- Stale versions, changed hashes, incomplete decisions, and unsupported
  decompositions fail before canonical mutation.
- Human interpretation supersedes analyzer evidence explicitly without erasing
  history or altering the frozen Spec v2.
- Approved corrections remain compatible with version history, inverse patches,
  point renders, QC, editable exports, and deterministic replay.
