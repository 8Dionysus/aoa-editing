# ADR-0015: Approved-semantics clean replay v2

- Status: accepted
- Date: 2026-07-19

## Context

The selected reconstruction pass `5r` contains a frozen Spec v2 and a bounded
phase correction derived from an earlier Comparison v2. Re-deriving that
correction during a clean replay would make the replay depend on the old
candidate and comparison workspace. Copying the old project or render would
prove only relocation, not independent reproducibility.

The phase correction is analyzer-owned approved execution semantics. It is not
the still-pending human editorial verdict, and replaying it must not change that
authority boundary.

## Decision

`clean-rerun-v2` starts from an absent output root and creates a separate
repo-local Editing Home. Its only media input is the hash-locked source PNG. It
loads the exact frozen Spec v2 and the selected passing reconstruction receipt,
then reuses the receipt's typed `ReferencePhaseCorrectionV2` directly. The
reconstruction runner records `phase_correction_source=approved_semantics` and
does not open the prior comparison report, candidate motion report, reference
video, or baseline render.

The replay still follows the ordinary application path: create project, seal
the forbidden reference hash, ingest and analyze the source, propose a
Treatment, approve its Decision Graph, create a reversible patch/version,
render preview and final, run QC, and validate Kdenlive and OTIO exports.

`CleanRerunReportV2` records complete hash and inode proofs for the baseline and
replay. The gate requires:

- distinct project, asset, version, paths, and inodes;
- identical normalized timeline semantics;
- byte-equivalent preview and final outputs;
- source-only render lineage and no copied reference bytes;
- absence of baseline paths from replay commands;
- recreated derivatives and canonical application records;
- passing preview/final QC and parser-backed editable exports;
- no hidden manual steps and zero mandatory skips.

The original user files remain outside both projects and are only read for
source ingest or hash verification. Runtime evidence remains untracked under
`var/evals`.

## Consequences

- Clean replay is independent of chat history and old derivatives.
- The selected temporal refinement is reproducible without turning comparison
  media into an execution dependency.
- A passing replay proves deterministic execution, not human editorial
  indistinguishability; the latter remains a separate attributable review.
- Any future change to approved semantics produces a different typed hash and
  requires a new baseline rather than silently weakening equivalence.
