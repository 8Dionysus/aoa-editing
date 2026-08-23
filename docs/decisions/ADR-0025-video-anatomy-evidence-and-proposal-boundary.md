# ADR-0025: Video Anatomy evidence and proposal boundary

- Status: accepted
- Date: 2026-08-23

## Context

AoA Editing can probe media, detect FFmpeg scene-score timestamps, transcribe
speech through a typed provider, and recover motion densely for bounded
reference evaluations.  Those capabilities do not yet form a general,
inspectable account of an arbitrary video's shots, transitions, sampled frames,
audio structure, semantic observations, coverage, and unresolved regions.

A useful video-understanding loop needs a cheap whole-video pass followed by
focused passes over uncertain or information-dense ranges.  The resulting
measurements must remain evidence.  Editorial interpretation and reconstruction
must remain separately reviewable proposals, and neither may mutate the
canonical edit graph without the existing patch/version acceptance boundary.

## Decision

AoA Editing owns a standalone `Video Anatomy` contour:

```text
immutable asset
  -> sampling plan
  -> structure/frame/audio/motion/visual evidence
  -> Video Anatomy aggregate
  -> editorial or reconstruction proposal
  -> explicit human review
  -> ordinary reversible edit patch and version
```

The contour has the following rules:

1. `VideoSamplingPlan` is deterministic and content-addressed.  Profile,
   range, budgets, pinned timestamps, detector parameters, and required
   capabilities participate in its hash.
2. Frame samples retain exact source time, selection reason, role, artifact
   hash, extraction provenance, and seek limitations.  Boundary and pinned
   samples are protected from deduplication.
3. `VideoStructureEvidence` covers the requested time range with explicit shot
   intervals, transition candidates, conflicts, gaps, and uncertainty.  An
   unsupported classification is `unknown`, never an inferred nearest type.
4. Visual, audio, and motion records keep their own provenance and may be
   partial independently.  Optional providers cannot erase deterministic
   sibling evidence or make the structural profile unavailable.
5. `VideoAnatomy` is an immutable aggregate over evidence references and
   content hashes.  It is not an edit decision.
6. `EditorialStructureProposal` and `ReferenceReconstructionProposal` are
   separate proposal artifacts.  Reconstruction semantics must be
   source-neutral and cannot contain reference pixels, copied audio, masks, or
   source-specific hidden inputs.
7. Proposal acceptance compiles through the existing Treatment/EditPatch,
   Editorial Decision Graph, and immutable ProjectVersion path.  Analysis never
   calls that acceptance path automatically.
8. All application surfaces call one Video Anatomy application service.  AI
   models are optional provider adapters behind path-free capability aliases;
   Claude, Codex, skills, Abyss services, and host runtimes are not product
   dependencies.
9. Human-readable reports, contact sheets, indexes, and UI projections are
   derived and rebuildable.  Canonical identity remains in typed JSON records,
   source hashes, plan hashes, and evidence lineage.

## Compatibility

The existing `video.scenes` evidence kind remains readable and usable.  The new
structure evidence is additive and may cite the legacy record as one detector
input.  Existing projects require no rewrite; new schemas use the current v1
schema directory while retaining explicit model schema versions.

The sealed-reference readiness gate remains unchanged.  Before readiness, the
named reference may only be statted and hashed.  After readiness, Video Anatomy
artifacts remain evaluation evidence and are forbidden from candidate render
lineage.

## Consequences

- General video understanding becomes a first-class product capability rather
  than a free-form multimodal prompt.
- Coverage and uncertainty are inspectable and can drive bounded rescans.
- Deep semantic and motion work can be added without changing edit authority.
- More typed artifacts and migration tests are required, but provider and UI
  implementations remain replaceable.
