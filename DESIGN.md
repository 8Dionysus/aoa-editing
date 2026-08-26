# Design

## Product contract

The product is an inspectable co-editor, not an opaque generator. A complete
editing loop is:

```text
immutable source + intent
  -> measured evidence
  -> one or more treatments
  -> accepted patch
  -> immutable project version
  -> preview/final render
  -> QC and comparison
  -> editable interchange export
```

Every arrow has a typed contract and provenance. No analysis result mutates a
timeline; no render output changes the project; no external NLE export becomes
the project source of truth.

## Bounded contexts

| Context | Owns | Does not own |
| --- | --- | --- |
| Project | assets, intent, versions, patches, decisions | media decoding |
| Evidence | measurements, transcripts, detections, confidence, provenance | editorial acceptance |
| Video Anatomy | sampling plans, shots, boundaries, selected frames, synchronized audio/visual/motion evidence | timeline mutation or editorial truth |
| Treatment | scenario-specific proposals and alternatives | canonical version state |
| Edit graph | timeline/layers/effects/keyframes and invariants | renderer commands |
| Rendering | deterministic compilation, jobs, artifacts | edit meaning |
| Quality | technical and editorial checks, comparison evidence | acceptance authority |
| Interchange | lossy, reported projections for external NLEs | canonical storage |
| Interaction | web, CLI, and agent request/response adapters | duplicated business rules |
| Preproduction | script beats, reviewed voiceover cues, capture instructions, terminal framing defaults | unreviewed analyzer timing or raw-screen source ranges as timeline authority |
| Evaluation | readiness, isolation, reconstruction, rerun, transfer | product shortcuts |
| Operational home | repo-local runtime, projects, evals, artifacts, cache/tmp/state | shared AI models or host truth |
| AI capability boundary | declarations, aliases, local bindings, invocation receipts, product evals | shared service composition, host deployment, model files, canonical edit authority |

## Operational ownership

The repository home is both the source root and the owner boundary for
project-specific runtime and state. `.venv/` and `var/` are untracked runtime
surfaces; moving the complete home must not change project identity. Canonical
records use IDs, hashes, and project-relative artifact references. Absolute
origin paths are allowed only as provenance.

Shared AI is deliberately outside this boundary. AoA Editing owns typed
capability declarations and adapter contracts; `abyss-stack` owns shared
services and recipes, while `abyss-machine` owns host resolution, resource
evidence, and physical shared-model paths. Tracked `local-ai://` declarations
resolve through an untracked overlay. Every invocation records live health,
model/license/hash authority, privacy, fallback, and evidence-or-proposal
authority; a provider output cannot directly become an edit.

## Growth seams

The first implementation is deliberately modular where future pressure is
expected: source analyzers, planners, segmentation/depth, render backends,
quality metrics, interchange targets, and agent transports. The stable center is
the project/evidence/treatment/edit-graph contract plus a relocatable
repo-local operational home.

Video Anatomy uses that stable center rather than a parallel graph. Its
progressive pipeline ends at immutable evidence; editorial and reconstruction
interpretations are separate proposals. Only explicit review can compile a
proposal through the existing reversible patch/version boundary. See
`docs/video-anatomy.md` for profiles, provider, and storage contracts.

## Non-goals of the first prototype

- replacing a full NLE UI;
- training foundation models;
- frame-perfect compatibility with every NLE;
- claiming semantic vision when only deterministic image decomposition ran;
- treating similarity to one reference as general product quality.
