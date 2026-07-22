# ADR-0007: Candidate editing knowledge and license-aware adapters

- Status: accepted
- Date: 2026-07-15

## Context

Scenario behavior must be reusable and reviewable without prematurely turning
one successful edit into universal law. Renderer, semantic-analysis, and NLE
alternatives also carry different Linux, headless, API, resource, and license
constraints.

## Decision

Store scenario and technique knowledge as versioned `TechniquePacket` documents
with applicability, contraindications, evidence, allowed operations, rules,
heuristics, variants, failure modes, QC, examples, application history, and a
candidate/evaluated/canonical status. The machine layer may be a graph; “Tree of
Editing” is a navigation and review projection. This repository emits
candidates only; Tree of Sophia remains the canon owner.

Keep adapters optional until their value and legal/runtime boundary is proven.
FFmpeg plus MLT is the baseline. Whisper, pyannote, image-text models,
segmentation, Blender, Premiere UXP, Resolve scripting, and Remotion remain
ports or evaluated alternatives. No model or proprietary SDK is silently
bundled. Every promoted adapter needs a version, license, model hash or external
owner reference, privacy behavior, compatibility tests, and provenance.

## Consequences

- One reference reconstruction can create only a candidate packet.
- Transfer to unrelated media is required before even considering evaluated
  status; canonical promotion requires stronger-owner review and broader cases.
- License or runtime changes do not force a canonical edit-graph migration.

## Verification

Checked-in scenario packets validate against the generated schema. The
reference extractor rejects failed comparisons and media-hash leakage; transfer
adds a second application record without target media.
