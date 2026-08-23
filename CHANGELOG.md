# Changelog

## 0.2.0 - 2026-08-23

- Added progressive Video Anatomy profiles, adaptive sampling, contiguous shot
  and boundary evidence, protected multi-signal deduplication, exact VFR timing,
  contact sheets and synchronized audio evidence.
- Added provider-neutral visual observations and dense per-shot motion with
  typed partial and failure behaviour.
- Added immutable anatomy, editorial/reconstruction proposals, explicit human
  review, reversible acceptance and dense normalized Motion v2 compilation.
- Added durable phase jobs, checkpoints, cancellation/retry/resume,
  revision-bound caches, resource admission and cache-prune receipts.
- Added shared CLI, HTTP, agent and workbench Video Anatomy surfaces plus an
  independent network-free evaluation corpus.
- Sealed reference analysis remains readiness-gated and no reference pixels,
  audio, masks or source-specific matrices can enter reconstruction lineage.
- Product provenance now reports package version `0.2.0` instead of hard-coded
  prototype version strings.

## Unreleased

- Added reviewed transcript/pause-backed voiceover timing for `screen.workflow`,
  with immutable narration assets, CLI/API/agent parity, and canonical audio
  mixing in reversible Treatments.
- Added a path-free `abyss-machine` STT normalization adapter so live owner
  health and partial/file-level transcripts satisfy the typed provider contract
  without leaking model or media paths.

- Returned the project-specific runtime, projects, evals, artifacts, and
  editor-owned state to one repo-local operational home.
- Added a typed `StorageLayout`, read-only legacy inspection, dry-run and
  verified relocation receipts, rollback preservation, and fail-closed cleanup
  boundaries.
- Rebuilt `.venv` and SQLite from declared sources, migrated four v1 projects
  and required proof artifacts, and preserved the v1 final render hash.
- Added a source-independent ground-truth motion laboratory with 23 fixture
  families, all-frame similarity/affine/homography recovery, temporal
  derivatives, model mismatch, uncertainty, and a fail-closed recovery gate.
- Added Reference Motion Evidence v2 and an immutable v2 reconstruction spec
  with every-frame matrices, residual optical flow, uncertainty intervals,
  phase/twist localization, diagnostics, and frozen comparison thresholds.
- Made Reference Motion Evidence v2 distinguish stable channel phase lag and
  curved center motion from derivative noise through local-polynomial
  kinematics and explicit scale/rotation/center coupling diagnostics.
- Added canonical Motion Language v2 with rational subframes, Bezier/Hermite
  and monotone curves, C1/C2 validation, bounded overshoot, angle unwrap,
  editable path/pivot curves, coupled phases, explicit transform order,
  matrix motion, deterministic evaluation/retiming, and a v1 compatibility
  migration.
- Added a derived matrix-compositor adapter with source-hash enforcement,
  premultiplied-alpha warping, deterministic uniform-shutter sampling,
  lossless disposable intermediates, direct-matrix profile scaling, bounded
  segment renders, cleanup, and render lineage.
- Added Comparison Protocol v2 with revision/spec binding, every-frame geometry
  and derivatives, phase/coupling and optical-flow agreement, VMAF/SSIM/PSNR,
  black/edge/blur diagnostics, review artifacts, source-only lineage, and a
  separate immutable human editorial verdict.

## 0.1.0 - Unreleased

- Established the AoA Editing owner, canonical edit graph, immutable versions,
  reversible patches, schemas, migrations, and filesystem store.
- Added FFmpeg ingest/evidence/render, all three planners, QC, MLT/Kdenlive and
  OTIO interchange.
- Added a local web workbench, desktop launcher, CLI, and allowlisted agent
  protocol over one application core.
- Added synthetic E2E evaluations and a fail-closed sealed-reference readiness
  gate.
- Added explicit contain/cover virtual-camera semantics, eased multi-keyframe
  compilation, rotation, and gated affine reference measurement.
