# AGENTS.md

Root route card for `aoa-editing`.

## Purpose

`aoa-editing` is the local-first AI co-editor owner for evidence-backed,
reversible video-editing decisions. It turns immutable source media and a user
intent into reviewable treatments, patches, versioned edit graphs, renders,
quality reports, and editable interchange exports.

## Owner lane

This repository owns:

- the versioned canonical edit-project schema and its invariants;
- analysis evidence contracts and provenance;
- scenario planning for `speech.clean`, `memory.montage`, and `still.motion`;
- reversible patch/version semantics and render compilation;
- the local web workbench, CLI, and agent interfaces over one application core;
- local evaluation gates, reference-isolation rules, and transfer evidence;
- derived Kdenlive/MLT and OpenTimelineIO interchange adapters.
- the repo-local project runtime and operational home containing projects,
  evals, artifacts, editor cache/tmp, and rebuildable state;
- typed declarations and adapter contracts for shared AI capabilities.

It does not own:

- user source media or the meaning of an external reference work;
- FFmpeg, Kdenlive/MLT, OpenTimelineIO, host AI runtimes, or their upstream
  contracts;
- Tree of Sophia canon. Candidate editing knowledge remains a review packet
  until its stronger owner accepts it.

## Start here

1. `README.md`
2. `DESIGN.md`
3. `docs/boundaries.md`
4. `docs/architecture.md`
5. `docs/decisions/README.md`
6. `docs/evaluation.md`
7. `docs/definition-of-done.md`
8. nearest nested `AGENTS.md`, when present
9. source and tests you plan to change

## Source-of-truth law

- Authored schemas and domain code own project meaning.
- A project workspace's immutable JSON version snapshots and referenced source
  hashes are canonical for that project instance.
- SQLite, thumbnails, proxies, renders, QC summaries, interchange files, and UI
  state are derived and may be rebuilt.
- Analysis output is evidence, never an edit decision by itself.
- A treatment is a proposal. Only an accepted patch creates a new version.
- Source assets are immutable after ingest; edits are non-destructive.
- External tools and AI models are adapters behind explicit ports. No provider
  may silently become canonical authority.

## Reference-evaluation isolation

Before `aoa-editing gate readiness` reports all generic checks green, the named
reference media may only be checked for path existence, readability, byte size,
and cryptographic hash. Do not decode frames, inspect audio, generate
thumbnails, transcribe, or otherwise analyze its content before that gate.

After the gate, reference content may be used only as evaluation evidence. It
must never become a render input, mask source, audio source, hidden fixture, or
copied project asset. Every reconstruction run must emit a source-lineage audit.

## Repo-local storage and host law

- This checkout is the default operational home. Product state lives under
  untracked `var/`; the project-specific runtime lives in untracked `.venv/`.
- Canonical projects: `var/projects`; runtime evals: `var/evals`; proof and
  product artifacts: `var/artifacts`; editor cache/tmp/state:
  `var/{cache,tmp,state}`.
- Resolve the home independently of the current working directory. An explicit
  `AOA_EDITING_HOME` may select another complete product home.
- Host-owned storage is not a default product-state root. Legacy roots are
  rollback evidence during migration, not continuing ownership.
- Shared models, host-managed AI runtimes, and large shared AI/compile/browser
  caches remain with `abyss-stack` and `abyss-machine`. Consume them through
  typed aliases and owner APIs, never filesystem links or embedded model paths.
- Read `/etc/abyss-machine/AGENTS.md` and storage policy before large writes or
  heavy AI work, and use the host resource gate for medium/heavy jobs.
- Never mutate a stack checkout or its model roots from this project.

## Mutation and verification

- Preserve unrelated user changes.
- Prefer test-first slices for stable behavior.
- Schema changes require compatibility tests and regenerated checked-in JSON
  schemas.
- Renderer changes require deterministic compiler tests plus a real FFmpeg
  smoke render.
- Interchange changes require parse validation and, when available, a real MLT
  load/render smoke.
- UI, CLI, and agent commands must call the same application services.
- Report changed owner surfaces, checks run, checks skipped, and remaining risk.
