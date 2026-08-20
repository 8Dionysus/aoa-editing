# AoA Editing

AoA Editing is a local-first AI video co-editor. It does not hide editing behind
a one-shot prompt. It records why an edit was proposed, lets a person accept or
amend the proposal, preserves every version, renders previews and finals, checks
the result, and exports an editable timeline.

The prototype is intentionally production-shaped:

- one typed application core shared by the web workbench, CLI, and agent API;
- immutable media ingest and explicit provenance;
- evidence-backed treatments for speech cleanup, memory montage, animated
  stills, and script-bound terminal workflows;
- reversible patches, versions, comparison, preview/final render, and QC;
- FFmpeg rendering with Kdenlive/MLT as the first editable interchange lane;
- nine typed local-AI capability aliases with untracked bindings, live health,
  redacted provider receipts, and deterministic fallbacks so the product
  remains usable offline;
- executable generic, isolation, reconstruction, clean-rerun, and transfer
  evaluations, including a fresh-home v2 replay from approved typed semantics.

The authoritative architecture is in [DESIGN.md](DESIGN.md), and completion
claims are governed by [docs/evaluation.md](docs/evaluation.md).

## Start

```bash
./scripts/bootstrap
./scripts/aoa-editing doctor
./scripts/aoa-editing serve --open
```

Or install the desktop launcher once and then open **AoA Editing** from the
applications menu:

```bash
./scripts/install-desktop
```

The workbench covers project creation, local upload, analysis, treatment review,
patch editing, acceptance/revert, timeline inspection, A/B preview, preview/final
render, QC, and editable export. A gated **Референс** workspace adds synchronized
reference/candidate playback, overlays and differences, motion/derivative
curves, uncertainty, typed correction previews, per-item approve/reject, and
point rerenders without admitting reference media into render lineage. The CLI
and line-delimited JSON agent protocol call the same application services.

The checkout is the repo-local operational home: `.venv/` contains the
project-specific runtime and untracked `var/` contains projects, evals,
artifacts, editor cache/tmp, and rebuildable state. Shared AI models and
host-managed services remain with their stack/machine owners and are reached
through typed provider aliases. No cloud service or API key is required.

## Verify

```bash
./scripts/test
./scripts/eval
./scripts/aoa-editing gate readiness
./scripts/aoa-editing eval provider-aliases --output /new/provider-alias-eval
./scripts/aoa-editing eval local-ai-live --help
```

The readiness gate does not decode the operator-supplied reference. It must pass
before the reference evaluation route can open.

For a terminal-first workflow video, planning can begin before capture exists:

```bash
./scripts/aoa-editing workflow plan-script \
  --title "Codex workflow" \
  --script /path/to/script.md \
  --output var/artifacts/capture-plans/codex-workflow.json
```

The script-only result is an editorial draft. Supplying authored beat JSON and
an attributable reviewer makes it capture-ready. After the raw recording is
ingested, a separate reviewed edit spec binds every beat to a source range and
camera target; `treatment workflow` then creates the ordinary reversible
Treatment. A delivered voice recording is ingested before the screen recording
and timed against the reviewed beats:

```bash
./scripts/aoa-editing workflow time-voiceover PROJECT_ID \
  --plan var/artifacts/capture-plans/codex-workflow.json \
  --asset VOICEOVER_ASSET_ID \
  --output var/artifacts/capture-plans/codex-workflow.voiceover-draft.json

./scripts/aoa-editing workflow review-voiceover \
  --plan var/artifacts/capture-plans/codex-workflow.json \
  --draft var/artifacts/capture-plans/codex-workflow.voiceover-draft.json \
  --cues /path/to/reviewed-voiceover-cues.json \
  --reviewed-by editor \
  --review-note "Listened through and approved every beat boundary" \
  --output var/artifacts/capture-plans/codex-workflow.voiceover-reviewed.json
```

The reviewed timing may then be embedded in the post-capture edit spec. It
drives every visual beat and adds the immutable narration asset as a normalized
audio track. Reference videos remain in the gated evaluation lane and cannot
be used by any of these steps as project media.

After an operator reviews one or more delivered revisions, a private typed
admission can bind the accepted, rejected, and deferred lessons to resolvable
external evidence without publishing that evidence:

```bash
./scripts/aoa-editing workflow admit-experience \
  --admission var/state/workflow-experience-intake/project-demo.json
```

The full immutable receipt stays under `var/state`. Candidate knowledge cites
only the hash of its reviewed source-neutral projection, so future project demos
can add independent evidence or counterexamples without turning one session
into a universal editing rule.

## Status

The production-shaped prototype and its fail-closed completion chain are
executable. Generic readiness, source-only reference reconstruction, all-frame
objective comparison, clean-room replay, heterogeneous transfer, provider
aliases, and live existing-ASR admission have runtime proof lanes. A complete
operator claim additionally requires an attributable human verdict for the
exact candidate, current revision receipts, and a local-only final tag; prose
does not substitute for them. The extracted reveal technique is checked in as
a reviewable candidate, not promoted canon. See the
[completion report](docs/completion-report.md) for the exact evidence boundary.

The v2 layer now includes a seven-class transfer corpus, normalized phase-aware
motion, typed pre-render resolution/composition refusals, and the canonical
local-AI alias boundary. The motion packet remains candidate pending a separate
human Tree of Editing review. Host models are not copied into this repository;
live bindings and any remaining model gaps are audited separately against
stack/machine owner evidence.

## Documentation

- [User guide](docs/user-guide.md)
- [Agent protocol](docs/agent-protocol.md)
- [Operations and storage](docs/operations.md)
- [Ports and adapters](docs/ports-and-adapters.md)
- [Schemas and migrations](docs/schemas.md)
- [Security and privacy](docs/security-privacy.md)
- [Evaluation](docs/evaluation.md)
- [Definition of done](docs/definition-of-done.md)
- [Prototype completion report](docs/completion-report.md)
- [Roadmap](docs/roadmap.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Third-party licenses](docs/third-party-licenses.md)

## License

Apache-2.0. External executables and models retain their own licenses.
