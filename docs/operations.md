# Operations

## Repo-local runtime and storage

| Purpose | Default |
| --- | --- |
| Repository home | `<repository-root>` |
| Python runtime | `<home>/.venv` |
| Canonical projects | `<home>/var/projects` |
| Runtime evaluations | `<home>/var/evals` |
| Product artifacts | `<home>/var/artifacts` |
| Regenerable editor cache | `<home>/var/cache` |
| Temporary work | `<home>/var/tmp` |
| Rebuildable state/indexes | `<home>/var/state` |

`AOA_EDITING_HOME` selects another complete home. A repo-local
`.aoa-editing.local.toml` may bind local provider aliases without entering Git.
Legacy per-root environment variables are compatibility inputs during
migration, not defaults. Resolution never depends on the current working
directory.

Shared models, host-managed AI runtimes, and shared AI/compile/browser caches
remain outside the product home with their stack/machine owners. Medium and
heavy work still passes through `abyss-machine resource launch`.

## Local AI aliases

Tracked declarations live in `manifests/ai-capabilities.json`. Copy only
owner-proven bindings from `.aoa-editing.local.example.toml` into the ignored
`.aoa-editing.local.toml`; replace every inventory placeholder from a current
stack/machine receipt. Do not put credentials or model paths in either file.

For the host Whisper binding, run `./scripts/host-stt-adapter health` and copy
its exact backend, model id, and model revision into the ignored overlay. The
adapter joins current owner read models, removes physical paths, and marks the
transcript partial when the host supplies only file-level text. Do not bind the
raw dictation JSON directly to the product alias.

```bash
./scripts/aoa-editing provider catalog
./scripts/aoa-editing provider bindings
./scripts/aoa-editing provider invoke \
  local-ai://speech/transcript/default \
  --request /path/to/local-request.json \
  --no-fallback
```

The resolver validates the tracked request/response schemas, privacy and live
health, then writes one immutable receipt under
`var/state/provider-receipts/`. Receipt parameters retain semantic options but
redact runtime media paths and credential-like fields. Disabled, unavailable,
and unbound capabilities are honest normal states; deterministic editing does
not depend on them.

The Phase-15 live admission is deliberately separate from the mocked alias
contract corpus. Capture the small official upstream metadata JSON, then run
the whole evaluator through the host resource gate:

```bash
abyss-machine resource launch \
  --class medium \
  --kind ai \
  --memory-demand-mib 1280 \
  --demand-key aoa-editing-local-ai-live \
  --demand-owner aoa-editing \
  --estimate-source warm-owner-service-plus-model-hash \
  --estimate-confidence high \
  -- \
  ./scripts/aoa-editing eval local-ai-live \
    --output /new/repo-local/eval-root \
    --model-owner-root /owner/stack/model-root \
    --upstream-revision-file /owner/cache/model/refs/main \
    --upstream-metadata-file /local/captured/model-metadata.json \
    --stack-source-registration /owner/stack/source/dictation.service \
    --stack-deployed-registration /owner/stack/deploy/dictation.service \
    --stack-managed-units /owner/stack/deploy/managed-units.txt
```

The 1280 MiB reservation is the observed client peak rounded up with operating
margin; it is not a static memory cap. The output hashes every selected
model-export file but does not copy weights. It generates unrelated known-text
speech, invokes the real alias, measures WER/CER, service and evaluator-unit
memory, and runtime fit, then compares against an isolated product home with no
AI binding. The reference and user media are not inputs.

## Bootstrap

`scripts/bootstrap` uses CPython 3.12, pinned runtime/dev dependencies, a pinned
build backend, and pinned pip. It installs the project editable without resolving
undeclared dependencies, then runs `doctor`.

No privileged operation is currently required. If a future required system
package is missing, use the operator-approved `pkexec` route; never request or
capture a password. An unavailable authorization must be reported as a blocker.

## Health and verification

```bash
./scripts/aoa-editing doctor --json
./scripts/verify
./scripts/ui-smoke --output var/tmp/ui-smoke
```

`doctor` distinguishes mandatory FFmpeg/ffprobe from optional Kdenlive/MLT and
host AI capabilities. The product binds to `127.0.0.1:8787` by default.

## Command map

| Intent | Command |
| --- | --- |
| Bootstrap pinned runtime | `./scripts/bootstrap` |
| Start no-terminal workbench | `./scripts/launch` or desktop entry |
| Test all contracts | `./scripts/test` |
| Generic evaluation | `./scripts/eval` |
| Video Anatomy estimate/run | `./scripts/aoa-editing anatomy estimate --help` / `anatomy run --help` |
| Video Anatomy synthetic gate | `./scripts/aoa-editing eval video-anatomy --output /new/root` |
| Video Anatomy cache preview/apply | `./scripts/aoa-editing anatomy prune-cache` / `anatomy prune-cache --apply` |
| Draft narration timing | `./scripts/aoa-editing workflow time-voiceover --help` |
| Review narration timing | `./scripts/aoa-editing workflow review-voiceover --help` |
| Admit reviewed demo experience | `./scripts/aoa-editing workflow admit-experience --help` |
| Provider alias contract gate | `./scripts/aoa-editing eval provider-aliases --output /new/root` |
| Live existing-model admission | `./scripts/aoa-editing eval local-ai-live --help` under the medium AI resource gate |
| Sealed readiness gate | `./scripts/aoa-editing gate readiness` |
| Freeze Comparison v2 | `./scripts/aoa-editing eval reference-freeze-comparison-v2 --help` |
| Run/review Comparison v2 | `./scripts/aoa-editing eval reference-compare-v2 --help` / `reference-review-v2 --help` |
| Run bounded Reconstruction v2 pass | `./scripts/aoa-editing eval reference-reconstruct-pass-v2 --help` |
| Verify and summarize Reconstruction v2 study | `./scripts/aoa-editing eval reference-summarize-study-v2 --help` |
| Attach reference understanding workspace | `./scripts/aoa-editing reference attach --help` |
| Inspect all-frame reference workspace | `./scripts/aoa-editing reference show --help` |
| Preview semantic motion correction | `./scripts/aoa-editing reference preview-language --help` |
| Review individual corrections | `./scripts/aoa-editing reference review --help` |
| Replay selected v2 semantics in a clean home | `./scripts/reproduce-reference --help` or `./scripts/aoa-editing eval clean-rerun-v2 --help` |
| Safe cleanup preview/apply | `./scripts/clean` / `./scripts/clean --apply` |

Video Anatomy keeps immutable packets and selected frame artifacts inside each
project. Temporary decode products use `var/tmp`; provider-result cache uses
`var/cache/video-anatomy`. `anatomy prune-cache` is dry-run first and cannot
delete project analysis, proposals, versions, renders, or evaluation roots. A
deep profile performs resource preflight and fails before decode when the
required free-space reserve is unavailable.

`scripts/reproduce-reference` requires a frozen Spec v2, the selected passing
`5r` reconstruction receipt, and a new output root. It performs the independent
Clean Rerun v2 only. Objective comparison, human review, transfer, provider
proof, and final completion remain separate authority lanes and are joined only
by `eval completion-audit`.

`workflow time-voiceover` requires a reviewed capture plan and an already
ingested narration asset. It writes a new timing draft and may invoke only the
typed local transcript alias; a missing provider falls back to measured pauses.
`workflow review-voiceover` requires a complete cue list plus reviewer and
review note, writes a separate immutable reviewed revision, and never mutates
the draft. Keep its delivery frame rate equal to the later screen recording.

`reference attach` accepts the immutable Reconstruction Study v2 and optionally
the host-local reference file. Only the reference hash and an untracked binding
are retained; the physical path does not enter the project registration.
`reference show`, artifact playback, and correction preview/review all fail
closed unless the current Git revision has a passing readiness receipt. A
language preview never writes a version. The `review` input must decide every
proposal item; only approved items pass through the normal reversible patch
service.

`clean-rerun-v2` requires the frozen Spec v2, the passing selected `5r`
reconstruction receipt, and a completely absent output root. It copies neither
the baseline project nor its renders. The command consumes the permitted PNG
plus the receipt's typed phase-correction semantics, recreates the full
application path, and fails unless semantic fingerprints, preview/final hashes,
lineage, inodes, QC, and editable exports all prove independence and
equivalence.

## Backup and restore

Back up a whole project directory under `var/projects/project_*`. The authoritative
files are `project.json`, `assets/*/asset.json`, immutable source files,
`evidence/*.json`, `treatments/*.json`, `patches/*.json`, and `versions/*.json`.
`index.sqlite3`, renders, QC, and exports are derived or rebuildable. Briefs,
decision graphs/logs, and style profiles are authoritative JSON and must be
included when relevant. Never back
up only SQLite and assume the project is preserved.

## Legacy rollback boundary

Legacy operator roots remain untouched during copy/rebuild and cutover. Raw
inventories and cutover receipts live only under the ignored
`var/artifacts/migrations` tree. Successful validation does not authorize
legacy cleanup; cleanup requires a separate report and fresh operator approval.
