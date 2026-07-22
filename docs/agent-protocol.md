# Agent protocol

Run:

```bash
./scripts/aoa-editing agent
```

The process reads one JSON object per line and writes one response per line. It
does not execute arbitrary shell commands. Every method is allowlisted and calls
the same application core as the UI and CLI.

Request:

```json
{"id":1,"method":"project.list","params":{}}
```

Success:

```json
{"id":1,"ok":true,"result":[]}
```

Failure:

```json
{"id":2,"ok":false,"error":{"code":"method_not_allowed","message":"shell.exec"}}
```

## Methods

| Method | Required params | Effect |
| --- | --- | --- |
| `project.list` | none | Read project manifests |
| `project.create` | `name`, `intent`, `scenario` | Create project |
| `project.show` | `project_id` | Read project/assets/versions |
| `project.revise_brief` | `project_id`, partial `intent`, `rationale` | Append Brief revision |
| `asset.ingest` | `project_id`, `source` | Immutable ingest |
| `asset.analyze` | `project_id`, `asset_id` | Produce evidence |
| `evidence.correct` | project/asset/kind/payload/supersedes/rationale | Append human correction |
| `style.confirm` | scope, `explicit_opt_in=true`, confirmations | Save explicit Style Memory |
| `style.list` | none | Read confirmed profiles |
| `treatment.propose` | `project_id` | Produce treatment |
| `treatment.options` | `project_id` | Produce executable creative variants |
| `treatment.accept` | `project_id`, `treatment_id` | Create version |
| `workflow.plan_script` | title, script, optional authored beats/reviewer/study ids | Media-free capture plan |
| `workflow.time_voiceover` | project, plan, ingested voiceover asset, optional frame rate/STT flag | Review-required narration cue draft |
| `workflow.review_voiceover` | plan, draft, complete cues, reviewer/note | Immutable reviewed narration timing |
| `workflow.propose_treatment` | project, plan, reviewed edit spec | Reversible screen-workflow Treatment |
| `reference.workflow_study.run` | plan path, new eval output root | Gated human-free batch study |
| `patch.preview_language` | project, command, optional version | Typed diff only |
| `provider.status` | optional `probe` | Read declarations, local binding states, and current health |
| `provider.invoke` | alias, payload, privacy/fallback options | Invoke through the shared resolver and return a noncanonical receipt |
| `reference.workspace.list` | `project_id` | Read project-local registrations |
| `reference.workspace.attach` | project, study path, optional local reference path | Hash-verified gated registration |
| `reference.workspace.show` | project, workspace | Derived all-frame workbench bundle |
| `reference.correction.preview_language` | project, workspace, command, optional version | Independent semantic diffs only |
| `reference.correction.preview_operations` | project, workspace, typed operations | Manual semantic diffs only |
| `reference.correction.review` | project, proposal, every approve/reject decision | Human evidence and optional reversible version |
| `version.revert` | `project_id`, `version_id` | Apply inverse as new version |
| `render.run` | project/version/profile, optional start/duration frames | Full or point render |
| `render.qc` | same | Produce QC report |
| `export.kdenlive` | project/version | MLT export and live validation |
| `export.otio` | project/version | OTIO export and round-trip |

Paths remain subject to project-store validation and sealed-hash policy. Agent
errors are returned as data; malformed requests do not terminate the stream.
