# Troubleshooting

## `doctor` is not ready

Run `./scripts/aoa-editing doctor --json`. FFmpeg and ffprobe are mandatory;
Kdenlive/MLT and host AI are optional adapters. The baseline remains usable
without an LLM or ASR model. If a mandatory system package truly needs a
privileged install, stop after research, show the exact package/command, and use
the operator-approved `pkexec` flow. Never request a password in chat.

The report begins with the resolved repo-local home and every owned sub-root.
If it points at a legacy operator location without an explicit compatibility
override, treat that as configuration drift.

## A source is rejected

The file must be readable and supported by ffprobe. A sealed reference hash is
intentionally rejected from project ingest. Original media is copied, re-hashed,
and made read-only; fix the input outside AoA Editing rather than modifying its
stored source.

## Derivatives are missing or stale

Re-run ingest for the same source or invoke analysis again. A valid derivative
manifest is reused. Missing or hash-mismatched regenerable artifacts are removed
and rebuilt; abandoned partials are cleaned first. The immutable source hash is
checked before and after.

## A render failed

Inspect the matching `jobs/job_*.json` and `render-plan.json`. Failed renders do
not promote `video.partial.mp4`. Correct the typed timeline or environment and
retry. Point renders use `--start-frame` and `--duration-frames`; both options
must be supplied together and must remain inside the timeline.

## The UI created data but did not update

Run `./scripts/ui-smoke --output var/tmp/ui-smoke` from the repository home. It
drives the real form, Brief revision, Style confirmation, project selection, and
tab interaction in headless Chromium. Check its JSON receipt and screenshot.

## State after a crash or deleted index

Restart normally. SQLite is reconstructed from canonical project JSON. Re-run
the interrupted idempotent action; immutable assets, decisions, and versions are
not overwritten. Never restore from SQLite alone.

## Clean safely

`./scripts/clean` is a dry run. `./scripts/clean --apply` removes only
repo-local test caches and the resolved home's `var/cache` and `var/tmp`
children. It refuses paths outside the product-owned home and never deletes
projects, sources, renders, or evaluation data.

## A migrated project contains old absolute paths

Old evidence, job, render-plan, and completion receipts preserve paths they
actually observed and remain historical proof. New canonical records must use
IDs, hashes, and project-relative artifact references. Run the relocation
validator; do not edit historical receipts in place or add a compatibility
filesystem link.

## Reference commands fail closed

The latest readiness receipt must pass and match the current Git revision. A
code or documentation commit intentionally invalidates the old gate. Run
`./scripts/aoa-editing gate readiness` again; do not bypass the revision check.
