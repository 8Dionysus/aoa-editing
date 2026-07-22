# AGENTS.md

This directory owns executable evaluation definitions and sealed-input metadata.

- Generic fixtures must be synthetic and materially unrelated to the supplied
  reference.
- `reference.lock.example.json` is the public template. The ignored
  `reference.lock.json` may contain paths, roles, sizes, and hashes before the
  readiness gate; it must not contain decoded content observations.
- Reference frames, thumbnails, masks, waveforms, transcripts, or audio are
  forbidden before a passing readiness receipt.
- After readiness, reference-derived artifacts live under the resolved
  repo-local `<home>/var/evals` root, never in Git. Shared AI or browser caches
  used by an adapter remain with their stack/machine owners.
- Evaluation evidence does not become product authority or Tree of Sophia canon.
- A skipped mandatory check is a failure.
