# ADR-0020: Path-free host STT normalization adapter

- Status: accepted
- Date: 2026-07-21
- Extends: ADR-0017 and ADR-0019

## Context

The host already exposes ready local Whisper through `abyss-machine dictation`,
but its owner JSON is not identical to the AoA Editing provider contract. Host
health does not directly publish the product protocol and inventory revision,
and current VAD segment records carry time spans without per-segment text.
Binding the raw owner commands directly would therefore fail the product's
health or transcript schema. Weakening validation would make an apparently
enabled provider less trustworthy than the honest unbound state.

## Decision

AoA Editing owns a narrow `host-stt-adapter` executable behind the existing
`abyss-machine-cli` binding kind. It performs no inference itself and never
reads model files directly.

For health, it joins fresh read-only `abyss-machine ai capabilities`,
`dictation status`, and `ai models` packets. It admits readiness only when STT,
the quality profile, warm server socket, and the selected OpenVINO inventory
entry agree. The adapter emits the product protocol, backend, model id, and a
stable SHA-256 revision over the permitted owner inventory-entry fields. It
does not expose physical owner paths.

For invocation, it calls `abyss-machine dictation transcribe` and maps only the
typed transcript evidence into the provider response. Host audio/model/cache
paths are discarded. If a future host response supplies text for each timed
segment, the adapter preserves segment precision. When the current host supplies
only a full transcript plus timing-only VAD segments, the adapter emits one
file-level segment and `partial=true`; it does not fabricate segment text. The
voiceover timing service then uses measured pauses and still requires the
ordinary human review.

The executable path and the exact live model revision remain in the ignored
`.aoa-editing.local.toml`. Tracked declarations and examples remain path-free
or use explicit operator placeholders.

## Consequences

- Live host STT can satisfy the product alias without importing host code or
  persisting host filesystem topology.
- Model inventory drift turns into a health version mismatch until the local
  binding is deliberately refreshed.
- Missing segment text reduces precision rather than being silently invented.
- Host model/runtime ownership and product evidence/decision authority remain
  separate.

## Verification

Unit tests prove stable inventory fingerprinting, path redaction, file-level
partial transcript semantics, and the existing provider fail-closed behavior.
The live local overlay is checked with `provider bindings`; a real invocation
receipt is deferred until operator-supplied narration exists so no synthetic
speech is confused with final voiceover evidence.
