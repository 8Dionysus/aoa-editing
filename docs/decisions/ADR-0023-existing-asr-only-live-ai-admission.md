# ADR-0023: Admit one existing ASR and defer unproven model expansion

- Status: accepted
- Date: 2026-07-23
- Extends: ADR-0017 and ADR-0020

## Context

The typed alias boundary and path-free host STT adapter make a live provider
possible, but neither proves that the selected model is the owner-registered
runtime, that its source revision and license are known, that its exported
bytes are stable, or that it improves a product capability over the
deterministic baseline.

The host also inventories embeddings, a resident text model, multimodal
projectors, and accelerator devices. Inventory presence is not a product gap,
a live stack capability registration, or permission to download another model.
The current product already has deterministic audio facts, image statistics,
saliency/depth proxies, scenario planners, typed corrections, and a complete
reference reconstruction path.

## Decision

Phase 15 admits exactly one existing capability:

`local-ai://speech/transcript/default`

It resolves through the ignored local overlay to the `abyss-machine` dictation
CLI, which consumes the `abyss-stack`-registered warm Whisper service. The
selected model remains owned by `abyss-stack`; hardware/runtime health remains
owned by `abyss-machine`; the normalization adapter, alias receipt, and product
quality gate remain owned by AoA Editing.

No model is downloaded, copied, converted, or installed for this admission.
The live gate must:

- match the local binding to fresh owner health;
- prove source/deploy stack service-registration parity and live service state;
- join the exact upstream revision with official license metadata;
- hash every regular file in the selected OpenVINO export and record a tree
  digest;
- measure device route, service memory, health latency, invocation latency, and
  real-time factor;
- transcribe known synthetic Russian speech through the real alias;
- compute frozen WER and CER thresholds;
- run the same fixture through an isolated no-AI product home and prove that
  technical audio evidence survives a localized `provider_missing`;
- preserve local-only privacy, `evidence` authority, and no-fallback status.

Every other declared capability receives an explicit gap decision:

- standalone VAD stays unbound because deterministic audio analysis and
  dictation-internal VAD already cover the current requirement;
- diarization stays deferred without a multi-speaker workflow and registered
  owner pipeline;
- description, semantic regions, and segmentation stay deferred until a
  reviewed editing operation needs them;
- image-text retrieval stays unbound until a corpus and retrieval workflow
  exist;
- editorial plan and critique stay unbound until a product utility eval proves
  an advantage over deterministic planning and human typed correction;
- a video-language model is not justified.

## Consequences

- Existing host capability becomes executable product evidence without moving
  model or runtime ownership into this repository.
- Model inventory drift, stack registration drift, license/revision mismatch,
  unavailable service health, quality regression, or a direct un-gated run
  fails closed.
- A partial file-level transcript may pass lexical quality while remaining
  explicitly partial temporal evidence.
- Future model additions require a new product gap, owner registration,
  resource/license/revision evidence, and a separate comparison gate.
- The deterministic application remains fully usable with no local binding.

## Verification

Run `aoa-editing eval local-ai-live` through
`abyss-machine resource launch --class medium --kind ai`. The revision-bound
`local-ai-integration.json` receipt and its model-file manifest are runtime
evidence under the repo-local `var/evals` root; physical owner paths never enter
the tracked declaration catalog or domain identity.
