# Ports and adapters

| Port | Core expectation | Current adapter | Replaceable by |
| --- | --- | --- | --- |
| Project store | Atomic immutable canonical JSON, rebuildable index | Filesystem + SQLite index | transactional/local object store |
| Media probe | Metadata plus provenance | ffprobe | another decoder probe |
| Analyzer | One typed evidence capability or localized failure | FFmpeg, Pillow/NumPy, provider-alias STT through path-free host normalization | OpenVINO, pyannote, CLIP, detectors |
| Video sampling/structure | Hash-bound plan, full coverage, competing transition evidence | FFprobe/FFmpeg keyframes plus OpenCV/NumPy multi-signal detector | another calibrated detector emitting the same evidence contracts |
| Video semantics | Selected frames -> normalized observations or typed partial failure | `vision.describe` provider alias with revision-bound cache | OCR/regions/tracking/segmentation/depth adapters |
| Video motion | Shot range -> measured global/independent hypotheses and uncertainty | dense OpenCV optical flow/transform summaries | GPU flow or tracked-object adapter |
| AI provider resolver | Declaration + untracked binding + live health -> redacted receipt | shell-free machine CLI and loopback HTTP | stack service adapters with the same receipt contract |
| Editorial planner | Evidence + Brief -> reversible Treatment | deterministic scenario planners | `local-ai://editorial/plan/default` proposal |
| Voiceover timing | immutable narration + transcript/pause evidence -> review-required beat cues | segment alignment with silence-weighted fallback | stronger local forced aligner behind the same human-review gate |
| Language patch | text -> validated preview only | deterministic allowlist parser | local LLM constrained to schema |
| Compiler | canonical timeline -> deterministic plan | FFmpeg + Motion v2 matrix compiler | MLT/Blender/Remotion adapter |
| Motion compositor | derived matrices + immutable pixels -> disposable lossless frames | OpenCV premultiplied-alpha warp + FFV1 | GPU/MLT/Blender compositor |
| Renderer | plan -> atomic output + receipt + lineage | FFmpeg subprocess | process worker or render farm |
| QC | output -> typed checks | ffprobe/FFmpeg metrics | stronger perceptual/compliance suite |
| Interchange | derived editable projection + loss report | Kdenlive/MLT and OTIO | Resolve/Premiere adapters |
| Agent transport | allowlisted typed application calls | line-delimited JSON | MCP or other RPC over same core |

An adapter may produce evidence, a proposal, a plan, or a derived artifact. It
cannot mutate canonical meaning outside the application services. Provider or
model identity, revision, parameters, determinism, timestamps, and relevant
commands are persisted in provenance. `manifests/analyzers.json` distinguishes
implemented capabilities from reserved ports so absence is never presented as
successful analysis. `manifests/ai-capabilities.json` owns the path-free AI
declarations; physical bindings never enter Git.
