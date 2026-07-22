# Architecture

## Shape

AoA Editing uses a ports-and-adapters shape around a typed domain and
application core.

```text
Web workbench   CLI   Agent JSON protocol
       \         |         /
        Application services
                 |
   Project / Evidence / Treatment / Edit graph
      |          |          |          |
 filesystem   analyzers   planners   compiler
 + SQLite     + AI ports             + QC
      |                               |
 immutable media                  FFmpeg
                                  MLT / OTIO exports
```

## Canonical and derived state

Canonical project state is stored as versioned JSON documents in a project
workspace beneath the repo-local operational home. Asset identity is
content-addressed. Versions are immutable; patches reference a base version and
produce a new version. SQLite is a rebuildable query/job index. Renders,
proxies, thumbnails, waveform data, interchange files, and UI caches are
derived.

The project manifest points to immutable Editorial Brief revisions. An approved
Editorial Decision Graph is the editing authority for each version. The timeline
is its executable projection; render plans and NLE files are downstream again.
Evidence has algorithm/model provenance, rational time bases and ranges,
confidence, authority, and explicit supersession. Human corrections win in the
effective view without deleting analyzer history.

## Runtime boundaries

FFmpeg/ffprobe is the baseline decoder and renderer. Kdenlive/MLT is an external
editable interchange consumer. Host Whisper and the resident local LLM are
optional adapters discovered by `doctor`; deterministic planners and analyzers
remain available if those services are absent.

The checkout resolves one repository home independent of the process CWD.
Project-specific Python lives in `.venv`; projects, evals, artifacts,
cache/tmp, and rebuildable state live under untracked `var/`. Shared AI does not
enter that tree: the repository declares typed capability aliases,
`abyss-stack` owns shared service composition, and `abyss-machine` resolves
host facts and resource gates. Provider resolution never changes canonical
project authority.

The provider boundary has three distinct layers. The tracked capability catalog
defines nine path-free `local-ai://` contracts. The ignored
`.aoa-editing.local.toml` resolves only locally proven stack services,
machine CLIs, loopback endpoints, or explicit inactive states. The application
resolver validates JSON input/output, privacy and fresh health, then atomically
writes a redacted `ResolvedProviderReceipt`. Missing, stale, mismatched,
timed-out, and malformed providers remain typed failures. AI output is only
evidence or proposal; human correction and ordinary Treatment approval remain
the authority boundary.

Still-image transforms preserve the source through an explicit `cover` or
`contain` fit before virtual-camera scale, rotation, and normalized placement.
Multi-keyframe easing is canonical edit data; FFmpeg expressions are only a
derived compilation. Post-gate source/reference registration is an OpenCV
analysis adapter and cannot contribute pixels to project state or render inputs.

Motion Language v2 is an additive canonical effect, not a renderer command. It
uses exact rational subframe times, scalar and path curves, cubic Bezier or
Hermite authoring controls, monotone interpolation and explicit overshoot
policy, angle unwrap, normalized pivot targeting, phase/coupling annotations,
ordered affine/homography operations, or an observed source-to-output matrix
curve where decomposition is not identifiable. The effect remains inside the
approved timeline projection and Editorial Decision Graph; deterministic
sampling and any compositor work are derived render-plan concerns.

The current Motion v2 renderer lowers approved curves into typed, auditable
per-frame source-to-output matrices. A derived compositor verifies the immutable
source hash, performs premultiplied-alpha warps and optional uniform-shutter
accumulation, and emits a disposable lossless intermediate for the ordinary
FFmpeg composition pipeline. Direct measured matrices are authored in canonical
output pixels and scaled only at profile compilation. Segment renders derive
only intersecting compositor frames. Unsupported implicit retiming and masks
fail closed rather than changing the motion.

The motion-recovery laboratory is a separate generic evaluation adapter. It
renders known synthetic matrices and temporal curves, then makes the analyzer
recover them from encoded frames without reading truth sidecars. Similarity,
affine, and homography are competing explanations; structured residual motion
fails to `unknown` rather than being hidden by a confident simpler label.
Velocity, acceleration, jerk, outliers, phase boundaries, and pivot
identifiability are preserved in versioned receipts before any v2 reference
claim is allowed.

Comparison Protocol v2 is frozen against the exact Spec v2 hash and a clean
implementation revision before any candidate render. Every candidate is
independently recovered on every frame, then evaluated across technical,
geometric, temporal, perceptual, editorial, and lineage axes. Numeric pivot
error is explicitly inapplicable when pivot/translation is gauge-equivalent;
the observable matrix is scored instead. Objective success cannot synthesize a
human verdict: an attributable rubric review creates a new immutable report
revision, and its absence leaves an otherwise passing result at `warn`.

The Reference Workspace is a project-local registry over that frozen evidence,
not a new media-ingest path. It joins the selected study, spec, reference and
candidate curves, comparison diagnostics, and exact base effect by hashes.
Reference media resolves through an untracked hash-keyed local binding, and
every workbench/media read requires current revision-bound readiness plus fresh
hash verification. The browser can therefore synchronize, overlay, differ, and
plot the reference without making it a project asset or render source.

Motion correction language is a semantic application service above Motion
Language v2. It emits independent typed operations and compact diffs. Every item
is explicitly approved or rejected by a human; accepted items compile against
the exact base-effect hash and enter the ordinary `EditingService` patch/version
path. The human interpretation is separate superseding evidence. Unsupported or
unidentifiable operations, including a gauge-equivalent pivot, remain visible
but fail closed.

Portable technique v2 is a separate knowledge-to-application boundary. It
retains dense normalized center, contain-relative scale, unwrapped rotation,
fixed-pivot gauge, and editorial phase/coupling meaning, never source-pixel
matrices or reference identity. Before a Treatment exists, the application
service joins ordinary image evidence with an explicit composition claim and a
pixel-density calculation. Missing evidence, an unsupported output aspect,
insufficient resolution, or independently moving layers produce a typed
refusal and no version or render. Eligible applications compile back into the
same canonical Motion Language v2 and reversible Decision Graph as any other
edit.

## Failure posture

Long-running analysis and render operations are jobs with persisted receipts.
A stopped process can resume or safely re-run an idempotent step. Partial output
is written to a temporary location and atomically promoted only after validation.
Analyzer capabilities implement a replaceable port. A capability failure is an
`analysis.failure` record and does not erase sibling evidence. SQLite is rebuilt
from canonical JSON at startup. Missing derived media is regenerated, while a
source hash mismatch fails closed.

## Interaction surfaces

The web workbench, CLI, and allowlisted JSON agent share application services.
Natural language yields a validated `PatchPreview`, never an execution command.
Direct timeline include/exclude and pin actions also create typed patches and new
versions. Segment renders select an exact canonical frame range and write an
independent render plan, job receipt, output, and lineage file.

Style profiles are written only by a distinct explicit-confirmation action.
They are not inferred from edits, media, prompts, or accept/reject history.

The `screen.workflow` path adds a preproduction boundary without bypassing the
edit graph. A script first becomes a media-free `ScreenWorkflowPlan`. It is
capture-ready only when exact semantic beats are authored and an attributable
reviewer approves them. After recording and immutable ingest, a separate
`ScreenWorkflowEditSpec` binds every beat to one or more reviewed source ranges,
focus targets, speeds, captions, and source-audio decisions. Several ordered
segments may form a named continuity group so an ongoing agent process can ramp
through constant speed tiers without omitting frames. Each segment carries
reviewed temporal intent and optional normalized live regions; prompt entry and
natural-source holds remain at 1x, continuity groups preserve source and camera
endpoints, and speeds above 24x require an explicit perceptual-risk override.
That binding and its temporal-integrity checks become human evidence; the
resulting Treatment still follows the normal patch, version, render, QC, and
interchange path. The current implementation deliberately uses the video-safe
v1 transform compiler while Motion Language v2 remains image-only in its
baseline compositor. Region-aware temporal-loop synthesis is not executable.

A delivered narration is an immutable project asset, not a post-render side
input. `ScreenWorkflowVoiceoverTiming` first derives a review-required cue draft
from typed transcript evidence or measured pauses, then requires an attributable
complete cue review. A reviewed timing can bind the visual durations in
`ScreenWorkflowEditSpec`; if a beat has several pacing segments, their durations
must sum to its one reviewed cue. Narration becomes a dedicated normalized audio
track in the same canonical timeline and lineage as the screen recording.

Post-delivery learning has a separate authority boundary.
`ScreenWorkflowExperienceAdmission` binds one immutable external evidence
snapshot to explicit owner dispositions. The full receipt remains in ignored
product state; a tracked candidate packet may cite only the content hash of the
source-neutral `ScreenWorkflowExperiencePublicProjection`. Session-memory or
another evidence system may produce the upstream review packet, but it does not
become a runtime dependency and cannot promote its own provisional candidates.
This makes repeated project demos a cumulative evaluation lane while keeping
private session identities, paths, wording, and raw refs outside the public
seed.

## Security and privacy

The default server binds to loopback. Project paths are validated beneath the
resolved repo-local projects root. Uploaded filenames never control destination paths. Agent
operations use an allowlisted command schema. No remote API is enabled by
default, and no source is uploaded without an explicit provider configuration.
