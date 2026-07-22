# Third-party licenses

AoA Editing source is Apache-2.0. Dependencies and external executables retain
their own licenses; this inventory is not a relicensing statement.

## Runtime Python packages

Exact versions are pinned in `requirements.lock`.

| License family | Packages |
| --- | --- |
| Apache-2.0 | coverage, OpenCV Python headless, OpenTimelineIO, python-multipart |
| MIT / MIT-CMU | annotated-doc, annotated-types, anyio, attrs, FastAPI, h11, idna, iniconfig, jsonschema, jsonschema-specifications, markdown-it-py, mdurl, mypy, Pillow, pip, platformdirs, pluggy, Pydantic, pytest, pytest-cov, referencing, Rich, rpds-py, Ruff, truststore |
| BSD-3-Clause | Click, httpcore2, httpx2, NumPy (plus bundled permissive notices), Starlette, Uvicorn |
| BSD-2-Clause or Apache-2.0 | packaging |
| BSD-2-Clause | Pygments |
| MPL-2.0 | pathspec |
| ISC | shellingham |
| PSF-2.0 | typing_extensions |

Use installed distribution metadata and upstream license files for complete
notices, especially NumPy's bundled components.

## External tools

| Tool | Live version | License / boundary |
| --- | --- | --- |
| FFmpeg/ffprobe | 8.1.2 Fedora build | GPL-3.0-or-later build; separate executable |
| Kdenlive Flatpak | 26.04.3 | GPL-3.0-only; editable external consumer |
| MLT | 7.40.0 | GPL/LGPL upstream components; Flatpak runtime |
| Chromium | 150.0.7871.46 | Chromium/BSD and bundled notices; test browser only |
| Git | 2.55.0 | GPL and bundled permissive components; development tool |

Capture local binary and Flatpak versions/hashes from
`manifests/environment.example.json` into an ignored runtime receipt when an
exact reproducibility audit is required.

## Models and evaluated alternatives

No model weights are distributed or downloaded by this repository. Optional
Whisper and text models stay in host-owned storage and use their model-card
licenses. OpenAI Whisper and faster-whisper code/weights are MIT, while a chosen
pyannote pipeline may require accepting a separate model agreement and may emit
opt-out telemetry. Those are future adapter decisions, not current dependencies.

Remotion is not a dependency. Its current terms are free for individuals and
organizations up to three people, while larger eligible organizations require a
company license; this made it a poor baseline owner for the renderer. Premiere
and DaVinci Resolve are proprietary host applications and remain optional future
interchange adapters, not bundled runtimes.
