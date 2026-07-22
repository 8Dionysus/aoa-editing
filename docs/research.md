# Technology research record

This record preserves why the baseline stack was selected. Live versions and
host capability receipts remain stronger evidence for what is installed now.

| Surface | Finding | Consequence |
| --- | --- | --- |
| FFmpeg filters | `loudnorm`, `silencedetect`, `zoompan`, `overlay`, `xfade`, and `libvmaf` are official filter surfaces. | Use FFmpeg for deterministic analysis/render/QC. |
| MLT XML | Producers, playlists, multitracks, filters, and transitions form the native serialization shape. | Generate an editable multitrack projection. |
| Kdenlive project | `.kdenlive` is MLT XML plus Kdenlive metadata, and MLT can render it. | Validate exports with the real installed `melt`. |
| OpenTimelineIO | Native OTIO is lossless for OTIO objects, while adapters have uneven effects/transition support. | Keep OTIO secondary and emit compatibility status. |
| Auto-Editor | v1 is stable but single-source/linear; v3 is nonlinear but partially stable. | Useful reference, not canonical project format. |
| PySceneDetect | Offers several scene algorithms and OTIO/FCP/EDL outputs. | Keep as a future analyzer adapter; baseline uses installed FFmpeg. |
| SAM 2 | Promptable image/video masks require a substantial PyTorch/checkpoint runtime and are optimized around CUDA. | Defer; expose segmentation port and honest deterministic proxy now. |
| Blender | Strong scripted compositor/render API but adds a large runtime and another project model. | Keep as a future render adapter when 2.5D/3D pressure justifies it. |
| OpenCV | SIFT descriptors and robust partial-affine estimation recover a source still's scale, rotation, and translation without making reference frames render assets. | Use the headless Apache-2.0 library only in the gated analysis adapter. |
| Whisper / faster-whisper | Whisper code and weights and faster-whisper are MIT; faster-whisper uses CTranslate2 and quantization. The host already owns proven OpenVINO Whisper profiles. | Use the host bridge; download no duplicate model. ASR remains correctable evidence. |
| pyannote.audio | MIT code supports local diarization, but community pretrained pipelines require model agreement/download; telemetry behavior must be reviewed or disabled for strict local privacy. | Reserve a diarization port; do not add heavy weights to the baseline. |
| Local image/video-text | Sampling frames into a local image-text embedding port is more portable than making a video foundation model mandatory. | Reserve retrieval ports; keep chronology/technical evidence as honest fallback. |
| Premiere UXP | Current UXP exposes Premiere projects, sequences, tracks, clips, effects, and export inside a proprietary desktop host with version-coupled APIs. | Future consumer adapter only; not Linux/headless baseline authority. |
| DaVinci Resolve scripting | Resolve/Fusion exposes Python/Lua scripting but requires the proprietary installed host and its project/runtime semantics. | Future optional interchange adapter, not canonical IR. |
| Remotion | Strong React/programmatic rendering, but current company licensing changes beyond individuals/teams of up to three and adds a Node/Chromium stack. | Reject as baseline renderer; reconsider only for a licensed template-render lane. |

Primary sources:

- <https://ffmpeg.org/ffmpeg-filters.html>
- <https://www.mltframework.org/docs/mltxml/>
- <https://docs.kdenlive.org/en/project_and_asset_management/file_management/project_files.html>
- <https://opentimelineio.readthedocs.io/en/latest/tutorials/adapters.html>
- <https://opentimelineio.readthedocs.io/en/v0.14/tutorials/feature-matrix.html>
- <https://auto-editor.com/docs/v3>
- <https://www.scenedetect.com/docs/latest/>
- <https://github.com/facebookresearch/sam2>
- <https://docs.blender.org/api/current/>
- <https://docs.opencv.org/5.x/d1/de0/tutorial_py_feature_homography.html>
- <https://docs.opencv.org/5.x/d9/d0c/group__calib3d.html>
- <https://github.com/openai/whisper>
- <https://github.com/SYSTRAN/faster-whisper>
- <https://github.com/pyannote/pyannote-audio>
- <https://developer.adobe.com/premiere-pro/uxp/>
- <https://documents.blackmagicdesign.com/UserManuals/Fusion8_Scripting_Guide.pdf>
- <https://www.remotion.pro/license>
