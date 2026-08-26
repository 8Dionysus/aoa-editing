"""Compile the canonical timeline into a deterministic FFmpeg plan."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np

from aoa_editing import __version__
from aoa_editing.domain.models import (
    Asset,
    AudioEffect,
    BlendEffect,
    BlurEffect,
    Clip,
    ColorEffect,
    CompositorFrameV2,
    CompositorJobV2,
    DissolveEffect,
    FrameRange,
    GlowEffect,
    MaskEffect,
    Matrix3x3V2,
    MatrixTransformMotionV2,
    MediaKind,
    ProjectVersion,
    Provenance,
    RenderPlan,
    RenderProfile,
    ScalarKeyframe,
    SpeedEffect,
    TextEffect,
    TrackKind,
    TransformEffect,
    TransformEffectV2,
    Vec2Keyframe,
)
from aoa_editing.domain.motion_v2 import evaluate_transform_matrix
from aoa_editing.infrastructure.store import ProjectStore


def render_profile(name: str, version: ProjectVersion) -> RenderProfile:
    if name == "preview":
        width = min(960, version.timeline.width)
        height = round(width * version.timeline.height / version.timeline.width)
        if height % 2:
            height += 1
        return RenderProfile(
            name="preview",
            width=width,
            height=height,
            crf=28,
            preset="veryfast",
            audio_bitrate="128k",
        )
    if name == "final":
        return RenderProfile(
            name="final",
            width=version.timeline.width,
            height=version.timeline.height,
            crf=18,
            preset="medium",
        )
    raise ValueError(f"unknown render profile: {name}")


class FFmpegCompiler:
    def __init__(self, store: ProjectStore):
        self.store = store

    def compile(
        self,
        project_id: str,
        version: ProjectVersion,
        profile_name: str,
        output_path: Path,
        frame_range: FrameRange | None = None,
    ) -> RenderPlan:
        profile = render_profile(profile_name, version)
        fps = version.timeline.frame_rate.fps
        total_seconds = version.timeline.duration_frames / fps
        if frame_range is not None and frame_range.end > version.timeline.duration_frames:
            raise ValueError("render range ends after the canonical timeline")
        background = version.timeline.background.replace("#", "0x")
        inputs: list[str] = [
            "-f",
            "lavfi",
            "-i",
            f"color=c={background}:s={profile.width}x{profile.height}:r={fps}:d={total_seconds}",
        ]
        filters: list[str] = ["[0:v]setpts=PTS-STARTPTS,format=rgba[base]"]
        current = "base"
        input_index = 1
        input_hashes: list[str] = []
        audio_labels: list[str] = []
        compositor_jobs: list[CompositorJobV2] = []
        normalize_audio = False
        visual_clips = [
            clip
            for track in version.timeline.tracks
            if track.kind is TrackKind.VIDEO and not track.muted
            for clip in track.clips
            if clip.enabled
            and (frame_range is None or _frame_range_intersection(clip.timeline_range, frame_range))
        ]
        audio_clips = [
            clip
            for track in version.timeline.tracks
            if track.kind is TrackKind.AUDIO and not track.muted
            for clip in track.clips
            if clip.enabled
        ]
        caption_clips = [
            clip
            for track in version.timeline.tracks
            if track.kind is TrackKind.CAPTION and not track.muted
            for clip in track.clips
            if clip.enabled
        ]
        for clip_number, clip in enumerate(visual_clips):
            asset = self.store.load_asset(project_id, clip.asset_id)
            source = self.store.asset_source_path(project_id, clip.asset_id)
            input_hashes.append(asset.sha256)
            label = f"clip{clip_number}"
            duration = clip.timeline_range.duration / fps
            speed = next(
                (effect.rate for effect in clip.effects if isinstance(effect, SpeedEffect)),
                1.0,
            )
            source_start = (clip.source_range.start / fps) if clip.source_range else 0.0
            input_duration = duration * speed
            transform_v2 = next(
                (effect for effect in clip.effects if isinstance(effect, TransformEffectV2)),
                None,
            )
            rendered_range = (
                _frame_range_intersection(clip.timeline_range, frame_range)
                if frame_range is not None
                else clip.timeline_range
            )
            if rendered_range is None:  # pragma: no cover - visual_clips is prefiltered
                raise ValueError("visual clip does not intersect the requested render range")
            if transform_v2 is not None:
                if speed != 1.0:
                    raise ValueError(
                        "Motion Language v2 must be retimed canonically before rendering"
                    )
                job = self._compositor_job(
                    clip,
                    asset,
                    source,
                    transform_v2,
                    label=label,
                    profile=profile,
                    version=version,
                    output_path=output_path,
                    rendered_range=rendered_range,
                )
                compositor_jobs.append(job)
                inputs.extend(["-i", str(job.output_path)])
            else:
                inputs.extend(_media_input(asset, source, input_duration, source_start, fps))
            clip_input = input_index
            input_index += 1
            mask = next((effect for effect in clip.effects if isinstance(effect, MaskEffect)), None)
            mask_input: int | None = None
            if mask is not None:
                if transform_v2 is not None:
                    raise ValueError(
                        "Motion Language v2 masks require a typed compositor mask stage"
                    )
                mask_path = (self.store.project_path(project_id) / mask.mask_asset_path).resolve()
                if not mask_path.is_relative_to(self.store.project_path(project_id).resolve()):
                    raise ValueError("mask path escapes project root")
                inputs.extend(
                    [
                        "-loop",
                        "1",
                        "-framerate",
                        str(fps),
                        "-t",
                        str(duration),
                        "-i",
                        str(mask_path),
                    ]
                )
                mask_input = input_index
                input_index += 1
            filters.extend(
                self._video_filters(
                    clip,
                    asset,
                    clip_input,
                    mask_input,
                    label,
                    profile,
                    fps,
                    rendered_range=rendered_range if transform_v2 is not None else None,
                )
            )
            composed = f"comp{clip_number}"
            composited_range = rendered_range if transform_v2 is not None else clip.timeline_range
            start_seconds = composited_range.start / fps
            end_seconds = composited_range.end / fps
            blend = next(
                (effect for effect in clip.effects if isinstance(effect, BlendEffect)), None
            )
            if blend is not None and blend.mode != "normal":
                filters.append(
                    f"[{current}][{label}]blend=all_mode={blend.mode}:"
                    f"all_opacity={blend.opacity}:eof_action=pass:shortest=0:"
                    f"enable='between(t,{start_seconds},{end_seconds})'[{composed}]"
                )
            else:
                filters.append(
                    f"[{current}][{label}]overlay=eof_action=pass:shortest=0:format=auto:"
                    f"enable='between(t,{start_seconds},{end_seconds})'[{composed}]"
                )
            current = composed
        for clip_number, clip in enumerate(audio_clips):
            asset = self.store.load_asset(project_id, clip.asset_id)
            source = self.store.asset_source_path(project_id, clip.asset_id)
            input_hashes.append(asset.sha256)
            duration = clip.timeline_range.duration / fps
            speed = next(
                (effect.rate for effect in clip.effects if isinstance(effect, SpeedEffect)),
                1.0,
            )
            source_start = (clip.source_range.start / fps) if clip.source_range else 0.0
            inputs.extend(_media_input(asset, source, duration * speed, source_start, fps))
            label = f"audio{clip_number}"
            chain = [
                f"[{input_index}:a]atrim=duration={duration * speed}",
                "asetpts=PTS-STARTPTS",
            ]
            if speed != 1.0:
                chain.extend([f"atempo={speed}", f"atrim=duration={duration}"])
            audio_effect = next(
                (effect for effect in clip.effects if isinstance(effect, AudioEffect)), None
            )
            if audio_effect is not None:
                if audio_effect.gain_keyframes:
                    gain = _scalar_expression(
                        audio_effect.gain_keyframes,
                        frame_count=max(1, clip.timeline_range.duration),
                        default=0.0,
                        variable="n",
                    )
                    chain.append(
                        f"volume='pow(10,(({audio_effect.gain_db})+({gain}))/20)':eval=frame"
                    )
                else:
                    chain.append(f"volume={audio_effect.gain_db}dB")
                fade_in = audio_effect.fade_in_frames / fps
                fade_out = audio_effect.fade_out_frames / fps
                if fade_in > 0:
                    chain.append(f"afade=t=in:st=0:d={fade_in}")
                if fade_out > 0 and duration > fade_out:
                    chain.append(f"afade=t=out:st={duration - fade_out}:d={fade_out}")
                normalize_audio = normalize_audio or audio_effect.normalize_lufs is not None
            delay_ms = round(clip.timeline_range.start / fps * 1000)
            chain.append(f"adelay={delay_ms}:all=1[{label}]")
            filters.append(",".join(chain))
            audio_labels.append(label)
            input_index += 1
        for caption_number, clip in enumerate(caption_clips):
            for text_number, text_effect in enumerate(
                effect for effect in clip.effects if isinstance(effect, TextEffect)
            ):
                titled = f"caption{caption_number}_{text_number}"
                start = clip.timeline_range.start / fps
                end = clip.timeline_range.end / fps
                filters.append(
                    f"[{current}]"
                    + _drawtext(text_effect, enable=f"between(t,{start},{end})")
                    + f"[{titled}]"
                )
                current = titled
        if frame_range is not None:
            start_seconds = frame_range.start / fps
            range_seconds = frame_range.duration / fps
            filters.append(
                f"[{current}]select='between(n,{frame_range.start},"
                f"{frame_range.end - 1})',setpts=N/({fps}*TB),"
                f"format={profile.pixel_format}[vout]"
            )
            audio_output = "afull"
        else:
            filters.append(f"[{current}]format={profile.pixel_format}[vout]")
            audio_output = "aout"
        if audio_labels:
            joined = "".join(f"[{label}]" for label in audio_labels)
            audio_chain = (
                f"{joined}amix=inputs={len(audio_labels)}:normalize=0:dropout_transition=0,"
                f"atrim=duration={total_seconds}"
            )
            if normalize_audio:
                audio_chain += ",loudnorm=I=-16:LRA=11:TP=-1.5"
            filters.append(audio_chain + f"[{audio_output}]")
        else:
            inputs.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    str(total_seconds),
                    "-i",
                    f"anullsrc=r={version.timeline.audio_sample_rate}:cl=stereo",
                ]
            )
            filters.append(f"[{input_index}:a]atrim=duration={total_seconds}[{audio_output}]")
        if frame_range is not None:
            filters.append(
                f"[afull]atrim=start={start_seconds}:duration={range_seconds},"
                "asetpts=PTS-STARTPTS[aout]"
            )
        filter_graph = ";".join(filters)
        range_limit = (
            ["-frames:v", str(frame_range.duration), "-t", str(range_seconds)]
            if frame_range is not None
            else []
        )
        shortest = [] if frame_range is not None else ["-shortest"]
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            filter_graph,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            profile.video_codec,
            "-preset",
            profile.preset,
            "-crf",
            str(profile.crf),
            "-pix_fmt",
            profile.pixel_format,
            "-r",
            str(fps),
            "-c:a",
            profile.audio_codec,
            "-b:a",
            profile.audio_bitrate,
            "-ar",
            str(version.timeline.audio_sample_rate),
            "-movflags",
            "+faststart",
            *range_limit,
            *shortest,
            str(output_path),
        ]
        return RenderPlan(
            project_id=project_id,
            version_id=version.id,
            command=command,
            filter_graph=filter_graph,
            output_path=output_path,
            input_hashes=sorted(set(input_hashes)),
            profile=profile,
            frame_range=frame_range,
            compositor_jobs=compositor_jobs,
            compiler=Provenance(
                tool="aoa-editing-ffmpeg-compiler",
                tool_version=__version__,
                parameters={
                    "profile": profile.name,
                    "canonical_input": "timeline-projection",
                    "frame_range": (frame_range.model_dump(mode="json") if frame_range else None),
                },
                deterministic=True,
            ),
        )

    def _compositor_job(
        self,
        clip: Clip,
        asset: Asset,
        source: Path,
        effect: TransformEffectV2,
        *,
        label: str,
        profile: RenderProfile,
        version: ProjectVersion,
        output_path: Path,
        rendered_range: FrameRange,
    ) -> CompositorJobV2:
        if asset.media_kind is not MediaKind.IMAGE:
            raise ValueError("the baseline Motion Language v2 compositor requires an image")
        source_width = asset.metadata.width
        source_height = asset.metadata.height
        if source_width is None or source_height is None:
            raise ValueError("Motion Language v2 source dimensions are unavailable")
        first, last = _motion_bounds(effect)
        expected_last = Fraction(clip.timeline_range.duration - 1)
        if first != 0 or last != expected_last:
            raise ValueError(
                "Motion Language v2 curves must cover the complete clip from frame zero "
                "through duration minus one"
            )
        local_start = rendered_range.start - clip.timeline_range.start
        local_end = rendered_range.end - clip.timeline_range.start
        if local_start < 0 or local_end > clip.timeline_range.duration:
            raise ValueError("compositor render range escapes the owning clip")
        frames: list[CompositorFrameV2] = []
        for output_frame, local_frame in enumerate(range(local_start, local_end)):
            matrices = []
            for sample_time in _motion_sample_times(effect, local_frame, expected_last):
                matrix = np.asarray(
                    evaluate_transform_matrix(
                        effect,
                        sample_time,
                        source_width=source_width,
                        source_height=source_height,
                        output_width=profile.width,
                        output_height=profile.height,
                    ),
                    dtype=np.float64,
                )
                if isinstance(effect.motion, MatrixTransformMotionV2):
                    matrix = (
                        np.asarray(
                            (
                                (
                                    profile.width / version.timeline.width,
                                    0.0,
                                    0.0,
                                ),
                                (
                                    0.0,
                                    profile.height / version.timeline.height,
                                    0.0,
                                ),
                                (0.0, 0.0, 1.0),
                            ),
                            dtype=np.float64,
                        )
                        @ matrix
                    )
                matrices.append(_matrix_tuple(matrix))
            frames.append(
                CompositorFrameV2(
                    output_frame=output_frame,
                    timeline_frame=clip.timeline_range.start + local_frame,
                    matrices_3x3=matrices,
                )
            )
        intermediate = output_path.parent / ".compositor" / f"{label}.mkv"
        fps = version.timeline.frame_rate
        encoder_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgra",
            "-video_size",
            f"{profile.width}x{profile.height}",
            "-framerate",
            f"{fps.numerator}/{fps.denominator}",
            "-i",
            "pipe:0",
            "-frames:v",
            str(len(frames)),
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgra",
            str(intermediate),
        ]
        return CompositorJobV2(
            clip_id=clip.id,
            source_path=source,
            source_sha256=asset.sha256,
            output_path=intermediate,
            width=profile.width,
            height=profile.height,
            frame_rate=fps,
            frames=frames,
            interpolation="lanczos4" if profile.name == "final" else "cubic",
            encoder_command=encoder_command,
        )

    def _video_filters(
        self,
        clip: Clip,
        asset: Asset,
        input_index: int,
        mask_input: int | None,
        label: str,
        profile: RenderProfile,
        fps: float,
        rendered_range: FrameRange | None = None,
    ) -> list[str]:
        effective_range = rendered_range or clip.timeline_range
        duration = effective_range.duration / fps
        speed = next(
            (effect.rate for effect in clip.effects if isinstance(effect, SpeedEffect)), 1.0
        )
        transform = next(
            (effect for effect in clip.effects if isinstance(effect, TransformEffect)), None
        )
        transform_v2 = next(
            (effect for effect in clip.effects if isinstance(effect, TransformEffectV2)),
            None,
        )
        if transform is not None and transform_v2 is not None:
            raise ValueError("a clip cannot contain both transform generations")
        contain_canvas = _contain_canvas(asset, transform, profile)
        base_label = f"{label}base"
        source_duration = duration * speed
        setpts = "PTS-STARTPTS" if speed == 1.0 else f"(PTS-STARTPTS)/{speed}"
        chain = (
            f"[{input_index}:v]trim=duration={source_duration},setpts={setpts},"
            f"trim=duration={duration}"
        )
        if transform_v2 is None:
            chain += "," + _fit_filters(transform, profile, contain_canvas)
        color = next((effect for effect in clip.effects if isinstance(effect, ColorEffect)), None)
        if color is not None:
            chain += (
                f",eq=brightness={color.brightness}:contrast={color.contrast}:"
                f"saturation={color.saturation}:gamma={color.gamma}"
            )
            if color.vignette > 0:
                chain += f",vignette=angle={0.15 + color.vignette * 0.45}"
        blur = next((effect for effect in clip.effects if isinstance(effect, BlurEffect)), None)
        if blur is not None and blur.sigma > 0:
            chain += f",gblur=sigma={blur.sigma}"
        chain += f",format=rgba[{base_label}]"
        filters = [chain]
        working = base_label
        if mask_input is not None:
            mask_label = f"{label}mask"
            masked_label = f"{label}masked"
            filters.append(
                f"[{mask_input}:v]trim=duration={duration},setpts=PTS-STARTPTS,"
                f"{_fit_filters(transform, profile, contain_canvas)},"
                f"format=gray[{mask_label}]"
            )
            filters.append(f"[{working}][{mask_label}]alphamerge[{masked_label}]")
            working = masked_label
        glow = next((effect for effect in clip.effects if isinstance(effect, GlowEffect)), None)
        if glow is not None and glow.intensity > 0:
            glow_base = f"{label}glowbase"
            glow_blur = f"{label}glowblur"
            glowed = f"{label}glowed"
            filters.append(f"[{working}]split=2[{glow_base}][{glow_blur}src]")
            filters.append(f"[{glow_blur}src]gblur=sigma={glow.radius}[{glow_blur}]")
            filters.append(
                f"[{glow_base}][{glow_blur}]blend=all_mode=screen:"
                f"all_opacity={glow.intensity}[{glowed}]"
            )
            working = glowed
        if transform is not None:
            if transform.rotation:
                rotated = f"{label}rotated"
                rotation = _scalar_expression(
                    transform.rotation,
                    frame_count=max(1, clip.timeline_range.duration),
                    default=0.0,
                    variable="n",
                )
                filters.append(
                    f"[{working}]rotate=a='({rotation})*PI/180':ow=iw:oh=ih:"
                    f"c=none:bilinear=1,format=rgba[{rotated}]"
                )
                working = rotated
            if contain_canvas is not None:
                padded = f"{label}padded"
                canvas_width, canvas_height = contain_canvas
                filters.append(
                    f"[{working}]pad={canvas_width}:{canvas_height}:"
                    f"(ow-iw)/2:(oh-ih)/2:color=0x00000000,"
                    f"format=rgba[{padded}]"
                )
                working = padded
            transformed = f"{label}motion"
            frame_count = max(1, clip.timeline_range.duration)
            zoom = _scalar_expression(transform.scale, frame_count, 1.0, variable="on")
            if transform.position_mode == "canvas_center":
                x_pos = _vector_expression(
                    transform.position, frame_count, axis="x", default=0.5, variable="on"
                )
                y_pos = _vector_expression(
                    transform.position, frame_count, axis="y", default=0.5, variable="on"
                )
                x_expression = f"iw/2-(({x_pos})*iw)/zoom"
                y_expression = f"ih/2-(({y_pos})*ih)/zoom"
            else:
                x_pos = _vector_expression(
                    transform.position, frame_count, axis="x", default=0.0, variable="on"
                )
                y_pos = _vector_expression(
                    transform.position, frame_count, axis="y", default=0.0, variable="on"
                )
                x_expression = f"iw/2-(iw/zoom/2)+({x_pos})*iw"
                y_expression = f"ih/2-(ih/zoom/2)+({y_pos})*ih"
            filters.append(
                f"[{working}]zoompan=z='{zoom}':"
                f"x='{x_expression}':"
                f"y='{y_expression}':"
                f"d=1:s={profile.width}x{profile.height}:fps={fps},"
                f"format=rgba[{transformed}]"
            )
            working = transformed
        blend = next((effect for effect in clip.effects if isinstance(effect, BlendEffect)), None)
        if blend is not None and blend.mode == "normal" and blend.opacity < 1:
            blended = f"{label}opacity"
            filters.append(f"[{working}]colorchannelmixer=aa={blend.opacity}[{blended}]")
            working = blended
        dissolve = next(
            (effect for effect in clip.effects if isinstance(effect, DissolveEffect)), None
        )
        if dissolve is not None:
            if dissolve.fade_in_frames:
                faded = f"{label}fadein"
                filters.append(
                    f"[{working}]fade=t=in:st=0:d={dissolve.fade_in_frames / fps}:alpha=1[{faded}]"
                )
                working = faded
            if dissolve.fade_out_frames:
                faded = f"{label}fadeout"
                fade_duration = dissolve.fade_out_frames / fps
                filters.append(
                    f"[{working}]fade=t=out:st={max(0.0, duration - fade_duration)}:"
                    f"d={fade_duration}:alpha=1[{faded}]"
                )
                working = faded
        text_effects = [effect for effect in clip.effects if isinstance(effect, TextEffect)]
        for index, text_effect in enumerate(text_effects):
            titled = f"{label}text{index}"
            filters.append(f"[{working}]" + _drawtext(text_effect) + f"[{titled}]")
            working = titled
        start = effective_range.start / fps
        filters.append(f"[{working}]setpts=PTS-STARTPTS+{start}/TB[{label}]")
        return filters


def _media_input(
    asset: Asset, source: Path, duration: float, source_start: float, fps: float
) -> list[str]:
    if asset.media_kind is MediaKind.IMAGE:
        return [
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            str(duration),
            "-i",
            str(source),
        ]
    return ["-ss", str(source_start), "-t", str(duration), "-i", str(source)]


def _motion_bounds(effect: TransformEffectV2) -> tuple[Fraction, Fraction]:
    motion = effect.motion
    keyframes = (
        motion.matrix.keyframes
        if isinstance(motion, MatrixTransformMotionV2)
        else motion.position.keyframes
    )
    return keyframes[0].time.as_fraction, keyframes[-1].time.as_fraction


def _frame_range_intersection(
    left: FrameRange,
    right: FrameRange,
) -> FrameRange | None:
    start = max(left.start, right.start)
    end = min(left.end, right.end)
    if start >= end:
        return None
    return FrameRange(start=start, duration=end - start)


def _motion_sample_times(
    effect: TransformEffectV2,
    frame: int,
    last: Fraction,
) -> list[Fraction]:
    sampling = effect.sampling
    center = Fraction(frame)
    if sampling.samples_per_frame == 1:
        return [center]
    shutter = Fraction(str(sampling.shutter_fraction))
    samples: list[Fraction] = []
    for index in range(sampling.samples_per_frame):
        position = Fraction(2 * index + 1, 2 * sampling.samples_per_frame) - Fraction(
            1,
            2,
        )
        selected = center + shutter * position
        samples.append(min(last, max(Fraction(0), selected)))
    return samples


def _matrix_tuple(matrix: np.ndarray) -> Matrix3x3V2:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


def _escape_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")


def _drawtext(effect: TextEffect, *, enable: str | None = None) -> str:
    text = _escape_drawtext(effect.text)
    options = [
        f"text='{text}'",
        f"x=(w-text_w)*{effect.x}",
        f"y=(h-text_h)*{effect.y}",
        f"fontsize={effect.font_size}",
        f"fontcolor={effect.color}",
    ]
    if effect.background:
        options.extend(("box=1", f"boxcolor={effect.background}", "boxborderw=12"))
    if enable:
        options.append(f"enable='{enable}'")
    return "drawtext=" + ":".join(options)


def _fit_filters(
    transform: TransformEffect | None,
    profile: RenderProfile,
    contain_canvas: tuple[int, int] | None,
) -> str:
    if transform is not None and contain_canvas is not None:
        canvas_width, canvas_height = contain_canvas
        return (
            f"scale={canvas_width}:{canvas_height}:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2,setsar=1"
        )
    return (
        f"scale={profile.width}:{profile.height}:force_original_aspect_ratio=increase,"
        f"crop={profile.width}:{profile.height},setsar=1"
    )


def _contain_canvas(
    asset: Asset,
    transform: TransformEffect | None,
    profile: RenderProfile,
) -> tuple[int, int] | None:
    if transform is None or transform.fit_mode != "contain":
        return None
    source_width = asset.metadata.width
    source_height = asset.metadata.height
    if source_width is None or source_height is None:
        return profile.width, profile.height
    output_fit_scale = min(profile.width / source_width, profile.height / source_height)
    preservation_factor = min(2.0, max(1.0, 1.0 / output_fit_scale))
    width = max(profile.width, round(profile.width * preservation_factor))
    height = max(profile.height, round(profile.height * preservation_factor))
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    return width, height


def _scalar_expression(
    keyframes: Sequence[ScalarKeyframe], frame_count: int, default: float, *, variable: str
) -> str:
    if not keyframes:
        return str(default)
    _ = frame_count
    return _piecewise_expression(
        keyframes,
        frame_count,
        variable=variable,
        value=lambda keyframe: keyframe.value,
    )


def _vector_expression(
    keyframes: Sequence[Vec2Keyframe],
    frame_count: int,
    *,
    axis: str,
    default: float,
    variable: str,
) -> str:
    if not keyframes:
        return str(default)
    _ = frame_count
    return _piecewise_expression(
        keyframes,
        frame_count,
        variable=variable,
        value=lambda keyframe: getattr(keyframe, axis),
    )


def _piecewise_expression[KeyframeT: (ScalarKeyframe, Vec2Keyframe)](
    keyframes: Sequence[KeyframeT],
    frame_count: int,
    *,
    variable: str,
    value: Callable[[KeyframeT], float],
) -> str:
    del frame_count  # The curve is explicitly bounded by canonical keyframe frames.
    getter = value
    if len(keyframes) == 1:
        return str(getter(keyframes[0]))
    segments: list[str] = []
    for left, right in pairwise(keyframes):
        start = float(getter(left))
        end = float(getter(right))
        progress = _easing_expression(variable, left.frame, right.frame, right.easing)
        interpolation = f"({start})+(({end})-({start}))*({progress})"
        segments.append(
            f"(gte({variable},{left.frame})*lt({variable},{right.frame})*({interpolation}))"
        )
    first = float(getter(keyframes[0]))
    last = float(getter(keyframes[-1]))
    interior = "+".join(segments)
    return (
        f"if(lte({variable},{keyframes[0].frame}),{first},"
        f"if(gte({variable},{keyframes[-1].frame}),{last},{interior}))"
    )


def _easing_expression(variable: str, start: int, end: int, easing: str) -> str:
    duration = max(1, end - start)
    progress = f"(({variable}-{start})/{duration})"
    if easing == "hold":
        return "0"
    if easing == "ease_in":
        return f"pow({progress},2)"
    if easing == "ease_out":
        return f"1-pow(1-{progress},2)"
    if easing == "ease_in_out":
        return f"({progress})*({progress})*(3-2*({progress}))"
    return progress
