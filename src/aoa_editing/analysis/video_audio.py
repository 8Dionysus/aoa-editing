"""Normalize deterministic and optional provider audio evidence onto one timeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aoa_editing import __version__
from aoa_editing.analysis.service import AnalysisService
from aoa_editing.domain.models import (
    EvidenceRecord,
    Provenance,
    TranscriptWord,
    VideoAudioEvent,
    VideoAudioTimelineEvidence,
    VideoSamplingPlan,
    VideoTranscriptSegment,
)
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.service import ProviderService


class VideoAudioAnalysisError(RuntimeError):
    """Audio decoding failed without a truthful partial representation."""


class VideoAudioTimelineService:
    def __init__(
        self,
        store: ProjectStore,
        *,
        providers: ProviderService | None = None,
    ):
        self.store = store
        self.providers = providers

    def analyze(
        self,
        plan: VideoSamplingPlan,
        *,
        transcribe: bool,
    ) -> tuple[VideoAudioTimelineEvidence, EvidenceRecord]:
        asset = self.store.load_asset(plan.project_id, plan.asset_id)
        duration = asset.metadata.duration_seconds or plan.duration_seconds
        if not asset.metadata.has_audio:
            timeline = VideoAudioTimelineEvidence(
                project_id=plan.project_id,
                asset_id=asset.id,
                source_sha256=asset.sha256,
                duration_seconds=duration,
                transcript_segments=[],
                events=[],
                speakers=[],
                unknown_intervals=[],
                evidence_refs=[],
                partial=False,
                provenance=[self._provenance(operation="verified-no-audio-stream")],
            )
            path = self.store.save_video_audio_timeline(plan.id, timeline)
            record = self._record(plan, timeline, path)
            self.store.save_evidence(record)
            return timeline, record

        AnalysisService(self.store, providers=self.providers).analyze(
            plan.project_id,
            plan.asset_id,
            transcribe=transcribe,
        )
        evidence = [
            item
            for item in self.store.list_effective_evidence(plan.project_id)
            if item.asset_id == plan.asset_id and item.source_sha256 == plan.source_sha256
        ]
        by_kind = {item.kind: item for item in evidence}
        references: list[str] = []
        provenance: list[Provenance] = []
        events: list[VideoAudioEvent] = []
        silence = by_kind.get("audio.silence")
        if silence is not None:
            references.append(silence.id)
            provenance.append(silence.provenance)
            for item in silence.payload.get("intervals", []):
                if not isinstance(item, dict):
                    continue
                events.append(
                    VideoAudioEvent(
                        kind="silence",
                        start_seconds=max(0.0, float(item.get("start_seconds", 0))),
                        end_seconds=min(duration, float(item.get("end_seconds", duration))),
                        label="deterministic silencedetect interval",
                        evidence_refs=[silence.id],
                    )
                )
        loudness = by_kind.get("audio.loudness")
        if loudness is not None:
            references.append(loudness.id)
            provenance.append(loudness.provenance)
            label = json.dumps(loudness.payload, sort_keys=True, ensure_ascii=False)[:500]
            events.append(
                VideoAudioEvent(
                    kind="loudness-change",
                    start_seconds=0,
                    end_seconds=duration,
                    label=f"global loudness measurement: {label}",
                    evidence_refs=[loudness.id],
                )
            )
        transcript_record = by_kind.get("speech.transcript") if transcribe else None
        transcript = self._transcript_segments(transcript_record, duration)
        if transcript_record is not None:
            references.append(transcript_record.id)
            provenance.append(transcript_record.provenance)
            events.extend(
                VideoAudioEvent(
                    kind="speech",
                    start_seconds=item.start_seconds,
                    end_seconds=item.end_seconds,
                    label=item.speaker,
                    evidence_refs=[transcript_record.id],
                )
                for item in transcript
            )
        source = self.store.asset_source_path(plan.project_id, plan.asset_id)
        onset_events, onset_provenance = self._energy_onsets(source, duration)
        events.extend(onset_events)
        provenance.append(onset_provenance)
        rhythm_events, rhythm_provenance = self._periodic_rhythm(onset_events, duration)
        events.extend(rhythm_events)
        provenance.append(rhythm_provenance)

        transcript_failure = next(
            (
                item
                for item in evidence
                if item.kind == "analysis.failure"
                and item.payload.get("failed_kind") == "speech.transcript"
            ),
            None,
        )
        if transcript_failure is not None and transcribe:
            references.append(transcript_failure.id)
            provenance.append(transcript_failure.provenance)
        unknown = (
            [(0.0, duration)] if not transcript else self._transcript_gaps(transcript, duration)
        )
        partial = bool(unknown) and transcribe
        if not transcribe:
            partial = False
        speakers = sorted({item.speaker for item in transcript if item.speaker is not None})
        timeline = VideoAudioTimelineEvidence(
            project_id=plan.project_id,
            asset_id=asset.id,
            source_sha256=asset.sha256,
            duration_seconds=duration,
            transcript_segments=transcript,
            events=sorted(
                events, key=lambda item: (item.start_seconds, item.end_seconds, item.kind)
            ),
            speakers=speakers,
            unknown_intervals=unknown if transcribe else [],
            evidence_refs=sorted(set(references)),
            partial=partial,
            provenance=provenance or [self._provenance(operation="audio-stream-probe-only")],
        )
        path = self.store.save_video_audio_timeline(plan.id, timeline)
        record = self._record(plan, timeline, path)
        self.store.save_evidence(record)
        return timeline, record

    def _record(
        self,
        plan: VideoSamplingPlan,
        timeline: VideoAudioTimelineEvidence,
        path: Path,
    ) -> EvidenceRecord:
        relative = str(path.relative_to(self.store.project_path(plan.project_id)))
        return EvidenceRecord(
            project_id=plan.project_id,
            asset_id=plan.asset_id,
            source_sha256=plan.source_sha256,
            kind="video.audio-timeline",
            confidence=0.7 if timeline.partial else 1.0,
            payload={
                "plan_id": plan.id,
                "event_count": len(timeline.events),
                "transcript_segment_count": len(timeline.transcript_segments),
                "unknown_intervals": timeline.unknown_intervals,
                "partial": timeline.partial,
                "evidence_refs": timeline.evidence_refs,
            },
            artifacts=[relative],
            time_base=plan.time_base,
            ranges=[plan.analysis_range],
            provenance=self._provenance(operation="audio-timeline-normalization"),
        )

    @staticmethod
    def _transcript_segments(
        record: EvidenceRecord | None,
        duration: float,
    ) -> list[VideoTranscriptSegment]:
        if record is None:
            return []
        segments: list[VideoTranscriptSegment] = []
        for raw in record.payload.get("segments", []):
            if not isinstance(raw, dict):
                continue
            raw_start = raw.get("start_seconds", raw.get("start", 0))
            start = float(raw_start if raw_start is not None else 0)
            raw_end = raw.get("end_seconds", raw.get("end", start))
            end = float(raw_end if raw_end is not None else start)
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            words: list[TranscriptWord] = []
            for word in raw.get("words", []):
                if not isinstance(word, dict) or not str(word.get("text", "")).strip():
                    continue
                raw_word_start = word.get("start_seconds", word.get("start", start))
                raw_word_end = word.get("end_seconds", word.get("end", end))
                words.append(
                    TranscriptWord(
                        text=str(word["text"]).strip(),
                        start_seconds=float(
                            raw_word_start if raw_word_start is not None else start
                        ),
                        end_seconds=float(raw_word_end if raw_word_end is not None else end),
                        confidence=(
                            float(word["confidence"])
                            if word.get("confidence") is not None
                            else None
                        ),
                    )
                )
            segments.append(
                VideoTranscriptSegment(
                    start_seconds=max(0.0, start),
                    end_seconds=min(duration, max(start, end)),
                    text=text,
                    speaker=(str(raw["speaker"]) if raw.get("speaker") is not None else None),
                    words=words,
                    provider_receipt_id=str(
                        record.provenance.parameters.get("provider_receipt_id", "")
                    )
                    or None,
                    partial=bool(record.payload.get("partial", False)),
                )
            )
        return sorted(segments, key=lambda item: (item.start_seconds, item.end_seconds))

    @staticmethod
    def _transcript_gaps(
        segments: list[VideoTranscriptSegment], duration: float
    ) -> list[tuple[float, float]]:
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for segment in segments:
            if segment.start_seconds > cursor + 0.05:
                gaps.append((cursor, segment.start_seconds))
            cursor = max(cursor, segment.end_seconds)
        if cursor < duration - 0.05:
            gaps.append((cursor, duration))
        return gaps

    def _energy_onsets(
        self, source: Path, duration: float
    ) -> tuple[list[VideoAudioEvent], Provenance]:
        sample_rate = 16_000
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ]
        result = subprocess.run(command, check=False, capture_output=True, timeout=600)
        if result.returncode != 0:
            raise VideoAudioAnalysisError(result.stderr.decode("utf-8", errors="replace")[-2000:])
        samples = np.frombuffer(result.stdout, dtype="<f4")
        window = 320
        usable = len(samples) // window * window
        if usable == 0:
            return [], self._provenance(operation="energy-onset-scan", empty=True)
        energy = np.sqrt(np.mean(samples[:usable].reshape(-1, window) ** 2, axis=1))
        positive = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        threshold = median + max(0.005, 6 * mad)
        candidates = np.flatnonzero(positive >= threshold)
        events: list[VideoAudioEvent] = []
        last_time = -1.0
        maximum = float(np.max(positive)) or 1.0
        for index in candidates:
            time_seconds = float(index * window / sample_rate)
            if time_seconds - last_time < 0.15 or time_seconds > duration:
                continue
            strength = min(1.0, float(positive[index] / maximum))
            events.append(
                VideoAudioEvent(
                    kind="accent" if strength >= 0.75 else "onset",
                    start_seconds=time_seconds,
                    end_seconds=min(duration, time_seconds + window / sample_rate),
                    strength=strength,
                    label="deterministic short-time-energy onset",
                )
            )
            last_time = time_seconds
        return events, self._provenance(
            operation="energy-onset-scan",
            command=[item if item != str(source) else "<project-media>" for item in command],
            sample_rate=sample_rate,
            window_samples=window,
            threshold=threshold,
        )

    def _periodic_rhythm(
        self,
        onset_events: list[VideoAudioEvent],
        duration: float,
    ) -> tuple[list[VideoAudioEvent], Provenance]:
        strong = [
            item for item in onset_events if item.strength is not None and item.strength >= 0.65
        ]
        if len(strong) < 3:
            return [], self._provenance(
                operation="periodic-onset-classification",
                strong_onset_count=len(strong),
                periodic=False,
            )
        times = np.asarray([item.start_seconds for item in strong], dtype=np.float64)
        intervals = np.diff(times)
        median = float(np.median(intervals))
        deviation = float(np.median(np.abs(intervals - median)))
        relative_deviation = deviation / max(median, 1e-9)
        periodic = 0.2 <= median <= 2.0 and relative_deviation <= 0.15
        if not periodic:
            return [], self._provenance(
                operation="periodic-onset-classification",
                strong_onset_count=len(strong),
                median_interval_seconds=median,
                relative_median_deviation=relative_deviation,
                periodic=False,
            )
        beats = [
            VideoAudioEvent(
                kind="beat",
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                strength=item.strength,
                label="periodic deterministic energy onset",
            )
            for item in strong
        ]
        music_end = min(duration, strong[-1].start_seconds + median)
        music = VideoAudioEvent(
            kind="music",
            start_seconds=strong[0].start_seconds,
            end_seconds=music_end,
            strength=float(np.mean([item.strength or 0.0 for item in strong])),
            label="periodic onset train; music is a conservative structural hypothesis",
        )
        return [music, *beats], self._provenance(
            operation="periodic-onset-classification",
            strong_onset_count=len(strong),
            median_interval_seconds=median,
            relative_median_deviation=relative_deviation,
            periodic=True,
            limitation="periodic energy supports music/beat structure but not genre or source",
        )

    @staticmethod
    def _provenance(**parameters: Any) -> Provenance:
        return Provenance(
            tool="aoa-editing-video-audio",
            tool_version=__version__,
            parameters=parameters,
            deterministic=True,
        )
