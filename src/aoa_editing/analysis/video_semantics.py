"""Provider-neutral semantic observations over retained Video Anatomy samples."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from aoa_editing import __version__
from aoa_editing.domain.models import (
    EvidenceRecord,
    Provenance,
    VideoFrameSampleManifest,
    VideoSamplingPlan,
    VideoStructureEvidence,
    VideoVisualObservation,
)
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.providers.service import ProviderService

VISION_ALIAS = "local-ai://vision/describe/default"
NORMALIZATION_VERSION = "video-anatomy-vision-v1"


class VideoSemanticService:
    def __init__(
        self,
        store: ProjectStore,
        *,
        providers: ProviderService | None = None,
    ):
        self.store = store
        self.providers = providers or ProviderService.for_settings(store.settings)

    def analyze(
        self,
        plan: VideoSamplingPlan,
        structure: VideoStructureEvidence,
        manifest: VideoFrameSampleManifest,
        *,
        prompt: str = (
            "Describe visible subjects, composition, focus, background, text, overlays, "
            "and meaningful change at each supplied timestamp. Report only visible evidence."
        ),
        explicit_opt_in: bool = False,
    ) -> tuple[list[VideoVisualObservation], EvidenceRecord]:
        samples = [item for item in manifest.samples if item.kept]
        times = sorted({round(item.time_seconds, 6) for item in samples})
        source = self.store.asset_source_path(plan.project_id, plan.asset_id)
        binding_revision = self._binding_revision()
        cache_key = self._cache_key(plan, times, prompt, binding_revision)
        cache_path = (
            self.store.settings.cache_root
            / "video-anatomy"
            / "vision-describe"
            / f"{cache_key}.json"
        )
        cached = self._read_cache(cache_path)
        cache_hit = cached is not None
        receipt_id: str | None
        receipt_path: str | None
        model_id: str | None
        model_revision: str | None
        outcome: str
        failure_code: str | None
        output: dict[str, Any] | None
        if cached is None:
            result = self.providers.invoke(
                VISION_ALIAS,
                {
                    "media_path": str(source),
                    "sample_times_seconds": times,
                    "prompt": prompt,
                },
                privacy_mode=self.store.load_project(plan.project_id).intent.privacy_mode,
                explicit_opt_in=explicit_opt_in,
                allow_fallback=False,
                project_id=plan.project_id,
                asset_id=plan.asset_id,
            )
            receipt = result.receipt
            receipt_id = receipt.id
            receipt_path = result.receipt_path
            model_id = receipt.model_id
            model_revision = receipt.model_revision
            outcome = receipt.outcome
            failure_code = receipt.failure_code
            output = result.output
            if output is not None:
                self._atomic_json(
                    cache_path,
                    {
                        "schema_version": "1.0.0",
                        "cache_key": cache_key,
                        "source_sha256": plan.source_sha256,
                        "plan_sha256": plan.plan_sha256,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "provider_receipt_id": receipt_id,
                        "provider_receipt_path": receipt_path,
                        "outcome": outcome,
                        "output": output,
                    },
                )
        else:
            output = cached.get("output")
            if not isinstance(output, dict):
                output = None
            receipt_id = str(cached.get("provider_receipt_id", "")) or None
            receipt_path = str(cached.get("provider_receipt_path", "")) or None
            model_id = str(cached.get("model_id", "")) or None
            model_revision = str(cached.get("model_revision", "")) or None
            outcome = str(cached.get("outcome", "succeeded"))
            failure_code = None

        if output is None or outcome not in {"succeeded", "partial"}:
            record = EvidenceRecord(
                project_id=plan.project_id,
                asset_id=plan.asset_id,
                source_sha256=plan.source_sha256,
                kind="video.visual-observations",
                confidence=0.0,
                payload={
                    "plan_id": plan.id,
                    "partial": True,
                    "observation_count": 0,
                    "provider_receipt_id": receipt_id,
                    "provider_receipt_path": receipt_path,
                    "provider_failure_code": failure_code,
                    "cache_hit": cache_hit,
                    "structural_profile_preserved": True,
                },
                time_base=plan.time_base,
                ranges=[plan.analysis_range],
                provenance=Provenance(
                    tool="aoa-editing-provider-alias",
                    tool_version=__version__,
                    parameters={
                        "requested_alias": VISION_ALIAS,
                        "normalization_version": NORMALIZATION_VERSION,
                        "cache_key": cache_key,
                    },
                    model_id=model_id,
                    model_revision=model_revision,
                    deterministic=False,
                ),
            )
            self.store.save_evidence(record)
            return [], record

        root = self.store.video_anatomy_path(plan.project_id, plan.id) / "visual"
        raw_path = root / f"raw-{cache_key}.json"
        self._atomic_json(raw_path, output)
        raw_relative = str(raw_path.relative_to(self.store.project_path(plan.project_id)))
        raw_hash = sha256_file(raw_path)
        descriptions = [item for item in output.get("descriptions", []) if isinstance(item, dict)]
        observations: list[VideoVisualObservation] = []
        typed_paths: list[str] = []
        for shot in structure.shots:
            shot_samples = [item for item in samples if item.shot_id == shot.id]
            matched = [
                item
                for item in descriptions
                if item.get("time_seconds") is not None
                and shot.frame_range.start
                <= round(float(item["time_seconds"]) * plan.time_base.fps)
                < shot.frame_range.end
            ]
            if not matched and descriptions and shot_samples:
                midpoint = (shot.frame_range.start + shot.frame_range.end - 1) / 2
                nearest = min(
                    descriptions,
                    key=lambda item: abs(
                        float(item.get("time_seconds") or 0) * plan.time_base.fps - midpoint
                    ),
                )
                nearest_time = float(nearest.get("time_seconds") or 0) * plan.time_base.fps
                if shot.frame_range.start - 1 <= nearest_time <= shot.frame_range.end:
                    matched = [nearest]
            if not matched:
                continue
            texts = [str(item.get("text", "")).strip() for item in matched]
            texts = [item for item in texts if item]
            if not texts:
                continue
            confidences = [
                float(item.get("confidence", 0.5))
                for item in matched
                if item.get("confidence") is not None
            ]
            observation = VideoVisualObservation(
                project_id=plan.project_id,
                asset_id=plan.asset_id,
                source_sha256=plan.source_sha256,
                shot_id=shot.id,
                frame_range=shot.frame_range,
                sample_ids=[item.id for item in shot_samples],
                description=" ".join(texts),
                subjects=[],
                composition_regions=[],
                visible_text=[],
                overlays=[],
                confidence=(sum(confidences) / len(confidences) if confidences else 0.5),
                raw_response_artifact=raw_relative,
                raw_response_sha256=raw_hash,
                normalization_version=NORMALIZATION_VERSION,
                provider_receipt_id=receipt_id,
                provenance=Provenance(
                    tool="aoa-editing-provider-alias",
                    tool_version=__version__,
                    parameters={
                        "requested_alias": VISION_ALIAS,
                        "provider_receipt_id": receipt_id,
                        "provider_receipt_path": receipt_path,
                        "cache_key": cache_key,
                        "cache_hit": cache_hit,
                        "normalization_version": NORMALIZATION_VERSION,
                    },
                    model_id=model_id,
                    model_revision=model_revision,
                    deterministic=False,
                ),
            )
            typed_path = self.store.save_video_visual_observation(plan.id, observation)
            observations.append(observation)
            typed_paths.append(
                str(typed_path.relative_to(self.store.project_path(plan.project_id)))
            )
        coverage = len({item.shot_id for item in observations}) / len(structure.shots)
        record = EvidenceRecord(
            project_id=plan.project_id,
            asset_id=plan.asset_id,
            source_sha256=plan.source_sha256,
            kind="video.visual-observations",
            confidence=(
                sum(item.confidence for item in observations) / len(observations)
                if observations
                else 0
            ),
            payload={
                "plan_id": plan.id,
                "partial": outcome == "partial" or coverage < 1,
                "observation_count": len(observations),
                "shot_coverage": coverage,
                "provider_receipt_id": receipt_id,
                "provider_receipt_path": receipt_path,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "model_id": model_id,
                "model_revision": model_revision,
            },
            artifacts=[raw_relative, *typed_paths],
            time_base=plan.time_base,
            ranges=[item.frame_range for item in observations] or [plan.analysis_range],
            provenance=Provenance(
                tool="aoa-editing-video-semantics",
                tool_version=__version__,
                parameters={
                    "normalization_version": NORMALIZATION_VERSION,
                    "cache_key": cache_key,
                    "cache_hit": cache_hit,
                },
                model_id=model_id,
                model_revision=model_revision,
                deterministic=False,
            ),
        )
        self.store.save_evidence(record)
        return observations, record

    def _binding_revision(self) -> str:
        status = self.providers.inspect_bindings(probe=False)
        entry = next(
            (item for item in status.get("bindings", []) if item.get("alias") == VISION_ALIAS),
            None,
        )
        if not isinstance(entry, dict):
            return "unbound"
        return ":".join(
            str(entry.get(name) or "unknown")
            for name in ("model_id", "model_revision", "binding_state")
        )

    @staticmethod
    def _cache_key(
        plan: VideoSamplingPlan,
        times: list[float],
        prompt: str,
        binding_revision: str,
    ) -> str:
        payload = json.dumps(
            {
                "source_sha256": plan.source_sha256,
                "plan_sha256": plan.plan_sha256,
                "profile": plan.profile.value,
                "times": times,
                "prompt": prompt,
                "binding_revision": binding_revision,
                "normalization_version": NORMALIZATION_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
