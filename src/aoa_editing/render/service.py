"""Atomic render execution, receipts, and source-lineage enforcement."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from aoa_editing.domain.models import FrameRange, JobReceipt, JobStatus, RenderPlan
from aoa_editing.infrastructure.media import sha256_file
from aoa_editing.infrastructure.store import ProjectStore
from aoa_editing.render.compiler import FFmpegCompiler
from aoa_editing.render.compositor import MatrixCompositor, cleanup_compositor_jobs


class RenderError(RuntimeError):
    """Render failed before an output was atomically promoted."""


class RenderService:
    def __init__(self, store: ProjectStore):
        self.store = store
        self.compiler = FFmpegCompiler(store)

    def render(
        self,
        project_id: str,
        version_id: str | None = None,
        *,
        profile: str = "preview",
        frame_range: FrameRange | None = None,
    ) -> JobReceipt:
        project = self.store.load_project(project_id)
        version = self.store.load_version(project_id, version_id)
        output_dir = self.store.project_path(project_id) / "renders" / version.id / profile
        if frame_range is not None:
            output_dir = (
                output_dir / "segments" / f"{frame_range.start:09d}-{frame_range.duration:09d}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "video.mp4"
        partial_path = output_dir / "video.partial.mp4"
        if partial_path.exists():
            partial_path.unlink()
        plan = self.compiler.compile(
            project_id, version, profile, partial_path, frame_range=frame_range
        )
        _atomic_text(output_dir / "render-plan.json", plan.model_dump_json(indent=2) + "\n")
        sealed = set(project.sealed_reference_hashes)
        leaked = sorted(sealed.intersection(plan.input_hashes))
        if leaked:
            raise RenderError(f"sealed reference hash reached render lineage: {leaked}")
        receipt = JobReceipt(
            kind=f"render.{profile}" + (".segment" if frame_range else ""),
            project_id=project_id,
            version_id=version.id,
            status=JobStatus.RUNNING,
            started_at=datetime.now(UTC),
            command=plan.command,
            input_hashes=plan.input_hashes,
        )
        self.store.save_job(receipt)
        try:
            compositor = MatrixCompositor()
            for job in plan.compositor_jobs:
                compositor.render(job)
            result = subprocess.run(
                plan.command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode != 0:
                raise RenderError(result.stderr[-5000:] or "FFmpeg returned a non-zero status")
            _validate_render(partial_path)
            os.replace(partial_path, final_path)
            digest = sha256_file(final_path)
            completed = receipt.model_copy(
                update={
                    "status": JobStatus.SUCCEEDED,
                    "finished_at": datetime.now(UTC),
                    "output_paths": [
                        str(final_path.relative_to(self.store.project_path(project_id)))
                    ],
                    "output_hashes": [digest],
                }
            )
            self.store.save_job(completed)
            self._write_lineage(project_id, version.id, profile, plan, digest)
            return completed
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            failed = receipt.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "finished_at": datetime.now(UTC),
                    "error": str(error),
                }
            )
            self.store.save_job(failed)
            if isinstance(error, RenderError):
                raise
            raise RenderError(str(error)) from error
        finally:
            cleanup_compositor_jobs(plan.compositor_jobs)

    def _write_lineage(
        self,
        project_id: str,
        version_id: str,
        profile: str,
        plan: RenderPlan,
        output_hash: str,
    ) -> None:
        import json

        path = plan.output_path.parent / "lineage.json"
        payload = {
            "schema": "aoa_editing_render_lineage_v1",
            "project_id": project_id,
            "version_id": version_id,
            "profile": profile,
            "frame_range": (plan.frame_range.model_dump(mode="json") if plan.frame_range else None),
            "input_hashes": plan.input_hashes,
            "output_sha256": output_hash,
            "forbidden_hashes_present": [],
            "command": plan.command,
            "compositor_jobs": [
                {
                    "motion_language_version": job.motion_language_version,
                    "clip_id": job.clip_id,
                    "source_sha256": job.source_sha256,
                    "frame_count": len(job.frames),
                    "samples_per_frame": len(job.frames[0].matrices_3x3),
                    "interpolation": job.interpolation,
                    "encoder_command": job.encoder_command,
                }
                for job in plan.compositor_jobs
            ],
        }
        _atomic_text(path, json.dumps(payload, indent=2) + "\n")


def _atomic_text(path: Path, payload: str) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(payload, encoding="utf-8")
    os.replace(partial, path)


def _validate_render(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RenderError("render output is missing or implausibly small")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or '"codec_name"' not in result.stdout:
        raise RenderError(result.stderr.strip() or "ffprobe could not validate output")
