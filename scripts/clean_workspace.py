#!/usr/bin/env python3
"""Remove only explicitly regenerable AoA Editing cache and temporary state."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aoa_editing.config import Settings


def safe_mutable_root(path: Path, editing_home: Path, category: str) -> Path:
    selected = path.expanduser().resolve()
    home = editing_home.expanduser().resolve()
    if selected == home or not selected.is_relative_to(home):
        raise SystemExit(f"refusing unsafe {category} root: {selected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Perform the displayed cleanup")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    settings = Settings.from_env(source_home=repo)
    mutable = [
        safe_mutable_root(settings.cache_root, settings.editing_home, "cache"),
        safe_mutable_root(settings.tmp_root, settings.editing_home, "temporary"),
    ]
    local = [repo / name for name in (".pytest_cache", ".mypy_cache", ".ruff_cache")]
    targets = [*local, *(child for root in mutable if root.exists() for child in root.iterdir())]
    for target in targets:
        print(target)
    if not args.apply:
        print(
            "Dry run only. Re-run with --apply. Durable projects, sources, renders, "
            "and evals are excluded."
        )
        return
    for target in targets:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    for cache in repo.rglob("__pycache__"):
        shutil.rmtree(cache)
    print("Regenerable cache and temporary state removed.")


if __name__ == "__main__":
    main()
