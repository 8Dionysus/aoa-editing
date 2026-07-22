#!/usr/bin/env python3
"""Resolve one StorageLayout field without requiring the project virtualenv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aoa_editing.config import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "field",
        choices=(
            "home",
            "runtime",
            "projects",
            "evals",
            "artifacts",
            "cache",
            "tmp",
            "state",
        ),
    )
    field = parser.parse_args().field
    settings = Settings.from_env(source_home=ROOT)
    paths = {
        "home": settings.editing_home,
        "runtime": settings.runtime_root,
        "projects": settings.projects_root,
        "evals": settings.evals_root,
        "artifacts": settings.artifacts_root,
        "cache": settings.cache_root,
        "tmp": settings.tmp_root,
        "state": settings.state_root,
    }
    print(paths[field])


if __name__ == "__main__":
    main()
