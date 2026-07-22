from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


def test_checked_in_schemas_are_current() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts" / "generate_schemas.py"), "--check"],
        check=True,
        cwd=root,
    )


def test_all_schemas_are_valid_draft_2020_12() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "schemas" / "v1").glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)
