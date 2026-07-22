#!/usr/bin/env python3
"""Generate checked-in JSON Schemas from canonical Pydantic models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aoa_editing.domain.models import SCHEMA_MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if checked-in schemas drift")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "schemas" / "v1"
    if not arguments.check:
        output.mkdir(parents=True, exist_ok=True)
    drift: list[Path] = []
    for name, model in sorted(SCHEMA_MODELS.items()):
        path = output / f"{name}.schema.json"
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://agents-of-abyss.local/aoa-editing/v1/{name}.schema.json"
        rendered = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
        if arguments.check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                drift.append(path)
        else:
            path.write_text(rendered, encoding="utf-8")
            print(path.relative_to(root))
    if drift:
        for path in drift:
            print(f"schema drift: {path.relative_to(root)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
