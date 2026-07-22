from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "",
    ".css",
    ".html",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_PUBLIC_FRAGMENTS = (
    "/home/" + "dionysus",
    "/srv/" + "AbyssOS",
    "/srv/" + "abyss-machine",
    "/dev/" + "nvme",
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [item.decode() for item in result.stdout.split(b"\0") if item]


def test_public_seed_excludes_operator_runtime_state_and_paths() -> None:
    tracked = _tracked_files()

    assert "evals/reference.lock.json" not in tracked
    assert "evals/reference.lock.example.json" in tracked
    assert "manifests/environment.json" not in tracked
    assert "manifests/environment.example.json" in tracked
    assert not any(path.startswith("var/") for path in tracked)

    violations: list[tuple[str, str]] = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for fragment in FORBIDDEN_PUBLIC_FRAGMENTS:
            if fragment in text:
                violations.append((relative, fragment))

    assert violations == []
