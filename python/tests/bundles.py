from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "weights"


def _read_model_type(bundle: Path) -> str:
    for line in (bundle / "config.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("model_type="):
            return line.split("=", 1)[1].strip()
    return ""


def _valid_bundle(path: Path) -> bool:
    return (path / "config.txt").exists() and (path / "components" / "manifest.json").exists()


def _find_bundle(preferred: list[str], types: set[str], on_missing=pytest.fail) -> Path:
    for name in preferred:
        candidate = WEIGHTS / name
        if candidate.exists() and _valid_bundle(candidate) and _read_model_type(candidate) in types:
            return candidate
    if not WEIGHTS.is_dir():
        on_missing(f"Weights directory not found: {WEIGHTS}")
    for candidate in sorted(WEIGHTS.iterdir()):
        if candidate.is_dir() and _valid_bundle(candidate) and _read_model_type(candidate) in types:
            return candidate
    on_missing(f"No valid live-test bundle found under {WEIGHTS} for model types: {sorted(types)}")
