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


def _iter_bundle_candidates(name: str):
    """Yield the bare bundle dir for `name` plus any suffixed `name-cq*` variants
    (e.g. `gemma-4-e2b-it-cq4`, `gemma-4-e2b-it-cq4-apple`) so callers that know a
    model's bare stem still find bundles built under the suffixed convention."""
    bare = WEIGHTS / name
    if bare.exists():
        yield bare
    if WEIGHTS.is_dir():
        for candidate in sorted(WEIGHTS.glob(f"{name}-cq*"), reverse=True):
            if candidate.is_dir():
                yield candidate


def _find_bundle(preferred: list[str], types: set[str], on_missing=pytest.fail) -> Path:
    for name in preferred:
        for candidate in _iter_bundle_candidates(name):
            if _valid_bundle(candidate) and _read_model_type(candidate) in types:
                return candidate
    if not WEIGHTS.is_dir():
        on_missing(f"Weights directory not found: {WEIGHTS}")
    for candidate in sorted(WEIGHTS.iterdir()):
        if candidate.is_dir() and _valid_bundle(candidate) and _read_model_type(candidate) in types:
            return candidate
    on_missing(f"No valid live-test bundle found under {WEIGHTS} for model types: {sorted(types)}")
