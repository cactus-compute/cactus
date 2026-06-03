"""Generate cross-check fixtures for the C++ KeyDiff KV-compression port.

Produces a JSON fixture file consumed by cactus-engine/tests/test_kv_compress.cpp.
Each fixture holds a random PRE-RoPE key matrix (n_kv_heads, n, head_dim) plus the
reference outputs from the Python source of truth so the C++ port can be checked
bit-for-bit:

  * keydiff_score (diagnostic_separability.py) -> per-head [n] scores
  * build_geom_keepset (geom_keepset.py)       -> per-head kept index lists

A separate "rope" fixture provides a single pre-RoPE vector + the rotate_half RoPE
applied at a chosen rank position, mirroring the cache's post-RoPE storage convention
(`out[i] = x[i]*cos - x[i+half]*sin`, `out[i+half] = x[i+half]*cos + x[i]*sin`,
`angle = pos * theta^(-2i/head_dim)`), so the C++ Route-B renumber math can be checked.

The generated fixture JSON (~360 KB) is checked into the repo, so the C++ test runs
with no Python step. Regenerating it requires the untracked MLX KeyDiff harness
(diagnostic_separability.py / geom_keepset.py) to be importable on the path.

Run:  python experiments/niah/gen_kv_compress_fixtures.py
Writes: experiments/niah/fixtures/kv_compress_fixtures.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostic_separability import keydiff_score
from geom_keepset import build_geom_keepset

OUT = Path(__file__).parent / "fixtures" / "kv_compress_fixtures.json"

# (name, n, head_dim, budget_frac, recent_frac, sink, seed)
# Kept small (n<=48, head_dim<=32) so the fixture covers the budget/recent/sink edge cases,
# the reserved-overflow fallback, and the planted-outlier geometry pick the tests assert on.
GRID = [
    ("base_keydiff",       48, 16, 0.25, 0.30, 4, 0),
    ("small_n_keydiff",    20,  8, 0.25, 0.30, 4, 2),
    ("tight_budget",       40,  8, 0.10, 0.30, 4, 3),
    ("reserved_overflow",  30,  8, 0.20, 0.80, 8, 4),
    ("full_budget",        40, 16, 1.00, 0.30, 4, 5),
    ("big_dim_keydiff",    40, 32, 0.25, 0.30, 4, 7),
]


def _outlier_keys(n_kv_heads, n, d, rng):
    """Tight cluster around one centroid direction, with planted mid-context outliers
    so the geometry selection is exercised (not just sink/recent)."""
    base = rng.normal(size=(d,))
    base /= np.linalg.norm(base)
    keys = base[None, None, :] + 0.05 * rng.normal(size=(n_kv_heads, n, d))
    mid = n // 2
    for h in range(n_kv_heads):
        keys[h, mid + h] = -3.0 * base + 0.01 * rng.normal(size=(d,))
    return keys.astype(np.float32)


def _budget(budget_frac, n):
    # Mirror the live harness: B = round(budget_frac * n), matching build_geom_keepset's
    # `budget` argument and keepset_for_head's B = min(budget, n).
    return max(1, int(round(budget_frac * n)))


def make_fixture(name, n, d, bf, rf, sink, seed):
    rng = np.random.default_rng(seed)
    keys = _outlier_keys(2 + (seed % 3), n, d, rng)
    n_kv_heads = keys.shape[0]
    budget = _budget(bf, n)

    scores = []
    for h in range(n_kv_heads):
        kh = np.asarray(keys[h], dtype=np.float64)
        scores.append([float(x) for x in keydiff_score(kh)])

    kept = build_geom_keepset(keys, budget=budget, signal="keydiff", sink=sink,
                              recent_frac=rf)
    return {
        "name": name,
        "n": n,
        "head_dim": d,
        "n_kv_heads": n_kv_heads,
        "budget": budget,
        "recent_frac": rf,
        "sink": sink,
        "keys": keys.reshape(-1).astype(np.float32).tolist(),
        "scores": scores,
        "kept": kept,
    }


def _rope_at(vec, pos, theta):
    """rotate_half RoPE applied to a single (head_dim,) vector at position `pos`,
    matching cactus-kernels/src/norms_rope.cpp."""
    d = vec.shape[0]
    half = d // 2
    out = np.empty_like(vec)
    for i in range(half):
        inv = theta ** (-(2.0 * i) / d)
        ang = pos * inv
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = vec[i], vec[i + half]
        out[i] = x1 * c - x2 * s
        out[i + half] = x2 * c + x1 * s
    return out


def make_rope_fixture():
    rng = np.random.default_rng(99)
    d, theta = 16, 1000000.0
    vec = rng.normal(size=(d,)).astype(np.float64)
    abs_pos, rank = 137, 5
    post_rope_at_abs = _rope_at(vec, abs_pos, theta)
    post_rope_at_rank = _rope_at(vec, rank, theta)
    return {
        "head_dim": d,
        "rope_theta": theta,
        "abs_pos": abs_pos,
        "rank": rank,
        "pre_rope": vec.tolist(),
        "post_rope_at_abs": post_rope_at_abs.tolist(),
        "post_rope_at_rank": post_rope_at_rank.tolist(),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixtures = [make_fixture(*spec) for spec in GRID]
    payload = {"fixtures": fixtures, "rope": make_rope_fixture()}
    OUT.write_text(json.dumps(payload))
    print(f"wrote {len(fixtures)} fixtures + rope to {OUT}")


if __name__ == "__main__":
    main()
