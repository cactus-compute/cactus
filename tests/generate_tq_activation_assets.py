#!/usr/bin/env python3
"""Generate raw activation inputs for direct Gemma4 TQ weight tests."""

from __future__ import annotations

import argparse
import pathlib
import struct

import numpy as np


def read_k(path: pathlib.Path) -> int:
    header = path.read_bytes()[:32]
    if len(header) < 32 or header[:4] != b"CACT":
        raise ValueError(f"{path} is not a cactus weight file")
    return struct.unpack_from("<Q", header, 24)[0]


def write_activation(path: pathlib.Path, m_rows: int, k_dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    acts = rng.normal(0.0, 0.40, size=(m_rows, k_dim)).astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<4sIII", b"TQAC", 1, m_rows, k_dim))
        f.write(acts.tobytes())
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=pathlib.Path,
                        default=pathlib.Path("weights/gemma-4-e2b-it-tqh-u4-codeorder"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("tests/assets/tq_kernels"))
    args = parser.parse_args()

    tq4_k = read_k(args.weights_root / "layer_0_ffn_gate.weights")
    tq2_k = read_k(args.weights_root / "embed_tokens_per_layer.weights")

    write_activation(args.out / "tq4_gemv_activations.bin", 1, tq4_k, 9401)
    write_activation(args.out / "tq4_gemm_activations.bin", 8, tq4_k, 9402)
    write_activation(args.out / "tq2_gemv_activations.bin", 1, tq2_k, 9201)
    write_activation(args.out / "tq2_gemm_activations.bin", 4, tq2_k, 9202)


if __name__ == "__main__":
    main()
