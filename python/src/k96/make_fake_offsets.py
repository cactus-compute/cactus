#!/usr/bin/env python3
"""Write a fake k96_cluster_offsets.bin into an existing dense Gemma-4 cactus
model dir. Uses uniform 32-aligned clusters (D_FFN/96 each), which is the same
synthetic layout used by tests/test_bench_grouped_mlp_int8.cpp. Lets us exercise
the runtime grouped_mlp_int8 graph op without waiting for the real K96 ckpt.

Usage:
    python -m python.src.k96.make_fake_offsets --model_dir weights/gemma-4-e2b-it
"""
import argparse
import struct
from pathlib import Path

N_LAYERS = 35
INTERMEDIATE = 6144
INTERMEDIATE_WIDE = 12288
DOUBLE_WIDE_START = 15
K_GROUPS = 96
ALIGN = 32


def d_ffn_at(i: int) -> int:
    return INTERMEDIATE_WIDE if i >= DOUBLE_WIDE_START else INTERMEDIATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()

    out = Path(args.model_dir) / "k96_cluster_offsets.bin"
    with open(out, "wb") as f:
        f.write(b"CK96")
        f.write(struct.pack("<II", 1, N_LAYERS))
        for i in range(N_LAYERS):
            d = d_ffn_at(i)
            g_size = d // K_GROUPS  # 64 or 128 — both multiples of 32
            assert g_size % ALIGN == 0, f"layer {i}: g_size {g_size} not aligned"
            f.write(struct.pack("<I", K_GROUPS))
            for k in range(K_GROUPS + 1):
                f.write(struct.pack("<I", k * g_size))
    print(f"Wrote {out}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
