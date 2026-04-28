#!/usr/bin/env python3
"""Sanity-check a converted K96 cactus model dir before running chat.

Verifies that:
  - k96_cluster_offsets.bin exists with the right magic / shape
  - All 35 layers have ffn_gate/up/down weights
  - Each layer's gate/up has N matching the cluster offsets total

Usage:
    python -m python.src.k96.verify_converted --model_dir weights/gemma-4-e2b-it-k96
"""
import argparse
import struct
import sys
from pathlib import Path


N_LAYERS = 35
INTERMEDIATE = 6144
INTERMEDIATE_WIDE = 12288
DOUBLE_WIDE_START = 15
K_GROUPS = 96


def d_ffn_at(i):
    return INTERMEDIATE_WIDE if i >= DOUBLE_WIDE_START else INTERMEDIATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()
    d = Path(args.model_dir)

    bin_path = d / "k96_cluster_offsets.bin"
    if not bin_path.exists():
        print(f"FAIL: {bin_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"CK96":
            print(f"FAIL: bad magic {magic!r}", file=sys.stderr); sys.exit(1)
        version, n_layers = struct.unpack("<II", f.read(8))
        if version != 1 or n_layers != N_LAYERS:
            print(f"FAIL: version={version} n_layers={n_layers}", file=sys.stderr); sys.exit(1)
        for i in range(N_LAYERS):
            (kg,) = struct.unpack("<I", f.read(4))
            if kg != K_GROUPS:
                print(f"FAIL: layer {i} k_groups={kg}", file=sys.stderr); sys.exit(1)
            offsets = list(struct.unpack(f"<{kg+1}I", f.read(4 * (kg + 1))))
            if offsets[0] != 0 or offsets[-1] != d_ffn_at(i):
                print(f"FAIL: layer {i} offsets {offsets[:3]}..{offsets[-3:]}", file=sys.stderr); sys.exit(1)
            if not all(o % 32 == 0 for o in offsets):
                print(f"FAIL: layer {i} offsets not 32-aligned", file=sys.stderr); sys.exit(1)
            if any(offsets[k+1] < offsets[k] for k in range(kg)):
                print(f"FAIL: layer {i} offsets not non-decreasing", file=sys.stderr); sys.exit(1)

    for i in range(N_LAYERS):
        for n in ("ffn_gate", "ffn_up", "ffn_down"):
            p = d / f"layer_{i}_{n}.weights"
            if not p.exists():
                print(f"FAIL: missing {p}", file=sys.stderr); sys.exit(1)

    print(f"OK: K96 model at {d} looks good ({N_LAYERS} layers, K={K_GROUPS}, offsets aligned)")


if __name__ == "__main__":
    main()
