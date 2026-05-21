#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import struct
from pathlib import Path

import numpy as np

from cactus.convert.export.qdq import (
    FLAG_ORTHOGONAL_ROTATION,
    read_cq_payload,
    read_header,
    unpack_lsb_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a folded rowwise-int8 LM-head sidecar.")
    parser.add_argument("input", type=Path, help="CQ4 one-group orthogonal LM-head/token_embeddings.weights")
    parser.add_argument("output", type=Path, help="Output .cache sidecar")
    parser.add_argument("--row-batch", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to overwrite")
    if args.row_batch <= 0 or args.row_batch % 4 != 0:
        raise ValueError("--row-batch must be a positive multiple of 4")

    header = read_header(args.input)
    if header.bits != 4 or header.num_groups != 1 or header.group_size != header.shape[1]:
        raise ValueError("expected one-group CQ4 LM-head")
    if (header.flags & FLAG_ORTHOGONAL_ROTATION) == 0:
        raise ValueError("expected orthogonal CQ rotation")

    n, k = header.shape
    if n % 4 != 0 or k % 16 != 0:
        raise ValueError(f"expected N multiple of 4 and K multiple of 16, got shape {header.shape}")
    scales_blob, packed = read_cq_payload(args.input, header)
    pos = 0
    codebook = np.frombuffer(scales_blob, dtype=np.float16, count=1 << header.bits, offset=pos).astype(np.float32)
    pos += (1 << header.bits) * 2
    pos += k * 2
    input_scale_recip = np.frombuffer(scales_blob, dtype=np.float16, count=k, offset=pos).astype(np.float32)
    pos += k * 2
    norms = np.frombuffer(scales_blob, dtype=np.float16, count=n, offset=pos).astype(np.float32)
    pos += n * 2
    rotation = np.frombuffer(scales_blob, dtype=np.float16, count=k * k, offset=pos).astype(np.float32).reshape(k, k)
    rotation_t = np.ascontiguousarray(rotation.T)

    packed_row_bytes = math.ceil(k * header.bits / 8)
    packed_rows = packed.reshape(n, packed_row_bytes)
    scales = np.empty(n, dtype=np.float32)
    row_sums = np.empty(n, dtype=np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights_tmp = args.output.with_suffix(args.output.suffix + ".weights.tmp")
    with weights_tmp.open("wb") as weights_handle:
        for start in range(0, n, args.row_batch):
            stop = min(start + args.row_batch, n)
            idx_np = np.stack([unpack_lsb_values(row, k, header.bits) for row in packed_rows[start:stop]])
            dq = codebook[idx_np]
            folded = (dq @ rotation_t) * norms[start:stop, None]
            folded *= input_scale_recip[None, :]
            max_abs = np.max(np.abs(folded), axis=1)
            batch_scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
            q = np.rint(folded / batch_scales[:, None]).clip(-127, 127).astype(np.int8)
            scales[start:stop] = batch_scales
            row_sums[start:stop] = q.astype(np.int32).sum(axis=1)
            weights_handle.write(np.ascontiguousarray(q.reshape(-1, 4, k).transpose(0, 2, 1)).tobytes())

    header_blob = struct.pack(
        "<4sIIIIQQQ",
        b"CLM8",
        1,
        k,
        n,
        1,
        scales.nbytes,
        row_sums.nbytes,
        n * k,
    )
    with args.output.open("wb") as out:
        out.write(header_blob)
        out.write(scales.tobytes())
        out.write(row_sums.tobytes())
        with weights_tmp.open("rb") as weights_handle:
            shutil.copyfileobj(weights_handle, out, length=16 * 1024 * 1024)
    weights_tmp.unlink()
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
