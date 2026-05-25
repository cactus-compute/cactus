# Temporary artifact helper: rewrites full-width orthogonal CQ4 weights into
# INTERLEAVED_4ROW storage without full model reconversion.
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from .export.qdq import (
    ALIGNMENT_DEFAULT,
    CACTUS_MAGIC,
    FLAG_INTERLEAVED_4ROW,
    FLAG_ORTHOGONAL_ROTATION,
    HEADER_SIZE,
    align_offset,
    read_cq_payload,
    read_header,
    unpack_lsb_values,
)
from .quantization.cq import PRECISION_CQ, pack_indices_interleaved_4row


def _padding(offset: int, alignment: int) -> bytes:
    return b"\0" * (align_offset(offset, alignment) - offset)


def interleave_orthogonal_cq4_file(input_path: Path, output_path: Path, *, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} exists; pass --force to overwrite")

    header = read_header(input_path)
    if header.ndim != 2:
        raise ValueError(f"{input_path}: expected 2D CQ tensor")
    n, k = header.shape
    if header.precision != PRECISION_CQ[4] or header.bits != 4:
        raise ValueError(f"{input_path}: expected CQ4")
    if (header.flags & FLAG_ORTHOGONAL_ROTATION) == 0:
        raise ValueError(f"{input_path}: expected orthogonal CQ rotation")
    if header.flags & FLAG_INTERLEAVED_4ROW:
        raise ValueError(f"{input_path}: already INTERLEAVED_4ROW")
    if header.num_groups != 1 or header.group_size != k:
        raise ValueError(
            f"{input_path}: expected one full-width group, got group_size={header.group_size} num_groups={header.num_groups}"
        )
    if n % 4 != 0 or k % 32 != 0:
        raise ValueError(f"{input_path}: expected N % 4 == 0 and K % 32 == 0, got shape={header.shape}")

    scales_blob, packed = read_cq_payload(input_path, header)
    rows = packed.reshape(n, k // 2)
    indices = np.stack([unpack_lsb_values(row, k, 4) for row in rows])
    interleaved = pack_indices_interleaved_4row(indices, k, 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    alignment = header.alignment or ALIGNMENT_DEFAULT
    scales_offset = align_offset(HEADER_SIZE, alignment)
    scales_end = scales_offset + len(scales_blob)
    data_offset = align_offset(scales_end, alignment)
    flags = header.flags | FLAG_INTERLEAVED_4ROW

    with output_path.open("wb") as out:
        out.write(CACTUS_MAGIC)
        out.write(struct.pack("<I", flags))
        out.write(struct.pack("<I", alignment))
        out.write(struct.pack("<I", header.ndim))
        for dim in header.dims:
            out.write(struct.pack("<Q", dim))
        out.write(struct.pack("<I", header.precision))
        out.write(struct.pack("<Q", int(interleaved.size)))
        out.write(struct.pack("<Q", len(scales_blob)))
        out.write(struct.pack("<I", header.group_size))
        out.write(struct.pack("<I", header.num_groups))
        out.write(struct.pack("<Q", header.original_n))
        out.write(_padding(HEADER_SIZE, alignment))
        out.write(scales_blob)
        out.write(_padding(scales_end, alignment))
        out.write(interleaved.tobytes())

    expected = data_offset + int(interleaved.size)
    actual = output_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{output_path}: wrote {actual} bytes, expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a full-width orthogonal CQ4 file to 4-row interleaved storage.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    interleave_orthogonal_cq4_file(args.input, args.output, force=args.force)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
