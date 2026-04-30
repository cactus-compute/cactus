#!/usr/bin/env python3
"""Generate model-shaped fixtures for the isolated TQ kernels.

The fixtures intentionally contain raw activations and input scales. The kernels
must fold input_scale into activations, rotate with the Hadamard signs, and then
consume packed codebook indices without materializing full weights.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import struct

import numpy as np


OP_GEMV = 1
OP_GEMM = 2
FLAG_CODE_ORDERED_INDICES = 1 << 1


def f16(x):
    return np.asarray(x, dtype=np.float16)


def hsum8(v: np.ndarray) -> np.float16:
    v = f16(v)
    sum4 = f16(v[:4] + v[4:])
    sum2 = f16(sum4 + np.roll(sum4, -2))
    sum1 = f16(sum2 + np.roll(sum2, -1))
    return np.float16(sum1[0])


def fma_f16(acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return f16(acc.astype(np.float32) + a.astype(np.float32) * b.astype(np.float32))


def fwht_f16(x: np.ndarray) -> np.ndarray:
    y = f16(x.copy())
    h = 1
    n = y.shape[0]
    while h < n:
        for i in range(0, n, h * 2):
            a = y[i : i + h].copy()
            b = y[i + h : i + h * 2].copy()
            y[i : i + h] = f16(a + b)
            y[i + h : i + h * 2] = f16(a - b)
        h *= 2
    return f16(y * np.float16(1.0 / math.sqrt(n)))


def make_rotation(group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234 + 17 * group_size)
    left = rng.choice(np.array([-1, 1], dtype=np.int8), size=group_size)
    right = rng.choice(np.array([-1, 1], dtype=np.int8), size=group_size)
    perm = rng.permutation(group_size).astype(np.uint32)
    return left, right, perm


def make_case(bits: int, op: int) -> dict[str, np.ndarray | int]:
    group_size = 128
    num_groups = 2
    k_dim = group_size * num_groups
    n_rows = 29
    m_rows = 1 if op == OP_GEMV else 5
    cb_size = 1 << bits
    rng = np.random.default_rng(9000 + bits * 101 + op)

    if bits == 2:
        codebook = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float16)
    else:
        codebook = np.linspace(-1.875, 1.875, cb_size, dtype=np.float16)

    input_scale = f16(rng.uniform(0.60, 1.70, size=k_dim))
    norms = f16(rng.uniform(0.35, 1.45, size=(n_rows, num_groups)))
    activations = f16(rng.normal(0.0, 0.40, size=(m_rows, k_dim)))
    indices = rng.integers(0, cb_size, size=(n_rows, num_groups, group_size), dtype=np.uint8)

    left, right, perm = make_rotation(group_size)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(group_size, dtype=np.uint32)

    # Match cactus TQ conversion: pre-apply inv_perm so runtime can skip gathers.
    code_ordered = indices[:, :, inv_perm].copy()
    flags = FLAG_CODE_ORDERED_INDICES
    packed = pack_indices(code_ordered, bits)
    expected = expected_for_kernel(bits, op, codebook, input_scale, norms, code_ordered,
                                   activations, left, right, flags)
    return {
        "bits": bits,
        "op": op,
        "M": m_rows,
        "K": k_dim,
        "N": n_rows,
        "group_size": group_size,
        "num_groups": num_groups,
        "flags": flags,
        "codebook": codebook,
        "input_scale": input_scale,
        "left": left,
        "right": right,
        "perm": perm,
        "norms": norms.reshape(-1),
        "packed": packed,
        "A": activations.reshape(-1),
        "expected": expected.reshape(-1),
    }


def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    rows, groups, group_size = indices.shape
    per_group_bytes = (group_size * bits) // 8
    out = np.zeros((rows, groups, per_group_bytes), dtype=np.uint8)
    if bits == 2:
        for k in range(0, group_size, 4):
            b = (indices[:, :, k].astype(np.uint8)
                 | (indices[:, :, k + 1].astype(np.uint8) << 2)
                 | (indices[:, :, k + 2].astype(np.uint8) << 4)
                 | (indices[:, :, k + 3].astype(np.uint8) << 6))
            out[:, :, k // 4] = b
    elif bits == 4:
        for k in range(0, group_size, 2):
            b = indices[:, :, k].astype(np.uint8) | (indices[:, :, k + 1].astype(np.uint8) << 4)
            out[:, :, k // 2] = b
    else:
        raise ValueError(bits)
    return out.reshape(-1)


def transform_activations(A: np.ndarray, input_scale: np.ndarray, left: np.ndarray,
                          right: np.ndarray, group_size: int, num_groups: int) -> np.ndarray:
    code_basis = np.empty_like(A)
    recip = f16(1.0 / input_scale.astype(np.float32))
    for m in range(A.shape[0]):
        for g in range(num_groups):
            s = g * group_size
            e = s + group_size
            work = f16(A[m, s:e] * recip[s:e])
            work = f16(work * left.astype(np.float16))
            work = fwht_f16(work)
            work = f16(work * right.astype(np.float16))
            code_basis[m, s:e] = work
    return code_basis


def expected_for_kernel(bits: int, op: int, codebook: np.ndarray, input_scale: np.ndarray,
                        norms: np.ndarray, indices: np.ndarray, A: np.ndarray,
                        left: np.ndarray, right: np.ndarray, flags: int) -> np.ndarray:
    del flags
    n_rows, num_groups, group_size = indices.shape
    code_basis = transform_activations(A, input_scale, left, right, group_size, num_groups)
    if op == OP_GEMV:
        return gemv_expected(bits, codebook, norms, indices, code_basis)
    return gemm_expected(bits, codebook, norms, indices, code_basis)


def gemv_expected(bits: int, codebook: np.ndarray, norms: np.ndarray, indices: np.ndarray,
                  code_basis: np.ndarray) -> np.ndarray:
    n_rows, num_groups, group_size = indices.shape
    out = np.zeros((n_rows,), dtype=np.float16)
    cb = f16(codebook)
    for n in range(n_rows):
        total = 0.0
        for g in range(num_groups):
            z = code_basis[0, g * group_size : (g + 1) * group_size]
            rn = float(norms[n, g])
            if bits == 4:
                acc0 = np.zeros(8, dtype=np.float16)
                acc1 = np.zeros(8, dtype=np.float16)
                for k in range(0, group_size, 16):
                    cv0 = cb[indices[n, g, k : k + 8]]
                    cv1 = cb[indices[n, g, k + 8 : k + 16]]
                    acc0 = fma_f16(acc0, z[k : k + 8], cv0)
                    acc1 = fma_f16(acc1, z[k + 8 : k + 16], cv1)
                total += rn * (float(hsum8(acc0)) + float(hsum8(acc1)))
            else:
                acc = np.zeros(8, dtype=np.float16)
                for k in range(0, group_size, 8):
                    cv = cb[indices[n, g, k : k + 8]]
                    acc = fma_f16(acc, z[k : k + 8], cv)
                total += rn * float(hsum8(acc))
        out[n] = np.float16(total)
    return out.reshape(1, n_rows)


def gemm_expected(bits: int, codebook: np.ndarray, norms: np.ndarray, indices: np.ndarray,
                  code_basis: np.ndarray) -> np.ndarray:
    n_rows, num_groups, group_size = indices.shape
    m_rows = code_basis.shape[0]
    out = np.zeros((m_rows, n_rows), dtype=np.float16)
    cb = f16(codebook)
    tile_n = 16
    for n_start in range(0, n_rows, tile_n):
        actual_n = min(tile_n, n_rows - n_start)
        out[:, n_start : n_start + actual_n] = np.float16(0)
        for g in range(num_groups):
            b_tile = np.empty((actual_n, group_size), dtype=np.float16)
            for ni in range(actual_n):
                row = n_start + ni
                b_tile[ni] = f16(cb[indices[row, g]] * norms[row, g])
            for m_start in range(0, m_rows, 4):
                actual_m = min(4, m_rows - m_start)
                acc = np.zeros((actual_m, actual_n, 8), dtype=np.float16)
                a_base = code_basis[m_start : m_start + actual_m,
                                    g * group_size : (g + 1) * group_size]
                for k in range(0, group_size, 16):
                    for ni in range(actual_n):
                        b_lo = b_tile[ni, k : k + 8]
                        b_hi = b_tile[ni, k + 8 : k + 16]
                        for mi in range(actual_m):
                            acc[mi, ni] = fma_f16(acc[mi, ni], a_base[mi, k : k + 8], b_lo)
                            acc[mi, ni] = fma_f16(acc[mi, ni], a_base[mi, k + 8 : k + 16], b_hi)
                for mi in range(actual_m):
                    for ni in range(actual_n):
                        dst = out[m_start + mi, n_start + ni]
                        out[m_start + mi, n_start + ni] = np.float16(float(dst) + float(hsum8(acc[mi, ni])))
    return out


def write_fixture(path: pathlib.Path, case: dict[str, np.ndarray | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<4sIIIIIIIII",
        b"TQFX",
        4,
        int(case["bits"]),
        int(case["op"]),
        int(case["M"]),
        int(case["K"]),
        int(case["N"]),
        int(case["group_size"]),
        int(case["num_groups"]),
        int(case["flags"]),
    )
    with path.open("wb") as f:
        f.write(header)
        for key in ("codebook", "input_scale", "left", "right", "perm", "norms", "packed", "A", "expected"):
            f.write(np.asarray(case[key]).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tests/assets/tq_kernels"))
    args = parser.parse_args()

    names = {
        (4, OP_GEMV): "tq4_gemv.bin",
        (4, OP_GEMM): "tq4_gemm.bin",
        (2, OP_GEMV): "tq2_gemv.bin",
        (2, OP_GEMM): "tq2_gemm.bin",
    }
    for (bits, op), name in names.items():
        write_fixture(args.out / name, make_case(bits, op))
        print(args.out / name)


if __name__ == "__main__":
    main()
