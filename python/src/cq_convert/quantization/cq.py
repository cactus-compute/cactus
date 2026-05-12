from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..cactus_adapters.tensor_io import (
    CACTUS_ALIGNMENT,
    CACTUS_MAGIC,
    align_offset,
    compute_padding,
)

HEADER_SIZE = 84
GROUP_SIZE = 128
PRECISION_CQ = {1: 3, 2: 4, 3: 5, 4: 6}
FLAG_ORTHOGONAL_ROTATION = 1 << 1


@dataclass(frozen=True)
class CQTensor:
    indices: np.ndarray
    norms: np.ndarray
    input_scale: np.ndarray
    bits: int
    group_size: int = GROUP_SIZE
    rotation_family: str = "hadamard"
    seed: int = 1234
    gptq_used: bool = False


_CODEBOOK_CACHE: dict[tuple[int, int], np.ndarray] = {}
_HADAMARD_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_ORTHO_CACHE: dict[tuple[int, int], np.ndarray] = {}


def make_codebook(group_dim: int, bits: int, grid_size: int = 200_001) -> np.ndarray:
    key = (int(group_dim), int(bits))
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]
    try:
        from scipy.special import gammaln
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for CQ codebook generation") from exc
    n_centroids = 1 << bits
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    log_c = gammaln(group_dim / 2.0) - 0.5 * math.log(math.pi) - gammaln((group_dim - 1.0) / 2.0)
    weights = math.exp(log_c) * np.power(np.clip(1.0 - grid * grid, 0.0, None), (group_dim - 3.0) / 2.0)
    quantiles = np.linspace(0.0, 1.0, n_centroids + 2, dtype=np.float64)[1:-1]
    cum = np.cumsum(weights)
    cum /= cum[-1]
    centroids = np.sort(np.clip(np.interp(quantiles, cum, grid), -1.0, 1.0))
    for _ in range(200):
        bounds = np.concatenate(([-1.0], (centroids[:-1] + centroids[1:]) / 2.0, [1.0]))
        updated = centroids.copy()
        for i in range(n_centroids):
            mask = (grid >= bounds[i]) & (grid <= bounds[i + 1] if i == n_centroids - 1 else grid < bounds[i + 1])
            ws = weights[mask]
            if ws.sum() > 0:
                updated[i] = float((grid[mask] * ws).sum() / ws.sum())
        if np.max(np.abs(updated - centroids)) < 1e-8:
            centroids = updated
            break
        centroids = np.sort(np.clip(updated, -1.0, 1.0))
    _CODEBOOK_CACHE[key] = centroids.astype(np.float32)
    return _CODEBOOK_CACHE[key]


def make_hadamard_components(group_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (int(group_dim), int(seed))
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for deterministic CQ rotations")
    g = torch.Generator(device="cpu").manual_seed(seed + 17 * group_dim)
    left = (2 * torch.randint(0, 2, (group_dim,), generator=g, dtype=torch.int64) - 1).to(torch.int8).numpy()
    right = (2 * torch.randint(0, 2, (group_dim,), generator=g, dtype=torch.int64) - 1).to(torch.int8).numpy()
    perm = torch.randperm(group_dim, generator=g).to(torch.int64).numpy().astype(np.uint32)
    _HADAMARD_CACHE[key] = (left, right, perm)
    return _HADAMARD_CACHE[key]


def make_hadamard_matrix(group_dim: int, seed: int) -> np.ndarray:
    try:
        from scipy.linalg import hadamard
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for Hadamard CQ rotation") from exc
    left, right, perm = make_hadamard_components(group_dim, seed)
    base = hadamard(group_dim, dtype=np.float32) / math.sqrt(group_dim)
    return (left.astype(np.float32)[:, None] * base * right.astype(np.float32)[None, :])[:, perm]


def make_orthogonal_rotation(group_dim: int, seed: int) -> np.ndarray:
    key = (int(group_dim), int(seed))
    if key in _ORTHO_CACHE:
        return _ORTHO_CACHE[key]
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for deterministic orthogonal CQ rotations")
    g = torch.Generator(device="cpu").manual_seed(seed + 17 * group_dim)
    a = torch.randn(group_dim, group_dim, generator=g)
    q, r = torch.linalg.qr(a, mode="reduced")
    d = torch.sign(torch.diagonal(r))
    d[d == 0] = 1
    _ORTHO_CACHE[key] = (q * d).numpy().astype(np.float32)
    return _ORTHO_CACHE[key]


def pack_indices_lsb(indices: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    n, k = indices.shape
    if bits == 8:
        return indices.astype(np.uint8, copy=False)
    if not 1 <= bits <= 4:
        raise ValueError(f"bits must be 1..4, got {bits}")
    if k % group_size != 0:
        raise ValueError(f"K={k} is not divisible by group_size={group_size}")
    chunk = 8
    bytes_per_chunk = chunk * bits // 8
    chunks_per_group = group_size // chunk
    grouped = indices.reshape(n, k // group_size, chunks_per_group, chunk).astype(np.uint64)
    word = np.zeros(grouped.shape[:-1], dtype=np.uint64)
    for i in range(chunk):
        word |= grouped[..., i] << (i * bits)
    out = np.zeros(word.shape + (bytes_per_chunk,), dtype=np.uint8)
    for byte in range(bytes_per_chunk):
        out[..., byte] = ((word >> (8 * byte)) & 0xFF).astype(np.uint8)
    return out.reshape(n, k * bits // 8)


def _as_numpy_fp32(weight) -> np.ndarray:
    if torch is not None and isinstance(weight, torch.Tensor):
        return weight.detach().float().cpu().numpy()
    return np.asarray(weight, dtype=np.float32)


def _prepare_input_scale(input_scale: np.ndarray | None, k: int) -> np.ndarray:
    if input_scale is None:
        return np.ones(k, dtype=np.float32)
    scale = np.asarray(input_scale, dtype=np.float32).reshape(-1)
    if scale.shape[0] > k:
        scale = scale[:k]
    elif scale.shape[0] < k:
        scale = np.pad(scale, (0, k - scale.shape[0]), constant_values=1.0)
    return np.clip(scale, 1e-8, 65504.0).astype(np.float32, copy=False)


def _gptq_correct_group(work: np.ndarray, recon: np.ndarray, h_inv: np.ndarray | None, start: int, stop: int) -> None:
    if h_inv is None or stop >= work.shape[1]:
        return
    try:
        m_bb = h_inv[start:stop, start:stop].astype(np.float32, copy=False)
        m_bs = h_inv[start:stop, stop:].astype(np.float32, copy=False)
        update = np.linalg.solve(m_bb + np.eye(m_bb.shape[0], dtype=np.float32) * 1e-6, m_bs)
        work[:, stop:] -= (work[:, start:stop] - recon) @ update
    except Exception:
        return


def quantize_hadamard(
    weight,
    bits: int,
    hessian: np.ndarray | None = None,
    use_gptq: bool = False,
    seed: int = 1234,
    input_scale: np.ndarray | None = None,
) -> CQTensor:
    w = _as_numpy_fp32(weight)
    if w.ndim != 2:
        raise ValueError(f"hadamard CQ requires a 2D tensor, got {w.shape}")
    n, k = w.shape
    original_k = k
    if k % GROUP_SIZE != 0:
        pad = GROUP_SIZE - (k % GROUP_SIZE)
        w = np.pad(w, ((0, 0), (0, pad)), mode="constant")
        k = w.shape[1]
    input_scale = _prepare_input_scale(input_scale, k)
    work = w * input_scale[None, :]
    rot = make_hadamard_matrix(GROUP_SIZE, seed)
    codebook = make_codebook(GROUP_SIZE, bits)
    groups = k // GROUP_SIZE
    indices = np.zeros((n, k), dtype=np.uint8)
    norms = np.zeros((n, groups), dtype=np.float16)
    h_inv = None
    if use_gptq and hessian is not None:
        try:
            h = np.asarray(hessian, dtype=np.float32)
            if h.shape[0] == original_k and h.shape[0] == h.shape[1]:
                if original_k != k:
                    h = np.pad(h, ((0, k - original_k), (0, k - original_k)), mode="constant")
                    np.fill_diagonal(h[original_k:, original_k:], 1.0)
                s = np.clip(input_scale.astype(np.float32), 1e-6, None)
                h = h / (s[:, None] * s[None, :])
                h = h + np.eye(h.shape[0], dtype=np.float32) * (0.01 * np.mean(np.diag(h)) + 1e-6)
                h_inv = np.linalg.inv(h)
        except Exception:
            h_inv = None
    for g in range(groups):
        start, stop = g * GROUP_SIZE, (g + 1) * GROUP_SIZE
        group = work[:, start:stop]
        row_norms = np.linalg.norm(group, axis=1).clip(min=1e-8).astype(np.float32)
        rotated = (group / row_norms[:, None]) @ rot
        idx = np.abs(rotated[..., None] - codebook[None, None, :]).argmin(axis=-1).astype(np.uint8)
        recon = (codebook[idx] @ rot.T) * row_norms[:, None]
        indices[:, start:stop] = idx
        norms[:, g] = row_norms.astype(np.float16)
        _gptq_correct_group(work, recon, h_inv, start, stop)
    return CQTensor(indices=indices, norms=norms, input_scale=input_scale.astype(np.float16), bits=bits, gptq_used=bool(use_gptq and h_inv is not None))


def quantize_orthogonal(weight, bits: int = 4, seed: int = 1234, input_scale: np.ndarray | None = None) -> CQTensor:
    w = _as_numpy_fp32(weight)
    if w.ndim != 2:
        raise ValueError(f"orthogonal CQ requires a 2D tensor, got {w.shape}")
    n, k = w.shape
    input_scale = _prepare_input_scale(input_scale, k)
    work = w * input_scale[None, :]
    rot = make_orthogonal_rotation(k, seed)
    codebook = make_codebook(k, bits)
    norms = np.linalg.norm(work, axis=1).clip(min=1e-8).astype(np.float32)
    indices = np.zeros((n, k), dtype=np.uint8)
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        rotated = (work[start:stop] / norms[start:stop, None]) @ rot
        indices[start:stop] = np.abs(rotated[..., None] - codebook[None, None, :]).argmin(axis=-1).astype(np.uint8)
    return CQTensor(indices=indices, norms=norms.astype(np.float16).reshape(n, 1), input_scale=input_scale.astype(np.float16), bits=bits, group_size=k, rotation_family="orthogonal")


def write_cq_tensor(out_path: Path, cq: CQTensor) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, k = cq.indices.shape
    group_size = int(cq.group_size)
    groups = k // group_size
    packed = pack_indices_lsb(cq.indices, group_size, cq.bits)
    codebook = make_codebook(group_size, cq.bits).astype(np.float16)
    input_scale = cq.input_scale.astype(np.float16, copy=False).reshape(k)
    recip = np.minimum(1.0 / np.maximum(input_scale.astype(np.float32), 1e-8), 65504.0).astype(np.float16)
    parts = [codebook.tobytes(), input_scale.tobytes(), recip.tobytes(), cq.norms.astype(np.float16, copy=False).tobytes()]
    flags = 0
    if cq.rotation_family == "orthogonal":
        flags |= FLAG_ORTHOGONAL_ROTATION
        parts.append(make_orthogonal_rotation(group_size, cq.seed).astype(np.float16).tobytes())
    else:
        left, right, perm = make_hadamard_components(group_size, cq.seed)
        parts.extend([left.tobytes(), right.tobytes(), perm.astype("<u4", copy=False).tobytes()])
    scales_blob = b"".join(parts)
    scales_bytes = len(scales_blob)
    data_bytes = packed.size
    scales_end = align_offset(HEADER_SIZE, CACTUS_ALIGNMENT) + scales_bytes
    data_offset = align_offset(scales_end, CACTUS_ALIGNMENT)
    with out_path.open("wb") as f:
        f.write(CACTUS_MAGIC)
        f.write(struct.pack("<I", flags))
        f.write(struct.pack("<I", CACTUS_ALIGNMENT))
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<Q", k))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<I", PRECISION_CQ[cq.bits]))
        f.write(struct.pack("<Q", data_bytes))
        f.write(struct.pack("<Q", scales_bytes))
        f.write(struct.pack("<I", group_size))
        f.write(struct.pack("<I", groups))
        f.write(struct.pack("<Q", n))
        f.write(compute_padding(HEADER_SIZE, CACTUS_ALIGNMENT))
        f.write(scales_blob)
        f.write(compute_padding(scales_end, CACTUS_ALIGNMENT))
        f.write(packed.tobytes())
    expected = data_offset + data_bytes
    actual = out_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{out_path}: wrote {actual} bytes, expected {expected}")
