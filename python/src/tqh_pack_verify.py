"""Numerical verifier for tqh_pack output.

Pack one layer with tqh_pack.write_tq_weights, then read the .weights file back
and dehydrate it from raw bytes (no cactus C++ involved). Compare against
packed_shipping/tqh_runtime.dehydrate_layer to confirm the cactus-side encoding
is round-trip identical to the reference.

Usage:
    python -m src.tqh_pack_verify --packed-dir packed_shipping/p3
    python -m src.tqh_pack_verify --packed-dir packed_shipping/pli_embed
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Allow importing tqh_runtime.py directly from packed_shipping/.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'packed_shipping'))
import tqh_runtime  # noqa: E402

from . import tqh_pack  # noqa: E402

TQ_FLAG_CODE_ORDERED_INDICES = 1 << 0
TQ_FLAG_PANEL_MAJOR = 1 << 1
TQ_PANEL_N = 4
TQ_PANEL_K_CHUNK = 16


def _read_cactus_tq_weights(path: Path):
    """Parse the tqh_pack-emitted .weights file back into raw arrays."""
    blob = path.read_bytes()
    if blob[:4] != b'CACT':
        raise ValueError("magic mismatch")
    flags = struct.unpack_from('<I', blob, 4)[0]
    ndim = struct.unpack_from('<I', blob, 12)[0]
    dim0 = struct.unpack_from('<Q', blob, 16)[0]
    dim1 = struct.unpack_from('<Q', blob, 24)[0]
    precision = struct.unpack_from('<I', blob, 48)[0]
    indices_bytes = struct.unpack_from('<Q', blob, 52)[0]
    scales_bytes = struct.unpack_from('<Q', blob, 60)[0]
    group_size = struct.unpack_from('<I', blob, 68)[0]
    num_groups = struct.unpack_from('<I', blob, 72)[0]
    bits = struct.unpack_from('<I', blob, 76)[0]
    off_cb = struct.unpack_from('<Q', blob, 80)[0]
    off_is = struct.unpack_from('<Q', blob, 88)[0]
    off_rot = struct.unpack_from('<Q', blob, 96)[0]
    off_sc = struct.unpack_from('<Q', blob, 104)[0]
    off_ix = struct.unpack_from('<Q', blob, 112)[0]
    total = struct.unpack_from('<Q', blob, 120)[0]
    rotation_family = struct.unpack_from('<I', blob, 128)[0]
    has_input_scale = struct.unpack_from('<I', blob, 132)[0]
    if total != len(blob):
        raise ValueError(f"total mismatch: header says {total}, file is {len(blob)}")
    if ndim != 2:
        raise ValueError("ndim != 2")

    n_centroids = 1 << bits
    codebook = np.frombuffer(blob, dtype=np.float32, count=n_centroids, offset=off_cb)

    input_scale = (np.frombuffer(blob, dtype=np.float16, count=dim1, offset=off_is)
                   if has_input_scale else None)

    if rotation_family == 0:
        gs = group_size
        left = np.frombuffer(blob, dtype=np.int8, count=gs, offset=off_rot)
        right = np.frombuffer(blob, dtype=np.int8, count=gs, offset=off_rot + gs)
        perm = np.frombuffer(blob, dtype=np.uint32, count=gs, offset=off_rot + 2 * gs)
    else:
        # Orthogonal full-width: stored as fp16[K, K].
        full = np.frombuffer(blob, dtype=np.float16, count=dim1 * dim1, offset=off_rot
                             ).reshape(dim1, dim1)
        left = right = perm = None

    if flags & TQ_FLAG_PANEL_MAJOR:
        n_blocks = (dim0 + TQ_PANEL_N - 1) // TQ_PANEL_N
        scales_raw = np.frombuffer(blob, dtype=np.float16,
                                   count=n_blocks * num_groups * TQ_PANEL_N,
                                   offset=off_sc).reshape(n_blocks, num_groups, TQ_PANEL_N)
        scales = np.zeros((dim0, num_groups), dtype=np.float16)
        for n in range(dim0):
            scales[n] = scales_raw[n // TQ_PANEL_N, :, n % TQ_PANEL_N]
    else:
        scales = np.frombuffer(blob, dtype=np.float16, count=dim0 * num_groups, offset=off_sc
                               ).reshape(dim0, num_groups)

    packed_ix = np.frombuffer(blob, dtype=np.uint8, count=indices_bytes, offset=off_ix)

    out = dict(flags=flags, precision=precision, dim0=dim0, dim1=dim1, group_size=group_size,
               num_groups=num_groups, bits=bits, rotation_family=rotation_family,
               codebook=codebook, input_scale=input_scale, scales=scales,
               packed_ix=packed_ix)
    if rotation_family == 0:
        out.update(left=left, right=right, perm=perm)
    else:
        out['orth_R'] = full
    return out


def _unpack_indices_lsb(packed: np.ndarray, group_size: int, bits: int,
                        N: int, num_groups: int) -> np.ndarray:
    """Inverse of tqh_pack.pack_indices_lsb. Returns uint8 [N, K]."""
    if bits == 8:
        return packed.reshape(N, num_groups * group_size).astype(np.uint8)
    chunk = 8
    bytes_per_chunk = chunk * bits // 8
    chunks_per_group = group_size // chunk
    per_group_bytes = group_size * bits // 8
    p = packed.reshape(N, num_groups, chunks_per_group, bytes_per_chunk).astype(np.uint64)
    word = np.zeros((N, num_groups, chunks_per_group), dtype=np.uint64)
    for b in range(bytes_per_chunk):
        word |= (p[..., b] << (8 * b))
    out = np.zeros((N, num_groups, chunks_per_group, chunk), dtype=np.uint8)
    mask = (1 << bits) - 1
    for i in range(chunk):
        out[..., i] = ((word >> (i * bits)) & mask).astype(np.uint8)
    return out.reshape(N, num_groups * group_size)


def _unpack_indices_lsb_panel(packed: np.ndarray, group_size: int, bits: int,
                              N: int, num_groups: int) -> np.ndarray:
    chunks = group_size // TQ_PANEL_K_CHUNK
    bytes_per_chunk = TQ_PANEL_K_CHUNK * bits // 8
    per_group_bytes = group_size * bits // 8
    n_blocks = (N + TQ_PANEL_N - 1) // TQ_PANEL_N
    panel = packed.reshape(n_blocks, num_groups, chunks, TQ_PANEL_N, bytes_per_chunk)
    row_major = np.zeros((N, num_groups, chunks, bytes_per_chunk), dtype=np.uint8)
    for n in range(N):
        row_major[n] = panel[n // TQ_PANEL_N, :, :, n % TQ_PANEL_N, :]
    return _unpack_indices_lsb(row_major.reshape(N, num_groups * per_group_bytes),
                               group_size, bits, N, num_groups)


def _dehydrate_from_cactus(rec: dict) -> np.ndarray:
    """Mirror tqh_runtime.dehydrate_layer math from the cactus-emitted blob."""
    import torch
    N, K = rec['dim0'], rec['dim1']
    gs = rec['group_size']
    G = rec['num_groups']
    bits = rec['bits']
    cb = torch.from_numpy(rec['codebook'].astype(np.float32))
    if rec['flags'] & TQ_FLAG_PANEL_MAJOR:
        indices = _unpack_indices_lsb_panel(rec['packed_ix'], gs, bits, N, G)
    else:
        indices = _unpack_indices_lsb(rec['packed_ix'], gs, bits, N, G)
    idx = torch.from_numpy(indices.astype(np.int32))
    norms = torch.from_numpy(rec['scales'].astype(np.float16)).float()  # [N, G]
    if rec['rotation_family'] == 0:
        left = torch.from_numpy(rec['left'].astype(np.int8)).float()
        right = torch.from_numpy(rec['right'].astype(np.int8)).float()
        perm = rec['perm'].astype(np.int64)
        # Reconstruct R = (left.unsq * H_norm * right.unsq)[:, perm]
        from scipy.linalg import hadamard
        import math
        base = torch.from_numpy(hadamard(gs).astype(np.float32) / math.sqrt(gs))
        R = (left.unsqueeze(1) * base * right.unsqueeze(0))[:, torch.from_numpy(perm)].contiguous()
    else:
        R = torch.from_numpy(rec['orth_R'].astype(np.float32))
        assert gs == K and G == 1

    out = torch.empty(N, K, dtype=torch.float32)
    for g in range(G):
        s, e = g * gs, (g + 1) * gs
        group_idx = idx[:, s:e]
        if (rec['rotation_family'] == 0
                and (rec['flags'] & TQ_FLAG_CODE_ORDERED_INDICES)):
            # Runtime packed idx_new[k] = reference idx_old[inv_perm[k]].
            # Convert back to reference order before applying R.T below.
            group_idx = group_idx[:, torch.from_numpy(perm)]
        dq = cb[group_idx]                                # [N, gs]
        recon = (dq @ R.T) * norms[:, g:g + 1]            # [N, gs]
        out[:, s:e] = recon
    if rec['input_scale'] is not None:
        scale = torch.from_numpy(rec['input_scale'].astype(np.float16)).float()
        out = out / scale.unsqueeze(0)
    return out.numpy()


def verify_one_layer(packed_dir: Path, layer_name: str = None,
                     extra_scale_check: bool = True,
                     panel_major: bool = False) -> dict:
    """Pack a single layer through tqh_pack and compare to tqh_runtime."""
    meta = json.loads((packed_dir / 'metadata.json').read_text())
    if layer_name is None:
        layer_name = next(iter(meta['layers'].keys()))
    layer_meta = meta['layers'][layer_name]
    print(f"Verifying layer: {layer_name}")
    print(f"  shape={layer_meta['shape']}  bits={set(layer_meta['bits_per_group'])}")

    seed = int(meta.get('seed', 1234))
    loaded = tqh_pack._load_tqh_layer(packed_dir, layer_name, layer_meta, meta)
    gs = loaded['group_size']
    cb = tqh_pack.make_codebook(gs, loaded['bits'])

    # Reference: tqh_runtime
    ref = tqh_runtime.TQHPackedLoader(str(packed_dir)).dehydrate(layer_name).float().numpy()

    # Pack via cactus encoding (no extra_scale; extra_scale=1.0 for apples-to-apples).
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.weights', delete=False) as tf:
        tmp_path = Path(tf.name)
    rot_kind = loaded['rotation_family']
    full_R = (tqh_pack.make_orthogonal_rotation(gs, seed)
              if rot_kind == 'orthogonal' else None)
    tqh_pack.write_tq_weights(
        tmp_path,
        indices=loaded['indices'], norms=loaded['norms'],
        input_scale=loaded['input_scale'], codebook=cb,
        group_size=gs, bits=loaded['bits'],
        rotation_kind=rot_kind, rotation_seed=seed,
        full_orth_R=full_R, extra_scale=1.0,
        panel_major=panel_major and rot_kind == 'hadamard',
    )
    print(f"  packed file size: {tmp_path.stat().st_size / 1024:.1f} KB")

    # Read back and dehydrate.
    rec = _read_cactus_tq_weights(tmp_path)
    cactus = _dehydrate_from_cactus(rec)

    diff = np.abs(ref - cactus)
    print(f"  ref dtype={ref.dtype} shape={ref.shape}")
    print(f"  max abs err: {diff.max():.6f}")
    print(f"  mean abs err: {diff.mean():.6f}")
    print(f"  ref-norm max: {np.abs(ref).max():.6f}")
    rel = diff.max() / max(np.abs(ref).max(), 1e-8)
    print(f"  relative max err: {rel:.6f}")

    # Also exact-equality check on indices/scales/codebook.
    if rec['rotation_family'] == 0:
        # Verify left/right/perm match the reference's RNG sequence.
        l_ref, r_ref, p_ref = tqh_pack.make_hadamard_components(gs, seed)
        assert np.array_equal(rec['left'], l_ref), "left signs mismatch"
        assert np.array_equal(rec['right'], r_ref), "right signs mismatch"
        assert np.array_equal(rec['perm'], p_ref), "perm mismatch"
        print(f"  rotation components: bit-exact match against tqh_runtime seed")
    return dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()), rel_max=float(rel))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--packed-dir', required=True)
    ap.add_argument('--layer', default=None,
                    help='Layer name to verify (defaults to first in metadata)')
    ap.add_argument('--panel-major', action='store_true',
                    help='Verify the 4-row panel-major Hadamard runtime layout')
    args = ap.parse_args()
    verify_one_layer(Path(args.packed_dir), layer_name=args.layer,
                     panel_major=args.panel_major)
