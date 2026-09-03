from __future__ import annotations

from dataclasses import replace
import struct

import numpy as np
import torch

from cactus.convert.cactus_adapters.tensor_io import (
    FLAG_HAS_SCALES,
    GROUP_SIZE,
    save_depthwise_conv_int8_with_header,
    save_pointwise_conv1d_int8_with_header,
    save_tensor_with_header,
)
from cactus.convert.export.qdq import FLAG_INTERLEAVED_4ROW, FLAG_ORTHOGONAL_ROTATION, dequantize_cq_file, read_header
from cactus.convert.interleave_orthogonal_cq4 import interleave_orthogonal_cq4_file
from cactus.convert.quantization.cq import PRECISION_CQ, pack_indices_lsb, quantize_hadamard, quantize_orthogonal, write_cq_tensor


def test_pack_indices_lsb_bits():
    idx = np.arange(128, dtype=np.uint8).reshape(1, 128)
    for bits in [1, 2, 3, 4]:
        packed = pack_indices_lsb(idx % (1 << bits), 128, bits)
        assert packed.size == 128 * bits // 8


def test_cq_header_roundtrip(tmp_path):
    w = np.random.default_rng(0).standard_normal((3, 128), dtype=np.float32)
    cq = quantize_hadamard(w, bits=3)
    out = tmp_path / "x.weights"
    write_cq_tensor(out, cq)
    data = out.read_bytes()[:84]
    magic, flags, alignment, ndim = struct.unpack_from("<4sIII", data, 0)
    dims = struct.unpack_from("<QQQQ", data, 16)
    precision = struct.unpack_from("<I", data, 48)[0]
    data_bytes = struct.unpack_from("<Q", data, 52)[0]
    scales_bytes = struct.unpack_from("<Q", data, 60)[0]
    assert magic == b"CACT"
    assert flags == 0
    assert alignment == 32
    assert ndim == 2
    assert dims[:2] == (3, 128)
    assert precision == PRECISION_CQ[3]
    assert data_bytes > 0
    assert scales_bytes > 0


def test_orthogonal_embedding_is_cq4(tmp_path):
    w = np.random.default_rng(1).standard_normal((4, 16), dtype=np.float32)
    cq = quantize_orthogonal(w, bits=4)
    out = tmp_path / "embed.weights"
    write_cq_tensor(out, cq)
    precision = struct.unpack_from("<I", out.read_bytes(), 48)[0]
    assert precision == PRECISION_CQ[4]
    assert cq.rotation_family == "orthogonal"


def test_orthogonal_interleaved_cq4_qdq_matches_row_major(tmp_path):
    w = np.random.default_rng(2).standard_normal((8, 32), dtype=np.float32)
    cq = quantize_orthogonal(w, bits=4)
    row_path = tmp_path / "row.weights"
    inter_path = tmp_path / "inter.weights"
    write_cq_tensor(row_path, cq)
    write_cq_tensor(inter_path, replace(cq, interleaved_4row=True))

    inter_header = read_header(inter_path)
    assert inter_header.flags & FLAG_ORTHOGONAL_ROTATION
    assert inter_header.flags & FLAG_INTERLEAVED_4ROW

    row = dequantize_cq_file(row_path, read_header(row_path), torch.float32, 4)
    inter = dequantize_cq_file(inter_path, inter_header, torch.float32, 4)
    assert torch.max(torch.abs(row - inter)).item() <= 1e-6


def test_interleave_orthogonal_cq4_file_preserves_qdq(tmp_path):
    w = np.random.default_rng(3).standard_normal((8, 32), dtype=np.float32)
    src = tmp_path / "src.weights"
    dst = tmp_path / "dst.weights"
    write_cq_tensor(src, quantize_orthogonal(w, bits=4))
    interleave_orthogonal_cq4_file(src, dst)

    src_tensor = dequantize_cq_file(src, read_header(src), torch.float32, 4)
    dst_tensor = dequantize_cq_file(dst, read_header(dst), torch.float32, 4)
    assert torch.max(torch.abs(src_tensor - dst_tensor)).item() <= 1e-6


def test_int8_bias_uses_cactus_grouped_layout(tmp_path):
    bias = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    out = tmp_path / "bias.weights"
    save_tensor_with_header(bias, out, precision="INT8", allow_int8_bias=True)
    raw = out.read_bytes()
    magic, flags, alignment, ndim = struct.unpack_from("<4sIII", raw, 0)
    dims = struct.unpack_from("<QQQQ", raw, 16)
    precision = struct.unpack_from("<I", raw, 48)[0]
    data_bytes = struct.unpack_from("<Q", raw, 52)[0]
    scales_bytes = struct.unpack_from("<Q", raw, 60)[0]
    group_size = struct.unpack_from("<I", raw, 68)[0]
    num_groups = struct.unpack_from("<I", raw, 72)[0]
    assert magic == b"CACT"
    assert flags & FLAG_HAS_SCALES
    assert alignment == 32
    assert ndim == 1
    assert dims[0] == GROUP_SIZE
    assert precision == 0
    assert data_bytes == GROUP_SIZE
    assert scales_bytes == 2
    assert group_size == GROUP_SIZE
    assert num_groups == 1


def test_depthwise_conv_int8_preserves_kernel_shape(tmp_path):
    weight = np.array([[[1.0, -2.0, 0.5]], [[0.25, 0.0, -0.75]]], dtype=np.float32)
    out = tmp_path / "layer_0_conv_depthwise.weights"
    save_depthwise_conv_int8_with_header(weight, out)
    raw = out.read_bytes()
    magic, flags, alignment, ndim = struct.unpack_from("<4sIII", raw, 0)
    dims = struct.unpack_from("<QQQQ", raw, 16)
    precision = struct.unpack_from("<I", raw, 48)[0]
    data_bytes = struct.unpack_from("<Q", raw, 52)[0]
    scales_bytes = struct.unpack_from("<Q", raw, 60)[0]
    group_size = struct.unpack_from("<I", raw, 68)[0]
    num_groups = struct.unpack_from("<I", raw, 72)[0]
    assert magic == b"CACT"
    assert flags & FLAG_HAS_SCALES
    assert alignment == 32
    assert ndim == 3
    assert dims[:3] == (2, 1, 3)
    assert precision == 0
    assert data_bytes == 6
    assert scales_bytes == 4
    assert group_size == 3
    assert num_groups == 1


def test_pointwise_conv1d_int8_preserves_rank3_shape(tmp_path):
    weight = np.random.default_rng(0).standard_normal((3, GROUP_SIZE * 2, 1), dtype=np.float32)
    out = tmp_path / "layer_0_conv_pointwise1.weights"
    save_pointwise_conv1d_int8_with_header(weight, out)
    raw = out.read_bytes()
    magic, flags, alignment, ndim = struct.unpack_from("<4sIII", raw, 0)
    dims = struct.unpack_from("<QQQQ", raw, 16)
    precision = struct.unpack_from("<I", raw, 48)[0]
    data_bytes = struct.unpack_from("<Q", raw, 52)[0]
    scales_bytes = struct.unpack_from("<Q", raw, 60)[0]
    group_size = struct.unpack_from("<I", raw, 68)[0]
    num_groups = struct.unpack_from("<I", raw, 72)[0]
    assert magic == b"CACT"
    assert flags & FLAG_HAS_SCALES
    assert alignment == 32
    assert ndim == 3
    assert dims[:3] == (3, GROUP_SIZE * 2, 1)
    assert precision == 0
    assert data_bytes == 3 * GROUP_SIZE * 2
    assert scales_bytes == 3 * 2 * 2
    assert group_size == GROUP_SIZE
    assert num_groups == 2


def _correlated_inputs(rng: np.random.Generator, samples: int, k: int, rank: int = 32) -> np.ndarray:
    latent = rng.standard_normal((samples, rank), dtype=np.float32)
    mixing = rng.standard_normal((rank, k), dtype=np.float32)
    noise = 0.1 * rng.standard_normal((samples, k), dtype=np.float32)
    return (latent @ mixing + noise).astype(np.float32)


def _damp_like_cq(h: np.ndarray) -> np.ndarray:
    return h + np.eye(h.shape[0], dtype=np.float32) * (0.01 * np.mean(np.diag(h)) + 1e-6)


def test_gptq_cholesky_block_update_matches_explicit_obs():
    from cactus.convert.quantization import cq as cq_mod

    rng = np.random.default_rng(10)
    k = 3 * cq_mod.GROUP_SIZE
    x = _correlated_inputs(rng, 2000, k)
    h = _damp_like_cq((x.T @ x / x.shape[0]).astype(np.float32))
    h64 = h.astype(np.float64)

    build_factor = getattr(cq_mod, "_gptq_cholesky_factor", None)
    factor = build_factor(h) if build_factor is not None else np.linalg.cholesky(np.linalg.inv(h64)).T.astype(np.float32)

    work = rng.standard_normal((8, k), dtype=np.float32)
    work_ref = work.astype(np.float64)
    for g in range(k // cq_mod.GROUP_SIZE):
        start, stop = g * cq_mod.GROUP_SIZE, (g + 1) * cq_mod.GROUP_SIZE
        recon = np.round(work[:, start:stop], 1).astype(np.float32)
        cq_mod._gptq_correct_group(work, recon, factor, start, stop)
        if stop < k:
            # Reference: the inverse Hessian restricted to the still-unquantized columns,
            # recomputed from scratch each step (equal to the Schur-updated OBS inverse).
            hinv_cur = np.linalg.inv(h64[start:, start:])
            gsz = stop - start
            update_ref = np.linalg.solve(hinv_cur[:gsz, :gsz], hinv_cur[:gsz, gsz:])
            err = work_ref[:, start:stop] - recon.astype(np.float64)
            work_ref[:, stop:] -= err @ update_ref
        np.testing.assert_allclose(work[:, stop:], work_ref[:, stop:], rtol=1e-4, atol=1e-4)
        work_ref[:, start:stop] = work[:, start:stop]


def test_gptq_reduces_calibrated_error(tmp_path):
    rng = np.random.default_rng(11)
    n, k = 16, 384
    w = rng.standard_normal((n, k), dtype=np.float32)
    x = _correlated_inputs(rng, 4096, k)
    h = (x.T @ x / x.shape[0]).astype(np.float32)

    plain = quantize_hadamard(w, bits=4)
    gptq = quantize_hadamard(w, bits=4, hessian=h, use_gptq=True)
    assert not plain.gptq_used
    assert gptq.gptq_used

    def reconstruct(cq, name):
        path = tmp_path / name
        write_cq_tensor(path, cq)
        return dequantize_cq_file(path, read_header(path), torch.float32, 4).numpy()

    w_plain = reconstruct(plain, "plain.weights")
    w_gptq = reconstruct(gptq, "gptq.weights")
    err_plain = np.linalg.norm(x @ (w - w_plain).T)
    err_gptq = np.linalg.norm(x @ (w - w_gptq).T)
    assert np.isfinite(err_gptq)
    assert err_gptq < err_plain, (err_gptq, err_plain)


def test_gptq_disabled_on_bad_hessian():
    rng = np.random.default_rng(12)
    w = rng.standard_normal((4, 256), dtype=np.float32)
    baseline = quantize_hadamard(w, bits=4)

    negative = quantize_hadamard(w, bits=4, hessian=-np.eye(256, dtype=np.float32) * 1e6, use_gptq=True)
    assert not negative.gptq_used
    np.testing.assert_array_equal(negative.indices, baseline.indices)


def test_gptq_disabled_on_nan_hessian():
    rng = np.random.default_rng(12)
    w = rng.standard_normal((4, 256), dtype=np.float32)
    baseline = quantize_hadamard(w, bits=4)
    nan_h = np.full((256, 256), np.nan, dtype=np.float32)
    broken = quantize_hadamard(w, bits=4, hessian=nan_h, use_gptq=True)
    np.testing.assert_array_equal(broken.indices, baseline.indices)
    assert not broken.gptq_used
