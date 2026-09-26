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
from cactus.convert.quantization.cq import (
    GROUP_SIZE as CQ_GROUP_SIZE,
    PRECISION_CQ,
    _gptq_correct_group,
    pack_indices_lsb,
    quantize_hadamard,
    quantize_orthogonal,
    write_cq_tensor,
)


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


def test_gptq_multiblock_compensation_matches_progressive_obs():
    rng = np.random.default_rng(12345)
    k = 384
    b = CQ_GROUP_SIZE
    assert k == 3 * b

    # Construct deterministic SPD correlated Hessian
    x = rng.standard_normal((k * 2, k)).astype(np.float32)
    h = (x.T @ x) / float(k * 2)
    h = h + np.eye(k, dtype=np.float32) * (0.01 * np.mean(np.diag(h)) + 1e-6)

    h_inv_global = np.linalg.inv(h)
    u_factor = np.linalg.cholesky(h_inv_global).T

    work_orig = rng.standard_normal((8, k)).astype(np.float32)
    recon_orig = rng.standard_normal((8, k)).astype(np.float32)
    tol = 1e-4

    for g in range(k // b - 1):
        start, stop = g * b, (g + 1) * b
        work = work_orig.copy()
        work_static = work_orig.copy()

        # 1. Independent ground-truth OBS reference from remaining sub-Hessian
        h_rem = h[start:, start:]
        h_rem_inv = np.linalg.inv(h_rem)
        ref_bb = h_rem_inv[0:b, 0:b]
        ref_bs = h_rem_inv[0:b, b:]
        ref_update = np.linalg.solve(ref_bb, ref_bs)

        residual = work_orig[:, start:stop] - recon_orig[:, start:stop]
        expected_work_rem = work_orig[:, stop:] - residual @ ref_update

        # 2. Production Cholesky implementation passes OBS invariant
        _gptq_correct_group(work, recon_orig[:, start:stop], u_factor, start, stop)
        actual_work_rem = work[:, stop:]
        actual_diff = float(np.max(np.abs(actual_work_rem - expected_work_rem)))
        assert actual_diff <= tol, f"Block {g} failed progressive OBS invariant: {actual_diff} > {tol}"

        # 3. Old static-inverse implementation violates the invariant on correlated blocks g >= 1
        old_m_bb = h_inv_global[start:stop, start:stop]
        old_m_bs = h_inv_global[start:stop, stop:]
        old_update = np.linalg.solve(old_m_bb + np.eye(b, dtype=np.float32) * 1e-6, old_m_bs)
        old_work_rem = work_static[:, stop:] - residual @ old_update
        old_diff = float(np.max(np.abs(old_work_rem - expected_work_rem)))

        if g == 0:
            assert old_diff <= tol, f"Block 0 unexpectedly diverged on static inverse: {old_diff} > {tol}"
        else:
            assert old_diff > tol, f"Old static inverse unexpectedly passed tolerance on block {g}: {old_diff} <= {tol}"


def test_quantize_hadamard_with_gptq_multiblock():
    rng = np.random.default_rng(2026)
    k = 384
    n = 16
    w = rng.standard_normal((n, k)).astype(np.float32)
    x = rng.standard_normal((k * 2, k)).astype(np.float32)
    h = (x.T @ x) / float(k * 2)

    cq_gptq = quantize_hadamard(w, bits=4, hessian=h, use_gptq=True)
    cq_rtn = quantize_hadamard(w, bits=4, hessian=h, use_gptq=False)

    assert cq_gptq.gptq_used is True
    assert cq_rtn.gptq_used is False
    assert cq_gptq.indices.shape == (n, k)
    assert cq_gptq.norms.shape == (n, k // CQ_GROUP_SIZE)
    # Block 0 has no prior compensation, so index decisions match RTN
    assert np.array_equal(cq_gptq.indices[:, :CQ_GROUP_SIZE], cq_rtn.indices[:, :CQ_GROUP_SIZE])
    # Multi-block compensation actively modifies subsequent quantization decisions
    assert not np.array_equal(cq_gptq.indices[:, CQ_GROUP_SIZE:], cq_rtn.indices[:, CQ_GROUP_SIZE:])
