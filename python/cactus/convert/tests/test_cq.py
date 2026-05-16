from __future__ import annotations

import struct

import numpy as np

from cactus.convert.cactus_adapters.tensor_io import (
    FLAG_HAS_SCALES,
    GROUP_SIZE,
    save_depthwise_conv_int8_with_header,
    save_pointwise_conv1d_int8_with_header,
    save_tensor_with_header,
)
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
