"""Equivalence of the transpiler's packed-panel writer with the engine's reference encoder.

The runtime kernels consume the panel bytes the transpiler writes; the engine ships
`cactus_quant_build_panels` as the byte-exact reference encoder (proven layout-invariant and
kernel-correct by the C++ matmul suite). These tests load that symbol via ctypes and assert the
transpiler's pure-numpy packer produces byte-identical panels and folded norms — so a passing
C++ kernel test transitively covers the transpiler output.
"""
from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import numpy as np
import pytest

from cactus.convert.quantization.cq import (
    canonicalize_codebook_indices,
    fold_panel_norms,
    make_codebook,
    pack_indices_lsb,
    pack_indices_panels,
    quantize_codebook_i8,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DYLIB_CANDIDATES = [
    _REPO_ROOT / "cactus-engine" / "build" / "libcactus_engine.dylib",
    _REPO_ROOT / "cactus-engine" / "build" / "libcactus_engine.so",
]
_BUILD_PANELS_SYMBOL = "_Z25cactus_quant_build_panelsPK17CactusQuantMatrixPhPf"


class _CactusQuantMatrix(ctypes.Structure):
    _fields_ = [
        ("bits", ctypes.c_uint32),
        ("K", ctypes.c_uint32),
        ("N", ctypes.c_uint32),
        ("group_size", ctypes.c_uint32),
        ("num_groups", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("codebook", ctypes.c_void_p),
        ("input_scale", ctypes.c_void_p),
        ("input_scale_recip", ctypes.c_void_p),
        ("norms", ctypes.c_void_p),
        ("packed_indices", ctypes.c_void_p),
        ("left_signs", ctypes.c_void_p),
        ("right_signs", ctypes.c_void_p),
        ("permutation", ctypes.c_void_p),
        ("rotation", ctypes.c_void_p),
        ("rotation_t", ctypes.c_void_p),
        ("expanded", ctypes.c_void_p),
        ("norm_f32", ctypes.c_void_p),
        ("packed_panels", ctypes.c_void_p),
        ("norm_panels", ctypes.c_void_p),
    ]


def _load_reference_encoder():
    for cand in _DYLIB_CANDIDATES:
        if not cand.exists():
            continue
        lib = ctypes.CDLL(str(cand))
        try:
            fn = getattr(lib, _BUILD_PANELS_SYMBOL)
        except AttributeError:
            continue
        fn.restype = None
        fn.argtypes = [ctypes.POINTER(_CactusQuantMatrix), ctypes.c_void_p, ctypes.c_void_p]
        return fn
    return None


_REFERENCE_ENCODER = _load_reference_encoder()
_skip = pytest.mark.skipif(
    _REFERENCE_ENCODER is None,
    reason="cactus-engine dylib (with cactus_quant_build_panels) not built; run cactus-engine/build.sh",
)


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(arr.ctypes.data)


def _reference_panels(bits, k, n, group_size, flags, codebook_f16, norms_f16, packed_indices):
    num_groups = k // group_size
    nkg = group_size // 4
    sb64 = (n + 63) // 64
    panels = np.zeros(sb64 * num_groups * nkg * 128, dtype=np.uint8)
    norms = np.zeros(sb64 * num_groups * 64, dtype=np.float32)
    w = _CactusQuantMatrix(
        bits=bits, K=k, N=n, group_size=group_size, num_groups=num_groups, flags=flags,
        codebook=_ptr(codebook_f16), input_scale=None, input_scale_recip=None,
        norms=_ptr(norms_f16), packed_indices=_ptr(packed_indices),
        left_signs=None, right_signs=None, permutation=None, rotation=None, rotation_t=None,
        expanded=None, norm_f32=None, packed_panels=None, norm_panels=None,
    )
    _REFERENCE_ENCODER(ctypes.byref(w), _ptr(panels), _ptr(norms))
    return panels, norms


def _python_panels(k, n, group_size, codebook_f16, norms_f16, raw_indices):
    cb_i8, cb_scale = quantize_codebook_i8(codebook_f16)
    canonical = canonicalize_codebook_indices(raw_indices, cb_i8)
    panels = pack_indices_panels(canonical, group_size)
    num_groups = k // group_size
    panel_norms = fold_panel_norms(norms_f16.reshape(n, num_groups), cb_scale)
    return panels, panel_norms.reshape(-1)


@_skip
@pytest.mark.parametrize(
    "n,k,group_size",
    [
        (64, 256, 128),    # one super-block, two groups
        (192, 512, 128),   # three super-blocks
        (100, 256, 128),   # ragged N (padded super-block)
        (256, 256, 256),   # single group spanning K
    ],
)
def test_transpiler_panels_match_reference_encoder(n, k, group_size):
    rng = np.random.default_rng(20240611 + n + k)
    bits = 4
    codebook = make_codebook(group_size, bits).astype(np.float16)
    raw = rng.integers(0, 1 << bits, size=(n, k), dtype=np.uint8)
    norms = (rng.standard_normal((n, k // group_size)) * 0.1).astype(np.float16)

    packed_lsb = np.ascontiguousarray(pack_indices_lsb(raw, group_size, bits))
    ref_panels, ref_norms = _reference_panels(
        bits, k, n, group_size, 0, codebook, np.ascontiguousarray(norms.reshape(-1)), packed_lsb
    )
    py_panels, py_norms = _python_panels(k, n, group_size, codebook, norms, raw)

    assert np.array_equal(py_panels, ref_panels)
    assert np.array_equal(py_norms, ref_norms)
