from __future__ import annotations

import ctypes
import importlib
import sys
from typing import Any

import numpy as np


class _MissingFFIFunction:
    _cactus_missing_symbol = True

    def __init__(self, name: str):
        self.__name__ = name
        self.argtypes = None
        self.restype = ctypes.c_int

    def __call__(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError(f"Cactus runtime is missing required symbol: {self.__name__}")


_ORIG_CDLL_GETATTR = ctypes.CDLL.__getattr__


def _patched_cdll_getattr(self: ctypes.CDLL, name: str):
    try:
        return _ORIG_CDLL_GETATTR(self, name)
    except AttributeError:
        if not name.startswith("cactus_"):
            raise
        missing = _MissingFFIFunction(name)
        setattr(self, name, missing)
        return missing


def _load_runtime_module():
    if "cactus.bindings.cactus" in sys.modules:
        return sys.modules["cactus.bindings.cactus"]

    ctypes.CDLL.__getattr__ = _patched_cdll_getattr
    try:
        return importlib.import_module("cactus.bindings.cactus")
    finally:
        ctypes.CDLL.__getattr__ = _ORIG_CDLL_GETATTR


def _patch_graph_runtime(cactus_module) -> None:
    Graph = cactus_module.Graph
    if getattr(Graph, "_transpile_runtime_compat_patched", False):
        return

    _lib = cactus_module._lib
    _err = cactus_module._err
    cactus_node_t = cactus_module.cactus_node_t

    def _has_symbol(name: str) -> bool:
        symbol = getattr(_lib, name, None)
        return symbol is not None and not getattr(symbol, "_cactus_missing_symbol", False)

    def _ensure_compare_tensor(self, tensor):
        tensor = self._ensure_tensor(tensor)
        if int(tensor.dtype) == int(Graph.FP16):
            return tensor
        return self.precision_cast(tensor, Graph.FP16)

    def _ensure_scalar_tensor(self, tensor):
        tensor = self._ensure_tensor(tensor)
        if int(tensor.dtype) == int(Graph.FP16):
            return tensor
        return self.precision_cast(tensor, Graph.FP16)

    def _ensure_fp16_activation(self, tensor):
        tensor = self._ensure_tensor(tensor)
        if int(tensor.dtype) == int(Graph.FP32):
            return self.precision_cast(tensor, Graph.FP16)
        return tensor

    def _approx_nonzero_mask(self, tensor):
        tensor = _ensure_scalar_tensor(self, tensor)
        # Gemma4 compare paths in v2 only need stable 0/1-style masks for
        # discrete values such as token-type ids and prebuilt boolean masks.
        magnitude = self.abs(tensor)
        shifted = self.scalar_add(magnitude, -0.5)
        sharpened = self.scalar_multiply(shifted, 16.0)
        return self.sigmoid(sharpened)

    orig_conv2d = Graph.conv2d
    orig_not_equal = Graph.not_equal
    orig_scalar_not_equal = Graph.scalar_not_equal
    orig_transpose = Graph.transpose
    orig_permute = Graph.permute
    orig_scalar_add = Graph.scalar_add
    orig_scalar_subtract = Graph.scalar_subtract
    orig_scalar_multiply = Graph.scalar_multiply
    orig_scalar_divide = Graph.scalar_divide
    orig_scalar_exp = Graph.scalar_exp
    orig_scalar_sqrt = Graph.scalar_sqrt
    orig_scalar_cos = Graph.scalar_cos
    orig_scalar_sin = Graph.scalar_sin
    orig_scalar_log = Graph.scalar_log
    orig_clamp = getattr(Graph, "clamp", None)
    orig_add = Graph.add
    orig_subtract = Graph.subtract
    orig_multiply = Graph.multiply
    orig_divide = Graph.divide
    orig_concat = Graph.concat
    orig_cat = Graph.cat
    orig_abs = Graph.abs
    orig_pow = Graph.pow
    orig_expand = Graph.expand

    def matmul(self, a, b, pretransposed_rhs=False, backend=None, output_dtype=None):
        a = _ensure_fp16_activation(self, a)
        b = _ensure_fp16_activation(self, b)
        out = cactus_node_t()
        rc = _lib.cactus_graph_matmul(
            self.h,
            cactus_node_t(a.id),
            cactus_node_t(b.id),
            ctypes.c_bool(bool(pretransposed_rhs)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_matmul failed"))
        result = self._apply_backend(self._tensor_from_node(out.value), backend)
        if output_dtype is not None and int(output_dtype) != int(result.dtype):
            result = self.precision_cast(result, int(output_dtype))
        return result

    def gather(self, tensor, indices, axis=0, backend=None):
        tensor = self._ensure_tensor(tensor)
        indices = self._ensure_tensor(indices)
        if int(axis) != 0:
            raise NotImplementedError(
                f"transpiler runtime compatibility only supports gather(axis=0), got axis={axis}"
            )
        out = cactus_node_t()
        rc = _lib.cactus_graph_gather(
            self.h,
            cactus_node_t(tensor.id),
            cactus_node_t(indices.id),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_gather failed")
        return self._apply_backend(self._tensor_from_node(out.value), backend)

    def conv2d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, backend=None):
        def _pair(value: Any) -> tuple[int, int]:
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise ValueError(f"expected pair for conv2d parameter, got {value!r}")
                return int(value[0]), int(value[1])
            return int(value), int(value)

        stride_hw = _pair(stride)
        padding_hw = _pair(padding)
        dilation_hw = _pair(dilation)
        groups_int = int(groups)
        kernel_hw = tuple(int(dim) for dim in self._ensure_tensor(weight).shape[-2:])

        if _has_symbol("cactus_graph_conv2d"):
            return orig_conv2d(
                self,
                x,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                backend=backend,
            )

        if (
            kernel_hw == (3, 3)
            and stride_hw == (2, 2)
            and padding_hw == (1, 1)
            and dilation_hw == (1, 1)
        ):
            if groups_int == 1:
                return self.conv2d_k3s2p1(x, weight, bias=bias, backend=backend)
            if len(weight.shape) >= 1 and groups_int == int(weight.shape[0]):
                return self.conv2d_depthwise_k3s2p1(x, weight, bias=bias, backend=backend)

        if (
            kernel_hw == (3, 3)
            and stride_hw == (1, 1)
            and padding_hw == (1, 1)
            and dilation_hw == (1, 1)
            and groups_int == 1
            and _has_symbol("cactus_graph_conv2d_k3s1p1")
        ):
            return self.conv2d_k3s1p1(x, weight, bias=bias, backend=backend)

        if (
            kernel_hw == (1, 1)
            and stride_hw == (1, 1)
            and padding_hw == (0, 0)
            and dilation_hw == (1, 1)
            and groups_int == 1
        ):
            return self.conv2d_pointwise_1x1(x, weight, bias=bias, backend=backend)

        raise NotImplementedError(
            "v2 runtime does not expose generic conv2d for this configuration: "
            f"kernel={kernel_hw} stride={stride_hw} padding={padding_hw} "
            f"dilation={dilation_hw} groups={groups_int}"
        )

    def scalar_not_equal(self, x, value, backend=None):
        if _has_symbol("cactus_graph_scalar_not_equal"):
            return orig_scalar_not_equal(self, x, value, backend=backend)
        x = _ensure_compare_tensor(self, x)
        delta = self.scalar_add(x, -float(value))
        return _approx_nonzero_mask(self, delta)

    def not_equal(self, a, b, backend=None):
        if _has_symbol("cactus_graph_not_equal"):
            return orig_not_equal(self, a, b, backend=backend)
        a = _ensure_compare_tensor(self, a)
        b = _ensure_compare_tensor(self, b)
        delta = self.subtract(a, b)
        return _approx_nonzero_mask(self, delta)

    def transpose(self, x, backend=None):
        return orig_transpose(self, _ensure_scalar_tensor(self, x), backend=backend)

    def permute(self, x, permutation, backend=None):
        return orig_permute(self, _ensure_scalar_tensor(self, x), permutation, backend=backend)

    def scalar_add(self, x, value, backend=None):
        return orig_scalar_add(self, self._ensure_tensor(x), value, backend=backend)

    def scalar_subtract(self, x, value, backend=None):
        return orig_scalar_subtract(self, self._ensure_tensor(x), value, backend=backend)

    def scalar_multiply(self, x, value, backend=None):
        return orig_scalar_multiply(self, self._ensure_tensor(x), value, backend=backend)

    def scalar_divide(self, x, value, backend=None):
        return orig_scalar_divide(self, self._ensure_tensor(x), value, backend=backend)

    def scalar_exp(self, x, backend=None):
        return orig_scalar_exp(self, self._ensure_tensor(x), backend=backend)

    def scalar_sqrt(self, x, backend=None):
        return orig_scalar_sqrt(self, self._ensure_tensor(x), backend=backend)

    def scalar_cos(self, x, backend=None):
        return orig_scalar_cos(self, self._ensure_tensor(x), backend=backend)

    def scalar_sin(self, x, backend=None):
        return orig_scalar_sin(self, self._ensure_tensor(x), backend=backend)

    def scalar_log(self, x, backend=None):
        return orig_scalar_log(self, self._ensure_tensor(x), backend=backend)

    def clamp(self, x, lo, hi, backend=None):
        if orig_clamp is None:
            raise RuntimeError("Cactus runtime is missing required symbol: cactus_graph_clamp")
        return orig_clamp(self, _ensure_scalar_tensor(self, x), lo, hi, backend=backend)

    def add(self, a, b, backend=None):
        return orig_add(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b), backend=backend)

    def subtract(self, a, b, backend=None):
        return orig_subtract(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b), backend=backend)

    def multiply(self, a, b, backend=None):
        return orig_multiply(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b), backend=backend)

    def divide(self, a, b, backend=None):
        return orig_divide(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b), backend=backend)

    def concat(self, a, b, axis=0, backend=None):
        return orig_concat(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b), axis=axis, backend=backend)

    def cat(self, tensors, axis=0, backend=None):
        legalized = [_ensure_scalar_tensor(self, tensor) for tensor in tensors]
        return orig_cat(self, legalized, axis=axis, backend=backend)

    def abs(self, x, backend=None):
        return orig_abs(self, self._ensure_tensor(x), backend=backend)

    def pow(self, x, exponent, backend=None):
        return orig_pow(self, self._ensure_tensor(x), exponent, backend=backend)

    def expand(self, x, shape, backend=None):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        if tuple(int(dim) for dim in x.shape) == shape:
            return x
        if _has_symbol("cactus_graph_expand"):
            return orig_expand(self, x, shape, backend=backend)

        # v2 builds do not always expose the generic expand symbol. For
        # broadcast-only expands, adding an embedded zero tensor with the target
        # shape gives the same logical result while staying inside existing
        # Cactus graph ops.
        dtype_map = {
            int(Graph.FP16): np.float16,
            int(Graph.FP32): np.float32,
        }
        x_dtype = int(x.dtype)
        if x_dtype not in dtype_map:
            raise NotImplementedError(
                "expand fallback only supports floating-point tensors; "
                f"got dtype={x_dtype} shape={tuple(int(dim) for dim in x.shape)} -> {shape}"
            )
        zero = self.input(shape, dtype=x_dtype)
        self.set_input(zero, np.zeros(shape, dtype=dtype_map[x_dtype]), dtype=x_dtype)
        self.mark_embedded_input(zero)
        return self.add(x, zero, backend=backend)

    Graph.matmul = matmul
    Graph.gather = gather
    Graph.conv2d = conv2d
    Graph.scalar_not_equal = scalar_not_equal
    Graph.not_equal = not_equal
    Graph.transpose = transpose
    Graph.permute = permute
    Graph.scalar_add = scalar_add
    Graph.scalar_subtract = scalar_subtract
    Graph.scalar_multiply = scalar_multiply
    Graph.scalar_divide = scalar_divide
    Graph.scalar_exp = scalar_exp
    Graph.scalar_sqrt = scalar_sqrt
    Graph.scalar_cos = scalar_cos
    Graph.scalar_sin = scalar_sin
    Graph.scalar_log = scalar_log
    Graph.clamp = clamp
    Graph.add = add
    Graph.subtract = subtract
    Graph.multiply = multiply
    Graph.divide = divide
    Graph.concat = concat
    Graph.cat = cat
    Graph.abs = abs
    Graph.pow = pow
    Graph.expand = expand
    Graph._transpile_runtime_compat_patched = True


_cactus_module = _load_runtime_module()

if hasattr(_cactus_module, "_lib"):
    _lib_obj = _cactus_module._lib
    if hasattr(_lib_obj, "cactus_graph_matmul"):
        _lib_obj.cactus_graph_matmul.argtypes = [
            ctypes.c_void_p,
            _cactus_module.cactus_node_t,
            _cactus_module.cactus_node_t,
            ctypes.c_bool,
            ctypes.POINTER(_cactus_module.cactus_node_t),
        ]
        _lib_obj.cactus_graph_matmul.restype = ctypes.c_int
    if hasattr(_lib_obj, "cactus_graph_gather"):
        _lib_obj.cactus_graph_gather.argtypes = [
            ctypes.c_void_p,
            _cactus_module.cactus_node_t,
            _cactus_module.cactus_node_t,
            ctypes.POINTER(_cactus_module.cactus_node_t),
        ]
        _lib_obj.cactus_graph_gather.restype = ctypes.c_int

_patch_graph_runtime(_cactus_module)

_lib = _cactus_module._lib
_err = _cactus_module._err
cactus_node_t = _cactus_module.cactus_node_t
cactus_tensor_info_t = _cactus_module.cactus_tensor_info_t
Graph = _cactus_module.Graph
Tensor = _cactus_module.Tensor
