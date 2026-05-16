from __future__ import annotations

import ctypes
import importlib
import sys
from typing import Any


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


def _load_runtime_modules():
    if "cactus.bindings.cactus" in sys.modules and "cactus.bindings.graph" in sys.modules:
        return sys.modules["cactus.bindings.cactus"], sys.modules["cactus.bindings.graph"]

    ctypes.CDLL.__getattr__ = _patched_cdll_getattr
    try:
        cactus_module = importlib.import_module("cactus.bindings.cactus")
        graph_module = importlib.import_module("cactus.bindings.graph")
    finally:
        ctypes.CDLL.__getattr__ = _ORIG_CDLL_GETATTR
    return cactus_module, graph_module


def _patch_graph_runtime(graph_module, cactus_module) -> None:
    Graph = graph_module.Graph
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

    def matmul(self, a, b, pretransposed_rhs=False, backend=Graph.CPU, output_dtype=None):
        a = _ensure_fp16_activation(self, a)
        b = _ensure_fp16_activation(self, b)
        out = cactus_node_t()
        rc = _lib.cactus_graph_matmul(
            self.h,
            cactus_node_t(a.id),
            cactus_node_t(b.id),
            ctypes.c_bool(bool(pretransposed_rhs)),
            ctypes.c_int32(int(backend)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_matmul failed"))
        result = self._tensor_from_node(out.value)
        if output_dtype is not None and int(output_dtype) != int(result.dtype):
            result = self.precision_cast(result, int(output_dtype))
        return result

    def gather(self, tensor, indices, axis=0):
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
        return self._tensor_from_node(out.value)

    def conv2d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
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
            )

        if (
            kernel_hw == (3, 3)
            and stride_hw == (2, 2)
            and padding_hw == (1, 1)
            and dilation_hw == (1, 1)
        ):
            if groups_int == 1:
                return self.conv2d_k3s2p1(x, weight, bias=bias)
            if len(weight.shape) >= 1 and groups_int == int(weight.shape[0]):
                return self.conv2d_depthwise_k3s2p1(x, weight, bias=bias)

        if (
            kernel_hw == (3, 3)
            and stride_hw == (1, 1)
            and padding_hw == (1, 1)
            and dilation_hw == (1, 1)
            and groups_int == 1
            and _has_symbol("cactus_graph_conv2d_k3s1p1")
        ):
            return self.conv2d_k3s1p1(x, weight, bias=bias)

        if (
            kernel_hw == (1, 1)
            and stride_hw == (1, 1)
            and padding_hw == (0, 0)
            and dilation_hw == (1, 1)
            and groups_int == 1
        ):
            return self.conv2d_pointwise_1x1(x, weight, bias=bias)

        raise NotImplementedError(
            "v2 runtime does not expose generic conv2d for this configuration: "
            f"kernel={kernel_hw} stride={stride_hw} padding={padding_hw} "
            f"dilation={dilation_hw} groups={groups_int}"
        )

    def scalar_not_equal(self, x, value):
        if _has_symbol("cactus_graph_scalar_not_equal"):
            return orig_scalar_not_equal(self, x, value)
        x = _ensure_compare_tensor(self, x)
        delta = self.scalar_add(x, -float(value))
        return _approx_nonzero_mask(self, delta)

    def not_equal(self, a, b):
        if _has_symbol("cactus_graph_not_equal"):
            return orig_not_equal(self, a, b)
        a = _ensure_compare_tensor(self, a)
        b = _ensure_compare_tensor(self, b)
        delta = self.subtract(a, b)
        return _approx_nonzero_mask(self, delta)

    def transpose(self, x, backend=Graph.CPU):
        return orig_transpose(self, _ensure_scalar_tensor(self, x), backend=backend)

    def permute(self, x, permutation, backend=Graph.CPU):
        return orig_permute(self, _ensure_scalar_tensor(self, x), permutation, backend=backend)

    def scalar_add(self, x, value):
        return orig_scalar_add(self, _ensure_scalar_tensor(self, x), value)

    def scalar_subtract(self, x, value):
        return orig_scalar_subtract(self, _ensure_scalar_tensor(self, x), value)

    def scalar_multiply(self, x, value):
        return orig_scalar_multiply(self, _ensure_scalar_tensor(self, x), value)

    def scalar_divide(self, x, value):
        return orig_scalar_divide(self, _ensure_scalar_tensor(self, x), value)

    def scalar_exp(self, x):
        return orig_scalar_exp(self, _ensure_scalar_tensor(self, x))

    def scalar_sqrt(self, x):
        return orig_scalar_sqrt(self, _ensure_scalar_tensor(self, x))

    def scalar_cos(self, x):
        return orig_scalar_cos(self, _ensure_scalar_tensor(self, x))

    def scalar_sin(self, x):
        return orig_scalar_sin(self, _ensure_scalar_tensor(self, x))

    def scalar_log(self, x):
        return orig_scalar_log(self, _ensure_scalar_tensor(self, x))

    def clamp(self, x, lo, hi):
        if orig_clamp is None:
            raise RuntimeError("Cactus runtime is missing required symbol: cactus_graph_clamp")
        return orig_clamp(self, _ensure_scalar_tensor(self, x), lo, hi)

    def add(self, a, b):
        return orig_add(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b))

    def subtract(self, a, b):
        return orig_subtract(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b))

    def multiply(self, a, b):
        return orig_multiply(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b))

    def divide(self, a, b):
        return orig_divide(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b))

    def concat(self, a, b, axis=0):
        return orig_concat(self, _ensure_scalar_tensor(self, a), _ensure_scalar_tensor(self, b), axis=axis)

    def cat(self, tensors, axis=0):
        legalized = [_ensure_scalar_tensor(self, tensor) for tensor in tensors]
        return orig_cat(self, legalized, axis=axis)

    def abs(self, x):
        return orig_abs(self, _ensure_scalar_tensor(self, x))

    def pow(self, x, exponent):
        return orig_pow(self, _ensure_scalar_tensor(self, x), exponent)

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
    Graph._transpile_runtime_compat_patched = True


_cactus_module, _graph_module = _load_runtime_modules()

if hasattr(_cactus_module, "_lib"):
    _lib_obj = _cactus_module._lib
    if hasattr(_lib_obj, "cactus_graph_matmul"):
        _lib_obj.cactus_graph_matmul.argtypes = [
            ctypes.c_void_p,
            _cactus_module.cactus_node_t,
            _cactus_module.cactus_node_t,
            ctypes.c_bool,
            ctypes.c_int32,
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

_patch_graph_runtime(_graph_module, _cactus_module)

_lib = _cactus_module._lib
_err = _cactus_module._err
cactus_node_t = _cactus_module.cactus_node_t
cactus_tensor_info_t = _cactus_module.cactus_tensor_info_t
Graph = _graph_module.Graph
Tensor = _graph_module.Tensor
