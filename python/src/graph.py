import ctypes
from os import wait
import numpy as np

from .cactus import _lib, cactus_node_t, cactus_tensor_info_t


class Graph:
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT4 = 3

    def __init__(self):
        self.h = _lib.cactus_graph_create()
        if not self.h:
            raise RuntimeError("cactus_graph_create failed")

    def __del__(self):
        h = getattr(self, "h", None)
        if h:
            _lib.cactus_graph_destroy(h)
            self.h = None

    def input(self, shape, dtype=FP16):
        shape = tuple(int(x) for x in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_input(self.h, arr, len(shape), int(dtype), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_input failed")
        return self._tensor_from_node(out.value)

    def set_input(self, tensor, data, dtype=None):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if tensor.g is not self:
            raise ValueError("tensor belongs to a different graph")
        target_dtype = int(tensor.dtype if dtype is None else dtype)
        arr = self._coerce_input_array(data, target_dtype)
        rc = _lib.cactus_graph_set_input(
            self.h,
            cactus_node_t(tensor.id),
            arr.ctypes.data_as(ctypes.c_void_p),
            target_dtype,
        )
        if rc != 0:
            raise RuntimeError("graph_set_input failed")

    def hard_reset(self):
        rc = _lib.cactus_graph_hard_reset(self.h)
        if rc != 0:
            raise RuntimeError("graph_hard_reset failed")

    def execute(self):
        rc = _lib.cactus_graph_execute(self.h)
        if rc != 0:
            raise RuntimeError("graph_execute failed")

    def add(self, a, b):
        return self._binary("cactus_graph_add", a, b)

    def subtract(self, a, b):
        return self._binary("cactus_graph_subtract", a, b)

    def multiply(self, a, b):
        return self._binary("cactus_graph_multiply", a, b)

    def divide(self, a, b):
        return self._binary("cactus_graph_divide", a, b)

    def abs(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_abs(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_abs failed")
        return self._tensor_from_node(out.value)

    def pow(self, x, exponent):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_pow(self.h, cactus_node_t(x.id), ctypes.c_float(float(exponent)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_pow failed")
        return self._tensor_from_node(out.value)

    def view(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_view(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_view failed")
        return self._tensor_from_node(out.value)

    def flatten(self, x, start_dim=0, end_dim=-1):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_flatten(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(start_dim)),
            ctypes.c_int32(int(end_dim)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_flatten failed")
        return self._tensor_from_node(out.value)

    def concat(self, a, b, axis=0):
        a = self._ensure_tensor(a)
        b = self._ensure_tensor(b)
        out = cactus_node_t()
        rc = _lib.cactus_graph_concat(
            self.h,
            cactus_node_t(a.id),
            cactus_node_t(b.id),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_concat failed")
        return self._tensor_from_node(out.value)

    def cat(self, tensors, axis=0):
        tensors = [self._ensure_tensor(t) for t in tensors]
        if not tensors:
            raise ValueError("cat requires at least one tensor")
        ids = (cactus_node_t * len(tensors))(*(cactus_node_t(t.id) for t in tensors))
        out = cactus_node_t()
        rc = _lib.cactus_graph_cat(
            self.h,
            ids,
            ctypes.c_size_t(len(tensors)),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_cat failed")
        return self._tensor_from_node(out.value)

    def group_norm(self, x, normalized_shape, eps=1e-5):
        x = self._ensure_tensor(x)
        normalized_shape = tuple(int(v) for v in normalized_shape)
        shape_arr = (ctypes.c_size_t * len(normalized_shape))(*normalized_shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_group_norm(
            self.h,
            cactus_node_t(x.id),
            shape_arr,
            len(normalized_shape),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_group_norm failed")
        return self._tensor_from_node(out.value)

    def layer_norm(self, x, normalized_shape, eps=1e-5):
        x = self._ensure_tensor(x)
        normalized_shape = tuple(int(v) for v in normalized_shape)
        shape_arr = (ctypes.c_size_t * len(normalized_shape))(*normalized_shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_layer_norm(
            self.h,
            cactus_node_t(x.id),
            shape_arr,
            len(normalized_shape),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_layer_norm failed")
        return self._tensor_from_node(out.value)
    
    def softmax(self, x, axis=-1):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_softmax(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_softmax failed")
        return self._tensor_from_node(out.value)
    
    def relu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_relu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_relu failed")
        return self._tensor_from_node(out.value)

    def gelu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gelu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_gelu failed")
        return self._tensor_from_node(out.value)

    def sigmoid(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_sigmoid(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_sigmoid failed")
        return self._tensor_from_node(out.value)

    def tanh(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_tanh(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_tanh failed")
        return self._tensor_from_node(out.value)

    def output_info(self, x):
        x = self._ensure_tensor(x)
        return self._get_output_info(x.id)

    def _binary(self, fn_name, a, b):
        a = self._ensure_tensor(a)
        b = self._ensure_tensor(b)
        out = cactus_node_t()
        rc = getattr(_lib, fn_name)(self.h, cactus_node_t(a.id), cactus_node_t(b.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed")
        return self._tensor_from_node(out.value)

    def _ensure_tensor(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("expected Tensor")
        if x.g is not self:
            raise ValueError("tensor belongs to a different graph")
        return x

    def _get_output_info(self, node_id):
        info = cactus_tensor_info_t()
        rc = _lib.cactus_graph_get_output_info(self.h, cactus_node_t(node_id), ctypes.byref(info))
        if rc != 0:
            raise RuntimeError("graph_get_output_info failed")
        shape = tuple(int(info.shape[i]) for i in range(int(info.rank)))
        return {
            "precision": int(info.precision),
            "rank": int(info.rank),
            "shape": shape,
            "num_elements": int(info.num_elements),
            "byte_size": int(info.byte_size),
        }

    def _tensor_from_node(self, node_id):
        meta = self._get_output_info(node_id)
        return Tensor(self, int(node_id), meta["shape"], meta["precision"])

    def _coerce_input_array(self, data, precision):
        if isinstance(data, Tensor):
            arr = data.numpy()
        else:
            arr = np.asarray(data)
        if precision == self.INT8:
            arr = np.ascontiguousarray(arr, dtype=np.int8)
        elif precision == self.FP16:
            arr = np.ascontiguousarray(arr, dtype=np.float16)
        elif precision == self.FP32:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        elif precision == self.INT4:
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
        else:
            raise RuntimeError("unsupported precision")
        return arr


class Tensor:
    def __init__(self, g, node_id, shape, dtype):
        self.g = g
        self.id = int(node_id)
        self.shape = tuple(shape)
        self.dtype = int(dtype)

    def __add__(self, other):
        return self.g.add(self, other)

    def __sub__(self, other):
        return self.g.subtract(self, other)

    def __mul__(self, other):
        return self.g.multiply(self, other)

    def __truediv__(self, other):
        return self.g.divide(self, other)

    def abs(self):
        return self.g.abs(self)

    def pow(self, exponent):
        return self.g.pow(self, exponent)

    def relu(self):
        return self.g.relu(self)

    def sigmoid(self):
        return self.g.sigmoid(self)

    def tanh(self):
        return self.g.tanh(self)

    def gelu(self):
        return self.g.gelu(self)

    def view(self, shape):
        return self.g.view(self, shape)

    def flatten(self, start_dim=0, end_dim=-1):
        return self.g.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def concat(self, other, axis=0):
        return self.g.concat(self, other, axis=axis)

    def cat(self, tensors, axis=0):
        return self.g.cat([self] + tensors, axis=axis)

    def group_norm(self, normalized_shape, eps=1e-5):
        return self.g.group_norm(self, normalized_shape, eps)

    def layer_norm(self, normalized_shape, eps=1e-5):
        return self.g.layer_norm(self, normalized_shape, eps)
    
    def softmax(self, axis=-1):
        return self.g.softmax(self, axis)

    def numpy(self):
        info = cactus_tensor_info_t()
        rc = _lib.cactus_graph_get_output_info(self.g.h, cactus_node_t(self.id), ctypes.byref(info))
        if rc != 0:
            raise RuntimeError("graph_get_output_info failed")

        out_ptr = ctypes.c_void_p()
        rc = _lib.cactus_graph_get_output_ptr(self.g.h, cactus_node_t(self.id), ctypes.byref(out_ptr))
        if rc != 0 or not out_ptr.value:
            raise RuntimeError("graph_get_output_ptr failed")

        rank = int(info.rank)
        shape = tuple(int(info.shape[i]) for i in range(rank))
        num_elements = int(info.num_elements)
        precision = int(info.precision)

        if precision == Graph.FP16:
            arr = np.ctypeslib.as_array((ctypes.c_uint16 * num_elements).from_address(out_ptr.value)).view(np.float16)
        elif precision == Graph.FP32:
            arr = np.ctypeslib.as_array((ctypes.c_float * num_elements).from_address(out_ptr.value))
        elif precision == Graph.INT8:
            arr = np.ctypeslib.as_array((ctypes.c_int8 * num_elements).from_address(out_ptr.value))
        elif precision == Graph.INT4:
            arr = np.ctypeslib.as_array((ctypes.c_uint8 * int(info.byte_size)).from_address(out_ptr.value))
            return arr.copy()
        else:
            raise RuntimeError("unsupported precision")

        return arr.reshape(shape).copy()

    def __repr__(self):
        return f"Tensor(id={self.id}, shape={self.shape}, dtype={self.dtype})"
