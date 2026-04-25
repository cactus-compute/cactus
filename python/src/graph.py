import ctypes
import numpy as np

from .cactus import _lib, cactus_node_t, cactus_tensor_info_t


class Graph:
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT4 = 3
    CPU = 0
    NPU = 1
    ACT_SILU = 0
    ACT_GELU = 1
    ACT_GELU_ERF = 2
    ACT_RELU = 3
    ACT_SIGMOID = 4
    ACT_TANH = 5

    def __init__(self):
        self.h = _lib.cactus_graph_create()
        if not self.h:
            raise RuntimeError("cactus_graph_create failed")
    
    def save(self, filename):
        rc = _lib.cactus_graph_save(self.h, str(filename).encode())
        if rc != 0:
            raise RuntimeError("graph_save failed")

    @classmethod
    def load(cls, filename):
        h = _lib.cactus_graph_load(str(filename).encode())
        if not h:
            raise RuntimeError("cactus_graph_load failed")
        obj = cls.__new__(cls)
        obj.h = h
        return obj

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

    def set_external_input(self, tensor, data_ptr, dtype=None):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if tensor.g is not self:
            raise ValueError("tensor belongs to a different graph")
        target_dtype = int(tensor.dtype if dtype is None else dtype)
        ptr = ctypes.c_void_p(data_ptr if isinstance(data_ptr, int) else int(data_ptr))
        rc = _lib.cactus_graph_set_external_input(
            self.h,
            cactus_node_t(tensor.id),
            ptr,
            target_dtype,
        )
        if rc != 0:
            raise RuntimeError("graph_set_external_input failed")

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

    def add_clipped(self, a, b):
        return self._binary("cactus_graph_add_clipped", a, b)

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

    def precision_cast(self, x, dtype):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_precision_cast(self.h, cactus_node_t(x.id), int(dtype), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_precision_cast failed")
        return self._tensor_from_node(out.value)

    def quantize_activations(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_quantize_activations(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_quantize_activations failed")
        return self._tensor_from_node(out.value)

    def _scalar(self, fn_name, x, value=None):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        fn = getattr(_lib, fn_name)
        if value is None:
            rc = fn(self.h, cactus_node_t(x.id), ctypes.byref(out))
        else:
            rc = fn(self.h, cactus_node_t(x.id), ctypes.c_float(float(value)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed")
        return self._tensor_from_node(out.value)

    def scalar_add(self, x, value):
        return self._scalar("cactus_graph_scalar_add", x, value)

    def scalar_subtract(self, x, value):
        return self._scalar("cactus_graph_scalar_subtract", x, value)

    def scalar_multiply(self, x, value):
        return self._scalar("cactus_graph_scalar_multiply", x, value)

    def scalar_divide(self, x, value):
        return self._scalar("cactus_graph_scalar_divide", x, value)

    def scalar_exp(self, x):
        return self._scalar("cactus_graph_scalar_exp", x)

    def scalar_sqrt(self, x):
        return self._scalar("cactus_graph_scalar_sqrt", x)

    def scalar_cos(self, x):
        return self._scalar("cactus_graph_scalar_cos", x)

    def scalar_sin(self, x):
        return self._scalar("cactus_graph_scalar_sin", x)

    def scalar_log(self, x):
        return self._scalar("cactus_graph_scalar_log", x)

    def view(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_view(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_view failed")
        return self._tensor_from_node(out.value)

    def reshape(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_reshape(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_reshape failed")
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

    def slice(self, x, axis, start, length=0):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_slice(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(axis)),
            ctypes.c_size_t(int(start)),
            ctypes.c_size_t(int(length)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_slice failed")
        return self._tensor_from_node(out.value)

    def index(self, x, index_value, axis=0):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_index(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_size_t(int(index_value)),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_index failed")
        return self._tensor_from_node(out.value)

    def transpose(self, x, backend=CPU):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_transpose(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(backend)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_transpose failed")
        return self._tensor_from_node(out.value)

    def permute(self, x, permutation, backend=CPU):
        x = self._ensure_tensor(x)
        permutation = tuple(int(v) for v in permutation)
        arr = (ctypes.c_size_t * len(permutation))(*permutation)
        out = cactus_node_t()
        rc = _lib.cactus_graph_transpose_n(
            self.h,
            cactus_node_t(x.id),
            arr,
            len(permutation),
            ctypes.c_int32(int(backend)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_transpose_n failed")
        return self._tensor_from_node(out.value)

    def matmul(self, a, b, pretransposed_rhs=False, backend=CPU):
        a = self._ensure_tensor(a)
        b = self._ensure_tensor(b)
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
            raise RuntimeError("graph_matmul failed")
        return self._tensor_from_node(out.value)

    def gather(self, tensor, indices):
        tensor = self._ensure_tensor(tensor)
        indices = self._ensure_tensor(indices)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gather(self.h, cactus_node_t(tensor.id), cactus_node_t(indices.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_gather failed")
        return self._tensor_from_node(out.value)

    def embedding_from_tensor(self, embedding_tensor, indices):
        embedding_tensor = self._ensure_tensor(embedding_tensor)
        indices = self._ensure_tensor(indices)
        out = cactus_node_t()
        rc = _lib.cactus_graph_embedding_from_tensor(
            self.h, cactus_node_t(embedding_tensor.id), cactus_node_t(indices.id), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_embedding_from_tensor failed")
        return self._tensor_from_node(out.value)

    def embedding_from_file(self, filename, indices):
        indices = self._ensure_tensor(indices)
        out = cactus_node_t()
        rc = _lib.cactus_graph_embedding_from_file(self.h, str(filename).encode(), cactus_node_t(indices.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_embedding_from_file failed")
        return self._tensor_from_node(out.value)

    def mmap_embeddings(self, filename):
        out = cactus_node_t()
        rc = _lib.cactus_graph_mmap_embeddings(self.h, str(filename).encode(), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_mmap_embeddings failed")
        return self._tensor_from_node(out.value)

    def mmap_weights(self, filename):
        out = cactus_node_t()
        rc = _lib.cactus_graph_mmap_weights(self.h, str(filename).encode(), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_mmap_weights failed")
        return self._tensor_from_node(out.value)

    def bilinear_interpolation(self, pos_embeds, dst_height, dst_width):
        pos_embeds = self._ensure_tensor(pos_embeds)
        out = cactus_node_t()
        rc = _lib.cactus_graph_bilinear_interpolation(
            self.h, cactus_node_t(pos_embeds.id), ctypes.c_size_t(int(dst_height)), ctypes.c_size_t(int(dst_width)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_bilinear_interpolation failed")
        return self._tensor_from_node(out.value)

    def set_grouped_scales(self, tensor, group_size, num_groups, scales):
        tensor = self._ensure_tensor(tensor)
        arr = np.ascontiguousarray(scales, dtype=np.float16)
        rc = _lib.cactus_graph_set_grouped_scales(
            self.h,
            cactus_node_t(tensor.id),
            ctypes.c_size_t(int(group_size)),
            ctypes.c_size_t(int(num_groups)),
            arr.ctypes.data_as(ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("graph_set_grouped_scales failed")

    def set_interleaved(self, tensor, interleaved=True, original_n=0):
        tensor = self._ensure_tensor(tensor)
        rc = _lib.cactus_graph_set_interleaved(
            self.h, cactus_node_t(tensor.id), ctypes.c_bool(bool(interleaved)), ctypes.c_size_t(int(original_n))
        )
        if rc != 0:
            raise RuntimeError("graph_set_interleaved failed")

    def release_weight_pages(self, tensor):
        tensor = self._ensure_tensor(tensor)
        rc = _lib.cactus_graph_release_weight_pages(self.h, cactus_node_t(tensor.id))
        if rc != 0:
            raise RuntimeError("graph_release_weight_pages failed")

    def prefetch_weight_pages(self, tensor):
        tensor = self._ensure_tensor(tensor)
        rc = _lib.cactus_graph_prefetch_weight_pages(self.h, cactus_node_t(tensor.id))
        if rc != 0:
            raise RuntimeError("graph_prefetch_weight_pages failed")

    def release_all_weight_pages(self):
        rc = _lib.cactus_graph_release_all_weight_pages(self.h)
        if rc != 0:
            raise RuntimeError("graph_release_all_weight_pages failed")

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

    def groupnorm(self, x, weight, bias, num_groups, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        bias = self._ensure_tensor(bias)
        out = cactus_node_t()
        rc = _lib.cactus_graph_groupnorm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            cactus_node_t(bias.id),
            ctypes.c_size_t(int(num_groups)),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_group_norm failed")
        return self._tensor_from_node(out.value)

    def group_norm(self, x, weight, bias, num_groups, eps=1e-5):
        return self.groupnorm(x, weight, bias, num_groups, eps=eps)

    def layernorm(self, x, weight, bias=None, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        has_bias = bias is not None
        bias_node = cactus_node_t(0)
        if has_bias:
            bias = self._ensure_tensor(bias)
            bias_node = cactus_node_t(bias.id)
        out = cactus_node_t()
        rc = _lib.cactus_graph_layernorm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            bias_node,
            ctypes.c_float(float(eps)),
            ctypes.c_bool(has_bias),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_layer_norm failed")
        return self._tensor_from_node(out.value)

    def layer_norm(self, x, weight, bias=None, eps=1e-5):
        return self.layernorm(x, weight, bias=bias, eps=eps)

    def batchnorm(self, x, weight, bias, running_mean, running_var, axis=1, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        bias = self._ensure_tensor(bias)
        running_mean = self._ensure_tensor(running_mean)
        running_var = self._ensure_tensor(running_var)
        out = cactus_node_t()
        rc = _lib.cactus_graph_batchnorm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            cactus_node_t(bias.id),
            cactus_node_t(running_mean.id),
            cactus_node_t(running_var.id),
            ctypes.c_int32(int(axis)),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_batchnorm failed")
        return self._tensor_from_node(out.value)

    def batch_norm(self, x, weight, bias, running_mean, running_var, axis=1, eps=1e-5):
        return self.batchnorm(x, weight, bias, running_mean, running_var, axis=axis, eps=eps)

    def rms_norm(self, x, weight, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rms_norm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_rms_norm failed")
        return self._tensor_from_node(out.value)

    def topk(self, x, k):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_topk(self.h, cactus_node_t(x.id), ctypes.c_size_t(int(k)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_topk failed")
        return self._tensor_from_node(out.value)

    def rope(self, x, theta, position_offset=0, backend=CPU):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rope(
            self.h, cactus_node_t(x.id), ctypes.c_float(float(theta)), ctypes.c_size_t(int(position_offset)),
            ctypes.c_int32(int(backend)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_rope failed")
        return self._tensor_from_node(out.value)

    def rope_gptj(self, x, theta, position_offset=0, rot_dim=0, backend=CPU):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rope_gptj(
            self.h, cactus_node_t(x.id), ctypes.c_float(float(theta)), ctypes.c_size_t(int(position_offset)),
            ctypes.c_size_t(int(rot_dim)), ctypes.c_int32(int(backend)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_rope_gptj failed")
        return self._tensor_from_node(out.value)

    def _reduce(self, fn_name, x, axis):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = getattr(_lib, fn_name)(self.h, cactus_node_t(x.id), ctypes.c_int32(int(axis)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed")
        return self._tensor_from_node(out.value)

    def sum(self, x, axis):
        return self._reduce("cactus_graph_sum", x, axis)

    def mean(self, x, axis):
        return self._reduce("cactus_graph_mean", x, axis)

    def variance(self, x, axis):
        return self._reduce("cactus_graph_variance", x, axis)

    def min(self, x, axis):
        return self._reduce("cactus_graph_min", x, axis)

    def max(self, x, axis):
        return self._reduce("cactus_graph_max", x, axis)
    
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

    def attention(self, query, key, value, scale, is_causal=True, position_offset=0, window_size=0,
                  backend=CPU, mask=None, additive_mask=False):
        query = self._ensure_tensor(query)
        key = self._ensure_tensor(key)
        value = self._ensure_tensor(value)
        mask_node = cactus_node_t(0)
        use_mask = mask is not None
        if use_mask:
            mask_node = cactus_node_t(self._ensure_tensor(mask).id)
        out = cactus_node_t()
        rc = _lib.cactus_graph_attention(
            self.h,
            cactus_node_t(query.id),
            cactus_node_t(key.id),
            cactus_node_t(value.id),
            ctypes.c_float(float(scale)),
            ctypes.c_bool(bool(is_causal)),
            ctypes.c_size_t(int(position_offset)),
            ctypes.c_size_t(int(window_size)),
            ctypes.c_int32(int(backend)),
            ctypes.c_bool(use_mask),
            mask_node,
            ctypes.c_bool(bool(additive_mask)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_attention failed")
        return self._tensor_from_node(out.value)

    def rel_pos_bias(self, query, relative_key, scale):
        query = self._ensure_tensor(query)
        relative_key = self._ensure_tensor(relative_key)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rel_pos_bias(
            self.h, cactus_node_t(query.id), cactus_node_t(relative_key.id), ctypes.c_float(float(scale)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_rel_pos_bias failed")
        return self._tensor_from_node(out.value)

    def attention_int8_hybrid(self, query, key_new, value_new, scale, position_offset,
                              cached_keys, cached_values, k_scales, v_scales,
                              cache_len, num_kv_heads, head_dim, window_size=0):
        query = self._ensure_tensor(query)
        key_new = self._ensure_tensor(key_new)
        value_new = self._ensure_tensor(value_new)
        ck = np.ascontiguousarray(cached_keys, dtype=np.int8)
        cv = np.ascontiguousarray(cached_values, dtype=np.int8)
        ks = np.ascontiguousarray(k_scales, dtype=np.float32)
        vs = np.ascontiguousarray(v_scales, dtype=np.float32)
        out = cactus_node_t()
        rc = _lib.cactus_graph_attention_int8_hybrid(
            self.h,
            cactus_node_t(query.id),
            cactus_node_t(key_new.id),
            cactus_node_t(value_new.id),
            ctypes.c_float(float(scale)),
            ctypes.c_size_t(int(position_offset)),
            ck.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            cv.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(int(cache_len)),
            ctypes.c_size_t(int(num_kv_heads)),
            ctypes.c_size_t(int(head_dim)),
            ctypes.c_size_t(int(window_size)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_attention_int8_hybrid failed")
        return self._tensor_from_node(out.value)
    
    def relu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_relu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_relu failed")
        return self._tensor_from_node(out.value)

    def silu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_silu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_silu failed")
        return self._tensor_from_node(out.value)

    def gelu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gelu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_gelu failed")
        return self._tensor_from_node(out.value)

    def gelu_erf(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gelu_erf(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_gelu_erf failed")
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

    def glu(self, x, axis=-1):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_glu(self.h, cactus_node_t(x.id), ctypes.c_int32(int(axis)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_glu failed")
        return self._tensor_from_node(out.value)

    def conv1d_causal(self, x, weight, kernel_size, dilation):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        out = cactus_node_t()
        rc = _lib.cactus_graph_conv1d_causal(
            self.h, cactus_node_t(x.id), cactus_node_t(weight.id),
            ctypes.c_size_t(int(kernel_size)), ctypes.c_size_t(int(dilation)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_conv1d_causal failed")
        return self._tensor_from_node(out.value)

    def conv1d_k3(self, x, weight, stride=1):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        out = cactus_node_t()
        rc = _lib.cactus_graph_conv1d_k3(
            self.h, cactus_node_t(x.id), cactus_node_t(weight.id), ctypes.c_size_t(int(stride)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_conv1d_k3 failed")
        return self._tensor_from_node(out.value)

    def conv1d_k7s3(self, x, weight, bias):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        bias = self._ensure_tensor(bias)
        out = cactus_node_t()
        rc = _lib.cactus_graph_conv1d_k7s3(
            self.h, cactus_node_t(x.id), cactus_node_t(weight.id), cactus_node_t(bias.id), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_conv1d_k7s3 failed")
        return self._tensor_from_node(out.value)

    def conv1d(self, x, weight, bias=None, stride=1):
        return self._conv_with_optional_bias("cactus_graph_conv1d", x, weight, bias, ctypes.c_size_t(int(stride)))

    def conv1d_same_depthwise_k9(self, x, weight, bias=None):
        return self._conv_with_optional_bias("cactus_graph_conv1d_same_depthwise_k9", x, weight, bias)

    def conv1d_pointwise(self, x, weight, bias=None):
        return self._conv_with_optional_bias("cactus_graph_conv1d_pointwise", x, weight, bias)

    def conv2d_k3s2p1(self, x, weight, bias=None):
        return self._conv_with_optional_bias("cactus_graph_conv2d_k3s2p1", x, weight, bias)

    def conv2d_depthwise_k3s2p1(self, x, weight, bias=None):
        return self._conv_with_optional_bias("cactus_graph_conv2d_depthwise_k3s2p1", x, weight, bias)

    def conv2d_pointwise_1x1(self, x, weight, bias=None):
        return self._conv_with_optional_bias("cactus_graph_conv2d_pointwise_1x1", x, weight, bias)

    def _conv_with_optional_bias(self, fn_name, x, weight, bias=None, *extra):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        has_bias = bias is not None
        bias_node = cactus_node_t(0 if bias is None else self._ensure_tensor(bias).id)
        out = cactus_node_t()
        rc = getattr(_lib, fn_name)(
            self.h, cactus_node_t(x.id), cactus_node_t(weight.id), ctypes.c_bool(has_bias), bias_node, *extra, ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed")
        return self._tensor_from_node(out.value)

    def lstm_cell(self, input, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh):
        tensors = [self._ensure_tensor(t) for t in (input, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh)]
        out = cactus_node_t()
        rc = _lib.cactus_graph_lstm_cell(self.h, *(cactus_node_t(t.id) for t in tensors), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_lstm_cell failed")
        return self._tensor_from_node(out.value)

    def gated_deltanet_decode(self, query, key, value, gate_log, beta, initial_state, scale):
        tensors = [self._ensure_tensor(t) for t in (query, key, value, gate_log, beta, initial_state)]
        out = cactus_node_t()
        rc = _lib.cactus_graph_gated_deltanet_decode(
            self.h, *(cactus_node_t(t.id) for t in tensors), ctypes.c_float(float(scale)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_gated_deltanet_decode failed")
        return self._tensor_from_node(out.value)

    def gated_deltanet_prefill(self, query, key, value, gate_log, beta, initial_state, chunk_size, scale):
        tensors = [self._ensure_tensor(t) for t in (query, key, value, gate_log, beta, initial_state)]
        out = cactus_node_t()
        rc = _lib.cactus_graph_gated_deltanet_prefill(
            self.h, *(cactus_node_t(t.id) for t in tensors), ctypes.c_size_t(int(chunk_size)), ctypes.c_float(float(scale)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_gated_deltanet_prefill failed")
        return self._tensor_from_node(out.value)

    def stft(self, x, weight, stride, num_fft_bins):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        out = cactus_node_t()
        rc = _lib.cactus_graph_stft(
            self.h, cactus_node_t(x.id), cactus_node_t(weight.id),
            ctypes.c_size_t(int(stride)), ctypes.c_size_t(int(num_fft_bins)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_stft failed")
        return self._tensor_from_node(out.value)

    def altup_predict(self, coefs, streams):
        coefs = self._ensure_tensor(coefs)
        streams = [self._ensure_tensor(t) for t in streams]
        ids = (cactus_node_t * len(streams))(*(cactus_node_t(t.id) for t in streams))
        out = cactus_node_t()
        rc = _lib.cactus_graph_altup_predict(self.h, cactus_node_t(coefs.id), ids, ctypes.c_size_t(len(streams)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_altup_predict failed")
        return self._tensor_from_node(out.value)

    def altup_correct(self, coefs, innovation, predictions):
        coefs = self._ensure_tensor(coefs)
        innovation = self._ensure_tensor(innovation)
        predictions = [self._ensure_tensor(t) for t in predictions]
        ids = (cactus_node_t * len(predictions))(*(cactus_node_t(t.id) for t in predictions))
        out = cactus_node_t()
        rc = _lib.cactus_graph_altup_correct(
            self.h, cactus_node_t(coefs.id), cactus_node_t(innovation.id), ids, ctypes.c_size_t(len(predictions)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_altup_correct failed")
        return self._tensor_from_node(out.value)

    def gaussian_topk(self, x, ppf):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gaussian_topk(self.h, cactus_node_t(x.id), ctypes.c_float(float(ppf)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_gaussian_topk failed")
        return self._tensor_from_node(out.value)

    def moe_layer_gated(self, hidden, routing_probs, topk_indices, w1_weights, w3_weights, w2_weights,
                        num_experts, num_experts_per_tok, normalize_routing=True, epsilon=1e-6, routed_scaling_factor=1.0):
        hidden = self._ensure_tensor(hidden)
        routing_probs = self._ensure_tensor(routing_probs)
        topk_indices = self._ensure_tensor(topk_indices)
        w1 = (cactus_node_t * len(w1_weights))(*(cactus_node_t(self._ensure_tensor(t).id) for t in w1_weights))
        w3 = (cactus_node_t * len(w3_weights))(*(cactus_node_t(self._ensure_tensor(t).id) for t in w3_weights))
        w2 = (cactus_node_t * len(w2_weights))(*(cactus_node_t(self._ensure_tensor(t).id) for t in w2_weights))
        out = cactus_node_t()
        rc = _lib.cactus_graph_moe_layer_gated(
            self.h, cactus_node_t(hidden.id), cactus_node_t(routing_probs.id), cactus_node_t(topk_indices.id),
            w1, w3, w2, ctypes.c_size_t(int(num_experts)), ctypes.c_size_t(int(num_experts_per_tok)),
            ctypes.c_bool(bool(normalize_routing)), ctypes.c_float(float(epsilon)),
            ctypes.c_float(float(routed_scaling_factor)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_moe_layer_gated failed")
        return self._tensor_from_node(out.value)

    def moe_layer_ungated(self, hidden, routing_probs, topk_indices, w1_weights, w2_weights,
                          num_experts, num_experts_per_tok, normalize_routing=True, epsilon=1e-6,
                          routed_scaling_factor=1.0, activation=ACT_GELU):
        hidden = self._ensure_tensor(hidden)
        routing_probs = self._ensure_tensor(routing_probs)
        topk_indices = self._ensure_tensor(topk_indices)
        w1 = (cactus_node_t * len(w1_weights))(*(cactus_node_t(self._ensure_tensor(t).id) for t in w1_weights))
        w2 = (cactus_node_t * len(w2_weights))(*(cactus_node_t(self._ensure_tensor(t).id) for t in w2_weights))
        out = cactus_node_t()
        rc = _lib.cactus_graph_moe_layer_ungated(
            self.h, cactus_node_t(hidden.id), cactus_node_t(routing_probs.id), cactus_node_t(topk_indices.id),
            w1, w2, ctypes.c_size_t(int(num_experts)), ctypes.c_size_t(int(num_experts_per_tok)),
            ctypes.c_bool(bool(normalize_routing)), ctypes.c_float(float(epsilon)),
            ctypes.c_float(float(routed_scaling_factor)), ctypes.c_int32(int(activation)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_moe_layer_ungated failed")
        return self._tensor_from_node(out.value)

    def sample(self, logits, temperature=1.0, top_p=1.0, top_k=0):
        logits = self._ensure_tensor(logits)
        out = cactus_node_t()
        rc = _lib.cactus_graph_sample(
            self.h, cactus_node_t(logits.id), ctypes.c_float(float(temperature)),
            ctypes.c_float(float(top_p)), ctypes.c_size_t(int(top_k)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_sample failed")
        return self._tensor_from_node(out.value)

    def scatter_topk(self, indices, values, num_classes):
        indices = self._ensure_tensor(indices)
        values = self._ensure_tensor(values)
        out = cactus_node_t()
        rc = _lib.cactus_graph_scatter_topk(
            self.h, cactus_node_t(indices.id), cactus_node_t(values.id), ctypes.c_size_t(int(num_classes)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError("graph_scatter_topk failed")
        return self._tensor_from_node(out.value)

    def persistent(self, source_node):
        source_node = self._ensure_tensor(source_node)
        out = cactus_node_t()
        rc = _lib.cactus_graph_persistent(self.h, cactus_node_t(source_node.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_persistent failed")
        return self._tensor_from_node(out.value)

    def is_populated(self, persistent_node):
        persistent_node = self._ensure_tensor(persistent_node)
        out_is_populated = ctypes.c_int32()
        rc = _lib.cactus_graph_is_populated(self.h, cactus_node_t(persistent_node.id), ctypes.byref(out_is_populated))
        if rc != 0:
            raise RuntimeError("graph_is_populated failed")
        return bool(out_is_populated.value)

    def invalidate_persistent(self, persistent_node):
        persistent_node = self._ensure_tensor(persistent_node)
        rc = _lib.cactus_graph_invalidate_persistent(self.h, cactus_node_t(persistent_node.id))
        if rc != 0:
            raise RuntimeError("graph_invalidate_persistent failed")

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

    def precision_cast(self, dtype):
        return self.g.precision_cast(self, dtype)

    def quantize_activations(self):
        return self.g.quantize_activations(self)

    def scalar_add(self, value):
        return self.g.scalar_add(self, value)

    def scalar_subtract(self, value):
        return self.g.scalar_subtract(self, value)

    def scalar_multiply(self, value):
        return self.g.scalar_multiply(self, value)

    def scalar_divide(self, value):
        return self.g.scalar_divide(self, value)

    def scalar_exp(self):
        return self.g.scalar_exp(self)

    def scalar_sqrt(self):
        return self.g.scalar_sqrt(self)

    def scalar_cos(self):
        return self.g.scalar_cos(self)

    def scalar_sin(self):
        return self.g.scalar_sin(self)

    def scalar_log(self):
        return self.g.scalar_log(self)

    def relu(self):
        return self.g.relu(self)

    def sigmoid(self):
        return self.g.sigmoid(self)

    def tanh(self):
        return self.g.tanh(self)

    def gelu(self):
        return self.g.gelu(self)

    def gelu_erf(self):
        return self.g.gelu_erf(self)

    def silu(self):
        return self.g.silu(self)

    def view(self, shape):
        return self.g.view(self, shape)

    def reshape(self, shape):
        return self.g.reshape(self, shape)

    def flatten(self, start_dim=0, end_dim=-1):
        return self.g.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def slice(self, axis, start, length=0):
        return self.g.slice(self, axis, start, length=length)

    def index(self, index_value, axis=0):
        return self.g.index(self, index_value, axis=axis)

    def transpose(self, backend=Graph.CPU):
        return self.g.transpose(self, backend=backend)

    def permute(self, permutation, backend=Graph.CPU):
        return self.g.permute(self, permutation, backend=backend)

    def concat(self, other, axis=0):
        return self.g.concat(self, other, axis=axis)

    def cat(self, tensors, axis=0):
        return self.g.cat([self] + tensors, axis=axis)

    def groupnorm(self, weight, bias, num_groups, eps=1e-5):
        return self.g.groupnorm(self, weight, bias, num_groups, eps=eps)

    def layernorm(self, weight, bias=None, eps=1e-5):
        return self.g.layernorm(self, weight, bias=bias, eps=eps)

    def batchnorm(self, weight, bias, running_mean, running_var, axis=1, eps=1e-5):
        return self.g.batchnorm(self, weight, bias, running_mean, running_var, axis=axis, eps=eps)

    def group_norm(self, weight, bias, num_groups, eps=1e-5):
        return self.groupnorm(weight, bias, num_groups, eps=eps)

    def layer_norm(self, weight, bias=None, eps=1e-5):
        return self.layernorm(weight, bias=bias, eps=eps)

    def batch_norm(self, weight, bias, running_mean, running_var, axis=1, eps=1e-5):
        return self.batchnorm(weight, bias, running_mean, running_var, axis=axis, eps=eps)

    def rms_norm(self, weight, eps=1e-5):
        return self.g.rms_norm(self, weight, eps=eps)
    
    def softmax(self, axis=-1):
        return self.g.softmax(self, axis)

    def glu(self, axis=-1):
        return self.g.glu(self, axis=axis)

    def matmul(self, other, pretransposed_rhs=False, backend=Graph.CPU):
        return self.g.matmul(self, other, pretransposed_rhs=pretransposed_rhs, backend=backend)

    def sum(self, axis):
        return self.g.sum(self, axis)

    def mean(self, axis):
        return self.g.mean(self, axis)

    def variance(self, axis):
        return self.g.variance(self, axis)

    def min(self, axis):
        return self.g.min(self, axis)

    def max(self, axis):
        return self.g.max(self, axis)

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
