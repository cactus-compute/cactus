"""Cactus Python FFI bindings."""
import ctypes
import json
import platform
from pathlib import Path

import numpy as np

TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

_LIB_NAME = "libcactus_engine.dylib" if platform.system() == "Darwin" else "libcactus_engine.so"


def _find_library():
    bundled = Path(__file__).parent / "lib" / _LIB_NAME
    if bundled.exists():
        return bundled

    dev_build = Path(__file__).parent.parent.parent.parent / "cactus-engine" / "build" / _LIB_NAME
    if dev_build.exists():
        return dev_build

    raise RuntimeError(
        f"Cactus library ({_LIB_NAME}) not found.\n"
        f"Install with: pip install cactus-compute\n"
        f"Or build from source: cactus build --python"
    )


_LIB_PATH = _find_library()
try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
except OSError as exc:
    raise RuntimeError(
        f"Cactus library found at {_LIB_PATH} but failed to load: {exc}.\n"
        "Common causes: wrong arch (the library is built for arm64 only), "
        "missing libcurl on Linux, or a stale build. Rebuild with "
        "`cactus build --python`."
    ) from exc


def _bind_optional(name, argtypes, restype):
    try:
        fn = getattr(_lib, name)
    except AttributeError:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn

cactus_graph_t = ctypes.c_void_p
cactus_node_t = ctypes.c_uint64

class cactus_tensor_info_t(ctypes.Structure):
    _fields_ = [
        ("precision", ctypes.c_int32),
        ("rank", ctypes.c_size_t),
        ("shape", ctypes.c_size_t * 8),
        ("num_elements", ctypes.c_size_t),
        ("byte_size", ctypes.c_size_t),
    ]

_lib.cactus_set_telemetry_environment.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_lib.cactus_set_telemetry_environment.restype = None
_lib.cactus_set_telemetry_environment(b"python", None, None)

_lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
_lib.cactus_init.restype = ctypes.c_void_p

_lib.cactus_set_backend.argtypes = [ctypes.c_char_p]
_lib.cactus_set_backend.restype = ctypes.c_int

# cactus graph API
_lib.cactus_graph_create.restype = cactus_graph_t
_lib.cactus_graph_destroy.argtypes = [cactus_graph_t]
_lib.cactus_graph_destroy.restype = None
_lib.cactus_graph_hard_reset.argtypes = [cactus_graph_t]
_lib.cactus_graph_hard_reset.restype = ctypes.c_int

_lib.cactus_graph_save.argtypes = [cactus_graph_t, ctypes.c_char_p]
_lib.cactus_graph_save.restype = ctypes.c_int

_lib.cactus_graph_load.argtypes = [ctypes.c_char_p]
_lib.cactus_graph_load.restype = cactus_graph_t

_lib.cactus_graph_input.argtypes = [
    cactus_graph_t,
    ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
    ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_input.restype = ctypes.c_int

_lib.cactus_graph_set_input.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_void_p, ctypes.c_int32
]
_lib.cactus_graph_set_input.restype = ctypes.c_int
_lib.cactus_graph_set_external_input.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_void_p, ctypes.c_int32
]
_lib.cactus_graph_set_external_input.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_mark_embedded_input",
    [cactus_graph_t, cactus_node_t],
    ctypes.c_int,
)
_bind_optional(
    "cactus_graph_set_runtime_input_shape",
    [cactus_graph_t, cactus_node_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t],
    ctypes.c_int,
)
_bind_optional(
    "cactus_graph_set_input_dynamic_dims",
    [cactus_graph_t, cactus_node_t, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t],
    ctypes.c_int,
)

_lib.cactus_graph_add.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_add.restype = ctypes.c_int
_lib.cactus_graph_add_clipped.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_add_clipped.restype = ctypes.c_int

_lib.cactus_graph_subtract.argtypes = [cactus_graph_t, cactus_node_t,
  cactus_node_t, ctypes.POINTER(cactus_node_t)]
_lib.cactus_graph_subtract.restype = ctypes.c_int

_lib.cactus_graph_multiply.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_multiply.restype = ctypes.c_int

_lib.cactus_graph_divide.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_divide.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_not_equal",
    [cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)

_lib.cactus_graph_precision_cast.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_precision_cast.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_quantize_activations",
    [cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)

_lib.cactus_graph_scalar_add.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_add.restype = ctypes.c_int
_lib.cactus_graph_scalar_subtract.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_subtract.restype = ctypes.c_int
_lib.cactus_graph_scalar_multiply.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_multiply.restype = ctypes.c_int
_lib.cactus_graph_scalar_divide.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_divide.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_scalar_floor_divide",
    [cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)
_bind_optional(
    "cactus_graph_scalar_not_equal",
    [cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)
_lib.cactus_graph_scalar_exp.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_exp.restype = ctypes.c_int
_lib.cactus_graph_scalar_sqrt.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_sqrt.restype = ctypes.c_int
_lib.cactus_graph_scalar_cos.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_cos.restype = ctypes.c_int
_lib.cactus_graph_scalar_sin.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_sin.restype = ctypes.c_int
_lib.cactus_graph_scalar_log.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scalar_log.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_masked_select_prefix",
    [cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)
_bind_optional(
    "cactus_graph_masked_scatter",
    [cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)

_lib.cactus_graph_abs.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_abs.restype = ctypes.c_int

_lib.cactus_graph_pow.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_pow.restype = ctypes.c_int

_lib.cactus_graph_view.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
    ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_view.restype = ctypes.c_int

_lib.cactus_graph_flatten.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_flatten.restype = ctypes.c_int
_lib.cactus_graph_reshape.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_reshape.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_expand",
    [cactus_graph_t, cactus_node_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)
_lib.cactus_graph_transpose.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_transpose.restype = ctypes.c_int
_lib.cactus_graph_transpose_n.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_transpose_n.restype = ctypes.c_int
_lib.cactus_graph_slice.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_slice.restype = ctypes.c_int
_lib.cactus_graph_index.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_size_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_index.restype = ctypes.c_int

_lib.cactus_graph_concat.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_concat.restype = ctypes.c_int

_lib.cactus_graph_cat.argtypes = [
    cactus_graph_t, ctypes.POINTER(cactus_node_t), ctypes.c_size_t, ctypes.c_int32,
    ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_cat.restype = ctypes.c_int
_lib.cactus_graph_matmul.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_matmul.restype = ctypes.c_int
_lib.cactus_graph_gather.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_gather.restype = ctypes.c_int
_lib.cactus_graph_embedding_from_tensor.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_embedding_from_tensor.restype = ctypes.c_int
_lib.cactus_graph_embedding_from_file.argtypes = [
    cactus_graph_t, ctypes.c_char_p, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_embedding_from_file.restype = ctypes.c_int
_lib.cactus_graph_mmap_embeddings.argtypes = [
    cactus_graph_t, ctypes.c_char_p, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_mmap_embeddings.restype = ctypes.c_int
_lib.cactus_graph_mmap_weights.argtypes = [
    cactus_graph_t, ctypes.c_char_p, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_mmap_weights.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_bind_mmap_weights",
    [cactus_graph_t, cactus_node_t, ctypes.c_char_p],
    ctypes.c_int,
)
_lib.cactus_graph_bilinear_interpolation.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_bilinear_interpolation.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_set_grouped_scales",
    [cactus_graph_t, cactus_node_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p],
    ctypes.c_int,
)
_bind_optional(
    "cactus_graph_set_interleaved",
    [cactus_graph_t, cactus_node_t, ctypes.c_bool, ctypes.c_size_t],
    ctypes.c_int,
)
_lib.cactus_graph_release_weight_pages.argtypes = [cactus_graph_t, cactus_node_t]
_lib.cactus_graph_release_weight_pages.restype = ctypes.c_int
_lib.cactus_graph_prefetch_weight_pages.argtypes = [cactus_graph_t, cactus_node_t]
_lib.cactus_graph_prefetch_weight_pages.restype = ctypes.c_int
_lib.cactus_graph_release_all_weight_pages.argtypes = [cactus_graph_t]
_lib.cactus_graph_release_all_weight_pages.restype = ctypes.c_int

_lib.cactus_graph_sum.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_sum.restype = ctypes.c_int
_lib.cactus_graph_mean.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_mean.restype = ctypes.c_int
_lib.cactus_graph_variance.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_variance.restype = ctypes.c_int
_lib.cactus_graph_min.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_min.restype = ctypes.c_int
_lib.cactus_graph_max.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_max.restype = ctypes.c_int
_lib.cactus_graph_cumsum.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_cumsum.restype = ctypes.c_int

_lib.cactus_graph_relu.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_relu.restype = ctypes.c_int
_lib.cactus_graph_silu.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_silu.restype = ctypes.c_int
_lib.cactus_graph_gelu.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_gelu.restype = ctypes.c_int
_lib.cactus_graph_gelu_erf.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_gelu_erf.restype = ctypes.c_int
_lib.cactus_graph_sigmoid.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_sigmoid.restype = ctypes.c_int
_lib.cactus_graph_tanh.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_tanh.restype = ctypes.c_int
_lib.cactus_graph_glu.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_glu.restype = ctypes.c_int

_lib.cactus_graph_layernorm.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.c_float, ctypes.c_bool, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_layernorm.restype = ctypes.c_int
_lib.cactus_graph_groupnorm.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.c_size_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_groupnorm.restype = ctypes.c_int
_lib.cactus_graph_batchnorm.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.c_int32, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_batchnorm.restype = ctypes.c_int
_lib.cactus_graph_rms_norm.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_rms_norm.restype = ctypes.c_int
_lib.cactus_graph_topk.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_topk.restype = ctypes.c_int
_lib.cactus_graph_rope.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.c_size_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_rope.restype = ctypes.c_int
_lib.cactus_graph_rope_gptj.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_rope_gptj.restype = ctypes.c_int
_lib.cactus_graph_softmax.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_softmax.restype = ctypes.c_int
_lib.cactus_graph_attention.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.c_float, ctypes.c_bool,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32, ctypes.c_bool, cactus_node_t, ctypes.c_bool,
    ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_attention.restype = ctypes.c_int
_lib.cactus_graph_kv_cache_state.argtypes = [
    cactus_graph_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_kv_cache_state.restype = ctypes.c_int
_lib.cactus_graph_kv_cache_append.argtypes = [
    cactus_graph_t,
    cactus_node_t,
    cactus_node_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_kv_cache_append.restype = ctypes.c_int
_lib.cactus_graph_attention_cached.argtypes = [
    cactus_graph_t,
    cactus_node_t,
    cactus_node_t,
    cactus_node_t,
    cactus_node_t,
    cactus_node_t,
    ctypes.c_float,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_attention_cached.restype = ctypes.c_int
_lib.cactus_graph_conv_cache_state.argtypes = [
    cactus_graph_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_conv_cache_state.restype = ctypes.c_int
_lib.cactus_graph_conv_cache_append.argtypes = [
    cactus_graph_t,
    cactus_node_t,
    cactus_node_t,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_conv_cache_append.restype = ctypes.c_int
_lib.cactus_graph_conv_cache_initialize.argtypes = [
    cactus_graph_t,
    cactus_node_t,
    cactus_node_t,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_conv_cache_initialize.restype = ctypes.c_int
_lib.cactus_graph_recurrent_cache_state.argtypes = [
    cactus_graph_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_recurrent_cache_state.restype = ctypes.c_int
_lib.cactus_graph_recurrent_cache_write.argtypes = [
    cactus_graph_t,
    cactus_node_t,
    cactus_node_t,
    ctypes.POINTER(cactus_node_t),
]
_lib.cactus_graph_recurrent_cache_write.restype = ctypes.c_int
_lib.cactus_graph_rel_pos_bias.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_rel_pos_bias.restype = ctypes.c_int
_lib.cactus_graph_attention_int8_hybrid.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.c_float, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_attention_int8_hybrid.restype = ctypes.c_int
_lib.cactus_graph_rfft.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_rfft.restype = ctypes.c_int
_lib.cactus_graph_irfft.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_irfft.restype = ctypes.c_int
_lib.cactus_graph_mel_filter_bank.argtypes = [
    cactus_graph_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float, ctypes.c_float, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_mel_filter_bank.restype = ctypes.c_int
_lib.cactus_graph_spectrogram.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float, ctypes.c_bool, ctypes.c_int,
    ctypes.c_float, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_bool,
    ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_spectrogram.restype = ctypes.c_int
_lib.cactus_graph_image_preprocess.argtypes = [
    cactus_graph_t, cactus_node_t,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_image_preprocess.restype = ctypes.c_int
_lib.cactus_graph_conv1d_causal.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv1d_causal.restype = ctypes.c_int
_lib.cactus_graph_conv1d_k3.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv1d_k3.restype = ctypes.c_int
_lib.cactus_graph_conv1d_k7s3.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv1d_k7s3.restype = ctypes.c_int
_lib.cactus_graph_conv1d.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, cactus_node_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv1d.restype = ctypes.c_int
_lib.cactus_graph_conv1d_same_depthwise_k9.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv1d_same_depthwise_k9.restype = ctypes.c_int
_lib.cactus_graph_conv1d_pointwise.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv1d_pointwise.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_clamp",
    [cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.c_float, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)
_lib.cactus_graph_conv2d_k3s2p1.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv2d_k3s2p1.restype = ctypes.c_int
_lib.cactus_graph_conv2d_depthwise_k3s2p1.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv2d_depthwise_k3s2p1.restype = ctypes.c_int
_lib.cactus_graph_conv2d_pointwise_1x1.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_conv2d_pointwise_1x1.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_conv2d_k3s1p1",
    [cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_bool, cactus_node_t, ctypes.POINTER(cactus_node_t)],
    ctypes.c_int,
)
_bind_optional(
    "cactus_graph_conv2d",
    [
        cactus_graph_t,
        cactus_node_t,
        cactus_node_t,
        ctypes.c_bool,
        cactus_node_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(cactus_node_t),
    ],
    ctypes.c_int,
)
_lib.cactus_graph_lstm_cell.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_lstm_cell.restype = ctypes.c_int
_lib.cactus_graph_gated_deltanet_decode.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_gated_deltanet_decode.restype = ctypes.c_int
_lib.cactus_graph_gated_deltanet_prefill.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t, ctypes.c_size_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_gated_deltanet_prefill.restype = ctypes.c_int
_lib.cactus_graph_stft.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_stft.restype = ctypes.c_int
_lib.cactus_graph_altup_predict.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t), ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_altup_predict.restype = ctypes.c_int
_lib.cactus_graph_altup_correct.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.POINTER(cactus_node_t), ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_altup_correct.restype = ctypes.c_int
_lib.cactus_graph_gaussian_topk.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_gaussian_topk.restype = ctypes.c_int
_lib.cactus_graph_moe_layer_gated.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t,
    ctypes.POINTER(cactus_node_t), ctypes.POINTER(cactus_node_t), ctypes.POINTER(cactus_node_t),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_bool, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_moe_layer_gated.restype = ctypes.c_int
_bind_optional(
    "cactus_graph_dense_mlp_tq_fused",
    [
        cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t, cactus_node_t,
        ctypes.c_float, ctypes.POINTER(cactus_node_t)
    ],
    ctypes.c_int,
)
_lib.cactus_graph_moe_layer_ungated.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, cactus_node_t,
    ctypes.POINTER(cactus_node_t), ctypes.POINTER(cactus_node_t),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_bool, ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_moe_layer_ungated.restype = ctypes.c_int
_lib.cactus_graph_sample.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_float, ctypes.c_float, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_sample.restype = ctypes.c_int
_lib.cactus_graph_scatter_topk.argtypes = [
    cactus_graph_t, cactus_node_t, cactus_node_t, ctypes.c_size_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_scatter_topk.restype = ctypes.c_int
_lib.cactus_graph_persistent.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_persistent.restype = ctypes.c_int
_lib.cactus_graph_is_populated.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(ctypes.c_int32)
]
_lib.cactus_graph_is_populated.restype = ctypes.c_int
_lib.cactus_graph_invalidate_persistent.argtypes = [cactus_graph_t, cactus_node_t]
_lib.cactus_graph_invalidate_persistent.restype = ctypes.c_int

_lib.cactus_graph_execute.argtypes = [cactus_graph_t]
_lib.cactus_graph_execute.restype = ctypes.c_int

_lib.cactus_graph_get_output_ptr.argtypes = [cactus_graph_t, cactus_node_t,
  ctypes.POINTER(ctypes.c_void_p)]
_lib.cactus_graph_get_output_ptr.restype = ctypes.c_int

_lib.cactus_graph_get_output_info.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_tensor_info_t)
]
_lib.cactus_graph_get_output_info.restype = ctypes.c_int

_lib.cactus_complete.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_char_p, TokenCallback, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_complete.restype = ctypes.c_int

_lib.cactus_prefill.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_prefill.restype = ctypes.c_int

_lib.cactus_transcribe.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_size_t, ctypes.c_char_p, TokenCallback, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_transcribe.restype = ctypes.c_int

_lib.cactus_stream_transcribe_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.cactus_stream_transcribe_start.restype = ctypes.c_void_p

_lib.cactus_stream_transcribe_process.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_size_t
]
_lib.cactus_stream_transcribe_process.restype = ctypes.c_int

_lib.cactus_stream_transcribe_stop.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
_lib.cactus_stream_transcribe_stop.restype = ctypes.c_int

_bind_optional(
    "cactus_detect_language",
    [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ],
    ctypes.c_int,
)

_lib.cactus_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_bool
]
_lib.cactus_embed.restype = ctypes.c_int

_lib.cactus_image_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_image_embed.restype = ctypes.c_int

_lib.cactus_audio_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_audio_embed.restype = ctypes.c_int

try:
    _lib.cactus_preprocess_audio_features.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_preprocess_audio_features.restype = ctypes.c_int
except AttributeError:
    pass


_lib.cactus_reset.argtypes = [ctypes.c_void_p]
_lib.cactus_reset.restype = None

_lib.cactus_stop.argtypes = [ctypes.c_void_p]
_lib.cactus_stop.restype = None

_lib.cactus_destroy.argtypes = [ctypes.c_void_p]
_lib.cactus_destroy.restype = None

_lib.cactus_get_last_error.argtypes = []
_lib.cactus_get_last_error.restype = ctypes.c_char_p

_lib.cactus_tokenize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.cactus_tokenize.restype = ctypes.c_int

_bind_optional(
    "cactus_render_prompt",
    [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    ctypes.c_int,
)

_bind_optional(
    "cactus_decode_tokens",
    [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
    ],
    ctypes.c_int,
)

_lib.cactus_score_window.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
]
_lib.cactus_score_window.restype = ctypes.c_int

_lib.cactus_rag_query.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_size_t, ctypes.c_size_t
]
_lib.cactus_rag_query.restype = ctypes.c_int

_lib.cactus_index_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.cactus_index_init.restype = ctypes.c_void_p

_lib.cactus_index_add.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_size_t,
    ctypes.c_size_t
]
_lib.cactus_index_add.restype = ctypes.c_int

_lib.cactus_index_delete.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t
]
_lib.cactus_index_delete.restype = ctypes.c_int

_lib.cactus_index_query.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_index_query.restype = ctypes.c_int

_lib.cactus_index_compact.argtypes = [ctypes.c_void_p]
_lib.cactus_index_compact.restype = ctypes.c_int

_lib.cactus_index_destroy.argtypes = [ctypes.c_void_p]
_lib.cactus_index_destroy.restype = None

_lib.cactus_index_get.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_index_get.restype = ctypes.c_int

_lib.cactus_set_app_id.argtypes = [ctypes.c_char_p]
_lib.cactus_set_app_id.restype = None

_lib.cactus_telemetry_flush.argtypes = []
_lib.cactus_telemetry_flush.restype = None

_lib.cactus_telemetry_shutdown.argtypes = []
_lib.cactus_telemetry_shutdown.restype = None

LogCallback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p)

_lib.cactus_log_set_level.argtypes = [ctypes.c_int]
_lib.cactus_log_set_level.restype = None

_lib.cactus_log_set_callback.argtypes = [LogCallback, ctypes.c_void_p]
_lib.cactus_log_set_callback.restype = None


def _enc(s):
    """Encode a string to bytes for C, or pass through None/bytes."""
    if s is None:
        return None
    return s.encode() if isinstance(s, str) else s


def _to_json(obj):
    """Accept str, bytes, dict, list, or None — return encoded bytes for C."""
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return json.dumps(obj).encode()
    if isinstance(obj, str):
        return obj.encode()
    return obj


def _from_json(buf):
    """Decode a ctypes string buffer to a dict. Returns {} on empty."""
    text = buf.value.decode("utf-8", errors="ignore")
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if len(buf.value) >= len(buf) - 1:
            raise RuntimeError(
                f"C engine filled the {len(buf)} byte response buffer; "
                "response was truncated before valid JSON could be parsed."
            ) from exc
        raise


def _prepare_pcm(pcm_data):
    """Marshal pcm_data bytes to (ctypes_ptr, size) for C. Returns (None, 0) if None."""
    if pcm_data is None:
        return None, 0
    pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
    return ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8)), len(pcm_data)


def _make_token_callback(callback):
    """Wrap a Python callback(text, token_id) into a C-compatible TokenCallback."""
    if not callback:
        return TokenCallback()
    def _bridge(token_bytes, token_id, _):
        callback(token_bytes.decode("utf-8", errors="ignore") if token_bytes else "", token_id)
    return TokenCallback(_bridge)


def cactus_get_last_error():
    """Returns the last error message from the C runtime, or None."""
    result = _lib.cactus_get_last_error()
    return result.decode() if result else None


def _err(default):
    return cactus_get_last_error() or default


# ── Telemetry ────────────────────────────────────────────────────────


def cactus_set_telemetry_environment(framework, cache_location, version):
    """Sets telemetry environment (framework name, cache path, version)."""
    _lib.cactus_set_telemetry_environment(_enc(framework), _enc(cache_location), _enc(version))


def cactus_set_app_id(app_id):
    """Sets the application identifier for telemetry."""
    _lib.cactus_set_app_id(_enc(app_id))


def cactus_telemetry_flush():
    """Flushes pending telemetry events."""
    _lib.cactus_telemetry_flush()


def cactus_telemetry_shutdown():
    """Shuts down the telemetry subsystem."""
    _lib.cactus_telemetry_shutdown()


# ── Model lifecycle ──────────────────────────────────────────────────


def cactus_set_backend(backend):
    """Select the inference backend ("auto", "cpu", or "metal"). Returns 0 on success, -1 on failure."""
    return _lib.cactus_set_backend(_enc(backend))


def cactus_init(model_path, corpus_dir=None, cache_index=False):
    """Load a model from disk.

    Args:
        model_path: Path to the converted model weights directory.
        corpus_dir: Optional path to a RAG corpus directory.
        cache_index: Whether to cache the RAG index to disk.

    Returns:
        An opaque model handle. Pass to other cactus_* functions.
        Call cactus_destroy() when done.
    """
    handle = _lib.cactus_init(_enc(model_path), _enc(corpus_dir), cache_index)
    if not handle:
        raise RuntimeError(_err("Failed to initialize model"))
    return handle


def cactus_destroy(model):
    """Free all resources associated with a model handle. Safe to call on
    a null/already-destroyed handle (no-op)."""
    if not model:
        return
    _lib.cactus_destroy(model)


def cactus_reset(model):
    """Clear the KV cache, resetting the model to a fresh conversation state.
    No-op when ``model`` is null."""
    if not model:
        return
    _lib.cactus_reset(model)


def cactus_stop(model):
    """Signal the model to stop the current generation early. No-op when
    ``model`` is null."""
    if not model:
        return
    _lib.cactus_stop(model)


# ── LLM completion ───────────────────────────────────────────────────


def cactus_complete(model, messages, options=None, tools=None, callback=None, pcm_data=None):
    """Run chat completion.

    Args:
        model:    Model handle from cactus_init().
        messages: List of message dicts, e.g. [{"role": "user", "content": "Hi"}].
                  Also accepts a pre-serialized JSON string.
        options:  Optional dict of generation options (temperature, max_tokens, etc.).
        tools:    Optional list of tool definitions for function calling.
        callback: Optional function(text, token_id) called on each generated token.
        pcm_data: Optional raw PCM audio bytes for audio-capable models.

    Returns:
        A dict with the completion response and metrics.
    """
    buf = ctypes.create_string_buffer(1 << 20)
    cb = _make_token_callback(callback)
    pcm_ptr, pcm_size = _prepare_pcm(pcm_data)
    rc = _lib.cactus_complete(
        model, _to_json(messages), buf, len(buf),
        _to_json(options), _to_json(tools), cb, None, pcm_ptr, pcm_size,
    )
    if rc < 0:
        raise RuntimeError(_err("Completion failed"))
    return _from_json(buf)


def cactus_prefill(model, messages, options=None, tools=None, pcm_data=None):
    """Pre-fill the KV cache with messages without generating a response.

    Returns:
        A dict with prefill stats (tokens processed, latency, etc.).
    """
    buf = ctypes.create_string_buffer(1 << 20)
    pcm_ptr, pcm_size = _prepare_pcm(pcm_data)
    rc = _lib.cactus_prefill(
        model, _to_json(messages), buf, len(buf),
        _to_json(options), _to_json(tools), pcm_ptr, pcm_size,
    )
    if rc < 0:
        raise RuntimeError(_err("Prefill failed"))
    return _from_json(buf)


# ── Audio / speech ───────────────────────────────────────────────────


def cactus_transcribe(model, audio_path, prompt=None, options=None, callback=None, pcm_data=None):
    """Transcribe audio to text.

    Args:
        model:      Model handle.
        audio_path: Path to a WAV audio file.
        prompt:     Optional prompt to guide transcription.
        options:    Optional dict of transcription options. Set ``timestamps: True`` (Whisper
                    only) to populate the ``segments`` array.
        callback:   Optional function(text, token_id) for streaming tokens.
        pcm_data:   Optional raw PCM audio bytes (alternative to audio_path).

    Returns:
        A dict with the transcribed text under ``response`` and a ``segments`` array of
        ``{start, end, text}`` objects, populated only for Whisper when ``timestamps`` is set
        (empty otherwise, including all Parakeet transcription).
    """
    buf = ctypes.create_string_buffer(1 << 20)
    cb = _make_token_callback(callback)
    pcm_ptr, pcm_size = _prepare_pcm(pcm_data)
    rc = _lib.cactus_transcribe(
        model, _enc(audio_path), _enc(prompt), buf, len(buf),
        _to_json(options), cb, None, pcm_ptr, pcm_size,
    )
    if rc < 0:
        raise RuntimeError(_err("Transcription failed"))
    return _from_json(buf)


def cactus_stream_transcribe_start(model, options=None):
    """Open a streaming transcription session on a Whisper or Parakeet TDT model.

    Args:
        model:   Model handle.
        options: Optional dict forwarded to the underlying transcribe call
                 (e.g. {"language": "en", "max_tokens": 256}); chunking is
                 handled internally.

    Returns:
        An opaque stream handle. Feed audio with cactus_stream_transcribe_process
        and finish with cactus_stream_transcribe_stop.
    """
    stream = _lib.cactus_stream_transcribe_start(model, _to_json(options))
    if not stream:
        raise RuntimeError(_err("Failed to start streaming transcription"))
    return stream


def cactus_stream_transcribe_process(stream, pcm_data):
    """Feed the next slice of audio to a streaming session.

    Args:
        stream:   Stream handle from cactus_stream_transcribe_start.
        pcm_data: 16 kHz mono 16-bit PCM audio bytes for this chunk.

    Returns:
        A dict with "confirmed" (newly finalized text, append it), "pending" (volatile tail,
        replace it each call), and per-call stats ("decode_tps",
        "total_time_ms", "time_to_first_token_ms", "decode_tokens").
    """
    buf = ctypes.create_string_buffer(1 << 16)
    pcm_ptr, pcm_size = _prepare_pcm(pcm_data)
    rc = _lib.cactus_stream_transcribe_process(stream, pcm_ptr, pcm_size, buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Stream transcription failed"))
    return _from_json(buf)


def cactus_stream_transcribe_stop(stream):
    """Flush buffered audio, return the final text, and destroy the session.

    Args:
        stream: Stream handle from cactus_stream_transcribe_start.

    Returns:
        A dict {"success": True, "confirmed": <remaining finalized text>, "pending": ""}. The handle is invalid
        after this call.
    """
    buf = ctypes.create_string_buffer(1 << 16)
    rc = _lib.cactus_stream_transcribe_stop(stream, buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Stream transcription stop failed"))
    return _from_json(buf)


def cactus_detect_language(model, audio_path, options=None, pcm_data=None):
    """Detect the spoken language in audio.

    Returns:
        A dict with detected language info.
    """
    buf = ctypes.create_string_buffer(65536)
    pcm_ptr, pcm_size = _prepare_pcm(pcm_data)
    rc = _lib.cactus_detect_language(
        model, _enc(audio_path), buf, len(buf), _to_json(options), pcm_ptr, pcm_size,
    )
    if rc < 0:
        raise RuntimeError(_err("Detect language failed"))
    return _from_json(buf)


def cactus_preprocess_audio_features(audio_path, model_type, mel_bins, capacity):
    """Compute mel spectrogram features from an audio file.

    Returns:
        A tuple (values, mel_bins, frames) where values is a list of floats.
    """
    if not hasattr(_lib, "cactus_preprocess_audio_features"):
        raise RuntimeError("cactus_preprocess_audio_features is unavailable; rebuild with cactus build --python")
    buf = (ctypes.c_float * int(capacity))()
    feature_count = ctypes.c_size_t()
    out_mels = ctypes.c_size_t()
    out_frames = ctypes.c_size_t()
    rc = _lib.cactus_preprocess_audio_features(
        _enc(audio_path), _enc(model_type),
        ctypes.c_size_t(int(mel_bins)),
        buf, ctypes.sizeof(buf),
        ctypes.byref(feature_count), ctypes.byref(out_mels), ctypes.byref(out_frames),
    )
    if rc < 0:
        raise RuntimeError(_err("Audio feature preprocessing failed"))
    return list(buf[:feature_count.value]), int(out_mels.value), int(out_frames.value)


# ── Embeddings ───────────────────────────────────────────────────────


def cactus_embed(model, text, normalize=True):
    """Compute a text embedding.

    Args:
        model:     Model handle.
        text:      The text to embed.
        normalize: Whether to L2-normalize the embedding (default True).

    Returns:
        A list of floats (the embedding vector).
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_embed(model, _enc(text), buf, ctypes.sizeof(buf), ctypes.byref(dim), normalize)
    if rc < 0:
        raise RuntimeError(_err("Embedding failed"))
    return list(buf[:dim.value])


def cactus_image_embed(model, image_path):
    """Compute an image embedding. Returns a list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_image_embed(model, _enc(image_path), buf, ctypes.sizeof(buf), ctypes.byref(dim))
    if rc < 0:
        raise RuntimeError(_err("Image embedding failed"))
    return list(buf[:dim.value])


def cactus_audio_embed(model, audio_path):
    """Compute an audio embedding. Returns a list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_audio_embed(model, _enc(audio_path), buf, ctypes.sizeof(buf), ctypes.byref(dim))
    if rc < 0:
        raise RuntimeError(_err("Audio embedding failed"))
    return list(buf[:dim.value])


# ── Tokenization ─────────────────────────────────────────────────────


def cactus_tokenize(model, text):
    """Tokenize text into token IDs. Returns a list of ints."""
    max_tokens = 8192
    arr = (ctypes.c_uint32 * max_tokens)()
    n = ctypes.c_size_t(0)
    rc = _lib.cactus_tokenize(model, _enc(text), arr, max_tokens, ctypes.byref(n))
    if rc < 0:
        raise RuntimeError(_err("Tokenization failed"))
    return list(arr[:n.value])


def cactus_render_prompt(model, messages, options=None, tools=None):
    """Render the chat-template prompt string for messages without generating."""
    if not hasattr(_lib, "cactus_render_prompt"):
        raise RuntimeError("cactus_render_prompt is unavailable; rebuild with cactus build --python")
    buf = ctypes.create_string_buffer(1 << 20)
    rc = _lib.cactus_render_prompt(
        model, _to_json(messages), _to_json(options), _to_json(tools), buf, len(buf),
    )
    if rc < 0:
        raise RuntimeError(_err("Prompt rendering failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_decode_tokens(model, tokens, temperature=0.0, top_p=1.0, top_k=1):
    """Decode the next token from a sequence. Returns the next token ID as an int."""
    if not tokens:
        raise ValueError("tokens must be non-empty")
    arr = (ctypes.c_uint32 * len(tokens))(*[int(t) for t in tokens])
    out = ctypes.c_uint32(0)
    rc = _lib.cactus_decode_tokens(
        model, arr, len(tokens),
        ctypes.c_float(float(temperature)),
        ctypes.c_float(float(top_p)),
        ctypes.c_size_t(int(top_k)),
        ctypes.byref(out),
    )
    if rc < 0:
        raise RuntimeError(_err("Token decode failed"))
    return int(out.value)


def cactus_score_window(model, tokens, start, end, context):
    """Score a window of tokens for log-probabilities.

    Returns:
        A dict with token-level log-probability scores.
    """
    buf = ctypes.create_string_buffer(65536)
    arr = (ctypes.c_uint32 * len(tokens))(*tokens)
    rc = _lib.cactus_score_window(model, arr, len(tokens), start, end, context, buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Score window failed"))
    return _from_json(buf)


# ── RAG ──────────────────────────────────────────────────────────────


def cactus_rag_query(model, query, top_k=5):
    """Query the RAG corpus for relevant documents.

    Returns:
        A dict with ranked results.
    """
    buf = ctypes.create_string_buffer(65536)
    rc = _lib.cactus_rag_query(model, _enc(query), buf, len(buf), top_k)
    if rc < 0:
        raise RuntimeError(_err("RAG query failed"))
    return _from_json(buf)


# ── Vector index ─────────────────────────────────────────────────────


def cactus_index_init(index_dir, embedding_dim):
    """Create a vector index for semantic search.

    Args:
        index_dir:     Directory to persist the index.
        embedding_dim: Dimensionality of the embedding vectors.

    Returns:
        An opaque index handle. Call cactus_index_destroy() when done.
    """
    handle = _lib.cactus_index_init(_enc(index_dir), embedding_dim)
    if not handle:
        raise RuntimeError(_err("Failed to initialize index"))
    return handle


def cactus_index_add(index, ids, documents, metadatas=None, embeddings=None):
    """Add documents with embeddings to the index.

    Args:
        index:      Index handle from cactus_index_init().
        ids:        List of integer document IDs.
        documents:  List of document strings.
        metadatas:  Optional list of metadata strings (one per document).
        embeddings: List of embedding vectors (list of floats each).
    """
    count = len(ids)
    if len(documents) != count:
        raise ValueError(f"documents length ({len(documents)}) must match ids length ({count})")
    if embeddings and len(embeddings) != count:
        raise ValueError(f"embeddings length ({len(embeddings)}) must match ids length ({count})")
    if metadatas and len(metadatas) != count:
        raise ValueError(f"metadatas length ({len(metadatas)}) must match ids length ({count})")
    embedding_dim = len(embeddings[0]) if embeddings else 0
    if embeddings and any(len(e) != embedding_dim for e in embeddings):
        raise ValueError("all embedding vectors must share the same dimension")

    ids_arr = (ctypes.c_int * count)(*ids)
    docs_arr = (ctypes.c_char_p * count)()
    for i, doc in enumerate(documents):
        docs_arr[i] = _enc(doc)

    meta_arr = None
    if metadatas:
        meta_arr = (ctypes.c_char_p * count)()
        for i, meta in enumerate(metadatas):
            meta_arr[i] = _enc(meta)

    emb_ptrs = (ctypes.POINTER(ctypes.c_float) * count)()
    emb_arrays = []
    for i, emb in enumerate(embeddings or []):
        arr = (ctypes.c_float * len(emb))(*emb)
        emb_arrays.append(arr)
        emb_ptrs[i] = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))

    rc = _lib.cactus_index_add(index, ids_arr, docs_arr, meta_arr, emb_ptrs, count, embedding_dim)
    if rc < 0:
        raise RuntimeError(_err("Failed to add to index"))


def cactus_index_delete(index, ids):
    """Remove documents from the index by ID."""
    ids_arr = (ctypes.c_int * len(ids))(*ids)
    rc = _lib.cactus_index_delete(index, ids_arr, len(ids))
    if rc < 0:
        raise RuntimeError(_err("Failed to delete from index"))


def cactus_index_query(index, embedding, options=None):
    """Query the index by embedding vector.

    Returns:
        A dict with "results" — a list of {"id": int, "score": float}.
    """
    result_capacity = 1000
    embedding_dim = len(embedding)
    emb_arr = (ctypes.c_float * embedding_dim)(*embedding)
    emb_ptr = ctypes.cast(emb_arr, ctypes.POINTER(ctypes.c_float))
    id_buffer = (ctypes.c_int * result_capacity)()
    score_buffer = (ctypes.c_float * result_capacity)()
    id_ptr = ctypes.cast(id_buffer, ctypes.POINTER(ctypes.c_int))
    score_ptr = ctypes.cast(score_buffer, ctypes.POINTER(ctypes.c_float))
    id_size = ctypes.c_size_t(result_capacity)
    score_size = ctypes.c_size_t(result_capacity)
    rc = _lib.cactus_index_query(
        index, ctypes.pointer(emb_ptr), 1, embedding_dim, _to_json(options),
        ctypes.pointer(id_ptr), ctypes.byref(id_size),
        ctypes.pointer(score_ptr), ctypes.byref(score_size),
    )
    if rc < 0:
        raise RuntimeError(_err("Index query failed"))
    n = id_size.value
    return {"results": [{"id": int(id_buffer[i]), "score": float(score_buffer[i])} for i in range(n)]}


_INDEX_DOC_BUF_SIZE = 4096
_INDEX_EMB_BUF_SIZE = 4096


def cactus_index_get(index, ids):
    """Retrieve documents by ID from the index.

    Returns:
        A dict with "results" — a list of {"document", "metadata", "embedding"}.
    """
    count = len(ids)
    ids_arr = (ctypes.c_int * count)(*ids)
    doc_raw = [ctypes.create_string_buffer(_INDEX_DOC_BUF_SIZE) for _ in range(count)]
    doc_ptrs = (ctypes.c_char_p * count)()
    doc_sizes = (ctypes.c_size_t * count)()
    meta_raw = [ctypes.create_string_buffer(_INDEX_DOC_BUF_SIZE) for _ in range(count)]
    meta_ptrs = (ctypes.c_char_p * count)()
    meta_sizes = (ctypes.c_size_t * count)()
    emb_raw = [(ctypes.c_float * _INDEX_EMB_BUF_SIZE)() for _ in range(count)]
    emb_ptrs = (ctypes.POINTER(ctypes.c_float) * count)()
    emb_sizes = (ctypes.c_size_t * count)()
    for i in range(count):
        doc_ptrs[i] = ctypes.cast(doc_raw[i], ctypes.c_char_p)
        doc_sizes[i] = _INDEX_DOC_BUF_SIZE
        meta_ptrs[i] = ctypes.cast(meta_raw[i], ctypes.c_char_p)
        meta_sizes[i] = _INDEX_DOC_BUF_SIZE
        emb_ptrs[i] = ctypes.cast(emb_raw[i], ctypes.POINTER(ctypes.c_float))
        emb_sizes[i] = _INDEX_EMB_BUF_SIZE
    rc = _lib.cactus_index_get(
        index, ids_arr, count,
        doc_ptrs, doc_sizes, meta_ptrs, meta_sizes, emb_ptrs, emb_sizes,
    )
    if rc < 0:
        raise RuntimeError(_err("Failed to get from index"))
    results = []
    for i in range(count):
        doc = doc_raw[i].value.decode("utf-8", errors="ignore")
        meta = meta_raw[i].value.decode("utf-8", errors="ignore")
        emb = list(emb_raw[i][:emb_sizes[i]])
        results.append({"document": doc, "metadata": meta or None, "embedding": emb})
    return {"results": results}


def cactus_index_compact(index):
    """Compact the index storage on disk to reclaim space from deletions."""
    rc = _lib.cactus_index_compact(index)
    if rc < 0:
        raise RuntimeError(_err("Failed to compact index"))


def cactus_index_destroy(index):
    """Free all resources associated with an index handle. Safe to call on a
    null/already-destroyed handle (no-op)."""
    if not index:
        return
    _lib.cactus_index_destroy(index)


# ── Logging ──────────────────────────────────────────────────────────


def cactus_log_set_level(level):
    """Set the log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE."""
    _lib.cactus_log_set_level(level)


_log_callback_ref = None


def cactus_log_set_callback(callback):
    """Set a log callback: callback(level, component, message). Pass None to clear."""
    global _log_callback_ref
    if callback is None:
        _log_callback_ref = None
        _lib.cactus_log_set_callback(LogCallback(), None)
        return

    def _bridge(level, component_bytes, message_bytes, _):
        callback(
            level,
            component_bytes.decode("utf-8", errors="ignore") if component_bytes else "",
            message_bytes.decode("utf-8", errors="ignore") if message_bytes else "",
        )

    _log_callback_ref = LogCallback(_bridge)
    _lib.cactus_log_set_callback(_log_callback_ref, None)


class Graph:
    INT8 = 0
    FP16 = 1
    FP32 = 2
    CQ1 = 3
    CQ2 = 4
    CQ3 = 5
    CQ4 = 6
    CPU = 0
    NPU = 1
    METAL = 2
    ACT_SILU = 0
    ACT_GELU = 1
    ACT_GELU_ERF = 2
    ACT_RELU = 3
    ACT_SIGMOID = 4
    ACT_TANH = 5

    def __init__(self):
        self.h = _lib.cactus_graph_create()
        if not self.h:
            raise RuntimeError(_err("cactus_graph_create failed"))
    
    def save(self, filename):
        rc = _lib.cactus_graph_save(self.h, str(filename).encode())
        if rc != 0:
            raise RuntimeError(_err("graph_save failed"))

    @classmethod
    def load(cls, filename):
        h = _lib.cactus_graph_load(str(filename).encode())
        if not h:
            raise RuntimeError(_err("cactus_graph_load failed"))
        obj = cls.__new__(cls)
        obj.h = h
        return obj

    def __del__(self):
        h = getattr(self, "h", None)
        if h:
            _lib.cactus_graph_destroy(h)
            self.h = None

    def input(self, shape, dtype=FP16, dynamic_dims=None):
        shape = tuple(int(x) for x in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_input(self.h, arr, len(shape), int(dtype), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_input failed"))
        tensor = self._tensor_from_node(out.value)
        if dynamic_dims is not None:
            mask = (ctypes.c_uint8 * len(dynamic_dims))(*[1 if int(d) else 0 for d in dynamic_dims])
            rc = _lib.cactus_graph_set_input_dynamic_dims(self.h, cactus_node_t(tensor.id), mask, len(dynamic_dims))
            if rc != 0:
                raise RuntimeError(_err("graph_set_input_dynamic_dims failed"))
        return tensor

    def set_input(self, tensor, data, dtype=None):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if tensor.g is not self:
            raise ValueError("tensor belongs to a different graph")
        target_dtype = int(tensor.dtype if dtype is None else dtype)
        arr = self._coerce_input_array(data, target_dtype)
        info = self._get_output_info(tensor.id)
        expected_shape = tuple(int(x) for x in info["shape"])
        expected_num_elements = int(info["num_elements"])
        expected_byte_size = int(info["byte_size"])
        if tuple(int(x) for x in arr.shape) != expected_shape:
            raise ValueError(
                "graph input shape mismatch for node "
                f"{tensor.id}: expected {expected_shape}, got {tuple(int(x) for x in arr.shape)}"
            )
        if int(arr.size) != expected_num_elements:
            raise ValueError(
                "graph input element-count mismatch for node "
                f"{tensor.id}: expected {expected_num_elements}, got {int(arr.size)}"
            )
        if int(arr.nbytes) != expected_byte_size:
            raise ValueError(
                "graph input byte-size mismatch for node "
                f"{tensor.id}: expected {expected_byte_size}, got {int(arr.nbytes)}"
            )
        rc = _lib.cactus_graph_set_input(
            self.h,
            cactus_node_t(tensor.id),
            arr.ctypes.data_as(ctypes.c_void_p),
            target_dtype,
        )
        if rc != 0:
            raise RuntimeError(_err("graph_set_input failed"))

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
            raise RuntimeError(_err("graph_set_external_input failed"))

    def set_runtime_input_shape(self, tensor, shape):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if tensor.g is not self:
            raise ValueError("tensor belongs to a different graph")
        shape = tuple(int(x) for x in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        rc = _lib.cactus_graph_set_runtime_input_shape(self.h, cactus_node_t(tensor.id), arr, len(shape))
        if rc != 0:
            raise RuntimeError(_err("graph_set_runtime_input_shape failed"))

    def mark_embedded_input(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if tensor.g is not self:
            raise ValueError("tensor belongs to a different graph")
        rc = _lib.cactus_graph_mark_embedded_input(self.h, cactus_node_t(tensor.id))
        if rc != 0:
            raise RuntimeError(_err("graph_mark_embedded_input failed"))

    def hard_reset(self):
        rc = _lib.cactus_graph_hard_reset(self.h)
        if rc != 0:
            raise RuntimeError(_err("graph_hard_reset failed"))

    def execute(self):
        rc = _lib.cactus_graph_execute(self.h)
        if rc != 0:
            raise RuntimeError(_err("graph_execute failed"))

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

    def not_equal(self, a, b):
        return self._binary("cactus_graph_not_equal", a, b)

    def abs(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_abs(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_abs failed"))
        return self._tensor_from_node(out.value)

    def pow(self, x, exponent):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_pow(self.h, cactus_node_t(x.id), ctypes.c_float(float(exponent)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_pow failed"))
        return self._tensor_from_node(out.value)

    def precision_cast(self, x, dtype):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_precision_cast(self.h, cactus_node_t(x.id), int(dtype), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_precision_cast failed"))
        return self._tensor_from_node(out.value)

    def quantize_activations(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_quantize_activations(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_quantize_activations failed"))
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

    def scalar_floor_divide(self, x, value):
        return self._scalar("cactus_graph_scalar_floor_divide", x, value)

    def scalar_not_equal(self, x, value):
        return self._scalar("cactus_graph_scalar_not_equal", x, value)

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

    def clamp(self, x, lo, hi):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_clamp(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_float(float(lo)),
            ctypes.c_float(float(hi)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_clamp failed"))
        return self._tensor_from_node(out.value)

    def masked_select_prefix(self, x, mask):
        return self._binary("cactus_graph_masked_select_prefix", x, mask)

    def masked_scatter(self, x, mask, source):
        x = self._ensure_tensor(x)
        mask = self._ensure_tensor(mask)
        source = self._ensure_tensor(source)
        out = cactus_node_t()
        rc = _lib.cactus_graph_masked_scatter(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(mask.id),
            cactus_node_t(source.id),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_masked_scatter failed"))
        return self._tensor_from_node(out.value)

    def view(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_view(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_view failed"))
        return self._tensor_from_node(out.value)

    def reshape(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_reshape(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_reshape failed"))
        return self._tensor_from_node(out.value)

    def expand(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_expand(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_expand failed"))
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
            raise RuntimeError(_err("graph_flatten failed"))
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
            raise RuntimeError(_err("graph_slice failed"))
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
            raise RuntimeError(_err("graph_index failed"))
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
            raise RuntimeError(_err("graph_transpose failed"))
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
            raise RuntimeError(_err("graph_transpose_n failed"))
        return self._tensor_from_node(out.value)

    def matmul(self, a, b, pretransposed_rhs=False, backend=CPU, output_dtype=None):
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
            raise RuntimeError(_err("graph_matmul failed"))
        return self._tensor_from_node(out.value)

    def gather(self, tensor, indices):
        tensor = self._ensure_tensor(tensor)
        indices = self._ensure_tensor(indices)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gather(
            self.h,
            cactus_node_t(tensor.id),
            cactus_node_t(indices.id),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_gather failed"))
        return self._tensor_from_node(out.value)

    def embedding_from_tensor(self, embedding_tensor, indices):
        embedding_tensor = self._ensure_tensor(embedding_tensor)
        indices = self._ensure_tensor(indices)
        out = cactus_node_t()
        rc = _lib.cactus_graph_embedding_from_tensor(
            self.h, cactus_node_t(embedding_tensor.id), cactus_node_t(indices.id), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_embedding_from_tensor failed"))
        return self._tensor_from_node(out.value)

    def embedding_from_file(self, filename, indices):
        indices = self._ensure_tensor(indices)
        out = cactus_node_t()
        rc = _lib.cactus_graph_embedding_from_file(self.h, str(filename).encode(), cactus_node_t(indices.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_embedding_from_file failed"))
        return self._tensor_from_node(out.value)

    def mmap_embeddings(self, filename):
        out = cactus_node_t()
        rc = _lib.cactus_graph_mmap_embeddings(self.h, str(filename).encode(), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_mmap_embeddings failed"))
        return self._tensor_from_node(out.value)

    def mmap_weights(self, filename):
        out = cactus_node_t()
        rc = _lib.cactus_graph_mmap_weights(self.h, str(filename).encode(), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_mmap_weights failed"))
        return self._tensor_from_node(out.value)

    def bind_mmap_weights(self, tensor, filename):
        tensor = self._ensure_tensor(tensor)
        rc = _lib.cactus_graph_bind_mmap_weights(self.h, cactus_node_t(tensor.id), str(filename).encode())
        if rc != 0:
            raise RuntimeError(_err("graph_bind_mmap_weights failed"))

    def bilinear_interpolation(self, pos_embeds, dst_height, dst_width):
        pos_embeds = self._ensure_tensor(pos_embeds)
        out = cactus_node_t()
        rc = _lib.cactus_graph_bilinear_interpolation(
            self.h, cactus_node_t(pos_embeds.id), ctypes.c_size_t(int(dst_height)), ctypes.c_size_t(int(dst_width)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_bilinear_interpolation failed"))
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
            raise RuntimeError(_err("graph_set_grouped_scales failed"))

    def set_interleaved(self, tensor, interleaved=True, original_n=0):
        tensor = self._ensure_tensor(tensor)
        rc = _lib.cactus_graph_set_interleaved(
            self.h, cactus_node_t(tensor.id), ctypes.c_bool(bool(interleaved)), ctypes.c_size_t(int(original_n))
        )
        if rc != 0:
            raise RuntimeError(_err("graph_set_interleaved failed"))

    def release_weight_pages(self, tensor):
        tensor = self._ensure_tensor(tensor)
        rc = _lib.cactus_graph_release_weight_pages(self.h, cactus_node_t(tensor.id))
        if rc != 0:
            raise RuntimeError(_err("graph_release_weight_pages failed"))

    def prefetch_weight_pages(self, tensor):
        tensor = self._ensure_tensor(tensor)
        rc = _lib.cactus_graph_prefetch_weight_pages(self.h, cactus_node_t(tensor.id))
        if rc != 0:
            raise RuntimeError(_err("graph_prefetch_weight_pages failed"))

    def release_all_weight_pages(self):
        rc = _lib.cactus_graph_release_all_weight_pages(self.h)
        if rc != 0:
            raise RuntimeError(_err("graph_release_all_weight_pages failed"))

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
            raise RuntimeError(_err("graph_concat failed"))
        return self._tensor_from_node(out.value)

    def cat(self, tensors, axis=0):
        tensors = [self._ensure_tensor(t) for t in tensors]
        if not tensors:
            raise ValueError("cat requires at least one tensor")
        if len(tensors) == 1:
            return tensors[0]
        ids = (cactus_node_t * len(tensors))(*(cactus_node_t(t.id) for t in tensors))
        out = cactus_node_t()
        rc = _lib.cactus_graph_cat(
            self.h, ids, ctypes.c_size_t(len(tensors)),
            ctypes.c_int32(int(axis)), ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_cat failed"))
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
            raise RuntimeError(_err("graph_group_norm failed"))
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
            raise RuntimeError(_err("graph_layer_norm failed"))
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
            raise RuntimeError(_err("graph_batchnorm failed"))
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
            raise RuntimeError(_err("graph_rms_norm failed"))
        return self._tensor_from_node(out.value)

    def topk(self, x, k):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_topk(self.h, cactus_node_t(x.id), ctypes.c_size_t(int(k)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_topk failed"))
        return self._tensor_from_node(out.value)

    def rope(self, x, theta, position_offset=0, backend=CPU):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rope(
            self.h, cactus_node_t(x.id), ctypes.c_float(float(theta)), ctypes.c_size_t(int(position_offset)),
            ctypes.c_int32(int(backend)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_rope failed"))
        return self._tensor_from_node(out.value)

    def rope_gptj(self, x, theta, position_offset=0, rot_dim=0, backend=CPU):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rope_gptj(
            self.h, cactus_node_t(x.id), ctypes.c_float(float(theta)), ctypes.c_size_t(int(position_offset)),
            ctypes.c_size_t(int(rot_dim)), ctypes.c_int32(int(backend)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_rope_gptj failed"))
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

    def cumsum(self, x, axis):
        return self._reduce("cactus_graph_cumsum", x, axis)
    
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
            raise RuntimeError(_err("graph_softmax failed"))
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
            raise RuntimeError(_err("graph_attention failed"))
        return self._tensor_from_node(out.value)

    def kv_cache_state(self, max_seq_len, num_kv_heads, head_dim, window_size=0, sink_size=0, num_slots=1):
        out = cactus_node_t()
        rc = _lib.cactus_graph_kv_cache_state(
            self.h,
            ctypes.c_size_t(int(max_seq_len)),
            ctypes.c_size_t(int(num_kv_heads)),
            ctypes.c_size_t(int(head_dim)),
            ctypes.c_size_t(int(window_size)),
            ctypes.c_size_t(int(sink_size)),
            ctypes.c_size_t(int(num_slots)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_kv_cache_state failed"))
        return self._tensor_from_node(out.value)

    def kv_cache_append(self, new_kv, cache_state, window_size=0, sink_size=0):
        new_kv = self._ensure_tensor(new_kv)
        cache_state = self._ensure_tensor(cache_state)
        out = cactus_node_t()
        rc = _lib.cactus_graph_kv_cache_append(
            self.h,
            cactus_node_t(new_kv.id),
            cactus_node_t(cache_state.id),
            ctypes.c_size_t(int(window_size)),
            ctypes.c_size_t(int(sink_size)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_kv_cache_append failed"))
        return self._tensor_from_node(out.value)

    def attention_cached(
        self,
        query,
        key_new,
        value_new,
        k_cache_state,
        v_cache_state,
        scale,
        position_offset=0,
        window_size=0,
        v_head_dim=0,
    ):
        query = self._ensure_tensor(query)
        key_new = self._ensure_tensor(key_new)
        value_new = self._ensure_tensor(value_new)
        k_cache_state = self._ensure_tensor(k_cache_state)
        v_cache_state = self._ensure_tensor(v_cache_state)
        out = cactus_node_t()
        rc = _lib.cactus_graph_attention_cached(
            self.h,
            cactus_node_t(query.id),
            cactus_node_t(key_new.id),
            cactus_node_t(value_new.id),
            cactus_node_t(k_cache_state.id),
            cactus_node_t(v_cache_state.id),
            ctypes.c_float(float(scale)),
            ctypes.c_size_t(int(position_offset)),
            ctypes.c_size_t(int(window_size)),
            ctypes.c_size_t(int(v_head_dim)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_attention_cached failed"))
        return self._tensor_from_node(out.value)

    def conv_cache_state(self, window_size, hidden_dim):
        out = cactus_node_t()
        rc = _lib.cactus_graph_conv_cache_state(
            self.h,
            ctypes.c_size_t(int(window_size)),
            ctypes.c_size_t(int(hidden_dim)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_conv_cache_state failed"))
        return self._tensor_from_node(out.value)

    def conv_cache_append(self, new_data, cache_state):
        new_data = self._ensure_tensor(new_data)
        cache_state = self._ensure_tensor(cache_state)
        out = cactus_node_t()
        rc = _lib.cactus_graph_conv_cache_append(
            self.h,
            cactus_node_t(new_data.id),
            cactus_node_t(cache_state.id),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_conv_cache_append failed"))
        return self._tensor_from_node(out.value)

    def conv_cache_initialize(self, rows, cache_state):
        rows = self._ensure_tensor(rows)
        cache_state = self._ensure_tensor(cache_state)
        out = cactus_node_t()
        rc = _lib.cactus_graph_conv_cache_initialize(
            self.h,
            cactus_node_t(rows.id),
            cactus_node_t(cache_state.id),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_conv_cache_initialize failed"))
        return self._tensor_from_node(out.value)

    def recurrent_cache_state(self, shape, dtype=FP16):
        shape = tuple(int(x) for x in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_recurrent_cache_state(
            self.h,
            arr,
            ctypes.c_size_t(len(shape)),
            ctypes.c_int(int(dtype)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_recurrent_cache_state failed"))
        return self._tensor_from_node(out.value)

    def recurrent_cache_write(self, new_value, cache_input):
        new_value = self._ensure_tensor(new_value)
        cache_input = self._ensure_tensor(cache_input)
        out = cactus_node_t()
        rc = _lib.cactus_graph_recurrent_cache_write(
            self.h,
            cactus_node_t(new_value.id),
            cactus_node_t(cache_input.id),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_recurrent_cache_write failed"))
        return self._tensor_from_node(out.value)

    def rel_pos_bias(self, query, relative_key, scale):
        query = self._ensure_tensor(query)
        relative_key = self._ensure_tensor(relative_key)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rel_pos_bias(
            self.h, cactus_node_t(query.id), cactus_node_t(relative_key.id), ctypes.c_float(float(scale)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_rel_pos_bias failed"))
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
            raise RuntimeError(_err("graph_relu failed"))
        return self._tensor_from_node(out.value)

    def silu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_silu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_silu failed"))
        return self._tensor_from_node(out.value)

    def gelu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gelu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_gelu failed"))
        return self._tensor_from_node(out.value)

    def gelu_erf(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gelu_erf(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_gelu_erf failed"))
        return self._tensor_from_node(out.value)

    def sigmoid(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_sigmoid(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_sigmoid failed"))
        return self._tensor_from_node(out.value)

    def tanh(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_tanh(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_tanh failed"))
        return self._tensor_from_node(out.value)

    def glu(self, x, axis=-1):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_glu(self.h, cactus_node_t(x.id), ctypes.c_int32(int(axis)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_glu failed"))
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

    def conv2d_k3s1p1(self, x, weight, bias=None):
        return self._conv_with_optional_bias("cactus_graph_conv2d_k3s1p1", x, weight, bias)

    def conv2d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return self._conv_with_optional_bias(
            "cactus_graph_conv2d",
            x,
            weight,
            bias,
            ctypes.c_size_t(int(stride)),
            ctypes.c_size_t(int(padding)),
            ctypes.c_size_t(int(dilation)),
            ctypes.c_size_t(int(groups)),
        )

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
            raise RuntimeError(_err("graph_lstm_cell failed"))
        return self._tensor_from_node(out.value)

    def gated_deltanet_decode(self, query, key, value, gate_log, beta, initial_state, scale):
        tensors = [self._ensure_tensor(t) for t in (query, key, value, gate_log, beta, initial_state)]
        out = cactus_node_t()
        rc = _lib.cactus_graph_gated_deltanet_decode(
            self.h, *(cactus_node_t(t.id) for t in tensors), ctypes.c_float(float(scale)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_gated_deltanet_decode failed"))
        return self._tensor_from_node(out.value)

    def gated_deltanet_prefill(self, query, key, value, gate_log, beta, initial_state, chunk_size, scale):
        tensors = [self._ensure_tensor(t) for t in (query, key, value, gate_log, beta, initial_state)]
        out = cactus_node_t()
        rc = _lib.cactus_graph_gated_deltanet_prefill(
            self.h, *(cactus_node_t(t.id) for t in tensors), ctypes.c_size_t(int(chunk_size)), ctypes.c_float(float(scale)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_gated_deltanet_prefill failed"))
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
            raise RuntimeError(_err("graph_stft failed"))
        return self._tensor_from_node(out.value)

    def altup_predict(self, coefs, streams):
        coefs = self._ensure_tensor(coefs)
        streams = [self._ensure_tensor(t) for t in streams]
        ids = (cactus_node_t * len(streams))(*(cactus_node_t(t.id) for t in streams))
        out = cactus_node_t()
        rc = _lib.cactus_graph_altup_predict(self.h, cactus_node_t(coefs.id), ids, ctypes.c_size_t(len(streams)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_altup_predict failed"))
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
            raise RuntimeError(_err("graph_altup_correct failed"))
        return self._tensor_from_node(out.value)

    def gaussian_topk(self, x, ppf):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gaussian_topk(self.h, cactus_node_t(x.id), ctypes.c_float(float(ppf)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_gaussian_topk failed"))
        return self._tensor_from_node(out.value)

    def moe_layer_gated(self, hidden, routing_probs, topk_indices, w1_weights, w3_weights, w2_weights,
                        num_experts, num_experts_per_tok, normalize_routing=True, epsilon=1e-6, routed_scaling_factor=1.0, activation=0):
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
            ctypes.c_float(float(routed_scaling_factor)), ctypes.c_int32(int(activation)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_moe_layer_gated failed"))
        return self._tensor_from_node(out.value)

    def dense_mlp_tq_fused(self, hidden, gate_weight, up_weight, down_weight, product_scale=1.0):
        hidden = self._ensure_tensor(hidden)
        gate_weight = self._ensure_tensor(gate_weight)
        up_weight = self._ensure_tensor(up_weight)
        down_weight = self._ensure_tensor(down_weight)
        out = cactus_node_t()
        rc = _lib.cactus_graph_dense_mlp_tq_fused(
            self.h,
            cactus_node_t(hidden.id),
            cactus_node_t(gate_weight.id),
            cactus_node_t(up_weight.id),
            cactus_node_t(down_weight.id),
            ctypes.c_float(float(product_scale)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_dense_mlp_tq_fused failed"))
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
            raise RuntimeError(_err("graph_moe_layer_ungated failed"))
        return self._tensor_from_node(out.value)

    def sample(self, logits, temperature=0.6, top_p=0.95, top_k=20):
        logits = self._ensure_tensor(logits)
        out = cactus_node_t()
        rc = _lib.cactus_graph_sample(
            self.h, cactus_node_t(logits.id), ctypes.c_float(float(temperature)),
            ctypes.c_float(float(top_p)), ctypes.c_size_t(int(top_k)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_sample failed"))
        return self._tensor_from_node(out.value)

    def scatter_topk(self, indices, values, num_classes):
        indices = self._ensure_tensor(indices)
        values = self._ensure_tensor(values)
        out = cactus_node_t()
        rc = _lib.cactus_graph_scatter_topk(
            self.h, cactus_node_t(indices.id), cactus_node_t(values.id), ctypes.c_size_t(int(num_classes)), ctypes.byref(out)
        )
        if rc != 0:
            raise RuntimeError(_err("graph_scatter_topk failed"))
        return self._tensor_from_node(out.value)

    def persistent(self, source_node):
        source_node = self._ensure_tensor(source_node)
        out = cactus_node_t()
        rc = _lib.cactus_graph_persistent(self.h, cactus_node_t(source_node.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_persistent failed"))
        return self._tensor_from_node(out.value)

    def is_populated(self, persistent_node):
        persistent_node = self._ensure_tensor(persistent_node)
        out_is_populated = ctypes.c_int32()
        rc = _lib.cactus_graph_is_populated(self.h, cactus_node_t(persistent_node.id), ctypes.byref(out_is_populated))
        if rc != 0:
            raise RuntimeError(_err("graph_is_populated failed"))
        return bool(out_is_populated.value)

    def invalidate_persistent(self, persistent_node):
        persistent_node = self._ensure_tensor(persistent_node)
        rc = _lib.cactus_graph_invalidate_persistent(self.h, cactus_node_t(persistent_node.id))
        if rc != 0:
            raise RuntimeError(_err("graph_invalidate_persistent failed"))

    # ── Audio / signal processing ───────────────────────────────────

    def rfft(self, x):
        """Real-to-complex FFT."""
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rfft(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(_err("graph_rfft failed"))
        return self._tensor_from_node(out.value)

    def irfft(self, x, output_length):
        """Complex-to-real inverse FFT."""
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_irfft(
            self.h, cactus_node_t(x.id),
            ctypes.c_size_t(int(output_length)), ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_irfft failed"))
        return self._tensor_from_node(out.value)

    def mel_filter_bank(self, num_frequency_bins, num_mel_filters,
                        min_frequency, max_frequency, sampling_rate,
                        norm_type=0, scale_type=0):
        """Generate a mel-scale filter bank tensor."""
        out = cactus_node_t()
        rc = _lib.cactus_graph_mel_filter_bank(
            self.h,
            ctypes.c_size_t(int(num_frequency_bins)),
            ctypes.c_size_t(int(num_mel_filters)),
            ctypes.c_float(float(min_frequency)),
            ctypes.c_float(float(max_frequency)),
            ctypes.c_size_t(int(sampling_rate)),
            ctypes.c_int(int(norm_type)),
            ctypes.c_int(int(scale_type)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_mel_filter_bank failed"))
        return self._tensor_from_node(out.value)

    def spectrogram(self, waveform, mel_filters, frame_length, hop_length,
                    fft_length, power=2.0, center=True, pad_mode=0,
                    mel_floor=1e-10, log_mel_mode=0, dither=0.0,
                    preemphasis=0.0, remove_dc_offset=False):
        """Compute a spectrogram from a waveform tensor."""
        waveform = self._ensure_tensor(waveform)
        mel_filters = self._ensure_tensor(mel_filters)
        out = cactus_node_t()
        rc = _lib.cactus_graph_spectrogram(
            self.h,
            cactus_node_t(waveform.id),
            cactus_node_t(mel_filters.id),
            ctypes.c_size_t(int(frame_length)),
            ctypes.c_size_t(int(hop_length)),
            ctypes.c_size_t(int(fft_length)),
            ctypes.c_float(float(power)),
            ctypes.c_bool(bool(center)),
            ctypes.c_int(int(pad_mode)),
            ctypes.c_float(float(mel_floor)),
            ctypes.c_int(int(log_mel_mode)),
            ctypes.c_float(float(dither)),
            ctypes.c_float(float(preemphasis)),
            ctypes.c_bool(bool(remove_dc_offset)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_spectrogram failed"))
        return self._tensor_from_node(out.value)

    # ── Image processing ─────────────────────────────────────────────

    def image_preprocess(self, pixel_input, src_width, src_height,
                         target_width, target_height, patch_size, channels,
                         rescale_factor, mean, std_dev):
        """Resize, normalize, and patch an image tensor.

        *mean* and *std_dev* are per-channel float sequences (length = channels).
        """
        pixel_input = self._ensure_tensor(pixel_input)
        mean_arr = (ctypes.c_float * len(mean))(*[float(v) for v in mean])
        std_arr = (ctypes.c_float * len(std_dev))(*[float(v) for v in std_dev])
        out = cactus_node_t()
        rc = _lib.cactus_graph_image_preprocess(
            self.h, cactus_node_t(pixel_input.id),
            ctypes.c_int(int(src_width)),
            ctypes.c_int(int(src_height)),
            ctypes.c_int(int(target_width)),
            ctypes.c_int(int(target_height)),
            ctypes.c_int(int(patch_size)),
            ctypes.c_int(int(channels)),
            ctypes.c_float(float(rescale_factor)),
            mean_arr, std_arr, ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(_err("graph_image_preprocess failed"))
        return self._tensor_from_node(out.value)

    # ── Introspection ────────────────────────────────────────────────

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
            raise RuntimeError(_err("graph_get_output_info failed"))
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
        elif precision in (self.CQ1, self.CQ2, self.CQ3, self.CQ4):
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

    def __ne__(self, other):
        if isinstance(other, Tensor):
            return self.g.not_equal(self, other)
        return self.g.scalar_not_equal(self, other)

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

    def scalar_floor_divide(self, value):
        return self.g.scalar_floor_divide(self, value)

    def scalar_not_equal(self, value):
        return self.g.scalar_not_equal(self, value)

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

    def clamp(self, lo, hi):
        return self.g.clamp(self, lo, hi)

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

    def expand(self, shape):
        return self.g.expand(self, shape)

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

    def matmul(self, other, pretransposed_rhs=False, backend=Graph.CPU, output_dtype=None):
        return self.g.matmul(
            self,
            other,
            pretransposed_rhs=pretransposed_rhs,
            backend=backend,
            output_dtype=output_dtype,
        )

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

    def cumsum(self, axis):
        return self.g.cumsum(self, axis)

    def numpy(self):
        info = cactus_tensor_info_t()
        rc = _lib.cactus_graph_get_output_info(self.g.h, cactus_node_t(self.id), ctypes.byref(info))
        if rc != 0:
            raise RuntimeError(_err("graph_get_output_info failed"))

        out_ptr = ctypes.c_void_p()
        rc = _lib.cactus_graph_get_output_ptr(self.g.h, cactus_node_t(self.id), ctypes.byref(out_ptr))
        if rc != 0 or not out_ptr.value:
            raise RuntimeError(_err("graph_get_output_ptr failed"))

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
        elif precision in (Graph.CQ1, Graph.CQ2, Graph.CQ3, Graph.CQ4):
            arr = np.ctypeslib.as_array((ctypes.c_uint8 * int(info.byte_size)).from_address(out_ptr.value))
            return arr.copy()
        else:
            raise RuntimeError("unsupported precision")

        return arr.reshape(shape).copy()

    def __repr__(self):
        return f"Tensor(id={self.id}, shape={self.shape}, dtype={self.dtype})"
