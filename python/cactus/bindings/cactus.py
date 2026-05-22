"""Cactus Python FFI bindings."""
import ctypes
import json
import os
import platform
from pathlib import Path

TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

_LIB_NAME = "libcactus.dylib" if platform.system() == "Darwin" else "libcactus.so"


def _find_library():
    override = os.environ.get("CACTUS_LIB_PATH")
    if override:
        return Path(override).expanduser().resolve()

    bundled = Path(__file__).parent / "lib" / _LIB_NAME
    if bundled.exists():
        return bundled

    dev_build = Path(__file__).parent.parent.parent.parent / "cactus" / "build" / _LIB_NAME
    if dev_build.exists():
        return dev_build

    raise RuntimeError(
        f"Cactus library ({_LIB_NAME}) not found.\n"
        f"Install with: pip install cactus-compute\n"
        f"Or build from source: cactus build --python"
    )


_LIB_PATH = _find_library()
_lib = ctypes.CDLL(str(_LIB_PATH))


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

# cactus graph API
_lib.cactus_graph_create.restype = cactus_graph_t
_lib.cactus_graph_destroy.argtypes = [cactus_graph_t]
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
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_bool, ctypes.c_float, ctypes.c_float, ctypes.POINTER(cactus_node_t)
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

_bind_optional("cactus_stream_transcribe_start", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_void_p)

_bind_optional(
    "cactus_stream_transcribe_process",
    [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
    ],
    ctypes.c_int,
)

_bind_optional(
    "cactus_stream_transcribe_stop",
    [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t],
    ctypes.c_int,
)

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
    return json.loads(text)


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
    """Free all resources associated with a model handle."""
    _lib.cactus_destroy(model)


def cactus_reset(model):
    """Clear the KV cache, resetting the model to a fresh conversation state."""
    _lib.cactus_reset(model)


def cactus_stop(model):
    """Signal the model to stop the current generation early."""
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
    buf = ctypes.create_string_buffer(65536)
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
    buf = ctypes.create_string_buffer(65536)
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
        options:    Optional dict of transcription options.
        callback:   Optional function(text, token_id) for streaming tokens.
        pcm_data:   Optional raw PCM audio bytes (alternative to audio_path).

    Returns:
        A dict with transcription text and segments.
    """
    buf = ctypes.create_string_buffer(65536)
    cb = _make_token_callback(callback)
    pcm_ptr, pcm_size = _prepare_pcm(pcm_data)
    rc = _lib.cactus_transcribe(
        model, _enc(audio_path), _enc(prompt), buf, len(buf),
        _to_json(options), cb, None, pcm_ptr, pcm_size,
    )
    if rc < 0:
        raise RuntimeError(_err("Transcription failed"))
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


# ── Streaming transcription ──────────────────────────────────────────


def cactus_stream_transcribe_start(model, options=None):
    """Start a streaming transcription session.

    Returns:
        An opaque stream handle. Feed audio with cactus_stream_transcribe_process(),
        then finalize with cactus_stream_transcribe_stop().
    """
    handle = _lib.cactus_stream_transcribe_start(model, _to_json(options))
    if not handle:
        raise RuntimeError(_err("Failed to start stream transcription"))
    return handle


def cactus_stream_transcribe_process(stream, pcm_data):
    """Feed a PCM audio chunk to a streaming transcription session.

    Returns:
        A dict with partial transcription results.
    """
    buf = ctypes.create_string_buffer(65536)
    pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
    pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
    rc = _lib.cactus_stream_transcribe_process(stream, pcm_ptr, len(pcm_data), buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Stream process failed"))
    return _from_json(buf)


def cactus_stream_transcribe_stop(stream):
    """Finalize a streaming transcription session.

    Returns:
        A dict with the final transcription.
    """
    buf = ctypes.create_string_buffer(65536)
    rc = _lib.cactus_stream_transcribe_stop(stream, buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Stream stop failed"))
    return _from_json(buf)


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
    rc = _lib.cactus_embed(model, _enc(text), buf, 4096, ctypes.byref(dim), normalize)
    if rc < 0:
        raise RuntimeError(_err("Embedding failed"))
    return list(buf[:dim.value])


def cactus_image_embed(model, image_path):
    """Compute an image embedding. Returns a list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_image_embed(model, _enc(image_path), buf, 4096, ctypes.byref(dim))
    if rc < 0:
        raise RuntimeError(_err("Image embedding failed"))
    return list(buf[:dim.value])


def cactus_audio_embed(model, audio_path):
    """Compute an audio embedding. Returns a list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_audio_embed(model, _enc(audio_path), buf, 4096, ctypes.byref(dim))
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
    embedding_dim = len(embeddings[0]) if embeddings else 0

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
    """Free all resources associated with an index handle."""
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
