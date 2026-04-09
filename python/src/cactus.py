"""Cactus Python FFI bindings."""
import ctypes
import platform
from pathlib import Path

TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

_DIR = Path(__file__).parent.parent.parent
if platform.system() == "Darwin":
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.dylib"
else:
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.so"

if not _LIB_PATH.exists():
    raise RuntimeError(
        f"Cactus library not found at {_LIB_PATH}\n"
        f"Please build first: cactus build --python"
    )

_lib = ctypes.CDLL(str(_LIB_PATH))

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

_lib.cactus_graph_precision_cast.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_precision_cast.restype = ctypes.c_int
_lib.cactus_graph_quantize_activations.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_quantize_activations.restype = ctypes.c_int

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
_lib.cactus_graph_softmax.argtypes = [
    cactus_graph_t, cactus_node_t, ctypes.c_int32, ctypes.POINTER(cactus_node_t)
]
_lib.cactus_graph_softmax.restype = ctypes.c_int

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

_lib.cactus_detect_language.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_detect_language.restype = ctypes.c_int

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

_lib.cactus_vad.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_vad.restype = ctypes.c_int

_lib.cactus_diarize.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_diarize.restype = ctypes.c_int

_lib.cactus_embed_speaker.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_embed_speaker.restype = ctypes.c_int

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

_lib.cactus_stream_transcribe_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.cactus_stream_transcribe_start.restype = ctypes.c_void_p

_lib.cactus_stream_transcribe_process.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_size_t
]
_lib.cactus_stream_transcribe_process.restype = ctypes.c_int

_lib.cactus_stream_transcribe_stop.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
]
_lib.cactus_stream_transcribe_stop.restype = ctypes.c_int

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
    if s is None:
        return None
    return s.encode() if isinstance(s, str) else s


def cactus_get_last_error():
    """Returns the last error message or None."""
    result = _lib.cactus_get_last_error()
    return result.decode() if result else None


def _err(default):
    return cactus_get_last_error() or default


def cactus_set_telemetry_environment(cache_location):
    """Sets the telemetry cache directory."""
    _lib.cactus_set_telemetry_environment(None, _enc(cache_location), None)


def cactus_set_app_id(app_id):
    """Sets the application identifier for telemetry."""
    _lib.cactus_set_app_id(_enc(app_id))


def cactus_telemetry_flush():
    """Flushes pending telemetry events."""
    _lib.cactus_telemetry_flush()


def cactus_telemetry_shutdown():
    """Flushes and shuts down the telemetry subsystem."""
    _lib.cactus_telemetry_shutdown()


def cactus_init(model_path, corpus_dir, cache_index):
    """Initializes a model from the given path. Returns handle."""
    handle = _lib.cactus_init(_enc(model_path), _enc(corpus_dir), cache_index)
    if not handle:
        raise RuntimeError(_err("Failed to initialize model"))
    return handle


def cactus_destroy(model):
    """Frees all model resources."""
    _lib.cactus_destroy(model)


def cactus_reset(model):
    """Clears the KV cache."""
    _lib.cactus_reset(model)


def cactus_stop(model):
    """Signals the current generation to stop."""
    _lib.cactus_stop(model)


def cactus_complete(model, messages_json, options_json, tools_json, callback, pcm_data=None):
    """Runs chat completion. Returns response string."""
    buf = ctypes.create_string_buffer(65536)
    if callback:
        def _bridge(token_bytes, token_id, _):
            callback(token_bytes.decode("utf-8", errors="ignore") if token_bytes else "", token_id)
        cb = TokenCallback(_bridge)
    else:
        cb = TokenCallback()
    if pcm_data is not None:
        pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
        pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
        pcm_size = len(pcm_data)
    else:
        pcm_ptr = None
        pcm_size = 0
    rc = _lib.cactus_complete(
        model, _enc(messages_json), buf, len(buf),
        _enc(options_json), _enc(tools_json), cb, None, pcm_ptr, pcm_size
    )
    if rc < 0:
        raise RuntimeError(_err("Completion failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_prefill(model, messages_json, options_json, tools_json, pcm_data=None):
    """Prefills the KV cache with messages."""
    buf = ctypes.create_string_buffer(65536)
    if pcm_data is not None:
        pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
        pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
        pcm_size = len(pcm_data)
    else:
        pcm_ptr = None
        pcm_size = 0
    rc = _lib.cactus_prefill(
        model, _enc(messages_json), buf, len(buf),
        _enc(options_json), _enc(tools_json), pcm_ptr, pcm_size
    )
    if rc < 0:
        raise RuntimeError(_err("Prefill failed"))
def cactus_detect_language(model, audio_path, options_json, pcm_data):
    """Detects the spoken language in audio. Returns JSON string."""
    buf = ctypes.create_string_buffer(65536)
    if pcm_data is not None:
        pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
        pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
        pcm_size = len(pcm_data)
    else:
        pcm_ptr = None
        pcm_size = 0
    rc = _lib.cactus_detect_language(
        model, _enc(audio_path), buf, len(buf), _enc(options_json), pcm_ptr, pcm_size
    )
    if rc < 0:
        raise RuntimeError(_err("Detect language failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_transcribe(model, audio_path, prompt, options_json, callback, pcm_data):
    """Transcribes audio to text. Returns transcription string."""
    buf = ctypes.create_string_buffer(65536)
    if callback:
        def _bridge(token_bytes, token_id, _):
            callback(token_bytes.decode("utf-8", errors="ignore") if token_bytes else "", token_id)
        cb = TokenCallback(_bridge)
    else:
        cb = TokenCallback()
    if pcm_data is not None:
        pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
        pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
        pcm_size = len(pcm_data)
    else:
        pcm_ptr = None
        pcm_size = 0
    rc = _lib.cactus_transcribe(
        model, _enc(audio_path), _enc(prompt), buf, len(buf),
        _enc(options_json), cb, None, pcm_ptr, pcm_size
    )
    if rc < 0:
        raise RuntimeError(_err("Transcription failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_embed(model, text, normalize):
    """Generates a text embedding. Returns list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_embed(model, _enc(text), buf, 4096, ctypes.byref(dim), normalize)
    if rc < 0:
        raise RuntimeError(_err("Embedding failed"))
    return list(buf[:dim.value])


def cactus_image_embed(model, image_path):
    """Generates an image embedding. Returns list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_image_embed(model, _enc(image_path), buf, 4096, ctypes.byref(dim))
    if rc < 0:
        raise RuntimeError(_err("Image embedding failed"))
    return list(buf[:dim.value])


def cactus_audio_embed(model, audio_path):
    """Generates an audio embedding. Returns list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    rc = _lib.cactus_audio_embed(model, _enc(audio_path), buf, 4096, ctypes.byref(dim))
    if rc < 0:
        raise RuntimeError(_err("Audio embedding failed"))
    return list(buf[:dim.value])


def cactus_vad(model, audio_path, options_json, pcm_data):
    """Runs voice activity detection. Returns JSON string."""
    buf = ctypes.create_string_buffer(65536)
    if pcm_data is not None:
        pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
        pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
        pcm_size = len(pcm_data)
    else:
        pcm_ptr = None
        pcm_size = 0
    rc = _lib.cactus_vad(
        model, _enc(audio_path), buf, len(buf), _enc(options_json), pcm_ptr, pcm_size
    )
    if rc < 0:
        raise RuntimeError(_err("VAD failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_diarize(model, audio_path, options_json, pcm_data):
    """Runs speaker diarization. Returns JSON string."""
    buf = ctypes.create_string_buffer(1 << 20)
    if pcm_data is not None:
        pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
        pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
        pcm_size = len(pcm_data)
    else:
        pcm_ptr = None
        pcm_size = 0
    rc = _lib.cactus_diarize(
        model, _enc(audio_path), buf, len(buf), _enc(options_json), pcm_ptr, pcm_size
    )
    if rc < 0:
        raise RuntimeError(_err("Diarize failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_embed_speaker(model, audio_path, options_json, pcm_data):
    """Extracts a speaker embedding vector. Returns JSON string."""
    buf = ctypes.create_string_buffer(65536)
    if pcm_data is not None:
        pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
        pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
        pcm_size = len(pcm_data)
    else:
        pcm_ptr = None
        pcm_size = 0
    rc = _lib.cactus_embed_speaker(
        model, _enc(audio_path), buf, len(buf), _enc(options_json), pcm_ptr, pcm_size
    )
    if rc < 0:
        raise RuntimeError(_err("EmbedSpeaker failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_tokenize(model, text):
    """Tokenizes text into token IDs. Returns list of ints."""
    max_tokens = 8192
    arr = (ctypes.c_uint32 * max_tokens)()
    n = ctypes.c_size_t(0)
    rc = _lib.cactus_tokenize(model, _enc(text), arr, max_tokens, ctypes.byref(n))
    if rc < 0:
        raise RuntimeError(_err("Tokenization failed"))
    return list(arr[:n.value])


def cactus_score_window(model, tokens, start, end, context):
    """Scores a window of tokens. Returns JSON string."""
    buf = ctypes.create_string_buffer(65536)
    token_len = len(tokens)
    arr = (ctypes.c_uint32 * token_len)(*tokens)
    rc = _lib.cactus_score_window(model, arr, token_len, start, end, context, buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Score window failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_rag_query(model, query, top_k):
    """Queries the RAG corpus. Returns JSON string."""
    buf = ctypes.create_string_buffer(65536)
    rc = _lib.cactus_rag_query(model, _enc(query), buf, len(buf), top_k)
    if rc < 0:
        raise RuntimeError(_err("RAG query failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_stream_transcribe_start(model, options_json):
    """Creates a streaming transcription session. Returns stream handle."""
    handle = _lib.cactus_stream_transcribe_start(model, _enc(options_json))
    if not handle:
        raise RuntimeError(_err("Failed to start stream transcription"))
    return handle


def cactus_stream_transcribe_process(stream, pcm_data):
    """Processes a chunk of PCM audio. Returns partial text."""
    buf = ctypes.create_string_buffer(65536)
    pcm_arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
    pcm_ptr = ctypes.cast(pcm_arr, ctypes.POINTER(ctypes.c_uint8))
    rc = _lib.cactus_stream_transcribe_process(stream, pcm_ptr, len(pcm_data), buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Stream process failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_stream_transcribe_stop(stream):
    """Finalizes transcription. Returns final text."""
    buf = ctypes.create_string_buffer(65536)
    rc = _lib.cactus_stream_transcribe_stop(stream, buf, len(buf))
    if rc < 0:
        raise RuntimeError(_err("Stream stop failed"))
    return buf.value.decode("utf-8", errors="ignore")


def cactus_index_init(index_dir, embedding_dim):
    """Initializes a vector index in the given directory. Returns handle."""
    handle = _lib.cactus_index_init(_enc(index_dir), embedding_dim)
    if not handle:
        raise RuntimeError(_err("Failed to initialize index"))
    return handle


def cactus_index_add(index, ids, documents, embeddings, metadatas):
    """Adds documents with embeddings to the index."""
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
    for i, emb in enumerate(embeddings):
        arr = (ctypes.c_float * len(emb))(*emb)
        emb_arrays.append(arr)
        emb_ptrs[i] = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))

    rc = _lib.cactus_index_add(index, ids_arr, docs_arr, meta_arr, emb_ptrs, count, embedding_dim)
    if rc < 0:
        raise RuntimeError(_err("Failed to add to index"))


def cactus_index_delete(index, ids):
    """Removes documents by ID."""
    count = len(ids)
    ids_arr = (ctypes.c_int * count)(*ids)
    rc = _lib.cactus_index_delete(index, ids_arr, count)
    if rc < 0:
        raise RuntimeError(_err("Failed to delete from index"))


def cactus_index_query(index, embedding, options_json):
    """Searches the index by embedding. Returns JSON string."""
    result_capacity = 1000
    embedding_dim = len(embedding)
    emb_arr = (ctypes.c_float * embedding_dim)(*embedding)
    emb_ptr = ctypes.cast(emb_arr, ctypes.POINTER(ctypes.c_float))
    emb_ptr_ptr = ctypes.pointer(emb_ptr)

    id_buffer = (ctypes.c_int * result_capacity)()
    score_buffer = (ctypes.c_float * result_capacity)()
    id_ptr = ctypes.cast(id_buffer, ctypes.POINTER(ctypes.c_int))
    score_ptr = ctypes.cast(score_buffer, ctypes.POINTER(ctypes.c_float))
    id_size = ctypes.c_size_t(result_capacity)
    score_size = ctypes.c_size_t(result_capacity)
    id_ptr_ptr = ctypes.pointer(id_ptr)
    score_ptr_ptr = ctypes.pointer(score_ptr)

    rc = _lib.cactus_index_query(
        index, emb_ptr_ptr, 1, embedding_dim, _enc(options_json),
        id_ptr_ptr, ctypes.byref(id_size), score_ptr_ptr, ctypes.byref(score_size)
    )
    if rc < 0:
        raise RuntimeError(_err("Index query failed"))
    n = id_size.value
    parts = [f'{{"id":{id_buffer[i]},"score":{score_buffer[i]}}}' for i in range(n)]
    return '{"results":[' + ",".join(parts) + ']}'


_INDEX_DOC_BUF_SIZE = 4096
_INDEX_EMB_BUF_SIZE = 4096


def cactus_index_get(index, ids):
    """Retrieves documents by ID. Returns JSON string."""
    count = len(ids)
    if count == 0:
        return '{"results":[]}'

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
        doc_ptrs, doc_sizes, meta_ptrs, meta_sizes, emb_ptrs, emb_sizes
    )
    if rc < 0:
        raise RuntimeError(_err("Failed to get from index"))

    parts = []
    for i in range(count):
        doc = doc_raw[i].value.decode("utf-8", errors="ignore")
        meta_bytes = meta_raw[i].value
        meta = meta_bytes.decode("utf-8", errors="ignore") if meta_bytes else None
        emb_dim = emb_sizes[i]
        emb = list(emb_raw[i][:emb_dim]) if emb_dim > 0 else []
        meta_str = f'"{meta}"' if meta is not None else "null"
        emb_str = "[" + ",".join(str(v) for v in emb) + "]"
        parts.append(f'{{"document":"{doc}","metadata":{meta_str},"embedding":{emb_str}}}')
    return '{"results":[' + ",".join(parts) + ']}'


def cactus_index_compact(index):
    """Compacts the index storage."""
    rc = _lib.cactus_index_compact(index)
    if rc < 0:
        raise RuntimeError(_err("Failed to compact index"))


def cactus_index_destroy(index):
    """Frees all index resources."""
    _lib.cactus_index_destroy(index)


def cactus_log_set_level(level):
    """Sets the log level. 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE."""
    _lib.cactus_log_set_level(level)


_log_callback_ref = None


def cactus_log_set_callback(callback):
    """Sets a log callback. Pass None to clear. Callback signature: (level: int, component: str, message: str)."""
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
