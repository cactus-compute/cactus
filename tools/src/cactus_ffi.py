"""
Python FFI bindings for Cactus - direct mapping of cactus_ffi.h
"""

import ctypes
import platform
from pathlib import Path

# Callback type
TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

# Find library
_DIR = Path(__file__).parent.parent.parent
if platform.system() == "Darwin":
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.dylib"
else:
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.so"

_lib = None
if _LIB_PATH.exists():
    _lib = ctypes.CDLL(str(_LIB_PATH))

    # cactus_init
    _lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    _lib.cactus_init.restype = ctypes.c_void_p

    # cactus_complete
    _lib.cactus_complete.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_char_p, TokenCallback, ctypes.c_void_p
    ]
    _lib.cactus_complete.restype = ctypes.c_int

    # cactus_transcribe
    _lib.cactus_transcribe.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_char_p, TokenCallback, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
    ]
    _lib.cactus_transcribe.restype = ctypes.c_int

    # cactus_embed
    _lib.cactus_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_embed.restype = ctypes.c_int

    # cactus_image_embed
    _lib.cactus_image_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_image_embed.restype = ctypes.c_int

    # cactus_audio_embed
    _lib.cactus_audio_embed.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.cactus_audio_embed.restype = ctypes.c_int

    # cactus_reset
    _lib.cactus_reset.argtypes = [ctypes.c_void_p]
    _lib.cactus_reset.restype = None

    # cactus_stop
    _lib.cactus_stop.argtypes = [ctypes.c_void_p]
    _lib.cactus_stop.restype = None

    # cactus_destroy
    _lib.cactus_destroy.argtypes = [ctypes.c_void_p]
    _lib.cactus_destroy.restype = None


def cactus_init(model_path, context_size=2048, corpus_dir=None):
    """Initialize a model. Returns model handle."""
    return _lib.cactus_init(
        model_path.encode() if isinstance(model_path, str) else model_path,
        context_size,
        corpus_dir.encode() if corpus_dir else None
    )


def cactus_complete(model, messages_json, options_json=None, tools_json=None, callback=None):
    """Run completion. Returns response JSON string."""
    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_complete(
        model,
        messages_json.encode() if isinstance(messages_json, str) else messages_json,
        buf, len(buf),
        options_json.encode() if options_json else None,
        tools_json.encode() if tools_json else None,
        cb, None
    )
    return buf.value.decode()


def cactus_transcribe(model, audio_path, prompt="", callback=None):
    """Transcribe audio. Returns response JSON string."""
    buf = ctypes.create_string_buffer(65536)
    cb = TokenCallback(callback) if callback else TokenCallback()
    _lib.cactus_transcribe(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        prompt.encode() if isinstance(prompt, str) else prompt,
        buf, len(buf),
        None, cb, None, None, 0
    )
    return buf.value.decode()


def cactus_embed(model, text):
    """Get text embeddings. Returns list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_embed(
        model,
        text.encode() if isinstance(text, str) else text,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_image_embed(model, image_path):
    """Get image embeddings. Returns list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_image_embed(
        model,
        image_path.encode() if isinstance(image_path, str) else image_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_audio_embed(model, audio_path):
    """Get audio embeddings. Returns list of floats."""
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_audio_embed(
        model,
        audio_path.encode() if isinstance(audio_path, str) else audio_path,
        buf, ctypes.sizeof(buf), ctypes.byref(dim)
    )
    return list(buf[:dim.value])


def cactus_reset(model):
    """Reset model state."""
    _lib.cactus_reset(model)


def cactus_stop(model):
    """Stop generation."""
    _lib.cactus_stop(model)


def cactus_destroy(model):
    """Destroy model and free memory."""
    _lib.cactus_destroy(model)
