"""
Python FFI bindings for Cactus - direct mapping of cactus_ffi.h
"""

import ctypes
import json
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
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_bool
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


def cactus_complete(
    model,
    messages,
    tools=None,
    temperature=None,
    top_p=None,
    top_k=None,
    max_tokens=None,
    stop_sequences=None,
    force_tools=False,
    callback=None
):
    """Run chat completion with tool support.

    Args:
        model: Model handle from cactus_init
        messages: List of message dicts [{"role": "user", "content": "..."}] or JSON string
        tools: List of tool dicts [{"name": "...", "description": "...", "parameters": {...}}] or JSON string
        temperature: Sampling temperature (default: model default)
        top_p: Top-p sampling (default: model default)
        top_k: Top-k sampling (default: model default)
        max_tokens: Maximum tokens to generate (default: 256)
        stop_sequences: List of stop sequences
        force_tools: If True, constrain output to valid tool call format (default: False)
        callback: Token callback function(token_str, token_id, user_data)

    Returns:
        Response JSON string with 'response', 'function_calls', and metrics
    """
    if isinstance(messages, list):
        messages_json = json.dumps(messages)
    else:
        messages_json = messages

    tools_json = None
    if tools is not None:
        if isinstance(tools, list):
            tools_json = json.dumps(tools)
        else:
            tools_json = tools

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if stop_sequences is not None:
        options["stop_sequences"] = stop_sequences
    if force_tools:
        options["force_tools"] = True

    options_json = json.dumps(options) if options else None

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


def cactus_embed(model, text, normalize=False):
    """Get text embeddings. Returns list of floats.

    Args:
        model: Model handle from cactus_init
        text: Text to embed
        normalize: If True, L2-normalize the embeddings (default: False)
    """
    buf = (ctypes.c_float * 4096)()
    dim = ctypes.c_size_t()
    _lib.cactus_embed(
        model,
        text.encode() if isinstance(text, str) else text,
        buf, ctypes.sizeof(buf), ctypes.byref(dim), normalize
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
