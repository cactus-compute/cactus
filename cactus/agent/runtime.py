"""The shared Cactus engine and its C ABI."""
import ctypes
import os
import warnings
from functools import wraps
import threading

lock = threading.RLock()


def synchronized(function):
    @wraps(function)
    def call(*args, **kwargs):
        with lock:
            return function(*args, **kwargs)
    return call


def bind(lib):
    text, integer = ctypes.c_char_p, ctypes.c_int
    samples = ctypes.POINTER(ctypes.c_float)
    signatures = {
        "load": ([text, ctypes.c_uint64], integer),
        "models": ([], integer),
        "last_error": ([], text),
        "init": ([text, text, text], integer),
        "complete": ([text, samples, integer, integer, text, integer], integer),
        "set_audio": ([text, text, integer], None),
        "reset": ([], None),
        "embed": ([text, samples, integer, samples, integer], integer),
        "transcribe": ([samples, integer, text, text, integer, text, integer], integer),
        "stream_transcribe_process": ([samples, integer, text, text, text, integer], integer),
        "stream_transcribe_stop": ([text, integer], integer),
        "describe": ([text, integer, integer, text, text, text, text, integer], integer),
    }
    for name, (arguments, result) in signatures.items():
        try:
            function = getattr(lib, "cactus_" + name)
        except AttributeError as exc:
            raise RuntimeError("incompatible engine: Cactus 4 requires the cactus_* ABI") from exc
        function.argtypes, function.restype = arguments, result
    return lib


def _library_path(generation=3):
    from . import fetch
    fetch._engine(generation)
    override = os.environ.get("CACTUS_LIB_PATH")
    if override:
        return override
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), fetch.lib_name())
    if os.path.isfile(local):
        return local
    return fetch.fetch_library(dest_dir=fetch.engine_cache_dir())


def _base_weights_path(generation=3):
    from . import fetch
    return fetch.fetch_weights(generation)


_lib_handles = {}
_active = {}
_loaded_base = {}


def _load_base(lib, generation):
    if _loaded_base.get(generation):
        return
    path = _base_weights_path(generation)
    with open(path, "rb") as handle:
        data = handle.read()
    if lib.cactus_load(data, len(data)) < 0:
        raise RuntimeError(f"cactus_load failed for {path}")
    _loaded_base[generation] = path
    _active.pop(generation, None)


def _load_cdll(generation):
    """Load the engine, retrying with the other libc build if the first is wrong.

    A musl system that reports itself as glibc (or the reverse) otherwise fails
    at load with a missing symbol such as strtoll_l.
    """
    path = _library_path(generation)
    try:
        return ctypes.CDLL(path)
    except OSError as first:
        from . import fetch

        other = fetch.other_libc_tag()
        if other is None:
            raise
        try:
            alt = fetch.fetch_library(dest_dir=os.path.dirname(path) or ".",
                                      tag=other, generation=generation)
            handle = ctypes.CDLL(alt)
        except Exception:
            raise first from None
        warnings.warn(f"the {fetch._platform_tag()} engine did not load ({first}); "
                      f"using the {other} build instead", stacklevel=3)
        return handle


def _lib(generation=3):
    from . import fetch
    fetch._engine(generation)
    if 3 not in _lib_handles:
        _lib_handles[3] = bind(_load_cdll(3))
    return _lib_handles[3]
