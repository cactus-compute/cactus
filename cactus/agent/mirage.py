"""Image understanding, OCR, object masks, and structured forms with Mirage."""
from __future__ import annotations

import ctypes
import io
import json
import os

from .runtime import synchronized
from . import fetch
from .audio import _options

_loaded = None


def _image(image, width=None, height=None):
    if width is not None or height is not None:
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError("raw RGB input requires positive integer width and height")
        if not isinstance(image, (bytes, bytearray, memoryview)):
            raise TypeError("explicit dimensions require packed RGB bytes")
        data = bytes(image)
        if len(data) != width * height * 3:
            raise ValueError("RGB data length must be width * height * 3")
        return data, width, height
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        if len(image.shape) != 3 or image.shape[2] != 3 or str(image.dtype) != "uint8":
            raise ValueError("an image array must have shape (height, width, 3) and dtype uint8")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return image.tobytes(), int(width), int(height)
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError('image files require Pillow: pip install -U cactus-compute') from exc
    if isinstance(image, Image.Image):
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    else:
        source = io.BytesIO(image) if isinstance(image, (bytes, bytearray, memoryview)) else os.fspath(image)
        with Image.open(source) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
    return rgb.tobytes(), rgb.width, rgb.height


class Mirage:
    """Describe images or fill constrained forms using the shared Cactus engine.

    Images can be paths, encoded image bytes, Pillow images, uint8 RGB arrays,
    or packed RGB bytes with explicit width and height. Options are the keys
    accepted by cactus_describe, including sections, fields, conf, mask_format,
    and form. Calls are serialized with Needle and Whistle.
    """

    def __init__(self, weights=None, buffer_size=16 * 1024 * 1024):
        self.weights = os.fspath(weights) if weights is not None else fetch.fetch_weights(fetch.MIRAGE)
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._buffer = ctypes.create_string_buffer(buffer_size)
        self._bind()

    @synchronized
    def _bind(self):
        global _loaded
        from .runtime import _lib
        lib = _lib()
        if _loaded != self.weights:
            with open(self.weights, "rb") as source:
                data = source.read()
            if lib.cactus_load(data, len(data)) < 0:
                _loaded = None
                raise RuntimeError(lib.cactus_last_error().decode("utf-8", "replace"))
            if not lib.cactus_models() & 4:
                raise ValueError("the selected archive is not a Mirage vision model")
            _loaded = self.weights
        return lib

    @synchronized
    def describe(self, image, *, options=None, language=None, keywords=None, width=None, height=None, **settings):
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        merged = dict(options or {})
        if merged.keys() & settings.keys():
            raise ValueError("an option was supplied both in options and as a keyword")
        merged.update(settings)
        encoded = json.dumps(merged, ensure_ascii=False, allow_nan=False).encode("utf-8")
        rgb, width, height = _image(image, width, height)
        language, keywords = _options(language, keywords)
        lib = self._bind()
        code = lib.cactus_describe(rgb, width, height, encoded, language, keywords, self._buffer, len(self._buffer))
        if code < 0:
            raise RuntimeError(lib.cactus_last_error().decode("utf-8", "replace"))
        return json.loads(self._buffer.value.decode("utf-8"))

    def extract(self, image, form, **options):
        """Fill ordered choice, bounded integer, text, or referenced-box fields."""
        if not isinstance(form, dict) or not form:
            raise ValueError("form must be a nonempty dictionary of fields")
        return self.describe(image, form=form, **options)


_shared = {}


@synchronized
def describe(image, *, weights=None, **options):
    key = os.fspath(weights) if weights is not None else None
    if key not in _shared:
        _shared[key] = Mirage(weights=weights)
    return _shared[key].describe(image, **options)
