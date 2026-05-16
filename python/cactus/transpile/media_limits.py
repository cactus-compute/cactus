from __future__ import annotations

import os

_DEFAULT_IMAGE_SIZE = 256


def static_image_size() -> int:
    raw = os.environ.get("CACTUS_TRANSPILER_IMAGE_SIZE", str(_DEFAULT_IMAGE_SIZE))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_IMAGE_SIZE


def resize_static_image(image: object) -> object:
    size = static_image_size()
    target = (size, size)
    if getattr(image, "size", None) == target or not hasattr(image, "resize"):
        return image
    try:
        from PIL import Image  # type: ignore

        resample = Image.Resampling.BILINEAR
    except AttributeError:  # pragma: no cover
        resample = Image.BILINEAR  # type: ignore[name-defined]
    return image.resize(target, resample=resample)
