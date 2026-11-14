"""Image loading, dynamic-resolution preprocessing, and image-token expansion.

Patch sizing follows NVIDIA's NemotronH Omni image processor (Apache-2.0).
Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
Modified by Cactus Compute: Metal resizing and MLX-LM processor interface.
Licensed under the Apache License, Version 2.0; see LICENSE.
"""
import base64
import io
import math
from functools import cache, lru_cache
from pathlib import Path
from urllib.request import urlopen

import mlx.core as mx
import numpy as np
from PIL import Image

from .ops import KERNELS


def _rgb_pixels(image):
    return np.asarray(image if image.mode == 'RGB' else image.convert('RGB'))


def load_image(source):
    if isinstance(source, np.ndarray):
        source = Image.fromarray(source)
    if isinstance(source, Image.Image):
        return _rgb_pixels(source)
    source = str(source)
    if source.startswith('data:'):
        source = io.BytesIO(base64.b64decode(source.partition(',')[2], validate=True))
    elif source.startswith(('https://', 'http://')):
        with urlopen(source, timeout=30) as response:
            source = io.BytesIO(response.read())
    else:
        source = Path(source).expanduser()
    with Image.open(source) as image:
        return _rgb_pixels(image)


@lru_cache(maxsize=32)
def _bicubic_taps(original, output):
    scale = np.float32(original / output)
    filter_scale = np.maximum(scale, np.float32(1))
    support = np.float32(2) * filter_scale
    center = scale * (np.arange(output, dtype=np.float32) + np.float32(0.5))
    first = np.maximum(0, (center - support + np.float32(0.5)).astype(np.int64))
    end = np.minimum(original, (center + support + np.float32(0.5)).astype(np.int64))
    position = first[:, None] + np.arange((end - first).max())
    x = np.abs((position.astype(np.float32) + np.float32(0.5) - center[:, None]) / filter_scale)
    weights = np.where(x < 1, ((np.float32(1.5) * x - np.float32(2.5)) * x) * x + 1,
                       ((np.float32(-0.5) * x + np.float32(2.5)) * x - 4) * x + 2)
    weights[(x >= 2) | (position >= end[:, None])] = 0
    weights /= weights.sum(axis=1, keepdims=True, dtype=np.float32)
    return mx.array(weights), mx.array(np.minimum(position, original - 1).astype(np.uint32))


@cache
def _resize_kernel():
    return mx.fast.metal_kernel(
        name='bicubic_resize', output_names=['out'],
        input_names=['pixels', 'column_weights', 'columns', 'row_weights', 'rows', 'mean', 'std'],
        source=(KERNELS / 'bicubic_resize.metal').read_text(), compile_options={'math_mode': 'safe'})


def resize_bicubic(image, height, width, mean, std):
    """Antialiased bicubic resize to float32 (3, height, width) as (x / 255 - mean) / std, with PyTorch's overshoot."""
    column_weights, columns = _bicubic_taps(image.shape[1], width)
    row_weights, rows = _bicubic_taps(image.shape[0], height)
    return _resize_kernel()(
        inputs=[mx.array(image), column_weights, columns, row_weights, rows,
                mx.array(mean, mx.float32), mx.array(std, mx.float32)],
        grid=(width, height, 1), threadgroup=(16, 16, 1),
        output_shapes=[(3, height, width)], output_dtypes=[mx.float32])[0]


class ImageProcessor:
    image_token = '<image>'

    def __init__(self, config=None):
        config = config or {}
        self.patch_size = config.get('patch_size', 16)
        self.min_patches = config.get('min_num_patches', 1024)
        self.max_patches = config.get('max_num_patches', 13312)
        self.mean = np.array(config.get('norm_mean', [0.48145466, 0.4578275, 0.40821073]), np.float32)
        self.std = np.array(config.get('norm_std', [0.26862954, 0.26130258, 0.27577711]), np.float32)
        if self.patch_size != 16 or config.get('downsample_ratio', 0.5) != 0.5:
            raise ValueError('This vision adapter requires patch size 16 and downsample ratio 0.5')

    def target_size(self, width, height):
        hp = max(1, round(height / self.patch_size + 0.5))
        wp = max(1, round(width / self.patch_size + 0.5))
        factor = min(math.sqrt(self.max_patches / (hp * wp)), 1.)
        hp, wp = max(2, math.floor(factor * hp)), max(2, math.floor(factor * wp))
        if self.max_patches > self.min_patches and hp * wp < self.min_patches:
            up = math.sqrt(self.min_patches / (hp * wp))
            hp, wp = math.ceil(up * hp), math.ceil(up * wp)
        if hp % 2:
            hp = hp + 1 if (hp + 1) * wp <= self.max_patches else max(2, hp - 1)
        if wp % 2:
            wp = wp + 1 if hp * (wp + 1) <= self.max_patches else max(2, wp - 1)
        return hp * self.patch_size, wp * self.patch_size

    def preprocess(self, source):
        pixels = load_image(source)
        height, width = self.target_size(pixels.shape[1], pixels.shape[0])
        return resize_bicubic(pixels, height, width, self.mean, self.std)

    def __call__(self, text, *, images):
        parts = text.split(self.image_token)
        if len(parts) != len(images) + 1:
            raise ValueError('The number of image placeholders must match the supplied images')
        pixel_values = [self.preprocess(image) for image in images]
        expanded = [parts[0]]
        for pixels, following in zip(pixel_values, parts[1:]):
            count = pixels.shape[1] * pixels.shape[2] // 1024
            expanded.extend(['<img>', self.image_token * count, '</img>', following])
        return ''.join(expanded), pixel_values
