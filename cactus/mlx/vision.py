"""Image loading, dynamic-resolution preprocessing, and image-token expansion.

Patch sizing follows NVIDIA's NemotronH Omni image processor (Apache-2.0).
Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
Modified by Cactus Compute: NumPy resizing and MLX-LM processor interface.
Licensed under the Apache License, Version 2.0; see LICENSE.
"""
import base64
import io
import math
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image


def load_image(source):
    if isinstance(source, Image.Image):
        return source.convert('RGB')
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert('RGB')
    source = str(source)
    if source.startswith('data:'):
        source = io.BytesIO(base64.b64decode(source.partition(',')[2], validate=True))
    elif source.startswith(('https://', 'http://')):
        with urlopen(source, timeout=30) as response:
            source = io.BytesIO(response.read())
    else:
        source = Path(source).expanduser()
    with Image.open(source) as image:
        return image.convert('RGB')


def resize_bicubic(image, height, width):
    """Float32 antialiased bicubic resize; retain overshoot as PyTorch does."""
    image = np.asarray(image, dtype=np.float32)

    def axis_resize(data, output, axis):
        original = data.shape[axis]
        if original == output:
            return data
        scale = np.float32(original / output)
        filter_scale = np.maximum(scale, np.float32(1))
        support = np.float32(2) * filter_scale
        shape = list(data.shape)
        shape[axis] = output
        result = np.empty(shape, dtype=np.float32)
        for index in range(output):
            center = scale * np.float32(index + 0.5)
            first = max(0, int(center - support + np.float32(0.5)))
            end = min(original, int(center + support + np.float32(0.5)))
            x = np.abs((np.arange(first, end, dtype=np.float32) + np.float32(0.5) - center) / filter_scale)
            coefficients = np.where(x < 1, ((np.float32(1.5) * x - np.float32(2.5)) * x) * x + 1,
                                    ((np.float32(-0.5) * x + np.float32(2.5)) * x - 4) * x + 2)
            coefficients[x >= 2] = 0
            coefficients /= coefficients.sum(dtype=np.float32)
            if axis == 1:
                result[:, index] = np.einsum('hkc,k->hc', data[:, first:end], coefficients)
            else:
                result[index] = np.einsum('kwc,k->wc', data[first:end], coefficients)
        return result

    return axis_resize(axis_resize(image, width, 1), height, 0)


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
        image = load_image(source)
        height, width = self.target_size(*image.size)
        pixels = resize_bicubic(np.asarray(image), height, width)
        pixels = (pixels / np.float32(255) - self.mean) / self.std
        return pixels.transpose(2, 0, 1).copy()

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
