"""C-RADIO image encoder and projector for consolidated Nemotron CQ weights.

Architecture follows NVIDIA's Apache-2.0 NemotronH Omni / RADIO definitions.
Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
Modified by Cactus Compute: MLX implementation and packed CQ weight loading.
Licensed under the Apache License, Version 2.0; see LICENSE.
"""
import json

import mlx.core as mx
import mlx.nn as nn

from ..checkpoint import Checkpoint, books
from ..ops import CQLinear


def resize_positions(position, height, width):
    """NVIDIA CPE: bilinear square interpolation, then a top-left crop."""
    side = int(position.shape[1] ** 0.5)
    size = max(height, width)
    grid = position.astype(mx.float32).reshape(side, side, -1)
    coords = mx.maximum((mx.arange(size, dtype=mx.float32) + 0.5) * (side / size) - 0.5, 0)
    lo = mx.floor(coords).astype(mx.int32)
    hi = mx.minimum(lo + 1, side - 1)
    frac = coords - lo
    grid = grid[lo] * (1 - frac[:, None, None]) + grid[hi] * frac[:, None, None]
    grid = grid[:, lo] * (1 - frac[None, :, None]) + grid[:, hi] * frac[None, :, None]
    return grid[:height, :width].reshape(1, height * width, -1).astype(position.dtype)


def pixel_shuffle(x, height, width):
    """Group 2×2 patches in NVIDIA's v2 channel order."""
    batch, _, dim = x.shape
    x = x.reshape(batch, height, width // 2, dim * 2).transpose(0, 2, 1, 3)
    return x.reshape(batch, width // 2, height // 2, dim * 4).transpose(0, 2, 1, 3).reshape(batch, -1, dim * 4)


class VisionEncoder(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self._prefix = 'vision_model.radio_model.model.'
        self._num_layers = len({name.split('.')[4] for name in weights if '.blocks.' in name})
        self._position_key = None
        self._position = None

    def __call__(self, pixels):
        if pixels.ndim != 3 or pixels.shape[0] != 3:
            raise ValueError('Expected normalized RGB pixels with shape (3, height, width)')
        _, height, width = pixels.shape
        if height % 32 or width % 32:
            raise ValueError('Image dimensions must be multiples of 32')
        rows, cols = height // 16, width // 16
        weights, prefix = self.weights, self._prefix

        def linear(x, name):
            y = x @ weights[name + '.weight'].T
            bias = weights.get(name + '.bias')
            return y if bias is None else y + bias

        def norm(x, name):
            return mx.fast.layer_norm(x, weights[name + '.weight'], weights[name + '.bias'], 1e-6)

        patch = prefix + 'patch_generator.'
        pixels = pixels.astype(weights[patch + 'embedder.weight'].dtype)
        x = pixels.reshape(3, rows, 16, cols, 16).transpose(1, 3, 0, 2, 4).reshape(1, rows * cols, 768)
        x = linear(x, patch + 'embedder')
        if self._position_key != (rows, cols):
            self._position = resize_positions(weights[patch + 'pos_embed'], rows, cols)
            mx.eval(self._position)
            self._position_key = (rows, cols)
        x = x + self._position
        cls = weights[patch + 'cls_token.token'][None]
        x = mx.concatenate([cls, x], axis=1)
        for index in range(self._num_layers):
            name = prefix + f'blocks.{index}.'
            qkv = linear(norm(x, name + 'norm1'), name + 'attn.qkv')
            q, k, v = qkv.reshape(1, x.shape[1], 3, 16, 80).transpose(2, 0, 3, 1, 4)
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=80 ** -0.5)
            x = x + linear(attended.transpose(0, 2, 1, 3).reshape(x.shape), name + 'attn.proj')
            y = nn.gelu(linear(norm(x, name + 'norm2'), name + 'mlp.fc1'))
            x = x + linear(y, name + 'mlp.fc2')
            # Bound temporary memory and avoid retaining the whole image graph.
            mx.eval(x)
        x = norm(x[:, cls.shape[1]:], 'vision_projector.vision_final_layernorm')
        x = pixel_shuffle(x, rows, cols)
        x = mx.fast.rms_norm(x, weights['mlp1.0.weight'], 1e-5)
        x = linear(nn.relu2(linear(x, 'mlp1.1')), 'mlp1.3')
        mx.eval(x)
        return x[0]


def load_vision(model_file):
    checkpoint = Checkpoint(model_file, component='vision')
    weights = {}
    try:
        book = books(checkpoint)['4-ffq']
        names = json.loads(checkpoint.header['__metadata__']['cq.components'])['vision']
        packed_names = {name + suffix for name in checkpoint.leaves
                        for suffix in ('.codes', '.norms', '.norm_scale')}
        for name in names:
            if name not in packed_names:
                weights[name] = checkpoint.array(name)
        for name, leaf in checkpoint.leaves.items():
            op = CQLinear(checkpoint, name, book)
            weights[name] = op.gather(mx.arange(leaf['shape'][0]))
            mx.eval(weights[name])
            del op
        mx.eval(weights)
    finally:
        checkpoint.close()
    encoder = VisionEncoder(weights)
    encoder.eval()
    return encoder
