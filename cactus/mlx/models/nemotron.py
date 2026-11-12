"""Use upstream Nemotron-H blocks with CQ matrices and GPU-selected experts."""
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.nemotron_h import Model, ModelArgs
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from ..checkpoint import Checkpoint, books
from ..ops import CQEmbedding, CQExperts, CQLinear


class CQModel(Model):
    def __init__(self, args):
        super().__init__(args)
        self._decode_graph = None

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            return self.lm_head(self._forward_embeddings(input_embeddings, cache))
        if inputs.shape == (1, 1) and cache and all(
            c[0] is not None and c.lengths is None and c.left_padding is None
            for c in cache if isinstance(c, ArraysCache)
        ):
            if self._decode_graph is None:
                from .nemotron_decode import DecodeGraph
                self._decode_graph = DecodeGraph(self)
            return self._decode_graph(inputs, cache)
        return super().__call__(inputs, cache=cache)

    def make_cache(self):
        cache = super().make_cache()
        index = 0
        for layer in self.layers:
            if layer.block_type == 'M':
                m = layer.mixer
                cache[index][1] = mx.zeros((1, m.num_heads, m.head_dim, m.ssm_state_size), mx.float32)
            if layer.block_type in ('M', '*'):
                index += 1
        return cache

    def forward_hidden(self, tokens, cache, input_embeddings=None):
        if input_embeddings is not None:
            hidden = self._forward_embeddings(input_embeddings, cache)
            return self.lm_head(hidden), hidden
        if tokens.shape == (1, 1) and all(c[0] is not None for c in cache if isinstance(c, ArraysCache)):
            if self._decode_graph is None:
                from .nemotron_decode import DecodeGraph
                self._decode_graph = DecodeGraph(self)
            return self._decode_graph(tokens, cache, return_hidden=True)
        hidden = self.backbone(tokens, cache=cache)
        return self.lm_head(hidden), hidden

    def _forward_embeddings(self, hidden, cache):
        if cache is None:
            cache = [None] * len(self.layers)
        masks = {'*': create_attention_mask(hidden, cache[self.backbone.fa_idx]),
                 'M': create_ssm_mask(hidden, cache[self.backbone.ssm_idx])}
        states = iter(cache)
        for layer in self.layers:
            recurrent = layer.block_type in ('M', '*')
            hidden = layer(hidden, mask=masks.get(layer.block_type),
                           cache=next(states) if recurrent else None)
        return self.backbone.norm_f(hidden)

    def embed_inputs(self, tokens, pixel_values=None):
        if tokens.ndim != 1:
            raise ValueError('Vision generation supports one sequence at a time')
        embeddings = self.backbone.embeddings(tokens)
        if not pixel_values:
            return embeddings
        if self._vision is None:
            from .nemotron_vision import load_vision
            self._vision = load_vision(self._model_file)
        features = mx.concatenate([self._vision(mx.array(pixels)) for pixels in pixel_values])
        positions = np.flatnonzero(np.asarray(tokens) == 18)
        if len(positions) != features.shape[0]:
            raise ValueError('Image token count does not match the encoded image patches')
        embeddings[mx.array(positions)] = features.astype(embeddings.dtype)
        return embeddings

    def load_mtp(self):
        if self._mtp is None:
            from .nemotron_mtp import load_mtp
            self._mtp = load_mtp(self, self._model_file)
        return self._mtp


def locate(model, path):
    target = model
    parts = path.split('.')
    for key in parts[:-1]:
        target = target[int(key)] if key.isdigit() else getattr(target, key)
    return target, parts[-1]


def load(model_file, config_file, progress=None):
    start = time.perf_counter()
    checkpoint = Checkpoint(model_file)
    config = json.loads(Path(config_file).read_text())
    # The original serialized Infinity is represented as an object in this export.
    if isinstance(config.get('time_step_limit', [None, None])[1], dict):
        config['time_step_limit'][1] = float('inf')
    args = ModelArgs.from_dict(config)
    model = CQModel(args)
    model._model_file = str(model_file)
    model._mtp = None
    model._vision = None
    bs = books(checkpoint)
    experts = {}
    for i, (name, leaf) in enumerate(checkpoint.leaves.items()):
        cls = CQEmbedding if name == 'model.embeddings.weight' else CQLinear
        op = cls(checkpoint, name, bs[leaf['rate']])
        path = name.replace('model.', 'backbone.', 1).removesuffix('.weight')
        if '.experts.' in name:
            prefix, which = path.rsplit('.experts.', 1)
            experts.setdefault(prefix, {})[which] = op
        elif path.endswith('.gate'):
            target, key = locate(model, path + '.weight')
            setattr(target, key, op.gather(mx.arange(op._n)))
            mx.eval(getattr(target, key))
        else:
            target, key = locate(model, path)
            setattr(target, key, op)
        if progress and (i % 40 == 0 or i + 1 == len(checkpoint.leaves)):
            progress(f'Loaded {i + 1}/{len(checkpoint.leaves)} CQ matrices; {mx.get_active_memory() / 2**30:.2f} GiB')
    for path, pair in experts.items():
        if set(pair) != {'up', 'down'}:
            raise ValueError(f'Incomplete CQ expert pair: {path}')
        target, key = locate(model, path + '.switch_mlp')
        setattr(target, key, CQExperts(pair['up'], pair['down'], args.n_routed_experts))
    for name in checkpoint.manifest['rest']:
        path = name.removeprefix('language_model.')
        value = checkpoint.array(name)
        if 'conv1d.weight' in path:
            value = value.moveaxis(2, 1)
        if path.endswith('A_log'):
            value = value.astype(mx.float32)
        target, key = locate(model, path)
        setattr(target, key, value)
    model.eval()
    mx.eval(model.parameters())
    checkpoint.close()
    gc.collect()
    mx.clear_cache()
    if progress:
        progress(f'Model ready in {time.perf_counter() - start:.1f}s; {mx.get_active_memory() / 2**30:.2f} GiB active')
    return model
