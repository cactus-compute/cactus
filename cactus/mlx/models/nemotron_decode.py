"""Compile fixed-shape recurrent/MoE segments while keeping indexed I/O outside."""
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache


class DecodeGraph:
    def __init__(self, model):
        self.model = model
        self.segments = []
        pending, indices = [], []
        cache_index = 0

        def flush():
            if not pending:
                return
            layers = tuple(pending)
            def run(x, *states):
                i = 0
                updated = []
                for layer in layers:
                    if layer.block_type == 'M':
                        c = ArraysCache(size=2)
                        c[0], c[1] = states[i:i + 2]
                        i += 2
                        x = layer(x, cache=c)
                        updated.extend([c[0], c[1]])
                    else:
                        x = layer(x)
                return x, updated

            self.segments.append((mx.compile(run), tuple(indices)))
            pending.clear()
            indices.clear()

        for layer in model.layers:
            if layer.block_type == '*':
                flush()
                self.segments.append((layer, cache_index))
                cache_index += 1
            else:
                pending.append(layer)
                if layer.block_type == 'M':
                    indices.append(cache_index)
                    cache_index += 1
        flush()
        def finish(x):
            h = model.backbone.norm_f(x)
            return model.lm_head(h), h
        self.finish = mx.compile(finish)

    def __call__(self, tokens, cache, return_hidden=False):
        if tokens.shape != (1, 1):
            raise ValueError('Compiled decode expects one token and one sequence')
        x = self.model.backbone.embeddings(tokens)
        for run, indices in self.segments:
            if isinstance(indices, int):
                x = run(x, cache=cache[indices])
            else:
                states = []
                for i in indices:
                    if cache[i][0] is None or cache[i].lengths is not None or cache[i].left_padding is not None:
                        raise ValueError('Compiled decode requires an initialized, unpadded cache')
                    states.extend([cache[i][0], cache[i][1]])
                x, updated = run(x, *states)
                for j, i in enumerate(indices):
                    cache[i][0], cache[i][1] = updated[2*j:2*j+2]
        logits, hidden = self.finish(x)
        return (logits, hidden) if return_hidden else logits
