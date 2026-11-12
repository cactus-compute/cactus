"""Verify short blocks and replay truncated prefixes for exact Mamba rollback."""
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.ssm import compute_dt

_scan = mx.fast.metal_kernel(
    name='cq_ssm_verify', input_names=['x', 'a_log', 'b', 'c', 'skip', 'dt', 'state'],
    output_names=['out', 'history'],
    source=(Path(__file__).parents[1] / 'kernels/ssm_verify.metal').read_text())


def mamba_verify(layer, x, conv_state, state, snapshots=True):
    m = layer.mixer
    z = m.in_proj(layer.norm(x))
    gate, conv, dt = mx.split(z, [m.intermediate_size, m.intermediate_size + m.conv_dim], axis=-1)
    padded = mx.concatenate([conv_state, conv], axis=1)
    conv = nn.silu(m.conv1d(padded))
    values, b, c = mx.split(conv, [m.intermediate_size, m.intermediate_size + m.n_groups * m.ssm_state_size], axis=-1)
    n = x.shape[1]
    dt = compute_dt(dt, m.dt_bias, m.time_step_limit)
    y, history = _scan(inputs=[values, m.A_log, b, c, m.D.astype(x.dtype), dt, state],
        template=[('T', values.dtype), ('H', m.num_heads), ('DH', m.head_dim),
                  ('DS', m.ssm_state_size), ('NG', m.n_groups), ('NT', n), ('SNAPSHOTS', snapshots)],
        grid=(32, m.head_dim, m.num_heads), threadgroup=(32, 8, 1),
        output_shapes=[values.shape, (n, *state.shape) if snapshots else state.shape], output_dtypes=[values.dtype, mx.float32])
    conv_history = (mx.stack([padded[:, t + 1:t + m.conv_kernel_size] for t in range(n)])
                    if snapshots else padded[:, -m.conv_kernel_size + 1:])
    return x + m.out_proj(m.norm(y, gate)), (conv_history, history)


def moe_verify(layer, x):
    m = layer.mixer
    normalized = layer.norm(x)
    # The native BF16 router switches GEMV/GEMM arithmetic with token count.
    # Keep its one-token arithmetic so a verification block selects the same experts.
    selected = [m.gate(normalized[:, i:i + 1]) for i in range(x.shape[1])]
    indices = mx.concatenate([s[0] for s in selected], axis=1)
    scores = mx.concatenate([s[1] for s in selected], axis=1)
    h = m.fc1_latent_proj(normalized) if m.moe_latent_size is not None else normalized
    y = m.switch_mlp(h, indices)
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if m.moe_latent_size is not None:
        y = m.fc2_latent_proj(y)
    if m.config.n_shared_experts is not None:
        y = y + m.shared_experts(normalized)
    return x + y


def attention_verify(layer, x, cache):
    m = layer.mixer
    normalized = layer.norm(x)
    batch, length, _ = x.shape
    queries = m.q_proj(normalized).reshape(batch, length, m.num_heads, -1).transpose(0, 2, 1, 3)
    keys = m.k_proj(normalized).reshape(batch, length, m.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    values = m.v_proj(normalized).reshape(batch, length, m.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    outputs = []
    # Preserve the native single-query attention reduction and causal prefix.
    # The expensive CQ projections still process the complete verification block.
    for i in range(length):
        k, v = cache.update_and_fetch(keys[:, :, i:i + 1], values[:, :, i:i + 1])
        outputs.append(scaled_dot_product_attention(queries[:, :, i:i + 1], k, v,
                                                    cache=cache, scale=m.scale, mask=None))
    output = mx.concatenate(outputs, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return x + m.o_proj(output)


class VerifyGraph:
    def __init__(self, model):
        self.model = model
        self.segments = []
        pending, indices = [], []
        ci = 0

        def flush():
            if not pending:
                return
            layers = tuple(pending)
            def run(x, *states):
                i = 0
                history = []
                for layer in layers:
                    if layer.block_type == 'M':
                        x, h = mamba_verify(layer, x, *states[i:i+2], snapshots=False)
                        i += 2
                        history.extend(h)
                    elif layer.block_type == 'E':
                        x = moe_verify(layer, x)
                    else:
                        x = layer(x)
                return x, history
            self.segments.append((mx.compile(run), tuple(indices)))
            pending.clear()
            indices.clear()

        for layer in model.layers:
            if layer.block_type == '*':
                flush()
                self.segments.append((layer, ci))
                ci += 1
            else:
                pending.append(layer)
                if layer.block_type == 'M':
                    indices.append(ci)
                    ci += 1
        flush()
        def finish(x):
            h = model.backbone.norm_f(x)
            return model.lm_head(h), h
        self.finish = mx.compile(finish)

    def __call__(self, tokens, cache):
        if tokens.shape[0] != 1 or not 1 <= tokens.shape[1] <= 8:
            raise ValueError('MTP verification requires batch 1 and 1–8 tokens')
        x = self.model.backbone.embeddings(tokens)
        history = VerificationState(self, tokens, cache)
        for run, indices in self.segments:
            if isinstance(indices, int):
                x = attention_verify(run, x, cache[indices])
            else:
                states = [s for i in indices for s in (cache[i][0], cache[i][1])]
                x, updated = run(x, *states)
                for j, i in enumerate(indices):
                    history[i] = updated[2*j:2*j+2]
        logits, hidden = self.finish(x)
        return logits, hidden, history

    @staticmethod
    def commit(cache, history, keep, total):
        if not 1 <= keep <= total:
            raise ValueError('Invalid accepted prefix length')
        if isinstance(history, VerificationState):
            history.commit(cache, keep, total)
            return
        for i, c in enumerate(cache):
            if isinstance(c, ArraysCache):
                c[0], c[1] = (h[keep - 1] for h in history[i])
            else:
                c.trim(total - keep)


class VerificationState(dict):
    """Keep final recurrent states; replay a prefix only when it is truncated."""
    def __init__(self, graph, tokens, cache):
        super().__init__()
        self.graph, self.tokens = graph, tokens
        self.before = {i: (c[0], c[1]) for i, c in enumerate(cache) if isinstance(c, ArraysCache)}

    def commit(self, cache, keep, total):
        if keep == self.tokens.shape[1]:
            for i, states in self.items():
                cache[i][0], cache[i][1] = states
            return
        for i, c in enumerate(cache):
            if isinstance(c, ArraysCache):
                c[0], c[1] = self.before[i]
            else:
                c.trim(total)
        _, _, prefix = self.graph(self.tokens[:, :keep], cache)
        prefix.commit(cache, keep, keep)
