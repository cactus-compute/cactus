from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_user_graph_bundle


MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = "Hello"
MAX_NEW_TOKENS = 4


def _time_ms(fn, *, iterations: int = 50) -> float:
    for _ in range(5):
        fn()
    start = perf_counter()
    for _ in range(iterations):
        fn()
    return (perf_counter() - start) * 1000.0 / iterations


def load_model_and_params(dtype=np.float16):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True).eval()
    params = {
        name: jnp.asarray(tensor.detach().cpu().numpy().astype(dtype))
        for name, tensor in model.state_dict().items()
    }
    return model, model.config, params


def rms_norm(x, weight, eps):
    xf = x.astype(jnp.float32)
    variance = jnp.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(variance + eps) * weight).astype(jnp.float16)


def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, position_ids, *, head_dim, theta):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = position_ids.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb)[:, None, :, :].astype(jnp.float16)
    sin = jnp.sin(emb)[:, None, :, :].astype(jnp.float16)
    return (x * cos + rotate_half(x) * sin).astype(jnp.float16)


def linear(x, weight):
    return x @ weight.T


def llama_logits(params, input_ids, attention_mask, position_ids, *, cfg):
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    num_layers = int(cfg.num_hidden_layers)
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    eps = float(cfg.rms_norm_eps)

    x = params["model.embed_tokens.weight"][input_ids].astype(jnp.float16)
    batch, seq_len = input_ids.shape

    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        residual = x
        h = rms_norm(x, params[prefix + "input_layernorm.weight"], eps)

        q = linear(h, params[prefix + "self_attn.q_proj.weight"])
        k = linear(h, params[prefix + "self_attn.k_proj.weight"])
        v = linear(h, params[prefix + "self_attn.v_proj.weight"])
        q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, position_ids, head_dim=head_dim, theta=rope_theta)
        k = apply_rope(k, position_ids, head_dim=head_dim, theta=rope_theta)

        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(head_dim))
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        mask = attention_mask[:, None, None, :].astype(jnp.bool_) & causal[None, None, :, :]
        scores = jnp.where(mask, scores, -1.0e4)
        probs = jax.nn.softmax(scores, axis=-1).astype(jnp.float16)
        attn = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden_size)
        x = (residual + linear(attn, params[prefix + "self_attn.o_proj.weight"])).astype(jnp.float16)

        residual = x
        h = rms_norm(x, params[prefix + "post_attention_layernorm.weight"], eps)
        gate = jax.nn.silu(linear(h, params[prefix + "mlp.gate_proj.weight"]))
        up = linear(h, params[prefix + "mlp.up_proj.weight"])
        x = (residual + linear((gate * up).astype(jnp.float16), params[prefix + "mlp.down_proj.weight"])).astype(jnp.float16)

    x = rms_norm(x, params["model.norm.weight"], eps)
    return linear(x, params["lm_head.weight"])


def diff_stats(actual, expected):
    diff = np.abs(np.asarray(actual).astype(np.float32) - np.asarray(expected).astype(np.float32))
    return float(diff.max()), float(diff.mean())


def greedy_compare(tokenizer, torch_model, cfg, params, cactus_graph, prompt_ids, max_len):
    torch_ids = np.asarray([prompt_ids], dtype=np.int64)
    jax_ids = np.zeros((1, max_len), dtype=np.int32)
    cactus_ids = np.zeros((1, max_len), dtype=np.int32)
    jax_ids[0, : len(prompt_ids)] = prompt_ids
    cactus_ids[0, : len(prompt_ids)] = prompt_ids
    positions = np.arange(max_len, dtype=np.int32)[None, :]
    jax_tokens = list(prompt_ids)
    cactus_tokens = list(prompt_ids)
    max_diffs = []
    mean_diffs = []

    jit_logits = jax.jit(lambda p, ids, mask, pos: llama_logits(p, ids, mask, pos, cfg=cfg))
    warm_mask = (jax_ids != 0).astype(np.float32)
    jit_logits(params, jnp.asarray(jax_ids), jnp.asarray(warm_mask), jnp.asarray(positions)).block_until_ready()

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            torch_logits = torch_model(input_ids=torch.tensor(torch_ids)).logits.detach().cpu().numpy()
            torch_next = int(torch_logits[0, -1].argmax())
            torch_ids = np.concatenate([torch_ids, np.asarray([[torch_next]], dtype=np.int64)], axis=1)

            used = len(jax_tokens)
            mask = np.zeros((1, max_len), dtype=np.float32)
            mask[0, :used] = 1.0
            jax_logits = np.asarray(jit_logits(params, jnp.asarray(jax_ids), jnp.asarray(mask), jnp.asarray(positions)))
            jax_next = int(jax_logits[0, used - 1].argmax())
            jax_tokens.append(jax_next)
            jax_ids[0, used] = jax_next

            cactus_graph.graph.set_inputs([cactus_ids, mask, positions])
            cactus_logits = cactus_graph.graph.execute()[0].numpy()
            cactus_next = int(cactus_logits[0, used - 1].argmax())
            max_diff, mean_diff = diff_stats(cactus_logits[:, used - 1], jax_logits[:, used - 1])
            max_diffs.append(max_diff)
            mean_diffs.append(mean_diff)
            cactus_tokens.append(cactus_next)
            cactus_ids[0, used] = cactus_next

    final_mask = np.zeros((1, max_len), dtype=np.float32)
    final_mask[0, : len(jax_tokens)] = 1.0
    jax_ms = _time_ms(lambda: jit_logits(params, jnp.asarray(jax_ids), jnp.asarray(final_mask), jnp.asarray(positions)).block_until_ready())
    cactus_ms = _time_ms(lambda: (cactus_graph.graph.set_inputs([cactus_ids, final_mask, positions]), cactus_graph.graph.execute()))

    return {
        "torch_tokens": torch_ids[0].tolist(),
        "jax_tokens": jax_tokens,
        "cactus_tokens": cactus_tokens,
        "torch_text": tokenizer.decode(torch_ids[0]),
        "jax_text": tokenizer.decode(jax_tokens),
        "cactus_text": tokenizer.decode(cactus_tokens),
        "per_step_max_diff": max_diffs,
        "per_step_mean_diff": mean_diffs,
        "jax_ms": jax_ms,
        "cactus_ms": cactus_ms,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    torch_model, cfg, params = load_model_and_params(np.float16)
    prompt_ids = tokenizer(PROMPT, return_tensors="np")["input_ids"][0].astype(np.int32).tolist()
    max_len = len(prompt_ids) + MAX_NEW_TOKENS
    input_ids = np.zeros((1, max_len), dtype=np.int32)
    input_ids[0, : len(prompt_ids)] = prompt_ids
    attention_mask = np.zeros((1, max_len), dtype=np.float32)
    attention_mask[0, : len(prompt_ids)] = 1.0
    position_ids = np.arange(max_len, dtype=np.int32)[None, :]

    result = build_jax_user_graph_bundle(
        params=params,
        specs=[
            JaxGraphSpec(
                name="full_context",
                fn=lambda p, ids, mask, pos: llama_logits(p, ids, mask, pos, cfg=cfg),
                example_args=(jnp.asarray(input_ids), jnp.asarray(attention_mask), jnp.asarray(position_ids)),
                input_names=("input_ids", "attention_mask", "position_ids"),
                output_names=("logits",),
            )
        ],
        output_dir=Path("/private/tmp/cactus_tiny_llama_jax_probe"),
        model_id=MODEL_ID,
        task="text-generation",
        graph_meta={"probe": "tiny_llama"},
    )

    graph = result.bundle.graphs["full_context"]
    op_counts = Counter(node.op for node in graph.ir_graph.nodes.values())
    comparison = greedy_compare(tokenizer, torch_model, cfg, params, graph, prompt_ids, max_len)

    print("\nTiny LLaMA-compatible JAX generic probe:")
    print(f"  model: {MODEL_ID}")
    print(f"  torch tokens:  {comparison['torch_tokens']}")
    print(f"  jax tokens:    {comparison['jax_tokens']}")
    print(f"  cactus tokens: {comparison['cactus_tokens']}")
    print(f"  torch text:  {comparison['torch_text']!r}")
    print(f"  jax text:    {comparison['jax_text']!r}")
    print(f"  cactus text: {comparison['cactus_text']!r}")
    print(f"  per-step max diff:  {[round(v, 6) for v in comparison['per_step_max_diff']]}")
    print(f"  per-step mean diff: {[round(v, 6) for v in comparison['per_step_mean_diff']]}")
    print(f"  jax jit fixed-shape execute: {comparison['jax_ms']:.4f} ms")
    print(f"  cactus set_input+execute:    {comparison['cactus_ms']:.4f} ms")
    print(f"  semantic op counts: rms_norm={op_counts.get('rms_norm', 0)}, rope={op_counts.get('rope', 0)}, silu={op_counts.get('silu', 0)}")
    print(f"  jax pattern counts: {graph.ir_graph.meta.get('jax_semantic_patterns', {})}")
    print(f"  bundle: {result.output_dir}")


if __name__ == "__main__":
    main()
