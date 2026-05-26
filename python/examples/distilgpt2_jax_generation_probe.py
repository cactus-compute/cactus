from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_user_graph_bundle
from cactus.transpile.runtime_compat import Graph, Tensor


MODEL_ID = "distilgpt2"
PROMPT = "The future of artificial intelligence is"
N_LAYER = 6
N_HEAD = 12
N_EMBD = 768
HEAD_DIM = N_EMBD // N_HEAD
MAX_NEW_TOKENS = 8


def _time_ms(fn, *, iterations: int = 20) -> float:
    for _ in range(3):
        fn()
    start = perf_counter()
    for _ in range(iterations):
        fn()
    return (perf_counter() - start) * 1000.0 / iterations


def load_params(dtype=np.float16):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True).eval()
    params = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        params[name] = jnp.asarray(tensor.detach().cpu().numpy().astype(dtype))
    return model, params


def layer_norm(x, weight, bias, eps=1e-5):
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    variance = jnp.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * jax.lax.rsqrt(variance + eps) * weight + bias).astype(jnp.float16)


def gelu_new(x):
    xf = x.astype(jnp.float32)
    return (0.5 * xf * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (xf + 0.044715 * (xf**3))))).astype(jnp.float16)


def embed(params, input_ids, position_ids=None):
    positions = position_ids
    if positions is None:
        positions = jnp.arange(input_ids.shape[1], dtype=jnp.int32)[None, :]
    token = params["transformer.wte.weight"][input_ids]
    pos = params["transformer.wpe.weight"][positions]
    return (token + pos).astype(jnp.float16)


def project_qkv(params, x, layer):
    prefix = f"transformer.h.{layer}."
    h = layer_norm(x, params[prefix + "ln_1.weight"], params[prefix + "ln_1.bias"])
    qkv = h @ params[prefix + "attn.c_attn.weight"] + params[prefix + "attn.c_attn.bias"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    batch, seq, _ = q.shape
    q = q.reshape(batch, seq, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    return q, k, v


def attend(q, k, v, attention_mask, *, causal: bool):
    query_seq = q.shape[2]
    key_seq = k.shape[2]
    scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(HEAD_DIM))
    mask = attention_mask[:, None, None, :].astype(jnp.bool_)
    if causal:
        causal_mask = jnp.tril(jnp.ones((query_seq, key_seq), dtype=jnp.bool_))
        mask = mask & causal_mask[None, None, :, :]
    scores = jnp.where(mask, scores, -1.0e4)
    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.float16)
    return (probs @ v).transpose(0, 2, 1, 3).reshape(q.shape[0], query_seq, N_EMBD)


def finish_block(params, x, attn_y, layer):
    prefix = f"transformer.h.{layer}."
    attn_proj = attn_y @ params[prefix + "attn.c_proj.weight"] + params[prefix + "attn.c_proj.bias"]
    x = (x + attn_proj).astype(jnp.float16)
    h2 = layer_norm(x, params[prefix + "ln_2.weight"], params[prefix + "ln_2.bias"])
    fc = h2 @ params[prefix + "mlp.c_fc.weight"] + params[prefix + "mlp.c_fc.bias"]
    proj = gelu_new(fc) @ params[prefix + "mlp.c_proj.weight"] + params[prefix + "mlp.c_proj.bias"]
    return (x + proj).astype(jnp.float16)


def full_context_logits(params, input_ids, attention_mask):
    x = embed(params, input_ids)
    for layer in range(N_LAYER):
        q, k, v = project_qkv(params, x, layer)
        x = finish_block(params, x, attend(q, k, v, attention_mask, causal=True), layer)
    x = layer_norm(x, params["transformer.ln_f.weight"], params["transformer.ln_f.bias"])
    return x @ params["transformer.wte.weight"].T


def decoder_prefill(params, input_ids, attention_mask):
    x = embed(params, input_ids)
    keys = []
    values = []
    for layer in range(N_LAYER):
        q, k, v = project_qkv(params, x, layer)
        keys.append(k)
        values.append(v)
        x = finish_block(params, x, attend(q, k, v, attention_mask, causal=True), layer)
    x = layer_norm(x, params["transformer.ln_f.weight"], params["transformer.ln_f.bias"])
    logits = x @ params["transformer.wte.weight"].T
    return logits, jnp.stack(keys, axis=0), jnp.stack(values, axis=0)


def decoder_step(params, input_ids, position_ids, past_keys, past_values, attention_mask):
    x = embed(params, input_ids, position_ids)
    next_keys = []
    next_values = []
    for layer in range(N_LAYER):
        q, k, v = project_qkv(params, x, layer)
        all_k = jnp.concatenate([past_keys[layer], k], axis=2)
        all_v = jnp.concatenate([past_values[layer], v], axis=2)
        next_keys.append(all_k)
        next_values.append(all_v)
        x = finish_block(params, x, attend(q, all_k, all_v, attention_mask, causal=False), layer)
    x = layer_norm(x, params["transformer.ln_f.weight"], params["transformer.ln_f.bias"])
    logits = x @ params["transformer.wte.weight"].T
    return logits, jnp.stack(next_keys, axis=0), jnp.stack(next_values, axis=0)


def numpy_outputs(outputs):
    result = []
    for output in outputs:
        result.append(output.numpy() if hasattr(output, "numpy") else np.asarray(output))
    return result


def diff_stats(actual, expected):
    diff = np.abs(np.asarray(actual).astype(np.float32) - np.asarray(expected).astype(np.float32))
    return float(diff.max()), float(diff.mean())


def greedy_full_context(tokenizer, torch_model, params, cactus_graph, prompt_ids, max_len):
    torch_ids = np.array(prompt_ids, dtype=np.int32)[None, :]
    jax_ids = np.zeros((1, max_len), dtype=np.int32)
    cactus_ids = np.zeros((1, max_len), dtype=np.int32)
    jax_ids[0, : len(prompt_ids)] = prompt_ids
    cactus_ids[0, : len(prompt_ids)] = prompt_ids

    jax_tokens = list(prompt_ids)
    cactus_tokens = list(prompt_ids)
    max_diffs = []
    mean_diffs = []

    jit_full = jax.jit(full_context_logits)
    jit_full(params, jnp.asarray(jax_ids), jnp.asarray(jax_ids != 0, dtype=jnp.float32)).block_until_ready()

    for _ in range(MAX_NEW_TOKENS):
        torch_logits = torch_model(input_ids=__import__("torch").tensor(torch_ids)).logits.detach().cpu().numpy()
        torch_next = int(torch_logits[0, -1].argmax())
        torch_ids = np.concatenate([torch_ids, np.array([[torch_next]], dtype=np.int32)], axis=1)

        used = len(jax_tokens)
        jax_mask = np.zeros((1, max_len), dtype=np.float32)
        jax_mask[0, :used] = 1.0
        jax_logits = np.asarray(jit_full(params, jnp.asarray(jax_ids), jnp.asarray(jax_mask)))
        jax_next = int(jax_logits[0, used - 1].argmax())
        jax_tokens.append(jax_next)
        jax_ids[0, used] = jax_next

        cactus_mask = np.zeros((1, max_len), dtype=np.float32)
        cactus_mask[0, : len(cactus_tokens)] = 1.0
        cactus_graph.graph.set_inputs([cactus_ids, cactus_mask])
        cactus_logits = cactus_graph.graph.execute()[0].numpy()
        cactus_next = int(cactus_logits[0, len(cactus_tokens) - 1].argmax())
        max_diff, mean_diff = diff_stats(cactus_logits[:, len(cactus_tokens) - 1], jax_logits[:, len(cactus_tokens) - 1])
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        cactus_tokens.append(cactus_next)
        cactus_ids[0, len(cactus_tokens) - 1] = cactus_next

    final_mask = np.zeros((1, max_len), dtype=np.float32)
    final_mask[0, : len(cactus_tokens)] = 1.0
    cactus_graph.graph.set_inputs([cactus_ids, final_mask])
    cactus_ms = _time_ms(lambda: (cactus_graph.graph.set_inputs([cactus_ids, final_mask]), cactus_graph.graph.execute()))
    jax_ms = _time_ms(lambda: jit_full(params, jnp.asarray(jax_ids), jnp.asarray(final_mask)).block_until_ready())

    return {
        "torch_text": tokenizer.decode(torch_ids[0]),
        "jax_text": tokenizer.decode(jax_tokens),
        "cactus_text": tokenizer.decode(cactus_tokens),
        "per_step_max_diff": max_diffs,
        "per_step_mean_diff": mean_diffs,
        "jax_jit_ms": jax_ms,
        "cactus_set_input_execute_ms": cactus_ms,
    }


def reloaded_graph_run(probe_path):
    probe = json.loads(Path(probe_path).read_text())
    arrays = np.load(probe["arrays"])
    loaded = Graph.load(probe["graph_path"])
    for binding in probe["bindings"]:
        node = Tensor(loaded, int(binding["node_id"]), (), Graph.FP16)
        loaded.bind_mmap_weights(node, binding["path"])
    loaded.set_input(
        Tensor(loaded, probe["runtime_input_node_ids"][0], arrays["input_ids"].shape, Graph.FP32),
        arrays["input_ids"],
    )
    loaded.set_input(
        Tensor(loaded, probe["runtime_input_node_ids"][1], arrays["attention_mask"].shape, Graph.FP32),
        arrays["attention_mask"],
    )
    loaded.execute()
    return [
        Tensor(loaded, node_id, shape, dtype).numpy()
        for node_id, shape, dtype in probe["output_specs"]
    ]


def reload_only() -> None:
    outputs = reloaded_graph_run(sys.argv[2])
    print(float(np.asarray(outputs[0]).reshape(-1)[0]))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    torch_model, params = load_params(np.float16)
    prompt_ids = tokenizer(PROMPT, return_tensors="np")["input_ids"][0].astype(np.int32).tolist()
    prompt_len = len(prompt_ids)
    max_len = prompt_len + MAX_NEW_TOKENS

    full_ids = np.zeros((1, max_len), dtype=np.int32)
    full_ids[0, :prompt_len] = prompt_ids
    full_mask = np.zeros((1, max_len), dtype=np.float32)
    full_mask[0, :prompt_len] = 1.0
    prefill_ids = np.asarray([prompt_ids], dtype=np.int32)
    prefill_mask = np.ones((1, prompt_len), dtype=np.float32)

    prefill_ref = decoder_prefill(params, jnp.asarray(prefill_ids), jnp.asarray(prefill_mask))
    first_token = np.asarray(prefill_ref[0])[0, -1].argmax().astype(np.int32)
    step_ids = np.asarray([[first_token]], dtype=np.int32)
    step_mask = np.ones((1, prompt_len + 1), dtype=np.float32)
    step_positions = np.asarray([[prompt_len]], dtype=np.int32)
    step_ref = decoder_step(
        params,
        jnp.asarray(step_ids),
        jnp.asarray(step_positions),
        prefill_ref[1],
        prefill_ref[2],
        jnp.asarray(step_mask),
    )

    output_dir = Path("/private/tmp/cactus_distilgpt2_jax_generation_probe")
    result = build_jax_user_graph_bundle(
        params=params,
        specs=[
            JaxGraphSpec(
                name="full_context",
                fn=full_context_logits,
                example_args=(jnp.asarray(full_ids), jnp.asarray(full_mask)),
                input_names=("input_ids", "attention_mask"),
                output_names=("logits",),
            ),
            JaxGraphSpec(
                name="decoder_prefill",
                fn=decoder_prefill,
                example_args=(jnp.asarray(prefill_ids), jnp.asarray(prefill_mask)),
                input_names=("input_ids", "attention_mask"),
                output_names=("logits", "cache_k", "cache_v"),
            ),
            JaxGraphSpec(
                name="decoder_step",
                fn=decoder_step,
                example_args=(
                    jnp.asarray(step_ids),
                    jnp.asarray(step_positions),
                    prefill_ref[1],
                    prefill_ref[2],
                    jnp.asarray(step_mask),
                ),
                input_names=("input_ids", "position_ids", "cache_k", "cache_v", "attention_mask"),
                output_names=("logits", "cache_k", "cache_v"),
            ),
        ],
        output_dir=output_dir,
        model_id=MODEL_ID,
        task="text-generation",
        inputs_metadata={"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS},
        graph_meta={"probe": "distilgpt2_generation"},
    )

    full = greedy_full_context(tokenizer, torch_model, params, result.bundle.graphs["full_context"], prompt_ids, max_len)

    prefill_graph = result.bundle.graphs["decoder_prefill"]
    prefill_graph.graph.set_inputs([prefill_ids, prefill_mask])
    prefill_actual = numpy_outputs(prefill_graph.graph.execute())
    prefill_logits_max, prefill_logits_mean = diff_stats(prefill_actual[0], np.asarray(prefill_ref[0]))
    prefill_k_max, prefill_k_mean = diff_stats(prefill_actual[1], np.asarray(prefill_ref[1]))

    step_graph = result.bundle.graphs["decoder_step"]
    step_graph.graph.set_inputs([step_ids, step_positions, prefill_actual[1], prefill_actual[2], step_mask])
    step_actual = numpy_outputs(step_graph.graph.execute())
    step_logits_max, step_logits_mean = diff_stats(step_actual[0], np.asarray(step_ref[0]))
    step_k_max, step_k_mean = diff_stats(step_actual[1], np.asarray(step_ref[1]))

    reload_status = "not_checked"
    try:
        manifest = json.loads((output_dir / "components" / "manifest.json").read_text())
        prefill_component = next(c for c in manifest["components"] if c["component"] == "decoder_prefill")
        graph_path = output_dir / prefill_component["graph"]
        output_specs = [
            (node_id, tuple(tensor.shape), int(tensor.dtype))
            for node_id, tensor in zip(prefill_component["output_node_ids"], prefill_graph.graph.outputs, strict=True)
        ]
        arrays_path = output_dir / "reload_probe_inputs.npz"
        np.savez(arrays_path, input_ids=prefill_ids, attention_mask=prefill_mask)
        probe_path = output_dir / "reload_probe.json"
        probe_path.write_text(
            json.dumps(
                {
                    "graph_path": str(graph_path),
                    "arrays": str(arrays_path),
                    "runtime_input_node_ids": prefill_component["runtime_input_node_ids"],
                    "output_specs": output_specs,
                    "bindings": prefill_component["bound_constant_bindings"],
                },
                indent=2,
            )
        )
        completed = subprocess.run(
            [sys.executable, __file__, "--reload-only", str(probe_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode == 0:
            reload_status = "ok"
        else:
            stderr_tail = completed.stderr.strip().splitlines()[-1:] or completed.stdout.strip().splitlines()[-1:]
            detail = stderr_tail[0] if stderr_tail else "no stderr"
            reload_status = f"execute failed in subprocess: returncode={completed.returncode}, {detail}"
    except Exception as exc:
        reload_status = f"failed: {type(exc).__name__}: {exc}"

    print("\nFull-context greedy generation:")
    print(f"  torch:  {full['torch_text']!r}")
    print(f"  jax:    {full['jax_text']!r}")
    print(f"  cactus: {full['cactus_text']!r}")
    print(f"  per-step max diff:  {[round(v, 6) for v in full['per_step_max_diff']]}")
    print(f"  per-step mean diff: {[round(v, 6) for v in full['per_step_mean_diff']]}")
    print(f"  jax jit fixed-shape execute:       {full['jax_jit_ms']:.4f} ms")
    print(f"  cactus set_input+execute:          {full['cactus_set_input_execute_ms']:.4f} ms")

    print("\nSplit prefill/decode:")
    print(f"  prefill logits diff: max={prefill_logits_max:.6f}, mean={prefill_logits_mean:.6f}")
    print(f"  prefill cache-k diff: max={prefill_k_max:.6f}, mean={prefill_k_mean:.6f}")
    print(f"  prefill next token: {tokenizer.decode([int(first_token)])!r}")
    print(f"  decode logits diff: max={step_logits_max:.6f}, mean={step_logits_mean:.6f}")
    print(f"  decode cache-k diff: max={step_k_max:.6f}, mean={step_k_mean:.6f}")
    print(f"  decode next token: {tokenizer.decode([int(np.asarray(step_actual[0])[0, -1].argmax())])!r}")

    print("\nSaved bundle/reload:")
    print(f"  output_dir: {output_dir}")
    print(f"  components: {result.components_manifest_path}")
    print(f"  reload: {reload_status}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reload-only":
        reload_only()
    else:
        main()
