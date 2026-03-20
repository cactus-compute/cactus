#!/usr/bin/env python3
"""ONNX Runtime GenAI E2E benchmark — CPU-only via raw onnxruntime session."""
import sys
import time
import json
import os
import numpy as np

import onnxruntime as ort


def load_tokenizer(model_dir):
    """Load tokenizer from genai config using onnxruntime_genai."""
    import onnxruntime_genai as og
    # We only use og for tokenizer — model inference uses raw ort with CPU EP
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    return tokenizer, model


def benchmark(model_dir, prompt, max_tokens, rounds):
    print(f"[onnxrt] Loading model from {model_dir} (CPU-only)...", file=sys.stderr)

    tokenizer, og_model = load_tokenizer(model_dir)

    # Load ONNX model with ONLY CPUExecutionProvider
    model_path = os.path.join(model_dir, "model.onnx")
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = os.cpu_count()
    sess_opts.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )

    active_providers = sess.get_providers()
    print(f"[onnxrt] Active providers: {active_providers}", file=sys.stderr)
    assert active_providers == ["CPUExecutionProvider"], f"Not CPU-only! Got: {active_providers}"

    # Get model info
    config_path = os.path.join(model_dir, "genai_config.json")
    with open(config_path) as f:
        config = json.load(f)
    num_layers = config["model"]["decoder"]["num_hidden_layers"]
    num_kv_heads = config["model"]["decoder"]["num_key_value_heads"]
    head_size = config["model"]["decoder"]["head_size"]
    eos_tokens = config["model"].get("eos_token_id", [151645])
    if isinstance(eos_tokens, int):
        eos_tokens = [eos_tokens]

    results = []

    # Warmup
    print("[onnxrt] Warmup...", file=sys.stderr)
    _run_generation(sess, tokenizer, prompt, 16, num_layers, num_kv_heads, head_size, eos_tokens)

    for r in range(rounds):
        res = _run_generation(sess, tokenizer, prompt, max_tokens, num_layers, num_kv_heads, head_size, eos_tokens)
        results.append({
            "backend": "onnxrt",
            "rep": r,
            **res
        })
        print(f"[onnxrt] round {r+1}: decode={res['decode_tps']:.1f} tps, prefill={res['prefill_tps']:.1f} tps",
              file=sys.stderr)

    return results


def _run_generation(sess, tokenizer, prompt, max_tokens, num_layers, num_kv_heads, head_size, eos_tokens):
    """Run one generation with explicit KV cache management."""
    input_ids = tokenizer.encode(prompt)
    n_prompt = len(input_ids)

    # Initialize KV cache (empty)
    past_kv = {}
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros((1, num_kv_heads, 0, head_size), dtype=np.float32)
        past_kv[f"past_key_values.{i}.value"] = np.zeros((1, num_kv_heads, 0, head_size), dtype=np.float32)

    seq_len = 0
    generated_tokens = []

    wall_start = time.perf_counter()

    # Prefill: process all input tokens at once
    prefill_start = time.perf_counter()

    feed = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "attention_mask": np.ones((1, n_prompt), dtype=np.int64),
    }
    feed.update(past_kv)

    outputs = sess.run(None, feed)
    prefill_end = time.perf_counter()

    # Parse outputs: logits + new KV cache
    logits = outputs[0]  # (1, seq_len, vocab_size)
    new_past_kv = {}
    for i in range(num_layers):
        new_past_kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
        new_past_kv[f"past_key_values.{i}.value"] = outputs[1 + i * 2 + 1]

    # Greedy: take argmax of last token logits
    next_token = int(np.argmax(logits[0, -1, :]))
    generated_tokens.append(next_token)
    seq_len = n_prompt

    prefill_ms = (prefill_end - prefill_start) * 1000.0

    # Decode loop
    decode_start = time.perf_counter()

    for _ in range(max_tokens - 1):
        if next_token in eos_tokens:
            break

        seq_len += 1
        feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        }
        feed.update(new_past_kv)

        outputs = sess.run(None, feed)
        logits = outputs[0]
        for i in range(num_layers):
            new_past_kv[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            new_past_kv[f"past_key_values.{i}.value"] = outputs[1 + i * 2 + 1]

        next_token = int(np.argmax(logits[0, -1, :]))
        generated_tokens.append(next_token)

    decode_end = time.perf_counter()
    wall_end = time.perf_counter()

    decode_ms = (decode_end - decode_start) * 1000.0
    n_gen = len(generated_tokens)

    return {
        "prefill_tokens": n_prompt,
        "decode_tokens": n_gen,
        "prefill_tps": round(n_prompt * 1000.0 / prefill_ms if prefill_ms > 0 else 0, 1),
        "decode_tps": round(n_gen * 1000.0 / decode_ms if decode_ms > 0 else 0, 1),
        "ttft_ms": round(prefill_ms, 1),
        "total_ms": round((wall_end - wall_start) * 1000.0, 1),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="qwen3-0.6b")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    results = benchmark(args.model_path, args.prompt, args.max_tokens, args.rounds)
    for r in results:
        print(json.dumps(r))
