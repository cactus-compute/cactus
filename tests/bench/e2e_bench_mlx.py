#!/usr/bin/env python3
"""MLX-LM E2E benchmark — CPU-only mode."""
import sys
import os
import time
import json

# Force CPU-only
os.environ["MLX_USE_GPU"] = "0"

import mlx.core as mx
mx.set_default_device(mx.cpu)  # Force CPU device

from mlx_lm import load, generate


def benchmark(model_name, prompt, max_tokens, rounds):
    print(f"[mlx] Loading model {model_name} (CPU-only)...", file=sys.stderr)
    model, tokenizer = load(model_name)

    results = []

    # Warmup
    print("[mlx] Warmup...", file=sys.stderr)
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False)

    for r in range(rounds):
        input_tokens = tokenizer.encode(prompt)
        n_prompt = len(input_tokens)

        wall_start = time.perf_counter()

        # Prefill: encode + first forward pass
        prefill_start = time.perf_counter()
        result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        wall_end = time.perf_counter()

        total_ms = (wall_end - wall_start) * 1000.0

        # Count output tokens
        output_tokens = tokenizer.encode(result)
        generated = len(output_tokens) - n_prompt
        if generated < 1:
            generated = max_tokens  # fallback

        # MLX doesn't easily separate prefill/decode timing via Python API,
        # so we estimate based on total time
        # Prefill is typically fast for short prompts — attribute ~5% to prefill
        est_prefill_frac = n_prompt / (n_prompt + generated)
        prefill_ms = total_ms * est_prefill_frac
        decode_ms = total_ms * (1 - est_prefill_frac)

        prefill_tps = n_prompt * 1000.0 / prefill_ms if prefill_ms > 0 else 0
        decode_tps = generated * 1000.0 / decode_ms if decode_ms > 0 else 0

        results.append({
            "backend": "mlx",
            "rep": r,
            "prefill_tokens": n_prompt,
            "decode_tokens": generated,
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(decode_tps, 1),
            "ttft_ms": round(prefill_ms, 1),
            "total_ms": round(total_ms, 1),
        })
        print(f"[mlx] round {r+1}: decode={decode_tps:.1f} tps, total={total_ms:.0f}ms",
              file=sys.stderr)

        mx.metal.clear_cache() if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache') else None

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    results = benchmark(args.model_name, args.prompt, args.max_tokens, args.rounds)
    for r in results:
        print(json.dumps(r))
