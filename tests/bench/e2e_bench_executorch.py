#!/usr/bin/env python3
"""ExecuTorch E2E benchmark — uses NativeLlamaRunner with .pte model, CPU only."""
import sys
import os
import time
import json
import argparse


def benchmark(pte_path, params_path, tokenizer_path, tokenizer_config_path, prompt, max_tokens, rounds):
    print(f"[executorch] Loading model: {pte_path}", file=sys.stderr)

    from executorch.examples.models.llama.runner.native import NativeLlamaRunner

    results = []

    for r in range(rounds + 1):  # +1 for warmup
        is_warmup = (r == 0)
        gen_len = 16 if is_warmup else max_tokens

        args = argparse.Namespace(
            params=params_path,
            pte=pte_path,
            kv_cache=True,
            max_len=gen_len + 64,  # context length > gen length
            tokenizer=tokenizer_path,
            tokenizer_config=tokenizer_config_path,
        )

        runner = NativeLlamaRunner(args)
        tok = runner.tokenizer
        input_tokens = tok.encode(prompt, bos=True, eos=False)
        n_prompt = len(input_tokens)

        wall_start = time.perf_counter()
        tokens = runner.text_completion(prompt, temperature=0.0)
        wall_end = time.perf_counter()

        total_ms = (wall_end - wall_start) * 1000.0
        n_gen = len(tokens) - n_prompt
        if n_gen < 1:
            n_gen = len(tokens)

        # ExecuTorch prints "Prefill time:" and "Generation tok/s:" — we can also
        # estimate from total time
        prefill_frac = n_prompt / (n_prompt + n_gen)
        prefill_ms = total_ms * prefill_frac
        decode_ms = total_ms * (1 - prefill_frac)

        prefill_tps = n_prompt * 1000.0 / prefill_ms if prefill_ms > 0 else 0
        decode_tps = n_gen * 1000.0 / decode_ms if decode_ms > 0 else 0

        if is_warmup:
            print(f"[executorch] Warmup done ({n_gen} tokens in {total_ms:.0f}ms)", file=sys.stderr)
            continue

        results.append({
            "backend": "executorch",
            "rep": r - 1,
            "prefill_tokens": n_prompt,
            "decode_tokens": n_gen,
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(decode_tps, 1),
            "ttft_ms": round(prefill_ms, 1),
            "total_ms": round(total_ms, 1),
        })
        print(f"[executorch] round {r}: decode={decode_tps:.1f} tps, total={total_ms:.0f}ms",
              file=sys.stderr)

        del runner

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pte-path", required=True)
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-config-path", default=None)
    parser.add_argument("--model-name", default="qwen3-0.6b")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    results = benchmark(
        args.pte_path, args.params_path,
        args.tokenizer_path, args.tokenizer_config_path,
        args.prompt, args.max_tokens, args.rounds
    )
    for r in results:
        print(json.dumps(r))
