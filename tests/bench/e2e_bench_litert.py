#!/usr/bin/env python3
"""LiteRT-LM E2E benchmark — uses litert_lm_main CLI with CPU backend."""
import sys
import time
import json
import os
import subprocess
import re

LITERT_BIN = "/Users/noahcylich/Documents/Desert/third_party/litert-lm/sdk/bin/litert_lm_main"
LITERT_LIB = "/Users/noahcylich/Documents/Desert/third_party/litert-lm/sdk/lib"


def benchmark(model_path, prompt, max_tokens, rounds):
    print(f"[litert] Using model: {model_path}", file=sys.stderr)
    print(f"[litert] Binary: {LITERT_BIN}", file=sys.stderr)

    if not os.path.exists(LITERT_BIN):
        print("[litert] Binary not found, skipping", file=sys.stderr)
        return []

    results = []

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = LITERT_LIB + ":" + env.get("DYLD_LIBRARY_PATH", "")

    # Warmup
    print("[litert] Warmup...", file=sys.stderr)
    subprocess.run(
        [LITERT_BIN, "--model_path", model_path, "--backend=cpu",
         "--input_prompt=Hello"],
        env=env, capture_output=True, timeout=120
    )

    for r in range(rounds):
        wall_start = time.perf_counter()
        proc = subprocess.run(
            [LITERT_BIN, "--model_path", model_path, "--backend=cpu",
             f"--input_prompt={prompt}"],
            env=env, capture_output=True, text=True, timeout=300
        )
        wall_end = time.perf_counter()
        total_ms = (wall_end - wall_start) * 1000.0

        output = proc.stdout + proc.stderr

        # Parse metrics from CLI output
        prefill_tps = 0.0
        decode_tps = 0.0
        ttft_ms = 0.0
        prefill_tokens = 0
        decode_tokens = 0

        # "Prefill Speed: 159.08 tokens/sec."
        m = re.search(r"Prefill Speed:\s*([\d.]+)\s*tokens/sec", output)
        if m:
            prefill_tps = float(m.group(1))

        # "Decode Speed: 62.37 tokens/sec."
        m = re.search(r"Decode Speed:\s*([\d.]+)\s*tokens/sec", output)
        if m:
            decode_tps = float(m.group(1))

        # "Time to first token: 0.08 s"
        m = re.search(r"Time to first token:\s*([\d.]+)\s*s", output)
        if m:
            ttft_ms = float(m.group(1)) * 1000.0

        # "Processed 10 tokens in 62.86ms" (prefill)
        m = re.search(r"Prefill Turn 1: Processed (\d+) tokens", output)
        if m:
            prefill_tokens = int(m.group(1))

        # "Processed 12 tokens in 192.413ms" (decode)
        m = re.search(r"Decode Turn 1: Processed (\d+) tokens", output)
        if m:
            decode_tokens = int(m.group(1))

        results.append({
            "backend": "litert",
            "rep": r,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(decode_tps, 1),
            "ttft_ms": round(ttft_ms, 1),
            "total_ms": round(total_ms, 1),
        })
        print(f"[litert] round {r+1}: decode={decode_tps:.1f} tps, prefill={prefill_tps:.1f} tps",
              file=sys.stderr)

    return results


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
