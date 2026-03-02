#!/usr/bin/env python3
"""E2E per-token decode timing benchmark. Outputs CSV for graphing."""

import sys, os, json, time, csv, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
import cactus

def run_benchmark(model_path, label, max_tokens=256, reps=3):
    rows = []
    for rep in range(reps):
        m = cactus.CactusModel(model_path)
        token_times = []
        last_time = [None]

        def on_token(token, token_id, user_data):
            now = time.perf_counter()
            if last_time[0] is not None:
                token_times.append(now - last_time[0])
            last_time[0] = now

        msgs = [{"role": "user", "content":
            "Write a detailed explanation of how transformers work in deep learning, "
            "covering the attention mechanism, positional encoding, and the encoder-decoder architecture."}]

        last_time[0] = time.perf_counter()
        result = json.loads(m.complete(msgs, max_tokens=max_tokens, temperature=0.0, callback=on_token))

        for i, dt in enumerate(token_times):
            rows.append({
                "variant": label,
                "rep": rep,
                "token_index": i,
                "time_ms": dt * 1000,
            })

        decode_tps = result.get("decode_tps", 0)
        print(f"  {label} rep {rep}: {decode_tps:.1f} tok/s, {len(token_times)} tokens timed")
        del m

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/qwen3-0.6b")
    parser.add_argument("--label", default="current")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", default="e2e_per_token.csv")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    print(f"Model: {args.model}, label: {args.label}, reps: {args.reps}")
    rows = run_benchmark(args.model, args.label, args.max_tokens, args.reps)

    mode = "a" if args.append else "w"
    write_header = not args.append or not os.path.exists(args.output)

    with open(args.output, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "rep", "token_index", "time_ms"])
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
