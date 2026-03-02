#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python3"
OUTFILE="${1:-e2e_decode_main_vs_accum16.csv}"

MODELS=("qwen3-0.6b")
MAX_TOKENS=256
REPS=3

CURRENT_BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)

echo "branch,model,rep,decode_tps,prefill_tps,decode_tokens,prefill_tokens,total_ms,ttft_ms" > "$OUTFILE"

run_bench() {
    local label="$1"
    echo ""
    echo "=== Benchmarking: $label ==="
    echo "  Building..."
    (cd "$ROOT/cactus/build" && cmake .. >/dev/null 2>&1 && make -j >/dev/null 2>&1)
    echo "  Build done."

    for model_name in "${MODELS[@]}"; do
        local model_path="$ROOT/weights/$model_name"
        if [ ! -d "$model_path" ]; then
            echo "  Skipping $model_name (not found)"
            continue
        fi

        echo "  Model: $model_name"
        for rep in $(seq 0 $((REPS - 1))); do
            result=$("$PYTHON" -c "
import sys, json
sys.path.insert(0, '$ROOT/python/src')
import cactus
m = cactus.CactusModel('$model_path')
msgs = [{'role': 'user', 'content': 'Write a detailed explanation of how transformers work in deep learning, covering the attention mechanism, positional encoding, and the encoder-decoder architecture.'}]
r = json.loads(m.complete(msgs, max_tokens=$MAX_TOKENS, temperature=0.0))
print(json.dumps({
    'decode_tps': r.get('decode_tps', 0),
    'prefill_tps': r.get('prefill_tps', 0),
    'decode_tokens': r.get('decode_tokens', 0),
    'prefill_tokens': r.get('prefill_tokens', 0),
    'total_time_ms': r.get('total_time_ms', 0),
    'time_to_first_token_ms': r.get('time_to_first_token_ms', 0),
}))
del m
" 2>/dev/null)

            decode_tps=$(echo "$result" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['decode_tps'])")
            prefill_tps=$(echo "$result" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['prefill_tps'])")
            decode_tokens=$(echo "$result" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['decode_tokens'])")
            prefill_tokens=$(echo "$result" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['prefill_tokens'])")
            total_ms=$(echo "$result" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['total_time_ms'])")
            ttft_ms=$(echo "$result" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin)['time_to_first_token_ms'])")

            echo "$label,$model_name,$rep,$decode_tps,$prefill_tps,$decode_tokens,$prefill_tokens,$total_ms,$ttft_ms" >> "$OUTFILE"
            printf "    rep %d: decode=%.1f tok/s  prefill=%.1f tok/s  (%s tokens in %.0fms)\n" \
                "$rep" "$decode_tps" "$prefill_tps" "$decode_tokens" "$total_ms"
        done
    done
}

# 1. Benchmark current branch
run_bench "$CURRENT_BRANCH"

# 2. Switch to main, benchmark, switch back
git -C "$ROOT" checkout main --quiet
run_bench "main"
git -C "$ROOT" checkout "$CURRENT_BRANCH" --quiet
# Rebuild current branch so we leave in a good state
(cd "$ROOT/cactus/build" && cmake .. >/dev/null 2>&1 && make -j >/dev/null 2>&1)

echo ""
echo "=== Results saved to $OUTFILE ==="
echo ""

# Print summary table
BENCH_OUTFILE="$OUTFILE" "$PYTHON" << 'PYEOF'
import csv, statistics
from collections import defaultdict

import os; rows = list(csv.DictReader(open(os.environ["BENCH_OUTFILE"])))
by = defaultdict(list)
for r in rows:
    by[(r["branch"], r["model"])].append({
        "decode_tps": float(r["decode_tps"]),
        "prefill_tps": float(r["prefill_tps"]),
    })

models = sorted(set(r["model"] for r in rows))
branches = sorted(set(r["branch"] for r in rows), key=lambda b: b != "main")

print(f"{'Branch':<25} {'Model':<20} {'Decode TPS':>12} {'±':>6} {'Prefill TPS':>13} {'±':>6}")
print("-" * 85)

for model in models:
    main_decode = None
    for branch in branches:
        vals = by.get((branch, model), [])
        if not vals:
            continue
        d = [v["decode_tps"] for v in vals]
        p = [v["prefill_tps"] for v in vals]
        dm, ds = statistics.mean(d), (statistics.stdev(d) if len(d) > 1 else 0)
        pm, ps = statistics.mean(p), (statistics.stdev(p) if len(p) > 1 else 0)

        if branch == "main":
            main_decode = dm
            suffix = ""
        elif main_decode and main_decode > 0:
            pct = (dm - main_decode) / main_decode * 100
            suffix = f"  ({pct:+.1f}%)"
        else:
            suffix = ""

        print(f"{branch:<25} {model:<20} {dm:>10.1f} {ds:>6.1f} {pm:>11.1f} {ps:>6.1f}{suffix}")
    print()
PYEOF
