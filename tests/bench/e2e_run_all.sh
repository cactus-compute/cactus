#!/bin/bash
# Master E2E benchmark script — runs ALL backends (C++ and Python) on ALL models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
WEIGHTS_DIR="$SCRIPT_DIR/../../weights"
CSV_OUT="${1:-$SCRIPT_DIR/e2e_results.csv}"

PROMPT="Write out the first 1k tokens of Romeo and Juliet's 1st chapter:"
MAX_TOKENS=512
ROUNDS=10

echo "═══════════════════════════════════════════════════════════════"
echo " E2E Generation Benchmark — ALL Backends, CPU Only"
echo " Prompt: ${#PROMPT} chars | Max tokens: $MAX_TOKENS | Rounds: $ROUNDS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# ── 1. Cactus + llama.cpp (C++ binary) — both models ────────────────────────
echo "▶ Running C++ backends (cactus, llama_cpp) on both models..."
"$BUILD_DIR/e2e_bench" \
    --model-config "$SCRIPT_DIR/e2e_models.json" \
    --rounds "$ROUNDS" \
    --max-tokens "$MAX_TOKENS" \
    --prompt "$PROMPT" \
    --output "$TMP_DIR/cpp.csv" \
    2>&1 | grep -E "^\[round|^Model:|^Backend|^----|^cactus|^llama|^=== " || true
echo ""

# ── 2. ONNX Runtime GenAI (Python, CPU only) — qwen3 ────────────────────────
ONNX_QWEN="$WEIGHTS_DIR/qwen3-0.6b-ortgenai"
if [ -d "$ONNX_QWEN" ]; then
    echo "▶ Running ONNX RT (CPU) on qwen3-0.6b..."
    python3 "$SCRIPT_DIR/e2e_bench_onnxrt.py" \
        --model-path "$ONNX_QWEN" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --rounds "$ROUNDS" \
        > "$TMP_DIR/onnxrt_qwen.jsonl" 2>&1 || true
    echo ""
fi

# ── 3. ONNX Runtime GenAI (Python, CPU only) — gemma3 ───────────────────────
ONNX_GEMMA="$WEIGHTS_DIR/gemma3-270m-ortgenai"
if [ -d "$ONNX_GEMMA" ]; then
    echo "▶ Running ONNX RT (CPU) on gemma3-270m..."
    python3 "$SCRIPT_DIR/e2e_bench_onnxrt.py" \
        --model-path "$ONNX_GEMMA" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --rounds "$ROUNDS" \
        --model-name gemma3-270m \
        > "$TMP_DIR/onnxrt_gemma.jsonl" 2>&1 || true
    echo ""
fi

# ── 4. ExecuTorch (Python, CPU only) — qwen3 only ───────────────────────────
ET_PTE="$WEIGHTS_DIR/qwen3-0.6b.pte"
ET_PARAMS="/tmp/qwen3_0.6b_params.json"
ET_TOKENIZER="$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B -name 'tokenizer.json' 2>/dev/null | head -1)"
ET_TOKENIZER_CONFIG="$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B -name 'tokenizer_config.json' 2>/dev/null | head -1)"
if [ -f "$ET_PTE" ] && [ -f "$ET_PARAMS" ] && [ -n "$ET_TOKENIZER" ]; then
    echo "▶ Running ExecuTorch (CPU) on qwen3-0.6b..."
    python3 "$SCRIPT_DIR/e2e_bench_executorch.py" \
        --pte-path "$ET_PTE" \
        --params-path "$ET_PARAMS" \
        --tokenizer-path "$ET_TOKENIZER" \
        --tokenizer-config-path "$ET_TOKENIZER_CONFIG" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --rounds "$ROUNDS" \
        > "$TMP_DIR/executorch_qwen.jsonl" 2>&1 || true
    echo ""
fi

# ── 4b. ExecuTorch (Python, CPU only) — gemma3 ──────────────────────────────
ET_GEMMA_PTE="$WEIGHTS_DIR/gemma3-270m.pte"
ET_GEMMA_PARAMS="/tmp/gemma3_270m_params.json"
ET_GEMMA_TOKENIZER="$(find ~/.cache/huggingface/hub/models--google--gemma-3-270m-it -name 'tokenizer.json' 2>/dev/null | head -1)"
ET_GEMMA_TOKENIZER_CONFIG="$(find ~/.cache/huggingface/hub/models--google--gemma-3-270m-it -name 'tokenizer_config.json' 2>/dev/null | head -1)"
if [ -f "$ET_GEMMA_PTE" ] && [ -f "$ET_GEMMA_PARAMS" ] && [ -n "$ET_GEMMA_TOKENIZER" ]; then
    echo "▶ Running ExecuTorch (CPU) on gemma3-270m..."
    python3 "$SCRIPT_DIR/e2e_bench_executorch.py" \
        --pte-path "$ET_GEMMA_PTE" \
        --params-path "$ET_GEMMA_PARAMS" \
        --tokenizer-path "$ET_GEMMA_TOKENIZER" \
        --tokenizer-config-path "$ET_GEMMA_TOKENIZER_CONFIG" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --rounds "$ROUNDS" \
        --model-name gemma3-270m \
        > "$TMP_DIR/executorch_gemma.jsonl" 2>&1 || true
    echo ""
fi

# ── 5. LiteRT-LM (CLI, CPU only) — qwen3 ───────────────────────────────────
LITERT_QWEN="$WEIGHTS_DIR/Qwen3-0.6B.litertlm"
if [ -f "$LITERT_QWEN" ]; then
    echo "▶ Running LiteRT-LM (CPU) on qwen3-0.6b..."
    python3 "$SCRIPT_DIR/e2e_bench_litert.py" \
        --model-path "$LITERT_QWEN" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --rounds "$ROUNDS" \
        > "$TMP_DIR/litert_qwen.jsonl" 2>&1 || true
    echo ""
fi

# ── 6. LiteRT-LM (CLI, CPU only) — gemma3 ──────────────────────────────────
LITERT_GEMMA="$WEIGHTS_DIR/gemma3-270m-litert/gemma3-270m-it-q8.litertlm"
if [ -f "$LITERT_GEMMA" ]; then
    echo "▶ Running LiteRT-LM (CPU) on gemma3-270m..."
    python3 "$SCRIPT_DIR/e2e_bench_litert.py" \
        --model-path "$LITERT_GEMMA" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --rounds "$ROUNDS" \
        --model-name gemma3-270m \
        > "$TMP_DIR/litert_gemma.jsonl" 2>&1 || true
    echo ""
fi

# ── Merge all results ───────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo " Merging results..."
echo "═══════════════════════════════════════════════════════════════"

python3 << 'PYEOF'
import csv, json, sys, os, statistics

records = []

# Read C++ CSV
cpp_csv = os.environ.get('TMP_DIR', '/tmp') + '/cpp.csv'
for d in [os.environ.get('TMP_DIR', '/tmp')]:
    p = os.path.join(d, 'cpp.csv')
    if os.path.exists(p):
        cpp_csv = p
        break

PYEOF

# Use a simpler approach - inline Python with the TMP_DIR
python3 -c "
import csv, json, sys, os, statistics

records = []
tmp = '$TMP_DIR'

# Read C++ CSV
cpp_csv = os.path.join(tmp, 'cpp.csv')
if os.path.exists(cpp_csv):
    with open(cpp_csv) as f:
        for row in csv.DictReader(f):
            records.append(row)

# Read all JSONL files
for jf in ['onnxrt_qwen.jsonl', 'onnxrt_gemma.jsonl', 'executorch_qwen.jsonl', 'executorch_gemma.jsonl', 'litert_qwen.jsonl', 'litert_gemma.jsonl']:
    path = os.path.join(tmp, jf)
    if not os.path.exists(path):
        continue
    model_name = 'qwen3-0.6b' if 'qwen' in jf else 'gemma3-270m'
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                m = d.get('model', model_name)
                records.append({
                    'backend': d['backend'], 'model': m,
                    'rep': str(d['rep']),
                    'prefill_tokens': str(d['prefill_tokens']),
                    'decode_tokens': str(d['decode_tokens']),
                    'prefill_tps': str(d['prefill_tps']),
                    'decode_tps': str(d['decode_tps']),
                    'ttft_ms': str(d['ttft_ms']),
                    'total_ms': str(d['total_ms']),
                })
            except: pass

csv_out = '$CSV_OUT'
with open(csv_out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['backend','model','rep','prefill_tokens','decode_tokens','prefill_tps','decode_tps','ttft_ms','total_ms'])
    w.writeheader()
    w.writerows(records)
print(f'Wrote {len(records)} records to {csv_out}')

# Print summary per model
models = sorted(set(r['model'] for r in records))
for model in models:
    model_records = [r for r in records if r['model'] == model]
    by_backend = {}
    order = []
    for r in model_records:
        b = r['backend']
        if b not in by_backend:
            by_backend[b] = {'decode_tps': [], 'prefill_tps': [], 'ttft_ms': []}
            order.append(b)
        by_backend[b]['decode_tps'].append(float(r['decode_tps']))
        by_backend[b]['prefill_tps'].append(float(r['prefill_tps']))
        by_backend[b]['ttft_ms'].append(float(r['ttft_ms']))

    cactus_decode = statistics.mean(by_backend.get('cactus', by_backend[order[0]])['decode_tps'])

    print()
    print(f'Model: {model}  |  Max tokens: $MAX_TOKENS  |  $ROUNDS rounds')
    print()
    print(f'{\"Backend\":<18} {\"Decode TPS\":>12} {\"±\":>8} {\"Prefill TPS\":>14} {\"±\":>8} {\"TTFT\":>10} {\"vs Cactus\":>14}')
    print('-' * 84)
    for b in order:
        s = by_backend[b]
        d_mean = statistics.mean(s['decode_tps'])
        d_std = statistics.stdev(s['decode_tps']) if len(s['decode_tps']) > 1 else 0
        p_mean = statistics.mean(s['prefill_tps'])
        p_std = statistics.stdev(s['prefill_tps']) if len(s['prefill_tps']) > 1 else 0
        t_mean = statistics.mean(s['ttft_ms'])
        vs = 'baseline' if b == 'cactus' else (f'{((d_mean - cactus_decode) / cactus_decode) * 100:+.1f}%' if cactus_decode > 0 else 'n/a')
        print(f'{b:<18} {d_mean:>12.1f} {d_std:>8.1f} {p_mean:>14.1f} {p_std:>8.1f} {t_mean:>7.0f}ms {vs:>14}')
    print()
"

echo "Results saved to: $CSV_OUT"
