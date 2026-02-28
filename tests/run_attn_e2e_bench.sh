#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="LiquidAI/LFM2-1.2B"
PRECISION="INT4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_NAME="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--model MODEL] [--precision INT4|INT8|FP16]"; exit 1 ;;
    esac
done

MODEL_DIR_NAME=$(echo "${MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]')
MODEL_DIR="weights/$MODEL_DIR_NAME"
CSV_OUT="attn_e2e_results.csv"
PNG_OUT="attn_e2e_comparison.png"

echo "=== Step 1: Build libcactus ==="
cactus build

echo ""
echo "=== Step 2: Download model (if needed) ==="
if [ ! -d "$MODEL_DIR" ]; then
    cactus download "$MODEL_NAME" --precision "$PRECISION"
else
    echo "Model already present at $MODEL_DIR"
fi

echo ""
echo "=== Step 3: Build attn_e2e_bench ==="
cmake -S tests -B tests/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCACTUS_ROOT="$ROOT_DIR/cactus" 2>&1 | tail -5
cmake --build tests/build --target attn_e2e_bench -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo ""
echo "=== Step 4: Run benchmark ==="
tests/build/attn_e2e_bench "$MODEL_DIR" "$CSV_OUT"

echo ""
echo "=== Step 5: Plot results ==="
pip install -q matplotlib numpy 2>/dev/null || true
python3 tests/plot_attn_e2e.py "$CSV_OUT" "$PNG_OUT"

echo ""
echo "=== Done ==="
echo "CSV:  $CSV_OUT"
echo "Plot: $PNG_OUT"
