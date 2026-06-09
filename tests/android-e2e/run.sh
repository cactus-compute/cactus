#!/bin/bash
# Push the cactus-only e2e benchmark + weights to a connected Android device,
# run it at the given decode budget, and pull the CSV back.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERIAL=${SERIAL:-}
MAX_TOKENS=${MAX_TOKENS:-512}
ROUNDS=${ROUNDS:-10}
PROMPT=${PROMPT:-"Write out the first 1k tokens of Romeo and Juliet's 1st chapter:"}
OUT_CSV=${1:-"$SCRIPT_DIR/e2e_results_android.csv"}
DEVICE_DIR=/data/local/tmp/cactus-e2e

BIN="$SCRIPT_DIR/build/e2e_bench"
[ -x "$BIN" ] || { echo "Error: $BIN not found. Run build.sh first." >&2; exit 1; }

# Model name -> local cactus bundle dir.
MODELS=("qwen3-0.6b:$REPO_ROOT/weights/qwen3-0.6b"
        "gemma3-270m:$REPO_ROOT/weights/gemma-3-270m-it")

ADB=(adb)
[ -n "$SERIAL" ] && ADB=(adb -s "$SERIAL")

n_dev=$("${ADB[@]}" devices | grep -cE "device$") || true
[ "${n_dev:-0}" -ge 1 ] || { echo "Error: no Android device connected." >&2; exit 1; }
echo "Device: $("${ADB[@]}" shell getprop ro.product.model | tr -d '\r') ($("${ADB[@]}" shell getprop ro.product.cpu.abi | tr -d '\r'))"

"${ADB[@]}" shell "mkdir -p $DEVICE_DIR/weights"

echo "==> Pushing binary"
"${ADB[@]}" push "$BIN" "$DEVICE_DIR/e2e_bench" >/dev/null
"${ADB[@]}" shell "chmod +x $DEVICE_DIR/e2e_bench"

# Build device-side model config JSON, pushing each bundle.
CFG="{\n  \"models\": [\n"
first=1
for entry in "${MODELS[@]}"; do
    name="${entry%%:*}"; local_path="${entry#*:}"
    bundle="$(basename "$local_path")"
    [ -d "$local_path" ] || { echo "Error: missing weights $local_path" >&2; exit 1; }
    echo "==> Pushing weights: $name ($bundle)"
    "${ADB[@]}" push "$local_path" "$DEVICE_DIR/weights/" >/dev/null
    [ "$first" -eq 1 ] || CFG+=",\n"
    CFG+="    { \"name\": \"$name\", \"variants\": { \"cactus\": \"$DEVICE_DIR/weights/$bundle\" } }"
    first=0
done
CFG+="\n  ]\n}\n"

TMP_CFG="$(mktemp)"
printf "%b" "$CFG" > "$TMP_CFG"
"${ADB[@]}" push "$TMP_CFG" "$DEVICE_DIR/e2e_models_android.json" >/dev/null
rm -f "$TMP_CFG"

echo "==> Running e2e_bench (max-tokens=$MAX_TOKENS, rounds=$ROUNDS)"
"${ADB[@]}" shell "cd $DEVICE_DIR && ./e2e_bench \
    --model-config $DEVICE_DIR/e2e_models_android.json \
    --backends cactus \
    --rounds $ROUNDS \
    --max-tokens $MAX_TOKENS \
    --prompt '$PROMPT' \
    --output $DEVICE_DIR/e2e_results_android.csv"

echo "==> Pulling results to $OUT_CSV"
"${ADB[@]}" pull "$DEVICE_DIR/e2e_results_android.csv" "$OUT_CSV" >/dev/null
echo "Done: $OUT_CSV"
