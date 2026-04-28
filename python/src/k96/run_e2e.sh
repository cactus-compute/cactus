#!/bin/bash
# End-to-end pipeline for the gemma4-e2b-grouped-k96 model on cactus.
# Run from the cactus repo root after activating the python venv.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
RAW=${REPO}/weights/k96_raw
OUT=${REPO}/weights/gemma-4-e2b-it-k96
BASE=${REPO}/weights/gemma-4-e2b-it
CKPT=${RAW}/Sw_grouped_50_K96_lora_long.pt
GROUPS=${RAW}/groups

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: ${CKPT} not found. Run the HF download first." >&2
  exit 1
fi

echo "==> Converting to cactus K96 INT8 weights..."
python -m python.src.k96.convert_k96 \
  --base_dir "${BASE}" \
  --ckpt "${CKPT}" \
  --groups_dir "${GROUPS}" \
  --out_dir "${OUT}"

echo
echo "==> Done. Model dir: ${OUT}"
echo "    Test with: ./tests/build/chat ${OUT} --prompt 'What is the capital of France?'"
