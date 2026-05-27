#!/usr/bin/env bash
set -euo pipefail

runs="${1:-5}"
audio_file="${AUDIO_FILE:-cactus-engine/tests/assets/test.wav}"

models=(
  "parakeet-tdt-0.6b-v3|weights/e2e-parakeet-tdt-0.6b-v3-cq"
  "parakeet-tdt-1.1b|weights/e2e-parakeet-tdt-1.1b-cq"
  "whisper-tiny|weights/e2e-whisper-tiny-cq"
  "whisper-small|weights/e2e-whisper-small-cq"
  "whisper-medium|weights/e2e-whisper-medium-cq"
  "whisper-large-v3-turbo|weights/e2e-whisper-large-v3-turbo-cq"
)

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

printf "audio=%s runs=%s\n" "$audio_file" "$runs"
printf "model,run,total_time_ms,time_to_first_token_ms,prefill_tokens,decode_tokens,total_tokens,ram_usage_mb,response\n" > "$tmp"

for item in "${models[@]}"; do
  IFS='|' read -r name bundle <<< "$item"
  if [[ ! -d "$bundle" ]]; then
    printf "missing bundle for %s: %s\n" "$name" "$bundle" >&2
    exit 1
  fi

  for ((i = 1; i <= runs; i++)); do
    printf "running %s %d/%d\n" "$name" "$i" "$runs" >&2
    output="$(venv/bin/cactus transcribe "$bundle" --file "$audio_file" --no-cloud-tele)"
    json="$(printf "%s\n" "$output" | tail -n 1)"
    MODEL_NAME="$name" RUN_INDEX="$i" JSON_LINE="$json" python3 - <<'PY' >> "$tmp"
import csv
import json
import os
import sys

try:
    data = json.loads(os.environ["JSON_LINE"])
except Exception as exc:
    print(f"failed to parse JSON for {os.environ['MODEL_NAME']} run {os.environ['RUN_INDEX']}: {exc}", file=sys.stderr)
    raise

writer = csv.writer(sys.stdout)
writer.writerow([
    os.environ["MODEL_NAME"],
    os.environ["RUN_INDEX"],
    data.get("total_time_ms"),
    data.get("time_to_first_token_ms"),
    data.get("prefill_tokens"),
    data.get("decode_tokens"),
    data.get("total_tokens"),
    data.get("ram_usage_mb"),
    data.get("response", "").replace("\n", " "),
])
PY
  done
done

cat "$tmp"
printf "\nsummary\n"
python3 - "$tmp" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="")))
groups = {}
for row in rows:
    groups.setdefault(row["model"], []).append(row)

print("model,runs,total_min,total_mean,total_median,total_max,ttft_min,ttft_mean,ttft_median,ttft_max")
for model, items in groups.items():
    totals = [float(row["total_time_ms"]) for row in items]
    ttfts = [float(row["time_to_first_token_ms"]) for row in items]
    print(",".join([
        model,
        str(len(items)),
        f"{min(totals):.2f}",
        f"{statistics.mean(totals):.2f}",
        f"{statistics.median(totals):.2f}",
        f"{max(totals):.2f}",
        f"{min(ttfts):.2f}",
        f"{statistics.mean(ttfts):.2f}",
        f"{statistics.median(ttfts):.2f}",
        f"{max(ttfts):.2f}",
    ]))
PY
