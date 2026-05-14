#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/venv/bin/python3"
CACTUS="$ROOT/venv/bin/cactus"
CONFIG="$ROOT/tests/bench/e2e_models.json"
OUTFILE="$ROOT/e2e_decode_results.csv"
PROMPT=""
PROMPT_SUITE="mixed"
LONG_CONTEXT_REPEATS=24
MAX_TOKENS=128
REPS=5
WARMUP=2
MODEL_FILTER=""
TEMPERATURE=0
TOP_P=0.95
TOP_K=1
MIN_P=0.15
SEED=0
MTP_MAX_DRAFTS="0"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --config PATH       Model config JSON (default: tests/bench/e2e_models.json)
  --output PATH       CSV output path (default: e2e_decode_results.csv)
  --model NAME        Run only one model name from the config
  --prompt TEXT       Run one custom prompt instead of the default prompt suite
  --prompt-suite NAME Prompt suite when --prompt is not set: mixed, long_context, all (default: mixed)
  --long-context-repeats N Deprecated; long_context now uses one fixed natural prose prompt
  --max-tokens N      Decode tokens per rep (default: 128)
  --reps N            Measured reps per model (default: 5)
  --warmup N          Warmup reps per model (default: 2)
  --temperature F     Sampling temperature (default: 0)
  --top-p F           Top-p sampling threshold (default: 0.95)
  --top-k N           Top-k sampling threshold (default: 1)
  --min-p F           Min-p sampling threshold (default: 0.15)
  --seed N            Sampling seed; 0 uses random seed (default: 0)
  --mtp-max-draft N   Max draft tokens per MTP round; 0 disables MTP
  --mtp-max-drafts L  Comma-separated draft run list (default: 0)
  --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTFILE="$2"
            shift 2
            ;;
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --prompt-suite)
            PROMPT_SUITE="$2"
            shift 2
            ;;
        --long-context-repeats)
            LONG_CONTEXT_REPEATS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --reps)
            REPS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --min-p)
            MIN_P="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --mtp-max-draft)
            MTP_MAX_DRAFTS="$2"
            shift 2
            ;;
        --mtp-max-drafts)
            MTP_MAX_DRAFTS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing venv Python at $PYTHON" >&2
    exit 1
fi

if [[ ! -x "$CACTUS" ]]; then
    echo "Missing cactus CLI at $CACTUS" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Model config not found: $CONFIG" >&2
    exit 1
fi

case "$PROMPT_SUITE" in
    mixed|long_context|all)
        ;;
    *)
        echo "--prompt-suite must be one of: mixed, long_context, all" >&2
        exit 1
        ;;
esac

if ! [[ "$LONG_CONTEXT_REPEATS" =~ ^[0-9]+$ ]] || [[ "$LONG_CONTEXT_REPEATS" -le 0 ]]; then
    echo "--long-context-repeats must be positive" >&2
    exit 1
fi

IFS=',' read -r -a MTP_DRAFT_RAW_VALUES <<< "$MTP_MAX_DRAFTS"
MTP_DRAFT_VALUES=()
for draft in "${MTP_DRAFT_RAW_VALUES[@]}"; do
    draft="${draft//[[:space:]]/}"
    if [[ -z "$draft" ]]; then
        continue
    fi
    if ! [[ "$draft" =~ ^[0-9]+$ ]]; then
        echo "--mtp-max-drafts must be a comma-separated list of non-negative integers" >&2
        exit 1
    fi
    MTP_DRAFT_VALUES+=("$draft")
done

if [[ "${#MTP_DRAFT_VALUES[@]}" -eq 0 ]]; then
    echo "--mtp-max-drafts must include at least one draft value" >&2
    exit 1
fi

source "$ROOT/venv/bin/activate"
cactus build

SRC="$ROOT/tests/bench/e2e_cactus_decode.cpp"
BIN="$ROOT/tests/build/e2e_cactus_decode"
LIB="$ROOT/cactus/build/libcactus.a"
mkdir -p "$ROOT/tests/build"

if [[ ! -f "$LIB" ]]; then
    echo "Cactus library not found after build: $LIB" >&2
    exit 1
fi

echo "Compiling e2e decode benchmark..."
if [[ "$(uname)" == "Darwin" ]]; then
    CXX="${CXX:-clang++}"
    if [[ -f "$ROOT/libs/curl/macos/libcurl.a" ]]; then
        CURL_ARGS=("$ROOT/libs/curl/macos/libcurl.a")
    elif [[ -f "$ROOT/cactus-engine/libs/curl/macos/libcurl.a" ]]; then
        CURL_ARGS=("$ROOT/cactus-engine/libs/curl/macos/libcurl.a")
    else
        CURL_ARGS=("-lcurl")
    fi
    "$CXX" -std=c++20 -O3 \
        -I"$ROOT" \
        -I"$ROOT/cactus" \
        -I"$ROOT/cactus-engine" \
        -I"$ROOT/cactus-graph" \
        -I"$ROOT/cactus-kernels" \
        "$SRC" "$LIB" \
        -o "$BIN" \
        "${CURL_ARGS[@]}" \
        -framework Accelerate \
        -framework CoreML \
        -framework Foundation \
        -framework Security \
        -framework SystemConfiguration \
        -framework CFNetwork
else
    CXX="${CXX:-g++}"
    "$CXX" -std=c++20 -O3 \
        -I"$ROOT" \
        -I"$ROOT/cactus" \
        -I"$ROOT/cactus-engine" \
        -I"$ROOT/cactus-graph" \
        -I"$ROOT/cactus-kernels" \
        "$SRC" "$LIB" \
        -o "$BIN" \
        -lcurl \
        -pthread
fi

PROMPT_NAMES=()
PROMPTS=()
if [[ -n "$PROMPT" ]]; then
    PROMPT_NAMES=("custom")
    PROMPTS=("$PROMPT")
else
    mapfile -t PROMPT_ROWS < <("$PYTHON" - "$PROMPT_SUITE" "$LONG_CONTEXT_REPEATS" <<'PY'
import sys

suite = sys.argv[1]
_repeat = int(sys.argv[2])

mixed = [
    ("json_object", "Return a JSON object with keys name, role, and status."),
    ("python_function", "Write a short Python function that adds two numbers."),
    ("explain_schrodinger", "Simply explain Schrodinger's cat."),
    ("rank_tradeoffs", "Compare memory bandwidth, arithmetic intensity, and cache locality for a small-batch transformer decode step. Rank which bottleneck matters most and justify the ranking."),
    ("debug_plan", "A speculative decoder is fast on repetitive prompts but slower on open-ended reasoning prompts. Give a concrete debugging plan with measurements, hypotheses, and expected failure modes."),
    ("constrained_json", "Return a compact JSON object with keys decision, confidence, risks, and next_steps for whether to enable speculative decoding by default on mixed workloads."),
    ("code_review", "Review this C++ snippet for correctness risks and performance traps: for (int i = 0; i <= n; ++i) { total += values[i] * scale; if (total > limit) break; }."),
]

long_context_brief = " ".join([
    "You are reviewing a private launch-readiness memo for a mobile speech product that uses on-device decoding, speculative drafting, and an optional cloud fallback.",
    "The memo covers three weeks of dogfood reports from support agents, QA engineers, and two external pilot customers.",
    "It is intentionally messy: some notes are measurements, some are opinions, and some contradict each other because the teams tested different builds.",
    "Do not assume that the most recent paragraph is the most accurate one.",
    "In week one, the iOS team reported that ordinary dictation felt faster after speculative decoding was enabled, especially for short messages and repeated corrections.",
    "Their dashboard showed median end-to-end latency down by 18 percent, but the same dashboard excluded sessions that switched languages mid-sentence.",
    "The Android team saw a smaller 7 percent latency gain and warned that battery drain rose during long meetings.",
    "A QA note says the Android sample was skewed toward older phones with thermal throttling already active.",
    "The platform lead argued that the battery signal should not block the launch because the device was already under stress before the decoder started.",
    "A support lead disagreed and wrote that users do not care whether the extra heat was caused by the decoder or by background sync.",
    "The assistant draft model improved speed on templated emails, calendar edits, and short search queries.",
    "It did poorly on legal dictation, medical terminology, code snippets, and speech with named entities that appeared only once.",
    "One pilot customer, Northstar Clinics, said the model often accepted the first two words of a phrase and then rejected the third, causing a subtle rhythm change that made clinicians pause.",
    "Another pilot, Harbor Logistics, praised the same rhythm because warehouse notes often reused stock phrases.",
    "The product manager wants one default policy, but the field team wants a policy that depends on context length, domain, and recent rejection rate.",
    "A kernel engineer traced the first multi-token verify path and found that target verification was dominated by a small-batch quantized matmul kernel, not attention.",
    "After a focused optimization, the M equals 2 verifier improved enough that one-draft speculation became profitable on mixed internal prompts.",
    "However, the benchmark suite still contained repetitive prompts that gave unrealistically high acceptance, including simple counting and identity questions.",
    "The benchmark owner removed those prompts and added harder tasks involving JSON structure, debugging plans, code review, and tradeoff ranking.",
    "Those harder prompts reduced acceptance but still showed a modest average speedup.",
    "The cloud fallback classifier is not ready for launch.",
    "It catches some low-confidence sessions, but it also sends too many private utterances to the cloud when users dictate names, addresses, and support case numbers.",
    "Legal approved the fallback only if it is opt-in for external pilots and disabled for regulated healthcare accounts.",
    "The privacy reviewer also asked for clearer logs explaining why a session was handed off.",
    "The model quality team found three correctness risks.",
    "First, speculative acceptance was too high on boilerplate where the assistant copied a phrase but missed a negation later in the sentence.",
    "Second, sampled decoding occasionally diverged after a rejected draft because a cache rollback bug had survived in an experimental branch.",
    "Third, benchmark summaries sometimes mixed prefill throughput with decode throughput and made the launch look safer than it was.",
    "The cache rollback bug is fixed on main, but the team has not rerun every long-context smoke test since the fix.",
    "The release manager is deciding between three options.",
    "Option A ships MTP enabled by default for all non-healthcare customers, with cloud fallback disabled.",
    "Option B ships MTP disabled by default while allowing pilots to enable a fixed draft limit after local validation.",
    "Option C delays MTP and ships only the baseline decoder plus extra telemetry.",
    "Marketing prefers option A because it gives the clearest speed story.",
    "Support prefers option B because it gives them a reasoned answer when customers report odd pauses.",
    "The infrastructure team prefers option C because telemetry dashboards are still hard to interpret.",
    "A senior engineer wrote that option B is the only plan that respects both the kernel win and the remaining product risk.",
    "The same engineer warned that unclear rollout criteria can hide regressions, so dashboards must report both enabled time and accepted-token rate.",
    "A customer success manager added that pilots will forgive a small speedup if the product behaves consistently, but they will escalate quickly if the cursor appears to hesitate after rare names or compliance terms.",
    "The QA lead wants one more run that separates long-context dictation from short command-and-control tasks because the current aggregate hides the exact category where users notice pauses.",
    "The engineering manager agreed, but said the launch decision cannot wait for a perfect benchmark and must rely on the best risk-adjusted evidence available this week.",
    "Your task: using only this memo, recommend one launch option, identify the two strongest technical reasons for that option, identify the strongest objection to it, and specify the first three measurements that must appear on the launch dashboard.",
])

long_context = [
    (
        "long_context_1k_triage",
        long_context_brief,
    ),
]

if suite == "mixed":
    selected = mixed
elif suite == "long_context":
    selected = long_context
elif suite == "all":
    selected = mixed + long_context
else:
    raise SystemExit(f"Unknown prompt suite: {suite}")

for name, prompt in selected:
    if "\t" in name or "\t" in prompt or "\n" in prompt:
        raise SystemExit("Prompt names and prompts must be single-line TSV-safe strings")
    print(f"{name}\t{prompt}")
PY
)
    for row in "${PROMPT_ROWS[@]}"; do
        IFS=$'\t' read -r prompt_name prompt_text <<< "$row"
        PROMPT_NAMES+=("$prompt_name")
        PROMPTS+=("$prompt_text")
    done
fi

if [[ "${#PROMPTS[@]}" -eq 0 ]]; then
    echo "No prompts selected" >&2
    exit 1
fi

echo "backend,model,prompt,shape,mtp_max_draft,rep,prefill_tokens,decode_tokens,prefill_tps,decode_tps,ttft_ms,total_ms,mtp_requested,mtp_enabled,mtp_drafted_tokens,mtp_accepted_tokens,mtp_rejected_tokens,mtp_rounds,mtp_fallback_reason,assistant_draft_ms,target_verify_ms,sampling_or_argmax_ms,kv_transaction_ms,callback_stream_ms" > "$OUTFILE"

mapfile -t MODELS < <("$PYTHON" - "$ROOT" "$CONFIG" "$MODEL_FILTER" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).expanduser()
model_filter = sys.argv[3]

if not config_path.is_absolute():
    config_path = (root / config_path).resolve()

with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)

for model in config.get("models", []):
    name = model.get("name", "")
    if model_filter and name != model_filter:
        continue
    path = model.get("path")
    if path is None:
        variants = model.get("variants", {})
        path = variants.get("cactus")
    if not name or not path:
        continue
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    print(f"{name}\t{path}")
PY
)

ran=0
for entry in "${MODELS[@]}"; do
    IFS=$'\t' read -r model_name model_path <<< "$entry"
    if [[ ! -d "$model_path" ]]; then
        echo "Skipping $model_name; missing $model_path"
        continue
    fi

    for idx in "${!PROMPTS[@]}"; do
        prompt_name="${PROMPT_NAMES[$idx]}"
        prompt_text="${PROMPTS[$idx]}"
        for draft in "${MTP_DRAFT_VALUES[@]}"; do
            echo ""
            echo "Benchmarking $model_name / $prompt_name / draft $draft"
            "$BIN" \
                --model-name "$model_name" \
                --model-path "$model_path" \
                --prompt-name "$prompt_name" \
                --prompt "$prompt_text" \
                --max-tokens "$MAX_TOKENS" \
                --reps "$REPS" \
                --warmup "$WARMUP" \
                --temperature "$TEMPERATURE" \
                --top-p "$TOP_P" \
                --top-k "$TOP_K" \
                --min-p "$MIN_P" \
                --seed "$SEED" \
                --mtp-max-draft "$draft" >> "$OUTFILE"
        done
    done
    ran=$((ran + 1))
done

if [[ "$ran" -eq 0 ]]; then
    echo "No configured model paths were found. Edit $CONFIG or pass --config." >&2
    exit 1
fi

echo ""
echo "Results saved to $OUTFILE"
"$PYTHON" - "$OUTFILE" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
by_case = defaultdict(list)
for row in rows:
    by_case[(row["model"], row.get("prompt", "custom"), row.get("shape") or row.get("mtp_max_draft", ""))].append(row)

print()
print(f"{'Model':<20} {'Prompt':<18} {'Shape':>10} {'Mean TPS':>10} {'Min TPS':>9} {'+/-':>8} {'Accept %':>9} {'Rejects':>8} {'TTFT ms':>10} {'Fallback':<18}")
print("-" * 130)
for (model, prompt, shape), vals in by_case.items():
    decode = [float(v["decode_tps"]) for v in vals]
    ttft = [float(v["ttft_ms"]) for v in vals]
    drafted = sum(float(v.get("mtp_drafted_tokens", 0) or 0) for v in vals)
    accepted = sum(float(v.get("mtp_accepted_tokens", 0) or 0) for v in vals)
    rejected = sum(float(v.get("mtp_rejected_tokens", 0) or 0) for v in vals)
    d_mean = statistics.mean(decode)
    d_min = min(decode)
    d_std = statistics.stdev(decode) if len(decode) > 1 else 0.0
    t_mean = statistics.mean(ttft)
    accept_rate = (accepted / drafted * 100.0) if drafted else 0.0
    fallbacks = sorted({v.get("mtp_fallback_reason", "") for v in vals if v.get("mtp_fallback_reason", "")})
    fallback = "|".join(fallbacks)[:18]
    print(f"{model:<20} {prompt:<18} {shape:>10} {d_mean:>10.1f} {d_min:>9.1f} {d_std:>8.1f} {accept_rate:>8.1f}% {rejected:>8.0f} {t_mean:>10.1f} {fallback:<18}")
PY
