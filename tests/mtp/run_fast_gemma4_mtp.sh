#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMEOUT_SECONDS=180
MAX_TOKENS=24

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --max-tokens N          Override the short real-model max token budget (default: 24)
  --timeout-seconds N     Kill the fast suite after N seconds (default: 180)
  --help                  Show this help

Environment:
  CACTUS_TEST_GEMMA4_TARGET     Converted Gemma 4 target model directory
  CACTUS_TEST_GEMMA4_ASSISTANT  Converted Gemma 4 assistant model directory
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --timeout-seconds)
            TIMEOUT_SECONDS="$2"
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

source "$ROOT/venv/bin/activate"
export CACTUS_TEST_GEMMA4_MAX_TOKENS="$MAX_TOKENS"
cactus build

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

mkdir -p "$ROOT/cactus-engine/tests/build"
cmake -S "$ROOT/cactus-engine/tests" -B "$ROOT/cactus-engine/tests/build" \
    -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF >/dev/null
cmake --build "$ROOT/cactus-engine/tests/build" -j "$JOBS"

mkdir -p "$ROOT/cactus-graph/build"
cmake -S "$ROOT/cactus-graph" -B "$ROOT/cactus-graph/build" \
    -DCACTUS_BUILD_TESTS=ON -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF >/dev/null
cmake --build "$ROOT/cactus-graph/build" -j "$JOBS"

run_with_timeout() {
    local exe="$1"
    if [[ ! -x "$exe" ]]; then
        echo "Missing test executable: $exe" >&2
        exit 1
    fi

    if command -v timeout >/dev/null 2>&1; then
        timeout "$TIMEOUT_SECONDS" "$exe"
    else
        "$exe"
    fi
}

ENGINE_BUILD="$ROOT/cactus-engine/tests/build"
GRAPH_BUILD="$ROOT/cactus-graph/build"

echo "Running Gemma 4 MTP fast test suite with max_tokens=$MAX_TOKENS"

run_with_timeout "$ENGINE_BUILD/test_mtp_sampler"
run_with_timeout "$GRAPH_BUILD/test_cache_transaction"
run_with_timeout "$ENGINE_BUILD/test_mtp_completion"
run_with_timeout "$ENGINE_BUILD/test_gemma4_mtp_fast"
