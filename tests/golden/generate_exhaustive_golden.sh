#!/bin/bash
# for running the golden tests locally
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
GOLDEN_DIR="$SCRIPT_DIR"
TEST_BUILD_DIR="$PROJECT_ROOT/tests/build"

if [ ! -f "$TEST_BUILD_DIR/test_exhaustive" ]; then
    echo "Error: test_exhaustive not found at $TEST_BUILD_DIR/test_exhaustive"
    echo "Build tests first: cd tests && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

MODELS=(
    "qwen:Qwen/Qwen3-0.6B:INT8:llm:CACTUS_TEST_MODEL"
    "nomic:nomic-ai/nomic-embed-text-v2-moe:INT8:embedding:CACTUS_TEST_MODEL"
    "lfm2:LiquidAI/LFM2-350M:INT8:llm:CACTUS_TEST_MODEL"
    "lfm2moe:LiquidAI/LFM2-8B-A1B:INT8:llm:CACTUS_TEST_MODEL"
    "gemma:google/gemma-3-270m-it:INT8:llm:CACTUS_TEST_MODEL"
    "whisper:openai/whisper-small:INT8:stt:CACTUS_TEST_TRANSCRIBE_MODEL"
    "moonshine:UsefulSensors/moonshine-base:INT8:stt:CACTUS_TEST_TRANSCRIBE_MODEL"
    "parakeet:nvidia/parakeet-ctc-0.6b:INT8:stt:CACTUS_TEST_TRANSCRIBE_MODEL"
    "lfm2vl:LiquidAI/LFM2-VL-450M:INT8:vlm:CACTUS_TEST_MODEL"
    "lfm2vl:LiquidAI/LFM2-VL-450M:INT4:vlm:CACTUS_TEST_MODEL"
    "lfm2vl:LiquidAI/LFM2-VL-450M:FP16:vlm:CACTUS_TEST_MODEL"
)

echo "=== Downloading VAD model ==="
cactus download "snakers4/silero-vad" --precision INT8

PASSED=0
FAILED=0

for entry in "${MODELS[@]}"; do
    IFS=: read -r family model precision test_type env_var <<< "$entry"

    echo ""
    echo "=== Generating golden output: $family / $precision ==="
    echo "    Model: $model"

    if ! cactus download "$model" --precision "$precision"; then
        echo "    FAILED to download $model"
        FAILED=$((FAILED + 1))
        continue
    fi

    MODEL_DIR=$(echo "$model" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    MODEL_PATH="$PROJECT_ROOT/weights/$MODEL_DIR"

    export CACTUS_NO_CLOUD_TELE=1
    export CACTUS_GOLDEN_GENERATE=1
    export CACTUS_TEST_GOLDEN_FILE="$GOLDEN_DIR/golden.json"
    export CACTUS_TEST_GOLDEN_FAMILY="$family"
    export CACTUS_TEST_GOLDEN_PRECISION="$precision"
    export CACTUS_TEST_ASSETS="$PROJECT_ROOT/tests/assets"
    export "$env_var"="$MODEL_PATH"

    if [ "$test_type" = "stt" ]; then
        export CACTUS_TEST_VAD_MODEL="$PROJECT_ROOT/weights/silero-vad"
    fi

    if "$TEST_BUILD_DIR/test_exhaustive"; then
        echo "    OK"
        PASSED=$((PASSED + 1))
    else
        echo "    FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Golden generation complete ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""
echo "Golden file: $GOLDEN_DIR/golden.json"
echo "After verifying outputs, update golden.json with the actual expected values,"
echo "then commit it to the repository."

exit $FAILED
