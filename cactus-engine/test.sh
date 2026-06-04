#!/bin/bash
set -e

cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)/.."
ASSETS_DIR="$(pwd)/tests/assets"

IOS_MODE=false
ANDROID_MODE=false
ONLY_EXEC=""
MODEL_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ios)
            IOS_MODE=true
            shift
            ;;
        --android)
            ANDROID_MODE=true
            shift
            ;;
        --only)
            [ -z "${2:-}" ] && echo "Error: --only requires an argument" && exit 1
            ONLY_EXEC="$2"
            shift 2
            ;;
        --model)
            [ -z "${2:-}" ] && echo "Error: --model requires an argument" && exit 1
            MODEL_ARG="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

MODEL_DIR="${CACTUS_TEST_MODEL:-${MODEL_ARG:-$PROJECT_ROOT/weights/LFM2-VL-450M}}"

if [ "$IOS_MODE" = true ]; then
    export CACTUS_TEST_ONLY="$ONLY_EXEC"
    exec "$(pwd)/tests/ios/run.sh" "$MODEL_DIR"
fi

if [ "$ANDROID_MODE" = true ]; then
    export CACTUS_TEST_ONLY="$ONLY_EXEC"
    exec "$(pwd)/tests/android/run.sh" "$MODEL_DIR"
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model weights not found at $MODEL_DIR"
    echo "Set CACTUS_TEST_MODEL or download weights first."
    exit 1
fi

echo "Building and testing cactus-engine..."
echo "Model: $MODEL_DIR"

cd "$PROJECT_ROOT/cactus"
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

cd "$PROJECT_ROOT/cactus-engine/tests"
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
export CACTUS_TEST_MODEL="$MODEL_DIR"
export CACTUS_TEST_ASSETS="$ASSETS_DIR"
export CACTUS_INDEX_PATH="$ASSETS_DIR"

FAILED=0

if [ -n "$ONLY_EXEC" ]; then
    target="./test_$ONLY_EXEC"
    if [ -x "$target" ]; then
        ./"$target" || FAILED=1
    else
        echo "Test not found: $target"
        FAILED=1
    fi
else
    for test_bin in test_*; do
        [ -x "$test_bin" ] && ./"$test_bin" || FAILED=1
    done
fi

exit $FAILED
