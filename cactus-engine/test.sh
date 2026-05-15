#!/bin/bash
set -e

cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)/.."
MODEL_DIR="${CACTUS_TEST_MODEL:-$PROJECT_ROOT/weights/gemma-4-e2b-it}"
ASSETS_DIR="$(pwd)/tests/assets"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model weights not found at $MODEL_DIR"
    echo "Set CACTUS_TEST_MODEL or download weights first."
    exit 1
fi

echo "Building and testing cactus-engine..."
echo "Model: $MODEL_DIR"

# Build the library
cd "$PROJECT_ROOT/cactus"
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Build tests
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
NATIVE_MODELS_AVAILABLE=1
if grep -q "Native model subclasses are not present in this build" "$PROJECT_ROOT/cactus-engine/src/model.cpp"; then
    NATIVE_MODELS_AVAILABLE=0
fi

for test_bin in test_*; do
    if [ ! -x "$test_bin" ]; then
        continue
    fi
    if [ "$NATIVE_MODELS_AVAILABLE" -eq 0 ]; then
        case "$test_bin" in
            test_embed|test_llm|test_rag|test_stt|test_vlm)
                echo "SKIP: $test_bin - native model subclasses are not present in this build"
                continue
                ;;
        esac
    fi
    ./"$test_bin" || FAILED=1
done

exit $FAILED
