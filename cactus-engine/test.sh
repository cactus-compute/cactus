#!/bin/bash
set -e
cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)/.."
ASSETS_DIR="$(pwd)/tests/assets"

IOS_MODE=false
ANDROID_MODE=false
SUITE=""
CACTUS_TEST_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ios)     IOS_MODE=true; shift ;;
        --android) ANDROID_MODE=true; shift ;;
        --suite)   SUITE="${2:?--suite needs an argument}"; shift 2 ;;
        --model)   CACTUS_TEST_MODEL="${2:?--model needs an argument}"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$CACTUS_TEST_MODEL" ]; then
    echo "Error: --model is required. Run engine tests via the CLI (cactus test ...) which sets it." >&2
    exit 2
fi
export CACTUS_TEST_MODEL

if [[ "$CACTUS_TEST_MODEL" == /* || "$CACTUS_TEST_MODEL" == ./* ]]; then
    BUNDLE_DIR="$CACTUS_TEST_MODEL"
else
    BUNDLE_DIR="$PROJECT_ROOT/weights/$(basename "$CACTUS_TEST_MODEL" | tr '[:upper:]' '[:lower:]')"
fi

if [ ! -f "$BUNDLE_DIR/components/manifest.json" ]; then
    echo "Bundle missing at $BUNDLE_DIR. Run: cactus transpile $CACTUS_TEST_MODEL" >&2
    exit 1
fi

if [ "$IOS_MODE" = true ]; then
    export CACTUS_TEST_MODEL="$BUNDLE_DIR" CACTUS_TEST_SUITE="$SUITE"
    exec "$(pwd)/tests/ios/run.sh" "$BUNDLE_DIR"
fi
if [ "$ANDROID_MODE" = true ]; then
    export CACTUS_TEST_MODEL="$BUNDLE_DIR" CACTUS_TEST_SUITE="$SUITE"
    exec "$(pwd)/tests/android/run.sh" "$BUNDLE_DIR"
fi

echo "Model:  $CACTUS_TEST_MODEL"
echo "Bundle: $BUNDLE_DIR"

cd "$PROJECT_ROOT/cactus-engine/tests"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

export CACTUS_TEST_MODEL="$BUNDLE_DIR"
export CACTUS_TEST_ASSETS="$ASSETS_DIR"
export CACTUS_INDEX_PATH="$ASSETS_DIR"

FAILED=0
if [ -n "$SUITE" ]; then
    target="./test_$SUITE"
    if [ -x "$target" ]; then
        "$target" || FAILED=1
    else
        echo "Test not found: $target" >&2
        FAILED=1
    fi
else
    for t in ./test_*; do
        [ -x "$t" ] || continue
        "$t" || FAILED=1
    done
fi
exit $FAILED
