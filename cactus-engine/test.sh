#!/bin/bash
set -e
cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)/.."
ASSETS_DIR="$(pwd)/tests/assets"

IOS_MODE=false
ANDROID_MODE=false
SUITE=""
BACKEND="auto"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ios)     IOS_MODE=true; shift ;;
        --android) ANDROID_MODE=true; shift ;;
        --suite)   SUITE="${2:?--suite needs an argument}"; shift 2 ;;
        --model)   CACTUS_TEST_MODEL="${2:?--model needs an argument}"; shift 2 ;;
        --transcription-model) CACTUS_TEST_TRANSCRIPTION_MODEL="${2:?--transcription-model needs an argument}"; shift 2 ;;
        --backend) BACKEND="${2:?--backend needs an argument}"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

require_bundle() {
    local dir="$1" label="$2"
    if [ -z "$dir" ] || [ ! -f "$dir/components/manifest.json" ]; then
        echo "Error: --$label must point to a prepared bundle. Run engine tests via 'cactus test', which prepares them." >&2
        exit 2
    fi
    echo "$dir"
}

BUNDLE_DIR="$(require_bundle "$CACTUS_TEST_MODEL" "model")"
TRANSCRIPTION_BUNDLE_DIR="$(require_bundle "$CACTUS_TEST_TRANSCRIPTION_MODEL" "transcription-model")"

if [ "$IOS_MODE" = true ]; then
    export CACTUS_TEST_MODEL="$BUNDLE_DIR" CACTUS_TEST_TRANSCRIPTION_MODEL="$TRANSCRIPTION_BUNDLE_DIR" CACTUS_TEST_SUITE="$SUITE" CACTUS_TEST_BACKEND="$BACKEND"
    exec "$(pwd)/tests/ios/run.sh"
fi
if [ "$ANDROID_MODE" = true ]; then
    export CACTUS_TEST_MODEL="$BUNDLE_DIR" CACTUS_TEST_TRANSCRIPTION_MODEL="$TRANSCRIPTION_BUNDLE_DIR" CACTUS_TEST_SUITE="$SUITE" CACTUS_TEST_BACKEND="$BACKEND"
    exec "$(pwd)/tests/android/run.sh"
fi

echo "Model:                $CACTUS_TEST_MODEL"
echo "Bundle:               $BUNDLE_DIR"
echo "Transcription model:  $CACTUS_TEST_TRANSCRIPTION_MODEL"
echo "Transcription bundle: $TRANSCRIPTION_BUNDLE_DIR"
echo "Backend:              $BACKEND"

cd "$PROJECT_ROOT/cactus-engine/tests"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

export CACTUS_TEST_MODEL="$BUNDLE_DIR"
export CACTUS_TEST_TRANSCRIPTION_MODEL="$TRANSCRIPTION_BUNDLE_DIR"
export CACTUS_TEST_ASSETS="$ASSETS_DIR"
export CACTUS_INDEX_PATH="$ASSETS_DIR"
export CACTUS_TEST_BACKEND="$BACKEND"

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
