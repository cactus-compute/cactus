#!/bin/bash

set -e


echo "Building Cactus chat..."
echo "======================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WEIGHTS_DIR="$PROJECT_ROOT/weights/lfm2-1.2B"
if [ ! -d "$WEIGHTS_DIR" ] || [ ! -f "$WEIGHTS_DIR/config.txt" ]; then
    echo ""
    echo "LFM2 weights not found. Generating weights..."
    echo "============================================="
    cd "$PROJECT_ROOT"
    if command -v python3 &> /dev/null; then
        echo "Running: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
        if ! python3 -c "import numpy, torch, transformers" 2>/dev/null; then
            echo "Warning: Required Python packages not found. Make sure to set up your env in accordance with the README."
            exit 1
        fi
        if python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8; then
            echo "Successfully generated Weights"
        else
            echo "Warning: Failed to generate Weights. Tests may fail."
            echo "Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
        fi

    else
        echo "Warning: Python3 not found. Cannot generate weights automatically."
        echo "Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
    fi
else
    echo ""
    echo "LFM2 weights found at $WEIGHTS_DIR"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$SCRIPT_DIR/.."
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"

cd "$ROOT_DIR/cactus"
if [ ! -f "build/libcactus.a" ]; then
    echo "Cactus library not found. Building..."
    ./build.sh
fi

cd "$BUILD_DIR"

echo "Compiling chat.cpp..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    clang++ -std=c++17 -O3 \
        -I"$ROOT_DIR" \
        "$SCRIPT_DIR/chat.cpp" \
        "$ROOT_DIR/cactus/build/libcactus.a" \
        -o chat \
        -framework Accelerate
else
    g++ -std=c++17 -O3 \
        -I"$ROOT_DIR" \
        "$SCRIPT_DIR/chat.cpp" \
        "$ROOT_DIR/cactus/build/libcactus.a" \
        -o chat \
        -pthread
fi

echo "Build complete: $BUILD_DIR/chat"
echo ""

clear 
echo "Usage: $BUILD_DIR/chat <model_path>"
echo ""

$BUILD_DIR/chat $PROJECT_ROOT/weights/lfm2-1.2B
