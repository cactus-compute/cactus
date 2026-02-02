#!/bin/bash

set -e

missing=()
if ! command -v cmake &> /dev/null; then
    missing+=("cmake")
fi

if ! command -v make &> /dev/null; then
    missing+=("make")
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    missing+=("g++")
fi

if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: Missing required build tools: ${missing[*]}"
    echo ""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Install with: sudo apt-get install cmake build-essential"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Install with: xcode-select --install && brew install cmake"
    else
        echo "Please install cmake and a C++ compiler for your platform."
    fi
    exit 1
fi

echo "Building Cactus library..."

cd "$(dirname "$0")/../cactus"

rm -rf build

mkdir -p build
cd build

cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Cactus library built successfully!"
echo "Library location: $(pwd)/libcactus.a"
