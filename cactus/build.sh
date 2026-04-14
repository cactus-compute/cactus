#!/bin/bash

set -e

# Allow overriding build tools via environment
CMAKE="${CMAKE:-cmake}"
MAKE="${MAKE:-make}"
CXX="${CXX:-}"

missing=()
if ! command -v "$CMAKE" &> /dev/null; then
    missing+=("cmake")
fi

if ! command -v "$MAKE" &> /dev/null; then
    missing+=("make")
fi

if [ -z "$CXX" ]; then
    if command -v g++ &> /dev/null; then
        CXX="g++"
    elif command -v clang++ &> /dev/null; then
        CXX="clang++"
    else
        missing+=("g++")
    fi
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

rm -rf build 2>/dev/null || true

mkdir -p build
cd build

CMAKE_EXTRA_ARGS=""
if [ -n "$CXX" ]; then
    CMAKE_EXTRA_ARGS="-DCMAKE_CXX_COMPILER=$CXX"
fi

$CMAKE .. $CMAKE_EXTRA_ARGS -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1
$MAKE -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Cactus library built successfully!"
echo "Library location: $(pwd)/libcactus.a"
