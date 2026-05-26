#!/bin/bash
set -e

missing=()
command -v cmake &> /dev/null || missing+=("cmake")
command -v make &> /dev/null || missing+=("make")
command -v g++ &> /dev/null || command -v clang++ &> /dev/null || missing+=("g++")

if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: Missing required build tools: ${missing[*]}"
    echo ""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Install with: sudo apt-get install cmake build-essential"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Install with: xcode-select --install && brew install cmake"
    fi
    exit 1
fi

cd "$(dirname "$0")"

echo "Building Cactus library..."

LOCK_DIR="${TMPDIR:-/tmp}/cactus-build.lock"
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    if [ -f "$LOCK_DIR/pid" ] && ! kill -0 "$(cat "$LOCK_DIR/pid" 2>/dev/null)" 2>/dev/null; then
        rm -rf "$LOCK_DIR"
        continue
    fi
    sleep 2
done
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

mkdir -p build
cd build

cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Cactus library built successfully!"
echo "  Static:  $(pwd)/libcactus.a"
echo "  Shared:  $(pwd)/libcactus.$([ "$(uname)" = Darwin ] && echo dylib || echo so)"
