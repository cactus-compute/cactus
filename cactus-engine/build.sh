#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Building cactus-engine..."

mkdir -p build
cd build

cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "cactus-engine built successfully!"
echo "  Static: $(pwd)/libcactus_engine.a"
echo "  Shared: $(pwd)/libcactus_engine.$([ "$(uname)" = Darwin ] && echo dylib || echo so)"
