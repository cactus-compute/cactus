#!/bin/bash

set -e

echo "Building Cactus library..."

cd "$(dirname "$0")/../cactus"

rm -rf build

mkdir -p build
cd build

cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1
make -j$(nproc 2>/dev/null || echo 4)

echo "Cactus library built successfully!"
echo "Library location: $(pwd)/lib/libcactus.a"
