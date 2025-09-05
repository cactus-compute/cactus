#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APPLE_DIR="$PROJECT_ROOT/apple"

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
BUILD_DIR="$APPLE_DIR/build"

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found, please install it"
    exit 1
fi

if ! xcode-select -p &> /dev/null; then
    echo "Error: Xcode command line tools not found"
    echo "Install with: xcode-select --install"
    exit 1
fi

n_cpu=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

echo "Building Cactus for Apple platforms (ARM64 only)..."
echo "Build type: $CMAKE_BUILD_TYPE"
echo "Using $n_cpu CPU cores"
echo "Apple CMakeLists.txt: $APPLE_DIR/CMakeLists.txt"


IOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path)
if [ -z "$IOS_SDK_PATH" ] || [ ! -d "$IOS_SDK_PATH" ]; then
    echo "Error: iOS SDK not found. Make sure Xcode is installed."
    exit 1
fi

echo "Using iOS SDK: $IOS_SDK_PATH"

cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
      -DCMAKE_OSX_SYSROOT="$IOS_SDK_PATH" \
      -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
      -S "$APPLE_DIR" \
      -B "$BUILD_DIR"

cmake --build "$BUILD_DIR" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

mkdir -p "$APPLE_DIR"
cp "$BUILD_DIR/libcactus.a" "$APPLE_DIR/" || \
   { echo "Error: Could not find libcactus.a"; exit 1; }

echo "Build complete!"
echo "Library location: $APPLE_DIR/libcactus.a"
echo "Target: APPLE_DIR ARM64 devices only"