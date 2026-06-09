#!/bin/bash
# Build the cactus-only e2e decode-tps benchmark for Android (arm64-v8a).
# Links the prebuilt cactus-v2 Android archives (see CMakeLists.txt) — the
# only engine that reads the CQ4 transpiled bundles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

ANDROID_PLATFORM=${ANDROID_PLATFORM:-android-24}

# Locate NDK (env var, Android SDK, or homebrew cask).
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    for cand in \
        "$ANDROID_HOME/ndk/"* \
        "$HOME/Library/Android/sdk/ndk/"* \
        /opt/homebrew/share/android-ndk \
        /opt/homebrew/Caskroom/android-ndk/*/AndroidNDK*.app/Contents/NDK; do
        [ -f "$cand/build/cmake/android.toolchain.cmake" ] && ANDROID_NDK_HOME="$cand" && break
    done
fi
if [ -z "${ANDROID_NDK_HOME:-}" ] || [ ! -f "$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" ]; then
    echo "Error: Android NDK not found. Set ANDROID_NDK_HOME." >&2
    exit 1
fi
echo "Using NDK: $ANDROID_NDK_HOME"

echo "==> Configuring + building e2e_bench"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo "==> Built: $BUILD_DIR/e2e_bench"
