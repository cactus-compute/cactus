#!/bin/bash
# =============================================================================
# Build script for Cactus Swift integration test
# =============================================================================
# Prerequisites:
#   - cactus.framework built at apple/build-macos/Release/
#   - Xcode command line tools installed
#
# Usage:
#   ./build.sh            # build
#   ./swift_test [path]   # run (optional model path)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRAMEWORK_DIR="$REPO_ROOT/apple/build-macos/Release"
FRAMEWORK="$FRAMEWORK_DIR/cactus.framework"
HEADER="$REPO_ROOT/cactus/ffi/cactus_ffi.h"
MODULEMAP="$REPO_ROOT/apple/module.modulemap"
OUTPUT="$SCRIPT_DIR/swift_test"
MODULE_DIR="$SCRIPT_DIR/.build/CactusModule"

# ── Validate prerequisites ──────────────────────────────────────────────────

if [ ! -d "$FRAMEWORK" ]; then
    echo "Error: cactus.framework not found at $FRAMEWORK"
    echo ""
    echo "Build it first:"
    echo "  cactus build --apple"
    echo "  # or: cd apple/build-macos && cmake --build . --config Release"
    exit 1
fi

if [ ! -f "$HEADER" ]; then
    echo "Error: cactus_ffi.h not found at $HEADER"
    exit 1
fi

# ── Set up C module for Swift import ────────────────────────────────────────
# The framework binary doesn't include Headers/ or Modules/, so we create
# a standalone C module that Swift can import as 'cactus'.

rm -rf "$MODULE_DIR"
mkdir -p "$MODULE_DIR"

cat > "$MODULE_DIR/module.modulemap" << 'MODULEMAP'
module cactus [system] {
    header "cactus_ffi.h"
    export *
}
MODULEMAP

cp "$HEADER" "$MODULE_DIR/cactus_ffi.h"

# ── Compile ─────────────────────────────────────────────────────────────────

echo "Building swift_test..."
swiftc \
    "$SCRIPT_DIR/main.swift" \
    "$REPO_ROOT/apple/Cactus.swift" \
    -I "$MODULE_DIR" \
    -F "$FRAMEWORK_DIR" \
    -framework cactus \
    -Xlinker -rpath -Xlinker "$FRAMEWORK_DIR" \
    -o "$OUTPUT" \
    -O \
    -swift-version 5 \
    2>&1

echo ""
echo "Build successful: $OUTPUT"
echo "Run:  $OUTPUT [model_path]"
echo "      Default model: ~/cactus/weights/lfm2-350m"
