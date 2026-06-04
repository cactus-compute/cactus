#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FLUTTER_DIR="$SCRIPT_DIR"

echo "Building Cactus for Flutter..."
echo ""

# Build Android
echo "=== Building Android ==="
if [ -f "$PROJECT_ROOT/android/build.sh" ]; then
    bash "$PROJECT_ROOT/android/build.sh"
    cp "$PROJECT_ROOT/android/libcactus.so" "$FLUTTER_DIR/"
    echo "Copied libcactus.so"
else
    echo "Warning: android/build.sh not found, skipping Android build"
fi

echo ""

# Build Apple (iOS/macOS)
echo "=== Building Apple (iOS/macOS) ==="
if [ -f "$PROJECT_ROOT/apple/build.sh" ]; then
    bash "$PROJECT_ROOT/apple/build.sh"

    rm -rf "$FLUTTER_DIR/cactus-ios.xcframework"
    cp -R "$PROJECT_ROOT/apple/cactus-ios.xcframework" "$FLUTTER_DIR/"
    echo "Copied cactus-ios.xcframework"

    rm -rf "$FLUTTER_DIR/cactus-macos.xcframework"
    cp -R "$PROJECT_ROOT/apple/cactus-macos.xcframework" "$FLUTTER_DIR/"
    echo "Copied cactus-macos.xcframework"
else
    echo "Warning: apple/build.sh not found, skipping Apple build"
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Output:"
echo "  flutter/libcactus.so"
echo "  flutter/cactus-ios.xcframework"
echo "  flutter/cactus-macos.xcframework"
