#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Cleaning all build artifacts from Cactus project..."
echo "Project root: $PROJECT_ROOT"
echo ""

remove_if_exists() {
    if [ -d "$1" ]; then
        echo "Removing: $1"
        rm -rf "$1"
    else
        echo "Not found: $1"
    fi
}

remove_if_exists "$PROJECT_ROOT/cactus/build"

remove_if_exists "$PROJECT_ROOT/android/build"
remove_if_exists "$PROJECT_ROOT/android/libs"
remove_if_exists "$PROJECT_ROOT/android/arm64-v8a"

remove_if_exists "$PROJECT_ROOT/apple/build"
remove_if_exists "$PROJECT_ROOT/apple/libcactus.a"

remove_if_exists "$PROJECT_ROOT/tests/build"

echo ""
echo "Removing compiled libraries and frameworks..."
find "$PROJECT_ROOT" -name "*.so" -type f -delete 2>/dev/null && echo "Removed .so files" || echo "No .so files found"
find "$PROJECT_ROOT" -name "*.a" -type f -delete 2>/dev/null && echo "Removed .a files" || echo "No .a files found"
find "$PROJECT_ROOT" -name "*.xcframework" -type d -exec rm -rf {} + 2>/dev/null && echo "Removed .xcframework directories" || echo "No .xcframework directories found"

echo ""
echo "Clean complete!"
echo "All build artifacts have been removed."