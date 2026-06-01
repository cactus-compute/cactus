#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/package"
BUILD_DIR="$SCRIPT_DIR/build"
PACKAGE_BIN_DIR="$PACKAGE_DIR/bin"
ARTIFACT_BUNDLE_PATH="$PACKAGE_BIN_DIR/cactus_artifact.artifactbundle"
ARTIFACT_BUNDLE_ZIP_PATH="$PACKAGE_BIN_DIR/cactus_artifact.artifactbundle.zip"
DARWIN_XCFRAMEWORK_ZIP_SRC="$ROOT_DIR/apple/cactus.xcframework.zip"
DARWIN_XCFRAMEWORK_ZIP_DST="$PACKAGE_BIN_DIR/cactus.xcframework.zip"

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake not found, please install it"
    exit 1
fi

n_cpu=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

VARIANT_PATHS=()
VARIANT_TRIPLES=()

artifactbundle_init() {
    rm -rf "$ARTIFACT_BUNDLE_PATH" "$ARTIFACT_BUNDLE_ZIP_PATH"
    mkdir -p "$ARTIFACT_BUNDLE_PATH/dist" "$ARTIFACT_BUNDLE_PATH/include"
    cp "$ROOT_DIR/cactus-engine/cactus_engine.h" "$ARTIFACT_BUNDLE_PATH/include/cactus.h"

    cat > "$ARTIFACT_BUNDLE_PATH/include/module.modulemap" << 'EOF'
module cactus_artifact {
    header "cactus.h"
    export *
}
EOF
}

artifactbundle_add_variant() {
    local relative_path="$1"
    local supported_triple="$2"
    VARIANT_PATHS+=("$relative_path")
    VARIANT_TRIPLES+=("$supported_triple")
}

artifactbundle_write_info_json() {
    local info_file="$ARTIFACT_BUNDLE_PATH/info.json"
    {
        echo "{"
        echo "  \"schemaVersion\": \"1.0\","
        echo "  \"artifacts\": {"
        echo "    \"cactus_artifact\": {"
        echo "      \"type\": \"staticLibrary\","
        echo "      \"version\": \"1.0.0\","
        echo "      \"variants\": ["
        local i
        for i in "${!VARIANT_PATHS[@]}"; do
            local comma=","
            if [ "$i" -eq $((${#VARIANT_PATHS[@]} - 1)) ]; then
                comma=""
            fi
            echo "        {"
            echo "          \"path\": \"${VARIANT_PATHS[$i]}\","
            echo "          \"supportedTriples\": [\"${VARIANT_TRIPLES[$i]}\"],"
            echo "          \"staticLibraryMetadata\": {"
            echo "            \"headerPaths\": [\"include\"],"
            echo "            \"moduleMapPath\": \"include/module.modulemap\""
            echo "          }"
            echo "        }$comma"
        done
        echo "      ]"
        echo "    }"
        echo "  }"
        echo "}"
    } > "$info_file"
}

artifactbundle_finalize() {
    artifactbundle_write_info_json
    ditto -c -k --norsrc --keepParent "$ARTIFACT_BUNDLE_PATH" "$ARTIFACT_BUNDLE_ZIP_PATH"
    rm -rf "$ARTIFACT_BUNDLE_PATH"
    echo "Artifact bundle zip: $ARTIFACT_BUNDLE_ZIP_PATH"
}

build_android_variant() {
    echo "Building Android artifact bundle variant..."
    "$ROOT_DIR/android/build.sh"

    mkdir -p "$ARTIFACT_BUNDLE_PATH/dist/android"
    cp "$ROOT_DIR/android/libcactus.a" "$ARTIFACT_BUNDLE_PATH/dist/android/libcactus.a"
    artifactbundle_add_variant "dist/android/libcactus.a" "aarch64-unknown-linux-android"
}

build_linux_arm_variant() {
    echo "Building Linux ARM artifact bundle variant..."

    local out="$BUILD_DIR/linux-arm"
    local host_os host_arch toolchain_file
    host_os="$(uname -s)"
    host_arch="$(uname -m)"
    toolchain_file=""

    if [ "$host_os" = "Linux" ] && { [ "$host_arch" = "aarch64" ] || [ "$host_arch" = "arm64" ]; }; then
        echo "Using native Linux ARM toolchain ($host_arch)."
    elif [ "$host_os" = "Darwin" ] && [ "$host_arch" = "arm64" ]; then
        toolchain_file="$SCRIPT_DIR/linux-arm/aarch64-macos-cross.toolchain.cmake"
        if [ ! -f "$toolchain_file" ]; then
            echo "Error: Missing Linux ARM toolchain file at $toolchain_file"
            exit 1
        fi
        echo "Using macOS cross toolchain file: $toolchain_file"
    else
        echo "Error: Unsupported host for Linux ARM build: $host_os/$host_arch"
        echo "Linux ARM build is supported on native Linux ARM or Apple Silicon macOS."
        exit 1
    fi

    local cmake_args=(
        -S "$SCRIPT_DIR/linux-arm"
        -B "$out"
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
        -DCMAKE_SYSTEM_PROCESSOR=aarch64
    )

    if [ -n "$toolchain_file" ]; then
        cmake_args+=( -DCMAKE_TOOLCHAIN_FILE="$toolchain_file" )
    fi

    cmake "${cmake_args[@]}" >/dev/null
    cmake --build "$out" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu" >/dev/null

    local linux_lib_path="$out/lib/libcactus.a"
    if [ ! -f "$linux_lib_path" ]; then
        linux_lib_path="$out/libcactus.a"
    fi

    if [ ! -f "$linux_lib_path" ]; then
        echo "Error: Could not find Linux ARM static library output."
        exit 1
    fi

    mkdir -p "$ARTIFACT_BUNDLE_PATH/dist/linux-arm64"
    cp "$linux_lib_path" "$ARTIFACT_BUNDLE_PATH/dist/linux-arm64/libcactus.a"
    artifactbundle_add_variant "dist/linux-arm64/libcactus.a" "aarch64-unknown-linux-gnu"
}

write_package_manifest() {
    cat > "$PACKAGE_DIR/Package.swift" << 'EOF'
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "CactusShims",
  products: [
    .library(name: "CactusShims", targets: ["CactusShims"])
  ],
  targets: [
    .target(
      name: "CactusShims",
      dependencies: [
        .target(name: "cactus_artifact", condition: .when(platforms: [.android, .linux])),
        .target(
          name: "cactus",
          condition: .when(platforms: [.iOS, .macOS, .visionOS, .tvOS, .watchOS, .macCatalyst])
        )
      ],
      linkerSettings: [
        .linkedLibrary("c++_shared", .when(platforms: [.android]))
      ]
    ),
    .binaryTarget(name: "cactus", path: "bin/cactus.xcframework.zip"),
    .binaryTarget(name: "cactus_artifact", path: "bin/cactus_artifact.artifactbundle.zip")
  ]
)
EOF
}

write_package_sources() {
    mkdir -p "$PACKAGE_DIR/Sources/CactusShims"
    cat > "$PACKAGE_DIR/Sources/CactusShims/CactusShims.swift" << 'EOF'
#if canImport(Darwin)
  @_exported import cactus
#else
  @_exported import cactus_artifact
#endif
EOF
}

prepare_package_dir() {
    rm -rf "$PACKAGE_DIR" "$BUILD_DIR"
    mkdir -p "$PACKAGE_BIN_DIR" "$BUILD_DIR"
}

copy_darwin_xcframework() {
    echo "Building Darwin XCFramework..."
    BUILD_STATIC=false BUILD_XCFRAMEWORK=true "$ROOT_DIR/apple/build.sh"

    if [ ! -f "$DARWIN_XCFRAMEWORK_ZIP_SRC" ]; then
        echo "Error: Darwin XCFramework zip not found at $DARWIN_XCFRAMEWORK_ZIP_SRC"
        exit 1
    fi

    cp "$DARWIN_XCFRAMEWORK_ZIP_SRC" "$DARWIN_XCFRAMEWORK_ZIP_DST"
}

prepare_package_dir
copy_darwin_xcframework
artifactbundle_init
build_android_variant
build_linux_arm_variant
artifactbundle_finalize
write_package_manifest
write_package_sources

echo "Swift package generated at $PACKAGE_DIR"
