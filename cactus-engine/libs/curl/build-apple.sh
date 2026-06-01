#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURL_VERSION="${CURL_VERSION:-8.18.0}"
CURL_URL="${CURL_URL:-https://curl.se/download/curl-${CURL_VERSION}.tar.gz}"
BUILD_ROOT="${BUILD_ROOT:-}"

if [ -z "$BUILD_ROOT" ]; then
    BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cactus-curl-apple.XXXXXX")"
fi

IOS_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET:-13.0}"
TVOS_DEPLOYMENT_TARGET="${TVOS_DEPLOYMENT_TARGET:-13.0}"
WATCHOS_DEPLOYMENT_TARGET="${WATCHOS_DEPLOYMENT_TARGET:-9.0}"
VISIONOS_DEPLOYMENT_TARGET="${VISIONOS_DEPLOYMENT_TARGET:-1.0}"
MACOS_DEPLOYMENT_TARGET="${MACOS_DEPLOYMENT_TARGET:-13.0}"

SOURCE_TARBALL="$BUILD_ROOT/curl-${CURL_VERSION}.tar.gz"
SOURCE_DIR="$BUILD_ROOT/curl-${CURL_VERSION}"

cleanup() {
    rm -rf "$BUILD_ROOT"
}
trap cleanup EXIT


if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake not found"
    exit 1
fi

if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "Error: Xcode command line tools not found"
    exit 1
fi

n_cpu=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

download_source() {
    echo "Downloading curl ${CURL_VERSION}..."
    curl -L "$CURL_URL" -o "$SOURCE_TARBALL"
    tar -xzf "$SOURCE_TARBALL" -C "$BUILD_ROOT"
}

copy_headers() {
    rm -rf "$SCRIPT_DIR/include/curl"
    mkdir -p "$SCRIPT_DIR/include"
    cp -R "$SOURCE_DIR/include/curl" "$SCRIPT_DIR/include/"
}

build_slice() {
    local platform_dir="$1"
    local variant_dir="$2"
    local system_name="$3"
    local sysroot="$4"
    local arch="$5"
    local deployment_target="$6"

    local build_dir="$BUILD_ROOT/build-${platform_dir}-${variant_dir}"
    local output_dir="$SCRIPT_DIR/$platform_dir/$variant_dir"
    local output_lib="$output_dir/libcurl.a"
    local built_lib=""

    echo "Building ${platform_dir}/${variant_dir} (${arch}, ${sysroot})..."

    rm -rf "$build_dir"

    cmake -S "$SOURCE_DIR" \
        -B "$build_dir" \
        -G Xcode \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CURL_EXE=OFF \
        -DBUILD_LIBCURL_DOCS=OFF \
        -DBUILD_MISC_DOCS=OFF \
        -DENABLE_ARES=OFF \
        -DCURL_USE_LIBPSL=OFF \
        -DCURL_ZLIB=OFF \
        -DCURL_BROTLI=OFF \
        -DCURL_ZSTD=OFF \
        -DUSE_NGHTTP2=OFF \
        -DUSE_LIBIDN2=OFF \
        -DUSE_APPLE_IDN=OFF \
        -DCURL_DISABLE_LDAP=ON \
        -DCURL_DISABLE_LDAPS=ON \
        -DCURL_USE_LIBSSH2=OFF \
        -DCURL_USE_OPENSSL=OFF \
        -DHAVE_PIPE2=OFF \
        -DCMAKE_SYSTEM_NAME="$system_name" \
        -DCMAKE_OSX_SYSROOT="$sysroot" \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" >/dev/null

    cmake --build "$build_dir" --config Release -j "$n_cpu" >/dev/null

    if [ -f "$build_dir/lib/Release-$sysroot/libcurl.a" ]; then
        built_lib="$build_dir/lib/Release-$sysroot/libcurl.a"
    elif [ -f "$build_dir/lib/Release/libcurl.a" ]; then
        built_lib="$build_dir/lib/Release/libcurl.a"
    else
        echo "Error: built libcurl.a not found for ${platform_dir}/${variant_dir}"
        exit 1
    fi

    mkdir -p "$output_dir"
    cp "$built_lib" "$output_lib"
}

build_macos() {
    local build_dir="$BUILD_ROOT/build-macos"
    local output_dir="$SCRIPT_DIR/macos"
    local output_lib="$output_dir/libcurl.a"

    echo "Building macos (arm64, macosx)..."

    rm -rf "$build_dir"

    cmake -S "$SOURCE_DIR" \
        -B "$build_dir" \
        -G Xcode \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CURL_EXE=OFF \
        -DBUILD_LIBCURL_DOCS=OFF \
        -DBUILD_MISC_DOCS=OFF \
        -DENABLE_ARES=OFF \
        -DCURL_USE_LIBPSL=OFF \
        -DCURL_ZLIB=OFF \
        -DCURL_BROTLI=OFF \
        -DCURL_ZSTD=OFF \
        -DUSE_NGHTTP2=OFF \
        -DUSE_LIBIDN2=OFF \
        -DUSE_APPLE_IDN=OFF \
        -DCURL_DISABLE_LDAP=ON \
        -DCURL_DISABLE_LDAPS=ON \
        -DCURL_USE_LIBSSH2=OFF \
        -DCURL_USE_OPENSSL=OFF \
        -DHAVE_PIPE2=OFF \
        -DCMAKE_SYSTEM_NAME=Darwin \
        -DCMAKE_OSX_SYSROOT=macosx \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET" >/dev/null

    cmake --build "$build_dir" --config Release -j "$n_cpu" >/dev/null

    mkdir -p "$output_dir"
    cp "$build_dir/lib/Release/libcurl.a" "$output_lib"
}

build_watchos_device() {
    local build_dir_arm64_32="$BUILD_ROOT/build-watchos-device-arm64_32"
    local build_dir_arm64="$BUILD_ROOT/build-watchos-device-arm64"
    local legacy_output_dir="$SCRIPT_DIR/watchos/device"
    local output_dir_arm64_32="$SCRIPT_DIR/watchos/device-arm64_32"
    local output_dir_arm64="$SCRIPT_DIR/watchos/device-arm64"
    local output_lib_arm64_32="$output_dir_arm64_32/libcurl.a"
    local output_lib_arm64="$output_dir_arm64/libcurl.a"
    local lib_arm64_32="$build_dir_arm64_32/lib/Release-watchos/libcurl.a"
    local lib_arm64="$build_dir_arm64/lib/Release-watchos/libcurl.a"

    echo "Building watchos/device (arm64_32 + arm64, watchos)..."

    rm -rf "$build_dir_arm64_32" "$build_dir_arm64"
    rm -rf "$legacy_output_dir"

    cmake -S "$SOURCE_DIR" \
        -B "$build_dir_arm64_32" \
        -G Xcode \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CURL_EXE=OFF \
        -DBUILD_LIBCURL_DOCS=OFF \
        -DBUILD_MISC_DOCS=OFF \
        -DENABLE_ARES=OFF \
        -DCURL_USE_LIBPSL=OFF \
        -DCURL_ZLIB=OFF \
        -DCURL_BROTLI=OFF \
        -DCURL_ZSTD=OFF \
        -DUSE_NGHTTP2=OFF \
        -DUSE_LIBIDN2=OFF \
        -DUSE_APPLE_IDN=OFF \
        -DCURL_USE_OPENSSL=OFF \
        -DHAVE_PIPE2=OFF \
        -DCMAKE_SYSTEM_NAME=watchOS \
        -DCMAKE_OSX_SYSROOT=watchos \
        -DCMAKE_OSX_ARCHITECTURES=arm64_32 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$WATCHOS_DEPLOYMENT_TARGET" >/dev/null

    cmake -S "$SOURCE_DIR" \
        -B "$build_dir_arm64" \
        -G Xcode \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CURL_EXE=OFF \
        -DBUILD_LIBCURL_DOCS=OFF \
        -DBUILD_MISC_DOCS=OFF \
        -DENABLE_ARES=OFF \
        -DCURL_USE_LIBPSL=OFF \
        -DCURL_ZLIB=OFF \
        -DCURL_BROTLI=OFF \
        -DCURL_ZSTD=OFF \
        -DUSE_NGHTTP2=OFF \
        -DUSE_LIBIDN2=OFF \
        -DUSE_APPLE_IDN=OFF \
        -DCURL_USE_OPENSSL=OFF \
        -DHAVE_PIPE2=OFF \
        -DCMAKE_SYSTEM_NAME=watchOS \
        -DCMAKE_OSX_SYSROOT=watchos \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$WATCHOS_DEPLOYMENT_TARGET" >/dev/null

    cmake --build "$build_dir_arm64_32" --config Release -j "$n_cpu" >/dev/null
    cmake --build "$build_dir_arm64" --config Release -j "$n_cpu" >/dev/null

    mkdir -p "$output_dir_arm64_32" "$output_dir_arm64"
    cp "$lib_arm64_32" "$output_lib_arm64_32"
    cp "$lib_arm64" "$output_lib_arm64"
}

download_source
copy_headers

build_slice "ios" "device" "iOS" "iphoneos" "arm64" "$IOS_DEPLOYMENT_TARGET"
build_slice "ios" "simulator" "iOS" "iphonesimulator" "arm64" "$IOS_DEPLOYMENT_TARGET"
build_slice "tvos" "device" "tvOS" "appletvos" "arm64" "$TVOS_DEPLOYMENT_TARGET"
build_slice "tvos" "simulator" "tvOS" "appletvsimulator" "arm64" "$TVOS_DEPLOYMENT_TARGET"
build_watchos_device
build_slice "watchos" "simulator" "watchOS" "watchsimulator" "arm64" "$WATCHOS_DEPLOYMENT_TARGET"
build_slice "visionos" "device" "visionOS" "xros" "arm64" "$VISIONOS_DEPLOYMENT_TARGET"
build_slice "visionos" "simulator" "visionOS" "xrsimulator" "arm64" "$VISIONOS_DEPLOYMENT_TARGET"
build_macos

echo "Vendored Apple libcurl builds updated under $SCRIPT_DIR"
