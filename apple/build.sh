#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APPLE_DIR="$ROOT_DIR/apple"

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
BUILD_STATIC=${BUILD_STATIC:-true}
BUILD_XCFRAMEWORK=${BUILD_XCFRAMEWORK:-true}
CACTUS_CURL_ROOT=${CACTUS_CURL_ROOT:-"$ROOT_DIR/cactus-engine/libs/curl"}

IOS_DEPLOYMENT_TARGET=${IOS_DEPLOYMENT_TARGET:-13.0}
TVOS_DEPLOYMENT_TARGET=${TVOS_DEPLOYMENT_TARGET:-13.0}
WATCHOS_DEPLOYMENT_TARGET=${WATCHOS_DEPLOYMENT_TARGET:-9.0}
VISIONOS_DEPLOYMENT_TARGET=${VISIONOS_DEPLOYMENT_TARGET:-1.0}
MACOS_DEPLOYMENT_TARGET=${MACOS_DEPLOYMENT_TARGET:-13.0}

XCFRAMEWORK_PATH="$APPLE_DIR/CXXCactusDarwin.xcframework"
XCFRAMEWORK_ZIP_PATH="$APPLE_DIR/CXXCactusDarwin.xcframework.zip"
LAST_FRAMEWORK_PATH=""

if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake not found, please install it"
    exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
    echo "Error: Xcode command line tools not found"
    echo "Install with: xcode-select --install"
    exit 1
fi

n_cpu=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

echo "Building Cactus for Apple platforms..."
echo "Build type: $CMAKE_BUILD_TYPE"
echo "Using $n_cpu CPU cores"
echo "Static library: $BUILD_STATIC"
echo "XCFramework: $BUILD_XCFRAMEWORK"
echo "Vendored libcurl root: $CACTUS_CURL_ROOT"

build_static_library() {
    echo "Building static library for iOS device..."
    local build_dir="$APPLE_DIR/build-static-device"
    local ios_sdk_path

    ios_sdk_path=$(xcrun --sdk iphoneos --show-sdk-path)
    if [ -z "$ios_sdk_path" ] || [ ! -d "$ios_sdk_path" ]; then
        echo "Error: iOS SDK not found. Make sure Xcode is installed."
        exit 1
    fi

    cmake -DCMAKE_SYSTEM_NAME=iOS \
          -DCMAKE_OSX_ARCHITECTURES=arm64 \
          -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
          -DCMAKE_OSX_SYSROOT="$ios_sdk_path" \
          -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
          -DBUILD_SHARED_LIBS=OFF \
          -DCACTUS_CURL_ROOT="$CACTUS_CURL_ROOT" \
          -S "$APPLE_DIR" \
          -B "$build_dir" >/dev/null

    cmake --build "$build_dir" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu" >/dev/null

    cp "$build_dir/libcactus.a" "$APPLE_DIR/libcactus-device.a"
    echo "Device static library built: $APPLE_DIR/libcactus-device.a"

    echo "Building static library for iOS simulator..."
    local build_dir_sim="$APPLE_DIR/build-static-simulator"
    local ios_sim_sdk_path

    ios_sim_sdk_path=$(xcrun --sdk iphonesimulator --show-sdk-path)
    if [ -z "$ios_sim_sdk_path" ] || [ ! -d "$ios_sim_sdk_path" ]; then
        echo "Error: iOS Simulator SDK not found. Make sure Xcode is installed."
        exit 1
    fi

    cmake -DCMAKE_SYSTEM_NAME=iOS \
          -DCMAKE_OSX_ARCHITECTURES=arm64 \
          -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
          -DCMAKE_OSX_SYSROOT="$ios_sim_sdk_path" \
          -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
          -DBUILD_SHARED_LIBS=OFF \
          -DCACTUS_CURL_ROOT="$CACTUS_CURL_ROOT" \
          -S "$APPLE_DIR" \
          -B "$build_dir_sim" >/dev/null

    cmake --build "$build_dir_sim" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu" >/dev/null

    cp "$build_dir_sim/libcactus.a" "$APPLE_DIR/libcactus-simulator.a"
    echo "Simulator static library built: $APPLE_DIR/libcactus-simulator.a"
}

build_framework_slice() {
    local platform_name="$1"
    local system_name="$2"
    local sdk_name="$3"
    local arch="$4"
    local deployment_target="$5"
    local build_dir="$6"
    local sdk_path
    local framework_path=""

    sdk_path=$(xcrun --sdk "$sdk_name" --show-sdk-path)
    if [ -z "$sdk_path" ] || [ ! -d "$sdk_path" ]; then
        echo "Error: SDK $sdk_name not found. Make sure Xcode is installed."
        exit 1
    fi

    echo "Building $platform_name ($system_name, $sdk_name, $arch)..."

    rm -rf "$build_dir"

    cmake -S "$ROOT_DIR/apple" \
        -B "$build_dir" \
        -GXcode \
        -DCMAKE_SYSTEM_NAME="$system_name" \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_SYSROOT="$sdk_path" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DBUILD_SHARED_LIBS=ON \
        -DCACTUS_CURL_ROOT="$CACTUS_CURL_ROOT" \
        -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
        -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
        -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO \
        -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY="" >/dev/null

    cmake --build "$build_dir" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu" >/dev/null

    if [ -d "$build_dir/lib/$CMAKE_BUILD_TYPE/CXXCactusDarwin.framework" ]; then
        framework_path="$build_dir/lib/$CMAKE_BUILD_TYPE/CXXCactusDarwin.framework"
    elif [ -d "$build_dir/$CMAKE_BUILD_TYPE-$sdk_name/CXXCactusDarwin.framework" ]; then
        framework_path="$build_dir/$CMAKE_BUILD_TYPE-$sdk_name/CXXCactusDarwin.framework"
    elif [ -d "$build_dir/$CMAKE_BUILD_TYPE/CXXCactusDarwin.framework" ]; then
        framework_path="$build_dir/$CMAKE_BUILD_TYPE/CXXCactusDarwin.framework"
    else
        framework_path=$(find "$build_dir" -path "*CXXCactusDarwin.framework" -not -path "*EagerLinkingTBDs*" | head -n 1)
    fi

    if [ -z "$framework_path" ] || [ ! -d "$framework_path" ]; then
        echo "Error: CXXCactusDarwin.framework not found for $platform_name"
        exit 1
    fi

    LAST_FRAMEWORK_PATH="$framework_path"
}

framework_binary_path() {
    local framework_path="$1"

    if [ -f "$framework_path/CXXCactusDarwin" ]; then
        printf '%s\n' "$framework_path/CXXCactusDarwin"
    elif [ -f "$framework_path/Versions/A/CXXCactusDarwin" ]; then
        printf '%s\n' "$framework_path/Versions/A/CXXCactusDarwin"
    else
        echo "Error: Framework binary not found in $framework_path" >&2
        exit 1
    fi
}

create_universal_framework() {
    local source_framework_a="$1"
    local source_framework_b="$2"
    local destination_framework="$3"
    local binary_a
    local binary_b
    local destination_binary

    mkdir -p "$(dirname "$destination_framework")"
    rm -rf "$destination_framework"
    cp -R "$source_framework_a" "$destination_framework"

    binary_a=$(framework_binary_path "$source_framework_a")
    binary_b=$(framework_binary_path "$source_framework_b")
    destination_binary=$(framework_binary_path "$destination_framework")

    lipo -create "$binary_a" "$binary_b" -output "$destination_binary"
}

build_combined_xcframework() {
    echo "Building combined Apple XCFramework..."

    rm -rf "$XCFRAMEWORK_PATH" "$XCFRAMEWORK_ZIP_PATH"

    local build_root="$APPLE_DIR/build-xcframework"
    local ios_framework
    local ios_sim_framework
    local macos_framework
    local tvos_framework
    local tvos_sim_framework
    local watchos_arm64_framework
    local watchos_arm64_32_framework
    local watchos_device_framework
    local watchos_sim_framework
    local visionos_framework
    local visionos_sim_framework

    build_framework_slice "iOS" "iOS" "iphoneos" "arm64" "$IOS_DEPLOYMENT_TARGET" "$build_root/ios"
    ios_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "iOS Simulator" "iOS" "iphonesimulator" "arm64" "$IOS_DEPLOYMENT_TARGET" "$build_root/ios-sim"
    ios_sim_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "macOS" "Darwin" "macosx" "arm64" "$MACOS_DEPLOYMENT_TARGET" "$build_root/macos"
    macos_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "tvOS" "tvOS" "appletvos" "arm64" "$TVOS_DEPLOYMENT_TARGET" "$build_root/tvos"
    tvos_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "tvOS Simulator" "tvOS" "appletvsimulator" "arm64" "$TVOS_DEPLOYMENT_TARGET" "$build_root/tvos-sim"
    tvos_sim_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "watchOS arm64" "watchOS" "watchos" "arm64" "$WATCHOS_DEPLOYMENT_TARGET" "$build_root/watchos-arm64"
    watchos_arm64_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "watchOS arm64_32" "watchOS" "watchos" "arm64_32" "$WATCHOS_DEPLOYMENT_TARGET" "$build_root/watchos-arm64_32"
    watchos_arm64_32_framework="$LAST_FRAMEWORK_PATH"

    watchos_device_framework="$build_root/watchos-device/CXXCactusDarwin.framework"
    create_universal_framework "$watchos_arm64_framework" "$watchos_arm64_32_framework" "$watchos_device_framework"

    build_framework_slice "watchOS Simulator" "watchOS" "watchsimulator" "arm64" "$WATCHOS_DEPLOYMENT_TARGET" "$build_root/watchos-sim"
    watchos_sim_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "visionOS" "visionOS" "xros" "arm64" "$VISIONOS_DEPLOYMENT_TARGET" "$build_root/visionos"
    visionos_framework="$LAST_FRAMEWORK_PATH"

    build_framework_slice "visionOS Simulator" "visionOS" "xrsimulator" "arm64" "$VISIONOS_DEPLOYMENT_TARGET" "$build_root/visionos-sim"
    visionos_sim_framework="$LAST_FRAMEWORK_PATH"

    xcodebuild -create-xcframework \
        -framework "$ios_framework" \
        -framework "$ios_sim_framework" \
        -framework "$macos_framework" \
        -framework "$tvos_framework" \
        -framework "$tvos_sim_framework" \
        -framework "$watchos_device_framework" \
        -framework "$watchos_sim_framework" \
        -framework "$visionos_framework" \
        -framework "$visionos_sim_framework" \
        -output "$XCFRAMEWORK_PATH" >/dev/null

    local macos_framework_dir="$XCFRAMEWORK_PATH/macos-arm64/CXXCactusDarwin.framework"
    if [ -d "$macos_framework_dir/Versions/A" ]; then
        rm -rf "$macos_framework_dir/Headers" "$macos_framework_dir/Modules"
        ln -s Versions/A/Headers "$macos_framework_dir/Headers"
        ln -s Versions/A/Modules "$macos_framework_dir/Modules"
    fi

    rm -rf "$XCFRAMEWORK_ZIP_PATH"
    ditto -c -k --norsrc --keepParent "$XCFRAMEWORK_PATH" "$XCFRAMEWORK_ZIP_PATH"
    echo "Combined XCFramework built: $XCFRAMEWORK_PATH"
    echo "Combined XCFramework zip: $XCFRAMEWORK_ZIP_PATH"
}

t0=$(date +%s)

if [ "$BUILD_STATIC" = "true" ]; then
    build_static_library
fi

if [ "$BUILD_XCFRAMEWORK" = "true" ]; then
    build_combined_xcframework
fi

t1=$(date +%s)
echo ""
echo "Build complete!"
echo "Total time: $((t1 - t0)) seconds"

if [ "$BUILD_STATIC" = "true" ]; then
    rm -rf "$APPLE_DIR/build-static-device" "$APPLE_DIR/build-static-simulator"
    echo "Static libraries:"
    echo "  Device: $APPLE_DIR/libcactus-device.a"
    echo "  Simulator: $APPLE_DIR/libcactus-simulator.a"
fi

if [ "$BUILD_XCFRAMEWORK" = "true" ]; then
    rm -rf "$APPLE_DIR/build-xcframework"
    echo "XCFramework:"
    echo "  Darwin: $XCFRAMEWORK_PATH"
    echo "  Darwin zip: $XCFRAMEWORK_ZIP_PATH"
fi
