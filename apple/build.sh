#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APPLE_DIR="$ROOT_DIR/apple"

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
BUILD_STATIC=${BUILD_STATIC:-true}
BUILD_XCFRAMEWORK=${BUILD_XCFRAMEWORK:-true}

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

echo "Building Cactus for Apple platforms..."
echo "Build type: $CMAKE_BUILD_TYPE"
echo "Using $n_cpu CPU cores"
echo "Static library: $BUILD_STATIC"
echo "XCFramework: $BUILD_XCFRAMEWORK"

function cp_headers() {
    mkdir -p "$ROOT_DIR/apple/cactus.xcframework/$1/cactus.framework/Headers"
    cp "$ROOT_DIR/cactus/ffi/"*.h "$ROOT_DIR/apple/cactus.xcframework/$1/cactus.framework/Headers/" 2>/dev/null || true
    cp "$ROOT_DIR/cactus/engine/"*.h "$ROOT_DIR/apple/cactus.xcframework/$1/cactus.framework/Headers/" 2>/dev/null || true
    cp "$ROOT_DIR/cactus/graph/"*.h "$ROOT_DIR/apple/cactus.xcframework/$1/cactus.framework/Headers/" 2>/dev/null || true
    cp "$ROOT_DIR/cactus/kernel/"*.h "$ROOT_DIR/apple/cactus.xcframework/$1/cactus.framework/Headers/" 2>/dev/null || true
    cp "$ROOT_DIR/cactus/"*.h "$ROOT_DIR/apple/cactus.xcframework/$1/cactus.framework/Headers/" 2>/dev/null || true
}

function create_xcframework_info_plist() {
    cat > "$ROOT_DIR/apple/cactus.xcframework/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>AvailableLibraries</key>
	<array>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64</string>
			<key>LibraryPath</key>
			<string>cactus.framework</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
		</dict>
		<dict>
			<key>LibraryIdentifier</key>
			<string>ios-arm64-simulator</string>
			<key>LibraryPath</key>
			<string>cactus.framework</string>
			<key>SupportedArchitectures</key>
			<array>
				<string>arm64</string>
			</array>
			<key>SupportedPlatform</key>
			<string>ios</string>
			<key>SupportedPlatformVariant</key>
			<string>simulator</string>
		</dict>
	</array>
	<key>CFBundlePackageType</key>
	<string>XFWK</string>
	<key>XCFrameworkFormatVersion</key>
	<string>1.0</string>
</dict>
</plist>
EOF
}

function build_static_library() {
    echo "Building static library for iOS device..."
    BUILD_DIR="$APPLE_DIR/build-static-device"
    
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
          -DBUILD_SHARED_LIBS=OFF \
          -S "$APPLE_DIR" \
          -B "$BUILD_DIR"

    cmake --build "$BUILD_DIR" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

    mkdir -p "$APPLE_DIR"
    cp "$BUILD_DIR/libcactus.a" "$APPLE_DIR/libcactus-device.a" || \
       { echo "Error: Could not find device libcactus.a"; exit 1; }

    echo "Device static library built: $APPLE_DIR/libcactus-device.a"
    
    echo "Building static library for iOS simulator..."
    BUILD_DIR_SIM="$APPLE_DIR/build-static-simulator"
    
    IOS_SIM_SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path)
    if [ -z "$IOS_SIM_SDK_PATH" ] || [ ! -d "$IOS_SIM_SDK_PATH" ]; then
        echo "Error: iOS Simulator SDK not found. Make sure Xcode is installed."
        exit 1
    fi

    echo "Using iOS Simulator SDK: $IOS_SIM_SDK_PATH"

    cmake -DCMAKE_SYSTEM_NAME=iOS \
          -DCMAKE_OSX_ARCHITECTURES=arm64 \
          -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
          -DCMAKE_OSX_SYSROOT="$IOS_SIM_SDK_PATH" \
          -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
          -DBUILD_SHARED_LIBS=OFF \
          -S "$APPLE_DIR" \
          -B "$BUILD_DIR_SIM"

    cmake --build "$BUILD_DIR_SIM" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

    cp "$BUILD_DIR_SIM/libcactus.a" "$APPLE_DIR/libcactus-simulator.a" || \
       { echo "Error: Could not find simulator libcactus.a"; exit 1; }

    echo "Simulator static library built: $APPLE_DIR/libcactus-simulator.a"
}

function build_framework() {
    echo "Building framework for $4..."
    cd "$5"

    cmake -S "$ROOT_DIR/apple" \
        -B . \
        -GXcode \
        -DCMAKE_SYSTEM_NAME=$1 \
        -DCMAKE_OSX_ARCHITECTURES="$2" \
        -DCMAKE_OSX_SYSROOT=$3 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
        -DCMAKE_IOS_INSTALL_COMBINED=YES

    cmake --build . --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

    DEST_DIR="$ROOT_DIR/apple/cactus.xcframework/$4"
    FRAMEWORK_SRC="$CMAKE_BUILD_TYPE-$3/cactus.framework"
    FRAMEWORK_DEST="$DEST_DIR/cactus.framework"

    rm -rf "$DEST_DIR"
    mkdir -p "$DEST_DIR"

    if [ -d "$FRAMEWORK_SRC" ]; then
        cp -R "$FRAMEWORK_SRC" "$FRAMEWORK_DEST"
        echo "Framework copied from $FRAMEWORK_SRC to $FRAMEWORK_DEST"
    else
        echo "Error: Expected framework not found at $FRAMEWORK_SRC"
        echo "Available files:"
        find . -name "*.framework" -o -name "libcactus*" 2>/dev/null || true
        exit 1
    fi

    cp_headers $4

    rm -rf ./*
    cd "$ROOT_DIR"
}

function build_xcframework() {
    echo "Building XCFramework..."
    
    rm -rf "$ROOT_DIR/apple/cactus.xcframework"
    rm -rf "$ROOT_DIR/apple/build-ios" "$ROOT_DIR/apple/build-ios-simulator"
    mkdir -p "$ROOT_DIR/apple/build-ios" "$ROOT_DIR/apple/build-ios-simulator"

    build_framework "iOS" "arm64" "iphoneos" "ios-arm64" "$ROOT_DIR/apple/build-ios"
    
    build_framework "iOS" "arm64" "iphonesimulator" "ios-arm64-simulator" "$ROOT_DIR/apple/build-ios-simulator"

    create_xcframework_info_plist

    rm -rf "$ROOT_DIR/apple/build-ios" "$ROOT_DIR/apple/build-ios-simulator"
    
    echo "XCFramework built: $ROOT_DIR/apple/cactus.xcframework"
}

t0=$(date +%s)

if [ "$BUILD_STATIC" = "true" ]; then
    build_static_library
fi

if [ "$BUILD_XCFRAMEWORK" = "true" ]; then
    build_xcframework
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
    echo "XCFramework: $APPLE_DIR/cactus.xcframework"
fi