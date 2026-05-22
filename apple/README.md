# Apple Build

Builds `libcactus` for iOS and macOS.

## Usage

```bash
cactus build --apple
```

Or directly:

```bash
bash apple/build.sh
```

## Output

- `libcactus-device.a` — iOS device (arm64)
- `libcactus-simulator.a` — iOS simulator (arm64)
- `cactus-ios.xcframework/` — iOS XCFramework (device + simulator)
- `cactus-macos.xcframework/` — macOS XCFramework (arm64)

## Options

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILD_STATIC` | `true` | Build static libraries |
| `BUILD_XCFRAMEWORK` | `true` | Build XCFrameworks |
| `CMAKE_BUILD_TYPE` | `Release` | CMake build type |
| `CACTUS_CURL_ROOT` | `cactus-engine/libs/curl` | Vendored libcurl path |

## Requirements

- Xcode with iOS SDK
- CMake 3.10+
