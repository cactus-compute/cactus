# Vendored libcurl

Prebuilt static `libcurl.a` archives live here for Cactus builds.

## Apple builds

Rebuild the Apple slices with:

```bash
bash cactus-engine/libs/curl/build-apple.sh
```

Outputs:

- `ios/device/libcurl.a`
- `ios/simulator/libcurl.a`
- `tvos/device/libcurl.a`
- `tvos/simulator/libcurl.a`
- `watchos/device/libcurl.a`
- `watchos/simulator/libcurl.a`
- `visionos/device/libcurl.a`
- `visionos/simulator/libcurl.a`
- `macos/libcurl.a`

Defaults:

- curl `8.18.0`
- `arm64` for iOS, tvOS, visionOS, and macOS
- `arm64_32` and `arm64` for watchOS devices
- `arm64` for all simulators
