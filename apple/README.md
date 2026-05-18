# Apple Build Step

Builds the native Apple artifacts used by language bindings.

## Outputs

- `apple/libcactus-device.a` — iOS device static library
- `apple/libcactus-simulator.a` — iOS simulator static library
- `apple/cactus-ios.xcframework/` — iOS xcframework
- `apple/cactus-macos.xcframework/` — macOS xcframework

## Usage

```bash
cactus build --apple
```

Or directly:

```bash
bash apple/build.sh
```

## Notes

- Requires macOS with Xcode, deployment target 13.0
- Swift bindings live in [`bindings/swift/`](/bindings/swift/)
- Vendored libcurl under `cactus-engine/libs/curl/ios/` and `cactus-engine/libs/curl/macos/`
