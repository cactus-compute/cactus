# Swift Bindings

C module map import of `cactus_engine.h`. Works on iOS and macOS.

## Integration

<!-- --8<-- [start:install] -->
```bash
cactus build --apple
```
<!-- --8<-- [end:install] -->

<!-- --8<-- [start:integration] -->
**XCFramework**: drag `apple/cactus-ios.xcframework` (or `apple/cactus-macos.xcframework`) into Xcode (Embed & Sign). The framework bundles `cactus_engine.h`, so `import cactus` works directly.

**Static library**: link `apple/libcactus_engine-device.a` (or `apple/libcactus_engine-simulator.a`), copy `bindings/swift/module.modulemap` into your project, and add `cactus-engine/` to Header Search Paths so the module map can find `cactus_engine.h`.
<!-- --8<-- [end:integration] -->

## Usage

<!-- --8<-- [start:example] -->
```swift
import cactus

let model = cactus_init("/path/to/model", nil, false)
var buf = [CChar](repeating: 0, count: 65536)
cactus_complete(model, messagesJson, &buf, buf.count, nil, nil, nil, nil, nil, 0)
let response = String(cString: buf)
cactus_destroy(model)
```
<!-- --8<-- [end:example] -->
