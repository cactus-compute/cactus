# Swift Bindings

C module map import of `cactus_engine.h`. Works on iOS and macOS.

## Integration

```bash
cactus build --apple
```

**XCFramework**: drag `apple/cactus-ios.xcframework` into Xcode (Embed & Sign).

**Static library**: add `apple/libcactus-device.a` to Link Binary With Libraries, copy `module.modulemap` and `cactus_engine.h` into Header Search Paths.

## Usage

```swift
import cactus

let model = cactus_init("/path/to/model", nil, false)
var buf = [CChar](repeating: 0, count: 65536)
cactus_complete(model, messagesJson, &buf, buf.count, nil, nil, nil, nil, nil, 0)
let response = String(cString: buf)
cactus_destroy(model)
```
