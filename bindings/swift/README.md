# Swift Bindings

Generated Swift package wrapping `cactus_engine.h`. Works on all Apple platforms, Android, and Linux.

## Integration

Run this command to generate the local Swift package in this directory.
```bash
cactus build --swift
```

**Xcode Project:** Add the generated package to you project from the menu `File > Add Package Dependencies > Add Local...` Then, select the generated package in the menu.

**Swift Package:** Add the generated package as a local SPM dependency in your `Package.swift`.
```swift
dependencies: [
  .package(path: "<path to generated package>"),
  // ...
],
targets: [
  .target(
    name: "YourTarget",
    dependencies: [
      .product(name: "CactusShims", package: "CactusShims"),
      // ...
    ]
  )
]
```

## Usage

```swift
import CactusShims

let model = cactus_init("/path/to/model", nil, false)
var buf = [CChar](repeating: 0, count: 65536)
cactus_complete(model, messagesJson, &buf, buf.count, nil, nil, nil, nil, nil, 0)
let response = String(cString: buf)
cactus_destroy(model)
```
