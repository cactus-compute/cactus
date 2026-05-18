# Flutter Bindings

Dart FFI bindings to `cactus_engine.h`.

## Integration

```bash
cactus build --apple
cactus build --android
```

1. Copy `libcactus.so` to `jniLibs/arm64-v8a/` (Android)
2. Add `cactus-ios.xcframework` to your Xcode project (iOS)
3. Copy `cactus.dart` into your Dart source tree
4. Add `ffi` to `pubspec.yaml`

## Usage

```dart
import 'cactus.dart';
import 'package:ffi/ffi.dart';

final modelPath = '/path/to/model'.toNativeUtf8();
final model = cactusInit(modelPath, nullptr, false);
calloc.free(modelPath);

final msgs = messagesJson.toNativeUtf8();
final buf = calloc<Int8>(65536);
cactusComplete(model, msgs, buf.cast(), 65536, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
final response = buf.cast<Utf8>().toDartString();
calloc.free(msgs);
calloc.free(buf);
cactusDestroy(model);
```
