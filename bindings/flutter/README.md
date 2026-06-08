# Flutter Bindings

Dart FFI bindings to `cactus_engine.h`.

## Integration

<!-- --8<-- [start:install] -->
```bash
cactus build --apple
cactus build --android
```
<!-- --8<-- [end:install] -->

<!-- --8<-- [start:integration] -->
1. Copy `android/libcactus_engine.so` to your app's `jniLibs/arm64-v8a/`
2. Add `apple/cactus-ios.xcframework` to your Xcode project (iOS)
3. Add `apple/cactus-macos.xcframework` to your Xcode project (macOS)
4. Copy `cactus.dart` into your Dart source tree
5. Add `ffi` to `pubspec.yaml`
<!-- --8<-- [end:integration] -->

## Usage

<!-- --8<-- [start:example] -->
```dart
import 'cactus.dart';
import 'package:ffi/ffi.dart';

final modelPath = '/path/to/model'.toNativeUtf8();
final model = cactusInit(modelPath, nullptr, false);
calloc.free(modelPath);

const messagesJson = '[{"role":"user","content":"Hello"}]';
final msgs = messagesJson.toNativeUtf8();
final buf = calloc<Int8>(65536);
cactusComplete(model, msgs, buf.cast(), 65536, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
final response = buf.cast<Utf8>().toDartString();
calloc.free(msgs);
calloc.free(buf);
cactusDestroy(model);
```
<!-- --8<-- [end:example] -->
