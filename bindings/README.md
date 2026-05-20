# Bindings

Language-facing interop layers live here.

These are bindings, not SDKs:

- `python/` uses `ctypes`
- `swift/` uses a C module map
- `kotlin/` uses JNI
- `flutter/` uses Dart FFI
- `rust/` uses raw `extern "C"` declarations
- `react-native/` contains a thin React Native bridge backed by the raw Kotlin and Swift bindings

Platform-native build and packaging stay outside this folder:

- [`android/`](/android/) builds Android native artifacts
- [`apple/`](/apple/) builds Apple native artifacts

Use `sdk` terminology only for packaged, consumer-ready layers with a stable public API.
