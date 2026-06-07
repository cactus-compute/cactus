# Android Build

Builds `libcactus_engine` for Android (arm64-v8a).

## Usage

```bash
cactus build --android
```

Or directly:

```bash
bash android/build.sh
```

## Output

- `libcactus_engine.so` — Shared library (JNI, for Android apps)
- `libcactus_engine.a` — Static library (for native test binaries)

## Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ANDROID_NDK_HOME` | Auto-detected | Android NDK path |
| `ANDROID_PLATFORM` | `android-21` | Minimum API level |
| `CMAKE_BUILD_TYPE` | `Release` | CMake build type |
| `CACTUS_CURL_ROOT` | `cactus-engine/libs/curl` | Vendored libcurl path |

## Requirements

- Android NDK (install via Android Studio > SDK Tools > NDK)
- CMake 3.10+
