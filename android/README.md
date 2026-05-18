# Android Build Step

Builds the native Android artifacts used by language bindings.

## Outputs

- `android/libcactus.so` — shared library for `arm64-v8a`
- `android/libcactus.a` — merged static archive

## Usage

```bash
cactus build --android
```

Or directly:

```bash
bash android/build.sh
```

## Notes

- Requires Android NDK, targets `arm64-v8a` only
- Kotlin/JNI bindings live in [`bindings/kotlin/`](/bindings/kotlin/)
- Vendored mbedTLS under `android/mbedtls/` and libcurl under `cactus-engine/libs/curl/android/`
