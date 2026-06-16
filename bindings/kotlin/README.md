# Kotlin Bindings

JNI bridge to `cactus_engine.h` for Android, with KMP support for iOS.

The JNI bridge itself (`android/cactus_jni.cpp`, compiled into `libcactus_engine.so`) is JVM-language-agnostic. Java consumers can use it by writing an equivalent `CactusJNI.java` with `native` declarations matching the signatures in `Cactus.kt` — same library, same `Java_com_cactus_CactusJNI_*` symbols.

## Android Integration

<!-- --8<-- [start:install] -->
```bash
cactus build --android
```
<!-- --8<-- [end:install] -->

<!-- --8<-- [start:integration] -->
1. Copy `android/libcactus_engine.so` to `app/src/main/jniLibs/arm64-v8a/`
2. Copy `Cactus.kt` and `CactusCallbacks.kt` to your Kotlin source tree
<!-- --8<-- [end:integration] -->

<!-- --8<-- [start:example] -->
```kotlin
val handle = CactusJNI.nativeInit("/path/to/model", null, false)
val buf = ByteArray(65536)
CactusJNI.nativeComplete(handle, messagesJson, buf, null, null, null, null)
val response = String(buf, 0, buf.indexOf(0))
CactusJNI.nativeDestroy(handle)
```
<!-- --8<-- [end:example] -->

Streaming transcription — push 16 kHz mono PCM16, read `{"success":true,"confirmed":...,"pending":...}` back each call:

```kotlin
val stream = CactusJNI.nativeStreamTranscribeStart(handle, "{\"language\":\"en\"}")
val out = ByteArray(65536)
for (chunk in pcmChunks) { // each: 16 kHz mono PCM16 bytes
    CactusJNI.nativeStreamTranscribeProcess(stream, chunk, out)
    // parse confirmed/pending from String(out, 0, out.indexOf(0))
}
CactusJNI.nativeStreamTranscribeStop(stream, out)
```

## Kotlin Multiplatform

```
commonMain/  Cactus.common.kt     expect declarations
commonMain/  CactusCallbacks.kt   shared callback interfaces
androidMain/ Cactus.android.kt    actual via JNI
iosMain/     Cactus.ios.kt        actual via cinterop
```

### build.gradle.kts

```kotlin
kotlin {
    androidTarget()

    listOf(iosArm64(), iosSimulatorArm64()).forEach { target ->
        target.compilations.getByName("main") {
            cinterops {
                val cactus by creating {
                    defFile("src/nativeInterop/cinterop/cactus.def")
                    includeDirs("/path/to/cactus/cactus-engine")
                }
            }
        }
        val libSuffix = if (target.name == "iosSimulatorArm64") "simulator" else "device"
        target.binaries.framework {
            linkerOpts("-L/path/to/cactus/apple", "-lcactus_engine-$libSuffix")
        }
    }
}
```

### Usage (shared code)

```kotlin
val model = cactusInit("/path/to/model", null, false)
val result = cactusComplete(model, messagesJson, null, null, null, null)

// streaming transcription: start, feed PCM16 chunks, stop
val stream = cactusStreamTranscribeStart(model, "{\"language\":\"en\"}")
val out = cactusStreamTranscribeProcess(stream, pcmChunk)   // {"success":true,"confirmed":...,"pending":...}
val tail = cactusStreamTranscribeStop(stream)

cactusDestroy(model)
```
