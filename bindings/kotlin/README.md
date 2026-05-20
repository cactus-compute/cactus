# Kotlin Bindings

JNI bridge to `cactus_engine.h` for Android, with KMP support for iOS.

## Android Integration

```bash
cactus build --android
```

1. Copy `android/libcactus.so` to `app/src/main/jniLibs/arm64-v8a/`
2. Copy `Cactus.kt` to your Kotlin source tree

```kotlin
val handle = CactusJNI.nativeInit("/path/to/model", null, false)
val buf = ByteArray(65536)
CactusJNI.nativeComplete(handle, messagesJson, buf, null, null, null, null)
val response = String(buf, 0, buf.indexOf(0))
CactusJNI.nativeDestroy(handle)
```

## Kotlin Multiplatform

```
commonMain/  Cactus.common.kt     expect declarations
androidMain/ Cactus.android.kt    actual via JNI
iosMain/     Cactus.ios.kt        actual via cinterop
```

### build.gradle.kts

```kotlin
kotlin {
    androidTarget()

    listOf(iosArm64(), iosSimulatorArm64()).forEach {
        it.compilations.getByName("main") {
            cinterops {
                val cactus by creating {
                    defFile("src/nativeInterop/cinterop/cactus.def")
                    includeDirs("/path/to/cactus/cactus-engine")
                }
            }
        }
        it.binaries.framework {
            linkerOpts("-L/path/to/cactus/apple", "-lcactus-device")
        }
    }
}
```

### Usage (shared code)

```kotlin
val model = cactusInit("/path/to/model", null, false)
val result = cactusComplete(model, messagesJson, null, null, null)
cactusDestroy(model)
```
