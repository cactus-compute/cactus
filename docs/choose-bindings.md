# Choose Your Binding

Not sure which binding to use? 

Pick the right one for your platform and use case:

|  | React Native | Flutter | Kotlin | Swift | Python | Rust | CLI | C++ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Platforms** | iOS, Android | iOS, Android, macOS | Android, iOS (KMP) | iOS, macOS | Arm Linux, macOS | Arm Linux, macOS | macOS, Arm Linux | Arm Linux, macOS, iOS, Android |
| **Install** | build / source | build | build | build | pip | build / source | brew / source | header |
| **LLM** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Streaming** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Vision** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Audio** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Transcription** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Function Calling** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **RAG / Embeddings** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Cloud Fallback** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Quick Recommendations

!!! tip "Building a mobile app?"
    - **React Native** -- already have a React Native project? Just `npm install` and go
    - **Flutter** -- cross-platform mobile + Mac with full native bindings
    - **Kotlin** -- native Android apps or Kotlin Multiplatform
    - **Swift** -- native iOS/macOS apps with Metal acceleration

!!! tip "Server-side or scripting?"
    - **Python** -- server-side inference, batch processing, or rapid prototyping
    - **CLI** -- quick model testing and interactive sessions without writing code

!!! tip "Embedding in a native app?"
    - **C++** -- game engines, native desktop apps, or any C/C++ project
    - **Rust** -- systems-level integration with safe FFI bindings

## Binding Documentation

- **[React Native](/bindings/react-native/)** -- Native bridge modules over the C API for iOS and Android
- **[Python](/python/)** -- Module-level FFI bindings, mirrors the C API
- **[Swift](/bindings/swift/)** -- XCFramework for iOS/macOS with Metal support
- **[Kotlin](/bindings/kotlin/)** -- JNI bindings + Kotlin Multiplatform support
- **[Flutter](/bindings/flutter/)** -- Dart FFI bindings for Android, iOS, and macOS
- **[Rust](/bindings/rust/)** -- Raw `extern "C"` FFI declarations
- **[C++ / Engine API](/docs/cactus_engine.md)** -- Direct C FFI for maximum control

## Getting Started

Once you've picked your binding, head to the **[Quickstart](quickstart.md)** to install and run your first completion.
