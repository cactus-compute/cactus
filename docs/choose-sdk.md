# Choose Your SDK

Not sure which SDK to use? 

Pick the right one for your platform and use case:

|  | React Native | Flutter | Kotlin | Swift | Python | Rust | CLI | C++ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Platforms** | iOS, Android | iOS, Android, macOS | Android, iOS (KMP) | iOS, macOS | Arm Linux, macOS | All | macOS, Linux | All |
| **Install** | npm | build | build | build | build | cargo | brew / source | header |
| **LLM** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Streaming** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Vision** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Transcription** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Function Calling** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **RAG / Embeddings** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Cloud Fallback** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Quick Recommendations

!!! tip "Building a mobile app?"
    - **React Native** -- already have a React Native project? Just `npm install` and go
    - **Flutter** -- cross-platform mobile + Mac with full native bindings
    - **Kotlin** -- native Android apps or Kotlin Multiplatform
    - **Swift** -- native iOS/macOS apps with Apple NPU acceleration

!!! tip "Server-side or scripting?"
    - **Python** -- server-side inference, batch processing, or rapid prototyping
    - **CLI** -- quick model testing and interactive sessions without writing code

!!! tip "Embedding in a native app?"
    - **C++** -- game engines, native desktop apps, or any C/C++ project
    - **Rust** -- systems-level integration with safe FFI bindings

## SDK Documentation

- **[React Native](https://github.com/cactus-compute/cactus-react-native)** -- npm package with React hooks (`useCactusLM`)
- **[Python](/python/)** -- Module-level FFI bindings, mirrors the C API
- **[Swift](/apple/)** -- XCFramework for iOS/macOS with NPU support
- **[Kotlin / Android](/android/)** -- JNI bindings + Kotlin Multiplatform support
- **[Flutter](/flutter/)** -- Dart FFI bindings for Android, iOS, and macOS
- **[Rust](/rust/)** -- Auto-generated FFI bindings via bindgen
- **[C++ / Engine API](/docs/cactus_engine.md)** -- Direct C FFI for maximum control

## Getting Started

Once you've picked your SDK, head to the **[Quickstart](quickstart.md)** to install and run your first completion.
