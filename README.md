![Logo](assets/banner.jpg)

[![Email][gmail-shield]][gmail-url]
[![Discord][discord-shield]][discord-url]
[![Design Docs][docs-shield]][docs-url]
![License](https://img.shields.io/github/license/cactus-compute/cactus?style=for-the-badge)
[![Stars][stars-shield]][github-url]
[![Forks][forks-shield]][github-url]


[gmail-shield]: https://img.shields.io/badge/Gmail-red?style=for-the-badge&logo=gmail&logoColor=white
[gmail-url]: founders@cactuscompute.com

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-blue.svg?style=for-the-badge&logo=linkedin&colorB=blue
[linkedin-url]: https://www.linkedin.com/company/106281696

[discord-shield]: https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white
[discord-url]: https://discord.gg/j4SS7Nwr

[docs-shield]: https://img.shields.io/badge/Design_Docs-009485?style=for-the-badge&logo=readthedocs&logoColor=white
[docs-url]: https://deepwiki.com/cactus-compute/cactus

[website-shield]: https://img.shields.io/badge/Website-black?style=for-the-badge&logo=safari&logoColor=white
[website-url]: https://cactuscompute.com

[stars-shield]: https://img.shields.io/github/stars/cactus-compute/cactus?style=for-the-badge&color=yellow
[forks-shield]: https://img.shields.io/github/forks/cactus-compute/cactus?style=for-the-badge&color=blue
[issues-shield]: https://img.shields.io/github/issues/cactus-compute/cactus?style=for-the-badge
[prs-shield]: https://img.shields.io/github/issues-pr/cactus-compute/cactus?style=for-the-badge
[github-url]: https://github.com/cactus-compute/cactus

Cactus is a lightweight, high-performance framework for running AI models on mobile phones. Cactus has clean and consistent APIs across 
Flutter and React-Native.

Cactus currently leverages GGML backends to support any GGUF model already compatible with [![Llama.cpp](https://img.shields.io/badge/Llama.cpp-000000?style=flat&logo=github&logoColor=white)](https://github.com/ggerganov/llama.cpp), while we focus on broadly supporting every moblie app development platform, as well as upcoming features like:

- phone tool use (gallery search, read email, DM...) 
- thinking mode (planning, evals...) 
- higher-level APIs (sentiments, OCR, TTS...) 

Functionalities that will enhance small models amd make them production-ready!

Contributors with any of the above experiences are welcome! Feel free to submit cool example apps you built with Cactus, issues or tests! 

## Table of Contents

- [Technical Architecture](#technical-architecture)
- [Benchmarks](#benchmarks)
- [Examples](#examples)
- [Getting Started](#getting-started)
- [Flutter (Dart)](#flutter-dart)
- [React Native (TypeScript/JavaScript)](#react-native-typescriptjavascript)

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Applications                        │
└───────────────┬─────────────────┬───────────────────────┘
                │                 │                
┌───────────────┼─────────────────┼───────────────────────-┐
│ ┌─────────────▼─────┐ ┌─────────▼───────┐ ┌─────────────┐|
│ │     React API     │ │   Flutter API   │ │  Native APIs│|
│ └───────────────────┘ └─────────────────┘ └─────────────┘|
│                Platform Bindings                         │
└───────────────┬─────────────────┬───────────────────────-┘
                │                 │                
┌───────────────▼─────────────────▼───────────────────────┐
│              Cactus Core (C++) / llama.rn patches       │
└───────────────┬─────────────────┬───────────────────────┘
                │                 │                
┌───────────────▼─────┐ ┌─────────▼───────────────────────┐
│   Llama.cpp Core    │ │    GGML/GGUF Model Format       │
└─────────────────────┘ └─────────────────────────────────┘
```
- **Features**:
  - Model download from HuggingFace 
  - Text completion and chat completion
  - Streaming token generation 
  - Embedding generation
  - JSON mode with schema validation
  - Chat templates with Jinja2 support
  - Low memory footprint
  - Battery-efficient inference
  - Background processing

## Benchmarks

we created a demo chat app we use for benchmarking:

[![Download App](https://img.shields.io/badge/Download_iOS_App-grey?style=for-the-badge&logo=apple&logoColor=white)](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
[![Download App](https://img.shields.io/badge/Download_Android_App-grey?style=for-the-badge&logo=android&logoColor=white)](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share)

These are day-to-day usage scores, not a controlled testing.

| Device                        | Gemma-3 1B Q8 (toks/sec) | Qwen-2.5 1.5B Q8 (toks/sec) | SmolLM2 360M Q8 (toks/sec) |
|-------------------------------|--------------------------|-----------------------------|----------------------------|
| iPhone 16 Pro Max             | 43                       | 29                          | 103                        |
| iPhone 16 Pro                 | -                        | 28                          | 103                        |
| iPhone 16                     | -                        | 29                          | -                          |
| OnePlus 13 5G                 | 37                       | -                           | -                          |
| Samsung Galaxy S24 Ultra      | 36                       | -                           | -                          |
| OnePlus Open                  | 33                       | -                           | -                          |
| Samsung Galaxy S23 5G         | 32                       | -                           | -                          |
| Samsung Galaxy S24            | 31                       | -                           | -                          |
| iPhone 15 Pro Max             | -                        | 23                          | -                          |
| iPhone 15 Pro                 | -                        | 25                          | 81                         |
| iPhone 15                     | -                        | 25                          | -                          |
| iPhone 14 Pro Max             | -                        | 25                          | -                          |
| iPhone 13 Pro                 | 30                       | -                           | -                          |
| OnePlus 12                    | 30                       | -                           | -                          |
| Galaxy S25 Ultra              | 25                       | -                           | -                          |
| OnePlus 11                    | 23                       | -                           | 64                         |
| iPhone 12 mini                | 22                       | -                           | -                          |
| Redmi K70 Ultra               | 21                       | -                           | -                          |
| Xiaomi 13                     | 21                       | -                           | -                          |
| Samsung Galaxy S24+           | 19                       | -                           | -                          |
| Samsung Galaxy Z Fold 4       | 19                       | -                           | -                          |
| Xiaomi Poco F6 5G             | 19                       | -                           | -                          |
| iPhone 13 mini                | -                        | -                           | 42                         |
| iPhone 12 Pro Max             | -                        | 17                          | -                          |
| Google Pixel 8                | 16                       | -                           | -                          |
| Realme GT2                    | 16                       | -                           | -                          |
| Google Pixel 6a               | 14                       | -                           | -                          |

## Examples
We have ready-to-run-and-deploy examples [here](https://github.com/cactus-compute/cactus/tree/main/examples), you can simply copy, modify and deploy! And reach out if stuck or need hand-holding.

## Getting Started

## Flutter (Dart)

**1. Add Dependency:**
Add `cactus` to your `pubspec.yaml`:
```yaml
dependencies:
  cactus: ^0.0.2
```

**2. Install Dependency:**
Then run:
```bash
flutter pub get
```

Full setup and API details are available in the [Flutter README](cactus-flutter/README.md).

## React Native (TypeScript/JavaScript)

**1. Install Package:**

*   **Using npm:**
    ```bash
    npm install cactus-react-native
    ```
*   **Using yarn:**
    ```bash
    yarn add cactus-react-native
    ```

**2. Install Pods (for iOS):**
If you are not using Expo and are building for iOS, install the pods:
```bash
npx pod-install
```

For more detailed documentation and examples, see the [React Native README](cactus-react/README.md).