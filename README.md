<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

[![Email][gmail-shield]][gmail-url]&nbsp;&nbsp;&nbsp;[![Discord][discord-shield]][discord-url]&nbsp;&nbsp;&nbsp;[![Design Docs][docs-shield]][docs-url]&nbsp;&nbsp;&nbsp;

[gmail-shield]: https://img.shields.io/badge/Gmail-red?style=for-the-badge&logo=gmail&logoColor=white
[gmail-url]: mailto:founders@cactuscompute.com

[discord-shield]: https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white
[discord-url]: https://discord.gg/j4SS7Nwr

[docs-shield]: https://img.shields.io/badge/DeepWiki-009485?style=for-the-badge&logo=readthedocs&logoColor=white
[docs-url]: https://deepwiki.com/cactus-compute/cactus

Cactus is a lightweight, high-performance framework for running AI models on mobile devices, with simple and consistent APIs across Flutter and React-Native. Cactus currently leverages GGML backends to support any GGUF model already compatible with Llama.cpp. 

## ![Shipped](https://img.shields.io/badge/Shipped_Features-grey.svg?style=for-the-badge)

- Text completion and chat completion
- Vision Language Models
- Streaming token generation 
- Embedding generation
- Text-to-speech model support (early stages)
- JSON mode with schema validation
- Chat templates with Jinja2 support
- Low memory footprint
- Battery-efficient inference
- Background processing

## ![Upcoming](https://img.shields.io/badge/Upcoming_Features-grey.svg?style=for-the-badge)

- Agentic workflows (cross-app interactions etc.)
- Phone tool use (gallery search, read email, DM...) 
- Thinking mode (planning, evals...) 
- Higher-level APIs (sentiments, OCR, TTS...) 

## ![Architecture](https://img.shields.io/badge/Architecture-grey.svg?style=for-the-badge)

```
┌─────────────────────────────────────────┐
│                   Apps                  │
└─────────────┬─────────────┬─────────────┘
              │             │              
┌─────────────▼─────────────▼─────────────┐
│   ┌───-───┐   ┌────-────┐   ┌───-───┐   │
│   │ React │   │ Flutter │   │ Native│   │
│   └───────┘   └─────────┘   └───────┘   │
│                 Bindings                │
└─────────────┬─────────────┬─────────────┘
              │             │              
┌─────────────▼─────────────▼─────────────┐
│              Cactus Core C++            │
└─────────────┬─────────────┬─────────────┘
              │             │       
       ┌─────-▼────┐  ┌─────▼─────┐  
       │ Llama.cpp │  │ GGML/GGUF │  
       └───────────┘  └───────────┘  
```

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **Update `pubspec.yaml`:**
    Add `cactus` to your project's dependencies. Ensure you have `flutter: sdk: flutter` (usually present by default).
    ```yaml
    dependencies:
      flutter:
        sdk: flutter
      cactus: ^0.0.3
    ```
2.  **Install dependencies:**
    Execute the following command in your project terminal:
    ```bash
    flutter pub get
    ```

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **Install the `cactus-react-native` package:**
    Using npm:
    ```bash
    npm install cactus-react-native
    ```
    Or using yarn:
    ```bash
    yarn add cactus-react-native
    ```
2.  **Install iOS Pods (if not using Expo):**
    For native iOS projects, ensure you link the native dependencies. Navigate to your `ios` directory and run:
    ```bash
    npx pod-install
    ```

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus backend is written in C/C++, layered on top of GGML/GGUF to support models in the GGUF format. Developers and contributors in this niche can easily get started with examples for:

N/B: Should have `CMake` installed, or install with `brew install cmake` (on macOS) or standard package managers on Linux.

*   **Language Models:**
    1.  Navigate to the example directory: `cd examples/cpp-llm`
    2.  Make the build script executable (only needs to be done once): `chmod +x build.sh`
    3.  Run the example: `./build.sh` (This will download the Qwen 3 model)
    4.  Alter model hyperparameters and prompts by modifying `examples/cpp-llm/main.cpp`. 

*   **Vision-Language Models:**
    1.  Navigate to the example directory: `cd examples/cpp-vlm`
    2.  Make the build script executable (only needs to be done once): `chmod +x build.sh`
    3.  Run the example: `./build.sh` (This will download the [SmolVLM model](https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF))
    4.  Alter model hyperparameters, prompts, and input image by modifying `examples/cpp-vlm/main.cpp`.

*   **Text-to-Speech:**
    1.  Navigate to the example directory: `cd examples/cpp-tts`
    2.  Make the build script executable (only needs to be done once): `chmod +x build.sh`
    3.  Run the example: `./build.sh` (This will download the OuteTTS model)
    4.  Alter model hyperparameters and prompts by modifying `examples/cpp-tts/main.cpp`.

## ![Docs](https://img.shields.io/badge/Documentations-grey.svg?style=for-the-badge)

We host our docs on Deep Wiki, so you can additionally ask Devin any question about Cactus! It does not index frequently enough to keep up with our update speed though, so we have manually written docs for the APIs

- [Deep Wiki](url)
- [C++ Docs](docs/core.md)
- [Flutter Docs](cactus-flutter/README.md)
- [React-Native Docs](cactus-react/README.md)


## ![Example Apps](https://img.shields.io/badge/Examples-grey.svg?style=for-the-badge)

We have ready-to-run-and-deploy example apps:

1. [Flutter Chat](examples/flutter-chat)
2. [Flutter Notes](examples/flutter-notes)
3. [React Chat](examples/react-chat)
4. [React Productivity](examples/react-productivity)
5. [React Diary](examples/react-diary)
6. [C++ Language Model (LLM)](examples/cpp-llm)
7. [C++ Vision-Language Model (VLM)](examples/cpp-vlm)
8. [C++ Text-to-Speech (TTS)](examples/cpp-tts)

## ![Contributions](https://img.shields.io/badge/Contributions-grey.svg?style=for-the-badge)

We welcome contributions! Here's how you can help:

1.  **Clone the Repository:** For simplicity at this stage, clone the repository to your local machine.
2.  **Create a Branch:** Create a new branch for your contribution.
3.  **Implement Changes:** Make your desired changes or additions.
4.  **Run Tests (for C/C++ contributors):**
    *   Ensure all tests pass by running the script: `scripts/test-cactus.sh`
5.  **Flutter & React-Native Testing:** (Testing procedures for these platforms will be updated soon.)
6.  **Submit a Pull Request (PR):** Once you're ready, submit a PR with your changes!
7.  **Contribution Ideas** Example apps, polishing the examples, features, submitting benchmarks, etc.

## ![Performance](https://img.shields.io/badge/Performance-grey.svg?style=for-the-badge)

| Device                        | Gemma-3 1B Q8 (toks/sec) | Qwen-2.5 1.5B Q8 (toks/sec) | SmolLM2 360M Q8 (toks/sec) |
|:------------------------------|:------------------------:|:---------------------------:|:--------------------------:|
| iPhone 16 Pro Max             |            43            |             29              |            103             |
| iPhone 16 Pro                 |            -             |             28              |            103             |
| iPhone 16                     |            -             |             29              |             -              |
| OnePlus 13 5G                 |            37            |             -               |             -              |
| Samsung Galaxy S24 Ultra      |            36            |             -               |             -              |
| OnePlus Open                  |            33            |             -               |             -              |
| Samsung Galaxy S23 5G         |            32            |             -               |             -              |
| Samsung Galaxy S24            |            31            |             -               |             -              |
| iPhone 15 Pro Max             |            -             |             23              |             -              |
| iPhone 15 Pro                 |            -             |             25              |             81             |
| iPhone 15                     |            -             |             25              |             -              |
| iPhone 14 Pro Max             |            -             |             25              |             -              |
| iPhone 13 Pro                 |            30            |             -               |             -              |
| OnePlus 12                    |            30            |             -               |             -              |
| Galaxy S25 Ultra              |            25            |             -               |             -              |
| OnePlus 11                    |            23            |             -               |             64             |
| iPhone 12 mini                |            22            |             -               |             -              |
| Redmi K70 Ultra               |            21            |             -               |             -              |
| Xiaomi 13                     |            21            |             -               |             -              |
| Samsung Galaxy S24+           |            19            |             -               |             -              |
| Samsung Galaxy Z Fold 4       |            19            |             -               |             -              |
| Xiaomi Poco F6 5G             |            19            |             -               |             -              |
| iPhone 13 mini                |            -             |             -               |             42             |
| iPhone 12 Pro Max             |            -             |             17              |             -              |
| Google Pixel 8                |            16            |             -               |             -              |
| Realme GT2                    |            16            |             -               |             -              |
| Google Pixel 6a               |            14            |             -               |             -              |

## ![Demo](https://img.shields.io/badge/Demo-grey.svg?style=for-the-badge)

We created a demo chat app we use for benchmarking:

[![Download App](https://img.shields.io/badge/Download_iOS_App-grey?style=for-the-badge&logo=apple&logoColor=white)](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
[![Download App](https://img.shields.io/badge/Download_Android_App-grey?style=for-the-badge&logo=android&logoColor=white)](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share)
