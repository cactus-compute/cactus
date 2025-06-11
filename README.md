<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

[![Email][gmail-shield]][gmail-url]&nbsp;&nbsp;&nbsp;[![Discord][discord-shield]][discord-url]&nbsp;&nbsp;&nbsp;[![Design Docs][docs-shield]][docs-url]&nbsp;&nbsp;&nbsp;

[gmail-shield]: https://img.shields.io/badge/Gmail-red?style=for-the-badge&logo=gmail&logoColor=white
[gmail-url]: mailto:founders@cactuscompute.com

[discord-shield]: https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white
[discord-url]: https://discord.gg/j4SS7Nwr

[docs-shield]: https://img.shields.io/badge/DeepWiki-009485?style=for-the-badge&logo=readthedocs&logoColor=white
[docs-url]: https://deepwiki.com/cactus-compute/cactus

Cactus is a lightweight, high-performance framework for running AI models on mobile devices, with simple and consistent APIs across C/C++, Dart/Flutter and Ts/React-Native. Cactus currently leverages GGML backends to support any GGUF model already compatible with Llama.cpp. 

## ![Features](https://img.shields.io/badge/Shipped_Features-grey.svg?style=for-the-badge)

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

## ![Why Cactus?](https://img.shields.io/badge/Shipped_Features-grey.svg?style=for-the-badge)

- APIs are increasingly becoming expensive, especially at scale 
- Private and local, data do not leave the device whatsoever 
- Low-latency anf fault-tolerant, no need for users to have internet connections 
- Small models excell at most tasks, big APIs are often only better at enterprise tasks like coding
- Freedom to use any GGUF model, unlike Apple Foundations and Google AI Core 
- React-Native and Flutter APIs, no need for separate Swift and Android setups 
- iOS xcframework and JNILibs ifworking in native setup
- Neat and tiny C++ build for custom hardware 

## ![Architecture](https://img.shields.io/badge/Architecture-grey.svg?style=for-the-badge)

```
┌─────────────────────────────────────────┐
│              Your apps                  │
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
      cactus: ^0.1.0
    ```
2.  **Install dependencies:**
    Execute the following command in your project terminal:
    ```bash
    flutter pub get
    ```
3. **Basic Flutter Text Completion**
    ```dart
    import 'package:cactus/cactus.dart';

    Future<String> basicCompletion() async {
    // Initialize context
    final context = await CactusContext.init(CactusInitParams(
        modelPath: '/path/to/model.gguf',
        contextSize: 2048,
        threads: 4,
    ));

    // Generate response
    final result = await context.completion(CactusCompletionParams(
        messages: [
        ChatMessage(role: 'user', content: 'Hello, how are you?')
        ],
        maxPredictedTokens: 100,
        temperature: 0.7,
    ));

    context.free();
    return result.text;
    }
    ```
  To learn more, see the [Flutter Docs](docs/flutter.md). It covers chat design, embeddings, multimodal models, text-to-speech, and more.

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

3. **Basic React-Native Text Completion**
    ```typescript
    import { initLlama } from 'cactus-react-native';

    // Initialize a context
    const context = await initLlama({
    model: '/path/to/your/model.gguf',
    n_ctx: 2048,
    n_threads: 4,
    });

    // Generate text
    const result = await context.completion({
    messages: [
        { role: 'user', content: 'Hello, how are you?' }
    ],
    n_predict: 100,
    temperature: 0.7,
    });

    console.log(result.text);
    ```
 To learn more, see the [React Docs](docs/react.md). It covers chat design, embeddings, multimodal models, text-to-speech, and more.

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus backend is written in C/C++ and can run directly on any ARM/X86/Raspberry PI hardware like phones, smart tvs, watches, speakers, cameras, laptops etc. 

1. **Setup**
    You need `CMake 3.14+` installed, or install with `brew install cmake` (on macOS) or standard package managers on Linux.

2. **Build from Source**
    ```bash
    git clone https://github.com/your-org/cactus.git
    cd cactus
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    ```

3. **CMake Integration**
    Add to your `CMakeLists.txt`:

    ```cmake
    # Add Cactus as subdirectory
    add_subdirectory(cactus)

    # Link to your target
    target_link_libraries(your_target cactus)
    target_include_directories(your_target PRIVATE cactus)

    # Requires C++17 or higher 
    ```

4. **Basic Text Completion**
    ```cpp
    #include "cactus/cactus.h"
    #include <iostream>

    int main() {
        cactus::cactus_context context;
        
        // Configure parameters
        common_params params;
        params.model.path = "model.gguf";
        params.n_ctx = 2048;
        params.n_threads = 4;
        params.n_gpu_layers = 99; // Use GPU acceleration
        
        // Load model
        if (!context.loadModel(params)) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        
        // Set prompt
        context.params.prompt = "Hello, how are you?";
        context.params.n_predict = 100;
        
        // Initialize sampling
        if (!context.initSampling()) {
            std::cerr << "Failed to initialize sampling" << std::endl;
            return 1;
        }
        
        // Generate response
        context.beginCompletion();
        context.loadPrompt();
        
        while (context.has_next_token && !context.is_interrupted) {
            auto token_output = context.doCompletion();
            if (token_output.tok == -1) break;
        }
        
        std::cout << "Response: " << context.generated_text << std::endl;
        return 0;
    }
    ```
 To learn more, see the [C++ Docs](docs/cpp.md). It covers chat design, embeddings, multimodal models, text-to-speech, and more.


## ![Using this Repo & Example Apps](https://img.shields.io/badge/Using_Repo_And_Examples-grey.svg?style=for-the-badge)

First, clone the repo and make all scripts executable with `git clone https://github.com/cactus-compute/cactus.git && chmod +x scripts/*.sh`

1. **Flutter**
    - Build the Android JNILibs with `scripts/build-flutter-android.sh`.
    - Build the Flutter Plugin with `scripts/build-flutter-android.sh`.
    - Navigate to the example app with `cd examples/flutter`.
    - Open your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
    - Always start app with this combo `flutter clean && flutter pub get && flutter run`.
    - Play with the app, and make changes either to the example app or plugin as desired.

2. **React Native**
    - Build the Android JNILibs with `scripts/build-react-android.sh`.
    - Build the Flutter Plugin with `scripts/build-react-android.sh`.
    - Navigate to the example app with `cd examples/react`.
    - Setup your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
    - Always start app with this combo `yarn && yarn ios` or `yarn && yarn android`.
    - Play with the app, and make changes either to the example app or package as desired.
    - For now, if changes are made in the package, you would manually copy the files/folders into the `examples/react/node_modules/cactus-react-native`.

2. **C/C++**
    - Navigate to the example app with `cd examples/cpp`.
    - There are multiple main files `main_vlm, main_llm, main_embed, main_tts`.
    - Build both the libraries and executable using `build.sh`.
    - Run with one of the executables `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`.
    - Try different models and make changes as desired.

4. **Contributing**
    - To contribute a bug fix, create a branch after making your changes with `git checkout -b <branch-name>` and submit a PR. 
    - To contribute a feature, please raise as issue first so it can be discussed, to avoid intersecting with someone else.
    - [Join our discord](https://discord.gg/j4SS7Nwr)

## ![Performance](https://img.shields.io/badge/Performance-grey.svg?style=for-the-badge)

| Device                        | Gemma-3 1B Q8 (toks/sec) | Qwen-2.5 1.5B Q8 (toks/sec) |  
|:------------------------------|:------------------------:|:---------------------------:|
| iPhone 16 Pro Max             |            46            |             37              |
| iPhone 16 Pro                 |            46            |             37              |
| iPhone 16                     |            42            |             36              |
| iPhone 15 Pro Max             |            39            |             31              |
| iPhone 15 Pro                 |            39            |             31              |
| iPhone 14 Pro Max             |            38            |             29              |
| OnePlus 13 5G                 |            37            |             -               |
| Samsung Galaxy S24 Ultra      |            36            |             -               |
| iPhone 15                     |            36            |             25              |
| OnePlus Open                  |            33            |             -               |
| Samsung Galaxy S23 5G         |            32            |             -               |
| Samsung Galaxy S24            |            31            |             -               |
| iPhone 13 Pro                 |            30            |             -               |
| OnePlus 12                    |            30            |             -               |
| Galaxy S25 Ultra              |            25            |             -               |
| OnePlus 11                    |            23            |             -               |
| iPhone 13 mini                |            22            |             -               |
| Redmi K70 Ultra               |            21            |             -               |
| Xiaomi 13                     |            21            |             -               |
| Samsung Galaxy S24+           |            19            |             -               |
| Samsung Galaxy Z Fold 4       |            19            |             -               |
| Xiaomi Poco F6 5G             |            19            |             -               |

## ![Demo](https://img.shields.io/badge/Demo-grey.svg?style=for-the-badge)

We created a demo chat app we use for benchmarking:

[![Download App](https://img.shields.io/badge/Download_iOS_App-grey?style=for-the-badge&logo=apple&logoColor=white)](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
[![Download App](https://img.shields.io/badge/Download_Android_App-grey?style=for-the-badge&logo=android&logoColor=white)](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share)

## ![Recommendations](https://img.shields.io/badge/Our_Recommendations-grey.svg?style=for-the-badge)
You can run up to 10B models at Q4 on most devices, but it is not recommended for production due to file size, speed, battery, heating performance. 
We generally give the following recommendation.

- **Language Generation**: `SmolLM2-360m`, `Qwen-3-600m-Q6`, `Gemma-3-1B-Q6`, `Qwen-3-1.7B-Q6`
- **Multimodal Language Generation**: `Smol-VLM-500m-Q6`, `Gemma-3n-2B-Q6`
- **Embeddings**: `nomic-v2-moe-300m-Q6`, `jina-v3-570m-Q6`
- **Text-To-Speech**: `OuteTTS-0.2-500m-Q6`

Gemma-3n-2B-Q6 is a great omni model and beats GPT 4.1 across many metrics. It is multimodal (vision, audio) and can perfectly be used for embedding
text, images, audio, as well as zero-shot classification and more, with clever prompt engineering. We are trying hard to get the weights.