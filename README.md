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
[discord-url]: https://discord.gg/cBT6jcCF

[docs-shield]: https://img.shields.io/badge/Design_Docs-009485?style=for-the-badge&logo=readthedocs&logoColor=white
[docs-url]: https://deepwiki.com/cactus-compute/cactus

[website-shield]: https://img.shields.io/badge/Website-black?style=for-the-badge&logo=safari&logoColor=white
[website-url]: https://cactuscompute.com

[stars-shield]: https://img.shields.io/github/stars/cactus-compute/cactus?style=for-the-badge&color=yellow
[forks-shield]: https://img.shields.io/github/forks/cactus-compute/cactus?style=for-the-badge&color=blue
[issues-shield]: https://img.shields.io/github/issues/cactus-compute/cactus?style=for-the-badge
[prs-shield]: https://img.shields.io/github/issues-pr/cactus-compute/cactus?style=for-the-badge
[github-url]: https://github.com/cactus-compute/cactus

Cactus is a lightweight, high-performance framework for running AI models on mobile phones. Cactus has unified and consistent APIs across 
- React-Native
- Android/Kotlin
- Android/Java
- iOS/Swift
- iOS/Objective-C++
- Flutter/Dart

Cactus currently leverages GGML backends to support any GGUF model already compatible with [![Llama.cpp](https://img.shields.io/badge/Llama.cpp-000000?style=flat&logo=github&logoColor=white)](https://github.com/ggerganov/llama.cpp), while we focus on broadly supporting every moblie app development platform, as well as upcoming features like:

- MCP
- phone tool use
- thinking
- prompt-enhancement
- higher-level APIs

Contributors with any of the above experiences are welcome! Feel free to submit cool example apps you built with Cactus, issues or tests! 

Cactus Models coming soon.

## Table of Contents

- [Technical Architecture](#technical-architecture)
- [Features](#features)
- [Benchmarks](#benchmarks)
- [Getting Started](#getting-started)
  - [React Native](#react-native-shipped)
  - [Android](#android-currently-testing)
  - [Swift](#ios-in-developement)
  - [Flutter](#flutter-in-development)
  - [C++ (Raw backend)](#c-raw-backend)
- [License](#license)

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
│                   Cactus Core (C++)                     │
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

we created a little chat app for demo, you can try other models and report your finding here, [download the app](https://lnkd.in/dYGR54hn)

Gemma 1B INT8:
- iPhone 13 Pro: ~30 toks/sec 
- Galaxy S21: ~14 toks/sec 
- Google Pixel 6a: ~14 toks/sec 
- Galaxy S24 plus: ~20 toks/sec 

SmollLM 135m INT8: 
- iPhone 13 Pro: ~180 toks/sec
- Galaxy S21: ~42 toks/sec
- Google Pixel 6a: ~38 toks/sec
- Huawei P60 Lite (Gran's phone) ~8toks/sec


## Getting Started

### ✅ React Native (shipped)

```bash
npm install cactus-react-native
# or
yarn add cactus-react-native

# For iOS, install pods if not on Expo
npx pod-install
```
```typescript
import { initLlama, LlamaContext } from 'cactus-react-native';

// Load model
const context = await initLlama({
  model: 'models/llama-2-7b-chat.gguf', // Path to your model
  n_ctx: 2048,
  n_batch: 512,
  n_threads: 4
});

// Generate completion
const result = await context.completion({
  prompt: 'Explain quantum computing in simple terms',
  temperature: 0.7,
  top_k: 40,
  top_p: 0.95,
  n_predict: 512
}, (token) => {
  // Process each token
  process.stdout.write(token.token);
});

// Clean up
await context.release();
```

For more detailed documentation and examples, see the [React Native README](react/README.md).

### 🚧 Android (currently testing)

```gradle
<!-- Add to your `build.gradle` -->
dependencies {
    implementation 'com.cactuscompute:cactus-android:x.y.z'
}
```
```kotlin
import com.cactus.LlamaContext

// Load model
val llamaContext = LlamaContext.createContext(
    applicationContext,
    "models/llama-2-7b-chat.gguf",
    LlamaContextParams(
        nCtx = 2048,
        nBatch = 512, 
        nThreads = 4
    )
)

// Set up completion
val result = llamaContext.completion(
    CompletionParams(
        prompt = "Explain quantum computing in simple terms",
        temperature = 0.7f,
        topK = 40,
        topP = 0.95f,
        nPredict = 512
    )
) { token ->
    // Stream tokens as they're generated
    print(token.text)
}

// Clean up
llamaContext.release()
```

For more detailed documentation and examples, see the [Android README](android/README.md).

### 🚧 Swift (in developement)

```ruby
# Simply copy the swift/CactusSwift into your project for now
```
```swift
import Cactus

// Load model
let context = try CactusContext(
    modelPath: "models/llama-2-7b-chat.gguf",
    contextParams: ContextParams(
        contextSize: 2048,
        batchSize: 512,
        threadCount: 4
    )
)

// Generate completion
try context.completion(
    params: CompletionParams(
        prompt: "Explain quantum computing in simple terms",
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxTokens: 512
    )
) { token in
    // Process each token as it's generated
    print(token.text, terminator: "")
}

// Clean up
context.release()
```

For more detailed documentation and examples, see the [iOS README](swift/README.md).

### 🚧 Flutter (in development)

```bash
flutter pub add cactus_flutter
```
```dart
import 'package:cactus_flutter/cactus_flutter.dart';

// Load model
final context = await CactusContext.initialize(
  modelPath: 'models/llama-2-7b-chat.gguf',
  contextSize: 2048,
  batchSize: 512,
  threadCount: 4,
);

// Generate completion
final result = await context.completion(
  prompt: 'Explain quantum computing in simple terms',
  temperature: 0.7,
  topK: 40,
  topP: 0.95,
  maxTokens: 512,
  onToken: (String token) {
    // Process each token as it's generated
    print(token);
  }
);

// Clean up
await context.release();
```

For more detailed documentation and examples, see the [Flutter README](flutter/README.md).

### ✅ C++ (Raw backend)
```bash
// Use see the test folder
chmod +x scripts/test-cactus.sh
scripts/test-cactus.sh
```

```cpp 
#include "cactus.h"

int main() {
    // Initialize parameters
    cactus::common_params params;
    params.model = "models/llama-2-7b-chat.gguf";
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.n_threads = 4;
    
    // Create context and load model
    cactus::cactus_context ctx;
    if (!ctx.loadModel(params)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // Set up completion parameters
    params.prompt = "Explain quantum computing in simple terms";
    params.n_predict = 512;
    params.sampling.temp = 0.7f;
    params.sampling.top_k = 40;
    params.sampling.top_p = 0.95f;
    
    // Generate completion
    ctx.loadPrompt();
    ctx.beginCompletion();
    
    std::string result;
    while (true) {
        auto token_output = ctx.doCompletion();
        if (!ctx.has_next_token) break;
        std::cout << ctx.generated_text;
        result += ctx.generated_text;
    }
    
    return 0;
}
```

For more detailed documentation and examples, see the [C++ README](cactus/README.md).
