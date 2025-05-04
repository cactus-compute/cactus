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
- iPhone 16 Pro Max: ~45 toks/sec
- iPhone 13 Pro: ~30 toks/sec
- Galaxy A14: ~6 toks/sec
- Galaxy S24 plus: ~20 toks/sec 
- Galaxy S21: ~14 toks/sec 
- Google Pixel 6a: ~14 toks/sec 

SmollLM 135m INT8: 
- iPhone 13 Pro: ~180 toks/sec
- Galaxy A14: ~30 toks/sec
- Galaxy S21: ~42 toks/sec
- Google Pixel 6a: ~38 toks/sec
- Huawei P60 Lite (Gran's phone) ~8toks/sec


## Getting Started

### ✅ React Native (TypeScript/JavaScript)

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

### ✅ Android (Kotlin/Java)

**1. Add Repository to `settings.gradle.kts`:**

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS) // Optional but recommended
    repositories {
        google()
        mavenCentral()
        // Add GitHub Packages repository for Cactus
        maven {
            name = "GitHubPackagesCactusCompute"
            url = uri("https://maven.pkg.github.com/cactus-compute/cactus")
        }
    }
}
```

**2. Add Dependency to Module's `build.gradle.kts`:**

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("io.github.cactus-compute:cactus-android:0.0.1")
}
```

**3. Basic Usage (Kotlin):**

```kotlin
import com.cactus.android.LlamaContext
import com.cactus.android.LlamaInitParams
import com.cactus.android.LlamaCompletionParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// In an Activity, ViewModel, or coroutine scope

suspend fun runInference() {
    var llamaContext: LlamaContext? = null
    try {
        // Initialize (off main thread)
        llamaContext = withContext(Dispatchers.IO) {
            LlamaContext.create(
                params = LlamaInitParams(
                    modelPath = "path/to/your/model.gguf",
                    nCtx = 2048, nThreads = 4
                )
            )
        }

        // Complete (off main thread)
        val result = withContext(Dispatchers.IO) {
            llamaContext?.complete(
                prompt = "Explain quantum computing in simple terms",
                params = LlamaCompletionParams(temperature = 0.7f, nPredict = 512)
            ) { partialResultMap ->
                val token = partialResultMap["token"] as? String ?: ""
                print(token) // Process stream on background thread
                true // Continue generation
            }
        }
        println("\nFinal text: ${result?.text}")

    } catch (e: Exception) {
        // Handle errors
        println("Error: ${e.message}")
    } finally {
        // Clean up (off main thread)
        withContext(Dispatchers.IO) {
             llamaContext?.close()
        }
    }
}
```

For more detailed documentation and examples, see the [Android README](android/README.md).

### 🚧 Swift (in developement)

### 🚧 Flutter (in developement)

```
