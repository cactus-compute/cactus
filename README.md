<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 Translations

🇬🇧 English | 🇪🇸 [Español](docs/README.es.md) | 🇫🇷 [Français](docs/README.fr.md) | 🇨🇳 [中文](docs/README.zh.md) | 🇯🇵 [日本語](docs/README.ja.md) | 🇮🇳 [हिंदी](docs/README.hi.md) | 🇩🇪 [Deutsch](docs/README.de.md)
<br/>

Cross-platform framework for deploying LLM/VLM/TTS models locally in your app.

- Available in Flutter, React-Native and Kotlin Multiplatform.
- Supports any GGUF model you can find on Huggingface; Qwen, Gemma, Llama, DeepSeek etc.
- Run LLMs, VLMs, Embedding Models, TTS models and more.
- Accommodates from FP32 to as low as 2-bit quantized models, for efficiency and less device strain. 
- Chat templates with Jinja2 support and token streaming.

[CLICK TO JOIN OUR DISCORD!](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[CLICK TO VISUALISE AND QUERY REPO](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **Install:**
    Execute the following command in your project terminal:
    ```bash
    flutter pub add cactus
    ```
2. **Flutter Text Completion**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
    );

    final messages = [ChatMessage(role: 'user', content: 'Hello!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);
    ```
3. **Flutter Embedding**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
        generateEmbeddings: True,
    );

    final text = 'Your text to embed';
    final result = await lm.embedding(text);
    ```
4. **Flutter VLM Completion**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
        mmprojUrl: 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
    );

    final messages = [ChatMessage(role: 'user', content: 'Describe this image')];

    final response = await vlm.completion(
        messages, 
        imagePaths: ['/absolute/path/to/image.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```
5. **Flutter Cloud Fallback**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
        cactusToken: 'enterprise_token_here', 
    );

    final messages = [ChatMessage(role: 'user', content: 'Hello!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local (default): strictly only run on-device
    // localfirst: fallback to cloud if device fails
    // remotefirst: primarily remote, run local if API fails
    // remote: strictly run on cloud 
    final embedding = await lm.embedding('Your text', mode: 'localfirst');
    ```

  N/B: See the [Flutter Docs](https://github.com/cactus-compute/cactus/blob/main/flutter) for more.

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **Install the `cactus-react-native` package:**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **React-Native Text Completion**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/path/to/model.gguf', // this is a local model file inside the app sandbox
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: 'Hello!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```
3. **React-Native Embedding**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/path/to/model.gguf', // local model file inside the app sandbox
        n_ctx: 2048,
        embedding: True,
    });

    const text = 'Your text to embed';
    const params = { normalize: True };
    const result = await lm.embedding(text, params);
    ```

4. **React-Native VLM**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/path/to/vision-model.gguf', // local model file inside the app sandbox
        mmproj: '/path/to/mmproj.gguf', // local model file inside the app sandbox
    });

    const messages = [{ role: 'user', content: 'Describe this image' }];

    const params = {
        images: ['/absolute/path/to/image.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
5. **React-Native Agents**
    
    ```typescript
    import { CactusAgent } from 'cactus-react-native';

    // we recommend Qwen 3 family, 0.6B is great
    const { agent, error } = await CactusAgent.init({
        model: '/path/to/model.gguf', 
        n_ctx: 2048,
    });

    const weatherTool = agent.addTool(
        (location: string) => `Weather in ${location}: 72°F, sunny`,
        'Get current weather for a location',
        {
            location: { type: 'string', description: 'City name', required: true }
        }
    );

    const messages = [{ role: 'user', content: 'What\'s the weather in NYC?' }];
    const result = await agent.completionWithTools(messages, {
    n_predict: 200,
    temperature: 0.7,
    });

    await agent.release();
    ```

Get started with an [example app](https://github.com/cactus-compute/example-react-agents/) built using `CactusAgent`.

See the [React Docs](https://github.com/cactus-compute/cactus/blob/main/react) for more.

## ![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **Add Maven Dependency:**
    Add to your KMP project's `build.gradle.kts`:
    ```kotlin
    kotlin {
        sourceSets {
            commonMain {
                dependencies {
                    implementation("com.cactus:library:0.2.4")
                }
            }
        }
    }
    ```

2. **Platform Setup:**
    - **Android:** Works automatically - native libraries included.
    - **iOS:** In Xcode: File → Add Package Dependencies → Paste `https://github.com/cactus-compute/cactus` → Click Add

3. **Kotlin Multiplatform Text Completion**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // Set to 99 for full GPU offload
        )
        
        val downloadSuccess = lm.download(
            url = "path/to/hugginface/gguf",
            filename = "model_filename.gguf"
        )
        val initSuccess = lm.init("qwen3-600m.gguf")
        
        val result = lm.completion(
            prompt = "Hello!",
            maxTokens = 100,
            temperature = 0.7f
        )
    }
    ```

4. **Kotlin Multiplatform Speech To Text**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val stt = CactusSTT(
            language = "en-US",
            sampleRate = 16000,
            maxDuration = 30
        )
        
        // Only supports default Vosk STT model for Android & Apple FOundation Model
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()
        
        val result = stt.transcribe()
        result?.let { sttResult ->
            println("Transcribed: ${sttResult.text}")
            println("Confidence: ${sttResult.confidence}")
        }
        
        // Or transcribe from audio file
        val fileResult = stt.transcribeFile("/path/to/audio.wav")
    }
    ```

5. **Kotlin Multiplatform VLM**
    ```kotlin
    import com.cactus.CactusVLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val vlm = CactusVLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // Set to 99 for full GPU offload
        )
        
        val downloadSuccess = vlm.download(
            modelUrl = "path/to/hugginface/gguf",
            mmprojUrl = "path/to/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")
        
        val result = vlm.completion(
            prompt = "Describe this image",
            imagePath = "/path/to/image.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

  N/B: See the [Kotlin Docs](https://github.com/cactus-compute/cactus/blob/main/kotlin) for more.

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus backend is written in C/C++ and can run directly on phones, smart tvs, watches, speakers, cameras, laptops etc. See the [C++ Docs](https://github.com/cactus-compute/cactus/blob/main/cpp) for more.


## ![Using this Repo & Example Apps](https://img.shields.io/badge/Using_Repo_And_Examples-grey.svg?style=for-the-badge)

First, clone the repo with `git clone https://github.com/cactus-compute/cactus.git`, cd into it and make all scripts executable with `chmod +x scripts/*.sh`

1. **Flutter**
    - Build the Android JNILibs with `scripts/build-flutter-android.sh`.
    - Build the Flutter Plugin with `scripts/build-flutter.sh`. (MUST run before using example)
    - Navigate to the example app with `cd flutter/example`.
    - Open your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
    - Always start app with this combo `flutter clean && flutter pub get && flutter run`.
    - Play with the app, and make changes either to the example app or plugin as desired.

2. **React Native**
    - Build the Android JNILibs with `scripts/build-react-android.sh`.
    - Build the Flutter Plugin with `scripts/build-react.sh`.
    - Navigate to the example app with `cd react/example`.
    - Setup your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
    - Always start app with this combo `yarn && yarn ios` or `yarn && yarn android`.
    - Play with the app, and make changes either to the example app or package as desired.
    - For now, if changes are made in the package, you would manually copy the files/folders into the `examples/react/node_modules/cactus-react-native`.

3. **Kotlin Multiplatform**
    - Build the Android JNILibs with `scripts/build-flutter-android.sh`. (Flutter & Kotlin share same JNILibs)
    - Build the Kotlin library with `scripts/build-kotlin.sh`. (MUST run before using example)
    - Navigate to the example app with `cd kotlin/example`.
    - Open your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
    - Always start app with `./gradlew :composeApp:run` for desktop or use Android Studio/Xcode for mobile.
    - Play with the app, and make changes either to the example app or library as desired.

4. **C/C++**
    - Navigate to the example app with `cd cactus/example`.
    - There are multiple main files `main_vlm, main_llm, main_embed, main_tts`.
    - Build both the libraries and executable using `build.sh`.
    - Run with one of the executables `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`.
    - Try different models and make changes as desired.

5. **Contributing**
    - To contribute a bug fix, create a branch after making your changes with `git checkout -b <branch-name>` and submit a PR. 
    - To contribute a feature, please raise as issue first so it can be discussed, to avoid intersecting with someone else.
    - [Join our discord](https://discord.gg/SdZjmfWQ)

## ![Performance](https://img.shields.io/badge/Performance-grey.svg?style=for-the-badge)

| Device                        |  Gemma3 1B Q4 (toks/sec) |    Qwen3 4B Q4 (toks/sec)   |  
|:------------------------------|:------------------------:|:---------------------------:|
| iPhone 16 Pro Max             |            54            |             18              |
| iPhone 16 Pro                 |            54            |             18              |
| iPhone 16                     |            49            |             16              |
| iPhone 15 Pro Max             |            45            |             15              |
| iPhone 15 Pro                 |            45            |             15              |
| iPhone 14 Pro Max             |            44            |             14              |
| OnePlus 13 5G                 |            43            |             14              |
| Samsung Galaxy S24 Ultra      |            42            |             14              |
| iPhone 15                     |            42            |             14              |
| OnePlus Open                  |            38            |             13              |
| Samsung Galaxy S23 5G         |            37            |             12              |
| Samsung Galaxy S24            |            36            |             12              |
| iPhone 13 Pro                 |            35            |             11              |
| OnePlus 12                    |            35            |             11              |
| Galaxy S25 Ultra              |            29            |             9               |
| OnePlus 11                    |            26            |             8               |
| iPhone 13 mini                |            25            |             8               |
| Redmi K70 Ultra               |            24            |             8               |
| Xiaomi 13                     |            24            |             8               |
| Samsung Galaxy S24+           |            22            |             7               |
| Samsung Galaxy Z Fold 4       |            22            |             7               |
| Xiaomi Poco F6 5G             |            22            |             6               |

## ![Demo](https://img.shields.io/badge/Demo-grey.svg?style=for-the-badge)

| <img src="assets/ChatDemo.gif" alt="Chat Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="Download iOS App" src="https://img.shields.io/badge/Try_iOS_Demo-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="Download Android App" src="https://img.shields.io/badge/Try_Android_Demo-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="assets/EmbeddingDemo.gif" alt="Embedding Demo" width="220"/> |
| --- | --- |

## ![Recommendations](https://img.shields.io/badge/Our_Recommendations-grey.svg?style=for-the-badge)
We provide a colleaction of recommended models on our [HuggingFace Page](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models)
