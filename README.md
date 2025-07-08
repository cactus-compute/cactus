<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

<br/>

The fastest cross-platform framework for deploying AI locally on phones. 

- Available in Flutter and React-Native for cross-platform developers.
- Supports any GGUF model you can find on Huggingface; Qwen, Gemma, Llama, DeepSeek etc.
- Run LLMs, VLMs, Embedding Models, TTS models and more.
- Accommodates from FP32 to as low as 2-bit quantized models, for efficiency and less device strain. 
- MCP tool-calls to make AI performant and helpful (set reminder, gallery search, reply messages) etc.
- iOS xcframework and JNILibs for native setups.
- Neat and tiny C++ build for custom hardware.
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

    // Initialize
    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
    );

    // Completion 
    final messages = [CactusMessage(role: CactusMessageRole.user, content: 'Hello!')];
    final params = CactusCompletionParams(nPredict: 100, temperature: 0.7);
    final response = await lm.completion(messages, params);

    // Embedding 
    final text = 'Your text to embed';
    final params = CactusEmbeddingParams(normalize: true);
    final result = await lm.embedding(text, params);
    ```
3. **Flutter VLM Completion**
    ```dart
    import 'package:cactus/cactus.dart';

    // Initialize (Flutter handles downloads automatically)
    final vlm = await CactusVLM.init(
        modelUrl: 'huggingface/gguf/link',
        mmprojUrl: 'huggingface/gguf/mmproj/link',
    );

    final messages = [CactusMessage(role: CactusMessageRole.user, content: 'Describe this image')];

    final params = CactusVLMParams(
        images: ['/absolute/path/to/image.jpg'],
        nPredict: 200,
        temperature: 0.3,
    );

    final response = await vlm.completion(messages, params);
    ```
  N/B: See the [Flutter Docs](https://github.com/cactus-compute/cactus/blob/main/flutter). It covers chat design, embeddings, multimodal models, text-to-speech, and more.

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **Install the `cactus-react-native` package:**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **React-Native Text Completion**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/path/to/model.gguf',
        n_ctx: 2048,
    });

    // Completion 
    const messages = [{ role: 'user', content: 'Hello!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);

    // Embedding 
    const text = 'Your text to embed';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

3. **React-Native VLM**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/path/to/vision-model.gguf',
        mmproj: '/path/to/mmproj.gguf',
    });

    const messages = [{ role: 'user', content: 'Describe this image' }];

    const params = {
        images: ['/absolute/path/to/image.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
N/B: See the [React Docs](https://github.com/cactus-compute/cactus/blob/main/react). It covers chat design, embeddings, multimodal models, text-to-speech, and various options.

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus backend is written in C/C++ and can run directly on any ARM/X86/Raspberry PI hardware like phones, smart tvs, watches, speakers, cameras, laptops etc. See the [C++ Docs](https://github.com/cactus-compute/cactus/blob/main/cactus). It covers chat design, embeddings, multimodal models, text-to-speech, and more.


## ![Using this Repo & Example Apps](https://img.shields.io/badge/Using_Repo_And_Examples-grey.svg?style=for-the-badge)

First, clone the repo with `git clone https://github.com/cactus-compute/cactus.git`, cd into it and make all scripts executable with `chmod +x scripts/*.sh`

1. **Flutter**
    - Build the Android JNILibs with `scripts/build-flutter-android.sh`.
    - Build the Flutter Plugin with `scripts/build-flutter-android.sh`.
    - Navigate to the example app with `cd flutter/example`.
    - Open your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
    - Always start app with this combo `flutter clean && flutter pub get && flutter run`.
    - Play with the app, and make changes either to the example app or plugin as desired.

2. **React Native**
    - Build the Android JNILibs with `scripts/build-react-android.sh`.
    - Build the Flutter Plugin with `scripts/build-react-android.sh`.
    - Navigate to the example app with `cd react/example`.
    - Setup your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
    - Always start app with this combo `yarn && yarn ios` or `yarn && yarn android`.
    - Play with the app, and make changes either to the example app or package as desired.
    - For now, if changes are made in the package, you would manually copy the files/folders into the `examples/react/node_modules/cactus-react-native`.

2. **C/C++**
    - Navigate to the example app with `cd cactus/example`.
    - There are multiple main files `main_vlm, main_llm, main_embed, main_tts`.
    - Build both the libraries and executable using `build.sh`.
    - Run with one of the executables `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`.
    - Try different models and make changes as desired.

4. **Contributing**
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

We created a demo chat app we use for benchmarking:

[![Download App](https://img.shields.io/badge/Download_iOS_App-grey?style=for-the-badge&logo=apple&logoColor=white)](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
[![Download App](https://img.shields.io/badge/Download_Android_App-grey?style=for-the-badge&logo=android&logoColor=white)](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share)

## ![Recommendations](https://img.shields.io/badge/Our_Recommendations-grey.svg?style=for-the-badge)
We provide a colleaction of recommended models on our [HuggingFace Page](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models)
