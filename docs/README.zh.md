<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 翻译

🇬🇧 [English](../README.md) | 🇪🇸 [Español](README.es.md) | 🇫🇷 [Français](README.fr.md) | 🇨🇳 中文 | 🇯🇵 [日本語](README.ja.md) | 🇮🇳 [हिंदी](README.hi.md)
<br/>

在您的应用中本地部署LLM/VLM/TTS模型的跨平台框架。

- 支持Flutter、React-Native和Kotlin多平台。
- 支持您在Huggingface上找到的任何GGUF模型；Qwen、Gemma、Llama、DeepSeek等。
- 运行LLM、VLM、嵌入模型、TTS模型等。
- 支持从FP32到低至2位量化模型，提高效率并减少设备负担。
- 支持Jinja2的聊天模板和令牌流。

[点击加入我们的DISCORD！](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[点击可视化和查询仓库](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **安装：**
    在您的项目终端中执行以下命令：
    ```bash
    flutter pub add cactus
    ```
2. **Flutter文本补全**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
    );

    final messages = [ChatMessage(role: 'user', content: '你好！')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);
    ```
3. **Flutter嵌入**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        generateEmbeddings: true,
    );

    final text = '您要嵌入的文本';
    final result = await lm.embedding(text);
    ```
4. **Flutter VLM补全**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.init(
        modelUrl: 'huggingface/gguf/link',
        mmprojUrl: 'huggingface/gguf/mmproj/link',
    );

    final messages = [ChatMessage(role: 'user', content: '描述这张图片')];

    final response = await vlm.completion(
        messages, 
        imagePaths: ['/绝对路径/到/图片.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```
5. **Flutter云端回退**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        cactusToken: 'enterprise_token_here', 
    );

    final messages = [ChatMessage(role: 'user', content: '你好！')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local（默认）：严格仅在设备上运行
    // localfirst：如果设备失败则回退到云端
    // remotefirst：主要远程，如果API失败则运行本地
    // remote：严格在云端运行
    final embedding = await lm.embedding('您的文本', mode: 'localfirst');
    ```

  注：更多信息请参见[Flutter文档](https://github.com/cactus-compute/cactus/blob/main/flutter)。

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **安装`cactus-react-native`包：**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **React-Native文本补全**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/路径/到/model.gguf',
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: '你好！' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```
3. **React-Native嵌入**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/路径/到/model.gguf',
        n_ctx: 2048,
        embedding: true,
    });

    const text = '您要嵌入的文本';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

4. **React-Native VLM**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/路径/到/vision-model.gguf',
        mmproj: '/路径/到/mmproj.gguf',
    });

    const messages = [{ role: 'user', content: '描述这张图片' }];

    const params = {
        images: ['/绝对路径/到/图片.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
5. **React-Native云端回退**
    ```typescript
    import { CactusLM } from 'cactus-react-native';

    const { lm, error } = await CactusLM.init({
        model: '/路径/到/model.gguf',
        n_ctx: 2048,
    }, undefined, 'enterprise_token_here');

    const messages = [{ role: 'user', content: '你好！' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);

    // local（默认）：严格仅在设备上运行
    // localfirst：如果设备失败则回退到云端
    // remotefirst：主要远程，如果API失败则运行本地
    // remote：严格在云端运行
    const embedding = await lm.embedding('您的文本', undefined, 'localfirst');
    ```
注：更多信息请参见[React文档](https://github.com/cactus-compute/cactus/blob/main/react)。

## ![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **添加Maven依赖：**
    添加到您的KMP项目的`build.gradle.kts`：
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

2. **平台设置：**
    - **Android：** 自动工作 - 包含原生库。
    - **iOS：** 在Xcode中：File → Add Package Dependencies → 粘贴`https://github.com/cactus-compute/cactus` → 点击Add

3. **Kotlin多平台文本补全**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // 设置为99以进行完全GPU卸载
        )
        
        val downloadSuccess = lm.download(
            url = "路径/到/hugginface/gguf",
            filename = "model_filename.gguf"
        )
        val initSuccess = lm.init("qwen3-600m.gguf")
        
        val result = lm.completion(
            prompt = "你好！",
            maxTokens = 100,
            temperature = 0.7f
        )
    }
    ```

4. **Kotlin多平台语音转文本**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val stt = CactusSTT(
            language = "zh-CN",
            sampleRate = 16000,
            maxDuration = 30
        )
        
        // 仅支持Android的默认Vosk STT模型和Apple Foundation模型
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()
        
        val result = stt.transcribe()
        result?.let { sttResult ->
            println("转录：${sttResult.text}")
            println("置信度：${sttResult.confidence}")
        }
        
        // 或从音频文件转录
        val fileResult = stt.transcribeFile("/路径/到/audio.wav")
    }
    ```

5. **Kotlin多平台VLM**
    ```kotlin
    import com.cactus.CactusVLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val vlm = CactusVLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // 设置为99以进行完全GPU卸载
        )
        
        val downloadSuccess = vlm.download(
            modelUrl = "路径/到/hugginface/gguf",
            mmprojUrl = "路径/到/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")
        
        val result = vlm.completion(
            prompt = "描述这张图片",
            imagePath = "/路径/到/图片.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

  注：更多信息请参见[Kotlin文档](https://github.com/cactus-compute/cactus/blob/main/kotlin)。

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus后端用C/C++编写，可以直接在手机、智能电视、手表、扬声器、摄像头、笔记本电脑等设备上运行。更多信息请参见[C++文档](https://github.com/cactus-compute/cactus/blob/main/cpp)。


## ![使用此仓库和示例应用](https://img.shields.io/badge/使用仓库和示例-grey.svg?style=for-the-badge)

首先，使用`git clone https://github.com/cactus-compute/cactus.git`克隆仓库，进入其中并使用`chmod +x scripts/*.sh`使所有脚本可执行

1. **Flutter**
    - 使用`scripts/build-flutter-android.sh`构建Android JNILibs。
    - 使用`scripts/build-flutter.sh`构建Flutter插件。（在使用示例前必须运行）
    - 使用`cd flutter/example`导航到示例应用。
    - 通过Xcode或Android Studio打开您的模拟器，如果您之前没有这样做过，请参考[演练](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)。
    - 始终使用此组合启动应用`flutter clean && flutter pub get && flutter run`。
    - 玩转应用，根据需要对示例应用或插件进行更改。

2. **React Native**
    - 使用`scripts/build-react-android.sh`构建Android JNILibs。
    - 使用`scripts/build-react.sh`构建Flutter插件。
    - 使用`cd react/example`导航到示例应用。
    - 通过Xcode或Android Studio设置您的模拟器，如果您之前没有这样做过，请参考[演练](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)。
    - 始终使用此组合启动应用`yarn && yarn ios`或`yarn && yarn android`。
    - 玩转应用，根据需要对示例应用或包进行更改。
    - 目前，如果在包中进行了更改，您需要手动将文件/文件夹复制到`examples/react/node_modules/cactus-react-native`。

3. **Kotlin多平台**
    - 使用`scripts/build-flutter-android.sh`构建Android JNILibs。（Flutter和Kotlin共享相同的JNILibs）
    - 使用`scripts/build-kotlin.sh`构建Kotlin库。（在使用示例前必须运行）
    - 使用`cd kotlin/example`导航到示例应用。
    - 通过Xcode或Android Studio打开您的模拟器，如果您之前没有这样做过，请参考[演练](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)。
    - 对于桌面，始终使用`./gradlew :composeApp:run`启动应用，或对于移动设备使用Android Studio/Xcode。
    - 玩转应用，根据需要对示例应用或库进行更改。

4. **C/C++**
    - 使用`cd cactus/example`导航到示例应用。
    - 有多个主文件`main_vlm, main_llm, main_embed, main_tts`。
    - 使用`build.sh`构建库和可执行文件。
    - 使用可执行文件之一运行`./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`。
    - 尝试不同的模型并根据需要进行更改。

5. **贡献**
    - 要贡献错误修复，在进行更改后使用`git checkout -b <分支名>`创建分支并提交PR。
    - 要贡献功能，请先提出问题以便讨论，避免与他人冲突。
    - [加入我们的discord](https://discord.gg/SdZjmfWQ)

## ![性能](https://img.shields.io/badge/性能-grey.svg?style=for-the-badge)

| 设备                          |  Gemma3 1B Q4 (tokens/秒) |    Qwen3 4B Q4 (tokens/秒)   |  
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

## ![演示](https://img.shields.io/badge/演示-grey.svg?style=for-the-badge)

| <img src="assets/ChatDemo.gif" alt="Chat Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="下载iOS应用" src="https://img.shields.io/badge/试用iOS演示-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="下载Android应用" src="https://img.shields.io/badge/试用Android演示-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="assets/EmbeddingDemo.gif" alt="Embedding Demo" width="220"/> |
| --- | --- |

## ![推荐](https://img.shields.io/badge/我们的推荐-grey.svg?style=for-the-badge)
我们在[HuggingFace页面](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models)上提供推荐模型的集合
