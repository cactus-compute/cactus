<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 翻訳

🇬🇧 [English](README.md) | 🇪🇸 [Español](README.es.md) | 🇫🇷 [Français](README.fr.md) | 🇨🇳 [中文](README.zh.md) | 🇯🇵 日本語 | 🇮🇳 [हिंदी](README.hi.md)
<br/>

アプリ内でLLM/VLM/TTSモデルをローカル展開するためのクロスプラットフォームフレームワーク。

- Flutter、React-Native、Kotlin Multiplatformで利用可能。
- Huggingfaceで見つけることができるあらゆるGGUFモデルをサポート；Qwen、Gemma、Llama、DeepSeekなど。
- LLM、VLM、埋め込みモデル、TTSモデルなどを実行。
- FP32から2ビット量子化モデルまで対応し、効率性とデバイス負荷軽減を実現。
- Jinja2サポートとトークンストリーミングを備えたチャットテンプレート。

[DISCORDに参加する！](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[リポジトリの可視化とクエリ](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **インストール：**
    プロジェクトターミナルで以下のコマンドを実行：
    ```bash
    flutter pub add cactus
    ```
2. **Flutterテキスト補完**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
    );

    final messages = [ChatMessage(role: 'user', content: 'こんにちは！')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);
    ```
3. **Flutter埋め込み**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        generateEmbeddings: true,
    );

    final text = '埋め込みするテキスト';
    final result = await lm.embedding(text);
    ```
4. **Flutter VLM補完**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.init(
        modelUrl: 'huggingface/gguf/link',
        mmprojUrl: 'huggingface/gguf/mmproj/link',
    );

    final messages = [ChatMessage(role: 'user', content: 'この画像を説明してください')];

    final response = await vlm.completion(
        messages, 
        imagePaths: ['/絶対パス/画像.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```
5. **Flutterクラウドフォールバック**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        cactusToken: 'enterprise_token_here', 
    );

    final messages = [ChatMessage(role: 'user', content: 'こんにちは！')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local（デフォルト）：厳密にデバイス上でのみ実行
    // localfirst：デバイスが失敗した場合クラウドにフォールバック
    // remotefirst：主にリモート、APIが失敗した場合ローカル実行
    // remote：厳密にクラウドで実行
    final embedding = await lm.embedding('あなたのテキスト', mode: 'localfirst');
    ```

  注：詳細は[Flutter文書](https://github.com/cactus-compute/cactus/blob/main/flutter)を参照してください。

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **`cactus-react-native`パッケージをインストール：**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **React-Nativeテキスト補完**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/パス/to/model.gguf',
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: 'こんにちは！' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```
3. **React-Native埋め込み**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/パス/to/model.gguf',
        n_ctx: 2048,
        embedding: true,
    });

    const text = '埋め込みするテキスト';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

4. **React-Native VLM**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/パス/to/vision-model.gguf',
        mmproj: '/パス/to/mmproj.gguf',
    });

    const messages = [{ role: 'user', content: 'この画像を説明してください' }];

    const params = {
        images: ['/絶対パス/画像.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
5. **React-Nativeクラウドフォールバック**
    ```typescript
    import { CactusLM } from 'cactus-react-native';

    const { lm, error } = await CactusLM.init({
        model: '/パス/to/model.gguf',
        n_ctx: 2048,
    }, undefined, 'enterprise_token_here');

    const messages = [{ role: 'user', content: 'こんにちは！' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);

    // local（デフォルト）：厳密にデバイス上でのみ実行
    // localfirst：デバイスが失敗した場合クラウドにフォールバック
    // remotefirst：主にリモート、APIが失敗した場合ローカル実行
    // remote：厳密にクラウドで実行
    const embedding = await lm.embedding('あなたのテキスト', undefined, 'localfirst');
    ```
注：詳細は[React文書](https://github.com/cactus-compute/cactus/blob/main/react)を参照してください。

## ![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **Maven依存関係を追加：**
    KMPプロジェクトの`build.gradle.kts`に追加：
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

2. **プラットフォーム設定：**
    - **Android：** 自動的に動作 - ネイティブライブラリが含まれています。
    - **iOS：** Xcodeで：File → Add Package Dependencies → `https://github.com/cactus-compute/cactus`を貼り付け → Addをクリック

3. **Kotlin Multiplatformテキスト補完**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // フルGPUオフロードの場合は99に設定
        )
        
        val downloadSuccess = lm.download(
            url = "パス/to/hugginface/gguf",
            filename = "model_filename.gguf"
        )
        val initSuccess = lm.init("qwen3-600m.gguf")
        
        val result = lm.completion(
            prompt = "こんにちは！",
            maxTokens = 100,
            temperature = 0.7f
        )
    }
    ```

4. **Kotlin Multiplatform音声認識**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val stt = CactusSTT(
            language = "ja-JP",
            sampleRate = 16000,
            maxDuration = 30
        )
        
        // AndroidのデフォルトVosk STTモデルとApple Foundation Modelのみサポート
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()
        
        val result = stt.transcribe()
        result?.let { sttResult ->
            println("転写: ${sttResult.text}")
            println("信頼度: ${sttResult.confidence}")
        }
        
        // または音声ファイルから転写
        val fileResult = stt.transcribeFile("/パス/to/audio.wav")
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
            gpuLayers = 0 // フルGPUオフロードの場合は99に設定
        )
        
        val downloadSuccess = vlm.download(
            modelUrl = "パス/to/hugginface/gguf",
            mmprojUrl = "パス/to/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")
        
        val result = vlm.completion(
            prompt = "この画像を説明してください",
            imagePath = "/パス/to/画像.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

  注：詳細は[Kotlin文書](https://github.com/cactus-compute/cactus/blob/main/kotlin)を参照してください。

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

CactusバックエンドはC/C++で書かれており、携帯電話、スマートTV、時計、スピーカー、カメラ、ラップトップなどで直接実行できます。詳細は[C++文書](https://github.com/cactus-compute/cactus/blob/main/cpp)を参照してください。


## ![このリポジトリとサンプルアプリの使用](https://img.shields.io/badge/リポジトリとサンプルの使用-grey.svg?style=for-the-badge)

まず、`git clone https://github.com/cactus-compute/cactus.git`でリポジトリをクローンし、その中に移動して`chmod +x scripts/*.sh`ですべてのスクリプトを実行可能にします

1. **Flutter**
    - `scripts/build-flutter-android.sh`でAndroid JNILibsをビルド。
    - `scripts/build-flutter.sh`でFlutterプラグインをビルド。（サンプル使用前に必須）
    - `cd flutter/example`でサンプルアプリに移動。
    - XcodeまたはAndroid Studioでシミュレータを開く、初回の場合は[ウォークスルー](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)を参照。
    - 常にこの組み合わせでアプリを開始`flutter clean && flutter pub get && flutter run`。
    - アプリで遊び、必要に応じてサンプルアプリまたはプラグインに変更を加えます。

2. **React Native**
    - `scripts/build-react-android.sh`でAndroid JNILibsをビルド。
    - `scripts/build-react.sh`でFlutterプラグインをビルド。
    - `cd react/example`でサンプルアプリに移動。
    - XcodeまたはAndroid Studioでシミュレータを設定、初回の場合は[ウォークスルー](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)を参照。
    - 常にこの組み合わせでアプリを開始`yarn && yarn ios`または`yarn && yarn android`。
    - アプリで遊び、必要に応じてサンプルアプリまたはパッケージに変更を加えます。
    - 現在、パッケージに変更が加えられた場合、手動で`examples/react/node_modules/cactus-react-native`にファイル/フォルダをコピーします。

3. **Kotlin Multiplatform**
    - `scripts/build-flutter-android.sh`でAndroid JNILibsをビルド。（FlutterとKotlinは同じJNILibsを共有）
    - `scripts/build-kotlin.sh`でKotlinライブラリをビルド。（サンプル使用前に必須）
    - `cd kotlin/example`でサンプルアプリに移動。
    - XcodeまたはAndroid Studioでシミュレータを開く、初回の場合は[ウォークスルー](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)を参照。
    - デスクトップの場合は常に`./gradlew :composeApp:run`でアプリを開始、モバイルの場合はAndroid Studio/Xcodeを使用。
    - アプリで遊び、必要に応じてサンプルアプリまたはライブラリに変更を加えます。

4. **C/C++**
    - `cd cactus/example`でサンプルアプリに移動。
    - 複数のメインファイル`main_vlm, main_llm, main_embed, main_tts`があります。
    - `build.sh`を使用してライブラリと実行ファイルの両方をビルド。
    - 実行ファイルの一つで実行`./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`。
    - 異なるモデルを試し、必要に応じて変更を加えます。

5. **貢献**
    - バグ修正を貢献するには、変更後に`git checkout -b <ブランチ名>`でブランチを作成しPRを提出。
    - 機能を貢献するには、他の人との重複を避けるため、まず問題を提起して議論してください。
    - [discordに参加](https://discord.gg/SdZjmfWQ)

## ![パフォーマンス](https://img.shields.io/badge/パフォーマンス-grey.svg?style=for-the-badge)

| デバイス                       |  Gemma3 1B Q4 (トークン/秒) |    Qwen3 4B Q4 (トークン/秒)   |  
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

## ![デモ](https://img.shields.io/badge/デモ-grey.svg?style=for-the-badge)

| <img src="assets/ChatDemo.gif" alt="Chat Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="iOSアプリダウンロード" src="https://img.shields.io/badge/iOSデモを試す-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="Androidアプリダウンロード" src="https://img.shields.io/badge/Androidデモを試す-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="assets/EmbeddingDemo.gif" alt="Embedding Demo" width="220"/> |
| --- | --- |

## ![推奨](https://img.shields.io/badge/私たちの推奨-grey.svg?style=for-the-badge)
[HuggingFaceページ](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models)で推奨モデルのコレクションを提供しています
