<img src="../assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 번역

🇬🇧 [English](../README.md) | 🇪🇸 [Español](README.es.md) | 🇫🇷 [Français](README.fr.md) | 🇨🇳 [中文](README.zh.md) | 🇯🇵 [日本語](README.ja.md) | 🇮🇳 [हिंदी](README.hi.md) | 🇩🇪 [Deutsch](README.de.md) | 🇰🇷 한국어
<br/>

앱 내에서 LLM/VLM/TTS 모델을 로컬로 배포하기 위한 크로스플랫폼 프레임워크입니다.

- Flutter, React-Native, Kotlin Multiplatform에서 사용 가능합니다.
- Huggingface에서 찾을 수 있는 모든 GGUF 모델을 지원합니다: Qwen, Gemma, Llama, DeepSeek 등.
- LLM, VLM, 임베딩 모델, TTS 모델 등을 실행합니다.
- 효율성과 디바이스 부하 절감을 위해 FP32부터 2비트 양자화 모델까지 지원합니다.
- Jinja2 지원과 토큰 스트리밍을 제공하는 채팅 템플릿.

[디스코드에 참여하세요!](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[저장소 시각화 및 쿼리](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **설치:**
    프로젝트 터미널에서 다음 명령을 실행하세요:
    ```bash
    flutter pub add cactus
    ```
2. **Flutter 텍스트 생성**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
    );

    final messages = [ChatMessage(role: 'user', content: '안녕하세요!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);
    ```
3. **Flutter 임베딩**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
        generateEmbeddings: true,
    );

    final text = '임베딩할 텍스트';
    final result = await lm.embedding(text);
    ```
4. **Flutter VLM 생성**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
        mmprojUrl: 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
    );

    final messages = [ChatMessage(role: 'user', content: '이 이미지를 설명해주세요')];

    final response = await vlm.completion(
        messages, 
        imagePaths: ['/absolute/path/to/image.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```
5. **Flutter 클라우드 폴백**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
        cactusToken: 'enterprise_token_here', 
    );

    final messages = [ChatMessage(role: 'user', content: '안녕하세요!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local (기본값): 엄격하게 디바이스에서만 실행
    // localfirst: 디바이스 실패 시 클라우드로 폴백
    // remotefirst: 주로 원격 실행, API 실패 시 로컬 실행
    // remote: 엄격하게 클라우드에서만 실행
    final embedding = await lm.embedding('텍스트', mode: 'localfirst');
    ```

  참고: 자세한 내용은 [Flutter 문서](https://github.com/cactus-compute/cactus/blob/main/flutter)를 참조하세요.

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **`cactus-react-native` 패키지 설치:**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **React-Native 텍스트 생성**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/path/to/model.gguf', // 앱 샌드박스 내 로컬 모델 파일
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: '안녕하세요!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```
3. **React-Native 임베딩**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/path/to/model.gguf', // 앱 샌드박스 내 로컬 모델 파일
        n_ctx: 2048,
        embedding: true,
    });

    const text = '임베딩할 텍스트';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

4. **React-Native VLM**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/path/to/vision-model.gguf', // 앱 샌드박스 내 로컬 모델 파일
        mmproj: '/path/to/mmproj.gguf', // 앱 샌드박스 내 로컬 모델 파일
    });

    const messages = [{ role: 'user', content: '이 이미지를 설명해주세요' }];

    const params = {
        images: ['/absolute/path/to/image.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
5. **React-Native 에이전트**
    
    ```typescript
    import { CactusAgent } from 'cactus-react-native';

    // Qwen 3 제품군을 권장하며, 0.6B가 좋습니다
    const { agent, error } = await CactusAgent.init({
        model: '/path/to/model.gguf', 
        n_ctx: 2048,
    });

    const weatherTool = agent.addTool(
        (location: string) => `${location}의 날씨: 22°C, 맑음`,
        '지역의 현재 날씨 정보 제공',
        {
            location: { type: 'string', description: '도시 이름', required: true }
        }
    );

    const messages = [{ role: 'user', content: '서울의 날씨는 어때?' }];
      const result = await agent.completionWithTools(messages, {
      n_predict: 200,
      temperature: 0.7,
    });

    await agent.release();
    ```

`CactusAgent`를 사용하여 만든 [예제 앱](https://github.com/cactus-compute/example-react-agents/)으로 시작해보세요.

자세한 내용은 [React 문서](https://github.com/cactus-compute/cactus/blob/main/react)를 참조하세요.

## ![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **Maven 의존성 추가:**
    KMP 프로젝트의 `build.gradle.kts`에 추가:
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

2. **플랫폼 설정:**
    - **Android:** 자동으로 작동 - 네이티브 라이브러리 포함됨.
    - **iOS:** Xcode에서: File → Add Package Dependencies → `https://github.com/cactus-compute/cactus` 붙여넣기 → Add 클릭

3. **Kotlin Multiplatform 텍스트 생성**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // 전체 GPU 오프로드는 99로 설정
        )
        
        val downloadSuccess = lm.download(
            url = "path/to/hugginface/gguf",
            filename = "model_filename.gguf"
        )
        val initSuccess = lm.init("qwen3-600m.gguf")
        
        val result = lm.completion(
            prompt = "안녕하세요!",
            maxTokens = 100,
            temperature = 0.7f
        )
    }
    ```

4. **Kotlin Multiplatform 음성 인식**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val stt = CactusSTT(
            language = "ko-KR",
            sampleRate = 16000,
            maxDuration = 30
        )
        
        // Android는 기본 Vosk STT 모델, Apple은 Foundation 모델만 지원
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()
        
        val result = stt.transcribe()
        result?.let { sttResult ->
            println("변환된 텍스트: ${sttResult.text}")
            println("신뢰도: ${sttResult.confidence}")
        }
        
        // 또는 오디오 파일에서 변환
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
            gpuLayers = 0 // 전체 GPU 오프로드는 99로 설정
        )
        
        val downloadSuccess = vlm.download(
            modelUrl = "path/to/hugginface/gguf",
            mmprojUrl = "path/to/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")
        
        val result = vlm.completion(
            prompt = "이 이미지를 설명해주세요",
            imagePath = "/path/to/image.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

  참고: 자세한 내용은 [Kotlin 문서](https://github.com/cactus-compute/cactus/blob/main/kotlin)를 참조하세요.

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus 백엔드는 C/C++로 작성되어 휴대폰, 스마트 TV, 시계, 스피커, 카메라, 노트북 등에서 직접 실행할 수 있습니다. 자세한 내용은 [C++ 문서](https://github.com/cactus-compute/cactus/blob/main/cpp)를 참조하세요.


## ![저장소 및 예제 앱 사용하기](https://img.shields.io/badge/저장소_및_예제_사용-grey.svg?style=for-the-badge)

먼저 `git clone https://github.com/cactus-compute/cactus.git`으로 저장소를 복제하고, 해당 디렉토리로 이동한 후 `chmod +x scripts/*.sh`로 모든 스크립트를 실행 가능하게 만드세요.

1. **Flutter**
    - `scripts/build-flutter-android.sh`로 Android JNILibs를 빌드합니다.
    - `scripts/build-flutter.sh`로 Flutter 플러그인을 빌드합니다. (예제 사용 전 반드시 실행)
    - `cd flutter/example`로 예제 앱으로 이동합니다.
    - Xcode 또는 Android Studio를 통해 시뮬레이터를 엽니다. 처음이시라면 [가이드](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)를 참조하세요.
    - 항상 `flutter clean && flutter pub get && flutter run` 조합으로 앱을 시작합니다.
    - 앱을 실행하고 원하는 대로 예제 앱이나 플러그인을 수정해보세요.

2. **React Native**
    - `scripts/build-react-android.sh`로 Android JNILibs를 빌드합니다.
    - `scripts/build-react.sh`로 React Native 패키지를 빌드합니다.
    - `cd react/example`로 예제 앱으로 이동합니다.
    - Xcode 또는 Android Studio를 통해 시뮬레이터를 설정합니다. 처음이시라면 [가이드](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)를 참조하세요.
    - 항상 `yarn && yarn ios` 또는 `yarn && yarn android` 조합으로 앱을 시작합니다.
    - 앱을 실행하고 원하는 대로 예제 앱이나 패키지를 수정해보세요.
    - 현재 패키지를 수정한 경우, 수동으로 파일/폴더를 `examples/react/node_modules/cactus-react-native`에 복사해야 합니다.

3. **Kotlin Multiplatform**
    - `scripts/build-flutter-android.sh`로 Android JNILibs를 빌드합니다. (Flutter와 Kotlin이 동일한 JNILibs 공유)
    - `scripts/build-kotlin.sh`로 Kotlin 라이브러리를 빌드합니다. (예제 사용 전 반드시 실행)
    - `cd kotlin/example`로 예제 앱으로 이동합니다.
    - Xcode 또는 Android Studio를 통해 시뮬레이터를 엽니다. 처음이시라면 [가이드](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)를 참조하세요.
    - 데스크톱은 `./gradlew :composeApp:run`으로, 모바일은 Android Studio/Xcode를 사용하여 앱을 시작합니다.
    - 앱을 실행하고 원하는 대로 예제 앱이나 라이브러리를 수정해보세요.

4. **C/C++**
    - `cd cpp/example`로 예제 앱으로 이동합니다.
    - 여러 메인 파일이 있습니다: `main_vlm, main_llm, main_embed, main_tts`.
    - `build.sh`를 사용하여 라이브러리와 실행 파일을 모두 빌드합니다.
    - 실행 파일 중 하나를 실행합니다: `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`.
    - 다양한 모델을 시도하고 원하는 대로 수정해보세요.

5. **기여하기**
    - 버그 수정을 기여하려면, 변경 후 `git checkout -b <branch-name>`으로 브랜치를 생성하고 PR을 제출하세요.
    - 기능을 기여하려면, 다른 사람과 중복되지 않도록 먼저 이슈로 제기하여 논의해주세요.
    - [디스코드에 참여하세요](https://discord.gg/bNurx3AXTJ)

## ![성능](https://img.shields.io/badge/성능-grey.svg?style=for-the-badge)

| 디바이스                       |  Gemma3 1B Q4 (토큰/초) |    Qwen3 4B Q4 (토큰/초)    |  
|:------------------------------|:----------------------:|:-------------------------:|
| iPhone 16 Pro Max             |           54           |            18             |
| iPhone 16 Pro                 |           54           |            18             |
| iPhone 16                     |           49           |            16             |
| iPhone 15 Pro Max             |           45           |            15             |
| iPhone 15 Pro                 |           45           |            15             |
| iPhone 14 Pro Max             |           44           |            14             |
| OnePlus 13 5G                 |           43           |            14             |
| Samsung Galaxy S24 Ultra      |           42           |            14             |
| iPhone 15                     |           42           |            14             |
| OnePlus Open                  |           38           |            13             |
| Samsung Galaxy S23 5G         |           37           |            12             |
| Samsung Galaxy S24            |           36           |            12             |
| iPhone 13 Pro                 |           35           |            11             |
| OnePlus 12                    |           35           |            11             |
| Galaxy S25 Ultra              |           29           |             9             |
| OnePlus 11                    |           26           |             8             |
| iPhone 13 mini                |           25           |             8             |
| Redmi K70 Ultra               |           24           |             8             |
| Xiaomi 13                     |           24           |             8             |
| Samsung Galaxy S24+           |           22           |             7             |
| Samsung Galaxy Z Fold 4       |           22           |             7             |
| Xiaomi Poco F6 5G             |           22           |             6             |

## ![데모](https://img.shields.io/badge/데모-grey.svg?style=for-the-badge)

| <img src="../assets/ChatDemo.gif" alt="Chat Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="iOS 앱 다운로드" src="https://img.shields.io/badge/iOS_데모_체험-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="Android 앱 다운로드" src="https://img.shields.io/badge/Android_데모_체험-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="../assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="../assets/EmbeddingDemo.gif" alt="Embedding Demo" width="220"/> |
| --- | --- |

## ![추천 모델](https://img.shields.io/badge/추천_모델-grey.svg?style=for-the-badge)
[HuggingFace 페이지](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models)에서 추천 모델 컬렉션을 제공합니다.