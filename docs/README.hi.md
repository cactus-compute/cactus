<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 अनुवाद

🇬🇧 [English](../README.md) | 🇪🇸 [Español](README.es.md) | 🇫🇷 [Français](README.fr.md) | 🇨🇳 [中文](README.zh.md) | 🇯🇵 [日本語](README.ja.md) | 🇮🇳 हिंदी
<br/>

आपके ऐप में LLM/VLM/TTS मॉडल्स को स्थानीय रूप से तैनात करने के लिए क्रॉस-प्लेटफॉर्म फ्रेमवर्क।

- Flutter, React-Native और Kotlin Multiplatform में उपलब्ध।
- Huggingface पर मिलने वाले किसी भी GGUF मॉडल का समर्थन करता है; Qwen, Gemma, Llama, DeepSeek आदि।
- LLM, VLM, Embedding मॉडल, TTS मॉडल और अधिक चलाएं।
- FP32 से लेकर 2-बिट क्वांटाइज़्ड मॉडल तक का समर्थन, दक्षता और कम डिवाइस तनाव के लिए।
- Jinja2 समर्थन और टोकन स्ट्रीमिंग के साथ चैट टेम्प्लेट।

[हमारे DISCORD में शामिल होने के लिए क्लिक करें!](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[रेपो को विज़ुअलाइज़ और क्वेरी करने के लिए क्लिक करें](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **इंस्टॉल करें:**
    अपने प्रोजेक्ट टर्मिनल में निम्नलिखित कमांड चलाएं:
    ```bash
    flutter pub add cactus
    ```
2. **Flutter टेक्स्ट कंप्लीशन**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
    );

    final messages = [ChatMessage(role: 'user', content: 'नमस्ते!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);
    ```
3. **Flutter एम्बेडिंग**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        generateEmbeddings: true,
    );

    final text = 'एम्बेड करने के लिए आपका टेक्स्ट';
    final result = await lm.embedding(text);
    ```
4. **Flutter VLM कंप्लीशन**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.init(
        modelUrl: 'huggingface/gguf/link',
        mmprojUrl: 'huggingface/gguf/mmproj/link',
    );

    final messages = [ChatMessage(role: 'user', content: 'इस छवि का वर्णन करें')];

    final response = await vlm.completion(
        messages, 
        imagePaths: ['/निरपेक्ष/पथ/छवि.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```
5. **Flutter क्लाउड फॉलबैक**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        cactusToken: 'enterprise_token_here', 
    );

    final messages = [ChatMessage(role: 'user', content: 'नमस्ते!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local (डिफ़ॉल्ट): केवल डिवाइस पर चलाएं
    // localfirst: डिवाइस विफल होने पर क्लाउड पर फॉलबैक
    // remotefirst: मुख्यतः रिमोट, API विफल होने पर स्थानीय चलाएं
    // remote: केवल क्लाउड पर चलाएं
    final embedding = await lm.embedding('आपका टेक्स्ट', mode: 'localfirst');
    ```

  नोट: अधिक जानकारी के लिए [Flutter डॉक्स](https://github.com/cactus-compute/cactus/blob/main/flutter) देखें।

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **`cactus-react-native` पैकेज इंस्टॉल करें:**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **React-Native टेक्स्ट कंप्लीशन**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/पथ/to/model.gguf',
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: 'नमस्ते!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```
3. **React-Native एम्बेडिंग**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/पथ/to/model.gguf',
        n_ctx: 2048,
        embedding: true,
    });

    const text = 'एम्बेड करने के लिए आपका टेक्स्ट';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

4. **React-Native VLM**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/पथ/to/vision-model.gguf',
        mmproj: '/पथ/to/mmproj.gguf',
    });

    const messages = [{ role: 'user', content: 'इस छवि का वर्णन करें' }];

    const params = {
        images: ['/निरपेक्ष/पथ/छवि.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
5. **React-Native क्लाउड फॉलबैक**
    ```typescript
    import { CactusLM } from 'cactus-react-native';

    const { lm, error } = await CactusLM.init({
        model: '/पथ/to/model.gguf',
        n_ctx: 2048,
    }, undefined, 'enterprise_token_here');

    const messages = [{ role: 'user', content: 'नमस्ते!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);

    // local (डिफ़ॉल्ट): केवल डिवाइस पर चलाएं
    // localfirst: डिवाइस विफल होने पर क्लाउड पर फॉलबैक
    // remotefirst: मुख्यतः रिमोट, API विफल होने पर स्थानीय चलाएं
    // remote: केवल क्लाउड पर चलाएं
    const embedding = await lm.embedding('आपका टेक्स्ट', undefined, 'localfirst');
    ```
नोट: अधिक जानकारी के लिए [React डॉक्स](https://github.com/cactus-compute/cactus/blob/main/react) देखें।

## ![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **Maven डिपेंडेंसी जोड़ें:**
    अपने KMP प्रोजेक्ट के `build.gradle.kts` में जोड़ें:
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

2. **प्लेटफॉर्म सेटअप:**
    - **Android:** स्वचालित रूप से काम करता है - नेटिव लाइब्रेरी शामिल।
    - **iOS:** Xcode में: File → Add Package Dependencies → `https://github.com/cactus-compute/cactus` पेस्ट करें → Add क्लिक करें

3. **Kotlin Multiplatform टेक्स्ट कंप्लीशन**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // पूर्ण GPU ऑफलोड के लिए 99 सेट करें
        )
        
        val downloadSuccess = lm.download(
            url = "पथ/to/hugginface/gguf",
            filename = "model_filename.gguf"
        )
        val initSuccess = lm.init("qwen3-600m.gguf")
        
        val result = lm.completion(
            prompt = "नमस्ते!",
            maxTokens = 100,
            temperature = 0.7f
        )
    }
    ```

4. **Kotlin Multiplatform स्पीच टू टेक्स्ट**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val stt = CactusSTT(
            language = "hi-IN",
            sampleRate = 16000,
            maxDuration = 30
        )
        
        // केवल Android के लिए डिफ़ॉल्ट Vosk STT मॉडल और Apple Foundation मॉडल का समर्थन
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()
        
        val result = stt.transcribe()
        result?.let { sttResult ->
            println("ट्रांस्क्राइब्ड: ${sttResult.text}")
            println("विश्वसनीयता: ${sttResult.confidence}")
        }
        
        // या ऑडियो फाइल से ट्रांस्क्राइब करें
        val fileResult = stt.transcribeFile("/पथ/to/audio.wav")
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
            gpuLayers = 0 // पूर्ण GPU ऑफलोड के लिए 99 सेट करें
        )
        
        val downloadSuccess = vlm.download(
            modelUrl = "पथ/to/hugginface/gguf",
            mmprojUrl = "पथ/to/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")
        
        val result = vlm.completion(
            prompt = "इस छवि का वर्णन करें",
            imagePath = "/पथ/to/छवि.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

  नोट: अधिक जानकारी के लिए [Kotlin डॉक्स](https://github.com/cactus-compute/cactus/blob/main/kotlin) देखें।

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus बैकएंड C/C++ में लिखा गया है और फोन, स्मार्ट टीवी, घड़ियां, स्पीकर, कैमरा, लैपटॉप आदि पर सीधे चल सकता है। अधिक जानकारी के लिए [C++ डॉक्स](https://github.com/cactus-compute/cactus/blob/main/cpp) देखें।


## ![इस रेपो और उदाहरण ऐप्स का उपयोग](https://img.shields.io/badge/रेपो_और_उदाहरण_का_उपयोग-grey.svg?style=for-the-badge)

पहले, `git clone https://github.com/cactus-compute/cactus.git` से रेपो क्लोन करें, इसमें जाएं और `chmod +x scripts/*.sh` से सभी स्क्रिप्ट्स को executable बनाएं

1. **Flutter**
    - `scripts/build-flutter-android.sh` से Android JNILibs बिल्ड करें।
    - `scripts/build-flutter.sh` से Flutter प्लगइन बिल्ड करें। (उदाहरण उपयोग से पहले चलाना आवश्यक)
    - `cd flutter/example` से उदाहरण ऐप पर जाएं।
    - Xcode या Android Studio के जरिए अपना सिम्यूलेटर खोलें, पहली बार के लिए [वॉकथ्रू](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)।
    - हमेशा इस कॉम्बो से ऐप शुरू करें `flutter clean && flutter pub get && flutter run`।
    - ऐप के साथ खेलें, और आवश्यकतानुसार उदाहरण ऐप या प्लगइन में बदलाव करें।

2. **React Native**
    - `scripts/build-react-android.sh` से Android JNILibs बिल्ड करें।
    - `scripts/build-react.sh` से Flutter प्लगइन बिल्ड करें।
    - `cd react/example` से उदाहरण ऐप पर जाएं।
    - Xcode या Android Studio के जरिए अपना सिम्यूलेटर सेटअप करें, पहली बार के लिए [वॉकथ्रू](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)।
    - हमेशा इस कॉम्बो से ऐप शुरू करें `yarn && yarn ios` या `yarn && yarn android`।
    - ऐप के साथ खेलें, और आवश्यकतानुसार उदाहरण ऐप या पैकेज में बदलाव करें।
    - फिलहाल, यदि पैकेज में बदलाव किए गए हैं, तो आप मैन्युअल रूप से फाइलें/फोल्डर को `examples/react/node_modules/cactus-react-native` में कॉपी करेंगे।

3. **Kotlin Multiplatform**
    - `scripts/build-flutter-android.sh` से Android JNILibs बिल्ड करें। (Flutter और Kotlin समान JNILibs साझा करते हैं)
    - `scripts/build-kotlin.sh` से Kotlin लाइब्रेरी बिल्ड करें। (उदाहरण उपयोग से पहले चलाना आवश्यक)
    - `cd kotlin/example` से उदाहरण ऐप पर जाएं।
    - Xcode या Android Studio के जरिए अपना सिम्यूलेटर खोलें, पहली बार के लिए [वॉकथ्रू](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda)।
    - डेस्कटॉप के लिए हमेशा `./gradlew :composeApp:run` से ऐप शुरू करें या मोबाइल के लिए Android Studio/Xcode का उपयोग करें।
    - ऐप के साथ खेलें, और आवश्यकतानुसार उदाहरण ऐप या लाइब्रेरी में बदलाव करें।

4. **C/C++**
    - `cd cactus/example` से उदाहरण ऐप पर जाएं।
    - कई मुख्य फाइलें हैं `main_vlm, main_llm, main_embed, main_tts`।
    - `build.sh` का उपयोग करके लाइब्रेरी और executable दोनों बिल्ड करें।
    - executables में से किसी एक से चलाएं `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`।
    - विभिन्न मॉडल आज़माएं और आवश्यकतानुसार बदलाव करें।

5. **योगदान**
    - बग फिक्स में योगदान करने के लिए, अपने बदलाव के बाद `git checkout -b <ब्रांच-नाम>` से ब्रांच बनाएं और PR सबमिट करें।
    - फीचर में योगदान करने के लिए, कृपया पहले एक issue उठाएं ताकि इस पर चर्चा हो सके, दूसरों के साथ टकराव से बचने के लिए।
    - [हमारे discord में शामिल हों](https://discord.gg/SdZjmfWQ)

## ![प्रदर्शन](https://img.shields.io/badge/प्रदर्शन-grey.svg?style=for-the-badge)

| डिवाइस                        |  Gemma3 1B Q4 (टोकन/सेकंड) |    Qwen3 4B Q4 (टोकन/सेकंड)   |  
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

## ![डेमो](https://img.shields.io/badge/डेमो-grey.svg?style=for-the-badge)

| <img src="assets/ChatDemo.gif" alt="Chat Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="iOS ऐप डाउनलोड करें" src="https://img.shields.io/badge/iOS_डेमो_आज़माएं-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="Android ऐप डाउनलोड करें" src="https://img.shields.io/badge/Android_डेमो_आज़माएं-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="assets/EmbeddingDemo.gif" alt="Embedding Demo" width="220"/> |
| --- | --- |

## ![सिफारिशें](https://img.shields.io/badge/हमारी_सिफारिशें-grey.svg?style=for-the-badge)
हम अपने [HuggingFace पेज](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models) पर सुझाए गए मॉडल्स का संग्रह प्रदान करते हैं
