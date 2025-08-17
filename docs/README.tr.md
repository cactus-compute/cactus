<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 Çeviriler

🇬🇧 [İngilizce](../README.md) | 🇪🇸 [İspanyolca](../README.es.md) | 🇫🇷 [Fransızca](../README.fr.md) | 🇨🇳 [Çince](../README.zh.md) | 🇯🇵 [Japonca](../README.ja.md) | 🇮🇳 [Hintçe](../README.hi.md) | 🇩🇪 [Almanca](../README.de.md) | 🇰🇷 [Korece](../README.ko.md)
<br/>

Uygulamanızda LLM/VLM/TTS modellerini yerel olarak çalıştırmak için çapraz platform framework.

- Flutter, React-Native ve Kotlin Multiplatform üzerinde kullanılabilir.
- Huggingface üzerinde bulabileceğiniz herhangi bir GGUF modelini destekler; Qwen, Gemma, Llama, DeepSeek vb.
- LLM’ler, VLM’ler, Embedding Modelleri, TTS modelleri ve daha fazlasını çalıştırın.
- Verimlilik ve cihaz üzerindeki yükü azaltmak için FP32’den 2-bit quantize edilmiş modellere kadar destekler.
- Jinja2 destekli sohbet şablonları ve token akışı.

[DISCORD SUNUCUMUZA KATILMAK İÇİN TIKLAYIN!](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[DEPOYU GÖRSELLEŞTİRMEK VE SORGULAMAK İÇİN TIKLAYIN](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **Kurulum:**
    Projenizin terminalinde aşağıdaki komutu çalıştırın:
    ```bash
    flutter pub add cactus
    ```

2. **Flutter Metin Tamamlama**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.download(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
    );

    lm.init()

    final messages = [ChatMessage(role: 'user', content: 'Merhaba!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);
    ```

3. **Flutter Gömme (Embedding)**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.download(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
        generateEmbeddings: true,
    );
    lm.init()

    final text = 'Gömülmek istenen metniniz';
    final result = await lm.embedding(text);
    ```

4. **Flutter VLM Tamamlama**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.download(
        modelUrl: 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
        mmprojUrl: 'https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
    );

    vlm.init()

    final messages = [ChatMessage(role: 'user', content: 'Bu resmi açıkla')];

    final response = await vlm.completion(
        messages,
        imagePaths: ['/absolute/path/to/image.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```

5. **Flutter Bulut Yedekleme**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.download(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
        contextSize: 2048,
        cactusToken: 'token_buraya',
    );

    lm.init()

    final messages = [ChatMessage(role: 'user', content: 'Merhaba!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local (varsayılan): yalnızca cihaz üzerinde çalıştır
    // localfirst: cihazda çalıştır, başarısız olursa buluta düş
    // remotefirst: öncelikli olarak bulut, başarısız olursa cihazda çalıştır
    // remote: yalnızca bulut üzerinde çalıştır
    final embedding = await lm.embedding('Metniniz', mode: 'localfirst');
    ```

6. **Flutter Aracı Araçlar (Agentic Tools)**
    ```dart
    import 'package:cactus/cactus.dart';

    final agent = await CactusAgent.download(
        modelUrl: 'https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
    );

    agent.init()

    agent!.addTool(
        'araç_adı',
        Tool(),
        'Araç Bilgisi',
        {
        'parametre': Parameter(
            type: 'string',
            description: 'İhtiyacınız olan parametre!',
            required: true,
        ),
        },
    );

    final messages = [ChatMessage(role: 'user', content: 'Merhaba!')];

    final response = await agent.completionWithTools(
        messages,
        maxTokens: 200,
        temperature: 0.3,
    );
    ```

Not: Daha fazla bilgi için [Flutter Dokümanları](https://github.com/cactus-compute/cactus/blob/main/flutter) bölümüne bakın.

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **`cactus-react-native` paketini yükleyin:**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **React-Native Metin Tamamlama**
    ```typescript
    import { CactusLM } from 'cactus-react-native';

    const { lm, error } = await CactusLM.init({
        model: '/path/to/model.gguf', // bu, uygulama sandbox'ı içinde yerel bir model dosyasıdır
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: 'Merhaba!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```

3. **React-Native Gömme (Embedding)**
    ```typescript
    import { CactusLM } from 'cactus-react-native';

    const { lm, error } = await CactusLM.init({
        model: '/path/to/model.gguf', // uygulama sandbox'ı içinde yerel model dosyası
        n_ctx: 2048,
        embedding: true,
    });

    const text = 'Gömülmek istenen metniniz';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

4. **React-Native VLM (Görsel Dil Modeli)**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/path/to/vision-model.gguf', // uygulama sandbox'ı içinde yerel model dosyası
        mmproj: '/path/to/mmproj.gguf', // uygulama sandbox'ı içinde yerel model dosyası
    });

    const messages = [{ role: 'user', content: 'Bu resmi açıkla' }];

    const params = {
        images: ['/absolute/path/to/image.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```

5. **React-Native Ajanlar (Agents)**

    ```typescript
    import { CactusAgent } from 'cactus-react-native';

    // Qwen 3 ailesini öneriyoruz, 0.6B oldukça iyi
    const { agent, error } = await CactusAgent.init({
        model: '/path/to/model.gguf',
        n_ctx: 2048,
    });

    const weatherTool = agent.addTool(
        (location: string) => `${location} için hava durumu: 24°C, güneşli`,
        'Bir konum için güncel hava durumunu al',
        {
            location: { type: 'string', description: 'Şehir Adı', required: true }
        }
    );

    const messages = [{ role: 'user', content: 'NYC’de hava nasıl?' }];
      const result = await agent.completionWithTools(messages, {
      n_predict: 200,
      temperature: 0.7,
    });

    await agent.release();
    ```

`CactusAgent` kullanılarak oluşturulmuş bir [örnek uygulama](https://github.com/cactus-compute/example-react-agents/) ile başlamaya hazır olun.

Daha fazla bilgi için [React Dokümanları](https://github.com/cactus-compute/cactus/blob/main/react) bölümüne bakın.

## ![Kotlin Çoklu Platform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **Maven Bağımlılığını Ekleyin:**
    KMP projenizin `build.gradle.kts` dosyasına ekleyin:
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

2. **Platform Kurulumu:**
    - **Android:** Otomatik olarak çalışır – yerel kütüphaneler dahildir.
    - **iOS:** Xcode içinde: Dosya → Paket bağımlılıkları ekleyin → Yapıştır `https://github.com/cactus-compute/cactus` → Ekle'ye tıklayın

3. **Kotlin Çoklu Platform Metin Tamamlama**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking

    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // Tam GPU offload için 99 olarak ayarlayın
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

4. **Kotlin Çoklu Platform Konuşmadan Metne**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking

    runBlocking {
        val stt = CactusSTT(
            language = "en-US",
            sampleRate = 16000,
            maxDuration = 30
        )

        // Yalnızca Android için varsayılan Vosk STT modeli ve Apple Foundation Model desteklenmektedir
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()

        val result = stt.transcribe()
        result?.let { sttResult ->
            println("Çözümlenen: ${sttResult.text}")
            println("Güven: ${sttResult.confidence}")
        }

        // Veya ses dosyasından çözümleme yapın
        val fileResult = stt.transcribeFile("/path/to/audio.wav")
    }
    ```

5. **Kotlin Çoku Platform VLM (Görsel Dil Modeli)**
    ```kotlin
    import com.cactus.CactusVLM
    import kotlinx.coroutines.runBlocking

    runBlocking {
        val vlm = CactusVLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // Tam GPU offload için 99 olarak ayarlayın
        )

        val downloadSuccess = vlm.download(
            modelUrl = "path/to/hugginface/gguf",
            mmprojUrl = "path/to/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")

        val result = vlm.completion(
            prompt = "Bu resmi açıkla",
            imagePath = "/path/to/image.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

Not: Daha fazlası için [Kotlin Dokümanları](https://github.com/cactus-compute/cactus/blob/main/kotlin) bölümüne bakın.

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Cactus backend C/C++ ile yazılmıştır ve doğrudan telefonlarda, akıllı TV’lerde, saatlerde, hoparlörlerde, kameralarda, dizüstü bilgisayarlarda vb. çalışabilir. Daha fazlası için [C++ Dokümanları](https://github.com/cactus-compute/cactus/blob/main/cpp) bölümüne bakın.

## ![Bu Depoyu ve Örnek Uygulamaları Kullanma](https://img.shields.io/badge/Repo_Ve_Örneklerini_Kullanma-grey.svg?style=for-the-badge)

Öncelikle repoyu şu komutla klonlayın:
`git clone https://github.com/cactus-compute/cactus.git`

Ardından dizine girin ve tüm script dosyalarını çalıştırılabilir hale getirin:
`chmod +x scripts/*.sh`

1. **Flutter**
    - Android JNILibs dosyalarını `scripts/build-flutter-android.sh` ile oluşturun.
    - Flutter Plugin’i `scripts/build-flutter.sh` ile oluşturun. (Örneği kullanmadan önce ÇALIŞTIRILMASI GEREKİR)
    - Örnek uygulamaya gitmek için `cd flutter/example` komutunu çalıştırın.
    - Simülatörünüzü Xcode veya Android Studio üzerinden açın. Daha önce yapmadıysanız [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) bağlantısını takip edin.
    - Uygulamayı her zaman şu kombinasyonla başlatın: `flutter clean && flutter pub get && flutter run`
    - Uygulamayla oynayın ve istediğiniz şekilde örnek uygulamada veya plugin üzerinde değişiklik yapın.

2. **React Native**
    - Android JNILibs dosyalarını `scripts/build-react-android.sh` ile oluşturun.
    - React Plugin’i `scripts/build-react.sh` ile oluşturun.
    - Örnek uygulamaya gitmek için `cd react/example` komutunu çalıştırın.
    - Simülatörünüzü Xcode veya Android Studio üzerinden ayarlayın. Daha önce yapmadıysanız [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) bağlantısını takip edin.
    - Uygulamayı her zaman şu kombinasyonla başlatın: `yarn && yarn ios` veya `yarn && yarn android`
    - Uygulamayla oynayın ve istediğiniz şekilde örnek uygulamada veya paket üzerinde değişiklik yapın.
    - Şimdilik, pakette değişiklik yaparsanız dosya/klasörleri manuel olarak `examples/react/node_modules/cactus-react-native` içine kopyalamanız gerekir.

3. **Kotlin Çoku Platform**
    - Android JNILibs dosyalarını `scripts/build-flutter-android.sh` ile oluşturun. (Flutter ve Kotlin aynı JNILibs’i paylaşır)
    - Kotlin kütüphanesini `scripts/build-kotlin.sh` ile oluşturun. (Örneği kullanmadan önce ÇALIŞTIRILMASI GEREKİR)
    - Örnek uygulamaya gitmek için `cd kotlin/example` komutunu çalıştırın.
    - Simülatörünüzü Xcode veya Android Studio üzerinden açın. Daha önce yapmadıysanız [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) bağlantısını takip edin.
    - Uygulamayı her zaman masaüstü için şu komutla başlatın: `./gradlew :composeApp:run`
    - Uygulamayla oynayın ve istediğiniz şekilde örnek uygulamada veya kütüphane üzerinde değişiklik yapın.

4. **C/C++**
    - Örnek uygulamaya gitmek için `cd cactus/example` komutunu çalıştırın.
    - Birden fazla ana dosya vardır: `main_vlm, main_llm, main_embed, main_tts`.
    - Hem kütüphaneleri hem de çalıştırılabilir dosyaları `build.sh` ile derleyin.
    - Çalıştırmak için şu çalıştırılabilir dosyalardan birini kullanın: `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`
    - Farklı modelleri deneyin ve istediğiniz değişiklikleri yapın.

5. **Katkıda Bulunma**
    - Bir hata düzeltmesi katkısında bulunmak için değişikliklerinizi yaptıktan sonra `git checkout -b <branch-name>` komutuyla bir branch oluşturun ve bir PR gönderin.
    - Yeni bir özellik eklemek için lütfen önce bir issue açın, böylece başkalarının çalışmalarıyla çakışmayı önlemek adına tartışılabilir.
    - [Discord sunucumuza katılın](https://discord.gg/bNurx3AXTJ)

## ![Performans](https://img.shields.io/badge/Performans-grey.svg?style=for-the-badge)

| Cihaz                         |  Gemma3 1B Q4 (token/sn) |    Qwen3 4B Q4 (token/sn)   |
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

| <img src="assets/ChatDemo.gif" alt="Sohbet Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="iOS Uygulamasını İndir" src="https://img.shields.io/badge/iOS_Demo'yu_Dene-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="Android Uygulamasını İndir" src="https://img.shields.io/badge/Android_Demo'yu_Dene-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="assets/EmbeddingDemo.gif" alt="Gömülü Demo" width="220"/> |
| --- | --- |

## ![Öneriler](https://img.shields.io/badge/Öneriler-grey.svg?style=for-the-badge)
[HuggingFace Sayfamızda](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models) önerilen modellerin bir koleksiyonunu sunuyoruz.
