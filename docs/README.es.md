<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 Traducciones

🇬🇧 [English](../.md) | 🇪🇸 Español | 🇫🇷 [Français](README.fr.md) | 🇨🇳 [中文](README.zh.md) | 🇯🇵 [日本語](README.ja.md) | 🇮🇳 [हिंदी](README.hi.md)
<br/>

Framework multiplataforma para desplegar modelos LLM/VLM/TTS localmente en tu aplicación.

- Disponible en Flutter, React-Native y Kotlin Multiplatform.
- Soporta cualquier modelo GGUF que puedas encontrar en Huggingface; Qwen, Gemma, Llama, DeepSeek etc.
- Ejecuta modelos LLM, VLM, de embedding, TTS y más.
- Acomoda desde modelos FP32 hasta cuantizaciones tan bajas como 2 bits, para eficiencia y menos tensión del dispositivo.
- Plantillas de chat con soporte Jinja2 y streaming de tokens.

[¡HAZ CLIC PARA UNIRTE A NUESTRO DISCORD!](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[HAZ CLIC PARA VISUALIZAR Y CONSULTAR EL REPO](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **Instalar:**
    Ejecuta el siguiente comando en tu terminal del proyecto:
    ```bash
    flutter pub add cactus
    ```
2. **Completado de Texto Flutter**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
    );

    final messages = [ChatMessage(role: 'user', content: '¡Hola!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);
    ```
3. **Embedding Flutter**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        generateEmbeddings: true,
    );

    final text = 'Tu texto para embedder';
    final result = await lm.embedding(text);
    ```
4. **Completado VLM Flutter**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.init(
        modelUrl: 'huggingface/gguf/link',
        mmprojUrl: 'huggingface/gguf/mmproj/link',
    );

    final messages = [ChatMessage(role: 'user', content: 'Describe esta imagen')];

    final response = await vlm.completion(
        messages, 
        imagePaths: ['/ruta/absoluta/a/imagen.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```
5. **Respaldo en la Nube Flutter**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        cactusToken: 'enterprise_token_here', 
    );

    final messages = [ChatMessage(role: 'user', content: '¡Hola!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local (por defecto): estrictamente solo ejecutar en el dispositivo
    // localfirst: respaldo a la nube si el dispositivo falla
    // remotefirst: principalmente remoto, ejecutar local si la API falla
    // remote: estrictamente ejecutar en la nube
    final embedding = await lm.embedding('Tu texto', mode: 'localfirst');
    ```

  N/B: Ve los [Docs de Flutter](https://github.com/cactus-compute/cactus/blob/main/flutter) para más información.

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **Instala el paquete `cactus-react-native`:**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **Completado de Texto React-Native**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/ruta/a/model.gguf',
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: '¡Hola!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```
3. **Embedding React-Native**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/ruta/a/model.gguf',
        n_ctx: 2048,
        embedding: true,
    });

    const text = 'Tu texto para embedder';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

4. **VLM React-Native**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/ruta/a/vision-model.gguf',
        mmproj: '/ruta/a/mmproj.gguf',
    });

    const messages = [{ role: 'user', content: 'Describe esta imagen' }];

    const params = {
        images: ['/ruta/absoluta/a/imagen.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
5. **Respaldo en la Nube React-Native**
    ```typescript
    import { CactusLM } from 'cactus-react-native';

    const { lm, error } = await CactusLM.init({
        model: '/ruta/a/model.gguf',
        n_ctx: 2048,
    }, undefined, 'enterprise_token_here');

    const messages = [{ role: 'user', content: '¡Hola!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);

    // local (por defecto): estrictamente solo ejecutar en el dispositivo
    // localfirst: respaldo a la nube si el dispositivo falla
    // remotefirst: principalmente remoto, ejecutar local si la API falla
    // remote: estrictamente ejecutar en la nube
    const embedding = await lm.embedding('Tu texto', undefined, 'localfirst');
    ```
N/B: Ve los [Docs de React](https://github.com/cactus-compute/cactus/blob/main/react) para más información.

## ![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **Agregar Dependencia Maven:**
    Agregar al `build.gradle.kts` de tu proyecto KMP:
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

2. **Configuración de Plataforma:**
    - **Android:** Funciona automáticamente - bibliotecas nativas incluidas.
    - **iOS:** En Xcode: File → Add Package Dependencies → Pegar `https://github.com/cactus-compute/cactus` → Hacer clic en Add

3. **Completado de Texto Kotlin Multiplatform**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // Establecer a 99 para descarga completa de GPU
        )
        
        val downloadSuccess = lm.download(
            url = "ruta/a/hugginface/gguf",
            filename = "model_filename.gguf"
        )
        val initSuccess = lm.init("qwen3-600m.gguf")
        
        val result = lm.completion(
            prompt = "¡Hola!",
            maxTokens = 100,
            temperature = 0.7f
        )
    }
    ```

4. **Reconocimiento de Voz Kotlin Multiplatform**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val stt = CactusSTT(
            language = "es-ES",
            sampleRate = 16000,
            maxDuration = 30
        )
        
        // Solo soporta modelo STT Vosk por defecto para Android y Apple Foundation Model
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()
        
        val result = stt.transcribe()
        result?.let { sttResult ->
            println("Transcrito: ${sttResult.text}")
            println("Confianza: ${sttResult.confidence}")
        }
        
        // O transcribir desde archivo de audio
        val fileResult = stt.transcribeFile("/ruta/a/audio.wav")
    }
    ```

5. **VLM Kotlin Multiplatform**
    ```kotlin
    import com.cactus.CactusVLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val vlm = CactusVLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // Establecer a 99 para descarga completa de GPU
        )
        
        val downloadSuccess = vlm.download(
            modelUrl = "ruta/a/hugginface/gguf",
            mmprojUrl = "ruta/a/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")
        
        val result = vlm.completion(
            prompt = "Describe esta imagen",
            imagePath = "/ruta/a/imagen.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

  N/B: Ve los [Docs de Kotlin](https://github.com/cactus-compute/cactus/blob/main/kotlin) para más información.

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

El backend de Cactus está escrito en C/C++ y puede ejecutarse directamente en teléfonos, smart TVs, relojes, altavoces, cámaras, laptops, etc. Ve los [Docs de C++](https://github.com/cactus-compute/cactus/blob/main/cpp) para más información.


## ![Usando este Repo y Apps de Ejemplo](https://img.shields.io/badge/Usando_Repo_Y_Ejemplos-grey.svg?style=for-the-badge)

Primero, clona el repo con `git clone https://github.com/cactus-compute/cactus.git`, ve a él y haz todos los scripts ejecutables con `chmod +x scripts/*.sh`

1. **Flutter**
    - Construye las JNILibs de Android con `scripts/build-flutter-android.sh`.
    - Construye el Plugin de Flutter con `scripts/build-flutter.sh`. (DEBE ejecutarse antes de usar el ejemplo)
    - Navega a la app de ejemplo con `cd flutter/example`.
    - Abre tu simulador vía Xcode o Android Studio, [tutorial](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) si no lo has hecho antes.
    - Siempre inicia la app con esta combinación `flutter clean && flutter pub get && flutter run`.
    - Juega con la app, y haz cambios ya sea a la app de ejemplo o al plugin como desees.

2. **React Native**
    - Construye las JNILibs de Android con `scripts/build-react-android.sh`.
    - Construye el Plugin de Flutter con `scripts/build-react.sh`.
    - Navega a la app de ejemplo con `cd react/example`.
    - Configura tu simulador vía Xcode o Android Studio, [tutorial](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) si no lo has hecho antes.
    - Siempre inicia la app con esta combinación `yarn && yarn ios` o `yarn && yarn android`.
    - Juega con la app, y haz cambios ya sea a la app de ejemplo o al paquete como desees.
    - Por ahora, si se hacen cambios en el paquete, copiarías manualmente los archivos/carpetas en `examples/react/node_modules/cactus-react-native`.

3. **Kotlin Multiplatform**
    - Construye las JNILibs de Android con `scripts/build-flutter-android.sh`. (Flutter y Kotlin comparten las mismas JNILibs)
    - Construye la biblioteca Kotlin con `scripts/build-kotlin.sh`. (DEBE ejecutarse antes de usar el ejemplo)
    - Navega a la app de ejemplo con `cd kotlin/example`.
    - Abre tu simulador vía Xcode o Android Studio, [tutorial](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) si no lo has hecho antes.
    - Siempre inicia la app con `./gradlew :composeApp:run` para escritorio o usa Android Studio/Xcode para móvil.
    - Juega con la app, y haz cambios ya sea a la app de ejemplo o a la biblioteca como desees.

4. **C/C++**
    - Navega a la app de ejemplo con `cd cactus/example`.
    - Hay múltiples archivos main `main_vlm, main_llm, main_embed, main_tts`.
    - Construye tanto las bibliotecas como el ejecutable usando `build.sh`.
    - Ejecuta con uno de los ejecutables `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`.
    - Prueba diferentes modelos y haz cambios como desees.

5. **Contribuyendo**
    - Para contribuir con una corrección de error, crea una rama después de hacer tus cambios con `git checkout -b <nombre-rama>` y envía un PR.
    - Para contribuir con una característica, por favor levanta un issue primero para que pueda ser discutido, para evitar intersección con alguien más.
    - [Únete a nuestro discord](https://discord.gg/SdZjmfWQ)

## ![Rendimiento](https://img.shields.io/badge/Rendimiento-grey.svg?style=for-the-badge)

| Dispositivo                   |  Gemma3 1B Q4 (toks/seg) |    Qwen3 4B Q4 (toks/seg)   |  
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

| <img src="assets/ChatDemo.gif" alt="Chat Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="Descargar App iOS" src="https://img.shields.io/badge/Probar_Demo_iOS-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="Descargar App Android" src="https://img.shields.io/badge/Probar_Demo_Android-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="assets/EmbeddingDemo.gif" alt="Embedding Demo" width="220"/> |
| --- | --- |

## ![Recomendaciones](https://img.shields.io/badge/Nuestras_Recomendaciones-grey.svg?style=for-the-badge)
Proporcionamos una colección de modelos recomendados en nuestra [Página de HuggingFace](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models)
