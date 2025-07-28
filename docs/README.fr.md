<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

<span>
  <img alt="Y Combinator" src="https://img.shields.io/badge/Combinator-F0652F?style=for-the-badge&logo=ycombinator&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Oxford Seed Fund" src="https://img.shields.io/badge/Oxford_Seed_Fund-002147?style=for-the-badge&logo=oxford&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
  <img alt="Google for Startups" src="https://img.shields.io/badge/Google_For_Startups-4285F4?style=for-the-badge&logo=google&logoColor=white" height="18" style="vertical-align:middle;border-radius:4px;">
</span>

## 🌍 Traductions

🇬🇧 [English](../README.md) | 🇪🇸 [Español](README.es.md) | 🇫🇷 Français | 🇨🇳 [中文](README.zh.md) | 🇯🇵 [日本語](README.ja.md) | 🇮🇳 [हिंदी](README.hi.md)
<br/>

Framework multiplateforme pour déployer des modèles LLM/VLM/TTS localement dans votre application.

- Disponible pour Flutter, React-Native et Kotlin Multiplateforme.
- Prend en charge tout modèle GGUF que vous pouvez trouver sur Huggingface ; Qwen, Gemma, Llama, DeepSeek etc.
- Exécutez des modèles LLM, VLM, d'embedding, TTS et plus encore.
- Prend en charge des modèles de FP32 jusqu'à des quantifications aussi basses que 2 bits, pour l'efficacité et moins de contrainte sur l'appareil.
- Templates de chat avec support Jinja2 et streaming de tokens.

[CLIQUEZ POUR REJOINDRE NOTRE DISCORD !](https://discord.gg/bNurx3AXTJ)
<br/>
<br/>
[CLIQUEZ POUR VISUALISER ET INTERROGER LE REPO](https://repomapr.com/cactus-compute/cactus)

## ![Flutter](https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white)

1.  **Installation :**
    Exécutez la commande suivante dans votre terminal de projet :
    ```bash
    flutter pub add cactus
    ```
2. **Complétion de texte Flutter**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
    );

    final messages = [ChatMessage(role: 'user', content: 'Bonjour!')];
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

    final text = 'Votre texte à incorporer';
    final result = await lm.embedding(text);
    ```
4. **Complétion VLM Flutter**
    ```dart
    import 'package:cactus/cactus.dart';

    final vlm = await CactusVLM.init(
        modelUrl: 'huggingface/gguf/link',
        mmprojUrl: 'huggingface/gguf/mmproj/link',
    );

    final messages = [ChatMessage(role: 'user', content: 'Décrivez cette image')];

    final response = await vlm.completion(
        messages, 
        imagePaths: ['/chemin/absolu/vers/image.jpg'],
        maxTokens: 200,
        temperature: 0.3,
    );
    ```
5. **Fallback Cloud Flutter**
    ```dart
    import 'package:cactus/cactus.dart';

    final lm = await CactusLM.init(
        modelUrl: 'huggingface/gguf/link',
        contextSize: 2048,
        cactusToken: 'enterprise_token_here', 
    );

    final messages = [ChatMessage(role: 'user', content: 'Bonjour!')];
    final response = await lm.completion(messages, maxTokens: 100, temperature: 0.7);

    // local (par défaut): strictement uniquement sur l'appareil
    // localfirst: fallback vers le cloud si l'appareil échoue
    // remotefirst: principalement distant, exécuter localement si l'API échoue
    // remote: strictement exécuter sur le cloud 
    final embedding = await lm.embedding('Votre texte', mode: 'localfirst');
    ```

  N/B: Voir la [Documentation Flutter](https://github.com/cactus-compute/cactus/blob/main/flutter) pour plus d'informations.

## ![React Native](https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

1.  **Installez le package `cactus-react-native` :**
    ```bash
    npm install cactus-react-native && npx pod-install
    ```

2. **Complétion de texte React-Native**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/chemin/vers/model.gguf',
        n_ctx: 2048,
    });

    const messages = [{ role: 'user', content: 'Bonjour!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);
    ```
3. **Embedding React-Native**
    ```typescript
    import { CactusLM } from 'cactus-react-native';
    
    const { lm, error } = await CactusLM.init({
        model: '/chemin/vers/model.gguf',
        n_ctx: 2048,
        embedding: true,
    });

    const text = 'Votre texte à incorporer';
    const params = { normalize: true };
    const result = await lm.embedding(text, params);
    ```

4. **VLM React-Native**
    ```typescript
    import { CactusVLM } from 'cactus-react-native';

    const { vlm, error } = await CactusVLM.init({
        model: '/chemin/vers/vision-model.gguf',
        mmproj: '/chemin/vers/mmproj.gguf',
    });

    const messages = [{ role: 'user', content: 'Décrivez cette image' }];

    const params = {
        images: ['/chemin/absolu/vers/image.jpg'],
        n_predict: 200,
        temperature: 0.3,
    };

    const response = await vlm.completion(messages, params);
    ```
5. **Fallback Cloud React-Native**
    ```typescript
    import { CactusLM } from 'cactus-react-native';

    const { lm, error } = await CactusLM.init({
        model: '/chemin/vers/model.gguf',
        n_ctx: 2048,
    }, undefined, 'enterprise_token_here');

    const messages = [{ role: 'user', content: 'Bonjour!' }];
    const params = { n_predict: 100, temperature: 0.7 };
    const response = await lm.completion(messages, params);

    // local (par défaut): strictement uniquement sur l'appareil
    // localfirst: fallback vers le cloud si l'appareil échoue
    // remotefirst: principalement distant, exécuter localement si l'API échoue
    // remote: strictement exécuter sur le cloud 
    const embedding = await lm.embedding('Votre texte', undefined, 'localfirst');
    ```
N/B: Voir la [Documentation React](https://github.com/cactus-compute/cactus/blob/main/react) pour plus d'informations.

## ![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin_Multiplatform-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white)

1.  **Ajoutez la dépendance Maven :**
    Ajoutez à votre `build.gradle.kts` de projet KMP :
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

2. **Configuration de plateforme :**
    - **Android :** Fonctionne automatiquement - bibliothèques natives incluses.
    - **iOS :** Dans Xcode : File → Add Package Dependencies → Collez `https://github.com/cactus-compute/cactus` → Cliquez Add

3. **Complétion de texte Kotlin Multiplatform**
    ```kotlin
    import com.cactus.CactusLM
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val lm = CactusLM(
            threads = 4,
            contextSize = 2048,
            gpuLayers = 0 // Définir à 99 pour le déchargement GPU complet
        )
        
        val downloadSuccess = lm.download(
            url = "chemin/vers/hugginface/gguf",
            filename = "model_filename.gguf"
        )
        val initSuccess = lm.init("qwen3-600m.gguf")
        
        val result = lm.completion(
            prompt = "Bonjour!",
            maxTokens = 100,
            temperature = 0.7f
        )
    }
    ```

4. **Reconnaissance vocale Kotlin Multiplatform**
    ```kotlin
    import com.cactus.CactusSTT
    import kotlinx.coroutines.runBlocking
    
    runBlocking {
        val stt = CactusSTT(
            language = "fr-FR",
            sampleRate = 16000,
            maxDuration = 30
        )
        
        // Prend uniquement en charge le modèle STT Vosk par défaut pour Android et Apple Foundation Model
        val downloadSuccess = stt.download()
        val initSuccess = stt.init()
        
        val result = stt.transcribe()
        result?.let { sttResult ->
            println("Transcrit: ${sttResult.text}")
            println("Confiance: ${sttResult.confidence}")
        }
        
        // Ou transcrire à partir d'un fichier audio
        val fileResult = stt.transcribeFile("/chemin/vers/audio.wav")
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
            gpuLayers = 0 // Définir à 99 pour le déchargement GPU complet
        )
        
        val downloadSuccess = vlm.download(
            modelUrl = "chemin/vers/hugginface/gguf",
            mmprojUrl = "chemin/vers/hugginface/mmproj/gguf",
            modelFilename = "model_filename.gguf",
            mmprojFilename = "mmproj_filename.gguf"
        )
        val initSuccess = vlm.init("smolvlm2-500m.gguf", "mmproj-smolvlm2-500m.gguf")
        
        val result = vlm.completion(
            prompt = "Décrivez cette image",
            imagePath = "/chemin/vers/image.jpg",
            maxTokens = 200,
            temperature = 0.3f
        )
    }
    ```

  N/B: Voir la [Documentation Kotlin](https://github.com/cactus-compute/cactus/blob/main/kotlin) pour plus d'informations.

## ![C++](https://img.shields.io/badge/C%2B%2B-grey.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Le backend Cactus est écrit en C/C++ et peut s'exécuter directement sur les téléphones, smart TVs, montres, haut-parleurs, caméras, ordinateurs portables etc. Voir la [Documentation C++](https://github.com/cactus-compute/cactus/blob/main/cpp) pour plus d'informations.


## ![Utilisation de ce Repo et Applications d'Exemple](https://img.shields.io/badge/Utilisation_Repo_Et_Exemples-grey.svg?style=for-the-badge)

D'abord, clonez le repo avec `git clone https://github.com/cactus-compute/cactus.git`, naviguez dedans et rendez tous les scripts exécutables avec `chmod +x scripts/*.sh`

1. **Flutter**
    - Construisez les JNILibs Android avec `scripts/build-flutter-android.sh`.
    - Construisez le Plugin Flutter avec `scripts/build-flutter.sh`. (DOIT être exécuté avant d'utiliser l'exemple)
    - Naviguez vers l'application d'exemple avec `cd flutter/example`.
    - Ouvrez votre simulateur via Xcode ou Android Studio, [tutoriel](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) si vous ne l'avez pas fait auparavant.
    - Démarrez toujours l'application avec cette combinaison `flutter clean && flutter pub get && flutter run`.
    - Jouez avec l'application, et apportez des modifications soit à l'application d'exemple soit au plugin comme souhaité.

2. **React Native**
    - Construisez les JNILibs Android avec `scripts/build-react-android.sh`.
    - Construisez le Plugin Flutter avec `scripts/build-react.sh`.
    - Naviguez vers l'application d'exemple avec `cd react/example`.
    - Configurez votre simulateur via Xcode ou Android Studio, [tutoriel](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) si vous ne l'avez pas fait auparavant.
    - Démarrez toujours l'application avec cette combinaison `yarn && yarn ios` ou `yarn && yarn android`.
    - Jouez avec l'application, et apportez des modifications soit à l'application d'exemple soit au package comme souhaité.
    - Pour l'instant, si des modifications sont apportées dans le package, vous copieriez manuellement les fichiers/dossiers dans `examples/react/node_modules/cactus-react-native`.

3. **Kotlin Multiplatform**
    - Construisez les JNILibs Android avec `scripts/build-flutter-android.sh`. (Flutter et Kotlin partagent les mêmes JNILibs)
    - Construisez la bibliothèque Kotlin avec `scripts/build-kotlin.sh`. (DOIT être exécuté avant d'utiliser l'exemple)
    - Naviguez vers l'application d'exemple avec `cd kotlin/example`.
    - Ouvrez votre simulateur via Xcode ou Android Studio, [tutoriel](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) si vous ne l'avez pas fait auparavant.
    - Démarrez toujours l'application avec `./gradlew :composeApp:run` pour desktop ou utilisez Android Studio/Xcode pour mobile.
    - Jouez avec l'application, et apportez des modifications soit à l'application d'exemple soit à la bibliothèque comme souhaité.

4. **C/C++**
    - Naviguez vers l'application d'exemple avec `cd cactus/example`.
    - Il y a plusieurs fichiers main `main_vlm, main_llm, main_embed, main_tts`.
    - Construisez à la fois les bibliothèques et l'exécutable en utilisant `build.sh`.
    - Exécutez avec l'un des exécutables `./cactus_vlm`, `./cactus_llm`, `./cactus_embed`, `./cactus_tts`.
    - Essayez différents modèles et apportez des modifications comme souhaité.

5. **Contribution**
    - Pour contribuer à une correction de bug, créez une branche après avoir fait vos modifications avec `git checkout -b <nom-de-branche>` et soumettez une PR.
    - Pour contribuer à une fonctionnalité, veuillez d'abord soulever un problème pour qu'il puisse être discuté, pour éviter l'intersection avec quelqu'un d'autre.
    - [Rejoignez notre discord](https://discord.gg/SdZjmfWQ)

## ![Performances](https://img.shields.io/badge/Performances-grey.svg?style=for-the-badge)

| Appareil                      |  Gemma3 1B Q4 (toks/sec) |    Qwen3 4B Q4 (toks/sec)   |  
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

## ![Démo](https://img.shields.io/badge/Démo-grey.svg?style=for-the-badge)

| <img src="assets/ChatDemo.gif" alt="Chat Demo" width="250"/> | <a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212"><img alt="Télécharger l'App iOS" src="https://img.shields.io/badge/Essayer_Démo_iOS-grey?style=for-the-badge&logo=apple&logoColor=white" height="25"/></a><br/><a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share"><img alt="Télécharger l'App Android" src="https://img.shields.io/badge/Essayer_Démo_Android-grey?style=for-the-badge&logo=android&logoColor=white" height="25"/></a> |
| --- | --- |

| <img src="assets/VLMDemo.gif" alt="VLM Demo" width="220"/> | <img src="assets/EmbeddingDemo.gif" alt="Embedding Demo" width="220"/> |
| --- | --- |

## ![Recommandations](https://img.shields.io/badge/Nos_Recommandations-grey.svg?style=for-the-badge)
Nous fournissons une collection de modèles recommandés sur notre [Page HuggingFace](https://huggingface.co/Cactus-Compute?sort_models=alphabetical#models)
