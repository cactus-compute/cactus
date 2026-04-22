# Quickstart

Install Cactus and run your first on-device AI completion.

## Installation

=== "React Native"

    ```bash
    npm install cactus-react-native react-native-nitro-modules
    ```

=== "Flutter"

    --8<-- "flutter/README.md:install"

    ### Platform Integration

    --8<-- "flutter/README.md:integration"

=== "Kotlin"

    --8<-- "android/README.md:install"

    ### Platform Integration

    --8<-- "android/README.md:integration"

=== "Swift"

    --8<-- "apple/README.md:install"

    ### Platform Integration

    --8<-- "apple/README.md:integration"

=== "Python"

    --8<-- "python/README.md:install"

=== "Rust"

    --8<-- "rust/README.md:install"

=== "CLI"

    **Homebrew (macOS):**

    ```bash
    brew install cactus-compute/cactus/cactus
    ```

    **From Source (macOS):**

    ```bash
    git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
    ```

    **From Source (Linux):**

    ```bash
    sudo apt-get install python3 python3-venv python3-pip cmake build-essential libcurl4-openssl-dev
    git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
    ```

=== "C++"

    Include the Cactus header in your project:

    ```cpp
    #include <cactus.h>
    ```

    See the [Cactus repository](https://github.com/cactus-compute/cactus) for CMake build instructions.

---

## Your First Completion

=== "React Native"

    ```tsx
    import { useCactusLM } from 'cactus-react-native';

    const App = () => {
      const cactusLM = useCactusLM();

      useEffect(() => {
        if (!cactusLM.isDownloaded) {
          cactusLM.download();
        }
      }, []);

      const handleGenerate = () => {
        cactusLM.complete({
          messages: [{ role: 'user', content: 'What is the capital of France?' }],
        });
      };

      if (cactusLM.isDownloading) {
        return <Text>Downloading: {Math.round(cactusLM.downloadProgress * 100)}%</Text>;
      }

      return (
        <>
          <Button onPress={handleGenerate} title="Generate" />
          <Text>{cactusLM.completion}</Text>
        </>
      );
    };
    ```

=== "Flutter"

    --8<-- "flutter/README.md:example"

=== "Kotlin"

    --8<-- "android/README.md:example"

=== "Swift"

    --8<-- "apple/README.md:example"

=== "Python"

    --8<-- "python/README.md:example"

=== "Rust"

    ```rust
    use cactus_sys::*;
    use std::ffi::CString;

    unsafe {
        let model_path = CString::new("path/to/weight/folder").unwrap();
        let model = cactus_init(model_path.as_ptr(), std::ptr::null(), false);

        let messages = CString::new(
            r#"[{"role": "user", "content": "What is the capital of France?"}]"#
        ).unwrap();

        let mut response = vec![0u8; 4096];
        cactus_complete(
            model, messages.as_ptr(),
            response.as_mut_ptr() as *mut i8, 4096,
            std::ptr::null(), std::ptr::null(),
            None, std::ptr::null_mut(),
        );

        println!("{}", String::from_utf8_lossy(&response));
        cactus_destroy(model);
    }
    ```

=== "CLI"

    ```bash
    cactus run LiquidAI/LFM2.5-350M
    ```

=== "C++"

    ```cpp
    #include <cactus.h>

    cactus_model_t model = cactus_init(
        "path/to/weight/folder",
        "path/to/rag/documents",
        false
    );

    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ])";

    char response[4096];
    int result = cactus_complete(
        model, messages, response, sizeof(response),
        nullptr, nullptr, nullptr, nullptr,
        nullptr, 0
    );
    ```

---

## Supported Models

- **LLMs:** Gemma-3 (270M, FunctionGemma-270M, 1B), Gemma-4 (E2B, E4B), Gemma-3n (E2B, E4B), LiquidAI LFM2 (350M, 700M, 2.6B) / LFM2.5 (1.2B-Instruct, 1.2B-Thinking) / LFM2-8B-A1B, Qwen3 (0.6B, 1.7B), Tencent Youtu-LLM-2B (completion, tools, embeddings)
- **Vision:** Gemma-4 (E2B, E4B), LFM2-VL, LFM2.5-VL (450M, 1.6B) (with Apple NPU), Qwen3.5 (0.8B, 2B)
- **Audio:** Gemma-4 (E2B, E4B) with native speech understanding
- **Transcription:** Whisper (Tiny/Base/Small/Medium with Apple NPU), Parakeet (CTC-0.6B/CTC-1.1B/TDT-0.6B-v3 with Apple NPU), Moonshine-Base
- **VAD & Diarization:** Silero VAD, PyAnnote segmentation-3.0, WeSpeaker speaker embeddings
- **Embeddings:** Nomic-Embed, Qwen3-Embedding

See the full list on [HuggingFace](https://huggingface.co/cactus-compute).

## Next Steps

- **[Engine API](cactus_engine.md)** -- Full inference API reference
- **[Graph API](cactus_graph.md)** -- Zero-copy computation graph for custom models
- **[Fine-tuning & Deployment](finetuning.md)** -- Convert and deploy custom fine-tunes
- **[Choose Your SDK](choose-sdk.md)** -- Help picking the right SDK for your project
