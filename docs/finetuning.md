---
title: "Deploying Unsloth Fine-Tunes to Cactus for Phones"
description: "Step-by-step guide to training LoRA fine-tunes with Unsloth and deploying them to iOS and Android devices using Cactus inference engine."
keywords: ["fine-tuning", "LoRA", "Unsloth", "mobile deployment", "on-device AI", "iOS", "Android", "Qwen", "Gemma"]
---

# Deploying Unsloth Fine-Tunes to Cactus for Phones

- Cactus is an inference engine for mobile devices, macs and ARM chips like Raspberry Pi.
- At INT8, Cactus runs `Qwen3-0.6B` and `LFM2-1.2B` at `60-70 toks/sec` on iPhone 17 Pro, `13-18 toks/sec` on budget Pixel 6a.
- INT4 quantization provides ~50% memory reduction with minimal quality loss.
- Task-Specific INT8 tunes of `Gemma3-270m` hit `150 toks/sec` on iPhone 17 Pro and `23 toks/sec` on Raspberry Pi. 

## Quick Start

### 1. Train (Google Colab / GPU)

Use the provided notebook or your own Unsloth training script:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# ... train with SFTTrainer ...

# Save adapter
model.save_pretrained("my-lora-adapter")
tokenizer.save_pretrained("my-lora-adapter")

# Push to Hub (optional)
model.push_to_hub("username/my-lora-adapter")
```

### 2. Setup Cactus

```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
```
<img src="../assets/setup.png" alt="Logo" style="border-radius: 30px; width: 70%;">

### 3. Convert for Cactus

```bash
# From local adapter: Use the correct base model!
cactus convert Qwen/Qwen3-0.6B ./my-qwen3-0.6b --lora ./my-lora-adapter 

# From HuggingFace Hub: Use the correct base model!
cactus convert Qwen/Qwen3-0.6B ./my-qwen3-0.6b  --lora username/my-lora-adapter 

```
<img src="../assets/lora.png" alt="Logo" style="border-radius: 30px; width: 70%;">

### 4. Run

Test your model on Mac:

```bash
cactus run ./my-qwen3-0.6b
```
<img src="../assets/run.png" alt="Logo" style="border-radius: 30px; width: 70%;">

### 5. Use in iOS/macOS App

Build the native library:

```bash
cactus build --apple
```
```bash
Build complete!
Total time: 58 seconds
Static libraries:
  Device: /Users/henry/Desktop/cactus/apple/libcactus-device.a
  Simulator: /Users/henry/Desktop/cactus/apple/libcactus-simulator.a
XCFrameworks:
  iOS: /Users/henry/Desktop/cactus/apple/cactus-ios.xcframework
  macOS: /Users/henry/Desktop/cactus/apple/cactus-macos.xcframework
Apple build complete!
(venv) henry@Henrys-MacBook-Air cactus % 
```

Link `cactus-ios.xcframework` to your Xcode project, then:

```swift
import Foundation

// Load model from app bundle
let modelPath = Bundle.main.path(forResource: "my-model", ofType: nil)!
let model = cactus_init(modelPath, nil)

// Run completion
let messages = "[{\"role\":\"user\",\"content\":\"Hello!\"}]"
var response = [CChar](repeating: 0, count: 4096)
cactus_complete(model, messages, &response, response.count, nil, nil, nil, nil)
print(String(cString: response))

// Cleanup
cactus_destroy(model)
```
<img src="../assets/app.png" alt="Logo" style="border-radius: 30px; width: 20%;">

You can now build iOS apps using the following code, 
but to see performance on any device while testing,
run cactus tests by plugging any iphone to your Mac then running:

```bash
cactus test --<model-path-or-name> --ios 
```

Cactus demo apps will eventually expand to using your custom fine-tunes.
Also, `cactus run` will allow plugging in a phone, 
such that the interactive session use the phone chips,
this way you can test before fully building out your apps.

### 6. Use in Android App

Build the native library:

```bash
cactus build --android
```
```bash
Build complete!
Shared library location: /Users/henry/Desktop/cactus/android/libcactus.so
Static library location: /Users/henry/Desktop/cactus/android/libcactus.a
Android build complete!
(venv) henry@Henrys-MacBook-Air cactus % 
```

Copy `libcactus.so` to `app/src/main/jniLibs/arm64-v8a/`, then:

```kotlin
class CactusWrapper {
    init { System.loadLibrary("cactus") }

    external fun init(modelPath: String, contextSize: Long, corpusDir: String?): Long
    external fun complete(model: Long, messagesJson: String, bufferSize: Int): String
    external fun destroy(model: Long)
}

// Usage
val cactus = CactusWrapper()
val model = cactus.init("/data/local/tmp/my-model", 2048, null)
val response = cactus.complete(model, """[{"role":"user","content":"Hello!"}]""", 4096)
cactus.destroy(model)
```

You can now build ANdroid apps using the following code, 
but to see performance on any device while testing,
run cactus tests by plugging any android phone to your Mac then running:

```bash
cactus test --<model-path-or-name> --android 
```

Cactus demo apps will eventually expand to using your custom fine-tunes.
Also, `cactus run` will allow plugging in a phone, 
such that the interactive session use the phone chips,
this way you can test before fully building out your apps.

## Resources

- Supported Base Models: `Qwen3, Gemma3, LFM2, SmolLM2` 
- Full API reference: [Cactus Engine](https://github.com/cactus-compute/cactus/blob/main/docs/cactus_engine.md)
- Learn more and report bugs: [Cactus](https://github.com/cactus-compute/cactus/tree/main)

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference for inference, streaming, and tool calling
- [Runtime Compatibility](/docs/compatibility.md) — Ensure your weights match your Cactus runtime version
- [Python SDK](/python/) — Use fine-tuned models from Python
- [Swift SDK](/apple/) — Deploy fine-tuned models in iOS/macOS apps
- [Kotlin/Android SDK](/android/) — Deploy fine-tuned models in Android apps
