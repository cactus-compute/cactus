# Cactus

<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

[![Docs][docs-shield]][docs-url]
[![Website][website-shield]][website-url]
[![GitHub][github-shield]][github-url]
[![HuggingFace][hf-shield]][hf-url]
[![Reddit][reddit-shield]][reddit-url]
[![Blog][blog-shield]][blog-url]

A hybrid edge-cloud AI engine for mobile devices & wearables.

```
┌─────────────────┐
│  Cactus Engine  │ ←── OpenAI-compatible APIs for text, speech, and vision.
└─────────────────┘     
         │
┌─────────────────┐
│  Cactus Graph   │ ←── Zero-copy computation graph 
└─────────────────┘     
         │
┌─────────────────┐
│ Cactus Kernels  │ ←── Fastest ARM SIMD kernels (Apple, Samsung, Pixel, etc)
└─────────────────┘     
         │
┌─────────────────┐
│ Cactus Quants   │ ←── Cactus Quants at 4-bit uniform matches f16.
└─────────────────┘  
         │
┌─────────────────┐
│Cactus Transpiler│ ←── Transpiles custom PyTorch model to Cactus.
└─────────────────┘
```

## Quick Demo (Mac)

- Step 1: `brew install cactus-compute/cactus/cactus`
- Step 2: `cactus run`

## Cactus Engine

```cpp
#include "cactus_engine.h"

cactus_model_t model = cactus_init(
    "path/to/weight/folder",
    "path to txt or dir of txts for auto-rag",
    false
);

const char* messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Henry Ndubuaku"}
])";

const char* options = R"({
    "max_tokens": 50,
    "stop_sequences": ["<|im_end|>"]
})";

char response[4096];
int result = cactus_complete(
    model,            // model handle
    messages,         // JSON chat messages
    response,         // response buffer
    sizeof(response), // buffer size
    options,          // generation options
    nullptr,          // tools JSON
    nullptr,          // streaming callback
    nullptr,          // user data
    nullptr,          // pcm audio buffer
    0                 // pcm buffer size
);
```
Example response from Gemma3-270m
```json
{
    "success": true,        // generation succeeded
    "error": null,          // error details if failed
    "cloud_handoff": false, // true if cloud model used
    "response": "Hi there!",
    "function_calls": [],   // parsed tool calls
    "segments": [],         // transcription segments (empty for chat)
    "confidence": 0.8193,   // model confidence
    "confidence_threshold": 0.7, // resolved handoff threshold (model-dependent)
    "time_to_first_token_ms": 45.23,
    "total_time_ms": 163.67,
    "prefill_tps": 1621.89,
    "decode_tps": 168.42,
    "ram_usage_mb": 245.67,
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

## Cactus Graph

```cpp
#include "cactus_graph.h"

CactusGraph graph;
auto a = graph.input({2, 3}, Precision::FP16);
auto b = graph.input({3, 4}, Precision::INT8);

auto x1 = graph.matmul(a, b, false);
auto x2 = graph.transpose(x1);
auto result = graph.matmul(b, x2, true);

float a_data[6] = {1.1f, 2.3f, 3.4f, 4.2f, 5.7f, 6.8f};
float b_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

graph.set_input(a, a_data, Precision::FP16);
graph.set_input(b, b_data, Precision::INT8);

graph.execute();
void* output_data = graph.get_output(result);

graph.hard_reset(); 
```

## Benchmarks

- LLM: Gemma-4-E2B-CQ4, 1k-prefill tps / 100-decode tps
- VLM: Gemma-4-E2B-CQ4  256px input, latency / decode tps
- Transcribe: Parakeet-TDT-0.6B-CQ4, 20s audio, latency / decode tps
- Missing latency == no NPU support for device

| Device | LLM | VLM | Transcribe | 1k-Context RAM |
|--------|----------|------------|---------------|-----|
| Mac M4 Pro | 2619tps / 139tps | 0.4s / 160tps | 0.1s / 11Mtps | 1327 MB |
| Mac M3 Pro | 390 / 26 | 2.76s / 28.06 | 0.32s / 2.29M | 1376 MB |
| iPhone 17 Pro | - | - | - | - |
| iPhone 13 Mini | - | - | - | - |
| Galaxy S26 | 248 / 21 | - / 16 | - / 5.7M | - |
| Galaxy A17 5G | - | - | - | - |
| Pixel 10 Pro | - | - | - | - |
| Pixel 6a | - | - | - | - |
| Raspberry Pi 5 | - | - | - | - |


## Supported Models

- Any HuggigFace model can be converted using `cactus convert [HF-Name]`, though experimental.
- Liquid, Gemma. whisper. parakeet and Qwen model families are especially tested. 
- Some models have been pre-uploaded [here](https://huggingface.co/Cactus-Compute), just run `cactus download [HF-Name]`.
- `cactus run [HF-Name]` albeit first downloads or convert the model if not found. 

## Learn More

| Reference | Language | Description |
|-----------|----------|-------------|
| [Cactus Engine](/docs/cactus_engine.md) | C | Chat completion, streaming, tool calling, transcription, embeddings, RAG, vision, vector index, cloud handoff |
| [Cactus Graph](/docs/cactus_graph.md) | C++ | Tensor operations, matrix multiplication, attention, normalization, activation functions |
| [Cactus Kernels](/docs/cactus_kernels.md) | C++ | ARM NEON SIMD kernels for matmul, attention, convolution, quantization, DSP, image processing |
| [Cactus Quants](/docs/cactus_quants.md) | C++ | Rotation-and-codebook quantization from 4-bit to 1-bit for all weight tensors |
| [Cactus Hybrid](/docs/cactus_hybrid.md) | C/Python | Route hard queries to the cloud automatically based on local model confidence |
| [Cactus Transpiler](/docs/cactus_transpiler.md) | Python | Convert any PyTorch model to a Cactus runtime graph for on-device inference |
| [Python Package](/python/) | Python | Python package and CLI |

## Bindings

- [Swift](/bindings/swift/)
- [Kotlin](/bindings/kotlin/)
- [Flutter](/bindings/flutter/)
- [React Native](/bindings/react-native/)
- [Python](/bindings/python/)
- [Rust](/bindings/rust/)

## Using this repo

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│ Step 0: if on Linux (Ubuntu/Debian)                                            │
│ sudo apt-get install python3.12 python3.12-venv python3-pip cmake              │
│   build-essential libcurl4-openssl-dev                                         │
│                                                                                │
│ Step 1: clone and setup                                                        │
│ git clone https://github.com/cactus-compute/cactus && cd cactus                │
│ source ./setup                                                                 │
│                                                                                │
│ Step 2: use the commands                                                       │
│────────────────────────────────────────────────────────────────────────────────│
│                                                                                │
│  cactus auth                         manage cloud API key                      │
│    --status                          show key status                           │
│    --clear                           remove saved key                          │
│                                                                                │
│  cactus run [model|path]             run a model (downloads if needed)         │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --weights general|apple           weights bundle variant (default: general) │
│    --backend auto|cpu|metal          inference backend (default: auto)         │
│    --image <path>                    image file for VLM inference              │
│    --audio <path>                    audio file for audio chat                 │
│    --system <prompt>                 system prompt                             │
│    --prompt <text>                   send prompt immediately                   │
│    --thinking                        enable thinking/reasoning mode            │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local rebuild from source           │
│                                                                                │
│  cactus transcribe [model]           live microphone transcription with a model│
│    --file <audio.wav>                audio file to transcribe (WAV)            │
│    --language <code>                 language code (default: en)               │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --weights general|apple           weights bundle variant (default: general) │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local rebuild from source           │
│                                                                                │
│  cactus download [model]             get a bundle (prebuilt, else build)       │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --weights general|apple           weights bundle variant (default: general) │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local rebuild from source           │
│                                                                                │
│  cactus convert <model> [dir]        HuggingFace -> runnable cactus bundle     │
│                                      (CQ weights + runtime graph)              │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --weights general|apple           weights bundle variant (default: general) │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local rebuild from source           │
│    --lora <path>                     merge a LoRA adapter before converting    │
│    --weights-only                    stop after CQ weights (skip the graph)    │
│    --artifact-dir <path>             bundle output (default: weights/<model>)  │
│                                                                                │
│  cactus serve [model]                OpenAI-compatible local HTTP server       │
│    --host <addr>                     bind address (default: 127.0.0.1)         │
│    --port <port>                     port (default: 8080)                      │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --weights general|apple           weights bundle variant (default: general) │
│    --backend auto|cpu|metal          inference backend (default: auto)         │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local rebuild from source           │
│    --no-cloud-handoff                disable automatic cloud handoff           │
│    --confidence-threshold <0..1>     handoff to cloud below this confidence    │
│    --cloud-timeout-ms <n>            max wait for cloud handoff                │
│                                                                                │
│  cactus code                         run the AI coding agent (TUI / print)     │
│    --serve-model <id>                auto-start a server with this model       │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --weights general|apple           weights bundle variant (default: general) │
│    --backend auto|cpu|metal          inference backend (default: auto)         │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local rebuild from source           │
│    --host <addr>                     server address (default: 127.0.0.1)       │
│    --port <port>                     server port (default: 8080)               │
│    --no-serve                        require a running server (no auto-start)  │
│    --no-cloud-handoff                disable automatic cloud handoff           │
│    --confidence-threshold <0..1>     handoff to cloud below this confidence    │
│    --cloud-timeout-ms <n>            max wait for cloud handoff                │
│    -- <args...>                      pass remaining args to the agent          │
│                                                                                │
│  cactus list                         list downloaded models                    │
│                                                                                │
│  cactus build                        build cactus libraries                    │
│    --apple                           Apple (iOS/macOS)                         │
│    --android                         Android                                   │
│    --python                          shared lib for Python FFI                 │
│                                                                                │
│  cactus test                         run the test suite                        │
│    --component <name>                kernels | graph | engine | all            │
│                                      (default: all)                            │
│    --model <hf-id>                   default: LiquidAI/LFM2-VL-450M            │
│    --transcription-model <hf-id>     default: openai/whisper-base              │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --weights general|apple           weights bundle variant (default: general) │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local rebuild of test models        │
│    --suite <name>                    run a single test suite by name           │
│                                      (resolved across components,              │
│                                      e.g. llm → engine)                        │
│    --list                            list components and suites                │
│    --ios                             run on connected iPhone                   │
│    --android                         run on connected Android                  │
│    --enable-telemetry                send cloud telemetry (off by default)     │
│                                                                                │
│  cactus clean                        delete build artifacts, weights, venv     │
│  cactus --help                       show this help                            │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Maintaining Organisations

1. [Cactus Compute, Inc. (YC S25)](https://cactuscompute.com/)
2. [UCLA's BruinAI](https://bruinai.org/)
3. [Char (YC S25)](https://char.com/)
4. [Yale's AI Society](https://www.yale-ai.org/team)
5. [National University of Singapore's AI Society](https://www.nusaisociety.org/)
6. [UC Irvine's AI@UCI](https://aiclub.ics.uci.edu/)
7. [Imperial College's AI Society](https://www.imperialcollegeunion.org/csp/1391)
8. [University of Pennsylvania's AI@Penn](https://ai-at-penn-main-105.vercel.app/)
9. [University of Michigan Ann-Arbor MSAIL](https://msail.github.io/)
10. [University of Colorado Boulder's AI Club](https://www.cuaiclub.org/)

## Citation 

If you use Cactus in your research, please cite it as follows:

```bibtex
@software{cactus,
  title        = {Cactus: AI Inference Engine for Phones & Wearables},
  author       = {Ndubuaku, Henry and Cactus Team},
  url          = {https://github.com/cactus-compute/cactus},
  year         = {2025}
}
```

**N/B:** Scroll all the way up and click the shields link for resources!

[docs-shield]: https://img.shields.io/badge/Docs-555?style=for-the-badge&logo=readthedocs&logoColor=white
[docs-url]: https://cactus-compute.github.io/cactus/

[website-shield]: https://img.shields.io/badge/Website-555?style=for-the-badge&logo=safari&logoColor=white
[website-url]: https://cactuscompute.com/

[github-shield]: https://img.shields.io/badge/GitHub-555?style=for-the-badge&logo=github&logoColor=white
[github-url]: https://github.com/cactus-compute/cactus

[hf-shield]: https://img.shields.io/badge/HuggingFace-555?style=for-the-badge&logo=huggingface&logoColor=white
[hf-url]: https://huggingface.co/Cactus-Compute

[reddit-shield]: https://img.shields.io/badge/Reddit-555?style=for-the-badge&logo=reddit&logoColor=white
[reddit-url]: https://www.reddit.com/r/cactuscompute/

[blog-shield]: https://img.shields.io/badge/Blog-555?style=for-the-badge&logo=hashnode&logoColor=white
[blog-url]: https://cactuscompute.com/blog
