# Cactus

<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

[![Docs][docs-shield]][docs-url]
[![Website][website-shield]][website-url]
[![GitHub][github-shield]][github-url]
[![HuggingFace][hf-shield]][hf-url]
[![Reddit][reddit-shield]][reddit-url]
[![Blog][blog-shield]][blog-url]

A low-latency AI engine for mobile devices & wearables.

- **Fast & accurate:** fastest inference on ARM CPU, Cactus quants at 4-bit matches f16
- **Low RAM:** zero-copy memory mapping ensures 10x lower RAM use than other engines
- **Multimodal:** one engine for speech, vision, and language models
- **Cloud fallback:** automatically route requests to cloud models if needed
- **Model-Agnostic:** Custom PyTorch models can be exported to the Cactus runtime. 

```
┌─────────────────┐
│  Cactus Engine  │ ←── OpenAI-compatible APIs for text, speech, and vision.
└─────────────────┘     
         │
┌─────────────────┐
│  Cactus Graph   │ ←── Zero-copy computation graph ensures 10x lower RAM 
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
    "confidence": 0.8193,   // model confidence
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

## Build

```bash
cactus build --apple       # iOS/macOS
cactus build --android     # Android
cactus build --python      # Python
cactus build               # default static lib
```

## Bindings

- [Swift](/bindings/swift/)
- [Kotlin](/bindings/kotlin/)
- [Flutter](/bindings/flutter/)
- [React Native](/bindings/react-native/)
- [Python](/bindings/python/)
- [Rust](/bindings/rust/)

> **Model weights:** Pre-converted weights for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Benchmarks (CPU-only, no GPU)

- All weights INT4 quantised
- LFM: 1k-prefill / 100-decode, values are prefill tps / decode tps
- LFM-VL: 256px input, values are latency / decode tps
- Parakeet: 20s audio input, values are latency / decode tps
- Missing latency = no NPU support yet

| Device | LFM 1.2B | LFMVL 1.6B | Parakeet 1.1B | RAM |
|--------|----------|------------|---------------|-----|
| Mac M4 Pro | 582/100 | 0.2s/98 | 0.1s/900k+ | 76MB |
| iPad/Mac M3 | 350/60 | 0.3s/69 | 0.3s/800k+ | 70MB |
| iPhone 17 Pro | 327/48 | 0.3s/48 | 0.3s/300k+ | 108MB |
| iPhone 13 Mini | 148/34 | 0.3s/35 | 0.7s/90k+ | 1GB |
| Galaxy S25 Ultra | 255/37 | -/34 | -/250k+ | 1.5GB |
| Pixel 6a | 70/15 | -/15 | -/17k+ | 1GB |
| Galaxy A17 5G | 32/10 | -/11 | -/40k+ | 727MB |
| CMF Phone 2 Pro | - | - | - | - |
| Raspberry Pi 5 | 69/11 | 13.3s/11 | 4.5s/180k+ | 869MB |

## Supported Transcription Model

- STT: 20s audio input on Macbook Air M3 chip
- Benchmark dataset: internal evals with production users

| Model | Params | End2End ms | Latency ms | Decode toks/sec | NPU | RTF | WER |
|-------|--------|------------|------------|------------|-----|-----|-----|
| UsefulSensors/moonshine-base | 61M | 361.35 | 182 | 262 | yes | 0.0180 | 0.1395 |
| openai/whisper-tiny | 39M | 232.03 | 137.38 | 581 | yes | 0.0116 | 0.1860 |
| openai/whisper-base | 74M | 329.37 | 178.65 | 358 | yes | 0.0164 | 0.1628 |
| openai/whisper-small | 244M | 856.79 | 332.63 | 108 | yes | 0.0428 | 0.0930 |
| openai/whisper-medium | 769M | 2085.87 | 923.33 | 49 | yes | 0.1041 | 0.0930 |
| openai/whisper-large-v3 | 1.55B | 5994 | 2050 | 15.72 | no | 0.2992 | - |
| nvidia/parakeet-ctc-0.6b | 600M | 201.77 | 201.44 | 5214285 | yes | 0.0101 | 0.0930 |
| nvidia/parakeet-tdt-0.6b-v3 | 600M | 718.91 | 718.82 | 3583333 | yes | 0.0359 | 0.0465 |
| nvidia/parakeet-ctc-1.1b | 1.1B | 279.03 | 278.92 | 4562500 | yes | 0.0139 | 0.1628 |

## Supported LLMs

- Gemma weights are often **gated** on HuggingFace, needs tokens 
- Run `huggingface-cli login` and input your huggingface token

| Model | Features |                                                      
|-------|----------|
| google/gemma-3-270m-it | completion |
| google/functiongemma-270m-it | tools |
| google/gemma-3-1b-it | completion, gated |
| google/gemma-4-E2B-it | vision, audio, completion, tools, Apple NPU |
| google/gemma-4-E4B-it | vision, audio, completion, tools, Apple NPU |
| google/gemma-3n-E2B-it | completion, tools |
| google/gemma-3n-E4B-it | completion, tools |
| Qwen/Qwen3-0.6B | completion, tools, embed | 
| Qwen/Qwen3-Embedding-0.6B | embed | 
| Qwen/Qwen3.5-0.8B | vision, completion, tools, embed |
| Qwen/Qwen3-1.7B | completion, tools, embed | 
| Qwen/Qwen3.5-2B | vision, completion, tools, embed | 
| LiquidAI/LFM2.5-350M | completion, tools, embed |
| LiquidAI/LFM2-700M | completion, tools, embed |
| LiquidAI/LFM2-8B-A1B | completion, tools, embed |
| LiquidAI/LFM2.5-1.2B-Thinking | completion, tools, embed |
| LiquidAI/LFM2.5-1.2B-Instruct | completion, tools, embed |
| LiquidAI/LFM2-2.6B | completion, tools, embed |
| LiquidAI/LFM2-VL-450M | vision, txt & img embed, Apple NPU |
| LiquidAI/LFM2.5-VL-450M | vision, txt & img embed, Apple NPU |
| LiquidAI/LFM2.5-VL-1.6B | vision, txt & img embed, Apple NPU |
| tencent/Youtu-LLM-2B | completion, tools, embed |
| nomic-ai/nomic-embed-text-v2-moe | embed |

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
│  cactus run <model|path>             run a model (downloads if needed)         │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --platform cpu|apple              target accelerator (default: cpu)         │
│    --image <path>                    image file for VLM inference              │
│    --audio <path>                    audio file for audio chat                 │
│    --system <prompt>                 system prompt                             │
│    --prompt <text>                   send prompt immediately                   │
│    --thinking                        enable thinking/reasoning mode            │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force local convert+transpile fallback    │
│                                                                                │
│  cactus transcribe [model]           transcribe audio with a model             │
│    --file <audio.wav>                audio file to transcribe (required)       │
│    --language <code>                 language code (default: en)               │
│    --token <token>                   HuggingFace token (gated models)          │
│    --reconvert                       force reconversion from source            │
│                                                                                │
│  cactus download <model>             download a pre-built bundle               │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --platform cpu|apple              target accelerator (default: cpu)         │
│    --token <token>                   HuggingFace token                         │
│                                                                                │
│  cactus convert <model> [dir]        convert HuggingFace weights to CQ         │
│    --bits 1|2|3|4                    CQ quantization (default: 4)              │
│    --token <token>                   HuggingFace token                         │
│    --reconvert                       force build from source                   │
│    --lora <path>                     merge a LoRA adapter before converting    │
│                                                                                │
│  cactus transpile <model>            build a runnable bundle from CQ weights   │
│    --weights-dir <path>              path to CQ weights (default: lookup)      │
│    --task <auto|...>                 force task type (default: auto)           │
│    --artifact-dir <path>             bundle output (default: weights/<model>)  │
│                                                                                │
│  cactus serve [model]                OpenAI-compatible local HTTP server       │
│    --host <addr>                     bind address (default: 127.0.0.1)         │
│    --port <port>                     port (default: 8080)                      │
│                                                                                │
│  cactus list                         list local converted weights and bundles  │
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
│    --suite <name>                    run a single test suite (resolved         │
│                                      across components; e.g. performance       │
│                                      → kernels + graph, llm → engine)          │
│    --list                            list components and suites                │
│    --ios                             run on connected iPhone                   │
│    --android                         run on connected Android                  │
│    --enable-telemetry                send cloud telemetry (off by default)     │
│                                                                                │
│  cactus clean                        delete build artifacts                    │
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
