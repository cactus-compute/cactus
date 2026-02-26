# Cactus

<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

[![Docs][docs-shield]][docs-url]
[![Website][website-shield]][website-url]
[![GitHub][github-shield]][github-url]
[![HuggingFace][hf-shield]][hf-url]
[![Reddit][reddit-shield]][reddit-url]
[![Blog][blog-shield]][blog-url]

A hybrid low-latency energy-efficient AI engine for mobile devices & wearables.

```
┌─────────────────┐
│  Cactus Engine  │ ←── OpenAI-compatible APIs for all major languages
└─────────────────┘     Chat, vision, STT, RAG, tool call, cloud handoff
         │
┌─────────────────┐
│  Cactus Graph   │ ←── Zero-copy computation graph (PyTorch for mobile)
└─────────────────┘     Custom models, optimised for RAM & quantisation
         │
┌─────────────────┐
│ Cactus Kernels  │ ←── ARM SIMD kernels (Apple, Snapdragon, Exynos, etc)
└─────────────────┘     Custom attention, KV-cache quant, chunked prefill
```

## Quick Demo 

- Step 1: `brew install cactus-compute/cactus/cactus`
- Step 2: `cactus transcribe` or `cactus run` 

## Cactus Engine

```cpp
#include cactus.h

cactus_model_t model = cactus_init(
    "path/to/weight/folder",
    "path to txt or dir of txts for auto-rag",
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
    nullptr           // user data
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
#include cactus.h

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

## API & SDK References

| Reference | Language | Description |
|-----------|----------|-------------|
| [Engine API](cactus_engine.md) | C | Chat completion, streaming, tool calling, transcription, embeddings, RAG, vision, VAD, vector index, cloud handoff |
| [Graph API](cactus_graph.md) | C++ | Tensor operations, matrix multiplication, attention, normalization, activation functions |
| [Python SDK](/python/) | Python | Mac, Linux |
| [Swift SDK](/apple/) | Swift | iOS, macOS, tvOS, watchOS, Android |
| [Kotlin SDK](/android/) | Kotlin | Android, iOS (via KMP) |
| [Flutter SDK](/flutter/) | Dart | iOS, macOS, Android |
| [Rust SDK](/rust/) | Rust | Mac, Linux |
| [React Native](https://github.com/cactus-compute/cactus-react-native) | JavaScript | iOS, Android |

## Benchmarks

- All weights INT4 quantised
- LFM: 1k-prefill / 100-decode, values are prefill tps / decode tps
- LFM-VL: 256px input, values are latency / decode tps
- Parakeet: 30s audio input, values are latency / decode tps
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

## Roadmap

| Date | Status | Milestone |
|------|--------|-----------|
| Sep 2025 | Done | Released v1 |
| Oct 2025 | Done | Chunked prefill, KVCache Quant (2x prefill) |
| Nov 2025 | Done | Cactus Attention (10 & 1k prefill = same decode) |
| Dec 2025 | Done | Team grows to +6 Research Engineers |
| Jan 2026 | Done | Apple NPU/RAM, 5-11x faster iOS/Mac |
| Feb 2026 | Done | Hybrid inference, INT4, lossless Quant (1.5x) |
| Mar 2026 | Coming | Qualcomm/Google NPUs, 5-11x faster Android |
| Apr 2026 | Coming | Mediatek/Exynos NPUs, Cactus@ICLR |
| May 2026 | Coming | Kernel→C++, Graph/Engine→Rust, Mac GPU & VR |
| Jun 2026 | Coming | Torch/JAX model transpilers |
| Jul 2026 | Coming | Wearables optimisations, Cactus@ICML |
| Aug 2026 | Coming | Orchestration |
| Sep 2026 | Coming | Full Cactus paper, chip manufacturer partners |

## Using this repo

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│ Step 0: if on Linux (Ubuntu/Debian)                                          │
│ sudo apt-get install python3 python3-venv python3-pip cmake                  │
│   build-essential libcurl4-openssl-dev                                       │
│                                                                              │
│ Step 1: clone and setup                                                      │
│ git clone https://github.com/cactus-compute/cactus && cd cactus              │
│ source ./setup                                                               │
│                                                                              │
│ Step 2: use the commands                                                     │
│──────────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  cactus auth                         manage Cloud API key                    │
│    --status                          show key status                         │
│    --clear                           remove saved key                        │
│                                                                              │
│  cactus run <model>                  opens playground (auto downloads)       │
│    --precision INT4|INT8|FP16        quantization (default: INT4)            │
│    --token <token>                   HF token (gated models)                 │
│    --reconvert                       force reconversion from source          │
│                                                                              │
│  cactus transcribe [model]           live mic transcription (parakeet-1.1b)  │
│    --file <audio.wav>                transcribe file instead of mic          │
│    --precision INT4|INT8|FP16        quantization (default: INT4)            │
│    --token <token>                   HF token (gated models)                 │
│    --reconvert                       force reconversion from source          │
│                                                                              │
│  cactus download <model>             downloads model to ./weights            │
│    --precision INT4|INT8|FP16        quantization (default: INT4)            │
│    --token <token>                   HuggingFace API token                   │
│    --reconvert                       force reconversion from source          │
│                                                                              │
│  cactus convert <model> [dir]        convert model, supports LoRA merge      │
│    --precision INT4|INT8|FP16        quantization (default: INT4)            │
│    --lora <path>                     LoRA adapter to merge                   │
│    --token <token>                   HuggingFace API token                   │
│                                                                              │
│  cactus build                        build for ARM → build/libcactus.a       │
│    --apple                           Apple (iOS/macOS)                       │
│    --android                         Android                                 │
│    --flutter                         Flutter (all platforms)                 │
│    --python                          shared lib for Python FFI               │
│                                                                              │
│  cactus test                         run unit tests and benchmarks           │
│    --model <model>                   default: LFM2-VL-450M                   │
│    --transcribe_model <model>        default: moonshine-base                 │
│    --benchmark                       use larger models                       │
│    --precision INT4|INT8|FP16        regenerate weights with precision       │
│    --reconvert                       force reconversion from source          │
│    --no-rebuild                      skip building library                   │
│    --only <test>                     specific test (llm, vlm, stt, etc)      │
│    --ios                             run on connected iPhone                 │
│    --android                         run on connected Android                │
│                                                                              │
│  cactus clean                        remove all build artifacts              │
│  cactus --help                       show all commands and flags             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Supported Models                                                                                                                                                     
                                                                                                                                                                          
| Model | Features |                                                                                                                                             
|-------|----------|                                                                                                                                             
| google/gemma-3-270m-it | completion |                                                                                                                         
| google/functiongemma-270m-it | completion, tools |                                                                                                            
| LiquidAI/LFM2-350M | completion, tools, embed |                                                                                                               
| Qwen/Qwen3-0.6B | completion, tools, embed |                                                                                                                  
| LiquidAI/LFM2-700M | completion, tools, embed |                                                                                                               
| LiquidAI/LFM2-8B-A1B | completion, tools, embed |                                                                                                                
| google/gemma-3-1b-it | completion |                                                                                                                           
| LiquidAI/LFM2.5-1.2B-Thinking | completion, tools, embed |                                                                                                    
| LiquidAI/LFM2.5-1.2B-Instruct | completion, tools, embed |                                                                                                      
| Qwen/Qwen3-1.7B | completion, tools, embed | 
| LiquidAI/LFM2-2.6B | completion, tools, embed |                                                                                                                
| LiquidAI/LFM2-VL-450M | vision, txt & img embed, Apple NPU |                                                                                                            
| LiquidAI/LFM2.5-VL-1.6B | vision, txt & img embed, Apple NPU |                                                                                                               
| UsefulSensors/moonshine-base | transcription, speech embed |                                                                                                         
| openai/whisper-small | transcription, speech embed, Apple NPU |                                                                                                                 
| openai/whisper-medium | transcribe, speech embed, Apple NPU |
| nvidia/parakeet-ctc-0.6b | transcribe, speech embed, Apple NPU |
| nvidia/parakeet-ctc-1.1b | transcribe, speech embed, Apple NPU |
| snakers4/silero-vad | vad |
| nomic-ai/nomic-embed-text-v2-moe | embed |                                                                                                                    
| Qwen/Qwen3-Embedding-0.6B | embed | 

## Maintaining Organisations

1. [Cactus Compute, Inc. (YC S25)](https://cactuscompute.com/)
2. [UCLA's BruinAI](https://bruinai.org/)
3. [Char (YC S25)](https://char.com/)
4. [Yale's AI Society](https://www.yale-ai.org/team)
5. [National Unoversity of Singapore's AI Society](https://www.nusaisociety.org/)
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
[blog-url]: https://cactus-compute.github.io/cactus/blog/README/
