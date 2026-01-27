<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

Cross-platform & energy-efficient kernels, runtime and AI inference engine for mobile devices. 

```
┌─────────────────┐
│   Cactus FFI    │ ←── OpenAI compatible C API for integration (tools, RAG, cloud handoff)
└─────────────────┘
         │
┌─────────────────┐
│  Cactus Engine  │ ←── High-level transformer engine (NPU support, INT4/INT8/FP16/MIXED)
└─────────────────┘
         │
┌─────────────────┐
│ Cactus Models   │ ←── Implements SOTA models using Cactus Graphs 
└─────────────────┘
         │
┌─────────────────┐  
│  Cactus Graph   │ ←── Unified zero-copy computation graph (think NumPy for mobile)
└─────────────────┘
         │
┌─────────────────┐
│ Cactus Kernels  │ ←── Low-level ARM-specific SIMD operations (think CUDA for mobile)
└─────────────────┘
```

# Cactus Graph & Kernel
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

# Cactus Engine & FFI
```cpp
#include cactus.h

cactus_set_pro_key("");                  // email founders@cactuscompute.com for optional key

cactus_model_t model = cactus_init(
    "path/to/weight/folder",             // section to generate weigths below
    "txt/or/md/file/or/dir/with/many",   // nullptr if none, cactus does automatic fast RAG
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
    model,                               // model handle from cactus_init
    messages,                            // JSON array of chat messages
    response,                            // buffer to store response JSON
    sizeof(response),                    // size of response buffer
    options,                             // optional: generation options (nullptr for defaults)
    nullptr,                             // optional: tools JSON for function calling 
    nullptr,                             // optional: streaming callback fn(token, id, user_data)
    nullptr                              // optional: user data passed to callback
);
```
Example response from Gemma3-270m
```json
{
    "success": true,                    // when successfully generated locally
    "error": null,                      // returns specific errors if success = false
    "cloud_handoff": false,             // true when model is unconfident, simply route to cloud
    "response": "Hi there!",            // null when error is not null or cloud_handoff = true
    "function_calls": [],               // parsed to [{"name":"set_alarm","arguments":{"hour":"10","minute":"0"}}]
    "confidence": 0.8193,               // how confident the model is with its response
    "time_to_first_token_ms": 45.23,    // latency (time to first token)
    "total_time_ms": 163.67,            // total execution time
    "prefill_tps": 1621.89,             // prefill tokens per second
    "decode_tps": 168.42,               // decode tokens per second
    "ram_usage_mb": 245.67,             // current process RAM usage in MB
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

## Vision + Tool Calling

VLM models support tool calling for structured image analysis:

```cpp
const char* tools = R"([{
    "type": "function",
    "function": {
        "name": "analyze_image",
        "description": "Analyze image content",
        "parameters": {
            "type": "object",
            "properties": {
                "desc": {"type": "string"},
                "category": {"type": "string", "enum": ["animals", "nature", "objects"]}
            }
        }
    }
}])";

const char* messages = R"([{
    "role": "user",
    "content": "Analyze this image",
    "images": ["path/to/image.png"]
}])";

cactus_complete(model, messages, response, sizeof(response), nullptr, tools, nullptr, nullptr);
// response["function_calls"] contains structured output
```

See `python/example_vlm_tools.py` for a complete working example.



# Performance

- <sub>**Models:** LFM2-VL-450m & Whisper-Small</sub>
- <sub>**Precision:** Cactus smartly blends INT4, INT8 and F16 for all weights.</sub>
- <sub>**Decode** = toks/sec, **P/D** = prefill/decode, **VLM** = 256×256 image, **STT** = 30s audio</sub>
- <sub>**Cactus Pro**: Uses NPU for realtime and large context (Apple for now), scores are marked with *</sub>

| Device | Short Decode | 4k-P/D | VLM-TTFT | VLM-Dec | STT-TTFT | STT-Dec |
|--------|--------|--------|----------|---------|----------|---------|
| Mac M4 Pro | 170 | 989/150 | 0.2s/0.1s* | 168 | 1.0s/0.2s* | 92 |
| Mac M3 Pro | 140 | 890/123 | 0.3s/0.1s* | 149 | 1.5s/0.4s* | 81 |
| iPad/Mac M4 | 134 | 603/106 | 0.3s/0.1s* | 129 | 1.8s0.3s* | 70 |
| iPad/Mac M3 | 117 | 525/93 | 0.4s/0.1s* | 111 | 2.8s/0.7s* | 61 |
| iPhone 17 Pro | 126 | 428/84 | 0.5s/0.1s* | 120 | 3.0s/0.6s* | 80 |
| iPhone 16 Pro | 106 | 380/81 | 0.6s/0.2s* | 101 | 4.3s/0.7s* | 75 |
| iPhone 15 Pro | 90 | 330/75 | 0.7s/0.3s* | 92 | 4.5s/0.8s* | 70 |
| Galaxy S25 Ultra | 80 | 355/52 | 0.7s | 70 | 3.6s/- | 32 |
| Nothing 3 | 56 | 320/46 | 0.8s | 54 | 4.5s | 55 |
| Pixel 6a | 25 | 108/24 | 2.3s | 25 | 9.6 | 15 |
| Raspberry Pi 5 | 20 | 292/18 | 1.7s | 23 | 15s | 16 |


# Supported models

- <sub>Cactus smartly and compactly blends INT4, INT8 and F16 for all weights.</sub>
- <sub>You could still quantize everything with one precision, but mixed is optimal</sub>

| Model | Zipped Size | Completion | Tools | Vision | Embed | Speech | Pro |
|-------|------------------|------------|-------|--------|-------|--------|-----|
| google/gemma-3-270m-it | 252MB | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| google/functiongemma-270m-it | 252MB | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| openai/whisper-small | 283MB | ✗ | ✗ | ✗ | ✓ | ✓ | Apple |
| LiquidAI/LFM2-350M | 244MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2-VL-450M | 448MB | ✓ | ✓ | ✓ | ✓ | ✗ | Apple |
| nomic-ai/nomic-embed-text-v2-moe | 451MB | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Qwen/Qwen3-0.6B | 514MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Qwen/Qwen3-Embedding-0.6B | 514MB | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2-700M | 498MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| google/gemma-3-1b-it | 642MB | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| LiquidAI/LFM2.5-1.2B-Instruct | 474MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2-1.2B-RAG | 474MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2-1.2B-Tool | 474MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| openai/whisper-medium | 658MB | ✗ | ✗ | ✗ | ✓ | ✓ | Apple |
| LiquidAI/LFM2.5-VL-1.6B | 954MB | ✓ | ✓ | ✓ | ✓ | ✗ | Apple |
| Qwen/Qwen3-1.7B | 749MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |

# Using this repo on a Mac

```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
```

- <sub> `[model]` is a HuggingFace name from the table above (default: `google/gemma-3-270m-it`)</sub>
- <sub> Common flags: `--precision INT4|INT8|FP16` (default: INT4), `--token <hf_token>`</sub>
- <sub>Always run `source ./setup` in any new terminal.</sub>

| Command | Description |
|---------|-------------|
| `cactus run [model]` | Opens playground (auto downloads model) |
| `cactus download [model]` | Downloads model to `./weights` |
| `cactus convert [model] [dir]` | Converts model, supports LoRA merging (`--lora <path>`) |
| `cactus build` | Builds for ARM (`--apple` or `--android`) |
| `cactus test` | Runs tests (`--ios` / `--android`, `--model [name/path]`) |
| `cactus clean` | Removes build artifacts |
| `cactus --help` | Shows all commands and flags |

# Using in your apps 

- [Python for Mac](/python/)
- [React Native SDK](https://github.com/cactus-compute/cactus-react-native)
- [Swift Multiplatform SDK](https://github.com/mhayes853/swift-cactus)
- [Kotlin Multiplatform SDK](https://github.com/cactus-compute/cactus-kotlin)
- [Flutter SDK](https://github.com/cactus-compute/cactus-flutter)
- [Rust SDK](https://github.com/mrsarac/cactus-rs)

# Try demo apps 

- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)

# Maintaining Organisations
1. [Cactus Compute, Inc](https://cactuscompute.com/) 
2. [UCLA's BruinAI](https://bruinai.org/) 