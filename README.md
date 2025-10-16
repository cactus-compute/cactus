<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

Energy-efficient kernels & inference engine for phones. 

## Why Cactus?
- Phones run on battery, GPUs drain energy and heat the devices. 
- 70% of phones today don't ship NPUs which most frameworks optimse for. 
- Cactus is optimsed for old and new ARM-CPU first, with NPU/DSP/ISP coming.
- Fast on all phones with less battery drain and heating.

## Performance (CPU only)

- Speed for various sizes can be estimated proportionally
- INT4 wiil give 30% gains when merged 
- GPUs yield gains but drain battery, will be passed on for NPUs

| Device                        |  Qwen3-INT8-600m (toks/sec) |  
|:------------------------------|:------------------------:|
| iPhone 17 Pro                 | 74 |
| Galaxy S25 Ultra / 16 Pro     | 58 |
| iPhone 16 / Galaxy S25 / Nothing 3 | 52 |
| iPhone 15 Pro                 | 48 |
| iPhone 14 Pro / OnePlus 13 5G | 47 |
| Galaxy S24 Ultra / iPhone 15  | 42 |
| OnePlus Open / Galaxy S23     | 41 |
| iPhone 13 Pro / OnePlus 12    | 38 |
| iPhone 13 mini / Redmi K70 Ultra / Xiaomi 13 / OnePlus 11 | 27 |
| Pixel 6a / Nothing 3a / iPhone X / Galaxy S21 | 16 |

## File Size Comparison

| Format | Size (Qwen3-0.6B-INT8) |
|--------|------------------------|
| Cactus | 370-420 MB |
| ONNX/TFLite/MLX | 600 MB |
| GGUF | 800 MB |
| Executorch | 944 MB |

## Battery drain

 - Newer devices have bigger battery 
 - NPUs are designed for less drain (2-10x)
 - Apple Intelligence drain 0.6 percent/min on iPhone 16 Pro Max

| Device                        |  Qwen3-INT8-600m (percent/min) |  
|:------------------------------|:------------------------:|
| OnePlus 13 5G | 0.33 |
| Redmi K70 Ultra / OnePlus 12 | 0.41 |
| Galaxy S25 Ultra / Iphone 17 Pro / Nothing 3 | 0.44 |
| Galaxy S24 Ultra / Nothing 3a / Pixel 6a | 0.48 |
| iPhone 16 Pro Max / Xiaomi 13 | 0.50  |

## Design 
```
┌─────────────────┐
│   Cactus FFI    │ ←── OpenAI compatible C API for integration  
└─────────────────┘
         │
┌─────────────────┐
│  Cactus Engine  │ ←── High-level transformer engine
└─────────────────┘
         │
┌─────────────────┐  
│  Cactus Graph   │ ←── Unified zero-copy computation graph 
└─────────────────┘
         │
┌─────────────────┐
│ Cactus Kernels  │ ←── Low-level ARM-specific SIMD operations
└─────────────────┘
```

## Cactus Graph & Kernels
Cactus Graph is a general numerical computing framework that runs on Cactus Kernels.
Great for implementing custom models and scientific computing, like JAX for phones.

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

## Cactus Engine & APIs
Cactus Engine is a transformer inference engine built on top of Cactus Graphs.
It is abstracted via Cactus Foreign Function Interface APIs.
Header files are self-documenting but documentation contributions are welcome.

```cpp
#include cactus.h

const char* model_path = "path/to/weight/folder";
cactus_model_t model = cactus_init(model_path, 2048);

const char* messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "/nothink My name is Henry Ndubuaku"}
])";

const char* options = R"({
    "max_tokens": 50,
    "stop_sequences": ["<|im_end|>"]
})";

char response[1024];
int result = cactus_complete(model, messages, response, sizeof(response), options, nullptr, nullptr, nullptr);
```

With tool support:
```cpp
const char* tools = R"([
    {
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                        "required": true
                    }
                },
                "required": ["location"]
            }
        }
    }
])";

int result = cactus_complete(model, messages, response, sizeof(response), options, tools, nullptr, nullptr);
```

## Using Cactus in your apps
Cactus SDKs run 500k+ weekly inference tasks in production today, try them!

<a href="https://github.com/cactus-compute/cactus-flutter" target="_blank">
  <img alt="Flutter" src="https://img.shields.io/badge/Flutter-grey.svg?style=for-the-badge&logo=Flutter&logoColor=white">
</a> <a href="https://github.com/cactus-compute/cactus-react" target="_blank">
  <img alt="React Native" src="https://img.shields.io/badge/React%20Native-grey.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB">
</a> <a href="https://github.com/cactus-compute/cactus-kotlin" target="_blank">
  <img alt="Kotlin" src="https://img.shields.io/badge/Kotlin_MP-grey.svg?style=for-the-badge&logo=kotlin&logoColor=white">
</a>

## Getting started
<a href="https://cactuscompute.com/docs" target="_blank">
  <img alt="Documentation" src="https://img.shields.io/badge/Documentation-4A90E2?style=for-the-badge&logo=gitbook&logoColor=white">
</a> <a href="https://discord.gg/bNurx3AXTJ" target="_blank">
  <img alt="Discord" src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white">
</a>

## Demo 
<a href="https://apps.apple.com/gb/app/cactus-chat/id6744444212" target="_blank">
  <img alt="Download iOS App" src="https://img.shields.io/badge/Try_iOS_Demo-grey?style=for-the-badge&logo=apple&logoColor=white">
</a> <a href="https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp&pcampaignid=web_share" target="_blank">
  <img alt="Download Android App" src="https://img.shields.io/badge/Try_Android_Demo-grey?style=for-the-badge&logo=android&logoColor=white">
</a>

## Using this repo
You can run these codes directly on M-series Macbooks since they are ARM-based.
Vanilla M3 CPU-only can run Qwen3-600m-INT8 at 60-70 toks/sec, just run the following: 

```bash
./tests/run.sh # chmod +x first time
```

## Generating weights from HuggingFace 
Use any of the following (270m, 360m, 600m, 1B, 1.7B activated params):
```bash
# Language models
python3 tools/convert_hf.py google/gemma-3-270m-it weights/gemma3-270m-i8/ --precision INT8
python3 tools/convert_hf.py HuggingFaceTB/SmolLM2-360m-Instruct weights/smollm2-360m-i8/ --precision INT8
python3 tools/convert_hf.py Qwen/Qwen3-0.6B weights/qwen3-600m-i8/ --precision INT8
python3 tools/convert_hf.py google/gemma-3-ib-it weights/gemma3-1b-i8/ --precision INT8
python3 tools/convert_hf.py Qwen/Qwen3-1.7B weights/qwen3-1.7B-i8/ --precision INT8
python3 tools/convert_hf.py HuggingFaceTB/SmolLM2-1.7B-Instruct weights/smollm2-1.7B-i8/ --precision INT8

# Embedding models
python3 tools/convert_hf.py Qwen/Qwen3-Embedding-0.6B weights/qwen3-embed-600m-i8/ --precision INT8
python3 tools/convert_hf.py nomic-ai/nomic-embed-text-v2-moe weights/nomic-i8/ --precision INT8
```

Simply replace the weight path `tests/test_engine.cpp` with your choice.

## Roadmap:
- Llama, LFM, SmolVLM, Whisper, Kitten, Neuphonic
- Python tools for porting any Torch/JAX to cactus
- GPTQ & NPU/DSP/ISP for high-end phones 

## Limitations
While Cactus can be used for all Apple devices including Macbooks, for computers/AMD/Intel/Nvidia generally, 
please use HuggingFace, Llama.cpp, Ollama, vLLM, MLX. They're built for those, support x86, and are all great! 

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
