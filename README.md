<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

Fastest kernels & energy-efficient inference engine for all phones: old, new, budget or high-end. 

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

cactus_model_t model = cactus_init("path/to/weight/folder", 2048);

const char* messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "/nothink My name is Henry Ndubuaku"}
])";

const char* options = R"({
    "max_tokens": 50,
    "stop_sequences": ["<|im_end|>"]
})";

char response[1024]; // 
int result = cactus_complete(model, messages, response, sizeof(response), options, nullptr, nullptr, nullptr);
```

## Performance (Qwen-0.6B-INT8-CPU)

| Device | Prefill-1k (toks/s) | Decode (toks/s) | Battery Drain (%/min) |
|:-------------------------------|:--------------------:|:----------------:|:---------------------:|
| iPhone 17 Pro                  | 420                  | 75               | 0.44                  |
| Galaxy S25 Ultra               | 336                  | 60               | 0.45                  |
| iPhone 16 Pro                  | 334                  | 60               | -                     |
| Nothing 3                      | 296                  | 53               | 0.44                  |
| iPhone 15 Pro                  | 274                  | 49               | -                     |
| iPhone 14 Pro                  | 269                  | 48               | -                     |
| OnePlus 13 5G                  | 268                  | 48               | 0.33                  |
| Galaxy S24 Ultra               | 240                  | 43               | 0.48                  |
| iPhone 15                      | 241                  | 43               | -                     |
| OnePlus Open                   | 235                  | 42               | -                     |
| Galaxy S23                     | 233                  | 42               | -                     |
| iPhone 13 Pro                  | 218                  | 39               | -                     |
| OnePlus 12                     | 216                  | 39               | 0.42                  |
| iPhone 13 mini                 | 156                  | 28               | -                     |
| Redmi K70 Ultra                | 154                  | 28               | 0.41                  |
| Xiaomi 13                      | 153                  | 28               | 0.50                  |
| OnePlus 11                     | 152                  | 28               | -                     |
| Pixel 6a                       | 95                   | 17               | 0.48                  |
| Nothing 3a                     | 93                   | 17               | 0.48                  |

## Performance notes

- These were collated from real-world runs, not controlled tests.
- Apple Intelligence drains 0.6 percent/min on iPhone 16 Pro Max
- Compressed file size is only 394mb (900mb+ in Executorch/GGUF)

## Coming improvements:

- INT4 to 2x speed, while reducing battery drain and file size 2x
- NPUs to improve energy-efficiency and prefill speed up to 11x
- VLM and Audio models like LFM-VL, Whisper, KittenTTS, etc. 

## Using this repo
You can run these codes directly on M-series Macbooks since they are ARM-based.
Vanilla M3 CPU-only can run Qwen3-600m-INT8 at 60+ toks/sec, just run the following: 

```bash
./tests/run.sh 
```

## Generating weights from HuggingFace 

Run one of the following 

```bash
# Language models
python3 tools/convert_hf.py google/gemma-3-270m-it weights/gemma3-270m/ --precision INT8
python3 tools/convert_hf.py LiquidAI/LFM2-350M weights/lfm2-350m/ --precision INT8
python3 tools/convert_hf.py HuggingFaceTB/SmolLM2-360m-Instruct weights/smollm2-360m/ --precision INT8
python3 tools/convert_hf.py Qwen/Qwen3-0.6B weights/qwen3-600m/ --precision INT8 # supports tool call
python3 tools/convert_hf.py LiquidAI/LFM2-700M weights/lfm2-700m/ --precision INT8
python3 tools/convert_hf.py google/gemma-3-1b-it weights/gemma3-1b/ --precision INT8
python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
python3 tools/convert_hf.py Qwen/Qwen3-1.7B weights/qwen3-1.7B/ --precision INT8

# Embedding-only models
python3 tools/convert_hf.py Qwen/Qwen3-Embedding-0.6B weights/qwen3-embed-600m/ --precision INT8
python3 tools/convert_hf.py nomic-ai/nomic-embed-text-v2-moe weights/nomic/ --precision INT8
```

Then replace the model path in `tests/test_engine.cpp` with your choice.

## Resources 

- [C++ Documentation](docs/)
- [Join Our Discord](https://discord.gg/bNurx3AXTJ)
- [Website](https://cactuscompute.com)
- [Contribution Guidelines](CONTRIBUTING.md)

## SDKs for app developers

- [Kotlin Multiplatform SDK](https://github.com/cactus-compute/cactus-kotlin)
- [Flutter SDK](https://github.com/cactus-compute/cactus-flutter)
- [React Native SDK](https://github.com/cactus-compute/cactus-react)

## Try demo apps

- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)

