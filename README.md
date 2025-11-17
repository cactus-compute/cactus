<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

Fast, lightweight, cross-platform & energy-efficient AI inference framework for all phones, from old and budget to high-end. 

## Cactus Graph 
Cactus Graph is a general numerical computing framework for implementing 
any model, like PyTorch for phones.

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

## Cactus Engine
Cactus Engine is an AI inference engine with OpenAI-compatible APIs built on top of Cactus Graphs.

```cpp
#include cactus.h

cactus_model_t model = cactus_init("path/to/weight/folder", 2048);

const char* messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Henry Ndubuaku"}
])";

const char* options = R"({
    "max_tokens": 50,
    "stop_sequences": ["<|im_end|>"]
})";

char response[1024];
int result = cactus_complete(model, messages, response, sizeof(response), options, nullptr, nullptr, nullptr);
```
Example response from Gemma3-270m-INT8
```json
{
    "success": true,
    "response": "Hi there! I'm just a friendly assistant.",
    "time_to_first_token_ms": 45.23,
    "total_time_ms": 163.67,
    "tokens_per_second": 168.42,
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

## INT8 CPU Performance (LFM2-1.2B) (722mb Compressed)

| Device | Prefill (toks/s) | Decode (toks/s) | Battery Drain (%/min) |
|:-------------------------------|:--------------------:|:----------------:|:---------------------:|
| Macbook M4 Pro                 | 590                  | 96               | -                     |
| Mac Mini M4 Pro                | 580                  | 93               | -                     |
| iPhone 17 Pro                  | 420                  | 81               | 0.44                  |
| Galaxy S25 Ultra               | 336                  | 64               | 0.45                  |
| iPhone 16 Pro                  | 334                  | 64               | -                     |
| Nothing 3a Pro                 | 296                  | 63               | 0.44                  |
| Macbook Pro M3                  | 462                  | 62               | -                     |
| iPhone 15 Pro                  | 274                  | 57               | -                     |
| iPhone 14 Pro                  | 269                  | 51               | -                     |
| OnePlus 13 5G                  | 268                  | 51               | 0.33                  |
| Macbook Air M3                 | 260                  | 50               | -                     |
| Galaxy S24 Ultra               | 240                  | 46               | 0.48                  |
| iPhone 15                      | 241                  | 46               | -                     |
| Galaxy S23                     | 233                  | 45               | -                     |
| iPhone 13 Pro                  | 218                  | 42               | -                     |
| OnePlus 12                     | 216                  | 42               | 0.42                  |
| iPhone 13 mini                 | 156                  | 30               | -                     |
| Redmi K70 Ultra                | 154                  | 30               | 0.41                  |
| Xiaomi 13                      | 153                  | 30               | 0.50                  |
| Pixel 6a                       | 85                   | 13               | 0.48                  |
| Nothing 3a                     | 83                   | 13               | 0.48                  |
| Raspberry Pi 5                 | 50                   | 8.5              | -                     |

## Coming improvements:

- INT4 to 2x speed, while reducing battery drain and file size 2x
- NPUs to improve energy-efficiency and prefill speed up to 11x
- VLM and Audio models like LFM-VL, Whisper, KittenTTS, etc. 

## Using this repo
You can run these codes directly on M-series Macbooks since they are ARM-based.
Vanilla M3 CPU-only can run LFM2-1.2B-INT8 at 50+ toks/sec, just run the following: 

```bash
tests/run.sh 
```

## Generating weights from HuggingFace 

Run one of the following 

```bash
# Language models (INT8)
python3 tools/convert_hf.py google/gemma-3-270m-it weights/gemma3-270m/
python3 tools/convert_hf.py LiquidAI/LFM2-350M weights/lfm2-350m/  # supports tool call
python3 tools/convert_hf.py HuggingFaceTB/SmolLM2-360m-Instruct weights/smollm2-360m/ 
python3 tools/convert_hf.py Qwen/Qwen3-0.6B weights/qwen3-600m/  # supports tool call
python3 tools/convert_hf.py LiquidAI/LFM2-700M weights/lfm2-700m/ # supports tool call
python3 tools/convert_hf.py google/gemma-3-1b-it weights/gemma3-1b/  
python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ # supports tool call
python3 tools/convert_hf.py Qwen/Qwen3-1.7B weights/qwen3-1.7B/ # supports tool call
python3 tools/convert_hf.py HuggingFaceTB/SmolLM2-1.7B-Instruct weights/smollm2-1.7b/ 

# Embedding-only models 
python3 tools/convert_hf.py Qwen/Qwen3-Embedding-0.6B weights/qwen3-embed-600m/ 
python3 tools/convert_hf.py nomic-ai/nomic-embed-text-v2-moe weights/nomic/ 
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
- [Swift SDK](https://github.com/mhayes853/swift-cactus)

## Try demo apps

- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)
