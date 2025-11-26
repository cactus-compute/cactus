<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

Fast, lightweight, cross-platform & energy-efficient AI inference framework for small consumer devices. 

## Cactus Graph 
Cactus Graph is a general numerical computing framework for implementing 
any model, like PyTorch for consumer devices.

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

## INT8 CPU-ONLY Performance 

- LLM/VLM Model: LFM2-VL-450m
- Transcribe Model: Whisper-Small
- [Peak RAM calculation logic](tests/test_utils.h#L160-L213) 
- Covers the full range of consumer devices

| Device | Short Prompt Decode | 1k prefill/decode | 4k prefill/decode | 4k Peak RAM | 256x256 VLM TTFT | 256x256 VLM Decode | 256x256 VLM Peak RAM | 30s Transcribe TTFT | 30s Transcribe Decode | 30s Transcribe Peak RAM |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Mac M4 Pro | 173 tps | 1574/115 tps | 1089/100 tps | 122 MB | 0.38s | 168 tps | 112 MB | 1.7s | 83 tps | 142 MB |
| Mac M3 Pro | - | - | - | - | - | - | - | - | - | - |
| Mac M2 Pro | - | - | - | - | - | - | - | - | - | - |
| Qualcomm PC X Elite | - | - | - | - | - | - | - | - | - | - |
| Qualcomm PC X Plus | - | - | - | - | - | - | - | - | - | - |
| iPad/Mac M5 | - | - | - | - | - | - | - | - | - | - |
| iPad/Mac M4 | - | - | - | - | - | - | - | - | - | - |
| iPad/Mac M3 | 112 tps | 786/78 tps | 446/60 tps | 98 MB | 0.58s | 111 tps | 54 MB | 4.24s | 58 tps | 42 MB |
| iPhone 17 Pro | - | - | - | - | - | - | - | - | - | - |
| iPhone 16 Pro | - | - | - | - | - | - | - | - | - | - |
| iPhone 15 Pro | 99 tps | 549/74 tps | - | - | 0.84s | 93 tps | - | - | - | - |
| Galaxy S25 Ultra | - | - | - | - | - | - | - | - | - | - |
| Galaxy S24 Ultra | - | - | - | - | - | - | - | - | - | - |
| Galaxy S23 Ultra | - | - | - | - | - | - | - | - | - | - |
| Pixel 10 Pro | - | - | - | - | - | - | - | - | - | - |
| Pixel 9 pro | - | - | - | - | - | - | - | - | - | - |
| Pixel 8 Pro | - | - | - | - | - | - | - | - | - | - |
| Oppo Find X9 | - | - | - | - | - | - | - | - | - | - |
| Xiaomi 15T Pro | - | - | - | - | - | - | - | - | - | - |
| Nothing CMF 3 Pro | - | - | - | - | - | - | - | - | - | - |
| Galaxy A56 | - | - | - | - | - | - | - | - | - | - |
| Galaxy A55 | - | - | - | - | - | - | - | - | - | - |
| Raspberry Pi 5 | - | - | - | - | - | - | - | - | - | - |

## Coming improvements:

- INT4 to 2x speed, while reducing battery drain and file size 2x
- NPUs to improve energy-efficiency and prefill speed up to 11x
- VLM and Audio models like LFM-VL, Whisper, KittenTTS, etc. 

## Using up this repo (on Mac)

Dependencies will be setup on first run automatically.

```bash
cli/cactus --help # to see all commands
cli/cactus run LiquidAI/LFM2-VL-450M # to interact with a model
cli/cactus test # to run unit tests during dev
cli/cactus download Qwen/Qwen3-0.6B # HF name, stored to weights/Qwen3-0.6B
```

## Supported models (INT8)

| Model | Completion | Tool Call | Vision | Embed | Speech
|-------|--------------------|-------------------|----------------|------|------|
| google/gemma-3-270m-it | ✓ | ✗ | ✗ | ✗ | ✗ |
| openai/whisper-small | ✗ | ✗ | ✗ | ✓ | ✓ |
| LiquidAI/LFM2-350M | ✓ | ✓ | ✗ | ✓ | ✗ |
| HuggingFaceTB/SmolLM2-360m-Instruct | ✓ | ✗ | ✗ | ✗ | ✗ |
| LiquidAI/LFM2-VL-450M | ✓ | ✗ | ✓ | ✓ | ✗ |
| Qwen/Qwen3-0.6B | ✓ | ✓ | ✗ | ✓ | ✗ |
| Qwen/Qwen3-Embedding-0.6B | ✗ | ✗ | ✗ | ✓ | ✗ |
| LiquidAI/LFM2-700M | ✓ | ✓ | ✗ | ✓ | ✗ |
| nomic-ai/nomic-embed-text-v2-moe | ✗ | ✗ | ✗ | ✓ | ✗ |
| openai/whisper-medium | ✗ | ✗ | ✗ | ✓ | ✓ |
| google/gemma-3-1b-it | ✓ | ✗ | ✗ | ✗ | ✗ |
| LiquidAI/LFM2-1.2B | ✓ | ✓ | ✗ | ✓ | ✗ |
| LiquidAI/LFM2-1.2B-RAG | ✓ | ✓ | ✗ | ✓ | ✗ |
| LiquidAI/LFM2-VL-1.6B | ✓ | ✗ | ✓ | ✓ | ✗ |
| Qwen/Qwen3-1.7B | ✓ | ✓ | ✗ | ✓ | ✗ |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | ✓ | ✗ | ✗ | ✓ | ✗ |


## Resources 

- [C++ Documentation](docs/)
- [Join Our Discord](https://discord.gg/bNurx3AXTJ)
- [Website](https://cactuscompute.com)
- [Contribution Guidelines](CONTRIBUTING.md)

## Using in your apps

```bash
android/build.sh # generate the `libcactus.so` and `libcactus.a` for android
apple/build.sh # generate the `.xcframeworks` for Apple
```

Or simply use the provided SDKs

- [Kotlin Multiplatform SDK](https://github.com/cactus-compute/cactus-kotlin)
- [Flutter SDK](https://github.com/cactus-compute/cactus-flutter)
- [React Native SDK](https://github.com/cactus-compute/cactus-react-native)
- [Swift SDK](https://github.com/mhayes853/swift-cactus)

## Try demo apps
i
- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)

## Windows ARM PC setup

```bash
# Needs C++, Python and MySys with Pacman, then install CMake and Python dependencies weight convertion dependencies 
pacman -S mingw-w64-clang-aarch64-cmake mingw-w64-clang-aarch64-toolchain mingw-w64-clang-aarch64-mman-win32
pip3 install -r tools/requirements.txt
tests/run.bat for Windows ARM
```
