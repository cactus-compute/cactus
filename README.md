<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

Cross-platform & energy-efficient kernels, runtime and AI inference engine for mobile devices. 

## Cactus Graph 
Cactus Graph is a general numerical computing framework for implementing 
any model, like PyTorch for mobile devices.

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

- <sub>**Models:** LFM2-VL-450m (text/vision) & Whisper-Small (speech)</sub>
- <sub>**Decode** = tokens/sec, **P/D** = prefill/decode toks/sec, **VLM** = 256×256 image, **STT** = 30s audio transcription</sub>
- <sub>**INT4 coming**: 2x speed, 2x smaller files</sub>
- <sub>**NPU support coming**: 5-11x prefill speed, better energy efficiency</sub>

| Device | Decode | 1k-P/D | 4k-P/D | 4k-RAM | VLM-TTFT | VLM-Dec | VLM-RAM | STT-TTFT | STT-Dec | STT-RAM |
|--------|--------|--------|--------|--------|----------|---------|---------|----------|---------|---------|
| Mac M4 Pro | 173 | 1574/115 | 1089/100 | 122MB | 0.38s | 168 | 112MB | 1.7s | 83 | 142MB |
| Mac M3 Pro | - | - | - | - | - | - | - | - | - | - |
| iPad/Mac M4 | 129 | 793/82 | 507/64 | 80MB | 0.46s | 113 | 45MB | 2.4s | 60 | 31MB |
| iPad/Mac M3 | 112 | 786/78 | 446/60 | 80MB | 0.58s | 111 | 54MB | 4.2s | 58 | 42MB |
| iPhone 17 Pro | - | - | - | - | - | - | - | - | - | - |
| iPhone 16 Pro | - | - | - | - | - | - | - | - | - | - |
| iPhone 15 Pro | 99 | 549/74 | - | - | 0.84s | 93 | - | - | - | - |
| Qualcomm X Elite | - | - | - | - | - | - | - | - | - | - |
| Qualcomm X Plus | - | - | - | - | - | - | - | - | - | - |
| Galaxy S25 Ultra | 91 | 230/63 | 173/57 | 128MB | 1.4s | 58 | - | - | - | - |
| Galaxy S24 Ultra | - | - | - | - | - | - | - | - | - | - |
| Pixel 10 Pro | - | - | - | - | - | - | - | - | - | - |
| Pixel 9 Pro | - | - | - | - | - | - | - | - | - | - |
| Oppo Find X9 | - | - | - | - | - | - | - | - | - | - |
| Xiaomi 15T Pro | - | - | - | - | - | - | - | - | - | - |
| Nothing CMF 3 Pro | - | - | - | - | - | - | - | - | - | - |
| Raspberry Pi 5 | 24 | 192/28 | - | - | 2.3s | 23 | - | 21s | 16 | - |


## Using up this repo on Mac

Dependencies will be setup on first run automatically.

```bash
cli/cactus --help # to see all commands
cli/cactus run LiquidAI/LFM2-VL-450M # to interact with a model
cli/cactus test # to run unit tests during dev + reproduce benchmarks
cli/cactus download Qwen/Qwen3-0.6B # HF name, stored to weights/Qwen3-0.6B
```

## Supported models (INT8)

| Model | Compressed Size | Completion | Tool Call | Vision | Embed | Speech
|-------|--------------------|-------------------|----------------|------|------|------|
| google/gemma-3-270m-it | 172  | ✓ | ✗ | ✗ | ✗ | ✗ |
| openai/whisper-small | 210  | ✗ | ✗ | ✗ | ✓ | ✓ |
| LiquidAI/LFM2-350M | 233  | ✓ | ✓ | ✗ | ✓ | ✗ |
| HuggingFaceTB/SmolLM2-360m-Instruct | 227  | ✓ | ✗ | ✗ | ✗ | ✗ |
| LiquidAI/LFM2-VL-450M | 420  | ✓ | ✗ | ✓ | ✓ | ✗ |
| Qwen/Qwen3-0.6B | 394  | ✓ | ✓ | ✗ | ✓ | ✗ |
| Qwen/Qwen3-Embedding-0.6B | 394  | ✗ | ✗ | ✗ | ✓ | ✗ |
| LiquidAI/LFM2-700M | 467  | ✓ | ✓ | ✗ | ✓ | ✗ |
| nomic-ai/nomic-embed-text-v2-moe | 533  | ✗ | ✗ | ✗ | ✓ | ✗ |
| google/gemma-3-1b-it | 642  | ✓ | ✗ | ✗ | ✗ | ✗ |
| openai/whisper-medium | 646  | ✗ | ✗ | ✗ | ✓ | ✓ |
| LiquidAI/LFM2-1.2B | 722  | ✓ | ✓ | ✗ | ✓ | ✗ |
| LiquidAI/LFM2-1.2B-RAG | 722  | ✓ | ✓ | ✗ | ✓ | ✗ |
| LiquidAI/LFM2-1.2B-Tools | 722  | ✓ | ✓ | ✗ | ✓ | ✗ |
| LiquidAI/LFM2-VL-1.6B | 1440  | ✓ | ✗ | ✓ | ✓ | ✗ |
| Qwen/Qwen3-1.7B | 1161  | ✓ | ✓ | ✗ | ✓ | ✗ |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1161  | ✓ | ✗ | ✗ | ✓ | ✗ |

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

- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)

## Windows ARM PC setup

```bash
# Needs C++, Python and MySys with Pacman, then install CMake and Python dependencies weight convertion dependencies 
pacman -S mingw-w64-clang-aarch64-cmake mingw-w64-clang-aarch64-toolchain mingw-w64-clang-aarch64-mman-win32
pip3 install -r tools/requirements.txt
tests/run.bat for Windows ARM
```
