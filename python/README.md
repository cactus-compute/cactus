---
title: "Cactus Python Package"
description: "Python package and ctypes bindings for the Cactus on-device AI inference engine."
keywords: ["Python package", "Python bindings", "on-device AI", "Python FFI", "embeddings", "transcription", "RAG"]
---

# Cactus Python Package

Python bindings for Cactus Engine via FFI. Auto-installed when you run `source ./setup`.

> **Model bundles:** Pre-built runtime bundles for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Getting Started

<!-- --8<-- [start:install] -->
```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
cactus build --python
```
<!-- --8<-- [end:install] -->

```bash
# Download pre-built bundles
cactus download LiquidAI/LFM2-VL-450M
cactus download openai/whisper-small

# Optional: set your Cactus Cloud API key for automatic cloud fallback
cactus auth
```

## Quick Example

<!-- --8<-- [start:example] -->
```python
from cactus import ensure_model, cactus_init, cactus_complete, cactus_destroy
import json

# Downloads the pre-built bundle from HuggingFace if not already present
bundle = ensure_model("LiquidAI/LFM2-VL-450M")

model = cactus_init(str(bundle), None, False)
messages = json.dumps([{"role": "user", "content": "What is 2+2?"}])
result = cactus_complete(model, messages, None, None, None)
print(result["response"])
cactus_destroy(model)
```
<!-- --8<-- [end:example] -->

## API Reference

All functions are module-level and mirror the C FFI directly. Handles are plain `int` values (C pointers).

### Model Downloads

Download pre-built bundles programmatically (no CLI needed):

```python
from cactus import ensure_model, get_bundle_dir

# ensure_model downloads the pre-built bundle if missing, returns its Path
bundle = ensure_model("openai/whisper-tiny")

# Or resolve the expected on-disk location explicitly
bundle_dir = get_bundle_dir("openai/whisper-tiny", bits=4)
# -> Path("weights/whisper-tiny-cq4")
```

### Init / Lifecycle

```python
model = cactus_init(model_path: str, corpus_dir: str | None, cache_index: bool) -> int
cactus_destroy(model: int)
cactus_reset(model: int)   # clear KV cache
cactus_stop(model: int)    # abort ongoing generation
cactus_get_last_error() -> str | None
```

### Completion

Returns a `dict` with `success`, `error`, `cloud_handoff`, `response`, optional `thinking` (only present when the model emits chain-of-thought content, placed before `function_calls`), `function_calls`, `segments` (always `[]` for completion — populated only for Whisper transcription with the `timestamps` option), `confidence`, timing stats (`time_to_first_token_ms`, `total_time_ms`, `prefill_tps`, `decode_tps`, `ram_usage_mb`), and token counts (`prefill_tokens`, `decode_tokens`, `total_tokens`).

```python
result = cactus_complete(
    model: int,
    messages: list | str,            # list of {role, content} dicts or JSON string
    options: dict | str | None,        # optional inference options
    tools: list | str | None,        # optional tool definitions
    callback: Callable[[str, int], None] | None,  # streaming token callback
    pcm_data: list[int] | None = None              # optional raw audio bytes
) -> dict
```

```python
# With options and streaming
options = json.dumps({"max_tokens": 256, "temperature": 0.7})
def on_token(token, token_id): print(token, end="", flush=True)

result = cactus_complete(model, messages, options, None, on_token)
if result["cloud_handoff"]:
    # response already contains cloud result
    pass
```

**Response format:**
```json
{
    "success": true,
    "error": null,
    "cloud_handoff": false,
    "response": "4",
    "function_calls": [],
    "segments": [],
    "confidence": 0.92,
    "confidence_threshold": 0.7,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 163.7,
    "prefill_tps": 619.5,
    "decode_tps": 168.4,
    "ram_usage_mb": 512.3,
    "prefill_tokens": 28,
    "decode_tokens": 12,
    "total_tokens": 40
}
```

### Prefill

Pre-processes input text and populates the KV cache without generating output tokens. This reduces latency for subsequent calls to `cactus_complete`.

```python
cactus_prefill(
    model: int,
    messages: list | str,            # list of {role, content} dicts or JSON string
    options: dict | str | None,        # optional inference options
    tools: list | str | None,        # optional tool definitions
    pcm_data: list[int] | None = None              # optional raw audio bytes
) -> dict
```

```python
tools = json.dumps([{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City, State, Country"}
            },
            "required": ["location"]
        }
    }
}])

messages = json.dumps([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "<|tool_call_start|>get_weather(location=\"Paris\")<|tool_call_end|>"},
    {"role": "tool", "content": "{\"name\": \"get_weather\", \"content\": \"Sunny, 72°F\"}"},
    {"role": "assistant", "content": "It's sunny and 72°F in Paris!"}
])
cactus_prefill(model, messages, None, tools)

completion_messages = json.dumps([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "<|tool_call_start|>get_weather(location=\"Paris\")<|tool_call_end|>"},
    {"role": "tool", "content": "{\"name\": \"get_weather\", \"content\": \"Sunny, 72°F\"}"},
    {"role": "assistant", "content": "It's sunny and 72°F in Paris!"},
    {"role": "user", "content": "What about SF?"}
])
result = cactus_complete(model, completion_messages, None, tools, None)
```

**Response format:**
```json
{
    "success": true,
    "error": null,
    "prefill_tokens": 25,
    "prefill_tps": 166.1,
    "total_time_ms": 150.5,
    "ram_usage_mb": 245.67
}
```

### Transcription

Returns a `dict` with the `response` field (transcribed text) and a `segments` array of `{start, end, text}` objects. `segments` is populated only for Whisper models when the `timestamps` option is set (`{"timestamps": True}`); it is empty otherwise, including for all Parakeet transcription.

```python
result = cactus_transcribe(
    model: int,
    audio_path: str | None,
    prompt: str | None,
    options: dict | str | None,
    callback: Callable[[str, int], None] | None,
    pcm_data: list[int] | bytes | None
) -> dict
```

**Custom vocabulary** biases the decoder toward domain-specific words (supported for Whisper and Moonshine models). Pass `custom_vocabulary` and `vocabulary_boost` in `options`:

```python
options = json.dumps({
    "custom_vocabulary": ["Omeprazole", "HIPAA", "Cactus"],
    "vocabulary_boost": 3.0
})
result = cactus_transcribe(model, "medical_notes.wav", None, options, None, None)
```

```python
result = cactus_transcribe(model, "/path/to/audio.wav", None, {"timestamps": True}, None, None)
print(result["response"])
for seg in result["segments"]:
    print(f"[{seg['start']:.3f}s - {seg['end']:.3f}s] {seg['text']}")
```

### Streaming transcription

Transcribe continuously while audio is still being captured (Whisper and Parakeet TDT). Open a session, push 16 kHz mono 16-bit PCM chunks, and read text back as it stabilizes: `confirmed` words are final (append them to your transcript), `pending` is the volatile tail (replace it each call, for live display only).

```python
stream = cactus_stream_transcribe_start(model: int, options: dict | str | None) -> int
result = cactus_stream_transcribe_process(stream: int, pcm_data: bytes) -> dict  # {"confirmed": str, "pending": str, plus per-call timing stats}
result = cactus_stream_transcribe_stop(stream: int) -> dict                      # {"confirmed": str, "pending": ""}; destroys the session
```

`options` is forwarded to `cactus_transcribe` for **Whisper only** (e.g. `language`, `max_tokens`); the Parakeet TDT path ignores it. Chunking is handled internally.

```python
stream = cactus_stream_transcribe_start(model, {"language": "en"})
transcript = ""
for chunk in pcm_chunks:                       # each chunk: 16 kHz mono 16-bit PCM bytes
    out = cactus_stream_transcribe_process(stream, chunk)
    transcript += out["confirmed"]             # show out["pending"] separately as a live preview
transcript += cactus_stream_transcribe_stop(stream)["confirmed"]
```

### Embeddings

```python
embedding = cactus_embed(model: int, text: str, normalize: bool) -> list[float]
embedding = cactus_image_embed(model: int, image_path: str) -> list[float]
embedding = cactus_audio_embed(model: int, audio_path: str) -> list[float]
```

### Tokenization

```python
tokens = cactus_tokenize(model: int, text: str) -> list[int]
result = cactus_score_window(model: int, tokens: list[int], start: int, end: int, context: int) -> dict
```

### RAG

```python
result = cactus_rag_query(model: int, query: str, top_k: int) -> dict
```

Returns a `dict` with a `chunks` array. Each chunk has `score` (float), `source` (str, from document metadata), and `content` (str):

```json
{
    "chunks": [
        {"score": 0.0142, "source": "doc.txt", "content": "relevant passage..."}
    ]
}
```

### Vector Index

```python
index = cactus_index_init(index_dir: str, embedding_dim: int) -> int
cactus_index_add(index: int, ids: list[int], documents: list[str],
                 metadatas: list[str] | None, embeddings: list[list[float]])
cactus_index_delete(index: int, ids: list[int])
result = cactus_index_get(index: int, ids: list[int]) -> dict
result = cactus_index_query(index: int, embedding: list[float], options: dict | str | None) -> dict
cactus_index_compact(index: int)
cactus_index_destroy(index: int)
```

`cactus_index_query` returns `{"results":[{"id":<int>,"score":<float>}, ...]}`. `cactus_index_get` returns `{"results":[{"document":"...","metadata":<str|null>,"embedding":[...]}, ...]}`.

### Logging

```python
cactus_log_set_level(level: int)  # 0=DEBUG 1=INFO 2=WARN (default) 3=ERROR 4=NONE
cactus_log_set_callback(callback: Callable[[int, str, str], None] | None)
```

### Telemetry

```python
cactus_set_telemetry_environment(framework: str, cache_location: str | None, version: str | None)
cactus_set_app_id(app_id: str)
cactus_telemetry_flush()
cactus_telemetry_shutdown()
```

Functions that return a value raise `RuntimeError` on failure. `cactus_index_add`, `cactus_index_delete`, and `cactus_index_compact` also raise `RuntimeError` on failure despite not returning a value. Truly void functions that never raise: `cactus_destroy`, `cactus_reset`, `cactus_stop`, `cactus_index_destroy`, logging and telemetry functions.

## Vision (VLM)

Pass images in the messages content for vision-language models (LFM2-VL, LFM2.5-VL, Gemma4, Qwen3.5):

```python
messages = json.dumps([{
    "role": "user",
    "content": "Describe this image",
    "images": ["path/to/image.png"]
}])
result = cactus_complete(model, messages, None, None, None)
print(result["response"])
```

## Audio (Multimodal)

Pass audio files in messages for models with native audio understanding (Gemma4):

```python
messages = json.dumps([{
    "role": "user",
    "content": "Transcribe the audio.",
    "audio": ["path/to/audio.wav"]
}])
result = cactus_complete(model, messages, None, None, None)
print(result["response"])

# Combined vision + audio
messages = json.dumps([{
    "role": "user",
    "content": "Describe the image and transcribe the audio.",
    "images": ["path/to/image.png"],
    "audio": ["path/to/audio.wav"]
}])
result = cactus_complete(model, messages, None, None, None)
```

## Compute Graph

The `Graph` API provides a tensor computation graph for building and executing dataflow pipelines on the Cactus kernel layer:

```python
from cactus.bindings.cactus import Graph
import numpy as np

g = Graph()
a = g.input((2, 2))
b = g.input((2, 2))
y = ((a - b) * (a + b)).abs().pow(2.0).view((4,))

g.set_input(a, np.array([[2, 4], [6, 8]], dtype=np.float16))
g.set_input(b, np.array([[1, 2], [3, 4]], dtype=np.float16))
g.execute()

print(y.numpy())  # [9. 144. 729. 2304.]
```

Supported ops: `+`, `-`, `*`, `/`, `abs`, `pow`, `view`, `flatten`, `concat`, `cat`, `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`.

Every op accepts a `backend=` kwarg (`Graph.CPU` or `Graph.METAL`) to pin that op; by default it follows the global backend — auto (best available), or the one forced with `cactus_set_backend("cpu"|"metal")`:

```python
m = g.matmul(a, b, backend=Graph.METAL)
```

## Testing

Run the full test suite:

```bash
python python/test.py        # compact output
python python/test.py -v     # verbose
```

Tests are in `python/tests/` — bindings, CLI, server, graph, model, transpile,
and component-partition coverage. Add a new `test_*.py` to extend.

## See Also

- `Cactus Engine API` — Full C API reference that the Python bindings wrap
- `Cactus Index API` — Vector database API for RAG applications
- `Fine-tuning Guide` — Train and deploy custom LoRA fine-tunes
- `Runtime Compatibility` — Weight versioning across releases
- [Apple Build Step](/apple/) — Builds Apple native artifacts used by bindings
- [Android Build Step](/android/) — Builds Android native artifacts used by bindings
- [Swift Bindings](/bindings/swift/) — Swift C-module bindings
- [Kotlin Bindings](/bindings/kotlin/) — Kotlin/JNI bindings
- [Flutter Bindings](/bindings/flutter/) — Dart FFI bindings
- [Rust Bindings](/bindings/rust/) — Raw Rust FFI declarations
