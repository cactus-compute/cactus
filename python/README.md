---
title: "Cactus Python SDK"
description: "Python bindings for Cactus on-device AI inference engine. Supports chat completion, vision, transcription, embeddings, RAG, tool calling, and streaming."
keywords: ["Python SDK", "on-device AI", "LLM inference", "Python FFI", "embeddings", "transcription", "RAG"]
---

# Cactus Python Package

Python bindings for Cactus Engine via FFI. Auto-installed when you run `source ./setup`.

> **Model weights:** Pre-converted weights for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Getting Started

<!-- --8<-- [start:install] -->
```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
cactus build --python
```
<!-- --8<-- [end:install] -->

```bash
# Download models
cactus download LiquidAI/LFM2-VL-450M
cactus download openai/whisper-small

# Optional: set your Cactus Cloud API key for automatic cloud fallback
cactus auth
```

## Quick Example

<!-- --8<-- [start:example] -->
```python
from cactus import cactus_init, cactus_complete, cactus_destroy
import json

model = cactus_init("weights/lfm2-vl-450m", None, False)
messages = json.dumps([{"role": "user", "content": "What is 2+2?"}])
result = json.loads(cactus_complete(model, messages, None, None, None))
print(result["response"])
cactus_destroy(model)
```
<!-- --8<-- [end:example] -->

## API Reference

All functions are module-level and mirror the C FFI directly. Handles are plain `int` values (C pointers).

### Init / Lifecycle

```python
model = cactus_init(model_path: str, corpus_dir: str | None, cache_index: bool) -> int
cactus_destroy(model: int)
cactus_reset(model: int)   # clear KV cache
cactus_stop(model: int)    # abort ongoing generation
cactus_get_last_error() -> str | None
```

### Completion

Returns a JSON string with `success`, `error`, `cloud_handoff`, `response`, optional `thinking` (only present when the model emits chain-of-thought content, placed before `function_calls`), `function_calls`, `segments` (always `[]` for completion — populated only in transcription responses), `confidence`, timing stats (`time_to_first_token_ms`, `total_time_ms`, `prefill_tps`, `decode_tps`, `ram_usage_mb`), and token counts (`prefill_tokens`, `decode_tokens`, `total_tokens`).

```python
result_json = cactus_complete(
    model: int,
    messages_json: str,              # JSON array of {role, content}
    options_json: str | None,        # optional inference options
    tools_json: str | None,          # optional tool definitions
    callback: Callable[[str, int], None] | None   # streaming token callback
) -> str
```

```python
# With options and streaming
options = json.dumps({"max_tokens": 256, "temperature": 0.7})
def on_token(token, token_id): print(token, end="", flush=True)

result = json.loads(cactus_complete(model, messages_json, options, None, on_token))
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
    messages_json: str,              # JSON array of {role, content}
    options_json: str | None,        # optional inference options
    tools_json: str | None           # optional tool definitions
) -> None
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
result = json.loads(cactus_complete(model, completion_messages, None, tools, None))
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

Returns a JSON string. Use `json.loads()` to access the `response` field (transcribed text), the `segments` array (timestamped segments as `{"start": <sec>, "end": <sec>, "text": "<str>"}` — Whisper: phrase-level from timestamp tokens; Parakeet TDT: word-level from frame timing; Parakeet CTC and Moonshine: one segment per transcription window (consecutive VAD speech regions up to 30s)), and other metadata.

```python
result_json = cactus_transcribe(
    model: int,
    audio_path: str | None,
    prompt: str | None,
    options_json: str | None,
    callback: Callable[[str, int], None] | None,
    pcm_data: bytes | None
) -> str
```

**Custom vocabulary** biases the decoder toward domain-specific words (supported for Whisper and Moonshine models). Pass `custom_vocabulary` and `vocabulary_boost` in `options_json`:

```python
options = json.dumps({
    "custom_vocabulary": ["Omeprazole", "HIPAA", "Cactus"],
    "vocabulary_boost": 3.0
})
result = json.loads(cactus_transcribe(model, "medical_notes.wav", None, options, None, None))
```

Streaming transcription:
Streaming transcription also returns JSON strings:

```python
stream       = cactus_stream_transcribe_start(model: int, options_json: str | None) -> int
partial_json = cactus_stream_transcribe_process(stream: int, pcm_data: bytes) -> str
final_json   = cactus_stream_transcribe_stop(stream: int) -> str
```

In `cactus_stream_transcribe_process` responses: `confirmed` is the stable text from segments that have been finalised across two consecutive decode passes (potentially replaced by a cloud result); `confirmed_local` is the same text before any cloud substitution; `pending` is the current window's unconfirmed transcription text; `segments` contains timestamped segments for the current audio window.

```python
result = json.loads(cactus_transcribe(model, "/path/to/audio.wav", None, None, None, None))
print(result["response"])
for seg in result["segments"]:
    print(f"[{seg['start']:.3f}s - {seg['end']:.3f}s] {seg['text']}")
```

Streaming also accepts `custom_vocabulary` in the options passed to `cactus_stream_transcribe_start`. The bias is applied for the lifetime of the stream session.

### Embeddings

```python
embedding = cactus_embed(model: int, text: str, normalize: bool) -> list[float]
embedding = cactus_image_embed(model: int, image_path: str) -> list[float]
embedding = cactus_audio_embed(model: int, audio_path: str) -> list[float]
```

### Tokenization

```python
tokens     = cactus_tokenize(model: int, text: str) -> list[int]
result_json = cactus_score_window(model: int, tokens: list[int], start: int, end: int, context: int) -> str
```

### Detect Language

```python
result_json = cactus_detect_language(
    model: int,
    audio_path: str | None,
    options_json: str | None,
    pcm_data: bytes | None
) -> str
```

Returns a JSON string with fields: `success`, `error`, `language` (BCP-47 code), `language_token`, `token_id`, `confidence`, `entropy`, `total_time_ms`, `ram_usage_mb`.

### VAD

```python
result_json = cactus_vad(
    model: int,
    audio_path: str | None,
    options_json: str | None,
    pcm_data: bytes | None
) -> str
```

Returns a JSON string: `{"success":true,"error":null,"segments":[{"start":<sample_index>,"end":<sample_index>},...],"total_time_ms":...,"ram_usage_mb":...}`. VAD segments contain only `start` and `end` as integer sample indices — no `text` field.

### RAG

```python
result_json = cactus_rag_query(model: int, query: str, top_k: int) -> str
```

Returns a JSON string with a `chunks` array. Each chunk has `score` (float), `source` (str, from document metadata), and `content` (str):

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
                 embeddings: list[list[float]], metadatas: list[str] | None)
cactus_index_delete(index: int, ids: list[int])
result_json = cactus_index_get(index: int, ids: list[int]) -> str
result_json = cactus_index_query(index: int, embedding: list[float], options_json: str | None) -> str
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
cactus_set_telemetry_environment(cache_location: str)
cactus_set_app_id(app_id: str)
cactus_telemetry_flush()
cactus_telemetry_shutdown()
```

Functions that return a value raise `RuntimeError` on failure. `cactus_prefill`, `cactus_index_add`, `cactus_index_delete`, and `cactus_index_compact` also raise `RuntimeError` on failure despite not returning a value. Truly void functions that never raise: `cactus_destroy`, `cactus_reset`, `cactus_stop`, `cactus_index_destroy`, logging and telemetry functions.

## Vision (VLM)

Pass images in the messages content for vision-language models:

```python
messages = json.dumps([{
    "role": "user",
    "content": "Describe this image",
    "images": ["path/to/image.png"]
}])
result = json.loads(cactus_complete(model, messages, None, None, None))
print(result["response"])
```

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference that the Python bindings wrap
- [Cactus Index API](/docs/cactus_index.md) — Vector database API for RAG applications
- [Fine-tuning Guide](/docs/finetuning.md) — Train and deploy custom LoRA fine-tunes
- [Runtime Compatibility](/docs/compatibility.md) — Weight versioning across releases
- [Swift SDK](/apple/) — Swift bindings for iOS/macOS
- [Kotlin/Android SDK](/android/) — Kotlin bindings for Android
- [Flutter SDK](/flutter/) — Dart bindings for cross-platform mobile
