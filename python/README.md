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
handle = cactus_init(model_path: str, corpus_dir: str | None, cache_index: bool) -> int
cactus_destroy(handle: int)
cactus_reset(handle: int)   # clear KV cache
cactus_stop(handle: int)    # abort ongoing generation
cactus_get_last_error() -> str | None
```

### Completion

Returns a JSON string with `response`, `function_calls`, timing stats, and `cloud_handoff`.

```python
result_json = cactus_complete(
    handle: int,
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
    # confidence below threshold — defer to cloud
    pass
```

**Response format:**
```json
{
    "success": true,
    "response": "4",
    "function_calls": [],
    "cloud_handoff": false,
    "confidence": 0.92,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 163.7,
    "prefill_tps": 619.5,
    "decode_tps": 168.4,
    "prefill_tokens": 28,
    "decode_tokens": 12,
    "total_tokens": 40
}
```

### Transcription

```python
result_json = cactus_transcribe(
    handle: int,
    audio_path: str | None,
    prompt: str | None,
    options_json: str | None,
    callback: Callable[[str, int], None] | None,
    pcm_data: bytes | None
) -> str
```

Streaming transcription:

```python
stream = cactus_stream_transcribe_start(handle: int, options_json: str | None) -> int
partial = cactus_stream_transcribe_process(stream: int, pcm_data: bytes) -> str
final   = cactus_stream_transcribe_stop(stream: int) -> str
```

### Embeddings

```python
embedding = cactus_embed(handle: int, text: str, normalize: bool) -> list[float]
embedding = cactus_image_embed(handle: int, image_path: str) -> list[float]
embedding = cactus_audio_embed(handle: int, audio_path: str) -> list[float]
```

### Tokenization

```python
tokens     = cactus_tokenize(handle: int, text: str) -> list[int]
result_json = cactus_score_window(handle: int, tokens: list[int], start: int, end: int, context: int) -> str
```

### Detect Language

```python
result_json = cactus_detect_language(
    handle: int,
    audio_path: str | None,
    options_json: str | None,
    pcm_data: bytes | None
) -> str
```

### VAD

```python
result_json = cactus_vad(
    handle: int,
    audio_path: str | None,
    options_json: str | None,
    pcm_data: bytes | None
) -> str
```

### RAG

```python
result_json = cactus_rag_query(handle: int, query: str, top_k: int) -> str
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

### Logging

```python
cactus_log_set_level(level: int)  # 0=DEBUG 1=INFO 2=WARN 3=ERROR 4=NONE
cactus_log_set_callback(callback: Callable[[int, str, str], None] | None)
```

### Telemetry

```python
cactus_set_telemetry_environment(cache_location: str)
cactus_set_app_id(app_id: str)
cactus_telemetry_flush()
cactus_telemetry_shutdown()
```

All functions raise `RuntimeError` on failure.

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
