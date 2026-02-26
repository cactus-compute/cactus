---
title: "Cactus Python SDK"
description: "Python bindings for Cactus on-device AI inference engine. Supports chat completion, vision, transcription, embeddings, RAG, tool calling, and streaming."
keywords: ["Python SDK", "on-device AI", "LLM inference", "Python FFI", "embeddings", "transcription", "RAG"]
---

# Cactus Python Package

Python bindings for Cactus Engine via FFI. Auto-installed when you run `source ./setup`.

## Getting Started

```bash
# Setup environment
source ./setup

# Build shared library for Python
cactus build --python

# Download models
cactus download LiquidAI/LFM2-VL-450M
cactus download openai/whisper-small

# Optional: set your Cactus Cloud API key for automatic cloud fallback
cactus auth
```

## Quick Example

```python
from cactus import CactusModel
import json

with CactusModel("weights/lfm2-vl-450m") as model:
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = json.loads(model.complete(messages))
    print(response["response"])
```

## API Reference

### `CactusModel(model_path, corpus_dir=None)`

Initialize a model and act as a context manager.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to model weights directory |
| `corpus_dir` | `str` | Optional path to RAG corpus directory for document Q&A |

```python
with CactusModel("weights/lfm2-vl-450m") as model:
    # use model...
    pass

# Or without context manager
rag_model = CactusModel("weights/lfm2-rag", corpus_dir="./documents")
# rag_model.destroy() must be called
```

### `model.complete(messages, **options)`

Run chat completion. Returns JSON string with response and metrics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list\|str` | List of message dicts or JSON string |
| `tools` | `list` | Optional tool definitions for function calling |
| `temperature` | `float` | Sampling temperature |
| `top_p` | `float` | Top-p sampling |
| `top_k` | `int` | Top-k sampling |
| `max_tokens` | `int` | Maximum tokens to generate |
| `stop_sequences` | `list` | Stop sequences |
| `include_stop_sequences` | `bool` | Include matched stop sequences in output (default: `False`) |
| `force_tools` | `bool` | Constrain output to tool call format |
| `tool_rag_top_k` | `int` | Select top-k relevant tools via Tool RAG (default: 2, 0 = use all tools) |
| `confidence_threshold` | `float` | Minimum confidence for local generation (default: 0.7, triggers cloud_handoff when below) |
| `callback` | `fn` | Streaming callback `fn(token, token_id, user_data)` |

```python
# Basic completion
with CactusModel("weights/lfm2-vl-450m") as model:
    messages = [{"role": "user", "content": "Hello!"}]
    response = model.complete(messages, max_tokens=100)
    print(json.loads(response)["response"])

# With tools
tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }
}]
with CactusModel("weights/lfm2-vl-450m") as model:
    response = model.complete(messages, tools=tools)

# Streaming
def on_token(token, token_id, user_data):
    print(token, end="", flush=True)

with CactusModel("weights/lfm2-vl-450m") as model:
    model.complete(messages, callback=on_token)
```

**Response format** (all fields always present):
```json
{
    "success": true,
    "error": null,
    "cloud_handoff": false,
    "response": "Hello! How can I help?",
    "function_calls": [],
    "confidence": 0.85,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 163.7,
    "prefill_tps": 619.5,
    "decode_tps": 168.4,
    "ram_usage_mb": 245.67,
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

**Cloud handoff response** (when model detects low confidence):
```json
{
    "success": false,
    "error": null,
    "cloud_handoff": true,
    "response": null,
    "function_calls": [],
    "confidence": 0.18,
    "time_to_first_token_ms": 45.2,
    "total_time_ms": 45.2,
    "prefill_tps": 619.5,
    "decode_tps": 0.0,
    "ram_usage_mb": 245.67,
    "prefill_tokens": 28,
    "decode_tokens": 0,
    "total_tokens": 28
}
```

When `cloud_handoff` is `True`, the model's confidence dropped below `confidence_threshold` (default: 0.7) and recommends deferring to a cloud-based model for better results. Handle this in your application:

```python
with CactusModel("weights/lfm2-vl-450m") as model:
    result = json.loads(model.complete(messages))
    if result["cloud_handoff"]:
        # Defer to cloud API (e.g., OpenAI, Anthropic)
        response = call_cloud_api(messages)
    else:
        response = result["response"]
```

### `model.transcribe(audio_path, prompt="")`

Transcribe audio using a Whisper model. Returns JSON string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_path` | `str` | Path to audio file (WAV) |
| `prompt` | `str` | Whisper prompt for language/task |

```python
with CactusModel("weights/whisper-small") as whisper:
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    response = whisper.transcribe("audio.wav", prompt=prompt)
    print(json.loads(response)["response"])
```

### `model.embed(text, normalize=False)`

Get text embeddings. Returns list of floats.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to embed |
| `normalize` | `bool` | L2-normalize embeddings (default: False) |

```python
with CactusModel("weights/lfm2-vl-450m") as model:
    embedding = model.embed("Hello world")
    print(f"Dimension: {len(embedding)}")
```

### `model.image_embed(image_path)`

Get image embeddings from a VLM. Returns list of floats.

```python
with CactusModel("weights/lfm2-vl-450m") as model:
    embedding = model.image_embed("image.png")
```

### `model.audio_embed(audio_path)`

Get audio embeddings from a Whisper model. Returns list of floats.

```python
with CactusModel("weights/whisper-small") as whisper:
    embedding = whisper.audio_embed("audio.wav")
```

### `model.reset()`

Reset model state (clear KV cache). Call between unrelated conversations.

```python
model.reset()
```

### `model.stop()`

Stop an ongoing generation (useful with streaming callbacks).

```python
model.stop()
```

### `model.tokenize(text)`

Tokenize text. Returns list of token IDs.

```python
with CactusModel("weights/lfm2-vl-450m") as model:
    tokens = model.tokenize("Hello world")
    print(tokens)  # [1234, 5678, ...]
```

### `model.rag_query(query, top_k=5)`

Query RAG corpus for relevant text chunks. Requires model initialized with `corpus_dir`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Query text |
| `top_k` | `int` | Number of chunks to retrieve (default: 5) |

```python
with CactusModel("weights/lfm2-rag", corpus_dir="./documents") as model:
    chunks = model.rag_query("What is machine learning?", top_k=3)
    for chunk in chunks:
        print(f"Score: {chunk['score']:.2f} - {chunk['text'][:100]}...")
```

## Vision (VLM)

Pass images in the messages for vision-language models:

```python
with CactusModel("weights/lfm2-vl-450m") as vlm:
    messages = [{
        "role": "user",
        "content": "Describe this image",
        "images": ["path/to/image.png"]
    }]
    response = vlm.complete(messages)
    print(json.loads(response)["response"])
```

## Full Example

See `python/example.py` for a complete example covering:
- Text completion
- Text/image/audio embeddings
- Vision (VLM)
- Speech transcription

```bash
python python/example.py
```

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference that the Python bindings wrap
- [Cactus Index API](/docs/cactus_index.md) — Vector database API for RAG applications
- [Fine-tuning Guide](/docs/finetuning.md) — Train and deploy custom LoRA fine-tunes
- [Runtime Compatibility](/docs/compatibility.md) — Weight versioning across releases
- [Swift SDK](/apple/) — Swift bindings for iOS/macOS
- [Kotlin/Android SDK](/android/) — Kotlin bindings for Android
- [Flutter SDK](/flutter/) — Dart bindings for cross-platform mobile
