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
from cactus import cactus_init, cactus_complete, cactus_destroy
import json

model = cactus_init("weights/lfm2-vl-450m")

messages = [{"role": "user", "content": "What is 2+2?"}]
response = json.loads(cactus_complete(model, messages))
print(response["response"])

cactus_destroy(model)
```

## API Reference

### `cactus_init(model_path, corpus_dir=None)`

Initialize a model and return its handle.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to model weights directory |
| `corpus_dir` | `str` | Optional path to RAG corpus directory for document Q&A |

```python
model = cactus_init("weights/lfm2-vl-450m")
rag_model = cactus_init("weights/lfm2-rag", corpus_dir="./documents")
```

### `cactus_complete(model, messages, **options)`

Run chat completion. Returns JSON string with response and metrics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle from `cactus_init` |
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
messages = [{"role": "user", "content": "Hello!"}]
response = cactus_complete(model, messages, max_tokens=100)
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
response = cactus_complete(model, messages, tools=tools)

# Streaming
def on_token(token, token_id, user_data):
    print(token, end="", flush=True)

cactus_complete(model, messages, callback=on_token)
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
result = json.loads(cactus_complete(model, messages))
if result["cloud_handoff"]:
    # Defer to cloud API (e.g., OpenAI, Anthropic)
    response = call_cloud_api(messages)
else:
    response = result["response"]
```

### `cactus_transcribe(model, audio_path, prompt="")`

Transcribe audio using a Whisper model. Returns JSON string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Whisper model handle |
| `audio_path` | `str` | Path to audio file (WAV) |
| `prompt` | `str` | Whisper prompt for language/task |

```python
whisper = cactus_init("weights/whisper-small")
prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
response = cactus_transcribe(whisper, "audio.wav", prompt=prompt)
print(json.loads(response)["response"])
cactus_destroy(whisper)
```

### `cactus_embed(model, text, normalize=False)`

Get text embeddings. Returns list of floats.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle |
| `text` | `str` | Text to embed |
| `normalize` | `bool` | L2-normalize embeddings (default: False) |

```python
embedding = cactus_embed(model, "Hello world")
print(f"Dimension: {len(embedding)}")
```

### `cactus_image_embed(model, image_path)`

Get image embeddings from a VLM. Returns list of floats.

```python
embedding = cactus_image_embed(model, "image.png")
```

### `cactus_audio_embed(model, audio_path)`

Get audio embeddings from a Whisper model. Returns list of floats.

```python
embedding = cactus_audio_embed(whisper, "audio.wav")
```

### `cactus_reset(model)`

Reset model state (clear KV cache). Call between unrelated conversations.

```python
cactus_reset(model)
```

### `cactus_stop(model)`

Stop an ongoing generation (useful with streaming callbacks).

```python
cactus_stop(model)
```

### `cactus_destroy(model)`

Free model memory. Always call when done.

```python
cactus_destroy(model)
```

### `cactus_get_last_error()`

Get the last error message, or `None` if no error.

```python
error = cactus_get_last_error()
if error:
    print(f"Error: {error}")
```

### `cactus_tokenize(model, text)`

Tokenize text. Returns list of token IDs.

```python
tokens = cactus_tokenize(model, "Hello world")
print(tokens)  # [1234, 5678, ...]
```

### `cactus_rag_query(model, query, top_k=5)`

Query RAG corpus for relevant text chunks. Requires model initialized with `corpus_dir`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | handle | Model handle (must have corpus_dir set) |
| `query` | `str` | Query text |
| `top_k` | `int` | Number of chunks to retrieve (default: 5) |

```python
model = cactus_init("weights/lfm2-rag", corpus_dir="./documents")
chunks = cactus_rag_query(model, "What is machine learning?", top_k=3)
for chunk in chunks:
    print(f"Score: {chunk['score']:.2f} - {chunk['text'][:100]}...")
```

## Vision (VLM)

Pass images in the messages for vision-language models:

```python
vlm = cactus_init("weights/lfm2-vl-450m")

messages = [{
    "role": "user",
    "content": "Describe this image",
    "images": ["path/to/image.png"]
}]
response = cactus_complete(vlm, messages)
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
