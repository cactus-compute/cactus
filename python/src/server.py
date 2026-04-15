import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .cactus import (
    cactus_init,
    cactus_destroy,
    cactus_reset,
    cactus_complete,
    cactus_embed,
    cactus_transcribe,
    cactus_get_last_error,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

app = FastAPI(title="Cactus", version="0.1.0")


class _ModelSlot:
    """A single loaded model with its own inference lock."""
    def __init__(self, model_id: str, handle):
        self.model_id = model_id
        self.handle = handle
        self.lock = asyncio.Lock()
        self.last_used: float = time.monotonic()

    def touch(self):
        self.last_used = time.monotonic()


class ModelManager:
    """Keeps up to ``max_warm`` models loaded simultaneously (LRU eviction)."""

    def __init__(self, context_length: Optional[int] = None, max_warm: int = 2):
        self.slots: dict[str, _ModelSlot] = {}   # model_id -> slot
        self.swap_lock = asyncio.Lock()           # protects slots dict mutations
        self.context_length = context_length
        self.max_warm = max_warm

    def _load(self, model_id: str):
        model_path = WEIGHTS_DIR / model_id
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in weights/")
        handle = cactus_init(str(model_path), context_length=self.context_length or 0)
        if not handle:
            err = cactus_get_last_error() or "unknown error"
            raise HTTPException(status_code=500, detail=f"Failed to load model: {err}")
        return handle

    async def get(self, model_id: str) -> tuple:
        """Returns ``(handle, per-model lock)``."""
        async with self.swap_lock:
            slot = self.slots.get(model_id)
            if slot:
                slot.touch()
                return slot.handle, slot.lock

            # Evict LRU if at capacity
            if len(self.slots) >= self.max_warm:
                lru_id = min(self.slots, key=lambda k: self.slots[k].last_used)
                lru_slot = self.slots.pop(lru_id)
                print(f"Evicting model: {lru_id}")
                cactus_destroy(lru_slot.handle)

            handle = await asyncio.get_event_loop().run_in_executor(None, self._load, model_id)
            slot = _ModelSlot(model_id, handle)
            self.slots[model_id] = slot
            print(f"Loaded model: {model_id} ({len(self.slots)}/{self.max_warm} slots)")
            return slot.handle, slot.lock

    def preload(self, model_id: str):
        if model_id in self.slots:
            return
        handle = self._load(model_id)
        self.slots[model_id] = _ModelSlot(model_id, handle)
        print(f"Loaded model: {model_id} ({len(self.slots)}/{self.max_warm} slots)")

    def shutdown(self):
        for slot in self.slots.values():
            cactus_destroy(slot.handle)
        self.slots.clear()


manager = ModelManager()


EMBEDDING_MODEL_TYPES = {"bert"}
LLM_MODEL_TYPES = {"lfm2", "qwen", "gemma"}
STT_MODEL_TYPES = {"whisper", "moonshine", "parakeet", "parakeet_tdt"}
PREFERRED_DEFAULT_MODEL = "lfm2-24b-a2b"


def _read_config_field(model_dir: Path, field: str) -> str:
    config = model_dir / "config.txt"
    if not config.exists():
        return ""
    prefix = f"{field}="
    for line in config.read_text().splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return ""


def _read_model_type(model_dir: Path) -> str:
    return _read_config_field(model_dir, "model_type")


def _pick_default_model() -> str:
    if not WEIGHTS_DIR.exists():
        raise RuntimeError("No weights/ directory found. Run 'cactus download <model>' first.")
    preferred = WEIGHTS_DIR / PREFERRED_DEFAULT_MODEL
    if preferred.is_dir() and (preferred / "config.txt").exists():
        return PREFERRED_DEFAULT_MODEL
    for entry in sorted(WEIGHTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "config.txt").exists():
            if _read_model_type(entry) in LLM_MODEL_TYPES:
                return entry.name
    for entry in sorted(WEIGHTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "config.txt").exists():
            return entry.name
    raise RuntimeError("No models found in weights/. Run 'cactus download <model>' first.")


def discover_models():
    models = []
    if not WEIGHTS_DIR.exists():
        return models
    effective_ctx = manager.context_length or 512
    for entry in sorted(WEIGHTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "config.txt").exists():
            stat = entry.stat()
            model_max = int(_read_config_field(entry, "context_length") or 0)
            ctx = min(effective_ctx, model_max) if model_max > 0 else effective_ctx
            models.append({
                "id": entry.name,
                "object": "model",
                "created": int(stat.st_mtime),
                "owned_by": "cactus",
                "context_window": ctx,
            })
    return models


# --- Request / Response Models ---

class Permissive(BaseModel):
    model_config = {"extra": "allow"}

class ChatMessage(Permissive):
    role: str
    content: Optional[str | list] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None

class ToolFunction(Permissive):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None

class Tool(Permissive):
    type: str = "function"
    function: ToolFunction

class ToolChoiceFunction(Permissive):
    name: str

class ToolChoiceObject(Permissive):
    type: str = "function"
    function: ToolChoiceFunction

class ChatRequest(Permissive):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    stream: bool = False
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[str | ToolChoiceObject] = None

class EmbeddingRequest(Permissive):
    model: str
    input: str | list[str]
    encoding_format: Optional[str] = None


# --- Helpers ---

def _flatten_message(msg: ChatMessage) -> dict:
    d = {"role": msg.role}
    content = msg.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        d["content"] = "\n".join(parts)
    elif content is not None:
        d["content"] = content
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls is not None:
        d["tool_calls"] = msg.tool_calls
    return d


def translate_tools(tools: Optional[list[Tool]], tool_choice) -> tuple[Optional[list[dict]], bool]:
    if tools is None:
        return None, False

    if isinstance(tool_choice, str) and tool_choice == "none":
        return None, False

    force_tools = False
    cactus_tools = None

    if isinstance(tool_choice, ToolChoiceObject):
        target_name = tool_choice.function.name
        for t in tools:
            if t.function.name == target_name:
                cactus_tools = [{"name": t.function.name, "description": t.function.description or "", "parameters": t.function.parameters or {}}]
                break
        force_tools = True
    else:
        cactus_tools = [
            {"name": t.function.name, "description": t.function.description or "", "parameters": t.function.parameters or {}}
            for t in tools
        ]
        if isinstance(tool_choice, str) and tool_choice == "required":
            force_tools = True

    return cactus_tools, force_tools


def make_tool_calls(function_calls: list) -> list[dict]:
    out = []
    for fc in function_calls:
        args = fc.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        out.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fc.get("name", ""),
                "arguments": args_str,
            },
        })
    return out


def build_chat_response(response_json: dict, model_id: str, request_id: str) -> dict:
    function_calls = response_json.get("function_calls", [])
    has_tool_calls = len(function_calls) > 0
    text = response_json.get("response") or ""

    tool_calls = make_tool_calls(function_calls) if has_tool_calls else None

    if has_tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    message: dict = {"role": "assistant", "content": text if not has_tool_calls else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    prefill = response_json.get("prefill_tokens", 0)
    decode = response_json.get("decode_tokens", 0)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": None,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prefill,
            "completion_tokens": decode,
            "total_tokens": prefill + decode,
        },
    }


# --- Endpoints ---

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": discover_models()}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    handle, model_lock = await manager.get(req.model)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

    messages = [_flatten_message(m) for m in req.messages]
    cactus_tools, force_tools = translate_tools(req.tools, req.tool_choice)

    messages_json = json.dumps(messages)

    options: dict = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.top_p is not None:
        options["top_p"] = req.top_p
    if req.top_k is not None:
        options["top_k"] = req.top_k
    max_tok = req.max_tokens or req.max_completion_tokens
    if max_tok is not None:
        options["max_tokens"] = max_tok
    if req.stop:
        options["stop_sequences"] = req.stop
    if force_tools:
        options["force_tools"] = True
    options_json = json.dumps(options) if options else None

    tools_json = json.dumps(cactus_tools) if cactus_tools else None

    if req.stream:
        return StreamingResponse(
            _stream_completion(handle, model_lock, messages_json, options_json, tools_json, request_id, req.model),
            media_type="text/event-stream",
        )

    async with model_lock:
        cactus_reset(handle)
        raw = await asyncio.get_event_loop().run_in_executor(
            None, lambda: cactus_complete(handle, messages_json, options_json, tools_json, None)
        )

    response_json = json.loads(raw)
    if not response_json.get("success", False):
        raise HTTPException(status_code=500, detail=response_json.get("error", "completion failed"))

    return build_chat_response(response_json, req.model, request_id)


async def _stream_completion(handle, model_lock, messages_json, options_json, tools_json, request_id, model_id):
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_token(token_str, token_id):
        loop.call_soon_threadsafe(queue.put_nowait, token_str)

    async def run_inference():
        async with model_lock:
            cactus_reset(handle)
            raw = await loop.run_in_executor(
                None,
                lambda: cactus_complete(handle, messages_json, options_json, tools_json, on_token),
            )
        loop.call_soon_threadsafe(queue.put_nowait, None)
        return raw

    task = asyncio.create_task(run_inference())

    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    while True:
        token = await queue.get()
        if token is None:
            break
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "system_fingerprint": None,
            "choices": [{"index": 0, "delta": {"content": token}, "logprobs": None, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    raw = await task
    response_json = json.loads(raw)

    function_calls = response_json.get("function_calls", [])
    has_tool_calls = len(function_calls) > 0

    if has_tool_calls:
        finish_reason = "tool_calls"
        tool_calls = make_tool_calls(function_calls)
        tool_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "system_fingerprint": None,
            "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "logprobs": None, "finish_reason": None}],
        }
        yield f"data: {json.dumps(tool_chunk)}\n\n"
    else:
        finish_reason = "stop"

    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish_reason}],
    }

    prefill = response_json.get("prefill_tokens", 0)
    decode = response_json.get("decode_tokens", 0)
    final_chunk["usage"] = {
        "prompt_tokens": prefill,
        "completion_tokens": decode,
        "total_tokens": prefill + decode,
    }

    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    model_type = _read_model_type(WEIGHTS_DIR / req.model)
    if model_type not in EMBEDDING_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' is not an embedding model. Use an embedding model like nomic-embed-text-v2-moe.",
        )
    handle, model_lock = await manager.get(req.model)

    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []

    async with model_lock:
        for i, text in enumerate(inputs):
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=text: cactus_embed(handle, t, True)
            )
            data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i,
            })

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def _pick_default_stt_model() -> Optional[str]:
    """Find the first available STT model in weights/."""
    if not WEIGHTS_DIR.exists():
        return None
    for entry in sorted(WEIGHTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "config.txt").exists():
            if _read_model_type(entry) in STT_MODEL_TYPES:
                return entry.name
    return None


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
    timestamp_granularities: Optional[str] = Form(None),
):
    if not model:
        model = _pick_default_stt_model() or ""
    if not model:
        raise HTTPException(status_code=400, detail="No STT model available. Download one with 'cactus download'.")

    model_type = _read_model_type(WEIGHTS_DIR / model)
    if model_type not in STT_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not a speech-to-text model (type={model_type}).",
        )

    suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Convert non-WAV formats to WAV via ffmpeg
    if suffix != ".wav":
        wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
        if not shutil.which("ffmpeg"):
            Path(tmp_path).unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="ffmpeg is required to decode non-WAV audio formats")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
            capture_output=True,
        )
        Path(tmp_path).unlink(missing_ok=True)
        if result.returncode != 0:
            Path(wav_path).unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"ffmpeg failed to decode audio: {result.stderr.decode(errors='ignore')[:200]}")
        tmp_path = wav_path

    try:
        handle, model_lock = await manager.get(model)

        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if language:
            options["language"] = language
        options_json = json.dumps(options) if options else None

        async with model_lock:
            cactus_reset(handle)
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: cactus_transcribe(handle, tmp_path, prompt, options_json, None, None)
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result = json.loads(raw)
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "transcription failed"))

    text = result.get("response", "")

    if response_format == "verbose_json":
        segments = result.get("segments", [])
        return {
            "task": "transcribe",
            "language": language or "",
            "duration": segments[-1]["end"] if segments else 0.0,
            "text": text,
            "segments": [
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for i, seg in enumerate(segments)
            ],
        }

    if response_format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(text)

    return {"text": text}


@app.on_event("shutdown")
def on_shutdown():
    manager.shutdown()


def create_app(context_length: Optional[int] = None) -> FastAPI:
    manager.context_length = context_length
    if context_length and context_length > 0:
        os.environ["CACTUS_KV_WINDOW_SIZE"] = str(context_length)
    default_model = _pick_default_model()
    print(f"Preloading model: {default_model}")
    manager.preload(default_model)
    print(f"Model ready: {default_model}")

    stt_model = _pick_default_stt_model()
    if stt_model and stt_model != default_model:
        print(f"Preloading STT model: {stt_model}")
        manager.preload(stt_model)
        print(f"STT model ready: {stt_model}")

    return app
