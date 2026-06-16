"""OpenAI-compatible local HTTP server for Cactus v2 bundles."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .bindings.cactus import (
    cactus_complete,
    cactus_destroy,
    cactus_embed,
    cactus_get_last_error,
    cactus_init,
    cactus_reset,
    cactus_transcribe,
)
from .cli.download import get_weights_dir
from .cli.common import DEFAULT_MODEL_ID, PROJECT_ROOT, is_repo_checkout

LOGGER = logging.getLogger(__name__)

LLM_MODEL_TYPES = {"gemma", "gemma3n", "gemma4", "lfm2", "qwen", "qwen3p5", "needle", "youtu"}
STT_MODEL_TYPES = {"whisper", "parakeet_tdt", "parakeet-tdt"}


@dataclass(frozen=True)
class ModelInfo:
    id: str
    path: Path
    model_type: str
    context_length: int
    created: int
    supports_embedding: bool = False


class ModelRegistry:
    def __init__(self, weights_root: Path, extra_model: Path | None = None):
        self.weights_root = weights_root
        self.models: dict[str, ModelInfo] = {}
        self._discover(weights_root)
        if extra_model is not None:
            info = self._info_for_dir(extra_model)
            if info is None:
                raise RuntimeError(f"Not a valid v2 Cactus bundle: {extra_model}")
            self.models[info.id] = info

    @staticmethod
    def _read_config_field(model_dir: Path, field: str) -> str:
        config = model_dir / "config.txt"
        if not config.exists():
            return ""
        prefix = f"{field}="
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip()
        return ""

    @staticmethod
    def _supports_embedding(model_dir: Path) -> bool:
        """A bundle can produce text embeddings if it carries a text_embedding
        component (Nomic/BERT encoder) or a decoder_embed_chunk component
        (causal-LM hidden-state embeddings)."""
        manifest = model_dir / "components" / "manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return False
        names = {str(c.get("component", "")) for c in data.get("components", []) if isinstance(c, dict)}
        return "text_embedding" in names or "decoder_embed_chunk" in names

    @classmethod
    def _info_for_dir(cls, path: Path) -> ModelInfo | None:
        display_id = path.expanduser().name
        resolved = path.expanduser().resolve()
        if not (resolved / "config.txt").exists():
            return None
        if not (resolved / "components" / "manifest.json").exists():
            return None
        context_raw = cls._read_config_field(resolved, "context_length")
        try:
            context_length = int(context_raw or 0)
        except ValueError:
            context_length = 0
        stat = resolved.stat()
        return ModelInfo(
            id=display_id,
            path=resolved,
            model_type=cls._read_config_field(resolved, "model_type"),
            context_length=context_length,
            created=int(stat.st_mtime),
            supports_embedding=cls._supports_embedding(resolved),
        )

    def _discover(self, root: Path) -> None:
        if not root.exists():
            return
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            info = self._info_for_dir(entry)
            if info is not None:
                self.models[info.id] = info

    def require(self, model_id: str) -> ModelInfo:
        info = self.models.get(model_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not available")
        return info

    def default_llm(self, preferred: str | None = None) -> ModelInfo:
        if preferred:
            info = self.models.get(preferred)
            if info is None:
                raise RuntimeError(f"Requested model '{preferred}' is not a valid v2 Cactus bundle")
            if info.model_type not in LLM_MODEL_TYPES:
                raise RuntimeError(f"Requested model '{preferred}' is not an LLM bundle")
            return info
        preferred_id = get_weights_dir(DEFAULT_MODEL_ID).name
        if preferred_id in self.models and self.models[preferred_id].model_type in LLM_MODEL_TYPES:
            return self.models[preferred_id]
        for info in sorted(self.models.values(), key=lambda x: x.id):
            if info.model_type in LLM_MODEL_TYPES:
                return info
        raise RuntimeError("No valid LLM bundles found. Prepare a transpiled bundle before running `cactus serve`.")

    def default_stt(self) -> ModelInfo | None:
        for info in sorted(self.models.values(), key=lambda x: x.id):
            if info.model_type in STT_MODEL_TYPES:
                return info
        return None

    def list_openai_models(self) -> list[dict[str, Any]]:
        out = []
        for info in sorted(self.models.values(), key=lambda x: x.id):
            out.append({
                "id": info.id,
                "object": "model",
                "created": info.created,
                "owned_by": "cactus",
                "context_window": info.context_length,
                "model_type": info.model_type,
            })
        return out


class _ModelSlot:
    def __init__(self, info: ModelInfo, handle):
        self.info = info
        self.handle = handle
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()
        self.active_requests = 0

    def touch(self) -> None:
        self.last_used = time.monotonic()


class ModelManager:
    def __init__(self, registry: ModelRegistry, *, max_warm: int = 2):
        self.registry = registry
        self.max_warm = max_warm
        self.slots: dict[str, _ModelSlot] = {}
        self.swap_lock = asyncio.Lock()

    def _load(self, info: ModelInfo):
        try:
            return cactus_init(str(info.path))
        except RuntimeError as exc:
            err = cactus_get_last_error() or str(exc)
            raise HTTPException(status_code=500, detail=f"Failed to load model '{info.id}': {err}") from exc

    async def _get_slot(self, model_id: str) -> _ModelSlot:
        info = self.registry.require(model_id)
        async with self.swap_lock:
            slot = self.slots.get(info.id)
            if slot is not None:
                slot.touch()
                return slot

            if len(self.slots) >= self.max_warm:
                idle = [s for s in self.slots.values() if s.active_requests == 0 and not s.lock.locked()]
                if not idle:
                    raise HTTPException(status_code=503, detail="All warm model slots are busy")
                victim = min(idle, key=lambda s: s.last_used)
                self.slots.pop(victim.info.id, None)
                cactus_destroy(victim.handle)

            handle = await asyncio.get_running_loop().run_in_executor(None, self._load, info)
            slot = _ModelSlot(info, handle)
            self.slots[info.id] = slot
            return slot

    @asynccontextmanager
    async def acquire(self, model_id: str):
        slot = await self._get_slot(model_id)
        async with self.swap_lock:
            if slot.info.id not in self.slots:
                raise HTTPException(status_code=503, detail="Model slot was evicted before use")
            slot.active_requests += 1
        try:
            yield slot
        finally:
            async with self.swap_lock:
                slot.active_requests = max(0, slot.active_requests - 1)
                slot.touch()

    async def preload(self, model_id: str) -> None:
        async with self.acquire(model_id):
            return

    def shutdown(self) -> None:
        for slot in self.slots.values():
            cactus_destroy(slot.handle)
        self.slots.clear()


class Permissive(BaseModel):
    model_config = {"extra": "allow"}


class ChatMessage(Permissive):
    role: str
    content: str | list[Any] | None = None
    tool_call_id: str | None = None
    tool_calls: list[Any] | None = None


class ToolFunction(Permissive):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


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
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: str | ToolChoiceObject | None = None


class EmbeddingRequest(Permissive):
    model: str
    input: str | list[str]


def _flatten_message(msg: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role}
    if isinstance(msg.content, list):
        parts = []
        for part in msg.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        out["content"] = "\n".join(parts)
    elif msg.content is not None:
        out["content"] = msg.content
    if msg.tool_call_id is not None:
        out["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls is not None:
        out["tool_calls"] = msg.tool_calls
    return out


def _translate_tools(tools: list[Tool] | None, tool_choice) -> tuple[list[dict[str, Any]] | None, bool]:
    if not tools or tool_choice == "none":
        return None, False
    translated = [
        {"type": "function", "function": {"name": t.function.name, "description": t.function.description or "", "parameters": t.function.parameters or {}}}
        for t in tools
    ]
    if isinstance(tool_choice, ToolChoiceObject):
        selected = [t for t in translated if t["function"]["name"] == tool_choice.function.name]
        return selected or None, True
    return translated, tool_choice == "required"


def _make_tool_calls(function_calls: list[Any], *, with_index: bool = False) -> list[dict[str, Any]]:
    out = []
    for i, call in enumerate(function_calls):
        if not isinstance(call, dict):
            continue
        args = call.get("arguments", {})
        entry: dict[str, Any] = {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
            },
        }
        if with_index:
            entry["index"] = len(out)
        out.append(entry)
    return out


def _chat_options(req: ChatRequest) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k"):
        value = getattr(req, key)
        if value is not None:
            options[key] = value
    max_tokens = req.max_tokens if req.max_tokens is not None else req.max_completion_tokens
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if req.stop:
        options["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    return options


def _build_chat_response(result: dict[str, Any], model_id: str, request_id: str) -> dict[str, Any]:
    function_calls = result.get("function_calls") or []
    tool_calls = _make_tool_calls(function_calls)
    has_tool_calls = bool(tool_calls)
    prefill = int(result.get("prefill_tokens") or 0)
    decode = int(result.get("decode_tokens") or 0)
    message: dict[str, Any] = {"role": "assistant", "content": None if has_tool_calls else result.get("response", "")}
    if has_tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": "tool_calls" if has_tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": prefill,
            "completion_tokens": decode,
            "total_tokens": prefill + decode,
        },
    }


def _event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_completion(manager: ModelManager, req: ChatRequest, request_id: str, messages, options, tools):
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_token(token: str, token_id: int) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))

    async def run_inference():
        error = None
        result = None
        try:
            async with manager.acquire(req.model) as slot:
                async with slot.lock:
                    cactus_reset(slot.handle)
                    result = await loop.run_in_executor(
                        None,
                        lambda: cactus_complete(slot.handle, messages, options, tools, on_token),
                    )
        except Exception as exc:
            LOGGER.exception("Streaming completion failed")
            error = exc
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", (result, error)))

    task = asyncio.create_task(run_inference())
    created = int(time.time())
    yield _event({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}],
    })

    result = None
    error = None
    while True:
        kind, value = await queue.get()
        if kind == "token":
            yield _event({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": value}, "logprobs": None, "finish_reason": None}],
            })
        elif kind == "done":
            result, error = value
            break

    await task
    if error is not None:
        # OpenAI streaming spec embeds errors in a `data:` line with an `error`
        # object; clients (openai-python, vercel ai) parse `event:` lines as
        # comments. We also emit a final chunk so the client gets a
        # finish_reason before [DONE] and never hangs.
        yield f"data: {json.dumps({'error': {'message': str(error), 'type': 'server_error'}})}\n\n"
        yield _event({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
        })
        yield "data: [DONE]\n\n"
        return

    function_calls = (result or {}).get("function_calls") or []
    tool_calls = _make_tool_calls(function_calls, with_index=True)
    finish_reason = "tool_calls" if tool_calls else "stop"
    if tool_calls:
        yield _event({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "logprobs": None, "finish_reason": None}],
        })
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": int((result or {}).get("prefill_tokens") or 0),
            "completion_tokens": int((result or {}).get("decode_tokens") or 0),
            "total_tokens": int((result or {}).get("total_tokens") or 0),
        },
    }
    yield _event(final)
    yield "data: [DONE]\n\n"


def _requested_granularities(primary: list[str] | None, bracketed: list[str] | None) -> list[str]:
    values = []
    for item in (primary or []) + (bracketed or []):
        if item:
            values.append(item)
    return values


def create_app(
    *,
    weights_root: Path | None = None,
    model_path: Path | None = None,
    default_model: str | None = None,
    max_warm: int = 2,
    preload: bool = True,
) -> FastAPI:
    root = Path(weights_root) if weights_root is not None else (
        PROJECT_ROOT / "weights" if is_repo_checkout() else Path.home() / ".cache" / "cactus" / "weights"
    )
    registry = ModelRegistry(root, extra_model=model_path)
    if default_model is not None:
        selected = registry.models.get(default_model)
        if selected is None:
            raise RuntimeError(f"Requested model '{default_model}' is not a valid v2 Cactus bundle")
    else:
        try:
            selected = registry.default_llm()
        except RuntimeError:
            # No LLM available — allow serving a non-LLM bundle (e.g. an embedding model).
            available = sorted(registry.models.values(), key=lambda info: info.id)
            if not available:
                raise
            selected = available[0]
    manager = ModelManager(registry, max_warm=max_warm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if preload:
            await manager.preload(selected.id)
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="Cactus", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry
    app.state.manager = manager
    app.state.default_model = selected.id
    app.state.default_stt_model = registry.default_stt().id if registry.default_stt() else None

    @app.get("/v1/models")
    async def list_models(request: Request):
        reg: ModelRegistry = request.app.state.registry
        return {"object": "list", "data": reg.list_openai_models()}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, req: ChatRequest):
        reg: ModelRegistry = request.app.state.registry
        info = reg.require(req.model)
        if info.model_type not in LLM_MODEL_TYPES:
            raise HTTPException(status_code=400, detail=f"Model '{req.model}' is not an LLM model")
        messages = [_flatten_message(m) for m in req.messages]
        tools, force_tools = _translate_tools(req.tools, req.tool_choice)
        options = _chat_options(req)
        if force_tools:
            options["force_tools"] = True
        request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        mgr: ModelManager = request.app.state.manager
        if req.stream:
            return StreamingResponse(
                _stream_completion(mgr, req, request_id, messages, options or None, tools),
                media_type="text/event-stream",
            )
        async with mgr.acquire(req.model) as slot:
            async with slot.lock:
                cactus_reset(slot.handle)
                result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: cactus_complete(slot.handle, messages, options or None, tools, None),
                )
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error") or "completion failed")
        return _build_chat_response(result, req.model, request_id)

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(""),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(None),
        timestamp_granularities: list[str] | None = Form(None),
        timestamp_granularities_array: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    ):
        if response_format not in {"json", "text", "verbose_json"}:
            raise HTTPException(status_code=400, detail=f"Unsupported transcription response_format: {response_format}")
        granularities = _requested_granularities(timestamp_granularities, timestamp_granularities_array)
        if "word" in granularities:
            raise HTTPException(status_code=400, detail="Word timestamp granularity is not supported")
        if any(g != "segment" for g in granularities):
            raise HTTPException(status_code=400, detail=f"Unsupported timestamp granularity: {', '.join(granularities)}")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix != ".wav":
            raise HTTPException(status_code=400, detail="Only .wav transcription uploads are supported for now")
        model_id = model or request.app.state.default_stt_model
        if not model_id:
            raise HTTPException(status_code=400, detail="No STT model available")
        reg: ModelRegistry = request.app.state.registry
        info = reg.require(model_id)
        if info.model_type not in STT_MODEL_TYPES:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not a speech-to-text model")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            options: dict[str, Any] = {}
            if temperature is not None:
                options["temperature"] = temperature
            if language:
                options["language"] = language
            if response_format == "verbose_json":
                options["timestamps"] = True
            mgr: ModelManager = request.app.state.manager
            async with mgr.acquire(model_id) as slot:
                async with slot.lock:
                    cactus_reset(slot.handle)
                    result = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: cactus_transcribe(slot.handle, str(tmp_path), prompt, options or None, None),
                    )
        finally:
            tmp_path.unlink(missing_ok=True)
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error") or "transcription failed")
        text = result.get("response", "")
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            segments = result.get("segments") or []
            return {
                "task": "transcribe",
                "language": language or "",
                "duration": float(segments[-1].get("end", 0.0)) if segments else 0.0,
                "text": text,
                "segments": [
                    {"id": i,
                     "start": float(seg.get("start", 0.0)),
                     "end": float(seg.get("end", 0.0)),
                     "text": seg.get("text", "")}
                    for i, seg in enumerate(segments)
                ],
            }
        return {"text": text}

    @app.post("/v1/embeddings")
    async def create_embeddings(request: Request, req: EmbeddingRequest):
        reg: ModelRegistry = request.app.state.registry
        info = reg.require(req.model)
        if not info.supports_embedding:
            raise HTTPException(status_code=400, detail=f"Model '{req.model}' is not an embedding model")
        inputs = [req.input] if isinstance(req.input, str) else list(req.input)
        if not inputs:
            raise HTTPException(status_code=400, detail="'input' must not be empty")
        mgr: ModelManager = request.app.state.manager

        def _embed_all(handle) -> list[list[float]]:
            vectors = []
            for text in inputs:
                cactus_reset(handle)
                vectors.append(cactus_embed(handle, text, True))
            return vectors

        async with mgr.acquire(req.model) as slot:
            async with slot.lock:
                try:
                    vectors = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: _embed_all(slot.handle)
                    )
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc))
        data = [
            {"object": "embedding", "index": i, "embedding": vector}
            for i, vector in enumerate(vectors)
        ]
        return {
            "object": "list",
            "data": data,
            "model": req.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    return app
