"""OpenAI-compatible local HTTP server for Cactus v2 bundles."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from .bindings.cactus import (
    cactus_complete,
    cactus_destroy,
    cactus_embed,
    cactus_get_last_error,
    cactus_init,
    cactus_reset,
    cactus_transcribe,
)
from .cli.download import get_model_dir_name, get_weights_dir
from .cli.common import DEFAULT_MODEL_ID, is_valid_bundle, weights_root as default_weights_root

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
        if not is_valid_bundle(resolved):
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

    def _resolve(self, model_id: str) -> ModelInfo | None:
        """Match an exact bundle id, or fall back to the HuggingFace id / bare
        stem (e.g. 'google/gemma-4-E2B-it' -> 'gemma-4-e2b-it-cq4') when it
        is unambiguous."""
        info = self.models.get(model_id)
        if info is not None:
            return info
        stem = get_model_dir_name(model_id)
        if stem in self.models:
            return self.models[stem]
        variants = [m for m in self.models.values() if re.match(rf"{re.escape(stem)}-cq\d", m.id)]
        return variants[0] if len(variants) == 1 else None

    def require(self, model_id: str) -> ModelInfo:
        info = self._resolve(model_id)
        if info is None:
            available = ", ".join(sorted(self.models)) or "none"
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' is not available (available: {available})",
            )
        return info

    def default_llm(self, preferred: str | None = None) -> ModelInfo:
        if preferred:
            info = self._resolve(preferred)
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
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    reasoning_effort: str | None = None
    tools: list[Tool] | None = None
    tool_choice: str | ToolChoiceObject | None = None


class EmbeddingRequest(Permissive):
    model: str
    input: str | list[str]


_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.DOTALL)

_ENGINE_IMAGE_FORMATS = {"PNG", "JPEG", "GIF"}


def _decodable_image(source: Any) -> bool:
    try:
        with Image.open(source) as im:
            if im.format not in _ENGINE_IMAGE_FORMATS or im.width <= 0 or im.height <= 0:
                return False
            im.load()
        return True
    except Exception:
        return False


def _materialize_image(url: str) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        match = _DATA_URI_RE.match(url)
        if not match:
            return None
        mime = (match.group("mime") or "image/png").lower()
        ext = "." + mime.split("/")[-1].split("+")[0] if "/" in mime else ".png"
        payload = match.group("data")
        try:
            raw = base64.b64decode(payload) if match.group("b64") else payload.encode("utf-8")
        except ValueError:
            return None
        if not _decodable_image(BytesIO(raw)):
            return None
        fd, path = tempfile.mkstemp(suffix=ext, prefix="cactus_img_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        return path
    if url.startswith("file://"):
        url = url[len("file://"):]
    if Path(url).exists():
        return url if _decodable_image(url) else None
    return None


def _flatten_message(msg: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role}
    if isinstance(msg.content, list):
        parts = []
        images: list[str] = []
        for part in msg.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                raw = part.get("image_url")
                url = raw.get("url") if isinstance(raw, dict) else (raw if isinstance(raw, str) else part.get("url"))
                resolved = _materialize_image(url) if isinstance(url, str) else None
                if resolved:
                    images.append(resolved)
            elif isinstance(part, str):
                parts.append(part)
        out["content"] = "\n".join(parts)
        if images:
            out["images"] = images
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


def _clamp_finite(value: Any, lo: float, hi: float) -> float | None:
    """Clamp a sampling param to a finite, sane range so extreme/non-finite
    values (e.g. 1e308, inf, nan) can't overflow the native sampler."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return max(lo, min(hi, v))


def _chat_options(req: ChatRequest) -> dict[str, Any]:
    options: dict[str, Any] = {}
    temperature = _clamp_finite(req.temperature, 0.0, 2.0)
    options["temperature"] = temperature if temperature is not None else 0.7
    top_p = _clamp_finite(req.top_p, 0.0, 1.0)
    if top_p is not None:
        options["top_p"] = top_p
    if req.top_k is not None:
        options["top_k"] = max(0, min(int(req.top_k), 1 << 20))
    max_tokens = req.max_tokens if req.max_tokens is not None else req.max_completion_tokens
    options["max_tokens"] = max_tokens if max_tokens is not None else 4096
    if req.stop:
        options["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.reasoning_effort and req.reasoning_effort.lower() not in ("none", "off"):
        options["enable_thinking_if_supported"] = True
    return options


_HARMONY_MARKER = "<|channel"
_HARMONY_THOUGHT_CHANNELS = {"thought", "thinking", "analysis", "reflection"}
_HARMONY_CHANNEL_RE = re.compile(r"<\|channel\|?>\s*([a-zA-Z_]+)\s*\n?")
_HARMONY_CONTROL_RE = re.compile(r"<\|[^>]*>")


def _split_harmony_channels(text: str) -> tuple[str, str | None]:
    if not text or _HARMONY_MARKER not in text:
        return text, None
    segments = _HARMONY_CHANNEL_RE.split(text)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    if segments[0].strip():
        content_parts.append(segments[0])
    for i in range(1, len(segments) - 1, 2):
        body = segments[i + 1]
        target = reasoning_parts if segments[i].lower() in _HARMONY_THOUGHT_CHANNELS else content_parts
        target.append(body)
    content = _HARMONY_CONTROL_RE.sub("", "".join(content_parts)).strip()
    reasoning = _HARMONY_CONTROL_RE.sub("", "\n".join(reasoning_parts)).strip() or None
    if not content and reasoning:
        return reasoning, None
    return content, reasoning


class _HarmonyStreamSplitter:
    def __init__(self) -> None:
        self._buf = ""
        self._channel = "final"

    def _kind(self) -> str:
        return "reasoning" if self._channel in _HARMONY_THOUGHT_CHANNELS else "content"

    def _hold_partial(self, s: str) -> str:
        for k in range(min(len(s), len(_HARMONY_MARKER) - 1), 0, -1):
            if _HARMONY_MARKER.startswith(s[-k:]):
                return s[:-k]
        return s

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while True:
            idx = self._buf.find(_HARMONY_MARKER)
            if idx == -1:
                safe = self._hold_partial(self._buf)
                if safe:
                    out.append((self._kind(), _HARMONY_CONTROL_RE.sub("", safe)))
                    self._buf = self._buf[len(safe):]
                break
            before = self._buf[:idx]
            if before:
                out.append((self._kind(), _HARMONY_CONTROL_RE.sub("", before)))
            rest = self._buf[idx + len(_HARMONY_MARKER):]
            m = re.match(r"\|?>\s*([a-zA-Z_]+)(?:[ \t]*\n|(?=<))", rest)
            if not m:
                self._buf = self._buf[idx:]
                break
            self._channel = m.group(1).lower()
            self._buf = rest[m.end():]
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        out = [(self._kind(), _HARMONY_CONTROL_RE.sub("", self._buf))]
        self._buf = ""
        return out

_TOOLCALL_PAIRS = (
    ("<|tool_call>", "<tool_call|>"),
    ("<start_function_call>", "<end_function_call>"),
    ("<tool_call>", "</tool_call>"),
    ("<|tool_call_start|>", "<|tool_call_end|>"),
)
_TOOLCALL_OPENERS = tuple(o for o, _ in _TOOLCALL_PAIRS)


def _strip_tool_call_markers(text: str) -> str:
    if not text or ("call:" not in text and "tool_call" not in text
                    and "function_call" not in text and '<|"|>' not in text):
        return text
    s = text
    for open_m, close_m in _TOOLCALL_PAIRS:
        while True:
            ci = s.find(close_m)
            if ci == -1:
                break
            oi = s.rfind(open_m, 0, ci)
            if oi == -1:
                oi = s.rfind("call:", 0, ci)
            if oi == -1:
                s = s[:ci] + s[ci + len(close_m):]
            else:
                s = s[:oi] + s[ci + len(close_m):]
    cut = None
    for open_m in _TOOLCALL_OPENERS:
        oi = s.find(open_m)
        if oi != -1:
            cut = oi if cut is None else min(cut, oi)
    if cut is not None:
        s = s[:cut]
    ci = s.find("call:")
    while ci != -1:
        j = ci + 5
        k = j
        while k < len(s) and (s[k].isalnum() or s[k] in "_-."):
            k += 1
        m = k
        while m < len(s) and s[m] == " ":
            m += 1
        if k > j and m < len(s) and s[m] == "{":
            s = s[:ci]
            break
        ci = s.find("call:", ci + 5)
    s = s.replace('<|"|>', '"')
    return s.strip() if s != text else text


def _build_chat_response(result: dict[str, Any], model_id: str, request_id: str) -> dict[str, Any]:
    function_calls = result.get("function_calls") or []
    tool_calls = _make_tool_calls(function_calls)
    has_tool_calls = bool(tool_calls)
    prefill = int(result.get("prefill_tokens") or 0)
    decode = int(result.get("decode_tokens") or 0)
    content, reasoning = _split_harmony_channels(result.get("response", "") or "")
    content = _strip_tool_call_markers(content)
    message: dict[str, Any] = {"role": "assistant", "content": None if has_tool_calls else content}
    if reasoning:
        message["reasoning_content"] = reasoning
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
        "cloud_handoff": bool(result.get("cloud_handoff", False)),
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
    splitter = _HarmonyStreamSplitter()

    def _delta_chunk(piece_kind: str, text: str) -> str:
        field = "reasoning_content" if piece_kind == "reasoning" else "content"
        return _event({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {field: text}, "logprobs": None, "finish_reason": None}],
        })

    tools_present = bool(tools)

    def _consume(pieces: list[tuple[str, str]]) -> list[str]:
        out: list[str] = []
        for piece_kind, piece_text in pieces:
            if not piece_text:
                continue
            if tools_present and piece_kind != "reasoning":
                continue
            out.append(_delta_chunk(piece_kind, piece_text))
        return out

    while True:
        kind, value = await queue.get()
        if kind == "token":
            for event in _consume(splitter.feed(value)):
                yield event
        elif kind == "done":
            result, error = value
            break

    for event in _consume(splitter.flush()):
        yield event

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
    if tools_present and not tool_calls:
        clean_content, _ = _split_harmony_channels((result or {}).get("response", "") or "")
        clean_content = _strip_tool_call_markers(clean_content)
        if clean_content:
            yield _delta_chunk("content", clean_content)
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
        "cloud_handoff": bool((result or {}).get("cloud_handoff", False)),
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
    auto_handoff: bool = True,
    confidence_threshold: float | None = None,
    cloud_timeout_ms: int | None = None,
) -> FastAPI:
    root = Path(weights_root) if weights_root is not None else default_weights_root()
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
    app.state.auto_handoff = auto_handoff
    app.state.confidence_threshold = confidence_threshold
    app.state.cloud_timeout_ms = cloud_timeout_ms

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
        if "max_tokens" not in options:
            options["max_tokens"] = info.context_length if info.context_length > 0 else 8192
        state = request.app.state
        if not state.auto_handoff:
            options["auto_handoff"] = False
        if state.confidence_threshold is not None:
            options["confidence_threshold"] = state.confidence_threshold
        if state.cloud_timeout_ms is not None:
            options["cloud_timeout_ms"] = state.cloud_timeout_ms
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
        if not inputs or any(not text for text in inputs):
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
