"""Class-based Python API for Cactus mirroring apple/Cactus.swift and android/Cactus.kt."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import cactus as _ffi


class CactusError(Exception):
    pass


class InitializationFailed(CactusError):
    pass


class CompletionFailed(CactusError):
    pass


class TranscriptionFailed(CactusError):
    pass


class EmbeddingFailed(CactusError):
    pass


class VADFailed(CactusError):
    pass


class InvalidResponse(CactusError):
    pass


class IndexOperationFailed(CactusError):
    pass


CactusError.InitializationFailed = InitializationFailed
CactusError.CompletionFailed = CompletionFailed
CactusError.TranscriptionFailed = TranscriptionFailed
CactusError.EmbeddingFailed = EmbeddingFailed
CactusError.VADFailed = VADFailed
CactusError.InvalidResponse = InvalidResponse
CactusError.IndexOperationFailed = IndexOperationFailed


@dataclass
class Message:
    role: str
    content: str = ""
    images: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = list(self.images)
        if self.audio:
            d["audio"] = list(self.audio)
        return d


@dataclass
class CompletionOptions:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    stop_sequences: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.0
    min_p: float | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop_sequences),
            "confidence_threshold": self.confidence_threshold,
        }
        if self.min_p is not None:
            d["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            d["repetition_penalty"] = self.repetition_penalty
        if self.enable_thinking is not None:
            d["enable_thinking"] = self.enable_thinking
        return d


@dataclass
class TranscriptionOptions:
    language: str | None = None
    translate_to_english: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"translate": self.translate_to_english}
        if self.language is not None:
            d["language"] = self.language
        return d


@dataclass
class VADOptions:
    threshold: float | None = None
    neg_threshold: float | None = None
    min_speech_duration_ms: int | None = None
    max_speech_duration_s: float | None = None
    min_silence_duration_ms: int | None = None
    speech_pad_ms: int | None = None
    window_size_samples: int | None = None
    sampling_rate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.neg_threshold is not None:
            d["neg_threshold"] = self.neg_threshold
        if self.min_speech_duration_ms is not None:
            d["min_speech_duration_ms"] = self.min_speech_duration_ms
        if self.max_speech_duration_s is not None:
            d["max_speech_duration_s"] = self.max_speech_duration_s
        if self.min_silence_duration_ms is not None:
            d["min_silence_duration_ms"] = self.min_silence_duration_ms
        if self.speech_pad_ms is not None:
            d["speech_pad_ms"] = self.speech_pad_ms
        if self.window_size_samples is not None:
            d["window_size_samples"] = self.window_size_samples
        if self.sampling_rate is not None:
            d["sampling_rate"] = self.sampling_rate
        return d


@dataclass
class CompletionResult:
    response: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    confidence: float = 1.0
    cloud_handoff: bool = False
    function_calls: list[dict[str, Any]] | None = None
    segments: list[dict[str, Any]] | None = None
    thinking: str | None = None

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CompletionResult":
        prompt_tokens = int(d.get("prefill_tokens", 0) or 0)
        completion_tokens = int(d.get("decode_tokens", 0) or 0)
        return cls(
            response=str(d.get("response", "") or ""),
            tokens=list(d.get("tokens", []) or []),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(d.get("total_tokens", prompt_tokens + completion_tokens) or 0),
            time_to_first_token_ms=float(d.get("time_to_first_token_ms", 0.0) or 0.0),
            total_time_ms=float(d.get("total_time_ms", 0.0) or 0.0),
            prefill_tps=float(d.get("prefill_tps", 0.0) or 0.0),
            decode_tps=float(d.get("decode_tps", 0.0) or 0.0),
            confidence=float(d.get("confidence", 1.0) if d.get("confidence") is not None else 1.0),
            cloud_handoff=bool(d.get("cloud_handoff", False) or False),
            function_calls=d.get("function_calls"),
            segments=d.get("segments"),
            thinking=d.get("thinking"),
        )


@dataclass
class TranscriptionResult:
    text: str = ""
    language: str | None = None
    segments: list[dict[str, Any]] | None = None
    duration_ms: float = 0.0
    total_time_ms: float = 0.0

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            text=str(d.get("response", "") or ""),
            language=d.get("language"),
            segments=d.get("segments"),
            duration_ms=float(d.get("duration_ms", 0.0) or 0.0),
            total_time_ms=float(d.get("total_time_ms", 0.0) or 0.0),
        )


@dataclass
class VADSegment:
    start: int = 0
    end: int = 0

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "VADSegment":
        return cls(start=int(d.get("start", 0) or 0), end=int(d.get("end", 0) or 0))


@dataclass
class VADResult:
    segments: list[VADSegment] = field(default_factory=list)
    total_time_ms: float = 0.0
    ram_usage_mb: float = 0.0

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "VADResult":
        segs = [VADSegment.from_json(s) for s in (d.get("segments") or [])]
        return cls(
            segments=segs,
            total_time_ms=float(d.get("total_time_ms", 0.0) or 0.0),
            ram_usage_mb=float(d.get("ram_usage_mb", 0.0) or 0.0),
        )


@dataclass
class IndexResult:
    id: int
    score: float


def _parse_json(s: str) -> dict[str, Any]:
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        raise InvalidResponse(f"Invalid JSON response: {e}") from e
    if not isinstance(obj, dict):
        raise InvalidResponse("Expected JSON object")
    return obj


def _pcm_bytes(data: bytes | bytearray | memoryview | None) -> bytes | None:
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return bytes(memoryview(data))


def _json_or_none(d: dict[str, Any] | None) -> str | None:
    if d is None:
        return None
    return json.dumps(d)


class Cactus:
    def __init__(self, model_path: str, corpus_dir: str | None = None, cache_index: bool = False) -> None:
        try:
            self._handle = _ffi.cactus_init(model_path, corpus_dir, cache_index)
        except RuntimeError as e:
            raise InitializationFailed(str(e)) from e
        self._closed = False

    def __enter__(self) -> "Cactus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check(self) -> None:
        if self._closed or self._handle is None:
            raise CactusError("Cactus has been closed")

    def close(self) -> None:
        if not self._closed and getattr(self, "_handle", None):
            _ffi.cactus_destroy(self._handle)
            self._handle = None
            self._closed = True

    def reset(self) -> None:
        self._check()
        _ffi.cactus_reset(self._handle)

    def stop(self) -> None:
        self._check()
        _ffi.cactus_stop(self._handle)

    def complete(
        self,
        messages_or_prompt: str | list[Message] | list[dict[str, Any]],
        options: CompletionOptions | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str, int], None] | None = None,
        pcm_data: bytes | bytearray | memoryview | None = None,
    ) -> CompletionResult:
        self._check()
        opts = options or CompletionOptions()

        if isinstance(messages_or_prompt, str):
            msgs: list[dict[str, Any]] = [Message.user(messages_or_prompt).to_dict()]
        else:
            msgs = [m.to_dict() if isinstance(m, Message) else dict(m) for m in messages_or_prompt]

        messages_json = json.dumps(msgs)
        options_json = json.dumps(opts.to_dict())
        tools_json = _json_or_none(tools) if tools is None or isinstance(tools, list) else _json_or_none(list(tools))
        pcm = _pcm_bytes(pcm_data)

        try:
            raw = _ffi.cactus_complete(
                self._handle, messages_json, options_json, tools_json, on_token, pcm
            )
        except RuntimeError as e:
            raise CompletionFailed(str(e)) from e

        d = _parse_json(raw)
        err = d.get("error")
        if isinstance(err, str) and err:
            raise CompletionFailed(err)
        return CompletionResult.from_json(d)

    def transcribe(
        self,
        audio: str | bytes | bytearray | memoryview,
        prompt: str = "",
        options: TranscriptionOptions | None = None,
        on_token: Callable[[str, int], None] | None = None,
    ) -> TranscriptionResult:
        self._check()
        opts = options or TranscriptionOptions()
        options_json = json.dumps(opts.to_dict())
        if isinstance(audio, str):
            audio_path: str | None = audio
            pcm: bytes | None = None
        else:
            audio_path = None
            pcm = _pcm_bytes(audio)

        try:
            raw = _ffi.cactus_transcribe(
                self._handle, audio_path, prompt or None, options_json, on_token, pcm
            )
        except RuntimeError as e:
            raise TranscriptionFailed(str(e)) from e

        d = _parse_json(raw)
        err = d.get("error")
        if isinstance(err, str) and err:
            raise TranscriptionFailed(err)
        return TranscriptionResult.from_json(d)

    def embed(self, text: str, normalize: bool = True) -> list[float]:
        self._check()
        try:
            return _ffi.cactus_embed(self._handle, text, normalize)
        except RuntimeError as e:
            raise EmbeddingFailed(str(e)) from e

    def image_embed(self, image_path: str) -> list[float]:
        self._check()
        try:
            return _ffi.cactus_image_embed(self._handle, image_path)
        except RuntimeError as e:
            raise EmbeddingFailed(str(e)) from e

    def audio_embed(self, audio_path: str) -> list[float]:
        self._check()
        try:
            return _ffi.cactus_audio_embed(self._handle, audio_path)
        except RuntimeError as e:
            raise EmbeddingFailed(str(e)) from e

    def vad(
        self,
        audio: str | bytes | bytearray | memoryview,
        options: VADOptions | None = None,
    ) -> VADResult:
        self._check()
        opts = options or VADOptions()
        opts_dict = opts.to_dict()
        options_json = json.dumps(opts_dict) if opts_dict else None
        if isinstance(audio, str):
            audio_path: str | None = audio
            pcm: bytes | None = None
        else:
            audio_path = None
            pcm = _pcm_bytes(audio)

        try:
            raw = _ffi.cactus_vad(self._handle, audio_path, options_json, pcm)
        except RuntimeError as e:
            raise VADFailed(str(e)) from e

        d = _parse_json(raw)
        err = d.get("error")
        if isinstance(err, str) and err:
            raise VADFailed(err)
        return VADResult.from_json(d)

    def rag_query(self, query: str, top_k: int = 5) -> str:
        self._check()
        try:
            return _ffi.cactus_rag_query(self._handle, query, top_k)
        except RuntimeError as e:
            raise CompletionFailed(str(e)) from e

    def tokenize(self, text: str) -> list[int]:
        self._check()
        try:
            return _ffi.cactus_tokenize(self._handle, text)
        except RuntimeError as e:
            raise CompletionFailed(str(e)) from e

    def score_window(self, tokens: list[int] | Iterable[int], start: int, end: int, context: int) -> str:
        self._check()
        token_list = list(tokens)
        try:
            return _ffi.cactus_score_window(self._handle, token_list, start, end, context)
        except RuntimeError as e:
            raise CompletionFailed(str(e)) from e

    def create_stream_transcriber(self, options: TranscriptionOptions | None = None) -> "StreamTranscriber":
        self._check()
        opts = options or TranscriptionOptions()
        options_json = json.dumps(opts.to_dict())
        try:
            stream_handle = _ffi.cactus_stream_transcribe_start(self._handle, options_json)
        except RuntimeError as e:
            raise TranscriptionFailed(str(e)) from e
        return StreamTranscriber(stream_handle)


class StreamTranscriber:
    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._closed = False

    def __enter__(self) -> "StreamTranscriber":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check(self) -> None:
        if self._closed or self._handle is None:
            raise TranscriptionFailed("Stream transcriber has been closed")

    def process(self, pcm_data: bytes | bytearray | memoryview) -> TranscriptionResult:
        self._check()
        pcm = _pcm_bytes(pcm_data) or b""
        try:
            raw = _ffi.cactus_stream_transcribe_process(self._handle, pcm)
        except RuntimeError as e:
            raise TranscriptionFailed(str(e)) from e
        d = _parse_json(raw)
        err = d.get("error")
        if isinstance(err, str) and err:
            raise TranscriptionFailed(err)
        return TranscriptionResult.from_json(d)

    def stop(self) -> TranscriptionResult:
        self._check()
        try:
            raw = _ffi.cactus_stream_transcribe_stop(self._handle)
        finally:
            self._handle = None
            self._closed = True
        d = _parse_json(raw)
        err = d.get("error")
        if isinstance(err, str) and err:
            raise TranscriptionFailed(err)
        return TranscriptionResult.from_json(d)

    def close(self) -> None:
        if not self._closed and getattr(self, "_handle", None):
            try:
                _ffi.cactus_stream_transcribe_stop(self._handle)
            except Exception:
                pass
            self._handle = None
            self._closed = True


class CactusIndex:
    def __init__(self, index_dir: str, embedding_dim: int) -> None:
        try:
            self._handle = _ffi.cactus_index_init(index_dir, embedding_dim)
        except RuntimeError as e:
            raise IndexOperationFailed(str(e)) from e
        self._closed = False

    def __enter__(self) -> "CactusIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check(self) -> None:
        if self._closed or self._handle is None:
            raise IndexOperationFailed("Index has been closed")

    def add(
        self,
        ids: list[int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[str] | None = None,
    ) -> None:
        self._check()
        if not (len(ids) == len(documents) == len(embeddings)):
            raise IndexOperationFailed("ids, documents, and embeddings must have the same length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise IndexOperationFailed("metadatas must match ids length")
        try:
            _ffi.cactus_index_add(self._handle, ids, documents, embeddings, metadatas)
        except RuntimeError as e:
            raise IndexOperationFailed(str(e)) from e

    def delete(self, ids: list[int]) -> None:
        self._check()
        try:
            _ffi.cactus_index_delete(self._handle, ids)
        except RuntimeError as e:
            raise IndexOperationFailed(str(e)) from e

    def query(self, embedding: list[float], top_k: int = 5) -> list[IndexResult]:
        self._check()
        options_json = json.dumps({"top_k": top_k})
        try:
            raw = _ffi.cactus_index_query(self._handle, embedding, options_json)
        except RuntimeError as e:
            raise IndexOperationFailed(str(e)) from e
        d = _parse_json(raw)
        results = d.get("results") or []
        return [IndexResult(id=int(r.get("id", 0)), score=float(r.get("score", 0.0))) for r in results][:top_k]

    def compact(self) -> None:
        self._check()
        try:
            _ffi.cactus_index_compact(self._handle)
        except RuntimeError as e:
            raise IndexOperationFailed(str(e)) from e

    def close(self) -> None:
        if not self._closed and getattr(self, "_handle", None):
            _ffi.cactus_index_destroy(self._handle)
            self._handle = None
            self._closed = True
