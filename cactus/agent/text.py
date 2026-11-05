from __future__ import annotations

import ctypes
import json
import os
import warnings

from .._telemetry import track as _track
from . import runtime
from .runtime import synchronized
from .tools import build_schema, pydantic_schema, _is_pydantic_model
from .grounding import (
    _jsonable, _source_years,
    _with_date_fact, _annotate_ungrounded, _validate_extraction,
    _ungrounded_paths, _grounded_number_paths,
)

UNRESET_TURNS = 4


_CACT_GENERATIONS = {
    0x05E12A83: 2,
    0x05E12A84: 3,
}


def _weight_generation(path):
    with open(path, "rb") as handle:
        tag_bytes = handle.read(4)
    if len(tag_bytes) != 4:
        raise RuntimeError(f"{path} is not a complete .cact archive")
    tag = int.from_bytes(tag_bytes, "little")
    try:
        return _CACT_GENERATIONS[tag]
    except KeyError as exc:
        raise RuntimeError(
            f"{path} has unknown .cact format tag 0x{tag:08x}; "
            "cannot choose a compatible Needle engine") from exc


_CACT_HEADER = "<48If"
_CACT_RECORD = "<BBHIIIIQQII"
_CACT_FP16, _CACT_RAW = 1, 4
_CONFIDENCE_HEAD_CODE = 2


def _confidence_head_present(path):
    """Whether a Needle 3 archive carries a confidence head, read from its head manifest."""
    import struct

    with open(path, "rb") as handle:
        header = handle.read(struct.calcsize(_CACT_HEADER))
        if len(header) < struct.calcsize(_CACT_HEADER):
            return False
        fields = struct.unpack(_CACT_HEADER, header)
        num_tensors, codebook = fields[1], fields[2]
        handle.seek(codebook * 4, 1)
        size = struct.calcsize(_CACT_RECORD)
        records = [struct.unpack(_CACT_RECORD, handle.read(size)) for _ in range(num_tensors)]
        if not records or records[-1][0] != _CACT_RAW:
            return False
        for heads in (1, 2, 3):
            index = num_tensors - 2 - 6 * heads
            if index < 0:
                continue
            dtype, ndim, _, length = records[index][:4]
            if dtype != _CACT_FP16 or ndim != 1 or length != heads:
                continue
            handle.seek(records[index][7])
            codes = struct.unpack(f"<{heads}e", handle.read(2 * heads))
            return _CONFIDENCE_HEAD_CODE in {int(round(c)) for c in codes}
    return False


class Cactus:
    def __init__(self, tools=None, system=None, weights=None, tool_index_path=None, buffer_size=65536,
                 auto_date=True, generation=None, stateless=False):
        self._functions = {}
        self._stateless = bool(stateless)
        self._turns = 0
        self._weights = os.fspath(weights) if weights is not None else None
        self._tuned = self._weights is not None
        if self._tuned:
            self._generation = _weight_generation(self._weights)
        else:
            self._generation = int(generation or 3)
        from . import fetch
        fetch._engine(self._generation)
        self._worker = None
        self._closed = False
        self._calibrated = (not self._tuned
                            or (self._generation >= 3 and _confidence_head_present(self._weights)))
        if not self._calibrated:
            warnings.warn("these weights carry no confidence head trained for them, so this "
                          "agent reports confidence as None; platform fine-tunes keep the head",
                          stacklevel=2)
        self._system_text = _with_date_fact(system or "") if auto_date else (system or "")
        self._system = self._system_text.encode("utf-8")
        tools_json = tools if isinstance(tools, str) else json.dumps(self._resolve(tools))
        self._tools_json = tools_json.encode("utf-8")
        try:
            parsed_tools = json.loads(tools_json)
            self._n_tools = len(parsed_tools)
        except (json.JSONDecodeError, TypeError):
            parsed_tools, self._n_tools = [], None
        self._tool_schemas = [entry for entry in (parsed_tools or [])
                              if isinstance(entry, dict)]
        self._seen_years = set()
        self._tool_index_path = (os.fspath(tool_index_path).encode("utf-8")
                                 if tool_index_path else None)
        self._buffer = ctypes.create_string_buffer(buffer_size)
        if self._weights:
            from .._worker import FineTuneWorker
            self._worker = FineTuneWorker(
                runtime._library_path(self._generation), self._weights,
                self._system.decode("utf-8"), self._tools_json.decode("utf-8"),
                os.fspath(tool_index_path) if tool_index_path else None,
                buffer_size)
        else:
            self._bind()

    @synchronized
    def _bind(self):
        if self._closed:
            raise RuntimeError("Cactus instance is closed")
        if self._worker is not None:
            return
        generation = self._generation
        lib = runtime._lib(generation)
        runtime._load_base(lib, generation)
        if runtime._active.get(generation) is self:
            return
        if lib.cactus_init(self._system, self._tools_json, self._tool_index_path) < 0:
            runtime._active.pop(generation, None)
            raise RuntimeError("cactus_init failed")
        runtime._active[generation] = self
        self._seen_years = set()

    def _resolve(self, tools):
        schemas = []
        for entry in tools or []:
            if _is_pydantic_model(entry):
                schema = pydantic_schema(entry)
                self._functions[schema["name"]] = entry
                schemas.append(schema)
            elif callable(entry):
                schema = getattr(entry, "_cactus_tool", None) or build_schema(entry)
                self._functions[schema["name"]] = entry
                schemas.append(schema)
            elif isinstance(entry, dict):
                schemas.append(entry)
        return schemas

    def _track_props(self):
        return {"n_tools": self._n_tools, "tuned": self._tuned,
                "generation": self._generation}

    @synchronized
    def complete(self, text: str = "", max_new_tokens: int = 512, *, audio=None, language=None,
                 keywords=None, word_timestamps=False) -> dict:
        _track("complete", self._track_props())
        if self._stateless:
            self.reset()
        if audio is not None and text:
            raise ValueError("complete accepts either text or audio")
        self._count_query()
        if audio is not None:
            return self._complete_audio(audio, max_new_tokens, language, keywords, word_timestamps)
        return self._complete(text, max_new_tokens)

    def _count_query(self):
        self._turns += 1
        if self._turns == UNRESET_TURNS + 1:
            warnings.warn(
                f"{UNRESET_TURNS} queries on this agent without reset(): every turn stays in "
                "the conversation, so unrelated queries lose accuracy and confidence. "
                "Call reset() between independent queries, or construct with stateless=True",
                stacklevel=3)

    @synchronized
    def _complete(self, text: str, max_new_tokens: int = 512,
                  ground: bool = True) -> dict:
        self._bind()
        self._seen_years |= _source_years(text or "")
        if self._worker is not None:
            raw = self._worker.complete(text, max_new_tokens)
        else:
            lib = runtime._lib(self._generation)
            audio = (None, 0)
            rc = lib.cactus_complete(
                text.encode("utf-8"), *audio, int(max_new_tokens), self._buffer,
                len(self._buffer))
            if rc < 0:
                detail = self._buffer.value.decode("utf-8", "replace")
                raise RuntimeError(detail or f"cactus_complete failed (code {rc})")
            raw = self._buffer.value.decode("utf-8")
        return self._response(raw, text, ground)

    def _complete_audio(self, audio, max_new_tokens, language, keywords, word_timestamps):
        from .audio import _samples, _options
        from .whistle import Whistle, _weights_path
        self._bind()
        samples, count = _samples(audio)
        if keywords is not None and not isinstance(keywords, str):
            keywords = "\n".join(keywords)
        if self._worker is not None:
            packed = ctypes.string_at(samples, count * ctypes.sizeof(ctypes.c_float))
            raw = self._worker.complete_audio(packed, _weights_path(), language, keywords, word_timestamps, max_new_tokens)
        else:
            speech = Whistle()
            lib = speech._bind()
            language, keywords = _options(language, keywords)
            lib.cactus_set_audio(language, keywords, int(bool(word_timestamps)))
            code = lib.cactus_complete(None, samples, count, int(max_new_tokens), self._buffer, len(self._buffer))
            if code < 0:
                raise RuntimeError(lib.cactus_last_error().decode("utf-8", "replace"))
            raw = self._buffer.value.decode("utf-8")
        response = json.loads(raw)
        heard = response.get("audio_text", "")
        self._seen_years |= _source_years(heard)
        return self._response(raw, heard)

    def _response(self, raw, text, ground=True):
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"engine returned an unparseable envelope ({err}); this is an "
                f"engine bug - please report it with the prompt and schema") from err
        if not self._calibrated:
            response["confidence"] = None
        if ground:
            _annotate_ungrounded(response, self._tool_schemas, self._seen_years,
                                 self._system_text, text)
        return response

    @synchronized
    def embed(self, text: str = "") -> list[float]:
        self._bind()
        if self._worker is not None:
            return self._worker.embed(text)
        lib = runtime._lib(self._generation)
        dim = lib.cactus_embed(text.encode("utf-8"), None, 0, None, 0)
        if dim <= 0:
            raise RuntimeError(f"cactus_embed failed (code {dim})")
        output = (ctypes.c_float * dim)()
        rc = lib.cactus_embed(text.encode("utf-8"), None, 0, output, dim)
        if rc != dim:
            raise RuntimeError(f"cactus_embed failed (code {rc})")
        return list(output)

    def run(self, query: str = "", max_steps: int = 8,
            max_new_tokens: int = 512, strict: bool = True) -> dict:
        _track("run", self._track_props())
        if self._stateless:
            self.reset()
        self._count_query()
        response = self._complete(query, max_new_tokens)
        executed = []
        for _ in range(max_steps):
            calls = response.get("function_calls") or []
            if response.get("type") != "call" or not calls:
                break
            ungrounded = _ungrounded_paths(response)
            results = []
            for call in calls:
                name = str(call.get("name"))
                fabricated = sorted(ungrounded.get(name, ()))
                if strict and fabricated:
                    grounded = _grounded_number_paths(
                        call.get("arguments") or {}, query, self._system_text)
                    fabricated = [path for path in fabricated
                                  if path not in grounded]
                if strict and fabricated:
                    results.append({"error": "ungrounded " + ", ".join(fabricated)})
                    continue
                fn = self._functions.get(call.get("name"))
                if fn is None:
                    results.append({"error": "unknown tool: " + name})
                    continue
                try:
                    results.append(fn(**(call.get("arguments") or {})))
                except Exception as exc:
                    results.append({"error": str(exc)})
            executed.extend(results)
            response = self._complete(json.dumps(results, default=_jsonable),
                                      max_new_tokens, ground=False)
        response["results"] = executed
        return response

    def extract(self, text: str, schema: type | dict, max_new_tokens: int = 512,
                strict: bool = True) -> object:
        return extract(text, schema, system=self._system_text or None,
                       max_new_tokens=max_new_tokens, weights=self._weights,
                       strict=strict, generation=self._generation)

    @synchronized
    def reset(self):
        self._bind()
        self._turns = 0
        if self._worker is not None:
            self._worker.reset()
        else:
            runtime._lib(self._generation).cactus_reset()
        self._seen_years = set()

    @synchronized
    def close(self):
        if self._worker is not None:
            try:
                self._worker.close()
            finally:
                self._worker = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def extract(text: str, schema: type | dict, system: str | None = None,
            max_new_tokens: int = 512, weights: str | None = None,
            strict: bool = True, generation: int | None = None) -> object:
    """One-shot structured extraction using the matching native engine.

    With ``strict=True`` (the default), temporal values that contradict a literal
    year in the input, plus engine-reported fabricated or negated values, raise
    :class:`ExtractionValidationError` instead of being returned silently.
    """
    selected = weights
    generation = _weight_generation(selected) if selected else int(generation or 3)
    _track("extract", {"n_tools": 1, "tuned": bool(selected),
                       "generation": generation})
    agent = Cactus(tools=[schema], system=system, weights=selected, generation=generation)


    facts = agent._system_text
    try:
        response = agent._complete(text, max_new_tokens)
    finally:
        agent.close()
    calls = response.get("function_calls") or response.get("suppressed_calls") or []
    if not calls:
        return None
    arguments = calls[0].get("arguments") or {}
    if strict:
        _validate_extraction(text, schema, arguments, response, facts)
    return schema(**arguments) if _is_pydantic_model(schema) else arguments
