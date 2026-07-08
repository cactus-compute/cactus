from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from .bundles import (
    PROJECT_ROOT, WEIGHTS, _find_bundle, _iter_bundle_candidates,
    _read_model_type, _valid_bundle,
)

ASSETS = PROJECT_ROOT / "cactus-engine" / "tests" / "assets"
LLM_TYPES = {"gemma", "gemma3n", "gemma4", "lfm2", "qwen", "qwen3p5", "needle", "youtu"}
STT_TYPES = {"whisper", "parakeet_tdt", "parakeet-tdt"}


def _default_llm_bundle() -> Path:
    """Locate the canonical gemma-4-e2b-it LLM bundle under whichever convention
    it was built with (bare `gemma-4-e2b-it` or suffixed `...-cq4`)."""
    for candidate in _iter_bundle_candidates("gemma-4-e2b-it"):
        if _valid_bundle(candidate) and _read_model_type(candidate) in LLM_TYPES:
            return candidate
    return WEIGHTS / "gemma-4-e2b-it"  # fall through so _require_bundle reports it


def _require_bundle(relative: Path, types: set[str]) -> Path:
    candidate = PROJECT_ROOT / relative
    if not candidate.exists():
        pytest.fail(
            f"Live-test model not found: {relative}\n"
            f"Prepare it with `cactus convert google/gemma-4-E2B-it` from {PROJECT_ROOT}."
        )
    if not _valid_bundle(candidate):
        pytest.fail(
            f"Live-test model is not a prepared v2 bundle: {relative}\n"
            "Expected config.txt and components/manifest.json.\n"
            f"Prepare it with `cactus convert google/gemma-4-E2B-it` from {PROJECT_ROOT}."
        )
    model_type = _read_model_type(candidate)
    if model_type not in types:
        pytest.fail(
            f"Live-test model has unsupported type {model_type!r}: {relative}\n"
            f"Expected one of: {sorted(types)}."
        )
    return candidate


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server(bundle: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    python_path = str(PROJECT_ROOT / "python")
    env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cactus",
            "serve",
            str(bundle),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_ready(proc: subprocess.Popen, base_url: str) -> None:
    deadline = time.time() + 120
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"server exited before becoming ready:\n{output}")
        try:
            with httpx.Client(timeout=2) as client:
                res = client.get(f"{base_url}/v1/models")
            if res.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    pytest.fail(f"server did not become ready: {last_error}")


@contextmanager
def _serve(bundle: Path):
    port = _free_port()
    proc = _start_server(bundle, port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(proc, base_url)
        yield base_url, bundle.name
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture(scope="module")
def live_server():
    bundle = _default_llm_bundle()
    _require_bundle(bundle, LLM_TYPES)
    with _serve(bundle) as server:
        yield server


def test_live_models(live_server) -> None:
    base_url, _ = live_server
    res = httpx.get(f"{base_url}/v1/models", timeout=10)
    assert res.status_code == 200
    models = res.json()["data"]
    assert models
    for model in models:
        bundle = WEIGHTS / model["id"]
        assert _valid_bundle(bundle)


def test_live_chat_rejects_unknown_model(live_server) -> None:
    base_url, _ = live_server
    res = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "definitely-not-a-live-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
        timeout=30,
    )
    assert res.status_code == 404


def test_live_chat_completion(live_server) -> None:
    base_url, model = live_server
    res = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "Reply with one word."}], "max_tokens": 4},
        timeout=120,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] > 0


def test_live_chat_text_content_parts(live_server) -> None:
    base_url, model = live_server
    res = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly "},
                    {"type": "text", "text": "OK"},
                ],
            }],
            "max_tokens": 8,
            "temperature": 0.0,
        },
        timeout=180,
    )
    assert res.status_code == 200
    assert res.json()["choices"][0]["message"]["content"]


def test_live_long_gemma_generation(live_server) -> None:
    base_url, model = live_server
    res = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": (
                    "Write a numbered list of ten short items about why local HTTP "
                    "inference servers need end-to-end tests. Keep each item concise."
                ),
            }],
            "max_tokens": 96,
            "temperature": 0.0,
        },
        timeout=180,
    )
    assert res.status_code == 200
    body = res.json()
    text = body["choices"][0]["message"]["content"]
    assert isinstance(text, str)
    assert len(text.split()) >= 20
    assert body["usage"]["completion_tokens"] >= 16
    assert body["choices"][0]["finish_reason"] == "stop"


def test_live_multi_turn_conversation(live_server) -> None:
    base_url, model = live_server
    messages = [
        {"role": "system", "content": "You answer directly and remember the user's chosen keyword."},
        {"role": "user", "content": "My keyword is cactus. Reply with only: remembered."},
        {"role": "assistant", "content": "remembered."},
        {"role": "user", "content": "What keyword did I give you? Reply with only the keyword."},
    ]
    res = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": 12,
            "temperature": 0.0,
        },
        timeout=180,
    )
    assert res.status_code == 200
    body = res.json()
    text = body["choices"][0]["message"]["content"].lower()
    assert "cactus" in text
    assert body["usage"]["prompt_tokens"] > body["usage"]["completion_tokens"]


def test_live_tool_call_completion(live_server) -> None:
    base_url, model = live_server
    res = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Use the weather tool for Paris."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "tool_choice": "required",
            "max_tokens": 64,
            "temperature": 0.0,
        },
        timeout=180,
    )
    assert res.status_code == 200
    body = res.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] in {"tool_calls", "stop"}
    if choice["finish_reason"] == "tool_calls":
        calls = choice["message"]["tool_calls"]
        assert calls
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"]


def test_live_concurrent_chat_requests(live_server) -> None:
    base_url, model = live_server

    def request(i: int) -> int:
        res = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"Reply with the number {i}."}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=180,
        )
        assert res.status_code == 200
        assert res.json()["choices"][0]["message"]["content"]
        return res.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(request, [1, 2])) == [200, 200]


def test_live_chat_stream(live_server) -> None:
    base_url, model = live_server
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4, "stream": True},
        timeout=120,
    ) as res:
        assert res.status_code == 200
        text = "".join(res.iter_text())
    assert "chat.completion.chunk" in text
    assert "data: [DONE]" in text


def test_live_long_streaming_generation(live_server) -> None:
    base_url, model = live_server
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Write one short paragraph about robust server testing."}],
            "max_tokens": 64,
            "stream": True,
        },
        timeout=180,
    ) as res:
        assert res.status_code == 200
        text = "".join(res.iter_text())
    assert "chat.completion.chunk" in text
    assert text.count('"delta"') >= 3
    assert "data: [DONE]" in text


def test_live_transcription_wav(live_server) -> None:
    base_url, _ = live_server
    stt = _find_bundle(["parakeet-tdt-0.6b-v3-transpiled"], STT_TYPES)
    audio = ASSETS / "test.wav"
    assert audio.exists()
    with audio.open("rb") as f:
        res = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            data={"model": stt.name, "response_format": "json"},
            files={"file": ("test.wav", f, "audio/wav")},
            timeout=180,
        )
    assert res.status_code == 200
    assert res.json()["text"]


def test_live_transcription_text_response(live_server) -> None:
    base_url, _ = live_server
    stt = _find_bundle(["parakeet-tdt-0.6b-v3-transpiled"], STT_TYPES)
    audio = ASSETS / "test.wav"
    with audio.open("rb") as f:
        res = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            data={"model": stt.name, "response_format": "text"},
            files={"file": ("test.wav", f, "audio/wav")},
            timeout=180,
        )
    assert res.status_code == 200
    assert res.text.strip()
    assert res.headers["content-type"].startswith("text/plain")


def test_live_transcription_verbose_json_segments(live_server) -> None:
    base_url, _ = live_server
    stt = _find_bundle(["parakeet-tdt-0.6b-v3-transpiled"], STT_TYPES)
    audio = ASSETS / "test.wav"
    with audio.open("rb") as f:
        res = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            data={
                "model": stt.name,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            files={"file": ("test.wav", f, "audio/wav")},
            timeout=180,
        )
    assert res.status_code == 200
    body = res.json()
    assert body["text"]
    assert "segments" in body


def test_live_transcription_rejects_unsupported_format(live_server) -> None:
    base_url, _ = live_server
    stt = _find_bundle(["parakeet-tdt-0.6b-v3-transpiled"], STT_TYPES)
    audio = ASSETS / "test.wav"
    with audio.open("rb") as f:
        res = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            data={"model": stt.name, "response_format": "srt"},
            files={"file": ("test.wav", f, "audio/wav")},
            timeout=30,
        )
    assert res.status_code == 400
    assert "Unsupported transcription response_format" in json.dumps(res.json())


def test_live_transcription_rejects_word_timestamps(live_server) -> None:
    base_url, _ = live_server
    stt = _find_bundle(["parakeet-tdt-0.6b-v3-transpiled"], STT_TYPES)
    audio = ASSETS / "test.wav"
    with audio.open("rb") as f:
        res = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            data={
                "model": stt.name,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
            files={"file": ("test.wav", f, "audio/wav")},
            timeout=30,
        )
    assert res.status_code == 400
    assert "Word timestamp" in json.dumps(res.json())


def test_live_transcription_rejects_non_wav(live_server) -> None:
    base_url, _ = live_server
    stt = _find_bundle(["parakeet-tdt-0.6b-v3-transpiled"], STT_TYPES)
    res = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        data={"model": stt.name},
        files={"file": ("test.mp3", b"nope", "audio/mpeg")},
        timeout=30,
    )
    assert res.status_code == 400
    assert "Only .wav" in json.dumps(res.json())


EMBED_TYPES = {"bert", "nomic"}


def _find_embed_bundle() -> Path:
    for candidate in sorted(WEIGHTS.iterdir()) if WEIGHTS.exists() else []:
        if candidate.is_dir() and _valid_bundle(candidate) and _read_model_type(candidate) in EMBED_TYPES:
            return candidate
    raise RuntimeError(
        "No embedding (nomic/bert) bundle under weights/; "
        "run `cactus convert nomic-ai/nomic-embed-text-v2-moe`"
    )


def _supports_embedding(bundle: Path) -> bool:
    try:
        data = json.loads((bundle / "components" / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    names = {str(c.get("component", "")) for c in data.get("components", []) if isinstance(c, dict)}
    return "text_embedding" in names or "decoder_embed_chunk" in names


def _find_non_embed_llm_bundle() -> Path | None:
    for candidate in sorted(WEIGHTS.iterdir()) if WEIGHTS.exists() else []:
        if (candidate.is_dir() and _valid_bundle(candidate)
                and _read_model_type(candidate) in LLM_TYPES
                and not _supports_embedding(candidate)):
            return candidate
    return None


@pytest.fixture(scope="module")
def embed_server():
    with _serve(_find_embed_bundle()) as server:
        yield server


@pytest.fixture(scope="module")
def non_embed_server():
    bundle = _find_non_embed_llm_bundle()
    if bundle is None:
        pytest.skip("No non-embedding LLM bundle under weights/")
    with _serve(bundle) as server:
        yield server


def test_live_embeddings_string(embed_server) -> None:
    base_url, model = embed_server
    res = httpx.post(
        f"{base_url}/v1/embeddings",
        json={"model": model, "input": "Paris is the capital of France."},
        timeout=60,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) > 0
    assert all(isinstance(x, float) for x in body["data"][0]["embedding"][:8])


def test_live_embeddings_list(embed_server) -> None:
    base_url, model = embed_server
    res = httpx.post(
        f"{base_url}/v1/embeddings",
        json={"model": model, "input": ["hello world", "a second sentence"]},
        timeout=60,
    )
    assert res.status_code == 200, res.text
    data = res.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert len(data[0]["embedding"]) == len(data[1]["embedding"]) > 0


def test_live_embeddings_rejects_llm_model(non_embed_server) -> None:
    base_url, model = non_embed_server
    res = httpx.post(
        f"{base_url}/v1/embeddings",
        json={"model": model, "input": "hello"},
        timeout=30,
    )
    assert res.status_code == 400
    assert "not an embedding model" in json.dumps(res.json())
