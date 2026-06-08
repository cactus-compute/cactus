from pathlib import Path
from types import SimpleNamespace
import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.testclient import TestClient

import cactus.server as server


def _bundle(root: Path, name: str, model_type: str, *, manifest: bool = True) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.txt").write_text(
        f"model_type={model_type}\ncontext_length=1024\n",
        encoding="utf-8",
    )
    if manifest:
        (path / "components").mkdir()
        (path / "components" / "manifest.json").write_text('{"components":[]}', encoding="utf-8")
    return path


def _app(tmp_path: Path, monkeypatch, *, preload: bool = False):
    _bundle(tmp_path, "llm", "gemma4")
    monkeypatch.setattr(server, "cactus_init", lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs))
    monkeypatch.setattr(server, "cactus_destroy", lambda handle: None)
    return server.create_app(weights_root=tmp_path, default_model="llm", preload=preload)


def test_models_lists_only_v2_bundles(tmp_path: Path, monkeypatch) -> None:
    _bundle(tmp_path, "valid", "gemma4")
    _bundle(tmp_path, "old_weights", "lfm2", manifest=False)
    monkeypatch.setattr(server, "cactus_init", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "cactus_destroy", lambda handle: None)

    app = server.create_app(weights_root=tmp_path, default_model="valid", preload=False)
    with TestClient(app) as client:
        data = client.get("/v1/models").json()["data"]

    assert [m["id"] for m in data] == ["valid"]
    assert data[0]["context_window"] == 1024


def test_chat_completion_shapes_openai_response(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch)

    def fake_complete(handle, messages, options, tools, callback):
        assert messages == [{"role": "user", "content": "hello"}]
        assert options["max_tokens"] == 3
        return {
            "success": True,
            "response": "hi",
            "function_calls": [],
            "prefill_tokens": 2,
            "decode_tokens": 1,
        }

    monkeypatch.setattr(server, "cactus_complete", fake_complete)
    monkeypatch.setattr(server, "cactus_reset", lambda handle: None)

    with TestClient(app) as client:
        res = client.post("/v1/chat/completions", json={
            "model": "llm",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
        })

    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def test_chat_tool_calls_are_translated(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "cactus_reset", lambda handle: None)
    monkeypatch.setattr(server, "cactus_complete", lambda *args, **kwargs: {
        "success": True,
        "response": "",
        "function_calls": [{"name": "lookup", "arguments": {"q": "x"}}],
        "prefill_tokens": 1,
        "decode_tokens": 1,
    })

    with TestClient(app) as client:
        res = client.post("/v1/chat/completions", json={
            "model": "llm",
            "messages": [{"role": "user", "content": "call tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            "tool_choice": "required",
        })

    choice = res.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'


def test_streaming_completion_error_terminates(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "cactus_reset", lambda handle: None)

    def fail_complete(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "cactus_complete", fail_complete)

    with TestClient(app) as client:
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "llm",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }) as res:
            body = "".join(res.iter_text())

    assert res.status_code == 200
    # OpenAI streaming spec carries errors in a `data:` JSON line, not in
    # `event:` lines. We also emit a final chunk with finish_reason so
    # clients don't hang before [DONE].
    assert '"error"' in body
    assert "boom" in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_transcription_rejects_non_wav(tmp_path: Path, monkeypatch) -> None:
    _bundle(tmp_path, "llm", "gemma4")
    _bundle(tmp_path, "stt", "parakeet_tdt")
    monkeypatch.setattr(server, "cactus_init", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "cactus_destroy", lambda handle: None)
    app = server.create_app(weights_root=tmp_path, default_model="llm", preload=False)

    with TestClient(app) as client:
        res = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", b"not wav", "audio/mpeg")},
            data={"model": "stt"},
        )

    assert res.status_code == 400
    assert "Only .wav" in res.json()["detail"]


def test_transcription_verbose_json_with_segments(tmp_path: Path, monkeypatch) -> None:
    _bundle(tmp_path, "llm", "gemma4")
    _bundle(tmp_path, "stt", "parakeet_tdt")
    monkeypatch.setattr(server, "cactus_init", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "cactus_destroy", lambda handle: None)
    monkeypatch.setattr(server, "cactus_reset", lambda handle: None)
    monkeypatch.setattr(server, "cactus_transcribe", lambda *args, **kwargs: {
        "success": True,
        "response": "hello",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
    })
    app = server.create_app(weights_root=tmp_path, default_model="llm", preload=False)

    with TestClient(app) as client:
        res = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"wav", "audio/wav")},
            data={
                "model": "stt",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
        )

    assert res.status_code == 200
    assert res.json()["segments"] == [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello"}]


def test_transcription_rejects_word_timestamps(tmp_path: Path, monkeypatch) -> None:
    _bundle(tmp_path, "llm", "gemma4")
    _bundle(tmp_path, "stt", "parakeet_tdt")
    monkeypatch.setattr(server, "cactus_init", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "cactus_destroy", lambda handle: None)
    app = server.create_app(weights_root=tmp_path, default_model="llm", preload=False)

    with TestClient(app) as client:
        res = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"wav", "audio/wav")},
            data={"model": "stt", "timestamp_granularities[]": "word"},
        )

    assert res.status_code == 400
    assert "Word timestamp" in res.json()["detail"]


def test_model_manager_does_not_evict_active_slots(tmp_path: Path, monkeypatch) -> None:
    for name in ("a", "b", "c"):
        _bundle(tmp_path, name, "gemma4")
    destroyed = []
    monkeypatch.setattr(server, "cactus_init", lambda path, **kwargs: Path(path).name)
    monkeypatch.setattr(server, "cactus_destroy", lambda handle: destroyed.append(handle))

    registry = server.ModelRegistry(tmp_path)
    manager = server.ModelManager(registry, max_warm=2)

    async def run():
        async with manager.acquire("a"):
            async with manager.acquire("b"):
                with pytest.raises(HTTPException) as exc:
                    async with manager.acquire("c"):
                        pass
                assert exc.value.status_code == 503
                assert destroyed == []

        async with manager.acquire("c"):
            pass

    asyncio.run(run())
    assert len(destroyed) == 1
