from cactus.cli import speech
from cactus.agent import runtime
import array
import json
import math
import os
import struct
import sys
import wave

import pytest

from conftest import _engine_available


def _whistle_weights():
    from cactus.agent import fetch

    return os.environ.get("CACTUS_WHISTLE_WEIGHTS") or os.path.join(fetch.cache_dir(fetch.WHISTLE), fetch.base_weights(fetch.WHISTLE))


requires_whistle = pytest.mark.skipif(not (_engine_available(3) and os.path.exists(_whistle_weights())),
                                      reason="Whistle engine or whistle.cact not installed (auto-fetched from HF on first real use)")


class _Stub:
    class _Fn:
        argtypes = None
        restype = None

    def __getattr__(self, name):
        return _Stub._Fn()


def test_speech_weights_come_from_their_own_repo(tmp_path, monkeypatch):
    """One engine runs both models, so only the weights have a channel of their own."""
    import cactus
    from cactus.agent import whistle
    from cactus.agent import fetch

    assert fetch.MODEL_REPOS[fetch.WHISTLE] == "Cactus-Compute/whistle"
    assert fetch.engine_repo(fetch.WHISTLE) == fetch.engine_repo(3)
    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    monkeypatch.setattr(cactus, "__file__", str(tmp_path / "package" / "__init__.py"))
    monkeypatch.setattr(fetch, "_register_download", lambda generation: None)
    monkeypatch.delenv("CACTUS_WHISTLE_WEIGHTS", raising=False)
    import hashlib
    from pathlib import Path
    monkeypatch.setitem(fetch.MODEL_SHA256, fetch.WHISTLE, hashlib.sha256(b"weights").hexdigest())

    weights, fetched = tmp_path / "published.cact", []
    weights.write_bytes(b"weights")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kwargs: fetched.append((kwargs["repo_id"], kwargs["filename"])) or str(weights))
    cache = Path(fetch.cache_dir(fetch.WHISTLE))
    assert whistle._weights_path() == str(cache / "whistle.cact")
    assert (cache / "whistle.cact").read_bytes() == b"weights"
    assert fetched == [("Cactus-Compute/whistle", "whistle.cact")]
    monkeypatch.setenv("CACTUS_WHISTLE_WEIGHTS", "/opt/whistle.cact")
    assert whistle._weights_path() == "/opt/whistle.cact"


def test_a_cached_model_still_counts_as_a_download(tmp_path, monkeypatch):
    """Hugging Face counts a model by requests for its config, so a warm cache must
    still ask for it: otherwise only a visitor's first ever load is ever counted."""
    import cactus
    from cactus.agent import whistle
    from cactus.agent import fetch

    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    monkeypatch.setattr(cactus, "__file__", str(tmp_path / "package" / "__init__.py"))
    monkeypatch.delenv("CACTUS_WHISTLE_WEIGHTS", raising=False)
    import hashlib
    from pathlib import Path
    monkeypatch.setitem(fetch.MODEL_SHA256, fetch.WHISTLE, hashlib.sha256(b"weights").hexdigest())

    cache = Path(fetch.cache_dir(fetch.WHISTLE))
    cache.mkdir(parents=True)
    (cache / "whistle.cact").write_bytes(b"weights")

    asked = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kwargs: asked.append((kwargs["repo_id"], kwargs["filename"],
                                                       kwargs.get("force_download"))) or str(cache / "whistle.cact"))
    assert whistle._weights_path() == str(cache / "whistle.cact")
    assert asked == [("Cactus-Compute/whistle", "config.json", True)]

    whistle._weights_path()
    assert len(asked) == 2, "every load counts, not just the first"


def test_counting_a_download_never_breaks_an_offline_load(tmp_path, monkeypatch):
    import cactus
    from cactus.agent import whistle
    from cactus.agent import fetch

    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    monkeypatch.setattr(cactus, "__file__", str(tmp_path / "package" / "__init__.py"))
    monkeypatch.delenv("CACTUS_WHISTLE_WEIGHTS", raising=False)
    import hashlib
    from pathlib import Path
    monkeypatch.setitem(fetch.MODEL_SHA256, fetch.WHISTLE, hashlib.sha256(b"weights").hexdigest())
    cache = Path(fetch.cache_dir(fetch.WHISTLE))
    cache.mkdir(parents=True)
    (cache / "whistle.cact").write_bytes(b"weights")

    def offline(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", offline)
    assert whistle._weights_path() == str(cache / "whistle.cact")


def test_module_level_transcribe_reuses_one_model(monkeypatch):
    import cactus
    from cactus.agent import whistle

    calls = []

    class _Fake:
        def __init__(self, weights=None):
            calls.append(weights)
            self.weights = weights

        def transcribe(self, audio, language=None, keywords=None, word_timestamps=False):
            return {"text": audio, "language": language, "words": word_timestamps}

    monkeypatch.setattr(whistle, "Whistle", _Fake)
    monkeypatch.setattr(whistle, "_shared", {})
    monkeypatch.setattr("cactus._telemetry.track", lambda *a, **k: None)
    assert cactus.transcribe("a.wav")["text"] == "a.wav"
    assert cactus.transcribe("b.wav", language="de")["language"] == "de"
    assert calls == [None]
    cactus.transcribe("c.wav", weights="tuned.cact")
    assert calls == [None, "tuned.cact"]


def test_the_speech_model_loads_into_needle_s_engine(monkeypatch):
    from cactus.agent import whistle

    loaded = []
    monkeypatch.setattr(runtime, "_lib_handles", {})
    monkeypatch.setattr("cactus.agent.runtime._load_cdll", lambda generation: loaded.append(generation) or _Stub())
    whistle._lib()
    monkeypatch.setattr(runtime, "_lib_handles", {})
    assert loaded == [3]


def _tone(seconds, rate=16000):
    return [0.3 * math.sin(2 * math.pi * 220 * i / rate) for i in range(int(seconds * rate))]


def _write_wav(path, samples, rate, channels=1, width=2):
    peak = (1 << (8 * width - 1)) - 1
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        for value in samples:
            level = int(value * peak)
            frame = bytes([level + 128]) if width == 1 else level.to_bytes(width, "little", signed=True)
            out.writeframes(frame * channels)


@pytest.mark.parametrize("rate,channels,width", [(16000, 1, 2), (8000, 1, 2), (44100, 2, 2), (16000, 1, 1), (22050, 2, 3), (48000, 1, 4)])
def test_wav_files_become_16k_mono_floats(tmp_path, rate, channels, width):
    from cactus.agent.whistle import _read_wav

    if rate != 16000:
        pytest.importorskip("soxr")
    path = tmp_path / "tone.wav"
    _write_wav(path, _tone(0.5, rate), rate, channels, width)
    samples = _read_wav(str(path))
    assert abs(len(samples) - 8000) <= 1
    assert 0.28 < max(samples) < 0.31 and -0.31 < min(samples) < -0.28
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if a < 0 <= b)
    assert 108 <= crossings <= 111


def _riff(tag, channels, rate, width, payload, extensible=False, before=b"", after=b""):
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, channels, rate, rate * channels * width, channels * width, 8 * width)
    if extensible:
        fmt += struct.pack("<HHIH", 22, 8 * width, 0, tag) + bytes(14)

    def chunk(name, body):
        return name + struct.pack("<I", len(body)) + body + b"\0" * (len(body) & 1)

    body = b"WAVE" + before + chunk(b"fmt ", fmt) + after + chunk(b"data", payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize("numpy_available", [True, False])
def test_float_and_extensible_wav_files_read_like_pcm(tmp_path, monkeypatch, numpy_available):
    from cactus.agent.whistle import _read_wav

    if not numpy_available:
        monkeypatch.setitem(sys.modules, "numpy", None)
    values = [0.0, 0.5, -0.25, 0.75]
    cases = {
        "float32": _riff(3, 1, 16000, 4, struct.pack("<4f", *values)),
        "float64 stereo": _riff(3, 2, 16000, 8, struct.pack("<8d", *[v for v in values for _ in range(2)])),
        "extensible pcm16": _riff(1, 1, 16000, 2, struct.pack("<4h", *[int(v * 32768) for v in values]), extensible=True),
        "extensible float32": _riff(3, 1, 16000, 4, struct.pack("<4f", *values), extensible=True),
        "odd chunk first": _riff(3, 1, 16000, 4, struct.pack("<4f", *values), before=b"LIST\x05\0\0\0abcde\0"),
    }
    for name, data in cases.items():
        path = tmp_path / f"{name}.wav"
        path.write_bytes(data)
        assert [round(float(v), 4) for v in _read_wav(str(path))] == values, name


def test_files_that_are_not_pcm_or_float_wav_fail(tmp_path):
    from cactus.agent.whistle import _read_wav

    path = tmp_path / "clip.wav"
    for data in (b"", b"fLaC" + bytes(40), b"RIFF\x04\0\0\0WAVE", _riff(7, 1, 8000, 1, bytes(8))):
        path.write_bytes(data)
        with pytest.raises(wave.Error):
            _read_wav(str(path))
    path.write_bytes(_riff(1, 1, 16000, 2, b""))
    assert len(_read_wav(str(path))) == 0


def test_samples_accept_lists_arrays_and_float32_bytes():
    from cactus.agent.whistle import _samples

    values = [0.0, 0.5, -0.25]
    inputs = [values, array.array("f", values), struct.pack("<3f", *values), (v for v in values)]
    numpy = pytest.importorskip("numpy")
    inputs.append(numpy.array(values, numpy.float64))
    for audio in inputs:
        buffer, count = _samples(audio)
        assert count == 3 and list(buffer) == values
    assert _samples([])[1] == 0


@requires_whistle
def test_silence_is_an_empty_transcript():
    from cactus import Whistle

    result = Whistle().transcribe([0.0] * 16000, word_timestamps=True)
    assert result == {"text": "", "language": "", "words": [], "ttft_ms": 0.0, "decode_tps": 0.0}


@requires_whistle
def test_transcribe_returns_text_language_and_timed_words(tmp_path):
    from cactus import Whistle
    from cactus.agent.whistle import LANGUAGES

    whistle = Whistle()
    sound = [v * (0.2 + 0.8 * abs(math.sin(math.pi * 3 * i / 16000))) for i, v in enumerate(_tone(3))]
    plain = whistle.transcribe(sound)
    assert set(plain) == {"text", "language", "ttft_ms", "decode_tps"} and plain["language"] in LANGUAGES + ("",)
    assert (plain["ttft_ms"] > 0) == bool(plain["language"]) and plain["decode_tps"] >= 0
    path = tmp_path / "sound.wav"
    _write_wav(path, sound, 16000)
    timed = whistle.transcribe(path, language="en", keywords=["Siobhan", "Krzysztof"], word_timestamps=True)
    assert timed["language"] == "en" and isinstance(timed["text"], str)
    starts = [word["start"] for word in timed["words"]]
    assert starts == sorted(starts)
    assert all(0 <= word["start"] <= word["end"] <= 3.01 and 0 <= word["probability"] <= 1 for word in timed["words"])


@requires_whistle
def test_embed_keeps_its_window_and_transcribe_takes_long_audio():
    from cactus import Whistle

    whistle = Whistle()
    embedding = whistle.embed(_tone(1))
    assert len(embedding) > 0 and all(isinstance(v, float) for v in embedding[:8])
    with pytest.raises(RuntimeError, match="30 s"):
        whistle.embed([0.0] * (30 * 16000 + 1))
    long = whistle.transcribe([0.0] * (31 * 16000), word_timestamps=True)
    assert long["text"] == "" and long["language"] == "" and long["words"] == []


@requires_whistle
def test_missing_weights_fail_clearly(tmp_path):
    from cactus import Whistle

    broken = tmp_path / "broken.cact"
    broken.write_bytes(b"not a model at all")
    with pytest.raises(RuntimeError):
        Whistle(weights=broken)
    whistle = Whistle()
    assert os.path.getsize(whistle.weights) > 0 and whistle.transcribe([0.0] * 16000)["text"] == ""


class _Whistle:
    weights = __file__

    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        return self.result


class _Microphone:
    PortAudioError = OSError

    def __init__(self, rate, seconds):
        self.rate, self.seconds, self.opened = rate, seconds, None

    def query_devices(self, kind):
        return {"default_samplerate": float(self.rate)}

    def InputStream(self, **options):
        import numpy

        microphone = self

        class Stream:
            def __enter__(self):
                microphone.opened = options
                options["callback"](numpy.full((microphone.rate * microphone.seconds, 1), 0.25, numpy.float32), None, None, None)

            def __exit__(self, *_):
                return False

        return Stream()


def test_record_resamples_the_microphone_to_16_khz_and_keeps_everything(monkeypatch, capsys):
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    from cactus.cli.speech import record

    microphone = _Microphone(48000, 1)
    monkeypatch.setitem(sys.modules, "sounddevice", microphone)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    audio = record()
    assert microphone.opened["samplerate"] == 48000 and microphone.opened["channels"] == 1
    assert audio.dtype == numpy.float32 and len(audio) == 16000 and abs(float(audio[8000]) - 0.25) < 0.01
    assert "recording, Enter to stop" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "sounddevice", _Microphone(16000, 31))
    assert len(record()) == 31 * 16000


def test_record_says_what_is_missing(monkeypatch):
    from cactus.cli.speech import record

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match=r"cactus-compute\[mic\]"):
        record()


class _StreamEngine:
    def __init__(self):
        self.calls = []

    def cactus_load(self, data, size):
        return 0

    def cactus_last_error(self):
        return b"stream broke"

    def _write(self, out, text, received):
        payload = json.dumps({"text": text, "words": [{"word": text, "start": 0.0, "end": 0.5, "probability": 1.0}] if text else [],
                              "pending": "tail", "language": "en", "received": received, "pass_ms": 7.0}).encode()
        out.value = payload
        return len(text.split())

    def cactus_stream_transcribe_process(self, samples, count, language, keywords, out, capacity):
        self.calls.append(("feed", count, language, keywords))
        if count < 0:
            return -1
        return self._write(out, str(count), len(self.calls))

    def cactus_stream_transcribe_stop(self, out, capacity):
        self.calls.append(("finish",))
        return self._write(out, "tail", len(self.calls))


class _TranscribeEngine:
    def __init__(self):
        self.capacities = []

    def cactus_load(self, data, size):
        return 0

    def cactus_transcribe(self, samples, count, language, keywords, word_timestamps, out, capacity):
        self.capacities.append(capacity)
        out.value = json.dumps({"text": "", "language": "", "ttft_ms": 0.0, "decode_tps": 0.0}).encode()
        return 0


def test_output_buffers_grow_with_the_audio(monkeypatch):
    from cactus.agent import whistle

    monkeypatch.setattr(whistle, "_loaded", None)
    engine = _TranscribeEngine()
    monkeypatch.setattr(whistle, "_lib", lambda: engine)
    model = whistle.Whistle(weights=__file__)
    model.transcribe([0.0] * 16000)
    model.transcribe([0.0] * (3600 * 16000))
    assert engine.capacities == [1 << 18, 3600 * whistle.OUTPUT_BYTES_PER_SECOND]
    monkeypatch.setattr(whistle, "_lib", lambda: _StreamEngine())
    streamed = whistle.Whistle(weights=__file__)
    list(streamed.stream([[0.0] * (120 * 16000)]))
    assert len(streamed._buffer) == 120 * whistle.OUTPUT_BYTES_PER_SECOND


def test_stream_feeds_each_chunk_and_flushes_the_tail(monkeypatch):
    from cactus.agent import whistle

    engine = _StreamEngine()
    monkeypatch.setattr(whistle, "_lib", lambda: engine)
    monkeypatch.setattr(whistle, "_loaded", None)
    model = whistle.Whistle(weights=__file__)
    steps = list(model.stream([[0.0] * 3, [0.0] * 2], language="de", keywords=["Siobhan", "Krzysztof"]))
    assert [step["text"] for step in steps] == ["3", "2", "tail"] and steps[-1]["received"] == 3
    assert engine.calls == [("feed", 3, b"de", b"Siobhan\nKrzysztof"), ("feed", 2, b"de", b"Siobhan\nKrzysztof"), ("finish",)]
    engine.calls.clear()
    live = model.stream([[0.0]])
    assert next(live)["text"] == "1"
    live.close()
    assert engine.calls == [("feed", 1, None, None), ("finish",)]


def test_module_level_stream_reuses_one_model(monkeypatch):
    import cactus
    from cactus.agent import whistle

    class _Fake:
        def __init__(self, weights=None):
            self.weights = weights

        def stream(self, chunks, language=None, keywords=None):
            for chunk in chunks:
                yield {"text": f"{len(chunk)} {language} {keywords} {self.weights}"}

    monkeypatch.setattr(whistle, "Whistle", _Fake)
    monkeypatch.setattr(whistle, "_shared", {})
    monkeypatch.setattr("cactus._telemetry.track", lambda *a, **k: None)
    assert [s["text"] for s in cactus.stream([[0.0] * 2], language="de", keywords=["Siobhan"])] == ["2 de ['Siobhan'] None"]
    assert next(cactus.stream([[0.0]], weights="tuned.cact"))["text"] == "1 None None tuned.cact"
    assert set(whistle._shared) == {None, "tuned.cact"}


def test_stream_raises_the_engine_error(monkeypatch):
    from cactus.agent import whistle

    class Broken(_StreamEngine):
        def cactus_stream_transcribe_process(self, *args):
            return -1

    monkeypatch.setattr(whistle, "_lib", lambda: Broken())
    monkeypatch.setattr(whistle, "_loaded", None)
    with pytest.raises(RuntimeError, match="stream broke"):
        list(whistle.Whistle(weights=__file__).stream([[0.0]]))


class _LiveWhistle:
    weights = __file__

    def __init__(self):
        self.chunks, self.options = [], None

    def stream(self, chunks, **options):
        self.options = options
        for chunk in chunks:
            self.chunks.append(len(chunk))
            yield {"text": f"chunk{len(self.chunks)}", "words": [{"word": f"chunk{len(self.chunks)}", "start": 0.1, "end": 0.4, "probability": 0.9}],
                   "pending": "tail", "language": "en", "received": float(len(self.chunks)), "pass_ms": 20.0}
        yield {"text": "tail", "words": [{"word": "tail", "start": 0.5, "end": 0.9, "probability": 0.8}], "pending": "", "language": "en",
               "received": float(len(self.chunks)), "pass_ms": 0.0}


def test_listen_feeds_the_microphone_in_seconds_and_prints_words_live(monkeypatch, capsys):
    pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    from cactus.cli.speech import listen

    monkeypatch.setitem(sys.modules, "sounddevice", _Microphone(48000, 3))
    monkeypatch.setattr("builtins.input", lambda *_: "")
    whistle = _LiveWhistle()
    listen(whistle, {"language": "de", "keywords": ["Siobhan"], "timestamps": True})
    assert whistle.options == {"language": "de", "keywords": ["Siobhan"]}
    assert abs(sum(whistle.chunks) - 3 * 16000) <= 2 and all(n >= 16000 for n in whistle.chunks[:-1])
    out = capsys.readouterr().out
    assert "recording, Enter to stop" in out and "chunk1 " in out and " tail" in out
    assert "\r\x1b[J" + "chunk1\n\x1b[90mtail\x1b[0m" in out and "\r\x1b[1A\x1b[J" + "chunk1 chunk2\n" in out
    assert " ".join(f"chunk{n}" for n in range(1, len(whistle.chunks) + 1)) + " tail\n\x1b[90m\x1b[0m" in out
    assert "0.10 -  0.40  chunk1" in out and "0.50 -  0.90  tail" in out
    assert f"audio   {len(whistle.chunks):.1f} s" in out and "pass   20 ms   en" in out


def test_listen_reports_a_broken_stream(monkeypatch, capsys):
    pytest.importorskip("numpy")
    pytest.importorskip("soxr")
    from cactus.cli.speech import listen

    class Broken:
        weights = __file__

        def stream(self, chunks, **options):
            raise RuntimeError("no speech model loaded")
            yield

    monkeypatch.setitem(sys.modules, "sounddevice", _Microphone(16000, 1))
    monkeypatch.setattr("builtins.input", lambda *_: "")
    with pytest.raises(RuntimeError, match="no speech model loaded"):
        listen(Broken(), {"language": None, "keywords": [], "timestamps": False})


def test_playground_listens_on_enter_once_streaming_is_on(monkeypatch, capsys):
    from cactus.agent import whistle

    lines = iter(["", "/stream", "", "/stream", "", "/quit"])
    heard = []
    monkeypatch.setattr(speech, "prompt", lambda: next(lines))
    monkeypatch.setattr(speech, "record", lambda: heard.append("recorded") or [0.0])
    monkeypatch.setattr(speech, "show_transcript", lambda model, audio, state: heard.append(audio))
    monkeypatch.setattr(speech, "listen", lambda model, state: heard.append(state["language"]))
    monkeypatch.setattr("cactus.agent.whistle.Whistle", lambda weights=None: _Whistle({}))
    speech.playground(type("Args", (), {"audio": None, "language": "en", "keywords": "", "word_timestamps": False, "weights": None})())
    assert heard == ["recorded", [0.0], "en", "recorded", [0.0]]
    assert "live transcription on" in capsys.readouterr().out


def test_rate_counts_steps_after_the_first_mark():
    from cactus.cli.speech import rate

    assert rate([1.0, 1.5, 2.0]) == 2.0
    assert rate([1.0]) == 0.0 and rate([]) == 0.0 and rate([2.0, 2.0]) == 0.0


def test_compare_prints_one_line_per_model_with_dashes_for_missing_timing(capsys):
    from cactus.cli.speech import run_models

    models = [("whistle", 17e6, lambda audio: ("hello there", 0.013, 1264.0)),
              ("moonshine tiny v2", 45e6, lambda audio: ("", None, None))]
    run_models(models, [0.0] * 16000)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("  whistle             17 MB  ttft  13 ms  decode 1264 tok/s  total")
    assert lines[0].endswith("ms   hello there")
    assert lines[1].startswith("  moonshine tiny v2   45 MB  ttft      -  decode          -  total")
    assert lines[1].endswith("ms   (no speech)")


def test_playground_and_compare_share_one_status_line():
    from cactus.cli.speech import status

    assert status("whistle", 17e6, 0.0137, 1264.4, 0.048) == "  whistle             17 MB  ttft  14 ms  decode 1264 tok/s  total   48 ms"
    assert status("moonshine tiny v2", 145e6, None, None, 0.21) == "  moonshine tiny v2  145 MB  ttft      -  decode          -  total  210 ms"


def test_status_reads_a_slow_run_in_seconds():
    from cactus.cli.speech import status

    assert status("whistle", 17e6, 0.0137, 1264.4, 12.5).endswith("total  12.5 s")
    assert status("whistle", 17e6, 0.0137, 1264.4, 9.9).endswith("total 9900 ms")


def test_audio_path_takes_a_path_as_a_terminal_hands_it_over():
    from cactus.cli.speech import audio_path

    assert audio_path(" clip.wav ") == "clip.wav"
    assert audio_path('"my clips/a b.wav"') == "my clips/a b.wav"
    assert audio_path("'my clips/a b.wav'") == "my clips/a b.wav"
    assert audio_path("my\\ clips/a\\ b.wav") == "my clips/a b.wav"


def test_playground_prints_text_words_and_timing(capsys):
    from cactus.cli.speech import show_transcript

    whistle = _Whistle({"text": "hello there", "language": "en", "ttft_ms": 12.6, "decode_tps": 900.4,
                        "words": [{"word": "hello", "start": 0.1, "end": 0.4, "probability": 0.98}]})
    show_transcript(whistle, "clip.wav", {"language": "en", "keywords": ["Siobhan"], "timestamps": True})
    out = capsys.readouterr().out.splitlines()
    assert whistle.calls == [("clip.wav", {"language": "en", "keywords": ["Siobhan"], "word_timestamps": True})]
    assert out[0] == "hello there"
    assert out[1].split() == ["0.10", "-", "0.40", "hello", "0.98"]
    assert out[2].startswith("  whistle           ") and "ttft  13 ms  decode  900 tok/s  total" in out[2] and out[2].endswith("ms   en")


def test_playground_reports_engine_errors_instead_of_raising(capsys):
    from cactus.cli.speech import show_transcript

    class Broken:
        def transcribe(self, audio, **options):
            raise RuntimeError("audio limit is 30 s")

    show_transcript(Broken(), "long.wav", {"language": None, "keywords": [], "timestamps": False})
    assert capsys.readouterr().out == "  audio limit is 30 s\n"


def test_playground_keeps_the_language_when_the_code_is_not_one_of_ours(monkeypatch, capsys):
    from cactus.agent import whistle

    lines = iter(["/language de", "/language EN", "/language", "/quit"])
    monkeypatch.setattr(speech, "prompt", lambda: next(lines))
    monkeypatch.setattr("cactus.agent.whistle.Whistle", lambda weights=None: _Whistle({}))
    speech.playground(type("Args", (), {"audio": None, "language": None, "keywords": "",
                                         "word_timestamps": False, "weights": None})())
    out = capsys.readouterr().out
    assert "language de" in out and "language is one of en de fr es it nl pl" in out and "language detected" in out


def test_compare_runs_a_file_the_same_way_the_playground_does(monkeypatch, capsys):
    pytest.importorskip("numpy")
    import types
    from cactus.agent import whistle

    for name in ("whisper", "moonshine_voice"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    seen, read = [], []
    monkeypatch.setattr(speech, "run_models", lambda models, audio: seen.append(len(audio)))
    monkeypatch.setattr("cactus.cli.speech._read_wav", lambda path: read.append(path) or [0.0] * 16000)
    lines = iter(['/file "my clips/a b.wav"', "/quit"])
    monkeypatch.setattr(speech, "prompt", lambda: next(lines))
    monkeypatch.setattr(speech, "load_whistle", lambda weights: (17e6, lambda audio: ("", None, None)))
    monkeypatch.setattr(speech, "load_whisper", lambda size: (76e6, lambda audio: ("", None, None)))
    monkeypatch.setattr(speech, "load_moonshine", lambda: (45e6, lambda audio: ("", None, None)))
    speech.compare(type("Args", (), {"audio": None, "weights": None})())
    assert read == ["my clips/a b.wav"] and seen == [16000]


def test_compare_needs_its_extra(monkeypatch):
    from cactus.agent import whistle

    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(SystemExit, match=r"cactus-compute\[mic,compare\]"):
        speech.compare(type("Args", (), {"audio": None, "weights": None})())


def test_cli_routes_the_whistle_commands(monkeypatch):
    import cactus._telemetry
    import cactus.cli
    import cactus.agent.whistle

    seen = []
    monkeypatch.setattr(cactus._telemetry, "track", lambda *a, **k: None)
    monkeypatch.setattr(speech, "playground", lambda args: seen.append(("playground", args)))
    monkeypatch.setattr(speech, "compare", lambda args: seen.append(("compare", args)))
    monkeypatch.setattr(sys, "argv", ["cactus", "whistle", "playground", "clip.wav", "--language", "de", "--keywords", "Siobhan, Krzysztof", "--word-timestamps"])
    cactus.cli.main()
    monkeypatch.setattr(sys, "argv", ["cactus", "whistle", "compare", "--weights", "w.cact"])
    cactus.cli.main()
    assert [s[0] for s in seen] == ["playground", "compare"]
    assert (seen[0][1].audio, seen[0][1].language, seen[0][1].keywords, seen[0][1].word_timestamps) == ("clip.wav", "de", "Siobhan, Krzysztof", True)
    assert (seen[1][1].audio, seen[1][1].weights) == (None, "w.cact")
    monkeypatch.setattr(sys, "argv", ["cactus", "whistle"])
    with pytest.raises(SystemExit, match="cactus whistle playground \\| compare"):
        cactus.cli.main()
