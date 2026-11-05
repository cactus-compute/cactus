from __future__ import annotations

import ctypes
import json
import os
import struct

from .runtime import synchronized, lock

from .audio import SAMPLE_RATE, _options, _samples
from .audio import _read_wav as _read_wav
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
OUTPUT_BYTES_PER_SECOND = 4096

_loaded = None


def _weights_path():
    from .runtime import _base_weights_path
    from . import fetch

    return os.environ.get("CACTUS_WHISTLE_WEIGHTS") or _base_weights_path(fetch.WHISTLE)


def _lib():
    from .runtime import _lib as runtime
    return runtime()


def languages(weights):
    try:
        with open(weights, "rb") as handle:
            raw = handle.read()
        tag, count, codebook = struct.unpack_from("<III", raw)
    except (OSError, struct.error):
        return LANGUAGES
    if tag != 0x05E12A84:
        return LANGUAGES
    record = 196 + codebook * 4
    for _ in range(count):
        if raw[record] == 4:
            offset, size = struct.unpack_from("<QQ", raw, record + 20)
            blob, found = raw[offset:offset + size], []
            at = 24
            for _ in range(struct.unpack_from("<I", blob)[0]):
                length = struct.unpack_from("<H", blob, at + 5)[0]
                piece = blob[at + 7:at + 7 + length].decode("utf-8", "replace")
                at += 7 + length
                found = found + [piece[2:-2]] if piece.startswith("<|") and piece.endswith("|>") else []
            return tuple(found) or LANGUAGES
        record += 44
    return LANGUAGES


class Whistle:
    """Speech-to-text in English, German, French, Spanish, Italian, Dutch and Polish.

    Audio is a WAV file path or 16 kHz mono float samples in [-1, 1], of any length.
    Calls share the Cactus runtime and are serialized across threads.
    """

    def __init__(self, weights=None, buffer_size=1 << 18):
        self.weights = os.fspath(weights) if weights is not None else _weights_path()
        self._buffer = ctypes.create_string_buffer(buffer_size)
        self._bind()

    @synchronized
    def _bind(self):
        global _loaded
        lib = _lib()
        if _loaded == self.weights:
            return lib
        with open(self.weights, "rb") as handle:
            data = handle.read()
        if lib.cactus_load(data, len(data)) < 0:
            _loaded = None
            raise RuntimeError(lib.cactus_last_error().decode("utf-8", "replace"))
        _loaded = self.weights
        return lib

    def _result(self, code):
        if code < 0:
            raise RuntimeError(_lib().cactus_last_error().decode("utf-8", "replace"))
        return json.loads(self._buffer.value.decode("utf-8", "replace"))

    def _reserve(self, size):
        if size > len(self._buffer):
            self._buffer = ctypes.create_string_buffer(size)

    @synchronized
    def transcribe(self, audio, language=None, keywords=None, word_timestamps=False) -> dict:
        """Returns {"text", "language", "ttft_ms", "decode_tps"}, plus "words" with start, end and probability when word_timestamps is set.

        language is one of LANGUAGES, or None to detect it. keywords are words or phrases for keyword biasing.
        """
        lib = self._bind()
        samples, count = _samples(audio)
        language, keywords = _options(language, keywords)
        self._reserve(count // SAMPLE_RATE * OUTPUT_BYTES_PER_SECOND)
        return self._result(lib.cactus_transcribe(samples, count, language, keywords, int(bool(word_timestamps)),
                                                  self._buffer, len(self._buffer)))

    def stream(self, chunks, language=None, keywords=None):
        """Live transcription: one dict per chunk, then one for the tail when the chunks end.

        chunks are 16 kHz mono float sample buffers, about a second each, with no limit on the total. Each dict has the
        "text" and "words" committed by that chunk, with times from the start of the stream, the unconfirmed "pending"
        tail, the "language", the seconds "received" and the "pass_ms" it took. Join the texts with a space.
        """
        with lock:
            lib = self._bind()
            language, keywords = _options(language, keywords)
            stopped = False
            try:
                for chunk in chunks:
                    samples, count = _samples(chunk)
                    self._reserve(count // SAMPLE_RATE * OUTPUT_BYTES_PER_SECOND)
                    yield self._result(lib.cactus_stream_transcribe_process(samples, count, language, keywords, self._buffer,
                                                                            len(self._buffer)))
                result = self._result(lib.cactus_stream_transcribe_stop(self._buffer, len(self._buffer)))
                stopped = True
                yield result
            finally:
                if not stopped:
                    lib.cactus_stream_transcribe_stop(self._buffer, len(self._buffer))

    @synchronized
    def embed(self, audio) -> list[float]:
        """The encoder output, one row of floats per 80 ms frame, flattened."""
        lib = self._bind()
        samples, count = _samples(audio)
        size = lib.cactus_embed(None, samples, count, None, 0)
        if size < 0:
            raise RuntimeError(lib.cactus_last_error().decode("utf-8", "replace"))
        output = (ctypes.c_float * size)()
        if lib.cactus_embed(None, samples, count, output, size) != size:
            raise RuntimeError(lib.cactus_last_error().decode("utf-8", "replace"))
        return list(output)


_shared = {}


def _shared_model(weights):
    key = os.fspath(weights) if weights is not None else None
    if key not in _shared:
        _shared[key] = Whistle(weights=weights)
    return _shared[key]


def transcribe(audio, language=None, keywords=None, word_timestamps=False, weights=None) -> dict:
    """Speech to text, on the model this process already has loaded.

    audio is a WAV file path or 16 kHz mono float samples in [-1, 1], of any length.
    Returns {"text", "language", "ttft_ms", "decode_tps"}, plus "words" with start,
    end and probability when word_timestamps is set. language is one of LANGUAGES,
    or None to detect it; keywords are words or phrases for keyword biasing.
    """
    from .._telemetry import track

    track("transcribe", {"tuned": bool(weights), "timestamps": bool(word_timestamps)})
    return _shared_model(weights).transcribe(audio, language=language, keywords=keywords,
                                             word_timestamps=word_timestamps)


def stream(chunks, language=None, keywords=None, weights=None):
    """Live transcription on the model this process already has loaded: Whistle.stream with no limit on length.

    chunks are 16 kHz mono float sample buffers, about a second each. Yields one dict per chunk with the "text"
    and "words" committed by it, the unconfirmed "pending" tail, the "language", the seconds "received" and
    "pass_ms", then one for the tail when the chunks end.
    """
    from .._telemetry import track

    track("stream", {"tuned": bool(weights)})
    return _shared_model(weights).stream(chunks, language=language, keywords=keywords)
