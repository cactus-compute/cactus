import array
import ctypes
import os
import struct
import sys
import wave

SAMPLE_RATE = 16000

_WAV_TYPES = {(1, 1): "B", (1, 2): "h", (1, 3): None, (1, 4): "i", (3, 4): "f", (3, 8): "d"}


def _wav_chunks(path):
    with open(path, "rb") as source:
        data = source.read()
    chunks, at = {}, 12
    while at + 8 <= len(data):
        size = struct.unpack_from("<I", data, at + 4)[0]
        chunks.setdefault(data[at:at + 4], data[at + 8:at + 8 + size])
        at += 8 + size + (size & 1)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE" or b"fmt " not in chunks or b"data" not in chunks:
        raise wave.Error(f"{path} is not a WAV file")
    tag, channels, rate, _, align, _ = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if tag == 0xFFFE:
        tag = struct.unpack_from("<H", chunks[b"fmt "], 24)[0]
    width = align // channels
    if (tag, width) not in _WAV_TYPES:
        raise wave.Error(f"{path} is not integer PCM or float WAV")
    return tag, channels, width, rate, chunks[b"data"]


def _read_wav(path):
    tag, channels, width, rate, raw = _wav_chunks(path)
    offset, scale = (0, 1.0) if tag == 3 else (128, 128.0) if width == 1 else (0, float(1 << (8 * width - 1)))
    try:
        import numpy
    except ImportError:
        numpy = None
    if numpy is not None:
        if width == 3:
            bytes3 = numpy.frombuffer(raw, numpy.uint8).reshape(-1, 3).astype(numpy.int32)
            values = (bytes3[:, 0] | bytes3[:, 1] << 8 | bytes3[:, 2] << 16) << 8 >> 8
        else:
            values = numpy.frombuffer(raw, "<" + _WAV_TYPES[tag, width])
        mono = (values.reshape(-1, channels).mean(axis=1, dtype=numpy.float32) - offset) / numpy.float32(scale)
    else:
        if width == 3:
            values = [int.from_bytes(raw[i:i + 3], "little", signed=True) for i in range(0, len(raw), 3)]
        else:
            values = array.array(_WAV_TYPES[tag, width])
            values.frombytes(raw)
            if sys.byteorder == "big":
                values.byteswap()
        mono = [(sum(frame) / channels - offset) / scale for frame in zip(*(values[c::channels] for c in range(channels)))]
    if rate == SAMPLE_RATE or not len(mono):
        return mono
    try:
        import numpy
        import soxr
    except ImportError:
        raise RuntimeError(f'resampling {rate} Hz audio to {SAMPLE_RATE} Hz needs soxr: pip install "cactus-compute[mic]"') from None
    return soxr.resample(numpy.asarray(mono, numpy.float32), rate, SAMPLE_RATE, quality="HQ")


def _options(language, keywords):
    if keywords is not None and not isinstance(keywords, str):
        keywords = "\n".join(keywords)
    return language.encode("utf-8") if language else None, keywords.encode("utf-8") if keywords else None


def _samples(audio):
    if isinstance(audio, (str, os.PathLike)):
        audio = _read_wav(os.fspath(audio))
    data = array.array("f")
    if isinstance(audio, (bytes, bytearray)):
        data.frombytes(audio)
    elif isinstance(audio, array.array) and audio.typecode == "f":
        data = audio
    elif hasattr(audio, "tobytes") and getattr(audio, "dtype", None) is not None:
        data.frombytes(audio.astype("float32", copy=False).tobytes())
    else:
        data.extend(audio)
    return (ctypes.c_float * len(data)).from_buffer(data) if len(data) else (ctypes.c_float * 1)(), len(data)

