"""Shared I/O: audio decode/resample, noise mixing, frame extraction, PCM packing."""
from __future__ import annotations

import math
import random
import struct
import subprocess
import wave
from pathlib import Path


TARGET_SR = 16000


def decode_to_mono16k_wav(src: Path, dst: Path) -> None:
    """Decode any audio (m4a/mp3/wav) to 16kHz mono 16-bit PCM WAV via ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16", str(dst)],
        check=True,
    )


def read_wav_int16(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == TARGET_SR, \
            f"expected 16k mono s16, got {w.getframerate()}Hz {w.getnchannels()}ch {w.getsampwidth()*8}bit for {path}"
        raw = w.readframes(w.getnframes())
    return list(struct.unpack(f"<{len(raw)//2}h", raw))


def write_wav_int16(path: Path, samples: list[int], sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _rms(xs: list[int]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def mix_at_snr(clean: list[int], noise: list[int], snr_db: float) -> list[int]:
    """Mix noise into clean at target SNR (dB). Loops noise to match length."""
    if len(noise) < len(clean):
        reps = len(clean) // len(noise) + 1
        noise = (noise * reps)[: len(clean)]
    else:
        # random offset so we don't always start at 0
        start = random.randint(0, len(noise) - len(clean))
        noise = noise[start : start + len(clean)]

    rms_s = _rms(clean)
    rms_n = _rms(noise)
    if rms_s == 0 or rms_n == 0:
        return list(clean)

    target_noise_rms = rms_s / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / rms_n

    out = []
    for s, n in zip(clean, noise):
        v = int(s + n * scale)
        if v > 32767: v = 32767
        elif v < -32768: v = -32768
        out.append(v)
    return out


def extract_random_frames(video: Path, count: int, out_dir: Path, seed: int = 0) -> list[Path]:
    """Extract `count` random frames as jpg. Returns written paths.

    If every expected frame file already exists, skips re-extraction — so the
    harness works even when the source .mov files aren't checked out (the
    frames themselves are committed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    expected = [out_dir / f"{stem}_f{i}.jpg" for i in range(count)]
    if all(p.exists() for p in expected):
        return expected
    if not video.exists():
        raise FileNotFoundError(
            f"Source video {video} is missing and frames are not all present "
            f"in {out_dir}. Re-download the video or commit the expected "
            f"frames: {[p.name for p in expected]}"
        )
    dur = _probe_duration(video)
    rng = random.Random(f"{video.name}:{seed}")
    for i, out in enumerate(expected):
        if out.exists():
            continue
        t = rng.uniform(0.2 * dur, 0.9 * dur)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
             "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)],
            check=True,
        )
    return expected


def _probe_duration(video: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(video)]
    )
    return float(out.strip())


def pcm_bytes_from_wav(path: Path) -> bytes:
    """Load 16k mono s16 WAV and return raw interleaved PCM bytes (ready for FFI)."""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == TARGET_SR
        return w.readframes(w.getnframes())
