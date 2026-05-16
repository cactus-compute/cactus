from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


_PARAKEET_SAMPLE_RATE = 16000
_PARAKEET_NUM_MELS = 128
_PARAKEET_N_FFT = 512
_PARAKEET_FRAME_LENGTH = 400
_PARAKEET_HOP_LENGTH = 160
_PARAKEET_PREEMPHASIS = 0.97
_PARAKEET_LOG_FLOOR = np.float32(2**-24)
_DEFAULT_MAX_AUDIO_SECONDS = 30.0


def audio_duration_limit_seconds() -> float:
    raw = os.environ.get("CACTUS_TRANSPILER_MAX_AUDIO_SECONDS", str(_DEFAULT_MAX_AUDIO_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AUDIO_SECONDS
    return max(0.0, value)


def normalize_audio_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.uint8:
        return ((samples.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        if scale <= 0:
            scale = 1.0
        return (samples.astype(np.float32) / scale).astype(np.float32)
    return samples.astype(np.float32)


def _limit_waveform_duration(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    max_seconds: float | None,
) -> np.ndarray:
    if max_seconds is None or max_seconds <= 0.0:
        return waveform
    max_samples = int(round(float(sample_rate) * float(max_seconds)))
    if max_samples <= 0 or int(waveform.shape[0]) <= max_samples:
        return waveform
    return np.ascontiguousarray(waveform[:max_samples])


def load_audio_waveform(
    audio_file: str | Path,
    *,
    target_sample_rate: int,
    max_seconds: float | None = None,
) -> np.ndarray:
    if max_seconds is None:
        max_seconds = audio_duration_limit_seconds()
    sample_rate, waveform = wavfile.read(str(audio_file))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = normalize_audio_samples(np.asarray(waveform))
    if int(sample_rate) != int(target_sample_rate):
        gcd = int(np.gcd(int(sample_rate), int(target_sample_rate)))
        waveform = resample_poly(
            waveform,
            int(target_sample_rate) // gcd,
            int(sample_rate) // gcd,
        ).astype(np.float32)
        sample_rate = int(target_sample_rate)
    waveform = _limit_waveform_duration(
        waveform.astype(np.float32),
        sample_rate=int(target_sample_rate),
        max_seconds=max_seconds,
    )
    return waveform.astype(np.float32)


@contextmanager
def limited_audio_file(
    audio_file: str | Path,
    *,
    target_sample_rate: int = _PARAKEET_SAMPLE_RATE,
    max_seconds: float | None = None,
) -> Iterator[Path]:
    if max_seconds is None:
        max_seconds = audio_duration_limit_seconds()
    if max_seconds <= 0.0:
        yield Path(audio_file).expanduser().resolve()
        return

    waveform = load_audio_waveform(
        audio_file,
        target_sample_rate=target_sample_rate,
        max_seconds=max_seconds,
    )
    temp_audio_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = Path(temp_file.name)
        wavfile.write(
            str(temp_audio_path),
            int(target_sample_rate),
            (np.clip(waveform, -1.0, 1.0) * np.float32(32767.0)).astype(np.int16),
        )
        yield temp_audio_path
    finally:
        if temp_audio_path is not None:
            try:
                temp_audio_path.unlink()
            except OSError:
                pass


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filter_bank(num_mels: int, sample_rate: int, n_fft: int) -> np.ndarray:
    try:
        from transformers.audio_utils import mel_filter_bank as transformers_mel_filter_bank  # type: ignore

        return transformers_mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=num_mels,
            min_frequency=0.0,
            max_frequency=sample_rate / 2.0,
            sampling_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).astype(np.float32)
    except Exception:
        pass

    num_freq_bins = n_fft // 2 + 1
    mel_points = np.linspace(
        _hz_to_mel(np.array([0.0], dtype=np.float32))[0],
        _hz_to_mel(np.array([sample_rate / 2.0], dtype=np.float32))[0],
        num_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)

    filters = np.zeros((num_mels, num_freq_bins), dtype=np.float32)
    for index in range(1, num_mels + 1):
        left = max(0, min(num_freq_bins - 1, int(bins[index - 1])))
        center = max(left + 1, min(num_freq_bins - 1, int(bins[index])))
        right = max(center + 1, min(num_freq_bins, int(bins[index + 1])))
        for freq in range(left, center):
            filters[index - 1, freq] = float(freq - left) / float(max(center - left, 1))
        for freq in range(center, right):
            filters[index - 1, freq] = float(right - freq) / float(max(right - center, 1))
    return filters


def generic_log_mel_features(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    num_mels: int,
    n_fft: int,
    hop_length: int,
    frame_length: int,
    preemphasis: float | None = None,
    mel_floor: np.float32 = _PARAKEET_LOG_FLOOR,
    normalize_active_frames_only: bool = True,
) -> tuple[np.ndarray, int]:
    waveform_tensor = torch.from_numpy(waveform).to(torch.float32).unsqueeze(0)
    if preemphasis is not None and abs(preemphasis) > 0.0:
        waveform_tensor = torch.cat(
            [
                waveform_tensor[:, :1],
                waveform_tensor[:, 1:] - float(preemphasis) * waveform_tensor[:, :-1],
            ],
            dim=1,
        )

    window = torch.hann_window(frame_length, periodic=False, dtype=torch.float32)
    stft = torch.stft(
        waveform_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="constant",
    )
    magnitudes = torch.view_as_real(stft)
    magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
    power = magnitudes.pow(2)

    feature_length = max(1, int(waveform.shape[0] // hop_length))
    mel_filters = torch.from_numpy(_mel_filter_bank(num_mels, sample_rate, n_fft)).to(torch.float32)
    mel_spec = power.transpose(1, 2) @ mel_filters
    mel_spec = torch.log(torch.clamp(mel_spec, min=float(mel_floor)))

    if normalize_active_frames_only:
        attention_mask = (
            torch.arange(mel_spec.shape[1], dtype=torch.long, device=mel_spec.device).unsqueeze(0)
            < feature_length
        )
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        masked = mel_spec * mask
        mean = masked.sum(dim=1) / float(feature_length)
        centered = masked - mean.unsqueeze(1)
        denom = float(max(1, feature_length - 1))
        variance = (centered.pow(2) * mask).sum(dim=1) / denom
        mel_spec = (mel_spec - mean.unsqueeze(1)) / torch.sqrt(variance.unsqueeze(1) + np.float32(1e-5))
        mel_spec = mel_spec * mask
    else:
        # Native Parakeet normalizes across the full centered-STFT frame count,
        # then trims to waveform_samples // hop_length. Keep that order for parity.
        frame_count = int(mel_spec.shape[1])
        mean = mel_spec.mean(dim=1)
        centered = mel_spec - mean.unsqueeze(1)
        denom = float(max(1, frame_count - 1))
        variance = centered.pow(2).sum(dim=1) / denom
        mel_spec = centered / torch.sqrt(variance.unsqueeze(1) + np.float32(1e-5))
    return mel_spec[0].cpu().numpy().astype(np.float32), feature_length


def prepare_cactus_audio_features(
    audio_file: str | Path,
    *,
    model_type: str,
    expected_frames: int | None,
    expected_mels: int,
    torch_dtype: torch.dtype,
    layout: str = "frames_mels",
) -> tuple[torch.Tensor, int]:
    from cactus.bindings.cactus import cactus_preprocess_audio_features

    capacity_frames = int(expected_frames) if isinstance(expected_frames, int) and expected_frames > 0 else 0
    if capacity_frames <= 0:
        lowered_model_type = model_type.lower()
        if "whisper" in lowered_model_type:
            capacity_frames = 3000
        else:
            sample_rate, waveform = wavfile.read(str(audio_file))
            samples = int(waveform.shape[0]) if hasattr(waveform, "shape") else len(waveform)
            if sample_rate <= 0:
                sample_rate = _PARAKEET_SAMPLE_RATE
            capacity_frames = max(
                1,
                int(np.ceil(samples * (_PARAKEET_SAMPLE_RATE / float(sample_rate)) / _PARAKEET_HOP_LENGTH)) + 8,
            )
    if "whisper" in model_type.lower():
        capacity_frames = max(capacity_frames, 3000)
    capacity = max(1, capacity_frames * int(expected_mels))
    with limited_audio_file(audio_file, target_sample_rate=_PARAKEET_SAMPLE_RATE) as frontend_audio_file:
        values, mel_bins, frames = cactus_preprocess_audio_features(
            str(frontend_audio_file),
            model_type,
            int(expected_mels),
            capacity,
        )
    if int(mel_bins) != int(expected_mels):
        raise ValueError(f"Cactus audio frontend returned {mel_bins} mel bins, expected {expected_mels}")
    feature_array = np.asarray(values, dtype=np.float32).reshape(int(mel_bins), int(frames))
    active_frames = int(frames)
    if isinstance(expected_frames, int) and expected_frames > 0:
        active_frames = min(active_frames, expected_frames)
    if layout == "frames_mels":
        feature_array = feature_array[:, :active_frames].T
        if isinstance(expected_frames, int) and expected_frames > active_frames:
            feature_array = np.pad(feature_array, ((0, expected_frames - active_frames), (0, 0)), mode="constant")
    elif layout == "mels_frames":
        feature_array = feature_array[:, :active_frames]
        if isinstance(expected_frames, int) and expected_frames > active_frames:
            feature_array = np.pad(feature_array, ((0, 0), (0, expected_frames - active_frames)), mode="constant")
    else:
        raise ValueError(f"unsupported Cactus audio feature layout: {layout!r}")
    tensor = torch.from_numpy(np.ascontiguousarray(feature_array)).unsqueeze(0).to(dtype=torch_dtype)
    return tensor, active_frames


def prepare_native_parakeet_audio_features(
    audio_file: str | Path,
    *,
    expected_frames: int | None,
    expected_mels: int,
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    try:
        return prepare_cactus_audio_features(
            audio_file,
            model_type="parakeet",
            expected_frames=expected_frames,
            expected_mels=expected_mels,
            torch_dtype=torch_dtype,
        )
    except Exception:
        pass

    waveform = load_audio_waveform(audio_file, target_sample_rate=_PARAKEET_SAMPLE_RATE)
    features, feature_length = generic_log_mel_features(
        waveform,
        sample_rate=_PARAKEET_SAMPLE_RATE,
        num_mels=expected_mels or _PARAKEET_NUM_MELS,
        n_fft=_PARAKEET_N_FFT,
        hop_length=_PARAKEET_HOP_LENGTH,
        frame_length=_PARAKEET_FRAME_LENGTH,
        preemphasis=_PARAKEET_PREEMPHASIS,
        mel_floor=_PARAKEET_LOG_FLOOR,
        normalize_active_frames_only=False,
    )
    active_frames = feature_length
    if isinstance(expected_frames, int) and expected_frames > 0:
        active_frames = min(active_frames, expected_frames)
    features = features[:active_frames, :]
    if features.shape[1] != expected_mels:
        raise ValueError(
            f"feature mel dimension mismatch: expected {expected_mels}, got {features.shape[1]}"
        )
    if isinstance(expected_frames, int) and expected_frames > active_frames:
        features = np.pad(features, ((0, expected_frames - active_frames), (0, 0)), mode="constant")
    tensor = torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0).to(dtype=torch_dtype)
    return tensor, active_frames


def prepare_native_gemma4_audio_features(
    audio_file: str | Path,
    *,
    expected_mels: int,
    torch_dtype: torch.dtype,
    max_seconds: float | None = None,
    pad_to_max_seconds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if max_seconds is None:
        max_seconds = audio_duration_limit_seconds()
    waveform = load_audio_waveform(
        audio_file,
        target_sample_rate=_PARAKEET_SAMPLE_RATE,
        max_seconds=max_seconds,
    )
    # Native Gemma4 pads to the nearest 320 samples, adds 160 samples of
    # semicausal left padding, and then runs a 321-point analysis frame.
    estimated_frames = int(np.ceil((len(waveform) + 640) / 160.0)) + 8
    capacity = max(1, estimated_frames * int(expected_mels))
    values, mel_bins, frames = prepare_cactus_audio_features_raw(
        audio_file,
        model_type="gemma4",
        mel_bins=int(expected_mels),
        capacity=capacity,
        max_seconds=max_seconds,
    )
    if int(mel_bins) != int(expected_mels):
        raise ValueError(f"Cactus Gemma4 audio frontend returned {mel_bins} mel bins, expected {expected_mels}")
    max_frames = estimated_frames
    if max_seconds is not None and max_seconds > 0.0:
        max_samples = int(round(float(_PARAKEET_SAMPLE_RATE) * float(max_seconds)))
        max_frames = int(np.ceil((max_samples + 640) / 160.0)) + 8
    active_frames = min(int(frames), int(max_frames))
    features = np.asarray(values, dtype=np.float32).reshape(int(frames), int(mel_bins))[:active_frames, :]
    output_frames = max(active_frames, int(max_frames)) if pad_to_max_seconds else active_frames
    if output_frames > active_frames:
        features = np.pad(features, ((0, output_frames - active_frames), (0, 0)), mode="constant")
    feature_tensor = torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0).to(dtype=torch_dtype)
    mask = torch.zeros((1, int(output_frames)), dtype=torch.bool)
    mask[:, :active_frames] = True
    return feature_tensor, mask, int(active_frames)


def prepare_cactus_audio_features_raw(
    audio_file: str | Path,
    *,
    model_type: str,
    mel_bins: int,
    capacity: int,
    max_seconds: float | None = None,
) -> tuple[list[float], int, int]:
    from cactus.bindings.cactus import cactus_preprocess_audio_features

    with limited_audio_file(
        audio_file,
        target_sample_rate=_PARAKEET_SAMPLE_RATE,
        max_seconds=max_seconds,
    ) as frontend_audio_file:
        return cactus_preprocess_audio_features(
            str(frontend_audio_file),
            model_type,
            int(mel_bins),
            int(capacity),
        )
