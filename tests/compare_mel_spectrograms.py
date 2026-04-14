#!/usr/bin/env python3
"""Compare mel spectrogram between HF and cactus-style for Gemma 4."""

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    fe = processor.feature_extractor

    audio_data, sr = sf.read(AUDIO_PATH)
    print(f"Audio: {len(audio_data)} samples, {sr} Hz")

    # HF feature extraction
    print("\n--- HF Audio Processing ---")
    print(f"  frame_length={fe.frame_length}, hop_length={fe.hop_length}")
    print(f"  fft_length={fe.fft_length}, mel_filters shape={fe.mel_filters.shape}")

    # Run HF feature extraction manually to get mel
    waveform = audio_data.copy()
    if waveform.ndim == 1:
        waveform = np.expand_dims(waveform, axis=0)

    # Semicausal padding
    pad_left = fe.frame_length // 2  # 160
    waveform = np.pad(waveform, ((0, 0), (pad_left, 0)), mode="constant")

    frame_size_for_unfold = fe.frame_length + 1  # 321

    # Unfold
    def unfold_1d(arr, size, step):
        n = arr.shape[-1]
        num_frames = (n - size) // step + 1
        idx = np.arange(size)[None, :] + np.arange(num_frames)[:, None] * step
        return arr[..., idx]

    frames = unfold_1d(waveform, frame_size_for_unfold, fe.hop_length)
    # Drop last sample (preemphasis=0 path)
    frames = frames[..., :-1]  # [1, num_frames, 320]

    # Apply window and FFT
    frames = frames * fe.window
    stft = np.fft.rfft(frames, n=fe.fft_length, axis=-1)
    magnitude = np.abs(stft)  # [1, num_frames, 257]

    # Apply mel filterbank
    hf_mel = np.matmul(magnitude, fe.mel_filters)  # [1, num_frames, 128]
    hf_log_mel = np.log(hf_mel + fe.mel_floor)  # additive floor, then log

    hf_log_mel = hf_log_mel[0]  # Remove batch dim: [num_frames, 128]
    print(f"  HF mel shape: {hf_log_mel.shape}")
    print(f"  HF mel range: [{hf_log_mel.min():.4f}, {hf_log_mel.max():.4f}]")
    print(f"  HF mel mean: {hf_log_mel.mean():.4f}, std: {hf_log_mel.std():.4f}")

    # Now replicate cactus-style mel computation
    print("\n--- Cactus-style Audio Processing ---")
    # Cactus params:
    # n_fft=321, frame_length=320, fft_override=512, hop_length=160
    # power=1.0, center=false, hann_periodic=true, mel_floor=0.001 (additive)
    # mel filters: 257 bins, 128 mel, htk scale, norm=nullptr (no slaney norm)
    # analysis_frame_length = n_fft = 321

    c_audio = audio_data.copy()
    # Pad to multiple of 320
    remainder = len(c_audio) % 320
    if remainder != 0:
        c_audio = np.concatenate([c_audio, np.zeros(320 - remainder)])

    # Semicausal padding (160 zeros at front)
    c_audio = np.concatenate([np.zeros(160), c_audio])

    # Window: 320-element periodic Hann, left-padded in a 321-element array
    # left_pad = (321 - 320) / 2 = 0 (integer division)
    analysis_frame_length = 321
    window_length = 320
    window = np.zeros(analysis_frame_length)
    denom = float(window_length)  # periodic
    for i in range(window_length):
        window[i] = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / denom)

    # Extract frames with hop=160, frame=321
    num_frames_cactus = 1 + (len(c_audio) - analysis_frame_length) // 160
    c_frames = np.zeros((num_frames_cactus, analysis_frame_length))
    for i in range(num_frames_cactus):
        start = i * 160
        c_frames[i] = c_audio[start:start+analysis_frame_length]

    # Apply window
    c_frames = c_frames * window

    # FFT with n=512
    c_stft = np.fft.rfft(c_frames, n=512, axis=-1)
    c_magnitude = np.abs(c_stft)  # [num_frames, 257]

    # On Apple, vDSP returns 2x, so cactus divides by 2 before floor+log
    # We simulate this: magnitude * 0.5
    c_magnitude_apple = c_magnitude * 0.5

    # Mel filterbank: 257 bins, 128 mel, htk, no slaney norm
    # Use HF's mel filters (should be the same since both use htk, 257 bins, 128 mel, 0-8000Hz)
    c_mel = np.matmul(c_magnitude, fe.mel_filters)
    c_mel_apple = np.matmul(c_magnitude_apple, fe.mel_filters)

    c_log_mel = np.log(c_mel + 0.001)
    c_log_mel_apple = np.log(c_mel_apple + 0.001)

    print(f"  Cactus mel shape: {c_log_mel.shape}")
    print(f"  Cactus mel range (no Apple fix): [{c_log_mel.min():.4f}, {c_log_mel.max():.4f}]")
    print(f"  Cactus mel mean (no Apple fix): {c_log_mel.mean():.4f}, std: {c_log_mel.std():.4f}")
    print(f"  Cactus mel range (Apple *0.5): [{c_log_mel_apple.min():.4f}, {c_log_mel_apple.max():.4f}]")
    print(f"  Cactus mel mean (Apple *0.5): {c_log_mel_apple.mean():.4f}, std: {c_log_mel_apple.std():.4f}")

    # Compare frame counts
    print(f"\n--- Frame Count Comparison ---")
    print(f"  HF frames: {hf_log_mel.shape[0]}")
    print(f"  Cactus frames: {c_log_mel.shape[0]}")

    # Compare mel values
    min_frames = min(hf_log_mel.shape[0], c_log_mel.shape[0])

    print(f"\n--- Mel Comparison (first {min_frames} frames) ---")

    # Without Apple fix
    diff = np.abs(hf_log_mel[:min_frames] - c_log_mel[:min_frames])
    flat_hf = hf_log_mel[:min_frames].flatten()
    flat_c = c_log_mel[:min_frames].flatten()
    cos_sim = np.dot(flat_hf, flat_c) / (np.linalg.norm(flat_hf) * np.linalg.norm(flat_c))
    print(f"  No Apple fix: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, cos={cos_sim:.6f}")

    # With Apple fix
    diff_apple = np.abs(hf_log_mel[:min_frames] - c_log_mel_apple[:min_frames])
    flat_ca = c_log_mel_apple[:min_frames].flatten()
    cos_sim_apple = np.dot(flat_hf, flat_ca) / (np.linalg.norm(flat_hf) * np.linalg.norm(flat_ca))
    print(f"  Apple *0.5:   max_diff={diff_apple.max():.6f}, mean_diff={diff_apple.mean():.6f}, cos={cos_sim_apple:.6f}")

    # Show first frame comparison
    print(f"\n--- First frame (first 10 mel bins) ---")
    print(f"  HF:           {hf_log_mel[0,:10]}")
    print(f"  Cactus:       {c_log_mel[0,:10]}")
    print(f"  Cactus Apple: {c_log_mel_apple[0,:10]}")

    # Check if the mel filter banks might differ
    # Cactus uses norm=nullptr (no normalization), HF uses htk mel scale
    # Let's also check with slaney norm
    from transformers.audio_utils import mel_filter_bank
    mel_htk_no_norm = mel_filter_bank(
        num_frequency_bins=257,
        num_mel_filters=128,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm=None,
        mel_scale="htk",
    )
    mel_htk_slaney_norm = mel_filter_bank(
        num_frequency_bins=257,
        num_mel_filters=128,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="htk",
    )

    print(f"\n--- Mel Filter Comparison ---")
    print(f"  HF mel_filters shape: {fe.mel_filters.shape}")
    print(f"  HTK no norm shape: {mel_htk_no_norm.shape}")
    print(f"  HTK slaney norm shape: {mel_htk_slaney_norm.shape}")

    diff_hf_nonorm = np.abs(fe.mel_filters - mel_htk_no_norm).max()
    diff_hf_slaney = np.abs(fe.mel_filters - mel_htk_slaney_norm).max()
    print(f"  HF vs HTK no-norm: max_diff={diff_hf_nonorm:.8f}")
    print(f"  HF vs HTK slaney: max_diff={diff_hf_slaney:.8f}")

    # Also try with no-norm mel filters
    c_mel_nonorm = np.matmul(c_magnitude, mel_htk_no_norm)
    c_log_mel_nonorm = np.log(c_mel_nonorm + 0.001)
    c_mel_nonorm_apple = np.matmul(c_magnitude_apple, mel_htk_no_norm)
    c_log_mel_nonorm_apple = np.log(c_mel_nonorm_apple + 0.001)

    diff_nonorm = np.abs(hf_log_mel[:min_frames] - c_log_mel_nonorm[:min_frames])
    diff_nonorm_apple = np.abs(hf_log_mel[:min_frames] - c_log_mel_nonorm_apple[:min_frames])
    print(f"\n  With no-norm mel filters:")
    print(f"    No Apple: max_diff={diff_nonorm.max():.6f}, mean_diff={diff_nonorm.mean():.6f}")
    print(f"    Apple *0.5: max_diff={diff_nonorm_apple.max():.6f}, mean_diff={diff_nonorm_apple.mean():.6f}")

    # Soft token count
    print(f"\n--- Soft Token Count ---")
    after_stage1 = (min_frames + 1) // 2
    num_soft_tokens = (after_stage1 + 1) // 2
    print(f"  Frames: {min_frames} -> after_stage1: {after_stage1} -> soft_tokens: {num_soft_tokens}")

    # Now run through HF model to get actual audio encoder output
    print(f"\n--- Running HF model to get audio encoder output ---")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe the audio."},
        {"type": "audio", "audio": audio_data, "sample_rate": sr},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )

    # Check how many audio tokens in the input
    audio_token_id = 258881  # Gemma4 audio token ID
    input_ids = inputs["input_ids"][0]
    num_audio_tokens = (input_ids == audio_token_id).sum().item()
    print(f"  Number of audio tokens in prompt: {num_audio_tokens}")
    print(f"  Total prompt tokens: {len(input_ids)}")

    # Check if audio_values are in inputs
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, (int, float)):
            print(f"  {k}: {v}")
        elif isinstance(v, list) and len(v) > 0:
            print(f"  {k}: list of {len(v)}")


if __name__ == "__main__":
    main()
