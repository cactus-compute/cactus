#!/usr/bin/env python3
"""Extract audio encoder output from HF and compare with cactus debug output."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)

    audio_data, sr = sf.read(AUDIO_PATH)
    print(f"Audio: {len(audio_data)} samples, {sr} Hz")

    # Get mel features from HF processor
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe the audio."},
        {"type": "audio", "audio": audio_data, "sample_rate": sr},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )

    input_features = inputs["input_features"]
    input_features_mask = inputs["input_features_mask"]
    print(f"Input features: {input_features.shape}")  # [1, num_frames, 128]

    # Run audio tower (conformer)
    with torch.no_grad():
        audio_outputs = model.model.audio_tower(
            input_features, input_features_mask, return_dict=True
        )

    last_hidden = audio_outputs.last_hidden_state
    print(f"Audio tower output: {last_hidden.shape}")  # [1, num_soft_tokens, hidden_dim]
    print(f"  range: [{last_hidden.min():.4f}, {last_hidden.max():.4f}]")
    print(f"  mean: {last_hidden.mean():.4f}, std: {last_hidden.std():.4f}")

    # Run through embed_audio (projector)
    with torch.no_grad():
        projected = model.model.embed_audio(inputs_embeds=last_hidden)
    print(f"Audio projected: {projected.shape}")
    print(f"  range: [{projected.min():.4f}, {projected.max():.4f}]")
    print(f"  mean: {projected.mean():.4f}, std: {projected.std():.4f}")

    # Get text embeddings for comparison
    embed_tokens = model.model.language_model.embed_tokens
    embed_scale = embed_tokens.embed_scale.item()
    print(f"\nText embedding scale: {embed_scale}")

    # Check a few text token embeddings
    test_tokens = torch.tensor([[2, 236761, 236764]])  # BOS, '.', ','
    with torch.no_grad():
        text_embeds = embed_tokens(test_tokens)
    print(f"Text embed range: [{text_embeds.min():.4f}, {text_embeds.max():.4f}]")
    print(f"Text embed mean: {text_embeds.mean():.4f}, std: {text_embeds.std():.4f}")

    # Compare norms
    audio_norms = projected[0].norm(dim=-1)  # [num_soft_tokens]
    text_norm_bos = text_embeds[0, 0].norm().item()
    text_norm_period = text_embeds[0, 1].norm().item()
    text_norm_comma = text_embeds[0, 2].norm().item()
    print(f"\nAudio embedding norms: mean={audio_norms.mean():.4f}, std={audio_norms.std():.4f}")
    print(f"  min={audio_norms.min():.4f}, max={audio_norms.max():.4f}")
    print(f"Text embedding norms: BOS={text_norm_bos:.4f}, '.'={text_norm_period:.4f}, ','={text_norm_comma:.4f}")

    # Save audio encoder output for comparison with cactus
    np.save("tests/hf_audio_encoder_output.npy", projected[0].cpu().numpy())
    np.save("tests/hf_mel_features.npy", input_features[0].cpu().numpy())
    print(f"\nSaved HF audio encoder output to tests/hf_audio_encoder_output.npy")
    print(f"Saved HF mel features to tests/hf_mel_features.npy")

    # Now check: what happens if we scale audio features differently?
    # Try the full forward pass with different audio scaling
    print(f"\n{'='*60}")
    print("Testing sensitivity to audio feature scaling...")

    # Get the full input embeddings
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        text_embeds_full = embed_tokens(input_ids)

    # Audio token mask
    audio_token_id = 258881
    audio_mask = (input_ids == audio_token_id)
    n_audio = audio_mask.sum().item()
    print(f"Audio tokens in prompt: {n_audio}")
    print(f"Projected audio tokens: {projected.shape[1]}")

    # Check attention mask from audio encoder
    if hasattr(audio_outputs, 'attention_mask') and audio_outputs.attention_mask is not None:
        valid_mask = audio_outputs.attention_mask
        n_valid = valid_mask.sum().item()
        print(f"Valid audio encoder output tokens: {n_valid}")
        # Get only valid tokens
        valid_projected = projected[valid_mask]
        print(f"After masking: {valid_projected.shape}")


if __name__ == "__main__":
    main()
