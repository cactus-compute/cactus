#!/usr/bin/env python3
"""Compare final merged embeddings between HF and cactus."""

import struct
import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"


def load_cactus_embeddings(path="/tmp/cactus_merged_embeddings.bin"):
    with open(path, "rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack("<Q", f.read(8))[0])
        total = 1
        for d in shape:
            total *= d
        data = np.frombuffer(f.read(total * 4), dtype=np.float32).reshape(shape)
    return data


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)

    audio_data, sr = sf.read(AUDIO_PATH)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe the audio."},
        {"type": "audio", "audio": audio_data, "sample_rate": sr},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )

    input_ids = inputs["input_ids"][0]
    print(f"HF token_ids ({len(input_ids)}): {input_ids[:20].tolist()} ...")
    print(f"HF token_ids (last 10): ... {input_ids[-10:].tolist()}")

    # Cactus token sequence from debug output
    cactus_tokens = [2, 105, 2364, 107, 5183, 17038, 506, 9855, 236761, 256000]
    print(f"\nCactus token_ids (first 10): {cactus_tokens}")

    # Decode both to text to compare
    tokenizer = processor.tokenizer
    hf_prefix = tokenizer.decode(input_ids[:10], skip_special_tokens=False)
    cactus_prefix = tokenizer.decode(cactus_tokens, skip_special_tokens=False)
    print(f"\nHF prefix decoded: {repr(hf_prefix)}")
    print(f"Cactus prefix decoded: {repr(cactus_prefix)}")

    # Check token-by-token
    print(f"\nToken-by-token comparison (first 15):")
    for i in range(min(15, len(input_ids))):
        hf_tok = input_ids[i].item()
        hf_text = tokenizer.decode([hf_tok])
        print(f"  [{i:3d}] HF={hf_tok:6d} '{hf_text}'")

    # Find where audio tokens start in HF
    audio_token_id = 258881
    audio_start = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
    if len(audio_start) > 0:
        print(f"\nHF audio tokens: start at position {audio_start[0].item()}, count={len(audio_start)}")

    # Suffix tokens (after audio)
    audio_end = audio_start[-1].item() + 1 if len(audio_start) > 0 else len(input_ids)
    print(f"HF suffix tokens: {input_ids[audio_end:].tolist()}")
    suffix_text = tokenizer.decode(input_ids[audio_end:], skip_special_tokens=False)
    print(f"HF suffix decoded: {repr(suffix_text)}")

    # Now get the HF merged embeddings
    embed_tokens = model.model.language_model.embed_tokens
    embed_scale = embed_tokens.embed_scale.item()
    print(f"\nHF embed_scale: {embed_scale}")

    with torch.no_grad():
        # Get text embeddings (scaled)
        text_embeds = embed_tokens(input_ids.unsqueeze(0))  # [1, seq_len, hidden]

        # Get audio features
        input_features = inputs["input_features"]
        input_features_mask = inputs["input_features_mask"]
        audio_outputs = model.model.audio_tower(input_features, input_features_mask, return_dict=True)
        audio_projected = model.model.embed_audio(inputs_embeds=audio_outputs.last_hidden_state)

        # Strip padding
        audio_mask_from_encoder = audio_outputs.attention_mask
        audio_valid = audio_projected[audio_mask_from_encoder]

    # Build merged embeddings like HF does
    hf_merged = text_embeds[0].clone()  # [seq_len, hidden]
    audio_mask = (input_ids == audio_token_id)
    n_audio = audio_mask.sum().item()
    print(f"Audio tokens in HF: {n_audio}, valid audio features: {len(audio_valid)}")

    # Replace audio positions
    hf_merged[audio_mask] = audio_valid.to(hf_merged.dtype)

    hf_merged_np = hf_merged.cpu().numpy()
    print(f"\nHF merged embeddings: shape={hf_merged_np.shape}")

    # Load cactus embeddings
    cactus_merged = load_cactus_embeddings()
    print(f"Cactus merged embeddings: shape={cactus_merged.shape}")

    # Compare shapes
    if hf_merged_np.shape != cactus_merged.shape:
        print(f"SHAPE MISMATCH: HF={hf_merged_np.shape} vs cactus={cactus_merged.shape}")

    seq_len = min(hf_merged_np.shape[0], cactus_merged.shape[0])

    # Compare row norms
    print(f"\n{'='*80}")
    print(f"Row-by-row comparison (first 20 rows, then audio region, then suffix):")
    print(f"{'='*80}")

    for i in range(min(20, seq_len)):
        hf_norm = np.linalg.norm(hf_merged_np[i])
        c_norm = np.linalg.norm(cactus_merged[i])
        cos = np.dot(hf_merged_np[i], cactus_merged[i]) / (hf_norm * c_norm + 1e-10)
        is_audio = audio_mask[i].item() if i < len(audio_mask) else False
        label = "AUDIO" if is_audio else "TEXT"
        print(f"  [{i:3d}] {label:5s} HF_norm={hf_norm:8.4f} C_norm={c_norm:8.4f} ratio={hf_norm/(c_norm+1e-10):6.2f} cos={cos:.6f}")

    # Audio region summary
    if len(audio_start) > 0:
        a_start = audio_start[0].item()
        a_end = audio_start[-1].item() + 1
        hf_audio_norms = np.linalg.norm(hf_merged_np[a_start:a_end], axis=1)
        c_audio_norms = np.linalg.norm(cactus_merged[a_start:a_end], axis=1)
        print(f"\n  Audio region [{a_start}:{a_end}]:")
        print(f"    HF norms:     mean={hf_audio_norms.mean():.4f} std={hf_audio_norms.std():.4f}")
        print(f"    Cactus norms: mean={c_audio_norms.mean():.4f} std={c_audio_norms.std():.4f}")
        print(f"    Norm ratio:   mean={np.mean(hf_audio_norms / (c_audio_norms + 1e-10)):.4f}")

        # Cosine similarities in audio region
        cos_sims = []
        for i in range(a_start, a_end):
            cos = np.dot(hf_merged_np[i], cactus_merged[i]) / (np.linalg.norm(hf_merged_np[i]) * np.linalg.norm(cactus_merged[i]) + 1e-10)
            cos_sims.append(cos)
        print(f"    Cosine sim:   mean={np.mean(cos_sims):.6f} min={np.min(cos_sims):.6f}")

    # Suffix tokens comparison
    if audio_end < seq_len:
        print(f"\n  Suffix tokens [{audio_end}:{seq_len}]:")
        for i in range(audio_end, min(seq_len, audio_end + 10)):
            hf_norm = np.linalg.norm(hf_merged_np[i])
            c_norm = np.linalg.norm(cactus_merged[i])
            cos = np.dot(hf_merged_np[i], cactus_merged[i]) / (hf_norm * c_norm + 1e-10)
            print(f"    [{i:3d}] HF_norm={hf_norm:8.4f} C_norm={c_norm:8.4f} ratio={hf_norm/(c_norm+1e-10):6.2f} cos={cos:.6f}")


if __name__ == "__main__":
    main()
