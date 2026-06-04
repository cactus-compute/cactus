#!/usr/bin/env python3
"""Generate reference audio encoder outputs from HF model for C++ validation.

Uses the SAME HF model weights that were converted to cactus format,
so the only differences should be from quantization and FP16 precision."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HF_MODEL_PATH = "gg-hf-gg/gemma-4-e2b-it"

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'assets')
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    num_frames = 100
    mel_bins = 128
    mel_input = torch.randn(1, num_frames, mel_bins, dtype=torch.float32)
    mel_mask = torch.zeros(1, num_frames, dtype=torch.bool)

    mel_flat = mel_input.squeeze(0).numpy()
    np.save(os.path.join(output_dir, 'audio_test_mel_input.npy'), mel_input.numpy())
    mel_flat.astype(np.float32).tofile(os.path.join(output_dir, 'audio_test_mel_input.bin'))
    print(f"Saved mel input: shape={mel_input.shape}")

    print(f"\nLoading HF model from {HF_MODEL_PATH}...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH, dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()

    audio_tower = model.model.audio_tower
    embed_audio = model.model.embed_audio

    hooks = {}
    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hooks[name] = output[0].detach().float()
            else:
                hooks[name] = output.detach().float()
        return hook_fn

    audio_tower.subsample_conv_projection.register_forward_hook(make_hook('sscp'))
    for i in range(min(12, len(audio_tower.conformer))):
        audio_tower.conformer[i].register_forward_hook(make_hook(f'conformer_{i}'))

    with torch.no_grad():
        encoder_out, encoder_mask = audio_tower(mel_input, mel_mask)
        projected = embed_audio(inputs_embeds=encoder_out)

    for name, tensor in sorted(hooks.items()):
        t = tensor.squeeze(0).numpy()
        fname = f'audio_ref_{name}.npy'
        np.save(os.path.join(output_dir, fname), t)
        t.astype(np.float32).tofile(os.path.join(output_dir, f'audio_ref_{name}.bin'))
        print(f"  {name}: shape={t.shape}, mean={t.mean():.4f}, std={t.std():.4f}")

    enc_np = encoder_out.squeeze(0).float().numpy()
    np.save(os.path.join(output_dir, 'audio_ref_encoder.npy'), enc_np)
    enc_np.astype(np.float32).tofile(os.path.join(output_dir, 'audio_ref_encoder.bin'))
    print(f"  encoder: shape={enc_np.shape}, mean={enc_np.mean():.4f}, std={enc_np.std():.4f}")

    proj_np = projected.squeeze(0).float().numpy()
    np.save(os.path.join(output_dir, 'audio_ref_projected.npy'), proj_np)
    proj_np.astype(np.float32).tofile(os.path.join(output_dir, 'audio_ref_projected.bin'))
    print(f"  projected: shape={proj_np.shape}, mean={proj_np.mean():.4f}, std={proj_np.std():.4f}")

    print("\nDone. Reference files in", output_dir)

if __name__ == '__main__':
    main()
