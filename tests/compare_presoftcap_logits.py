#!/usr/bin/env python3
"""Compare pre-softcap logits between HF and cactus at step 0."""

import struct
import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)
    tokenizer = processor.tokenizer

    audio_data, sr = sf.read(AUDIO_PATH)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe the audio."},
        {"type": "audio", "audio": audio_data, "sample_rate": sr},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )

    # Hook into the model to get pre-softcap logits
    # The LM head computes: logits = hidden @ lm_head.weight.T
    # Then softcap: tanh(logits/30)*30
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # Get final hidden state (post output_norm)
    last_hidden = outputs.hidden_states[0][-1][0, -1, :]  # [1536]

    # Compute pre-softcap logits manually
    lm_head_w = model.lm_head.weight  # [vocab, 1536]
    hf_pre_softcap = (last_hidden @ lm_head_w.T).detach().cpu().numpy()  # [vocab]

    print(f"HF pre-softcap: range=[{hf_pre_softcap.min():.2f}, {hf_pre_softcap.max():.2f}]")
    print(f"HF final hidden norm: {last_hidden.norm():.2f}")

    # Load cactus pre-softcap logits from the saved debug binary
    # We need to re-run cactus and save them. For now, let's simulate.
    # Actually, I saved merged_embeddings but not pre-softcap logits.
    # Let me compute what cactus logits SHOULD be if the hidden state direction matched.

    # Load cactus merged embeddings
    with open("/tmp/cactus_merged_embeddings.bin", "rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
        total = 1
        for d in shape:
            total *= d
        cactus_merged = np.frombuffer(f.read(total * 4), dtype=np.float32).reshape(shape)

    hf_merged = np.load("tests/hf_audio_encoder_output.npy")  # This is audio only

    # Instead, let's just compute what HF logits would be with 1/16 LM head
    # to isolate the softcap effect
    hf_post_softcap = np.tanh(hf_pre_softcap / 30.0) * 30.0

    # Simulate cactus: same hidden state, but LM head /16
    cactus_sim_pre = hf_pre_softcap / 16.0  # If hidden matches and LM head is /16
    cactus_sim_post = np.tanh(cactus_sim_pre / 30.0) * 30.0

    # Compare rankings
    hf_order = np.argsort(-hf_post_softcap)
    sim_order = np.argsort(-cactus_sim_post)

    print(f"\nIf the only difference were 1/16 LM head (same hidden state):")
    print(f"  Simulated pre-softcap: range=[{cactus_sim_pre.min():.2f}, {cactus_sim_pre.max():.2f}]")
    print(f"  Simulated post-softcap: range=[{cactus_sim_post.min():.2f}, {cactus_sim_post.max():.2f}]")

    print(f"\n  Top-10 comparison (HF vs simulated-cactus):")
    for i in range(10):
        hf_t = hf_order[i]
        sim_t = sim_order[i]
        match = "OK  " if hf_t == sim_t else "DIFF"
        print(f"    [{match}] rank {i}: HF={hf_t} '{tokenizer.decode([hf_t])}' (logit={hf_post_softcap[hf_t]:.2f}) "
              f"vs sim={sim_t} '{tokenizer.decode([sim_t])}' (logit={cactus_sim_post[sim_t]:.2f})")

    # Check if softcap changes rankings when logits are /16
    print(f"\n  Do rankings change with /16 logits through softcap?")
    # Pre-softcap rankings (these should be identical since /16 is monotonic)
    pre_order_hf = np.argsort(-hf_pre_softcap)
    pre_order_sim = np.argsort(-cactus_sim_pre)
    rankings_match = np.array_equal(pre_order_hf, pre_order_sim)
    print(f"    Pre-softcap rankings identical: {rankings_match}")

    # Post-softcap rankings
    post_rankings_match = np.array_equal(hf_order, sim_order)
    print(f"    Post-softcap rankings identical: {post_rankings_match}")

    if not post_rankings_match:
        # Find first divergence
        for i in range(len(hf_order)):
            if hf_order[i] != sim_order[i]:
                print(f"    First divergence at rank {i}: HF={hf_order[i]} vs sim={sim_order[i]}")
                break

    # Key punctuation analysis
    print(f"\n  Punctuation token analysis:")
    for tid, name in [(236761, '.'), (236764, ','), (236789, "'")]:
        hf_pre = hf_pre_softcap[tid]
        hf_post = hf_post_softcap[tid]
        sim_pre = cactus_sim_pre[tid]
        sim_post = cactus_sim_post[tid]
        hf_rank = (hf_post_softcap > hf_post).sum() + 1
        sim_rank = (cactus_sim_post > sim_post).sum() + 1
        print(f"    '{name}': HF pre={hf_pre:.2f} post={hf_post:.2f} rank={hf_rank} | "
              f"sim pre={sim_pre:.2f} post={sim_post:.2f} rank={sim_rank}")

    # Now THE KEY TEST: what do actual cactus logits look like?
    # From our debug: pre_softcap range=[-128.75, 7.50]
    # If 16x/16 cancelled perfectly, we'd expect same as HF: [-164, 22.8]
    # But we got [-128.75, 7.50] - that's NOT /16 of HF, it's different
    print(f"\n{'='*60}")
    print(f"ACTUAL cactus pre-softcap (from debug): [-128.75, 7.50]")
    print(f"HF pre-softcap: [{hf_pre_softcap.min():.2f}, {hf_pre_softcap.max():.2f}]")
    print(f"If 16x/16 cancelled: should match HF")
    print(f"Ratio max: {hf_pre_softcap.max() / 7.50:.2f}x")
    print(f"Ratio min: {hf_pre_softcap.min() / -128.75:.2f}x")


if __name__ == "__main__":
    main()
