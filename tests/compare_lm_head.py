#!/usr/bin/env python3
"""Check if the LM head produces the same logits when embeddings are 1/16 scaled."""

import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"


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

    # Run HF forward to get the final hidden state before LM head
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # hidden_states[0] is for the prefill step
    # It's a tuple of layer outputs; [-1] is the last layer
    last_layer_hidden = outputs.hidden_states[0][-1]  # [batch, seq_len, hidden]
    final_hidden = last_layer_hidden[0, -1, :]  # [hidden] - last position

    print(f"HF final hidden state: norm={final_hidden.norm():.4f}")
    print(f"  range=[{final_hidden.min():.4f}, {final_hidden.max():.4f}]")
    print(f"  mean={final_hidden.mean():.4f}, std={final_hidden.std():.4f}")

    # Get the LM head weight
    lm_head = model.lm_head.weight  # [vocab, hidden]
    print(f"\nHF LM head weight: shape={lm_head.shape}")
    norms = [f"{lm_head[i].norm().item():.4f}" for i in range(5)]
    print(f"  norm per row (first 5): {norms}")
    print(f"  avg row norm: {lm_head.norm(dim=1).mean():.4f}")

    # Compute logits manually
    logits = final_hidden @ lm_head.T  # [vocab]
    softcap = 30.0
    logits_capped = torch.tanh(logits / softcap) * softcap

    print(f"\nHF pre-softcap logits: range=[{logits.min():.2f}, {logits.max():.2f}]")
    print(f"HF post-softcap logits: range=[{logits_capped.min():.2f}, {logits_capped.max():.2f}]")

    # Top token
    top_idx = logits_capped.argmax().item()
    top_text = processor.tokenizer.decode([top_idx])
    print(f"HF top token: {top_idx} '{top_text}' logit={logits_capped[top_idx]:.2f}")

    # Now simulate what happens if LM head weights were /16
    # (as in cactus where token_embeddings = HF/16)
    lm_head_div16 = lm_head / 16.0
    logits_div16 = final_hidden @ lm_head_div16.T
    logits_div16_capped = torch.tanh(logits_div16 / softcap) * softcap

    print(f"\nWith LM head /16:")
    print(f"  Pre-softcap: range=[{logits_div16.min():.2f}, {logits_div16.max():.2f}]")
    print(f"  Post-softcap: range=[{logits_div16_capped.min():.2f}, {logits_div16_capped.max():.2f}]")
    top16 = logits_div16_capped.argmax().item()
    top16_text = processor.tokenizer.decode([top16])
    print(f"  Top token: {top16} '{top16_text}' logit={logits_div16_capped[top16]:.2f}")

    # Check if token rankings change
    _, hf_order = logits_capped.sort(descending=True)
    _, div16_order = logits_div16_capped.sort(descending=True)
    print(f"\n  Top-10 comparison:")
    for i in range(10):
        hf_t = hf_order[i].item()
        d16_t = div16_order[i].item()
        match = "OK" if hf_t == d16_t else "DIFF"
        print(f"    [{match}] rank {i}: HF={hf_t} '{processor.tokenizer.decode([hf_t])}' vs /16={d16_t} '{processor.tokenizer.decode([d16_t])}'")

    # Key punctuation tokens
    print(f"\n  Punctuation logits:")
    for tid, name in [(236761, '.'), (236764, ','), (236789, "'")]:
        hf_val = logits_capped[tid].item()
        d16_val = logits_div16_capped[tid].item()
        hf_rank = (logits_capped > logits_capped[tid]).sum().item() + 1
        d16_rank = (logits_div16_capped > logits_div16_capped[tid]).sum().item() + 1
        print(f"    '{name}': HF logit={hf_val:.2f} rank={hf_rank} | /16 logit={d16_val:.2f} rank={d16_rank}")

    # Now also check: what if both hidden AND lm_head are /16?
    # (this is what happens in cactus if the transformer normalizes the hidden state)
    # The output_norm (RMSNorm) will normalize the final hidden to a consistent scale
    # So the hidden state entering the LM head should be similar regardless of input scale.
    # The LM head being /16 means logits are /16.
    print(f"\n{'='*60}")
    print("Checking pre-softcap logit scale vs softcap effect:")
    print(f"  HF pre-softcap max: {logits.max():.2f}")
    print(f"  /16 pre-softcap max: {logits_div16.max():.2f}")
    print(f"  Ratio: {logits.max() / logits_div16.max():.2f}")
    print()
    print(f"  HF tanh compression at max: {torch.tanh(logits.max()/30):.4f}")
    print(f"  /16 tanh compression at max: {torch.tanh(logits_div16.max()/30):.4f}")
    print(f"  At comma (HF): pre={logits[236764]:.2f} tanh={torch.tanh(logits[236764]/30):.4f} post={logits_capped[236764]:.2f}")
    print(f"  At comma (/16): pre={logits_div16[236764]:.2f} tanh={torch.tanh(logits_div16[236764]/30):.4f} post={logits_div16_capped[236764]:.2f}")


if __name__ == "__main__":
    main()
