#!/usr/bin/env python3
"""Test whether INT8 quantization of the LM head causes punctuation loss."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"
TASK_TEXT = "Transcribe the audio."

PUNCT_TOKENS = {236761: '.', 236764: ',', 236789: "'"}


def quantize_per_group_int8(weight, group_size=32):
    """Simulate per-group INT8 quantization matching cactus format."""
    rows, cols = weight.shape
    # Group along columns (K dimension)
    num_groups = (cols + group_size - 1) // group_size

    q_weight = torch.zeros_like(weight, dtype=torch.int8)
    scales = torch.zeros(rows, num_groups, dtype=torch.float16)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, cols)
        group = weight[:, start:end].float()

        # Per-row per-group quantization
        max_abs = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        scale = max_abs / 127.0
        scales[:, g] = scale.squeeze().half()

        q_weight[:, start:end] = torch.clamp(
            torch.round(group / scale), -128, 127
        ).to(torch.int8)

    return q_weight, scales


def dequantize_per_group(q_weight, scales, group_size=32):
    """Dequantize INT8 weights back to float."""
    rows, cols = q_weight.shape
    num_groups = scales.shape[1]

    result = torch.zeros(rows, cols, dtype=torch.float32)
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, cols)
        result[:, start:end] = q_weight[:, start:end].float() * scales[:, g:g+1].float()

    return result


def int8_matmul(activation_fp16, q_weight, scales, group_size=32):
    """Simulate INT8 matmul: quantize activations, do int8 mul, dequantize."""
    # activation: [1, K], weight: [N, K] (transposed), output: [1, N]
    M, K = activation_fp16.shape
    N = q_weight.shape[0]

    # Quantize activations per-row
    act_float = activation_fp16.float()
    act_max = act_float.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    act_scale = act_max / 127.0
    act_int8 = torch.clamp(torch.round(act_float / act_scale), -128, 127).to(torch.int8)

    # Do grouped int8 matmul
    num_groups = scales.shape[1]
    output = torch.zeros(M, N, dtype=torch.float32)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, K)

        # int8 partial dot product
        partial = act_int8[:, start:end].float() @ q_weight[:, start:end].float().T
        # Scale by both scales
        output += partial * (act_scale * scales[:, g:g+1].float().T)

    return output.half()  # Return as FP16 like cactus


def main():
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)

    print("Loading audio...")
    audio_data, sr = sf.read(AUDIO_PATH)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": TASK_TEXT},
        {"type": "audio", "audio": audio_data, "sample_rate": sr},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )

    # Get the embedding/LM head weight
    lm_head_weight = model.lm_head.weight.data  # [vocab_size, hidden_dim]
    print(f"LM head weight shape: {lm_head_weight.shape}, dtype: {lm_head_weight.dtype}")

    # Quantize to INT8
    print("Quantizing LM head to INT8 (group_size=32)...")
    q_weight, scales = quantize_per_group_int8(lm_head_weight, group_size=32)

    # Check quantization error for punctuation tokens
    deq_weight = dequantize_per_group(q_weight, scales, group_size=32)
    for tid, char in PUNCT_TOKENS.items():
        orig = lm_head_weight[tid].float()
        deq = deq_weight[tid]
        cos_sim = torch.nn.functional.cosine_similarity(orig.unsqueeze(0), deq.unsqueeze(0)).item()
        rmse = ((orig - deq) ** 2).mean().sqrt().item()
        print(f"  Token '{char}' (id={tid}): cos_sim={cos_sim:.6f}, rmse={rmse:.6f}")

    # Run model forward to get hidden states
    print("\nRunning model forward pass...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: {text}")

    # Get hidden states at each step and compare FP32 vs INT8 logits
    softcap = 30.0

    print(f"\n{'='*80}")
    print(f"Comparing FP32 vs INT8-quantized logits at each step:")
    print(f"{'='*80}")

    for step_idx, hidden_states in enumerate(outputs.hidden_states):
        if step_idx >= 60:
            break

        # hidden_states is a tuple of layer outputs; last layer is [-1]
        # Shape: [batch, seq_len, hidden_dim]
        last_hidden = hidden_states[-1][0, -1, :]  # [hidden_dim]

        # FP32 logits
        fp32_logits = (last_hidden @ lm_head_weight.T)  # [vocab_size]
        fp32_logits = torch.tanh(fp32_logits / softcap) * softcap

        # INT8 logits (simulating cactus)
        int8_logits = int8_matmul(
            last_hidden.half().unsqueeze(0),  # [1, hidden_dim]
            q_weight, scales, group_size=32
        )[0].float()  # [vocab_size]
        int8_logits = torch.tanh(int8_logits / softcap) * softcap

        fp32_top = torch.argmax(fp32_logits).item()
        int8_top = torch.argmax(int8_logits).item()
        fp32_text = tokenizer.decode([fp32_top])
        int8_text = tokenizer.decode([int8_top])

        match = "OK  " if fp32_top == int8_top else "DIFF"
        actual_token = generated_ids[step_idx].item() if step_idx < len(generated_ids) else -1
        actual_text = tokenizer.decode([actual_token]) if actual_token >= 0 else '?'

        print(f"\nStep {step_idx:3d} [{match}]: actual='{actual_text}' FP32='{fp32_text}' INT8='{int8_text}'")

        for tid, char in PUNCT_TOKENS.items():
            fp32_val = fp32_logits[tid].item()
            int8_val = int8_logits[tid].item()
            fp32_rank = (fp32_logits > fp32_logits[tid]).sum().item() + 1
            int8_rank = (int8_logits > int8_logits[tid]).sum().item() + 1
            print(f"  '{char}': FP32 rank={fp32_rank:5d} logit={fp32_val:8.2f} | INT8 rank={int8_rank:5d} logit={int8_val:8.2f} | rank_delta={int8_rank-fp32_rank:+d}")


if __name__ == "__main__":
    main()
