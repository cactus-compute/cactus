#!/usr/bin/env python3
"""Pinpoint which INT4-quantized weights cause punctuation loss."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"
GROUP_SIZE = 32


def quantize_to_int4(weight):
    """Simulate INT4 group-wise quantization (same as cactus conversion)."""
    rows, cols = weight.shape
    q = torch.zeros_like(weight, dtype=torch.float16)
    for g in range(0, cols, GROUP_SIZE):
        end = min(g + GROUP_SIZE, cols)
        group = weight[:, g:end].float()
        max_abs = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        scale = max_abs / 7.0  # INT4 range is [-8, 7]
        quantized = torch.clamp(torch.round(group / scale), -8, 7)
        q[:, g:end] = (quantized * scale).half()
    return q


def run_transcribe(model, processor, audio_data, sr):
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Transcribe the audio.'},
        {'type': 'audio', 'audio': audio_data, 'sample_rate': sr},
    ]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
        return_tensors='pt', add_generation_prompt=True)
    prompt_len = inputs['input_ids'].shape[1]

    fp16_inputs = {}
    for k, v in inputs.items():
        fp16_inputs[k] = v.to(torch.float16) if v.dtype == torch.float32 else v

    with torch.no_grad():
        out = model.generate(**fp16_inputs, max_new_tokens=80, do_sample=False)
    return processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def has_punctuation(text):
    return any(c in text for c in '.,;:!?')


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    audio_data, sr = sf.read(AUDIO_PATH)

    print("Loading FP16 model...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float16)

    # Baseline
    baseline = run_transcribe(model, processor, audio_data, sr)
    print(f"Baseline FP16: {baseline}")
    print(f"Has punctuation: {has_punctuation(baseline)}")

    # Group weights by type
    weight_groups = {
        'attn_qkv': [],
        'attn_output': [],
        'ffn_gate_up': [],
        'ffn_down': [],
        'per_layer': [],
        'lm_head': [],
    }

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            weight_groups['attn_qkv'].append(name)
        elif 'o_proj' in name:
            weight_groups['attn_output'].append(name)
        elif 'gate_proj' in name or 'up_proj' in name:
            weight_groups['ffn_gate_up'].append(name)
        elif 'down_proj' in name:
            weight_groups['ffn_down'].append(name)
        elif 'per_layer' in name and 'norm' not in name:
            weight_groups['per_layer'].append(name)
        elif 'lm_head' in name:
            weight_groups['lm_head'].append(name)

    for group_name, param_names in weight_groups.items():
        if not param_names:
            continue
        print(f"\n{'='*60}")
        print(f"Quantizing {group_name} ({len(param_names)} weights) to INT4...")

        # Save originals
        originals = {}
        for name in param_names:
            param = dict(model.named_parameters())[name]
            originals[name] = param.data.clone()
            param.data = quantize_to_int4(param.data)

        result = run_transcribe(model, processor, audio_data, sr)
        punct = has_punctuation(result)
        status = "OK" if punct else "BROKEN"
        print(f"  [{status}] {result}")

        # Restore
        for name in param_names:
            dict(model.named_parameters())[name].data = originals[name]

    # Now test ALL weights quantized together
    print(f"\n{'='*60}")
    print("Quantizing ALL 2D weights to INT4...")
    originals = {}
    for name, param in model.named_parameters():
        if param.ndim == 2:
            originals[name] = param.data.clone()
            param.data = quantize_to_int4(param.data)

    result = run_transcribe(model, processor, audio_data, sr)
    punct = has_punctuation(result)
    status = "OK" if punct else "BROKEN"
    print(f"  [{status}] {result}")

    for name in originals:
        dict(model.named_parameters())[name].data = originals[name]


if __name__ == "__main__":
    main()
