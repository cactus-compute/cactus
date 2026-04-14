#!/usr/bin/env python3
"""Test whether per-row INT8 activation quantization causes punctuation loss."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"
GROUP_SIZE = 32


def quantize_weight_int4(weight):
    """Simulate INT4 group-wise weight quantization."""
    rows, cols = weight.shape
    q = torch.zeros_like(weight)
    for g in range(0, cols, GROUP_SIZE):
        end = min(g + GROUP_SIZE, cols)
        group = weight[:, g:end].float()
        max_abs = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        scale = max_abs / 7.0
        quantized = torch.clamp(torch.round(group / scale), -8, 7)
        q[:, g:end] = (quantized * scale).to(weight.dtype)
    return q


def quantize_activation_int8_per_row(x):
    """Simulate cactus per-row INT8 activation quantization."""
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = max_abs / 127.0
    quantized = torch.clamp(torch.round(x / scale), -128, 127)
    return quantized * scale


class ActivationQuantHook:
    """Hook that quantizes activations to INT8 before every Linear layer."""
    def __init__(self):
        self.handles = []

    def hook_fn(self, module, inputs):
        x = inputs[0]
        return (quantize_activation_int8_per_row(x),) + inputs[1:]

    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                h = module.register_forward_pre_hook(self.hook_fn)
                self.handles.append(h)
        print(f"Registered {len(self.handles)} activation quantization hooks")

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


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


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    audio_data, sr = sf.read(AUDIO_PATH)

    print("Loading FP16 model...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float16)

    # Test 1: FP16 baseline
    print("\n=== Test 1: FP16 baseline ===")
    result = run_transcribe(model, processor, audio_data, sr)
    print(f"  {result}")

    # Test 2: INT4 weights only (no activation quantization)
    print("\n=== Test 2: INT4 weights only ===")
    originals = {}
    for name, param in model.named_parameters():
        if param.ndim == 2 and 'norm' not in name and 'embed' not in name.split('.')[-1]:
            originals[name] = param.data.clone()
            param.data = quantize_weight_int4(param.data)
    result = run_transcribe(model, processor, audio_data, sr)
    print(f"  {result}")
    for name in originals:
        dict(model.named_parameters())[name].data = originals[name]

    # Test 3: Activation quantization only (no weight quantization)
    print("\n=== Test 3: Activation INT8 quantization only (FP16 weights) ===")
    hook = ActivationQuantHook()
    hook.register(model)
    result = run_transcribe(model, processor, audio_data, sr)
    print(f"  {result}")
    hook.remove()

    # Test 4: Both INT4 weights AND activation quantization (= what cactus does)
    print("\n=== Test 4: INT4 weights + Activation INT8 (= cactus equivalent) ===")
    for name, param in model.named_parameters():
        if param.ndim == 2 and 'norm' not in name and 'embed' not in name.split('.')[-1]:
            originals[name] = param.data.clone()
            param.data = quantize_weight_int4(param.data)
    hook = ActivationQuantHook()
    hook.register(model)
    result = run_transcribe(model, processor, audio_data, sr)
    print(f"  {result}")
    hook.remove()
    for name in originals:
        dict(model.named_parameters())[name].data = originals[name]


if __name__ == "__main__":
    main()
