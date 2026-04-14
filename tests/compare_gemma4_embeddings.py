#!/usr/bin/env python3
"""Compare Gemma 4 e2b-it logits between FP32 and BF16, focusing on punctuation."""

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"
TASK_TEXT = "Transcribe the audio."

PUNCT_TOKENS = {
    236761: '.',
    236764: ',',
    236789: "'",
}


class LogitCapture:
    """Captures logits and punctuation token ranks at each generation step."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.steps = []

    def __call__(self, input_ids, scores):
        logits = scores[0]  # [vocab_size]
        top_token = torch.argmax(logits).item()
        top_text = self.tokenizer.decode([top_token])

        # Get top-5
        top5_vals, top5_idx = torch.topk(logits.float(), 5)

        step_info = {
            'top_token': top_token,
            'top_text': top_text,
            'top5': [(top5_idx[i].item(), top5_vals[i].item()) for i in range(5)],
            'punct': {}
        }

        for tid, char in PUNCT_TOKENS.items():
            if tid < logits.shape[0]:
                val = logits[tid].float().item()
                rank = (logits.float() > logits[tid].float()).sum().item() + 1
                step_info['punct'][char] = {'logit': val, 'rank': rank, 'id': tid}

        self.steps.append(step_info)
        return scores


def run_model(dtype_name, dtype, processor, audio_data, sr):
    print(f"\n{'='*60}")
    print(f"Loading {dtype_name} model...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": TASK_TEXT},
        {"type": "audio", "audio": audio_data, "sample_rate": sr},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    )
    prompt_len = inputs["input_ids"].shape[1]

    capture = LogitCapture(processor.tokenizer)

    print(f"Generating ({dtype_name})...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            logits_processor=[capture],
        )

    generated_ids = output_ids[0][prompt_len:]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens = [t.item() for t in generated_ids]
    print(f"{dtype_name} output: {text}")

    del model
    return text, tokens, capture.steps


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    audio_data, sr = sf.read(AUDIO_PATH)
    print(f"Audio: {len(audio_data)} samples, {sr} Hz")

    fp32_text, fp32_tokens, fp32_steps = run_model("FP32", torch.float32, processor, audio_data, sr)
    bf16_text, bf16_tokens, bf16_steps = run_model("BF16", torch.bfloat16, processor, audio_data, sr)

    print(f"\n{'='*60}")
    print(f"STEP-BY-STEP COMPARISON:")
    print(f"{'='*60}")

    max_steps = max(len(fp32_steps), len(bf16_steps))
    for i in range(min(max_steps, 60)):
        fp32_s = fp32_steps[i] if i < len(fp32_steps) else None
        bf16_s = bf16_steps[i] if i < len(bf16_steps) else None

        fp32_tok = fp32_s['top_text'] if fp32_s else '---'
        bf16_tok = bf16_s['top_text'] if bf16_s else '---'
        match = "OK" if (fp32_s and bf16_s and fp32_s['top_token'] == bf16_s['top_token']) else "DIFF"

        print(f"\nStep {i:3d} [{match}]: FP32='{fp32_tok}' vs BF16='{bf16_tok}'")

        if fp32_s:
            for char, info in fp32_s['punct'].items():
                bf16_info = bf16_s['punct'].get(char) if bf16_s else None
                bf16_rank = bf16_info['rank'] if bf16_info else '?'
                bf16_logit = f"{bf16_info['logit']:.2f}" if bf16_info else '?'
                print(f"  '{char}': FP32 rank={info['rank']:5d} logit={info['logit']:8.2f} | BF16 rank={bf16_rank} logit={bf16_logit}")

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"FP32: {fp32_text}")
    print(f"BF16: {bf16_text}")

    diverge_at = None
    for i, (a, b) in enumerate(zip(fp32_tokens, bf16_tokens)):
        if a != b:
            diverge_at = i
            print(f"First divergence at step {i}: FP32={a} '{processor.tokenizer.decode([a])}' vs BF16={b} '{processor.tokenizer.decode([b])}'")
            break
    if diverge_at is None:
        print("No divergence in overlapping tokens!")


if __name__ == "__main__":
    main()
