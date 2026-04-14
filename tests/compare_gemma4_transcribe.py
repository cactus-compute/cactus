#!/usr/bin/env python3
"""Compare Gemma 4 e2b-it transcription output between HuggingFace and Cactus."""

import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForImageTextToText

AUDIO_PATH = "tests/assets/test.wav"
MODEL_ID = "google/gemma-4-e2b-it"

# Match the exact prompt cactus uses (from cactus_transcribe.cpp line 195-196)
TASK_TEXT = "Transcribe the audio."

def main():
    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
    )

    print(f"Loading audio: {AUDIO_PATH}")
    audio_data, sr = sf.read(AUDIO_PATH)
    print(f"Audio: {len(audio_data)} samples, {sr} Hz, {len(audio_data)/sr:.2f}s")

    # Build the chat messages matching cactus format:
    # <bos><|turn>user\nTranscribe the audio.<|audio>[tokens]<audio|><turn|>\n<|turn>model\n
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TASK_TEXT},
                {"type": "audio", "audio": audio_data, "sample_rate": sr},
            ],
        }
    ]

    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    # Print the tokenized prompt for comparison
    input_ids = inputs["input_ids"][0]
    print(f"\nPrompt tokens ({len(input_ids)} total):")
    decoded_prompt = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
    # Show first/last parts (middle is audio tokens)
    if len(decoded_prompt) > 500:
        print(f"  Start: {decoded_prompt[:200]}")
        print(f"  ...({len(decoded_prompt)} chars)...")
        print(f"  End: {decoded_prompt[-200:]}")
    else:
        print(f"  {decoded_prompt}")

    print("\nGenerating transcription...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,  # greedy for reproducibility
        )

    # Decode only the generated tokens (skip the prompt)
    generated_ids = output_ids[0][len(input_ids):]
    transcription = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"HuggingFace output:")
    print(f"  {transcription}")
    print(f"{'='*60}")
    print(f"\nCactus output (from user):")
    print(f"  Hello hello hello just um quickly testing out creating a wave file through voice memos um the goal is to use this wave file to test out whisper hopefully this will transcribe properly that's all i can hope for all right here we go")
    print(f"{'='*60}")

    # Also print token-by-token for debugging
    print(f"\nGenerated tokens ({len(generated_ids)}):")
    for i, tid in enumerate(generated_ids[:50]):
        piece = processor.tokenizer.decode([tid])
        print(f"  [{i:3d}] token={tid:6d}  '{piece}'")
    if len(generated_ids) > 50:
        print(f"  ... ({len(generated_ids) - 50} more tokens)")


if __name__ == "__main__":
    main()
