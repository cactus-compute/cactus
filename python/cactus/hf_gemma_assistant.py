#!/usr/bin/env python3
import argparse

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, TextStreamer


TARGET_MODEL_ID = "google/gemma-4-E2B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-E2B-it-assistant"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 E2B IT with its Hugging Face MTP assistant."
    )
    parser.add_argument("prompt", help="Single user prompt to generate one response for.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not hasattr(transformers, "Gemma4AssistantForCausalLM"):
        raise RuntimeError(
            "Gemma 4 Assistant requires transformers with gemma4_assistant support. "
            "Install with: pip install -U 'transformers>=5.8.0'"
        )

    processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID, padding_side="left")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    assistant_model = AutoModelForCausalLM.from_pretrained(
        ASSISTANT_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [{"role": "user", "content": args.prompt}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(target_model.device)
    streamer = TextStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    target_model.generate(
        **inputs,
        assistant_model=assistant_model,
        max_new_tokens=256,
        do_sample=False,
        streamer=streamer,
    )


if __name__ == "__main__":
    main()
