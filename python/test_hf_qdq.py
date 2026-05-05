"""Run Gemma-4 inference with QDQ-reconstructed weights for ground-truth comparison.

Loads the HF Gemma4ForConditionalGeneration model, patches every quantized
layer with dequantized fp16 from the cactus v3 weights directory, then runs
a short prompt and prints token IDs to stdout (one per line, matching
test_fast output format).

Usage:
    python test_hf_qdq.py \
        --weights /path/to/gemma-4-e2b-it-tqh-prod-v3 \
        --hf_model google/gemma-4-2b-it \
        [--n_tokens 10]

    # or if you have a local HF checkout:
    python test_hf_qdq.py --weights ... --hf_model /local/hf/gemma-4-2b-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How are you?"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                        help="cactus v3 weights directory")
    parser.add_argument("--hf_model", type=str, required=True,
                        help="HF model id or local path with config.json / tokenizer")
    parser.add_argument("--n_tokens", type=int, default=10)
    parser.add_argument("--qdq_safetensors", type=Path, default=None,
                        help="pre-built QDQ safetensors (skip re-dequantizing if given)")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoProcessor
        from safetensors.torch import load_file, save_file
    except ImportError:
        sys.exit("pip install transformers torch safetensors")

    # --- build or load QDQ state dict ---
    if args.qdq_safetensors and args.qdq_safetensors.exists():
        print(f"Loading pre-built QDQ from {args.qdq_safetensors}", file=sys.stderr)
        qdq_state = load_file(str(args.qdq_safetensors))
    else:
        print("Dequantizing weights (this takes ~2 min for full model)...", file=sys.stderr)
        sys.path.insert(0, str(Path(__file__).parent))
        import qdq_weights as q

        qdq_state = {}
        for wf in sorted(args.weights.glob("*.weights")):
            stem = wf.stem
            hf = q.hf_name(stem)
            if hf is None:
                continue
            arr = q.dequant_file(wf, hf_scale=hf_scale)
            if arr is None:
                continue
            hf_scale = q.cactus_to_hf_scale(wf.stem)
            print(f"  {wf.name} → {hf} (scale={hf_scale})", file=sys.stderr)
            qdq_state[hf] = torch.from_numpy(arr)

        if args.qdq_safetensors:
            args.qdq_safetensors.parent.mkdir(parents=True, exist_ok=True)
            save_file(qdq_state, str(args.qdq_safetensors))
            print(f"Saved QDQ to {args.qdq_safetensors}", file=sys.stderr)

    # Patch weights to include .weight suffix that HF expects
    # e.g. model.language_model.layers.0.self_attn.q_proj → .weight
    qdq_patched = {}
    for k, v in qdq_state.items():
        qdq_patched[k + ".weight"] = v
    qdq_state = qdq_patched

    # --- load model ---
    print(f"Loading model from {args.hf_model}...", file=sys.stderr)
    try:
        from transformers import Gemma4ForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForCausalLM as Gemma4ForConditionalGeneration

    config_path = args.weights  # has config.json
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    # patch quantized layers
    missing, unexpected = model.load_state_dict(qdq_state, strict=False)
    patched = len(qdq_state) - len(unexpected)
    print(f"Patched {patched} tensors ({len(missing)} model keys not covered, "
          f"{len(unexpected)} QDQ keys not in model)", file=sys.stderr)

    model.eval()

    # --- tokenize ---
    try:
        proc = AutoProcessor.from_pretrained(args.hf_model)
        tokenizer = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    try:
        input_text = tokenizer.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
    except Exception:
        input_text = "\n".join(f"{m['role']}: {m['content']}" for m in PROMPT) + "\nassistant: "

    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Prompt tokens: {input_ids.shape[1]}", file=sys.stderr)

    # --- generate (manual loop to capture logits) ---
    DUMP_STEPS = {0, 1, 2, 3}  # dump top-20 logits at these steps
    past_key_values = None
    cur_ids = input_ids.clone()

    with torch.no_grad():
        for step in range(args.n_tokens):
            out = model(input_ids=cur_ids if past_key_values is None else cur_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True)
            logits = out.logits[0, -1, :]  # [vocab]
            past_key_values = out.past_key_values

            tok_id = int(logits.argmax())

            if step in DUMP_STEPS:
                top = logits.topk(20)
                print(f"[step {step} logits top-20]", file=sys.stderr)
                for rank, (val, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
                    txt = tokenizer.decode([idx])
                    print(f"  [{rank}] tok={idx} val={val:.4f} text={txt!r}", file=sys.stderr)

            text = tokenizer.decode([tok_id])
            print(f"[step {step}] sampled_tok={tok_id} (text={text!r})", file=sys.stderr)
            print(tok_id)

            cur_ids = torch.cat([cur_ids, torch.tensor([[tok_id]])], dim=1)
            stop_tokens = [tokenizer.eos_token_id,
                           tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                           tokenizer.convert_tokens_to_ids("<|im_end|>")]
            if tok_id in stop_tokens:
                break


if __name__ == "__main__":
    main()
