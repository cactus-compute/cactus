"""Numerical-parity oracle: transformers reference vs cactus engine, greedy.

Modes:
  ref <src_checkpoint_dir> <out.json>      transformers greedy token ids
  engine <bundle_dir> <out.json>           cactus engine greedy token ids (FFI, temp 0.0)
  compare <ref.json> <engine.json>         agreement stats; exits nonzero below floor

The 4-layer truncated model produces low-quality text by construction; the
metric is cross-stack agreement, not quality. The CQ4 bundle's agreement vs
reference is the calibrated floor the repack path must meet or beat.
"""

import json
import sys

MAX_NEW_TOKENS = 64
PROMPTS = [
    "What is 17 multiplied by 23?",
    "Name the capital of Australia.",
    "Write a python function that reverses a string.",
    "Explain photosynthesis in one sentence.",
    "List three prime numbers greater than 100.",
    "Translate 'good morning' to French.",
    "What year did the Apollo 11 mission land on the moon?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]
SYSTEM = "You are a helpful assistant"


def messages_for(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def run_ref(src_dir, out_path):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = json.loads(open(f"{src_dir}/config.json").read())
    text_cfg = Qwen3_5TextConfig(**config["text_config"])
    model = Qwen3_5ForCausalLM(text_cfg)
    state = {}
    with safe_open(f"{src_dir}/model.safetensors", framework="pt") as f:
        for name in f.keys():
            new = name.replace("model.language_model.", "model.")
            state[new] = f.get_tensor(name)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not [m for m in missing if not m.startswith(("mtp.", "model.visual."))], missing
    model = model.to(torch.bfloat16).eval()

    tok = AutoTokenizer.from_pretrained(src_dir)
    results = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = tok.apply_chat_template(
                messages_for(prompt), add_generation_prompt=True, return_tensors="pt", return_dict=False
            )
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=text_cfg.eos_token_id,
            )
            completion = out[0, ids.shape[1]:].tolist()
            results[str(i)] = {
                "prompt_ids": ids[0].tolist(),
                "completion_ids": completion,
                "text": tok.decode(completion, skip_special_tokens=True),
            }
            print(f"ref[{i}] {len(completion)} tokens")
    json.dump(results, open(out_path, "w"))


def run_engine(bundle_dir, out_path):
    from cactus.bindings.cactus import cactus_complete, cactus_destroy, cactus_init

    model = cactus_init(bundle_dir)
    results = {}
    try:
        for i, prompt in enumerate(PROMPTS):
            resp = cactus_complete(
                model,
                messages_for(prompt),
                options={"temperature": 0.0, "max_tokens": MAX_NEW_TOKENS, "auto_handoff": False},
            )
            text = resp.get("response", "")
            results[str(i)] = {"text": text, "decode_tokens": resp.get("decode_tokens", 0)}
            print(f"engine[{i}] {resp.get('decode_tokens', 0)} tokens: {text[:60]!r}")
    finally:
        cactus_destroy(model)
    json.dump(results, open(out_path, "w"))


def compare(ref_path, engine_path, floor=None):
    ref = json.load(open(ref_path))
    eng = json.load(open(engine_path))
    scores = []
    for key in sorted(ref, key=int):
        r, e = ref[key]["text"], eng[key]["text"]
        n = max(len(r), len(e)) or 1
        prefix = 0
        for a, b in zip(r, e):
            if a != b:
                break
            prefix += 1
        score = prefix / n
        scores.append(score)
        print(f"prompt {key}: text prefix {prefix}/{n} = {score:.3f}  ref={r[:40]!r} eng={e[:40]!r}")
    mean = sum(scores) / len(scores)
    print(f"MEAN_PREFIX_AGREEMENT {mean:.4f}")
    if floor is not None and mean < float(floor):
        print(f"FAIL: below floor {floor}")
        sys.exit(1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ref":
        run_ref(sys.argv[2], sys.argv[3])
    elif mode == "engine":
        run_engine(sys.argv[2], sys.argv[3])
    elif mode == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        sys.exit(f"unknown mode {mode}")
