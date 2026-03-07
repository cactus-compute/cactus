"""
Compare Gemma 3n block-composition captures between Cactus FP16 and HF.

Expected Cactus captures:
  CACTUS_CAPTURE_DIR=/tmp/cactus_attn_caps \
  CACTUS_CAPTURE_PREFILL_ONLY=1 \
  CACTUS_DISABLE_SPLIT_PREFILL=1 \
  CACTUS_CAPTURE_STDOUT=1 \
  CACTUS_THREADS=1 \
  echo "who are you" | tests/build/chat weights/gemma-3n-e2b-it-fp16
"""
import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np

CAPTURE_DIR = "/tmp/cactus_attn_caps"
HF_DIR = "/tmp/hf_block_caps"
PROMPT_WITH_BOS = "<bos><start_of_turn>user\nwho are you<end_of_turn>\n<start_of_turn>model\n"

BLOCK_SUFFIXES = [
    "block_normed",
    "block_attn_raw",
    "block_post_attn_norm",
    "block_laurel",
    "block_combined_pre_rsqrt2",
    "block_post_attn",
    "block_pre_ffn_norm",
    "block_mlp_gate_raw",
    "block_mlp_up",
    "block_mlp_gate_topk",
    "block_mlp_activated",
    "block_mlp_raw",
    "block_post_ffn_norm",
    "block_output_pre_altup",
    "post_correct_stream0",
    "post_pli_stream0",
]


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_cactus(name: str, capture_dir: str) -> np.ndarray | None:
    path = os.path.join(capture_dir, f"{name}.bin")
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.float16).astype(np.float32)


def load_hf(name: str, hf_dir: str) -> np.ndarray | None:
    path = os.path.join(hf_dir, f"{name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def infer_hf_offset(cactus_block_normed: np.ndarray | None, hf_block_normed: np.ndarray | None) -> Tuple[int, int | None]:
    if cactus_block_normed is None or hf_block_normed is None:
        return 0, None
    if hf_block_normed.ndim < 3:
        return 0, None

    hidden_dim = hf_block_normed.shape[-1]
    cactus_seq = cactus_block_normed.size // hidden_dim
    hf_seq = hf_block_normed.shape[1]
    if hidden_dim == 0 or cactus_seq == 0 or hf_seq < cactus_seq:
        return 0, None

    cactus_first = cactus_block_normed[:hidden_dim]
    max_offset = min(2, hf_seq - cactus_seq)
    best_offset = 0
    best_cos = -1.0
    for offset in range(max_offset + 1):
        hf_first = hf_block_normed[0, offset].reshape(-1)[:hidden_dim]
        score = cos_sim(cactus_first, hf_first)
        if score > best_cos:
            best_cos = score
            best_offset = offset
    return best_offset, cactus_seq


def generate_hf_captures(layers: List[int], hf_dir: str, dtype_name: str) -> None:
    import torch
    import types
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    if dtype_name == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_name == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype {dtype_name}")

    os.makedirs(hf_dir, exist_ok=True)
    for filename in os.listdir(hf_dir):
        if filename.endswith(".npy"):
            os.remove(os.path.join(hf_dir, filename))

    print(f"Loading HF model ({dtype_name})...")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it", torch_dtype=torch_dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
    input_ids = tokenizer.encode(PROMPT_WITH_BOS, add_special_tokens=False)
    lm = model.model.language_model

    ops: Dict[str, np.ndarray] = {}

    def store(name: str, tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        ops[name] = tensor.detach().float().cpu().numpy()

    originals = {}

    def make_patched_forward(layer_idx: int):
        pfx = f"L{layer_idx}_"

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor = None,
            per_layer_input: torch.Tensor = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            cache_position: torch.LongTensor | None = None,
            **kwargs,
        ):
            predictions = self.altup.predict(hidden_states)
            active_prediction = predictions[self.config.altup_active_idx]

            active_prediction_normed = self.input_layernorm(active_prediction)
            store(pfx + "block_normed", active_prediction_normed)

            laurel_output = self.laurel(active_prediction_normed)
            store(pfx + "block_laurel", laurel_output)

            attn, _ = self.self_attn(
                hidden_states=active_prediction_normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            store(pfx + "block_attn_raw", attn)

            attn = self.post_attention_layernorm(attn)
            store(pfx + "block_post_attn_norm", attn)

            combined = active_prediction + attn + laurel_output
            store(pfx + "block_combined_pre_rsqrt2", combined)

            attn_laurel = combined / math.sqrt(2.0)
            store(pfx + "block_post_attn", attn_laurel)

            attn_norm = self.pre_feedforward_layernorm(attn_laurel)
            store(pfx + "block_pre_ffn_norm", attn_norm)

            gate = self.mlp.gate_proj(attn_norm)
            store(pfx + "block_mlp_gate_raw", gate)

            up = self.mlp.up_proj(attn_norm)
            store(pfx + "block_mlp_up", up)

            if self.mlp.activation_sparsity > 0.0:
                gate = self.mlp._gaussian_topk(gate)
            store(pfx + "block_mlp_gate_topk", gate)

            activated = self.mlp.act_fn(gate)
            store(pfx + "block_mlp_activated", activated)

            attn_ffw = self.mlp.down_proj(activated * up)
            store(pfx + "block_mlp_raw", attn_ffw)

            attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
            store(pfx + "block_post_ffn_norm", attn_ffw_norm)

            attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
            store(pfx + "block_output_pre_altup", attn_ffw_laurel_gated)

            corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)
            stream0 = corrected_predictions[self.config.altup_active_idx]
            store(pfx + "post_correct_stream0", stream0)

            first_prediction = stream0.clone()
            if self.config.altup_correct_scale:
                first_prediction = self.altup.scale_corrected_output(first_prediction)

            first_prediction = self.per_layer_input_gate(first_prediction)
            first_prediction = self.act_fn(first_prediction)
            first_prediction = torch.multiply(first_prediction, per_layer_input)
            first_prediction = self.per_layer_projection(first_prediction)
            first_prediction = self.post_per_layer_input_norm(first_prediction)
            corrected_predictions[1:] += first_prediction

            store(pfx + "post_pli_stream0", corrected_predictions[self.config.altup_active_idx])
            return corrected_predictions

        return patched_forward

    try:
        for layer_idx in layers:
            layer = lm.layers[layer_idx]
            originals[layer_idx] = layer.forward
            layer.forward = types.MethodType(make_patched_forward(layer_idx), layer)

        with torch.no_grad():
            model(torch.tensor([input_ids]), use_cache=True)
    finally:
        for layer_idx, original in originals.items():
            lm.layers[layer_idx].forward = original

    for name, arr in ops.items():
        np.save(os.path.join(hf_dir, f"{name}.npy"), arr)
    print(f"Saved {len(ops)} HF captures to {hf_dir}")


def compare_layer(layer: int, capture_dir: str, hf_dir: str) -> Dict[str, float]:
    pfx = f"L{layer}_"
    align_c = load_cactus(pfx + "block_normed", capture_dir)
    align_h = load_hf(pfx + "block_normed", hf_dir)
    hf_offset, hf_target_seq = infer_hf_offset(align_c, align_h)

    print(f"\nLayer {layer} (HF offset={hf_offset}, target_seq={hf_target_seq})")
    print(f"{'Node':<34} {'Cos':>10}")
    print("-" * 46)

    results: Dict[str, float] = {}
    for suffix in BLOCK_SUFFIXES:
        c = load_cactus(pfx + suffix, capture_dir)
        h = load_hf(pfx + suffix, hf_dir)

        if c is None or h is None:
            print(f"{suffix:<34} {'MISSING':>10}")
            continue

        if hf_target_seq is not None and h.ndim >= 3 and h.shape[1] >= hf_offset + hf_target_seq:
            h = h[:, hf_offset:hf_offset + hf_target_seq, ...]

        cs = cos_sim(c, h)
        results[suffix] = cs
        print(f"{suffix:<34} {cs:>10.6f}")

    return results


def compare_hf_precisions(layers: List[int], hf_bf16_dir: str, hf_fp32_dir: str) -> None:
    print("\nHF bf16 vs HF fp32")
    print(f"{'Layer':<8} {'Node':<34} {'Cos':>10}")
    print("-" * 58)
    for layer in layers:
        pfx = f"L{layer}_"
        for suffix in BLOCK_SUFFIXES:
            b = load_hf(pfx + suffix, hf_bf16_dir)
            f = load_hf(pfx + suffix, hf_fp32_dir)
            if b is None or f is None:
                continue
            cs = cos_sim(b, f)
            print(f"{layer:<8} {suffix:<34} {cs:>10.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gemma 3n block captures")
    parser.add_argument("--layers", type=int, nargs="+", default=[4], help="Layers to compare")
    parser.add_argument("--capture-dir", default=CAPTURE_DIR, help="Cactus capture dir")
    parser.add_argument("--hf-dir", default=HF_DIR, help="HF capture dir")
    parser.add_argument("--gen-hf", action="store_true", help="Generate HF captures before comparing")
    parser.add_argument("--hf-dtype", choices=["bf16", "fp32"], default="bf16", help="HF dtype for --gen-hf")
    parser.add_argument("--hf-precision-check", action="store_true", help="Run bf16/fp32 HF precision-only comparison")
    parser.add_argument("--hf-bf16-dir", default="/tmp/hf_block_caps_bf16", help="HF bf16 capture dir")
    parser.add_argument("--hf-fp32-dir", default="/tmp/hf_block_caps_fp32", help="HF fp32 capture dir")
    args = parser.parse_args()

    layers = sorted(set(args.layers))

    if args.hf_precision_check:
        generate_hf_captures(layers, args.hf_bf16_dir, "bf16")
        generate_hf_captures(layers, args.hf_fp32_dir, "fp32")
        compare_hf_precisions(layers, args.hf_bf16_dir, args.hf_fp32_dir)
        return

    if args.gen_hf:
        generate_hf_captures(layers, args.hf_dir, args.hf_dtype)

    for layer in layers:
        compare_layer(layer, args.capture_dir, args.hf_dir)


if __name__ == "__main__":
    main()
