"""
Compare Cactus PREFILL captures with HF bf16 prefill for Layer 0.

This is the definitive comparison: same prompt, same 11 tokens, prefill only.
Compares every attention sub-operation to pinpoint where divergence occurs.
"""
import os, sys
import numpy as np

CAPTURE_DIR = "/tmp/cactus_attn_caps"
HF_DIR = "/tmp/hf_attn_caps"

# Cactus chat formatting prepends BOS internally for this prompt.
# We run HF with explicit BOS and then auto-select the best sequence offset.
PROMPT_WITH_BOS = "<bos><start_of_turn>user\nwho are you<end_of_turn>\n<start_of_turn>model\n"
PROMPT_NO_BOS = "<start_of_turn>user\nwho are you<end_of_turn>\n<start_of_turn>model\n"


def build_fallback_attention_mask(seq_len, device, dtype, sliding_window=None):
    import torch
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    allowed = k_pos <= q_pos
    if sliding_window is not None:
        allowed = allowed & ((q_pos - k_pos) < sliding_window)
    min_val = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype)
    mask = mask.masked_fill(allowed, 0)
    return mask.unsqueeze(0).unsqueeze(0)


def cos_sim(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def load_cactus(name):
    path = os.path.join(CAPTURE_DIR, f"{name}.bin")
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.float16).astype(np.float32)


def load_hf(name):
    path = os.path.join(HF_DIR, f"{name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def infer_hf_offset(cactus_block_normed, hf_block_normed):
    if cactus_block_normed is None or hf_block_normed is None:
        return 0, None
    if hf_block_normed.ndim < 3:
        return 0, None
    hidden_dim = hf_block_normed.shape[-1]
    if hidden_dim == 0:
        return 0, None

    cactus_seq = cactus_block_normed.size // hidden_dim
    hf_seq = hf_block_normed.shape[1]
    if cactus_seq == 0 or hf_seq < cactus_seq:
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


def generate_hf_captures(layer=0):
    """Run HF model and save all attention intermediates."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb, repeat_kv

    os.makedirs(HF_DIR, exist_ok=True)

    print("Loading HF model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it", dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")

    # Use BOS prompt - Cactus processes BOS first, then 11 tokens
    # We run HF with all 12 tokens and will compare tokens 1-11
    input_ids = tokenizer.encode(PROMPT_WITH_BOS, add_special_tokens=False)
    print(f"Tokens ({len(input_ids)}): {input_ids}")
    seq_len = len(input_ids)

    lm = model.model.language_model
    L = lm.layers[layer]
    attn_mod = L.self_attn
    ops = {}
    pfx = f"L{layer}_"

    # Monkey-patch attention forward
    def patched_forward(hidden_states, position_embeddings=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_mod.config.head_dim)
        cos, sin = position_embeddings

        q_proj = attn_mod.q_proj(hidden_states)
        ops[pfx + "attn_q_proj"] = q_proj.detach().float().numpy()

        query_states = q_proj.view(hidden_shape)
        query_states = attn_mod.q_norm(query_states)
        ops[pfx + "attn_q_norm"] = query_states.detach().float().numpy()

        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        ops[pfx + "attn_q_rope"] = query_states.detach().float().numpy()
        query_states = query_states.transpose(1, 2)

        k_proj = attn_mod.k_proj(hidden_states)
        ops[pfx + "attn_k_proj"] = k_proj.detach().float().numpy()

        key_states = k_proj.view(hidden_shape)
        key_states = attn_mod.k_norm(key_states)
        ops[pfx + "attn_k_norm"] = key_states.detach().float().numpy()

        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        ops[pfx + "attn_k_rope"] = key_states.detach().float().numpy()
        key_states = key_states.transpose(1, 2)

        v_proj = attn_mod.v_proj(hidden_states)
        ops[pfx + "attn_v_proj"] = v_proj.detach().float().numpy()

        value_states = v_proj.view(hidden_shape)
        value_states = attn_mod.v_norm(value_states)
        ops[pfx + "attn_v_norm"] = value_states.detach().float().numpy()
        value_states = value_states.transpose(1, 2)

        key_states_exp = repeat_kv(key_states, attn_mod.num_key_value_groups)
        value_states_exp = repeat_kv(value_states, attn_mod.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_exp.transpose(2, 3)) * attn_mod.scaling
        attention_mask = kwargs.get('attention_mask')
        if attention_mask is None:
            attention_mask = build_fallback_attention_mask(
                query_states.shape[2],
                query_states.device,
                query_states.dtype,
                attn_mod.sliding_window,
            )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights_sm = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights_sm, value_states_exp)
        ops[pfx + "attn_output_pre_oproj"] = attn_output.transpose(1, 2).contiguous().detach().float().numpy()

        attn_output_flat = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
        o_proj = attn_mod.o_proj(attn_output_flat)
        ops[pfx + "attn_o_proj"] = o_proj.detach().float().numpy()

        return o_proj, None

    import types
    attn_mod.forward = types.MethodType(lambda self, *a, **kw: patched_forward(*a, **kw), attn_mod)

    L.input_layernorm.register_forward_hook(
        lambda m, i, o: ops.update({pfx + "block_normed": o.detach().float().numpy()})
    )

    with torch.no_grad():
        model(torch.tensor([input_ids]))

    for name, arr in ops.items():
        np.save(os.path.join(HF_DIR, f"{name}.npy"), arr)
        print(f"  Saved {name}: shape={arr.shape}")

    return ops


def compare(layer=0):
    pfx = f"L{layer}_"

    # Comparison pairs: (cactus_name, hf_name, description, how_to_flatten_hf)
    comparisons = [
        ("block_normed", "block_normed", "Input LayerNorm"),
        ("attn_q_proj", "attn_q_proj", "Q Projection"),
        ("attn_q_norm", "attn_q_norm", "Q RMSNorm (per-head)"),
        ("attn_q_rope", "attn_q_rope", "Q after RoPE"),
        ("attn_k_proj", "attn_k_proj", "K Projection"),
        ("attn_k_norm", "attn_k_norm", "K RMSNorm (per-head)"),
        ("attn_k_rope", "attn_k_rope", "K after RoPE"),
        ("attn_v_proj", "attn_v_proj", "V Projection"),
        ("attn_v_norm", "attn_v_norm", "V RMSNorm (no weight)"),
        ("attn_output_pre_oproj", "attn_output_pre_oproj", "Attention*V (pre o_proj)"),
        ("attn_o_proj", "attn_o_proj", "Output Projection (o_proj)"),
        ("block_attn_raw", "attn_o_proj", "Full Attn Output"),
    ]

    print(f"\n{'='*100}")
    print(f"PREFILL COMPARISON: Layer {layer} (Cactus FP16 vs HF bf16)")
    print(f"{'='*100}")
    print(f"\n{'Sub-operation':<35} {'Cos Sim':>10} {'C norm':>10} {'H norm':>10} {'Ratio':>10} {'C elems':>10} {'H elems':>10}")
    print("-" * 100)

    align_c = load_cactus(pfx + "block_normed")
    align_h = load_hf(pfx + "block_normed")
    hf_offset, hf_target_seq = infer_hf_offset(align_c, align_h)
    if hf_target_seq is not None:
        first_cos = cos_sim(align_c[:align_h.shape[-1]], align_h[0, hf_offset][:align_h.shape[-1]])
        print(f"Alignment: HF offset={hf_offset}, target_seq={hf_target_seq}, first-token cos={first_cos:.6f}")

    prev_cos = None
    for c_name, h_name, desc in comparisons:
        c_full = pfx + c_name
        h_full = pfx + h_name

        c_val = load_cactus(c_full)
        h_val = load_hf(h_full)

        if c_val is None:
            print(f"{desc:<35} {'N/A-C':>10}")
            continue
        if h_val is None:
            print(f"{desc:<35} {'N/A-H':>10}")
            continue

        if hf_target_seq is not None and h_val.ndim >= 3 and h_val.shape[1] >= hf_offset + hf_target_seq:
            h_val = h_val[:, hf_offset:hf_offset + hf_target_seq, ...]

        c_flat = c_val.flatten()
        h_flat = h_val.flatten()

        n = min(len(c_flat), len(h_flat))
        if n == 0:
            print(f"{desc:<35} {'EMPTY':>10}")
            continue

        cs = cos_sim(c_flat[:n], h_flat[:n])
        c_norm = np.linalg.norm(c_flat[:n].astype(np.float64))
        h_norm = np.linalg.norm(h_flat[:n].astype(np.float64))
        ratio = c_norm / h_norm if h_norm > 1e-12 else float('inf')

        drop_str = ""
        if prev_cos is not None:
            drop = prev_cos - cs
            if drop > 0.005:
                drop_str = f" <<< DROP {drop:.4f}"

        print(f"{desc:<35} {cs:>10.6f} {c_norm:>10.2f} {h_norm:>10.2f} {ratio:>10.4f} {len(c_flat):>10} {len(h_flat):>10}{drop_str}")
        prev_cos = cs

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--hf-only", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    if args.compare_only:
        compare(args.layer)
        return

    if not args.hf_only:
        print("Cactus captures expected in", CAPTURE_DIR)
        print("Run with CACTUS_CAPTURE_PREFILL_ONLY=1 first")

    generate_hf_captures(args.layer)
    compare(args.layer)


if __name__ == "__main__":
    main()
