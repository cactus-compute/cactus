"""
Generate matching profile files for Cactus and HuggingFace Gemma 3n Layer 0.

Cactus: Runs the model via chat binary with CACTUS_PROFILE_FILE, extracts the
        prefill execution (seq_len > 1), and annotates with debug node labels.

HF:     Runs the model with hooks on every submodule, formats output to match
        the Cactus profile format.

Usage:
    python tests/generate_profiles.py [--weights PATH] [--layer N] [--prompt TEXT]

Outputs:
    tests/cactus_L0_profile.txt
    tests/hf_L0_profile.txt
"""
import argparse
import os
import re
import signal
import subprocess
import time
from collections import OrderedDict

CACTUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_BIN = os.path.join(CACTUS_DIR, "tests/build/chat")
DEFAULT_WEIGHTS = os.path.join(CACTUS_DIR, "weights/gemma-3n-e2b-it")
OUTPUT_DIR = os.path.join(CACTUS_DIR, "tests")


def parse_shape_from_line(line):
    match = re.search(r"\[([0-9,\s]+)\]", line)
    if not match:
        return None
    dims = []
    for part in match.group(1).split(","):
        part = part.strip()
        if part:
            dims.append(int(part))
    return dims if dims else None


def extract_op_lines(profile_lines, start, end):
    op_lines = []
    for line in profile_lines[start + 3:end]:
        stripped = line.strip()
        if stripped and stripped[0].isupper() and not stripped.startswith("==="):
            op_lines.append(line)
    return op_lines


def pick_prefill_section(profile_lines):
    section_starts = [i for i, l in enumerate(profile_lines) if "=== Graph Execution Profile ===" in l]
    if not section_starts:
        raise RuntimeError("No graph execution sections found in profile")

    sections = []
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else len(profile_lines)
        op_lines = extract_op_lines(profile_lines, start, end)
        seq_len = None
        for op in op_lines:
            if op.startswith("EMBEDDING"):
                shape = parse_shape_from_line(op)
                if shape:
                    seq_len = shape[0]
                break
        sections.append((start, end, op_lines, seq_len))

    for section in sections:
        if section[3] is not None and section[3] > 1:
            return section
    return sections[0]


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


def generate_cactus_profile(weights_path, prompt, layer):
    profile_path = "/tmp/cactus_gen_profile.txt"
    if os.path.exists(profile_path):
        os.remove(profile_path)

    env = os.environ.copy()
    env["CACTUS_PROFILE_FILE"] = profile_path
    env["CACTUS_CAPTURE_STDOUT"] = "1"
    env["CACTUS_CAPTURE_PREVIEW_COUNT"] = "5"
    env["CACTUS_CAPTURE_PREFILL_ONLY"] = "1"
    env["CACTUS_DISABLE_SPLIT_PREFILL"] = "1"

    proc = subprocess.Popen(
        [CHAT_BIN, weights_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env
    )
    try:
        proc.stdin.write((prompt + "\n").encode())
        proc.stdin.flush()
        time.sleep(10)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

    # Parse debug node IDs from capture stdout
    capture_text = stdout.decode(errors="ignore")
    debug_nodes = {}
    pattern = re.compile(r"Layer\s+(\d+)\s+-\s+(.+?)\s+\(node\s+(\d+)\)")
    for line in capture_text.split("\n"):
        m = pattern.search(line)
        if not m:
            continue
        layer_idx = int(m.group(1))
        name = m.group(2)
        node_id = int(m.group(3))
        if layer_idx == layer or (layer_idx == 0 and name.startswith("preamble")):
            if node_id not in debug_nodes:
                debug_nodes[node_id] = name

    # Parse profile
    with open(profile_path) as f:
        profile_lines = f.readlines()

    start, end, op_lines, seq_len = pick_prefill_section(profile_lines)
    header_lines = profile_lines[start:start + 3]

    if not debug_nodes:
        raise RuntimeError("No debug nodes parsed from Cactus stdout capture")

    preamble_node = None
    for node_id, name in debug_nodes.items():
        if name == "preamble_embed_scaled":
            preamble_node = node_id
            break

    if preamble_node is not None:
        scalar_multiply_idx = None
        for idx, line in enumerate(op_lines):
            if line.startswith("SCALAR_MULTIPLY"):
                scalar_multiply_idx = idx
                break
        base = preamble_node - scalar_multiply_idx if scalar_multiply_idx is not None else min(debug_nodes.keys()) - 1
    else:
        base = min(debug_nodes.keys()) - 1

    layer_prefix = f"L{layer}_"
    layer_node_ids = [nid for nid, name in debug_nodes.items() if name.startswith(layer_prefix)]

    max_node = max(layer_node_ids) if layer_node_ids else max(debug_nodes.keys())
    max_op_idx = min(max_node - base + 10, len(op_lines))
    if max_op_idx <= 0:
        max_op_idx = len(op_lines)

    out_path = os.path.join(OUTPUT_DIR, f"cactus_L{layer}_profile.txt")
    with open(out_path, "w") as f:
        for h in header_lines:
            f.write(h)
        for i in range(max_op_idx):
            node_id = base + i
            if node_id in debug_nodes:
                f.write(f"  *** {debug_nodes[node_id]} (node {node_id}) ***\n")
            if i < len(op_lines):
                f.write(op_lines[i])

    print(f"Cactus profile: {out_path} ({max_op_idx} ops, {len(debug_nodes)} debug nodes, seq_len={seq_len})")
    return out_path, seq_len


def generate_hf_profile(prompt, layer, target_seq_len=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HF model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3n-E2B-it", dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Tokens ({len(input_ids)}): {input_ids}")

    lm = model.model.language_model
    ops = OrderedDict()

    def store(name, out):
        if isinstance(out, torch.Tensor):
            t = out
        elif isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
            t = out[0]
        else:
            return
        t = t.detach().float().cpu()
        if target_seq_len is not None and t.dim() >= 3 and t.shape[1] > target_seq_len:
            t = t[:, :target_seq_len, ...]
        ops[name] = t

    def hook(name):
        def fn(mod, inp, out):
            store(name, out)
        return fn

    # Preamble hooks
    lm.embed_tokens.register_forward_hook(hook("EMBEDDING"))
    lm.embed_tokens_per_layer.register_forward_hook(hook("EMBEDDING_PER_LAYER"))
    lm.per_layer_model_projection.register_forward_hook(hook("PER_LAYER_MODEL_PROJ"))
    lm.per_layer_projection_norm.register_forward_hook(hook("PER_LAYER_PROJ_NORM"))
    for i in range(3):
        lm.altup_projections[i].register_forward_hook(hook(f"ALTUP_PROJ_{i}"))

    # Layer hooks
    L = lm.layers[layer]
    pfx = f"L{layer}_"
    L.altup.router_norm.register_forward_hook(hook(pfx + "ALTUP_ROUTER_NORM"))
    L.altup.modality_router.register_forward_hook(hook(pfx + "ALTUP_ROUTER"))
    L.altup.prediction_coefs.register_forward_hook(hook(pfx + "ALTUP_PRED_COEFS"))
    L.input_layernorm.register_forward_hook(hook(pfx + "block_normed"))
    L.laurel.linear_left.register_forward_hook(hook(pfx + "LAUREL_LEFT"))
    L.laurel.linear_right.register_forward_hook(hook(pfx + "LAUREL_RIGHT"))
    L.laurel.post_laurel_norm.register_forward_hook(hook(pfx + "LAUREL_NORM"))
    L.laurel.register_forward_hook(hook(pfx + "LAUREL"))
    attn_mod = L.self_attn
    original_forward = attn_mod.forward

    def patched_forward(self, hidden_states, position_embeddings=None, **kwargs):
        from transformers.models.gemma3n.modeling_gemma3n import apply_rotary_pos_emb, repeat_kv

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)
        cos, sin = position_embeddings

        q_proj = self.q_proj(hidden_states)
        store(pfx + "attn_q_proj", q_proj)
        query_states = self.q_norm(q_proj.view(hidden_shape))
        store(pfx + "attn_q_norm", query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        store(pfx + "attn_q_rope", query_states)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and kwargs.get("past_key_values") is not None:
            key_states, value_states = kwargs["past_key_values"].shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
            store(pfx + "attn_k_rope", key_states.transpose(1, 2))
            store(pfx + "attn_v_norm", value_states.transpose(1, 2))
        else:
            k_proj = self.k_proj(hidden_states)
            store(pfx + "attn_k_proj", k_proj)
            key_states = self.k_norm(k_proj.view(hidden_shape))
            store(pfx + "attn_k_norm", key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            store(pfx + "attn_k_rope", key_states)
            key_states = key_states.transpose(1, 2)

            v_proj = self.v_proj(hidden_states)
            store(pfx + "attn_v_proj", v_proj)
            value_states = self.v_norm(v_proj.view(hidden_shape))
            store(pfx + "attn_v_norm", value_states)
            value_states = value_states.transpose(1, 2)

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": kwargs.get("cache_position"),
                "sliding_window": self.sliding_window,
            }
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        key_states_exp = repeat_kv(key_states, self.num_key_value_groups)
        value_states_exp = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_exp.transpose(2, 3)) * self.scaling
        store(pfx + "attn_scores_raw", attn_weights)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = build_fallback_attention_mask(
                query_states.shape[2],
                query_states.device,
                query_states.dtype,
                self.sliding_window,
            )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights_sm = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        store(pfx + "attn_weights_softmax", attn_weights_sm)
        attn_output = torch.matmul(attn_weights_sm, value_states_exp)
        attn_output_seq = attn_output.transpose(1, 2).contiguous()
        store(pfx + "attn_output_pre_oproj", attn_output_seq)

        attn_output_flat = attn_output_seq.reshape(*input_shape, -1).contiguous()
        o_proj = self.o_proj(attn_output_flat)
        store(pfx + "attn_o_proj", o_proj)
        return o_proj, None

    import types
    attn_mod.forward = types.MethodType(patched_forward, attn_mod)

    L.post_attention_layernorm.register_forward_hook(hook(pfx + "POST_ATTN_NORM"))
    L.pre_feedforward_layernorm.register_forward_hook(hook(pfx + "PRE_FFN_NORM"))
    L.mlp.gate_proj.register_forward_hook(hook(pfx + "MLP_GATE"))
    L.mlp.up_proj.register_forward_hook(hook(pfx + "MLP_UP"))
    L.mlp.down_proj.register_forward_hook(hook(pfx + "MLP_DOWN"))
    L.mlp.register_forward_hook(hook(pfx + "MLP"))
    L.post_feedforward_layernorm.register_forward_hook(hook(pfx + "POST_FFN_NORM"))
    L.altup.correction_coefs.register_forward_hook(hook(pfx + "ALTUP_CORR_COEFS"))
    L.per_layer_input_gate.register_forward_hook(hook(pfx + "PLI_GATE"))
    L.per_layer_projection.register_forward_hook(hook(pfx + "PLI_PROJ"))
    L.post_per_layer_input_norm.register_forward_hook(hook(pfx + "PLI_NORM"))
    L.register_forward_hook(hook(pfx + "LAYER_OUTPUT"))

    lm.norm.register_forward_hook(hook("OUTPUT_NORM"))

    with torch.no_grad():
        model(torch.tensor([input_ids]), output_hidden_states=True)

    attn_mod.forward = original_forward

    def fmt(v, n=5):
        return "[" + ",".join(f"{x:.4f}" for x in v[:n]) + ",...]"

    out_path = os.path.join(OUTPUT_DIR, f"hf_L{layer}_profile.txt")
    with open(out_path, "w") as f:
        f.write("=== Graph Execution Profile (HF bf16) ===\n")
        f.write(f"{'Operation':<30}{'Time (ms)':<12}{'Output Shape':<24}Backend\n")
        f.write("-" * 72 + "\n")

        for name, t in ops.items():
            if t.dim() == 4:
                if t.shape[1] <= 8 and t.shape[2] > 8:
                    v = t[0, 0, -1].numpy()
                else:
                    v = t[0, -1, 0].numpy()
            elif t.dim() == 3:
                v = t[0, -1].numpy()
            elif t.dim() == 2:
                v = t[-1].numpy() if t.shape[0] > 1 else t[0].numpy()
            else:
                v = t.numpy()

            shape_str = str(list(t.shape))
            f.write(f"{name:<30}{'0.000':<12}{shape_str:<24} values={fmt(v)}\n")

    print(f"HF profile: {out_path} ({len(ops)} ops)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Cactus and HF profile files for comparison")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Cactus weight directory")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to profile")
    parser.add_argument("--prompt", default="who are you", help="User prompt")
    parser.add_argument("--cactus-only", action="store_true", help="Only generate Cactus profile")
    parser.add_argument("--hf-only", action="store_true", help="Only generate HF profile")
    args = parser.parse_args()

    chat_prompt = f"<bos><start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
    target_seq_len = None

    if not args.hf_only:
        _, target_seq_len = generate_cactus_profile(args.weights, args.prompt, args.layer)

    if not args.cactus_only:
        generate_hf_profile(chat_prompt, args.layer, target_seq_len)


if __name__ == "__main__":
    main()
