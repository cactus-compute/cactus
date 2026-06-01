#!/usr/bin/env python3
"""Long generation DeepSeek V4 cactus-vs-HF parity."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer


T = 1060
PREFILL = 530
GENERATE = 530
V = 32
D = 8
HC = 4
MIX = 24
H = 2
HD = 4
RD = 2
QA = 6
OG = 2
OR = 3
E = 16
I = 5
K = 4
IH = 2
ID = 4
NLAYERS = 5


class SqrtSoftplusActivation(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.nn.functional.softplus(x))


ACT2FN.setdefault("sqrtsoftplus", SqrtSoftplusActivation)


def seed_values(count: int, scale: float, phase: float, fp16: bool) -> np.ndarray:
    values = np.array([scale * math.sin(phase + 0.173 * (i + 1)) for i in range(count)], dtype=np.float32)
    return values.astype(np.float16).astype(np.float32) if fp16 else values


def default_tokens() -> np.ndarray:
    return np.array([(t * 7 + 3) % V for t in range(T)], dtype=np.int64)


def import_local_hf(repo: pathlib.Path):
    package_root = pathlib.Path(tempfile.mkdtemp(prefix="cactus_dsv4_hf_import_"))
    package_dir = package_root / "deepseek_v4"
    package_dir.mkdir()
    source_dir = repo / "hf_implementation" / "transformers"
    shutil.copy(source_dir / "configuration_deepseek_v4.py", package_dir / "configuration_deepseek_v4.py")
    shutil.copy(source_dir / "modeling_deepseek_v4.py", package_dir / "modeling_deepseek_v4.py")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    import transformers.models

    transformers.models.__path__.append(str(package_root))
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4CSACache,
        DeepseekV4ForCausalLM,
        DeepseekV4HCACache,
    )

    return DeepseekV4Config, DeepseekV4ForCausalLM, DeepseekV4CSACache, DeepseekV4HCACache


def assign_param(param: torch.nn.Parameter, values: np.ndarray, shape: tuple[int, ...]) -> None:
    param.data.copy_(torch.from_numpy(values.reshape(shape)).to(dtype=param.dtype))


def fill_layer(layer: torch.nn.Module, layer_idx: int) -> None:
    p = 0.35 + float(layer_idx) * 4.0
    assign_param(layer.attn_hc.fn, seed_values(MIX * HC * D, 0.013, p + 0.1, False), (MIX, HC * D))
    assign_param(layer.attn_hc.base, seed_values(MIX, 0.008, p + 0.2, False), (MIX,))
    layer.attn_hc.scale.data.copy_(torch.tensor([0.55 + 0.03 * layer_idx, -0.28 + 0.02 * layer_idx, 0.17]))
    assign_param(layer.ffn_hc.fn, seed_values(MIX * HC * D, 0.012, p + 0.3, False), (MIX, HC * D))
    assign_param(layer.ffn_hc.base, seed_values(MIX, 0.007, p + 0.4, False), (MIX,))
    layer.ffn_hc.scale.data.copy_(torch.tensor([-0.31, 0.42 + 0.02 * layer_idx, 0.19]))

    layer.input_layernorm.weight.data.copy_(torch.tensor([0.9 + 0.003 * ((i + layer_idx) % 5) for i in range(D)]))
    layer.post_attention_layernorm.weight.data.copy_(
        torch.tensor([0.88 + 0.004 * ((i + 2 * layer_idx) % 5) for i in range(D)])
    )
    attn = layer.self_attn
    assign_param(attn.q_a_proj.weight, seed_values(QA * D, 0.024, p + 0.5, True), (QA, D))
    attn.q_a_norm.weight.data.copy_(torch.tensor([0.91 + 0.002 * ((i + layer_idx) % 4) for i in range(QA)]))
    assign_param(attn.q_b_proj.weight, seed_values(H * HD * QA, 0.022, p + 0.6, True), (H * HD, QA))
    assign_param(attn.kv_proj.weight, seed_values(HD * D, 0.023, p + 0.7, True), (HD, D))
    attn.kv_norm.weight.data.copy_(torch.tensor([0.89 + 0.004 * ((i + layer_idx) % 3) for i in range(HD)]))
    assign_param(attn.o_a_proj.weight, seed_values(OG * OR * (H * HD // OG), 0.021, p + 0.8, True), (OG * OR, H * HD // OG))
    assign_param(attn.o_b_proj.weight, seed_values(D * OG * OR, 0.020, p + 0.9, True), (D, OG * OR))
    assign_param(attn.sinks, seed_values(H, 0.012, p + 1.0, False), (H,))

    if attn.compressor is not None and attn.layer_type == "heavily_compressed_attention":
        comp = attn.compressor
        assign_param(comp.kv_proj.weight, seed_values(HD * D, 0.018, p + 1.1, True), (HD, D))
        assign_param(comp.gate_proj.weight, seed_values(HD * D, 0.017, p + 1.2, True), (HD, D))
        assign_param(comp.position_bias, seed_values(128 * HD, 0.010, p + 1.3, False), (128, HD))
        comp.kv_norm.weight.data.copy_(torch.tensor([0.87 + 0.004 * ((i + layer_idx) % 3) for i in range(HD)]))
    if attn.compressor is not None and attn.layer_type == "compressed_sparse_attention":
        comp = attn.compressor
        assign_param(comp.kv_proj.weight, seed_values(2 * HD * D, 0.018, p + 1.4, True), (2 * HD, D))
        assign_param(comp.gate_proj.weight, seed_values(2 * HD * D, 0.017, p + 1.5, True), (2 * HD, D))
        assign_param(comp.position_bias, seed_values(4 * 2 * HD, 0.010, p + 1.6, False), (4, 2 * HD))
        comp.kv_norm.weight.data.copy_(torch.tensor([0.86 + 0.004 * ((i + layer_idx) % 3) for i in range(HD)]))
        idx = comp.indexer
        assign_param(idx.kv_proj.weight, seed_values(2 * ID * D, 0.019, p + 1.7, True), (2 * ID, D))
        assign_param(idx.gate_proj.weight, seed_values(2 * ID * D, 0.018, p + 1.8, True), (2 * ID, D))
        assign_param(idx.q_b_proj.weight, seed_values(IH * ID * QA, 0.020, p + 1.9, True), (IH * ID, QA))
        assign_param(idx.weights_proj.weight, seed_values(IH * D, 0.021, p + 2.0, True), (IH, D))
        assign_param(idx.position_bias, seed_values(4 * 2 * ID, 0.010, p + 2.1, False), (4, 2 * ID))
        idx.kv_norm.weight.data.copy_(torch.tensor([0.85 + 0.003 * ((i + layer_idx) % 4) for i in range(ID)]))

    mlp = layer.mlp
    assign_param(mlp.gate.weight, seed_values(E * D, 0.021, p + 2.2, True), (E, D))
    if hasattr(mlp.gate, "e_score_correction_bias"):
        assign_param(mlp.gate.e_score_correction_bias, seed_values(E, 0.006, p + 2.3, True), (E,))
    if hasattr(mlp.gate, "tid2eid"):
        table = np.array([[(tok + k * 3 + layer_idx) % E for k in range(K)] for tok in range(V)], dtype=np.int64)
        mlp.gate.tid2eid.data.copy_(torch.from_numpy(table))
    for expert in range(E):
        gate = seed_values(I * D, 0.019, p + 2.4 + 0.13 * expert, True).reshape(I, D)
        up = seed_values(I * D, 0.018, p + 2.8 + 0.13 * expert, True).reshape(I, D)
        mlp.experts.gate_up_proj[expert].data.copy_(torch.from_numpy(np.concatenate([gate, up], axis=0)))
        assign_param(mlp.experts.down_proj[expert], seed_values(D * I, 0.017, p + 3.2 + 0.13 * expert, True), (D, I))
    assign_param(mlp.shared_experts.gate_proj.weight, seed_values(I * D, 0.018, p + 3.6, True), (I, D))
    assign_param(mlp.shared_experts.up_proj.weight, seed_values(I * D, 0.017, p + 3.7, True), (I, D))
    assign_param(mlp.shared_experts.down_proj.weight, seed_values(D * I, 0.016, p + 3.8, True), (D, I))


def make_model(repo: pathlib.Path):
    DeepseekV4Config, DeepseekV4ForCausalLM, DeepseekV4CSACache, DeepseekV4HCACache = import_local_hf(repo)
    config = DeepseekV4Config(
        vocab_size=V,
        hidden_size=D,
        moe_intermediate_size=I,
        num_hidden_layers=NLAYERS,
        num_attention_heads=H,
        num_key_value_heads=1,
        head_dim=HD,
        q_lora_rank=QA,
        num_experts_per_tok=K,
        n_routed_experts=E,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe", "moe"],
        hc_mult=HC,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        sliding_window=128,
        o_groups=OG,
        o_lora_rank=OR,
        index_n_heads=IH,
        index_head_dim=ID,
        index_topk=K,
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=1,
        partial_rotary_factor=RD / HD,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        attention_dropout=0.0,
        output_router_logits=False,
        use_cache=False,
    )
    config._experts_implementation = "eager"
    model = DeepseekV4ForCausalLM(config).eval()

    def make_hca_cache():
        cache_layer = DeepseekV4HCACache.__new__(DeepseekV4HCACache)
        DynamicSlidingWindowLayer.__init__(cache_layer, sliding_window=config.sliding_window)
        cache_layer.compress_rate = config.compress_rates["heavily_compressed_attention"]
        cache_layer.buffer_kv = {"compressor": None}
        cache_layer.buffer_gate = {"compressor": None}
        cache_layer.compressed_kv = {"compressor": None}
        cache_layer.entry_count = {"compressor": 0}
        return cache_layer

    def make_csa_cache():
        cache_layer = DeepseekV4CSACache.__new__(DeepseekV4CSACache)
        DynamicSlidingWindowLayer.__init__(cache_layer, sliding_window=config.sliding_window)
        cache_layer.compress_rate = config.compress_rates["compressed_sparse_attention"]
        cache_layer.buffer_kv = {"compressor": None, "indexer": None}
        cache_layer.buffer_gate = {"compressor": None, "indexer": None}
        cache_layer.compressed_kv = {"compressor": None, "indexer": None}
        cache_layer.entry_count = {"compressor": 0, "indexer": 0}
        cache_layer.overlap_kv = {"compressor": None, "indexer": None}
        cache_layer.overlap_gate = {"compressor": None, "indexer": None}
        return cache_layer

    with torch.no_grad():
        assign_param(model.model.embed_tokens.weight, seed_values(V * D, 0.055, 0.1, True), (V, D))
        assign_param(model.lm_head.weight, seed_values(V * D, 0.030, 0.2, True), (V, D))
        assign_param(model.model.hc_head.hc_fn, seed_values(HC * HC * D, 0.012, 0.3, False), (HC, HC * D))
        assign_param(model.model.hc_head.hc_base, seed_values(HC, 0.006, 0.4, False), (HC,))
        model.model.hc_head.hc_scale.data.copy_(torch.tensor([0.5]))
        model.model.norm.weight.data.copy_(torch.tensor([0.92 + 0.003 * i for i in range(D)]))
        for layer_idx, layer in enumerate(model.model.layers):
            fill_layer(layer, layer_idx)
    return model, config, make_csa_cache, make_hca_cache


def make_cache(config, make_csa_cache, make_hca_cache):
    cache = DynamicCache(config=config)
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "compressed_sparse_attention":
            cache.layers[layer_idx] = make_csa_cache()
        elif layer_type == "heavily_compressed_attention":
            cache.layers[layer_idx] = make_hca_cache()
    return cache


def hf_logits(model, config, make_csa_cache, make_hca_cache, tokens: np.ndarray) -> np.ndarray:
    input_ids = torch.tensor([tokens.tolist()], dtype=torch.long)
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            past_key_values=make_cache(config, make_csa_cache, make_hca_cache),
            use_cache=False,
            logits_to_keep=0,
        ).logits.squeeze(0).cpu().numpy()


def cactus_logits(binary: pathlib.Path, tokens: np.ndarray) -> np.ndarray:
    token_csv = ",".join(str(int(t)) for t in tokens.tolist())
    proc = subprocess.run(
        [str(binary)],
        check=True,
        text=True,
        capture_output=True,
        env={
            **dict(os.environ),
            "CACTUS_DSV4_DUMP_5L_LOGITS": "1",
            "CACTUS_DSV4_5L_TOKENS": token_csv,
        },
    )
    match = re.search(r"CACTUS_DSV4_HF_5L_LOGITS\s+(\[[^\n]+\])", proc.stdout)
    if not match:
        raise RuntimeError(f"cactus logits marker missing from output:\n{proc.stdout}\n{proc.stderr}")
    return np.array(json.loads(match.group(1)), dtype=np.float32).reshape(T, V)


def parity_case(name: str, hf: np.ndarray, cactus: np.ndarray, max_abs_tol: float, mean_abs_tol: float) -> bool:
    diff = cactus - hf
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    max_idx = tuple(int(i) for i in np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape))
    print(
        f"{name}: max_abs={max_abs:.8f} at {max_idx}: "
        f"cactus={cactus[max_idx]:.8f} hf={hf[max_idx]:.8f}; mean_abs={mean_abs:.8f}"
    )
    return max_abs <= max_abs_tol and mean_abs <= mean_abs_tol


def long_generation_parity(model, config, make_csa_cache, make_hca_cache, binary: pathlib.Path,
                           max_abs_tol: float, mean_abs_tol: float) -> bool:
    prefix = [int((t * 7 + 3) % V) for t in range(PREFILL)]
    hf_rows: list[np.ndarray] = []
    generated: list[int] = []
    for step in range(GENERATE):
        tokens = np.array(prefix + [0] * (T - len(prefix)), dtype=np.int64)
        logits = hf_logits(model, config, make_csa_cache, make_hca_cache, tokens)
        row = logits[len(prefix) - 1].copy()
        next_token = int(np.argmax(row))
        hf_rows.append(row)
        generated.append(next_token)
        prefix.append(next_token)
        if (step + 1) % 50 == 0 or step == 0:
            print(f"HF generated {step + 1}/{GENERATE}; latest token={next_token}")

    full_tokens = np.array(prefix, dtype=np.int64)
    cactus = cactus_logits(binary, full_tokens)
    ok = True
    max_abs_seen = 0.0
    mean_abs_seen = 0.0
    mismatches: list[tuple[int, int, int]] = []
    for step, hf_row in enumerate(hf_rows):
        pos = PREFILL - 1 + step
        cactus_row = cactus[pos]
        diff = cactus_row - hf_row
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        max_abs_seen = max(max_abs_seen, max_abs)
        mean_abs_seen = max(mean_abs_seen, mean_abs)
        hf_next = generated[step]
        cactus_next = int(np.argmax(cactus_row))
        if hf_next != cactus_next:
            mismatches.append((step + 1, hf_next, cactus_next))
        ok = ok and hf_next == cactus_next and max_abs <= max_abs_tol and mean_abs <= mean_abs_tol
        if step < 5 or (step + 1) % 100 == 0:
            print(
                f"generation step {step + 1}: pos={pos} hf_next={hf_next} cactus_next={cactus_next} "
                f"max_abs={max_abs:.8f} mean_abs={mean_abs:.8f}"
            )
    print(f"generated_count={len(generated)}")
    print(f"first_20_generated={generated[:20]}")
    print(f"last_20_generated={generated[-20:]}")
    print(f"worst_generation_max_abs={max_abs_seen:.8f}")
    print(f"worst_generation_mean_abs={mean_abs_seen:.8f}")
    if mismatches:
        print(f"argmax mismatches ({len(mismatches)}): {mismatches[:20]}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--cactus-binary", type=pathlib.Path, required=True)
    parser.add_argument("--max-abs-tol", type=float, default=4.0e-2)
    parser.add_argument("--mean-abs-tol", type=float, default=8.0e-3)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model, config, make_csa_cache, make_hca_cache = make_model(args.repo)
    ok = long_generation_parity(
        model,
        config,
        make_csa_cache,
        make_hca_cache,
        args.cactus_binary,
        args.max_abs_tol,
        args.mean_abs_tol,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
