#!/usr/bin/env python3
"""Compare the cactus DeepSeek V4 toy graph against the actual HF runtime."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers.activations import ACT2FN


class SqrtSoftplusActivation(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.nn.functional.softplus(x))


ACT2FN.setdefault("sqrtsoftplus", SqrtSoftplusActivation)


T = 3
V = 9
D = 8
HC = 4
MIX = 24
H = 2
HD = 4
RD = 2
QA = 6
OG = 2
OR = 3
E = 3
I = 5
K = 2


def seed_values(count: int, scale: float, phase: float, fp16: bool) -> np.ndarray:
    values = np.array(
        [scale * math.sin(phase + 0.173 * float(i + 1)) for i in range(count)],
        dtype=np.float32,
    )
    if fp16:
        values = values.astype(np.float16).astype(np.float32)
    return values


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
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    return DeepseekV4Config, DeepseekV4ForCausalLM


def assign_param(param: torch.nn.Parameter, values: np.ndarray, shape: tuple[int, ...]) -> None:
    tensor = torch.from_numpy(values.reshape(shape)).to(dtype=param.dtype)
    param.data.copy_(tensor)


def fill_model(model: torch.nn.Module) -> None:
    layer = model.model.layers[0]
    with torch.no_grad():
        assign_param(model.model.embed_tokens.weight, seed_values(V * D, 0.12, 0.1, True), (V, D))

        assign_param(layer.attn_hc.fn, seed_values(MIX * HC * D, 0.025, 0.2, False), (MIX, HC * D))
        assign_param(layer.attn_hc.base, seed_values(MIX, 0.015, 0.3, False), (MIX,))
        layer.attn_hc.scale.data.copy_(torch.tensor([0.7, -0.35, 0.2], dtype=layer.attn_hc.scale.dtype))

        assign_param(layer.ffn_hc.fn, seed_values(MIX * HC * D, 0.022, 0.4, False), (MIX, HC * D))
        assign_param(layer.ffn_hc.base, seed_values(MIX, 0.014, 0.5, False), (MIX,))
        layer.ffn_hc.scale.data.copy_(torch.tensor([-0.4, 0.55, 0.25], dtype=layer.ffn_hc.scale.dtype))

        assign_param(model.model.hc_head.hc_fn, seed_values(HC * HC * D, 0.02, 0.6, False), (HC, HC * D))
        assign_param(model.model.hc_head.hc_base, seed_values(HC, 0.015, 0.7, False), (HC,))
        model.model.hc_head.hc_scale.data.copy_(torch.tensor([0.6], dtype=model.model.hc_head.hc_scale.dtype))

        layer.input_layernorm.weight.data.copy_(torch.tensor([0.86 + 0.011 * i for i in range(D)]))
        layer.post_attention_layernorm.weight.data.copy_(torch.tensor([0.91 - 0.007 * i for i in range(D)]))
        model.model.norm.weight.data.copy_(torch.tensor([0.82 + 0.014 * i for i in range(D)]))

        assign_param(layer.self_attn.q_a_proj.weight, seed_values(QA * D, 0.055, 0.8, True), (QA, D))
        layer.self_attn.q_a_norm.weight.data.copy_(torch.tensor([0.88 + 0.009 * i for i in range(QA)]))
        assign_param(layer.self_attn.q_b_proj.weight, seed_values(H * HD * QA, 0.05, 0.9, True), (H * HD, QA))
        assign_param(layer.self_attn.kv_proj.weight, seed_values(HD * D, 0.052, 1.0, True), (HD, D))
        layer.self_attn.kv_norm.weight.data.copy_(torch.tensor([0.9 - 0.012 * i for i in range(HD)]))
        assign_param(
            layer.self_attn.o_a_proj.weight,
            seed_values(OG * OR * (H * HD // OG), 0.048, 1.1, True),
            (OG * OR, H * HD // OG),
        )
        assign_param(layer.self_attn.o_b_proj.weight, seed_values(D * OG * OR, 0.046, 1.2, True), (D, OG * OR))
        assign_param(layer.self_attn.sinks, seed_values(H, 0.025, 1.3, False), (H,))

        assign_param(layer.mlp.gate.weight, seed_values(E * D, 0.05, 1.4, True), (E, D))
        assign_param(layer.mlp.gate.e_score_correction_bias, seed_values(E, 0.01, 1.5, True), (E,))
        for expert in range(E):
            gate = seed_values(I * D, 0.045, 1.6 + float(expert), True).reshape(I, D)
            up = seed_values(I * D, 0.043, 2.0 + float(expert), True).reshape(I, D)
            layer.mlp.experts.gate_up_proj[expert].data.copy_(torch.from_numpy(np.concatenate([gate, up], axis=0)))
            assign_param(
                layer.mlp.experts.down_proj[expert],
                seed_values(D * I, 0.041, 2.4 + float(expert), True),
                (D, I),
            )
        assign_param(layer.mlp.shared_experts.gate_proj.weight, seed_values(I * D, 0.04, 3.1, True), (I, D))
        assign_param(layer.mlp.shared_experts.up_proj.weight, seed_values(I * D, 0.038, 3.4, True), (I, D))
        assign_param(layer.mlp.shared_experts.down_proj.weight, seed_values(D * I, 0.036, 3.7, True), (D, I))

        assign_param(model.lm_head.weight, seed_values(V * D, 0.06, 4.0, True), (V, D))


def hf_logits(repo: pathlib.Path) -> np.ndarray:
    DeepseekV4Config, DeepseekV4ForCausalLM = import_local_hf(repo)
    config = DeepseekV4Config(
        vocab_size=V,
        hidden_size=D,
        moe_intermediate_size=I,
        num_hidden_layers=1,
        num_attention_heads=H,
        num_key_value_heads=1,
        head_dim=HD,
        q_lora_rank=QA,
        num_experts_per_tok=K,
        n_routed_experts=E,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        hc_mult=HC,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        sliding_window=128,
        o_groups=OG,
        o_lora_rank=OR,
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=1,
        partial_rotary_factor=RD / HD,
        rope_theta=10000.0,
        attention_dropout=0.0,
        output_router_logits=False,
        use_cache=False,
    )
    config._experts_implementation = "eager"
    model = DeepseekV4ForCausalLM(config).eval()
    fill_model(model)
    input_ids = torch.tensor([[1, 3, 5]], dtype=torch.long)
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False, logits_to_keep=0).logits.squeeze(0).cpu().numpy()


def cactus_logits(binary: pathlib.Path) -> np.ndarray:
    proc = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
    match = re.search(r"CACTUS_DSV4_HF_LOGITS\s+(\[[^\n]+\])", proc.stdout)
    if not match:
        raise RuntimeError(f"cactus logits marker missing from output:\n{proc.stdout}\n{proc.stderr}")
    return np.array(json.loads(match.group(1)), dtype=np.float32).reshape(T, V)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--cactus-binary", type=pathlib.Path, required=True)
    parser.add_argument("--max-abs-tol", type=float, default=2.0e-2)
    parser.add_argument("--mean-abs-tol", type=float, default=5.0e-3)
    args = parser.parse_args()

    torch.set_num_threads(1)
    hf = hf_logits(args.repo)
    cactus = cactus_logits(args.cactus_binary)
    diff = cactus - hf
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    max_idx = tuple(int(i) for i in np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape))
    print(f"HF logits shape: {hf.shape}")
    print(f"cactus logits shape: {cactus.shape}")
    print(f"max_abs={max_abs:.8f} at {max_idx}: cactus={cactus[max_idx]:.8f} hf={hf[max_idx]:.8f}")
    print(f"mean_abs={mean_abs:.8f}")
    if max_abs > args.max_abs_tol or mean_abs > args.mean_abs_tol:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
