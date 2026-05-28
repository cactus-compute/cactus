"""GPU dispatch plan — bridges HF model architecture to per-layer Metal kernel calls.

A ``GPUPlan`` is the JSON object the runtime reads to know what kernels
to dispatch in which order. It's per-model (one per bundle), not per
inference. Built once at convert time from the HF model's config.

The schema (intentionally narrow, room to grow):

    {
      "version": 1,
      "model_family": "gemma4",
      "hidden_dim": 4096,
      "head_dim": 128,
      "num_q_heads": 32,
      "num_kv_heads": 8,
      "num_layers": 24,
      "vocab_size": 256000,
      "rope_theta": 10000.0,
      "rope_neox": true,
      "mlp_kind": "swiglu",
      "weight_format": "cq4_group128",
      "kv_cache_dtype": "fp16",
      "layers": [
        {
          "layer_idx": 0,
          "ops": [
            {"kind": "rms_norm", "weight_offset_bytes": ..., "axis_size": 4096},
            {"kind": "mul_mv_int4_fp16", "weight_offset_bytes": ..., "scale_offset_bytes": ..., "K": 4096, "N": 4096, "tag": "q_proj"},
            ...
          ]
        },
        ...
      ],
      "final_norm": {...},
      "lm_head": {...}
    }

Op kinds map 1:1 to ``cactus_gpu.h`` ``pipeline_*`` functions.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Op:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.args}


@dataclass
class Layer:
    layer_idx: int
    ops: list[Op] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"layer_idx": self.layer_idx, "ops": [o.to_dict() for o in self.ops]}


@dataclass
class GPUPlan:
    model_family: str
    hidden_dim: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int
    rope_theta: float
    rope_neox: bool
    mlp_kind: str
    weight_format: str
    kv_cache_dtype: str
    layers: list[Layer] = field(default_factory=list)
    final_norm: dict[str, Any] | None = None
    lm_head: dict[str, Any] | None = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_family": self.model_family,
            "hidden_dim": self.hidden_dim,
            "head_dim": self.head_dim,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "rope_theta": self.rope_theta,
            "rope_neox": self.rope_neox,
            "mlp_kind": self.mlp_kind,
            "weight_format": self.weight_format,
            "kv_cache_dtype": self.kv_cache_dtype,
            "layers": [l.to_dict() for l in self.layers],
            "final_norm": self.final_norm,
            "lm_head": self.lm_head,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


# ----------------------------------------------------------------------------
# Plan extraction from HF models
# ----------------------------------------------------------------------------

def _detect_family(model: torch.nn.Module) -> str:
    name = type(model).__name__.lower()
    for fam in ("gemma4", "gemma3", "gemma", "qwen", "lfm2", "llama"):
        if fam in name:
            return fam
    return "unknown"


def _detect_text_config(model: torch.nn.Module):
    """Locate the text-decoder config. For multimodal HF models, this is
    nested under ``model.config.text_config`` or similar."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        # Try the multimodal nesting first.
        for attr in ("text_config", "language_config"):
            sub = getattr(cfg, attr, None)
            if sub is not None and getattr(sub, "hidden_size", None):
                return sub
        if getattr(cfg, "hidden_size", None):
            return cfg
    return None


def build_gpu_plan(model: torch.nn.Module, *, quantize_bits: int | None) -> GPUPlan:
    """Inspect the HF model + config and emit a GPU dispatch plan.

    This is intentionally a *thin* skeleton — it captures the dims and
    layer count, but doesn't yet enumerate every per-layer op. Op
    enumeration (RMSNorm → QKV → RoPE → FA → out_proj → ...) is the next
    deliverable; right now we record dims so the C++ runtime can at
    least size buffers correctly."""
    family = _detect_family(model)
    cfg = _detect_text_config(model)
    if cfg is None:
        raise RuntimeError("gpu.plan: could not find a text-decoder config on this model")

    hidden_dim   = int(getattr(cfg, "hidden_size", 0))
    head_dim     = int(getattr(cfg, "head_dim",
                       getattr(cfg, "hidden_size", 0) // max(1, int(getattr(cfg, "num_attention_heads", 1)))))
    num_q_heads  = int(getattr(cfg, "num_attention_heads", 0))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_q_heads))
    num_layers   = int(getattr(cfg, "num_hidden_layers", 0))
    vocab_size   = int(getattr(cfg, "vocab_size", 0))
    rope_theta   = float(getattr(cfg, "rope_theta", 10000.0))
    # Most modern LLMs (LLaMA, Gemma, Qwen) use neox-style RoPE. GPT-J uses
    # interleaved. Default true unless the config says otherwise.
    rope_neox    = bool(getattr(cfg, "rope_neox", True))
    mlp_kind     = "swiglu"  # all of LLaMA/Gemma/Qwen MLPs are SwiGLU-shaped
    weight_format = "cq4_group128" if quantize_bits == 4 else "fp16"
    kv_cache_dtype = "fp16"

    plan = GPUPlan(
        model_family=family,
        hidden_dim=hidden_dim,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        rope_theta=rope_theta,
        rope_neox=rope_neox,
        mlp_kind=mlp_kind,
        weight_format=weight_format,
        kv_cache_dtype=kv_cache_dtype,
    )

    # M1: enumerate per-layer ops with placeholder weight offsets. The real
    # offsets are filled in by ``weight_pack`` (it builds the offsets as a
    # side-effect of laying out the binary file).
    for li in range(num_layers):
        layer = Layer(layer_idx=li)
        # canonical Llama-style decoder layer:
        #   in -> rms_norm_attn -> q_proj | k_proj | v_proj -> rope -> kv_append
        #      -> flash_attn -> out_proj -> +residual
        #      -> rms_norm_mlp -> gate_proj | up_proj -> swiglu -> down_proj -> +residual
        layer.ops.append(Op("rms_norm", {"tag": "attn_norm", "axis_size": hidden_dim}))
        for tag, in_dim, out_dim in (
            ("q_proj", hidden_dim, num_q_heads  * head_dim),
            ("k_proj", hidden_dim, num_kv_heads * head_dim),
            ("v_proj", hidden_dim, num_kv_heads * head_dim),
        ):
            layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                                {"tag": tag, "K": in_dim, "N": out_dim}))
        layer.ops.append(Op("rope_apply",
                            {"head_dim": head_dim,
                             "num_q_heads": num_q_heads,
                             "num_kv_heads": num_kv_heads,
                             "is_neox": rope_neox,
                             "theta": rope_theta}))
        layer.ops.append(Op("kv_cache_append",
                            {"num_kv_heads": num_kv_heads, "head_dim": head_dim}))
        layer.ops.append(Op("flash_attn",
                            {"head_dim_q": head_dim, "head_dim_v": head_dim,
                             "num_query_groups": num_q_heads // max(1, num_kv_heads),
                             "causal": True, "has_softcap": False}))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "out_proj", "K": num_q_heads * head_dim, "N": hidden_dim}))
        layer.ops.append(Op("residual_add", {"axis_size": hidden_dim}))
        layer.ops.append(Op("rms_norm", {"tag": "mlp_norm", "axis_size": hidden_dim}))
        intermediate = int(getattr(cfg, "intermediate_size", 4 * hidden_dim))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "gate_proj", "K": hidden_dim, "N": intermediate}))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "up_proj",   "K": hidden_dim, "N": intermediate}))
        layer.ops.append(Op("swiglu_fwd", {"hidden_dim": intermediate}))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "down_proj", "K": intermediate, "N": hidden_dim}))
        layer.ops.append(Op("residual_add", {"axis_size": hidden_dim}))
        plan.layers.append(layer)

    plan.final_norm = {"kind": "rms_norm", "axis_size": hidden_dim}
    plan.lm_head = {
        "kind": "mul_mv_fp16",  # LM head usually stays fp16 for accuracy
        "K": hidden_dim,
        "N": vocab_size,
    }
    return plan
