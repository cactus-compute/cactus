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
    # "full"  → this layer computes its own K/V (q,k,v,o projections present)
    # "sliding" → reuses K/V from the kv_source_layer (only q,o projections present)
    attention_kind: str = "full"
    kv_source_layer: int = -1   # -1 means self (i.e. layer_idx)
    ops: list[Op] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "attention_kind": self.attention_kind,
            "kv_source_layer": self.kv_source_layer,
            "ops": [o.to_dict() for o in self.ops],
        }


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
    # Optional scalar multiplier applied to the embedding lookup output.
    # Gemma 3/4 scale embeddings by sqrt(hidden_size); other families = 1.0.
    embed_scale: float = 1.0
    # Optional final-logit softcap (cap * tanh(logits / cap)). Argmax-
    # invariant, but matters if we ever sample with temperature.
    final_logit_softcap: float = 0.0
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
            "embed_scale": self.embed_scale,
            "final_logit_softcap": self.final_logit_softcap,
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
    num_q_heads  = int(getattr(cfg, "num_attention_heads", 0))
    # head_dim may be None on configs that derive it implicitly (Qwen2.5).
    _hd = getattr(cfg, "head_dim", None)
    head_dim = int(_hd) if _hd else hidden_dim // max(1, num_q_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_q_heads))
    num_layers   = int(getattr(cfg, "num_hidden_layers", 0))
    vocab_size   = int(getattr(cfg, "vocab_size", 0))
    # rope_theta is sometimes on the config root and sometimes nested under
    # rope_scaling (Qwen2.5 uses the latter). Try both.
    _rt = getattr(cfg, "rope_theta", None)
    if _rt is None:
        _rs = getattr(cfg, "rope_scaling", None)
        if isinstance(_rs, dict):
            _rt = _rs.get("rope_theta")
    rope_theta = float(_rt) if _rt is not None else 10000.0
    # Most modern LLMs (LLaMA, Gemma, Qwen) use neox-style RoPE. GPT-J uses
    # interleaved. Default true unless the config says otherwise.
    rope_neox    = bool(getattr(cfg, "rope_neox", True))
    mlp_kind     = "swiglu"  # all of LLaMA/Gemma/Qwen MLPs are SwiGLU-shaped
    weight_format = "cq4_group128" if quantize_bits == 4 else "fp16"
    kv_cache_dtype = "fp16"

    # Gemma 3/4 family: pre-scale embedding by sqrt(hidden_size) and apply a
    # final-logit softcap. We resolve the embed_scale by inspecting the
    # embedding module (HF Gemma's ScaledWordEmbedding stores .embed_scale as
    # a buffer); fall back to sqrt(hidden_size) per the HF source.
    embed_scale = 1.0
    final_softcap = 0.0
    if family.startswith("gemma"):
        embed_scale = float(hidden_dim) ** 0.5
        for path in ("model.language_model.embed_tokens",
                     "model.model.embed_tokens",
                     "language_model.embed_tokens",
                     "model.embed_tokens"):
            mod = model
            ok = True
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if ok:
                for attr in ("scalar_embed_scale", "embed_scale"):
                    v = getattr(mod, attr, None)
                    if v is not None:
                        try:
                            embed_scale = float(v) if not hasattr(v, "item") else float(v.item())
                        except Exception:
                            pass
                        break
                break
        final_softcap = float(getattr(cfg, "final_logit_softcapping", 30.0) or 0.0)

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
        embed_scale=embed_scale,
        final_logit_softcap=final_softcap,
    )

    # Gemma 3n (text decoder) layers have 5 extra RMSNorms vs LLaMA:
    # q_norm, k_norm, v_norm (per-head, applied right after projection),
    # post_attention_layernorm (applied to attn output BEFORE residual add),
    # post_feedforward_layernorm (applied to mlp output BEFORE residual add).
    # We detect this by family rather than by config — the structural
    # difference matters far more than the literal config flag.
    gemma_norms = family.startswith("gemma")
    # Whether the Q/K/V projections carry a bias term. Qwen2.x does; LLaMA /
    # Gemma don't. The kernel always reads buffer(4) when HAS_BIAS=true, so we
    # plumb this through and weight_pack emits the bias bytes when set.
    qkv_has_bias = family.startswith("qwen")

    # Per-layer attention kinds. Gemma 4 uses an interleaved schedule
    # (e.g. ["sliding","sliding","sliding","sliding","full",...]) where
    # *some* sliding layers reuse K/V from the nearest preceding `full`
    # layer (these have only q+o projections). Other architectures
    # (LLaMA, Qwen) have uniform `full` everywhere.
    #
    # Empirical rule: if the layer module HAS k_proj/v_proj, treat as
    # "full" regardless of layer_types tag. Only mark as "sliding" if
    # the module is actually missing those Linears AND there's a prior
    # full-attention layer to source from. This handles Gemma 4's
    # first-4-layer edge case (config says sliding but module has k/v).
    layer_types = list(getattr(cfg, "layer_types", []))

    def _module_has_kv(li: int) -> bool:
        # Walk the HF model to find the layer's self_attn and check for an
        # actually-present k_proj weight tensor (not just the attribute, which
        # may exist as None on sliding layers).
        for path in ("model.language_model.layers", "model.layers",
                     "language_model.layers", "model.model.layers"):
            mod = model
            ok = True
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if ok and li < len(mod):
                sa = getattr(mod[li], "self_attn", None)
                if sa is None:
                    return False
                kp = getattr(sa, "k_proj", None)
                return kp is not None and hasattr(kp, "weight")
        return True  # conservative: assume full

    def _kv_shared_for(li: int) -> tuple[bool, int]:
        """Return (is_kv_shared, source_layer_idx) from the HF attention
        module flags when present. Gemma 4 / Gemma 3n attention modules
        carry ``is_kv_shared_layer`` + ``kv_shared_layer_index`` set by HF
        per the model's actual KV-sharing schedule — this is more accurate
        than guessing "nearest preceding layer that has its own KV"."""
        for path in ("model.language_model.layers", "model.layers",
                     "language_model.layers", "model.model.layers"):
            mod = model
            ok = True
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if ok and li < len(mod):
                sa = getattr(mod[li], "self_attn", None)
                if sa is not None:
                    flag = getattr(sa, "is_kv_shared_layer", None)
                    if flag is True:
                        idx = getattr(sa, "kv_shared_layer_index", None)
                        if idx is not None:
                            return True, int(idx)
                    elif flag is False:
                        return False, -1
                break
        return False, -1

    def _kind_for(li: int) -> tuple[str, int]:
        # 1) Trust the model's KV-sharing flags when they're set (Gemma 4).
        is_shared, src = _kv_shared_for(li)
        if is_shared:
            return "sliding", src
        if _module_has_kv(li):
            return "full", -1
        # 2) Fallback heuristic for older Gemma checkpoints without the flag
        # explicitly set: assume KV reuses from the nearest prior layer that
        # actually has its own k_proj weights.
        for prev in range(li - 1, -1, -1):
            if _module_has_kv(prev):
                return "sliding", prev
        return "full", -1

    for li in range(num_layers):
        kind, src = _kind_for(li)
        layer = Layer(layer_idx=li, attention_kind=kind, kv_source_layer=src)
        # The op sequence below is LLaMA-style by default; if ``gemma_norms``
        # is set, we insert the extra Gemma 3/4 normalizations (q_norm,
        # k_norm, v_norm, post_attention_layernorm, post_feedforward_layernorm).
        layer.ops.append(Op("rms_norm", {"tag": "attn_norm", "axis_size": hidden_dim}))
        # q_proj is always present. k_proj / v_proj only on full-attention layers.
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "q_proj", "K": hidden_dim, "N": num_q_heads * head_dim,
                             "has_bias": qkv_has_bias}))
        if gemma_norms:
            # Per-head Q RMSNorm: axis = head_dim, rows = num_q_heads.
            layer.ops.append(Op("rms_norm",
                                {"tag": "q_norm",
                                 "axis_size": head_dim,
                                 "rows": num_q_heads,
                                 "in_buf": "q_buf", "out_buf": "q_buf"}))
        if kind == "full":
            layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                                {"tag": "k_proj", "K": hidden_dim, "N": num_kv_heads * head_dim,
                                 "has_bias": qkv_has_bias}))
            if gemma_norms:
                layer.ops.append(Op("rms_norm",
                                    {"tag": "k_norm",
                                     "axis_size": head_dim,
                                     "rows": num_kv_heads,
                                     "in_buf": "k_buf", "out_buf": "k_buf"}))
            layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                                {"tag": "v_proj", "K": hidden_dim, "N": num_kv_heads * head_dim,
                                 "has_bias": qkv_has_bias}))
            if gemma_norms:
                layer.ops.append(Op("rms_norm",
                                    {"tag": "v_norm",
                                     "axis_size": head_dim,
                                     "rows": num_kv_heads,
                                     "in_buf": "v_buf", "out_buf": "v_buf"}))
        # RoPE — full layers apply to both Q and K (their own freshly-computed
        # K). Sliding layers only apply to Q (K from the source layer was
        # already RoPE'd when that layer ran).
        layer.ops.append(Op("rope_apply",
                            {"head_dim": head_dim,
                             "num_q_heads": num_q_heads,
                             "num_kv_heads": num_kv_heads if kind == "full" else 0,
                             "is_neox": rope_neox,
                             "theta": rope_theta}))
        if kind == "full":
            layer.ops.append(Op("kv_cache_append",
                                {"num_kv_heads": num_kv_heads, "head_dim": head_dim}))
        layer.ops.append(Op("flash_attn",
                            {"head_dim_q": head_dim, "head_dim_v": head_dim,
                             "num_query_groups": num_q_heads // max(1, num_kv_heads),
                             "causal": True, "has_softcap": False,
                             "kv_source_layer": src if kind == "sliding" else li}))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "out_proj", "K": num_q_heads * head_dim, "N": hidden_dim}))
        if gemma_norms:
            # post_attention_layernorm runs on the attention output (which the
            # runtime parks in mlp_down) BEFORE the residual add.
            layer.ops.append(Op("rms_norm",
                                {"tag": "post_attn_norm",
                                 "axis_size": hidden_dim,
                                 "in_buf": "mlp_down", "out_buf": "mlp_down"}))
        layer.ops.append(Op("residual_add", {"axis_size": hidden_dim}))
        layer.ops.append(Op("rms_norm",
                            {"tag": "pre_ff_norm" if gemma_norms else "mlp_norm",
                             "axis_size": hidden_dim}))
        intermediate = int(getattr(cfg, "intermediate_size", 4 * hidden_dim))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "gate_proj", "K": hidden_dim, "N": intermediate}))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "up_proj",   "K": hidden_dim, "N": intermediate}))
        layer.ops.append(Op("swiglu_fwd", {"hidden_dim": intermediate}))
        layer.ops.append(Op("mul_mv_int4_fp16" if quantize_bits == 4 else "mul_mv_fp16",
                            {"tag": "down_proj", "K": intermediate, "N": hidden_dim}))
        if gemma_norms:
            layer.ops.append(Op("rms_norm",
                                {"tag": "post_ff_norm",
                                 "axis_size": hidden_dim,
                                 "in_buf": "mlp_down", "out_buf": "mlp_down"}))
        layer.ops.append(Op("residual_add", {"axis_size": hidden_dim}))
        plan.layers.append(layer)

    plan.final_norm = {"kind": "rms_norm", "axis_size": hidden_dim}
    plan.lm_head = {
        "kind": "mul_mv_fp16",  # LM head usually stays fp16 for accuracy
        "K": hidden_dim,
        "N": vocab_size,
    }
    return plan
