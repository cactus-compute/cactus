from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldPattern:
    name: str
    semantic_ops: tuple[str, ...]
    description: str


GOLD_PATTERNS: tuple[GoldPattern, ...] = (
    GoldPattern(
        name="decoder_attention_gqa",
        semantic_ops=("rms_norm", "rope", "attention", "attention_int8_hybrid"),
        description=(
            "Q/K/V projections from the same normalized hidden state; Q and K optionally get per-head RMSNorm, "
            "Q/K are rotary-embedded, then Cactus attention consumes native Q-head and KV-head counts."
        ),
    ),
    GoldPattern(
        name="gated_mlp_gelu",
        semantic_ops=("matmul", "gelu", "multiply"),
        description="Two projection branches from the same normalized input: GELU(gate) * up, then a down projection.",
    ),
    GoldPattern(
        name="gated_mlp_silu",
        semantic_ops=("matmul", "silu", "multiply"),
        description="Two projection branches from the same normalized input: SiLU(gate) * up, then a down projection.",
    ),
    GoldPattern(
        name="decoder_block_post_attn_norm",
        semantic_ops=("rms_norm", "attention", "add_clipped"),
        description=(
            "Pre-norm attention block with post-attention RMSNorm before the residual add, followed by a second "
            "pre-FFN RMSNorm and a post-FFN RMSNorm before the final residual add."
        ),
    ),
    GoldPattern(
        name="decoder_block_simple_residual",
        semantic_ops=("rms_norm", "attention", "add"),
        description=(
            "Pre-norm attention block with a direct residual add after attention, followed by a second RMSNorm and "
            "a gated MLP branch with another direct residual add."
        ),
    ),
    GoldPattern(
        name="gemma4_partial_rope_attention",
        semantic_ops=("rms_norm", "rope", "concat", "attention"),
        description=(
            "Gemma4 attention where only a prefix of the head dimension is rotary-embedded, optionally with shared "
            "K/V heads and sliding-window attention."
        ),
    ),
    GoldPattern(
        name="sliding_window_attention_mask",
        semantic_ops=("attention",),
        description=(
            "Exported PyTorch may build a boolean mask with diff/cumsum/comparisons, but the handwritten Cactus "
            "models encode the same behavior directly as an attention `window_size`."
        ),
    ),
)
