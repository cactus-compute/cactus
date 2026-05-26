from __future__ import annotations

from dataclasses import dataclass

from .naming import NameMatch


@dataclass(frozen=True)
class TensorPolicy:
    action: str
    precision: str
    bits: int | None
    component: str
    use_gptq: bool
    rotation: str
    fallback_reason: str | None = None
    layout: str = "row_major"


def policy_for_tensor(match: NameMatch, shape: tuple[int, ...], user_bits: int, family: str) -> TensorPolicy:
    component = match.component
    name = match.source_name
    out = match.output_name or ""
    if match.output_name is None:
        return TensorPolicy("ignored", "none", None, component, False, "none", "no output filename")
    if "norm" in out.lower():
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "norm tensor")
    if family == "gemma4" and name == "model.embed_vision.embedding_projection.weight":
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "vision embedding projection scale-sensitive")
    if family in {"parakeet", "parakeet_tdt"} and out.endswith(".bias") and (
        "conv_" in out or out.startswith("subsampling_") or out.startswith("ctc_head_")
    ):
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "conv bias tensor")
    if family in {"parakeet", "parakeet_tdt"} and "lstm" in out.lower():
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "lstm recurrent tensor")
    if family == "parakeet_tdt" and out.startswith("tdt_"):
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "tdt decoder tensor")
    if family in {"parakeet", "parakeet_tdt"} and "self_attn_bias_" in out:
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "relative attention bias tensor")
    if family in {"parakeet", "parakeet_tdt"} and "conv_pointwise" in out and len(shape) == 3 and shape[2] == 1:
        return TensorPolicy("fallback", "INT8", 8, component, False, "none", "pointwise conv tensor")
    if "conv_depthwise.weights" in out and len(shape) == 3 and shape[1] == 1:
        return TensorPolicy("fallback", "INT8", 8, component, False, "none", "depthwise conv tensor")
    if family in {"parakeet", "parakeet_tdt"} and out.startswith("layer_") and (
        "conv_pointwise" in out or "conv_depthwise" in out
    ):
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "conformer conv tensor")
    if family == "whisper" and out.startswith("encoder.layer_"):
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "whisper encoder tensor")
    if family == "gemma4" and component == "audio" and (name.endswith(".bias") or out.endswith(".bias") or ".bias." in out):
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "audio bias tensor")
    if name.endswith(".bias") or out.endswith(".bias") or ".bias." in out:
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "bias tensor")
    if len(shape) != 2:
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "non-2d tensor")
    if "position_embedding" in out.lower() or "pos_embed" in out.lower() or "embed_positions" in out.lower():
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "position embedding tensor")
    if family == "gemma4" and name == "model.language_model.embed_tokens_per_layer.weight":
        return TensorPolicy("convert", f"CQ{user_bits}", user_bits, component, False, "hadamard")
    if family == "gemma4" and component in {"audio", "vision"}:
        return TensorPolicy("fallback", "FP16", None, component, False, "none", "gemma4 media tower accuracy")
    output_head_or_tied_embedding = out in {"token_embeddings.weights", "decoder_token_embeddings.weights", "output_weight.weights"}
    if component == "embedding" or output_head_or_tied_embedding:
        layout = "interleaved_4row" if output_head_or_tied_embedding else "row_major"
        return TensorPolicy("convert", "CQ4", 4, component, False, "orthogonal", layout=layout)
    if component == "audio" or component == "transcription":
        return TensorPolicy("convert", f"CQ{user_bits}", user_bits, component, False, "hadamard")
    return TensorPolicy("convert", f"CQ{user_bits}", user_bits, component, True, "hadamard")
