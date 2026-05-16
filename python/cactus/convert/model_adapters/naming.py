from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..cactus_adapters import weight_patterns as wp


@dataclass(frozen=True)
class NameMatch:
    source_name: str
    output_name: str | None
    component: str
    recognized: bool
    transpose: bool = False
    hf_name: str | None = None
    adapter_name: str | None = None


GEMMA4_WEIGHT_SCALE = 16.0
GEMMA4_MULT_BASENAMES = {"ffn_gate", "ffn_up", "per_layer_gate", "moe_gate_proj", "moe_up_proj"}
GEMMA4_DIV_BASENAMES = {
    "token_embeddings",
    "output_weight",
    "embed_vision_proj",
    "embed_vision_embedding",
}


def remap_gemma4_audio_key(key: str) -> str:
    if "audio_tower" not in key:
        return key
    new_key = re.sub(r"subsample_conv_projection\.layer(\d+)\.", r"subsample_conv_projection.conv_\1.", key)
    new_key = re.sub(r"audio_tower\.layers\.", "audio_tower.conformer.", new_key)
    new_key = new_key.replace(".feed_forward1.", ".ffw_layer_start.")
    new_key = new_key.replace(".feed_forward2.", ".ffw_layer_end.")
    new_key = re.sub(r"\.self_attn\.(q_proj|k_proj|v_proj)\.", r".attention.attn.\1.", new_key)
    new_key = new_key.replace(".self_attn.per_dim_scale", ".attention.attn.per_dim_scale")
    new_key = new_key.replace(".self_attn.relative_k_proj.", ".attention.attn.relative_position_embedding.pos_proj.")
    new_key = new_key.replace(".self_attn.post.", ".attention.post.")
    new_key = new_key.replace(".norm_pre_attn.", ".attention.pre_attn_norm.")
    new_key = new_key.replace(".norm_post_attn.", ".attention.post_norm.")
    new_key = re.sub(r"\.norm_out\.", ".norm.", new_key)
    return new_key


def restore_gemma4_audio_key(key: str) -> str:
    if not key.startswith("model.audio_tower."):
        return key
    new_key = re.sub(r"subsample_conv_projection\.conv_(\d+)\.", r"subsample_conv_projection.layer\1.", key)
    new_key = new_key.replace("model.audio_tower.conformer.", "model.audio_tower.layers.")
    new_key = new_key.replace(".ffw_layer_start.", ".feed_forward1.")
    new_key = new_key.replace(".ffw_layer_end.", ".feed_forward2.")
    new_key = re.sub(r"\.attention\.attn\.([qkv])_proj\.", r".self_attn.\1_proj.", new_key)
    new_key = new_key.replace(".attention.attn.per_dim_scale", ".self_attn.per_dim_scale")
    new_key = new_key.replace(".attention.attn.relative_position_embedding.pos_proj.", ".self_attn.relative_k_proj.")
    new_key = new_key.replace(".attention.post.", ".self_attn.post.")
    new_key = new_key.replace(".attention.pre_attn_norm.", ".norm_pre_attn.")
    new_key = new_key.replace(".attention.post_norm.", ".norm_post_attn.")
    new_key = re.sub(r"^model\.audio_tower\.layers\.(\d+)\.norm\.weight$", r"model.audio_tower.layers.\1.norm_out.weight", new_key)
    return new_key


def remap_whisper_key(key: str) -> str:
    if key.startswith("model.encoder.") or key.startswith("model.decoder."):
        return key[len("model."):]
    return key


def adapter_key_for_family(key: str, family: str) -> str:
    if family == "gemma4":
        return remap_gemma4_audio_key(key)
    if family == "whisper":
        return remap_whisper_key(key)
    return key


def restore_hf_key_for_family(key: str, family: str) -> str:
    if family == "gemma4":
        return restore_gemma4_audio_key(key)
    return key


def gemma4_scale_factor(out_name: str) -> float:
    base = out_name.removesuffix(".weights").removesuffix(".bias")
    parts = base.split("_", 2)
    if len(parts) == 3 and parts[0] == "layer" and parts[1].isdigit():
        base = parts[2]
    if base in GEMMA4_MULT_BASENAMES:
        return GEMMA4_WEIGHT_SCALE
    if base in GEMMA4_DIV_BASENAMES:
        return 1.0 / GEMMA4_WEIGHT_SCALE
    if any(x in out_name for x in [
        "input_norm",
        "post_attn_norm",
        "pre_ffn_norm",
        "post_ffn_norm",
        "post_per_layer_norm",
        "post_proj_norm",
    ]):
        return 1.0 / GEMMA4_WEIGHT_SCALE
    if "router_scale" in out_name:
        return 1.0 / GEMMA4_WEIGHT_SCALE
    if out_name == "output_norm.weights":
        return GEMMA4_WEIGHT_SCALE
    return 1.0


def remap_state_dict_for_family(state_dict: dict, family: str) -> dict:
    if family != "gemma4":
        return state_dict
    return {remap_gemma4_audio_key(k): v for k, v in state_dict.items()}


def _tower_output_name(hf_key: str, strip_prefix: str, add_prefix: str) -> str:
    name = hf_key[len(strip_prefix):]
    if name.endswith(".weight"):
        name = name[:-7]
        ext = ".weights"
    elif name.endswith(".bias"):
        name = name[:-5]
        ext = ".bias"
    else:
        ext = ".weights"
    if name.endswith(".linear"):
        name = name[:-7]
    elif name.endswith("_linear"):
        name = name[:-7]
    return add_prefix + name.replace(".", "_") + ext


def _vision_layer_match(name: str) -> str | None:
    prefixes = (
        "model.vision_tower.vision_model.encoder.layers.",
        "model.vision_model.encoder.layers.",
    )
    for prefix in prefixes:
        m = re.match(rf"^{re.escape(prefix)}(\d+)\.", name)
        if not m:
            continue
        idx = int(m.group(1))
        layer_prefix = f"{prefix}{idx}."
        for source, out in wp.get_vision_layer_weights(idx, layer_prefix):
            if name == source:
                return out
    return None


def _qwen_visual_match(name: str) -> str | None:
    global_map = {
        "model.visual.patch_embed.proj.weight": "vision_patch_embedding.weights",
        "model.visual.patch_embed.proj.bias": "vision_patch_embedding.bias.weights",
        "model.visual.pos_embed.weight": "vision_position_embedding.weights",
        "model.visual.merger.norm.weight": "vision_merger_norm.weights",
        "model.visual.merger.norm.bias": "vision_merger_norm.bias.weights",
        "model.visual.merger.linear_fc1.weight": "vision_merger_linear_fc1.weights",
        "model.visual.merger.linear_fc1.bias": "vision_merger_linear_fc1.bias.weights",
        "model.visual.merger.linear_fc2.weight": "vision_merger_linear_fc2.weights",
        "model.visual.merger.linear_fc2.bias": "vision_merger_linear_fc2.bias.weights",
    }
    if name in global_map:
        return global_map[name]

    m = re.match(r"^model\.visual\.blocks\.(\d+)\.(.+)$", name)
    if m:
        idx = int(m.group(1))
        suffix = m.group(2)
        suffix_map = {
            "norm1.weight": "layer_norm1.weights",
            "norm1.bias": "layer_norm1.bias.weights",
            "norm2.weight": "layer_norm2.weights",
            "norm2.bias": "layer_norm2.bias.weights",
            "mlp.linear_fc1.weight": "ffn_fc1.weights",
            "mlp.linear_fc1.bias": "ffn_fc1.bias.weights",
            "mlp.linear_fc2.weight": "ffn_fc2.weights",
            "mlp.linear_fc2.bias": "ffn_fc2.bias.weights",
            "attn.qkv.weight": "self_attn_qkv.weights",
            "attn.qkv.bias": "self_attn_qkv.bias.weights",
            "attn.proj.weight": "self_attn_out.weights",
            "attn.proj.bias": "self_attn_out.bias.weights",
        }
        out_suffix = suffix_map.get(suffix)
        if out_suffix:
            return f"vision_layer_{idx}_{out_suffix}"

    m = re.match(r"^model\.visual\.deepstack_merger_list\.(\d+)\.(.+)$", name)
    if m:
        idx = int(m.group(1))
        suffix = m.group(2)
        suffix_map = {
            "norm.weight": "norm.weights",
            "norm.bias": "norm.bias.weights",
            "linear_fc1.weight": "linear_fc1.weights",
            "linear_fc1.bias": "linear_fc1.bias.weights",
            "linear_fc2.weight": "linear_fc2.weights",
            "linear_fc2.bias": "linear_fc2.bias.weights",
        }
        out_suffix = suffix_map.get(suffix)
        if out_suffix:
            return f"vision_deepstack_merger_{idx}_{out_suffix}"
    return None


def _candidate_table_match(name: str, table: Iterable) -> str | None:
    for item in table:
        candidates, out_name = item[0], item[1]
        if isinstance(candidates, str):
            candidates = (candidates,)
        if name in candidates:
            return out_name
    return None


def _whisper_layer_match(name: str, i: int) -> tuple[str, bool] | None:
    for block in ("encoder", "decoder"):
        prefix = f"{block}.layers.{i}."
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        for patterns, _precision, out_name, transpose in wp.get_layer_weight_patterns(i, "CQ", model_type="whisper"):
            for pat in patterns:
                if suffix == pat:
                    return f"{block}.{out_name}", transpose
    return None


def _parakeet_layer_match(name: str, i: int) -> str | None:
    prefix = f"encoder.layers.{i}."
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    out_suffix = _candidate_table_match(suffix, wp.PARAKEET_LAYER_WEIGHTS)
    return f"layer_{i}_{out_suffix}" if out_suffix else None


def _parakeet_tdt_predictor_match(name: str) -> str | None:
    old = re.fullmatch(r"decoder\.prediction\.dec_rnn\.lstm\.(\d+)\.(Wx|Wh|bias)", name)
    if old:
        suffix = {"Wx": "weight_ih.weights", "Wh": "weight_hh.weights", "bias": "bias.weights"}[old.group(2)]
        return f"tdt_predictor_lstm_{int(old.group(1))}_{suffix}"
    new = re.fullmatch(r"decoder\.lstm\.(weight_ih|weight_hh|bias)_l(\d+)", name)
    if new:
        suffix = {"weight_ih": "weight_ih.weights", "weight_hh": "weight_hh.weights", "bias": "bias.weights"}[new.group(1)]
        return f"tdt_predictor_lstm_{int(new.group(2))}_{suffix}"
    return None


def _suffix_layer_match(name: str, i: int, family: str) -> tuple[str, bool] | None:
    if family == "whisper":
        return _whisper_layer_match(name, i)
    if family in {"parakeet", "parakeet_tdt"}:
        out = _parakeet_layer_match(name, i)
        if out:
            return out, False
    prefixes = [p.format(i=i) for p in wp.LAYER_PREFIXES] + [f"h.{i}."]
    for p in prefixes:
        if not name.startswith(p):
            continue
        suffix = name[len(p):]
        for patterns, _precision, out_name, transpose in wp.get_layer_weight_patterns(i, "CQ", model_type=family):
            for pat in patterns:
                if "{channel}" in pat:
                    rx = re.escape(pat).replace(r"\{channel\}", r"(\d+)")
                    m = re.fullmatch(rx, suffix)
                    if m:
                        return out_name.format(channel=m.group(1)), transpose
                elif suffix == pat:
                    return out_name, transpose
        if suffix.endswith(".weight"):
            return f"layer_{i}_{suffix[:-7].replace('.', '_')}.weights", False
        if suffix.endswith(".bias"):
            return f"layer_{i}_{suffix[:-5].replace('.', '_')}.bias", False
    return None


def _global_match(name: str, family: str) -> str | None:
    if family == "whisper":
        found = _candidate_table_match(name, wp.WHISPER_GLOBAL_WEIGHTS)
        if found:
            return found
    elif family == "parakeet":
        found = _candidate_table_match(name, wp.PARAKEET_GLOBAL_WEIGHTS)
        if found:
            return found
    elif family == "parakeet_tdt":
        for table in (wp.PARAKEET_GLOBAL_WEIGHTS, wp.PARAKEET_TDT_GLOBAL_WEIGHTS):
            found = _candidate_table_match(name, table)
            if found:
                return found
    if name in {"wte.weight", "word_embeddings.weight"}:
        return "token_embeddings.weights"
    if name in {"wpe.weight", "position_embeddings.weight"}:
        return "position_embeddings.weights"
    if name == "ln_f.weight":
        return "output_norm.weights"
    if name == "ln_f.bias":
        return "output_norm.bias"
    if name in wp.EMBED_NAMES:
        return "token_embeddings.weights"
    if name in wp.OUTPUT_NAMES:
        return "output_weight.weights"
    if name in wp.OUTPUT_NORM_NAMES:
        return "output_norm.weights"
    tables: list[Iterable] = [wp.VISION_ITEMS, wp.PROJECTOR_WEIGHTS]
    if family == "gemma4":
        tables.append(wp.GEMMA4_GLOBAL_WEIGHTS)
    elif family == "gemma3n":
        tables.append(wp.GEMMA3N_GLOBAL_WEIGHTS)
    elif family == "moonshine":
        tables.append(wp.MOONSHINE_GLOBAL_WEIGHTS)
    for table in tables:
        found = _candidate_table_match(name, table)
        if found:
            return found
    if family == "parakeet_tdt":
        predictor_name = _parakeet_tdt_predictor_match(name)
        if predictor_name:
            return predictor_name
    return None


def component_for_name(name: str, output_name: str | None = None) -> str:
    joined = f"{name} {output_name or ''}".lower()
    if "parakeet" in joined or "tdt_" in joined or "ctc_" in joined:
        return "transcription"
    if "audio" in joined or "encoder.conv" in joined or "subsample" in joined:
        return "audio"
    if "vision" in joined or "visual" in joined or "image" in joined or "projector" in joined:
        return "vision"
    if "embed" in joined or "token_embedding" in joined or "lm_head" in joined or "output_weight" in joined:
        return "embedding"
    if "decoder" in joined and ("whisper" in joined or "moonshine" in joined):
        return "transcription"
    return "language"


def _component_for_family(name: str, output_name: str | None, family: str) -> str:
    if family in {"parakeet", "parakeet_tdt"}:
        return "transcription"
    if family == "whisper":
        out = output_name or ""
        if out in {"decoder_token_embeddings.weights", "output_weight.weights"}:
            return "embedding"
        return "transcription"
    return component_for_name(name, output_name)


def cactus_name_for_tensor(name: str, family: str, num_layers: int | None = None) -> NameMatch:
    hf_name = name
    adapter_name = adapter_key_for_family(name, family)
    if family in {"parakeet", "parakeet_tdt"} and adapter_name.endswith(".conv.norm.num_batches_tracked"):
        return NameMatch(name, None, "transcription", True, hf_name=hf_name, adapter_name=adapter_name)
    global_name = _global_match(adapter_name, family)
    if global_name:
        return NameMatch(name, global_name, _component_for_family(hf_name, global_name, family), True, hf_name=hf_name, adapter_name=adapter_name)
    vision_name = _vision_layer_match(adapter_name)
    if vision_name:
        return NameMatch(name, vision_name, "vision", True, hf_name=hf_name, adapter_name=adapter_name)
    if family == "qwen":
        qwen_vision_name = _qwen_visual_match(adapter_name)
        if qwen_vision_name:
            return NameMatch(name, qwen_vision_name, "vision", True, hf_name=hf_name, adapter_name=adapter_name)
    if family == "gemma4":
        if adapter_name.startswith(wp.GEMMA4_VISION_TOWER_PREFIX):
            out = _tower_output_name(adapter_name, wp.GEMMA4_VISION_TOWER_PREFIX, "vision_")
            return NameMatch(name, out, "vision", True, hf_name=hf_name, adapter_name=adapter_name)
        if adapter_name.startswith(wp.GEMMA4_AUDIO_TOWER_PREFIX):
            out = _tower_output_name(adapter_name, wp.GEMMA4_AUDIO_TOWER_PREFIX, "audio_")
            return NameMatch(name, out, "audio", True, hf_name=hf_name, adapter_name=adapter_name)
    max_layers = int(num_layers or 160)
    for i in range(max_layers):
        layer = _suffix_layer_match(adapter_name, i, family)
        if layer:
            out, transpose = layer
            return NameMatch(name, out, _component_for_family(hf_name, out, family), True, transpose, hf_name=hf_name, adapter_name=adapter_name)
    if adapter_name.endswith(".weight") or adapter_name.endswith(".bias"):
        generic = adapter_name.replace(".", "_")
        if generic.endswith("_weight"):
            generic = generic[:-7] + ".weights"
        elif generic.endswith("_bias"):
            generic = generic[:-5] + ".bias"
        return NameMatch(name, generic, component_for_name(hf_name, generic), False, hf_name=hf_name, adapter_name=adapter_name)
    return NameMatch(name, None, component_for_name(hf_name), False, hf_name=hf_name, adapter_name=adapter_name)
