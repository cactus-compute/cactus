from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class WeightBinding:
    path: str
    kind: str  # "weight" | "embedding"
    source_name: str


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _candidate_model_dir_names(model_name_or_path: str) -> list[str]:
    candidates: list[str] = []
    raw = model_name_or_path.strip()
    if not raw:
        return candidates

    def _add(name: str) -> None:
        name = name.strip().lower()
        if name and name not in candidates:
            candidates.append(name)

    _add(raw.split("/")[-1])

    path = Path(raw)
    parts = path.parts
    for part in parts:
        if part.startswith("models--"):
            _add(part[len("models--"):].split("--")[-1])
            break

    return candidates


def _default_weights_dir_for_model_name(model_name_or_path: str) -> str | None:
    if not model_name_or_path:
        return None
    for model_dir_name in _candidate_model_dir_names(model_name_or_path):
        candidate = _PROJECT_ROOT / "weights" / model_dir_name
        if candidate.exists():
            return str(candidate)
    return None


def _normalized_source_candidates(source_name: str) -> list[str]:
    candidates: list[str] = []

    def _add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    raw = source_name.strip()
    _add(raw)

    for prefix in ("p_", "b_", "c_"):
        if raw.startswith(prefix):
            _add(raw[len(prefix):])

    stripped = candidates[-1]

    if stripped.startswith("module.backbone."):
        tail = stripped[len("module.backbone."):]
        _add(f"model.{tail}")
        _add(f"model.language_model.{tail}")
        if tail.startswith("layers."):
            _add(f"model.{tail}")
            _add(f"model.language_model.{tail}")

    layer_match = re.match(
        r"^(?:module_)?backbone_layers_slice_none__\d+__none____modules__(\d+)___(.+)$",
        stripped,
    )
    if layer_match:
        layer_index = int(layer_match.group(1))
        tail = layer_match.group(2)
        tail_map = {
            "self_attn_q_proj_weight": "self_attn.q_proj.weight",
            "self_attn_k_proj_weight": "self_attn.k_proj.weight",
            "self_attn_v_proj_weight": "self_attn.v_proj.weight",
            "self_attn_o_proj_weight": "self_attn.o_proj.weight",
            "self_attn_q_norm_weight": "self_attn.q_norm.weight",
            "self_attn_k_norm_weight": "self_attn.k_norm.weight",
            "mlp_gate_proj_weight": "mlp.gate_proj.weight",
            "mlp_up_proj_weight": "mlp.up_proj.weight",
            "mlp_down_proj_weight": "mlp.down_proj.weight",
            "input_layernorm_weight": "input_layernorm.weight",
            "post_attention_layernorm_weight": "post_attention_layernorm.weight",
            "linear_attn_in_proj_qkv_weight": "linear_attn.in_proj_qkv.weight",
            "linear_attn_conv1d_weight": "linear_attn.conv1d.weight",
            "linear_attn_norm_weight": "linear_attn.norm.weight",
            "linear_attn_dt_bias": "linear_attn.dt_bias",
            "linear_attn_A_log": "linear_attn.A_log",
        }
        dotted_tail = tail_map.get(tail)
        if dotted_tail is not None:
            _add(f"model.layers.{layer_index}.{dotted_tail}")
            _add(f"model.language_model.layers.{layer_index}.{dotted_tail}")

    embed_map = {
        "module_backbone_embed_tokens_weight": [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ],
        "module_backbone_embed_tokens_per_layer_weight": [
            "model.embed_tokens_per_layer.weight",
            "model.language_model.embed_tokens_per_layer.weight",
        ],
        "module_backbone_per_layer_model_projection_weight": [
            "model.per_layer_model_projection.weight",
            "model.language_model.per_layer_model_projection.weight",
        ],
        "module_backbone_norm_weight": [
            "model.norm.weight",
            "model.language_model.norm.weight",
        ],
        "module_model_lm_head_weight": [
            "lm_head.weight",
        ],
    }
    for key, mapped in embed_map.items():
        if stripped == key:
            for item in mapped:
                _add(item)

    return candidates


def resolve_transpile_weights_dir(graph_meta: dict[str, object]) -> str | None:
    explicit = graph_meta.get("weights_dir")
    if isinstance(explicit, str) and explicit:
        return explicit

    family = str(graph_meta.get("adapter_family", "")).upper()
    family_env = f"CACTUS_TRANSPILER_WEIGHTS_DIR_{family}"
    if family and family_env in os.environ and os.environ[family_env]:
        return os.environ[family_env]

    generic = os.environ.get("CACTUS_TRANSPILER_WEIGHTS_DIR")
    if generic:
        return generic

    model_name_or_path = graph_meta.get("model_name_or_path")
    if isinstance(model_name_or_path, str) and model_name_or_path:
        default_dir = _default_weights_dir_for_model_name(model_name_or_path)
        if default_dir:
            return default_dir
    return None


def resolve_weight_binding(*, weights_dir: str | None, source_name: str) -> WeightBinding | None:
    if not weights_dir:
        return None
    root = Path(weights_dir)
    if not root.exists():
        return None
    manifest_path = root / "weights_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None
    for candidate_name in _normalized_source_candidates(source_name):
        entry = manifest.get(candidate_name)
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        kind = entry.get("kind", "weight")
        if not isinstance(filename, str) or not isinstance(kind, str):
            continue
        candidate = root / filename
        if not candidate.exists():
            continue
        return WeightBinding(path=str(candidate), kind=kind, source_name=candidate_name)
    return None
