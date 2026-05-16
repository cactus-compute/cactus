from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..cactus_adapters.config_utils import (
    extract_audio_config,
    extract_base_config,
    extract_complex_gemma_config,
    extract_parakeet_config,
    extract_parakeet_tdt_config,
    extract_vision_config,
    extract_whisper_config,
)
from .naming import NameMatch, cactus_name_for_tensor, gemma4_scale_factor, restore_hf_key_for_family
from .policy import TensorPolicy, policy_for_tensor
from ..compat import patch_transformers_import_compat


@dataclass(frozen=True)
class TensorProvenance:
    source_names: list[str]
    transform: str = "none"
    qdq_restore: str = "hf_key"


@dataclass
class NormalizedState:
    state_dict: dict[str, Any]
    provenance: dict[str, TensorProvenance] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorEmission:
    output_name: str
    tensor: Any
    transform: str = "none"
    source_names: list[str] | None = None
    qdq_restore: str | None = None


class FamilyAdapter:
    family = "generic"

    def __init__(self, family: str | None = None) -> None:
        if family is not None:
            self.family = family

    def runtime_config(self, cfg: Any) -> dict[str, Any]:
        text_cfg = _cfg_get(cfg, "text_config", None)
        base_cfg = text_cfg if text_cfg is not None else cfg
        return extract_base_config(base_cfg, cfg)

    def runtime_model_type(self) -> str:
        return self.family

    def model_class(self, cfg: Any):
        patch_transformers_import_compat()
        from transformers import AutoModel, AutoModelForCausalLM

        arch = " ".join(_cfg_get(cfg, "architectures", []) or []).lower()
        if not any(x in arch for x in ["causallm", "conditionalgeneration", "forctc"]):
            return AutoModel
        return AutoModelForCausalLM

    def load_processor(self, model_id_or_path: str):
        patch_transformers_import_compat()
        from transformers import AutoProcessor, AutoTokenizer

        try:
            return AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True, local_files_only=Path(model_id_or_path).exists())
        except Exception:
            try:
                return AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True, local_files_only=Path(model_id_or_path).exists())
            except Exception:
                return None

    def normalize_state_dict(self, state_dict: dict[str, Any]) -> NormalizedState:
        return NormalizedState(state_dict=dict(state_dict))

    def name_tensor(self, source_name: str, _tensor: Any, num_layers: int | None) -> NameMatch:
        return cactus_name_for_tensor(source_name, self.family, num_layers)

    def policy(self, match: NameMatch, shape: tuple[int, ...], requested_bits: int) -> TensorPolicy:
        return policy_for_tensor(match, shape, requested_bits, self.family)

    def transform_tensor(self, match: NameMatch, tensor: Any) -> tuple[Any, str]:
        return tensor, "none"

    def expand_tensor(self, match: NameMatch, tensor: Any) -> list[TensorEmission]:
        if match.output_name is None:
            return []
        tensor, transform = self.transform_tensor(match, tensor)
        if match.transpose:
            tensor = _transpose_last_two(tensor)
            transform = f"{transform}+transpose" if transform != "none" else "transpose"
        return [TensorEmission(match.output_name, tensor, transform)]

    def module_target_name(self, source_name: str, _model: Any) -> str | None:
        return source_name[:-7] if source_name.endswith(".weight") else source_name

    def build_calibration_inputs(self, row: dict[str, Any], processor: Any, modality: str, base_dir) -> dict[str, Any] | None:
        if modality == "transcription":
            return None
        return None

    def qdq_output_keys(self, row: dict[str, Any]) -> list[str]:
        if row.get("qdq_restore") == "runtime_key":
            output_file = row.get("output_file")
            if not output_file:
                raise ValueError(f"manifest row has no runtime output file: {row}")
            key = str(output_file)
            return [key[:-8] if key.endswith(".weights") else key]
        key = row.get("hf_name") or row.get("source_name")
        if not key:
            raise ValueError(f"manifest row has no hf/source key: {row}")
        return [restore_hf_key_for_family(str(key), self.family)]

    def scale_factor(self, output_name: str) -> float:
        return 1.0


class Gemma4Adapter(FamilyAdapter):
    family = "gemma4"

    def runtime_config(self, cfg: Any) -> dict[str, Any]:
        text_cfg = _cfg_get(cfg, "text_config", None)
        base_cfg = text_cfg if text_cfg is not None else cfg
        config = extract_base_config(base_cfg, cfg)
        config.update(extract_complex_gemma_config(base_cfg, cfg))
        vision_cfg = _cfg_get(cfg, "vision_config", None)
        if vision_cfg:
            config.update(extract_vision_config(cfg, vision_cfg))
        audio_cfg = _cfg_get(cfg, "audio_config", None)
        if audio_cfg:
            config.update(extract_audio_config(cfg, audio_cfg))
        return config

    def model_class(self, cfg: Any):
        from transformers import AutoModel

        try:
            from transformers import Gemma4ForConditionalGeneration

            return Gemma4ForConditionalGeneration
        except Exception:
            return AutoModel

    def scale_factor(self, output_name: str) -> float:
        return gemma4_scale_factor(output_name)

    def policy(self, match: NameMatch, shape: tuple[int, ...], requested_bits: int) -> TensorPolicy:
        policy = super().policy(match, shape, requested_bits)
        name = match.source_name
        if name == "model.embed_vision.embedding_projection.weight":
            return replace(policy, use_gptq=False, fallback_reason=policy.fallback_reason or "vision embedding projection has no stable Hessian target")
        if ".self_attn." in name and (name.endswith(".k_proj.weight") or name.endswith(".v_proj.weight")):
            parts = name.split(".")
            try:
                layer = int(parts[3]) if parts[:3] == ["model", "language_model", "layers"] else None
            except Exception:
                layer = None
            if layer is not None and layer >= 15:
                return replace(policy, use_gptq=False, fallback_reason=policy.fallback_reason or "shared KV tensor has no per-layer hook module")
        return policy

    def module_target_name(self, source_name: str, _model: Any) -> str | None:
        target = super().module_target_name(source_name, _model)
        if target and source_name.startswith("model.vision_tower."):
            modules = dict(_model.named_modules()) if _model is not None else {}
            if target in modules and hasattr(modules[target], "weight"):
                return target
            linear_target = target if target.endswith(".linear") else f"{target}.linear"
            if linear_target in modules:
                return linear_target
        return target


class QwenAdapter(FamilyAdapter):
    family = "qwen"

    def model_class(self, cfg: Any):
        from transformers import AutoModel, AutoModelForCausalLM

        arch = " ".join(_cfg_get(cfg, "architectures", []) or []).lower()
        model_type = str(_cfg_get(cfg, "model_type", "") or "").lower().replace("_", "-")
        if "qwen3vlforconditionalgeneration" in arch or model_type == "qwen3-vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration

                return Qwen3VLForConditionalGeneration
            except Exception:
                return AutoModel
        return AutoModelForCausalLM


class WhisperAdapter(FamilyAdapter):
    family = "whisper"

    def model_class(self, cfg: Any):
        from transformers import AutoModelForSpeechSeq2Seq

        return AutoModelForSpeechSeq2Seq

    def runtime_config(self, cfg: Any) -> dict[str, Any]:
        return extract_whisper_config(cfg)

    def build_calibration_inputs(self, row: dict[str, Any], processor: Any, modality: str, base_dir) -> dict[str, Any] | None:
        if modality != "transcription" or processor is None:
            return None
        audio_rel = row.get("audio_path")
        if not audio_rel:
            return None
        audio_path = base_dir / audio_rel
        if not audio_path.exists():
            return None
        audio = _load_audio_16k(audio_path)
        try:
            return processor(audio, sampling_rate=16000, return_tensors="pt")
        except Exception:
            return None


class ParakeetAdapter(FamilyAdapter):
    family = "parakeet"

    def model_class(self, cfg: Any):
        from transformers import AutoModel, AutoModelForCTC

        arch = " ".join(_cfg_get(cfg, "architectures", []) or []).lower()
        model_type = str(_cfg_get(cfg, "model_type", "") or "").lower()
        if "parakeetforctc" in arch or model_type == "parakeet_ctc":
            try:
                return AutoModelForCTC
            except Exception:
                try:
                    from transformers import ParakeetForCTC

                    return ParakeetForCTC
                except Exception:
                    return AutoModel
        return AutoModel

    def runtime_config(self, cfg: Any) -> dict[str, Any]:
        return extract_parakeet_config(cfg)

    def transform_tensor(self, match: NameMatch, tensor: Any) -> tuple[Any, str]:
        out = match.output_name
        if out not in {
            "subsampling_conv0_weight.weights",
            "subsampling_depthwise1_weight.weights",
            "subsampling_pointwise1_weight.weights",
            "subsampling_depthwise2_weight.weights",
            "subsampling_pointwise2_weight.weights",
        }:
            return tensor, "none"
        if len(_tensor_shape(tensor)) != 4:
            return tensor, "none"
        shape = _tensor_shape(tensor)
        is_hwio_k3 = shape[1] == 3 and shape[2] == 3 and shape[3] >= 1
        is_hwio_pw = shape[1] == 1 and shape[2] == 1 and shape[3] > 1
        if not (is_hwio_k3 or is_hwio_pw):
            return tensor, "none"
        if torch is not None and isinstance(tensor, torch.Tensor):
            return tensor.permute(0, 3, 1, 2).contiguous(), "parakeet_conv_hwio_to_oihw"
        return np.transpose(np.asarray(tensor), (0, 3, 1, 2)).copy(), "parakeet_conv_hwio_to_oihw"


class NomicAdapter(FamilyAdapter):
    family = "nomic"

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 0

    def runtime_model_type(self) -> str:
        return "bert"

    def model_class(self, cfg: Any):
        from transformers import AutoModel

        return AutoModel

    def runtime_config(self, cfg: Any) -> dict[str, Any]:
        hidden_dim = int(_cfg_get(cfg, "n_embd", _cfg_get(cfg, "hidden_size", 0)))
        heads = int(_cfg_get(cfg, "n_head", _cfg_get(cfg, "num_attention_heads", 0)))
        self.num_experts = int(_cfg_get(cfg, "num_experts", 0) or 0)
        return {
            "vocab_size": int(_cfg_get(cfg, "vocab_size", 0)),
            "hidden_dim": hidden_dim,
            "num_layers": int(_cfg_get(cfg, "n_layer", _cfg_get(cfg, "num_hidden_layers", 0))),
            "attention_heads": heads,
            "attention_kv_heads": heads,
            "attention_head_dim": int(hidden_dim // max(1, heads)),
            "ffn_intermediate_dim": int(_cfg_get(cfg, "n_inner", _cfg_get(cfg, "intermediate_size", 0))),
            "context_length": int(_cfg_get(cfg, "n_positions", _cfg_get(cfg, "max_position_embeddings", 0))),
            "rope_theta": float(_cfg_get(cfg, "rotary_emb_base", _cfg_get(cfg, "rope_theta", 10000.0)) or 10000.0),
            "layer_norm_eps": float(_cfg_get(cfg, "layer_norm_epsilon", _cfg_get(cfg, "layer_norm_eps", 1e-5)) or 1e-5),
            "num_experts": self.num_experts,
            "num_shared_experts": int(_cfg_get(cfg, "num_shared_experts", 0) or 0),
            "num_top_experts": int(_cfg_get(cfg, "moe_top_k", _cfg_get(cfg, "num_top_experts", 0)) or 0),
            "num_experts_per_tok": int(_cfg_get(cfg, "moe_top_k", _cfg_get(cfg, "num_experts_per_tok", 0)) or 0),
            "moe_every_n_layers": int(_cfg_get(cfg, "moe_every_n_layers", 0) or 0),
            "tie_word_embeddings": True,
        }

    def normalize_state_dict(self, state_dict: dict[str, Any]) -> NormalizedState:
        out = dict(state_dict)
        provenance: dict[str, TensorProvenance] = {}
        word = state_dict.get("embeddings.word_embeddings.weight")
        if word is not None:
            token_type = state_dict.get("embeddings.token_type_embeddings.weight")
            if token_type is not None:
                out["token_embeddings"] = word + token_type
                sources = ["embeddings.word_embeddings.weight", "embeddings.token_type_embeddings.weight"]
                transform = "nomic_word_token_type_embedding_sum"
            else:
                out["token_embeddings"] = word
                sources = ["embeddings.word_embeddings.weight"]
                transform = "nomic_word_embedding_alias"
            out.pop("embeddings.word_embeddings.weight", None)
            out.pop("embeddings.token_type_embeddings.weight", None)
            provenance["token_embeddings"] = TensorProvenance(sources, transform, "adapter_key")
        if "emb_ln.weight" in out:
            out["embedding_layernorm.weight"] = out.pop("emb_ln.weight")
            provenance["embedding_layernorm.weight"] = TensorProvenance(["emb_ln.weight"], "nomic_embedding_layernorm_rename", "adapter_key")
        if "emb_ln.bias" in out:
            out["embedding_layernorm.bias"] = out.pop("emb_ln.bias")
            provenance["embedding_layernorm.bias"] = TensorProvenance(["emb_ln.bias"], "nomic_embedding_layernorm_rename", "adapter_key")
        return NormalizedState(out, provenance)

    def name_tensor(self, source_name: str, tensor: Any, num_layers: int | None) -> NameMatch:
        globals_map = {
            "token_embeddings": "token_embeddings.weights",
            "embedding_layernorm.weight": "embedding_layernorm.weight",
            "embedding_layernorm.bias": "embedding_layernorm.bias",
        }
        if source_name in globals_map:
            return NameMatch(source_name, globals_map[source_name], "embedding", True, hf_name=source_name, adapter_name=source_name)
        norm2 = _nomic_layer_suffix(source_name, ".norm2.weight")
        if norm2 is not None:
            return NameMatch(source_name, f"layer_{norm2}_norm2.weights", "language", True, hf_name=source_name, adapter_name=source_name)
        return cactus_name_for_tensor(source_name, self.family, num_layers)

    def policy(self, match: NameMatch, shape: tuple[int, ...], requested_bits: int) -> TensorPolicy:
        policy = super().policy(match, shape, requested_bits)
        if policy.use_gptq and ".mlp.experts.mlp." in match.source_name:
            return replace(policy, use_gptq=False)
        return policy

    def module_target_name(self, source_name: str, _model: Any) -> str | None:
        if ".mlp.experts.mlp." in source_name:
            return None
        return super().module_target_name(source_name, _model)

    def expand_tensor(self, match: NameMatch, tensor: Any) -> list[TensorEmission]:
        if match.output_name is None:
            return []
        out = match.output_name
        if "{channel}" in out:
            if "attn_{channel}" in out:
                return split_channel_tensor(tensor, out, ["q", "k", "v"], "nomic_qkv_split", match.transpose)
            if "mlp_expert_{channel}" in out:
                num_experts = self.num_experts or 1
                return split_channel_tensor(tensor, out, [str(i) for i in range(num_experts)], "nomic_moe_expert_split", match.transpose)
        return super().expand_tensor(match, tensor)


class ParakeetTDTAdapter(ParakeetAdapter):
    family = "parakeet_tdt"

    def model_class(self, cfg: Any):
        from transformers import AutoModel

        return AutoModel

    def runtime_config(self, cfg: Any) -> dict[str, Any]:
        return extract_parakeet_tdt_config(cfg)

    def normalize_state_dict(self, state_dict: dict[str, Any]) -> NormalizedState:
        augmented = dict(state_dict)
        provenance: dict[str, TensorProvenance] = {}
        i = 0
        while True:
            ih_key = f"decoder.lstm.bias_ih_l{i}"
            hh_key = f"decoder.lstm.bias_hh_l{i}"
            out_key = f"decoder.lstm.bias_l{i}"
            if ih_key not in state_dict and hh_key not in state_dict:
                break
            sources = [k for k in (ih_key, hh_key) if k in state_dict]
            if ih_key in state_dict and hh_key in state_dict:
                augmented[out_key] = state_dict[ih_key] + state_dict[hh_key]
            elif ih_key in state_dict:
                augmented[out_key] = state_dict[ih_key]
            else:
                augmented[out_key] = state_dict[hh_key]
            augmented.pop(ih_key, None)
            augmented.pop(hh_key, None)
            provenance[out_key] = TensorProvenance(sources, "parakeet_tdt_lstm_bias_sum", "hf_key")
            i += 1
        return NormalizedState(augmented, provenance)


class Lfm2Adapter(FamilyAdapter):
    family = "lfm2"

    def model_class(self, cfg: Any):
        from transformers import AutoModelForCausalLM

        arch = " ".join(_cfg_get(cfg, "architectures", []) or []).lower()
        model_type = str(_cfg_get(cfg, "model_type", "") or "").lower().replace("_", "-")
        if "lfm2vlforconditionalgeneration" in arch or model_type == "lfm2-vl":
            try:
                from transformers import Lfm2VlForConditionalGeneration

                return Lfm2VlForConditionalGeneration
            except Exception:
                pass
        return AutoModelForCausalLM

    def load_processor(self, model_id_or_path: str):
        processor = super().load_processor(model_id_or_path)
        if processor is not None and hasattr(processor, "tokenizer"):
            return processor
        root = Path(model_id_or_path)
        if not root.exists() or not (root / "tokenizer.json").exists():
            return None
        from transformers import Lfm2VlImageProcessorFast, Lfm2VlProcessor, PreTrainedTokenizerFast

        cfg_path = root / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "tokenizer.json"),
            bos_token=cfg.get("bos_token", "<|startoftext|>"),
            eos_token=cfg.get("eos_token", "<|im_end|>"),
            pad_token=cfg.get("pad_token", "<|pad|>"),
            clean_up_tokenization_spaces=cfg.get("clean_up_tokenization_spaces", True),
            model_max_length=cfg.get("model_max_length", int(1e30)),
        )
        for key in ("image_token", "image_start_token", "image_end_token", "image_thumbnail"):
            value = cfg.get(key)
            if value is None:
                continue
            setattr(tokenizer, key, value)
            setattr(tokenizer, f"{key}_id", tokenizer.convert_tokens_to_ids(value))
        tokenizer.model_specific_special_tokens = cfg.get("model_specific_special_tokens", {})
        image_processor = Lfm2VlImageProcessorFast.from_pretrained(str(root), local_files_only=True)
        return Lfm2VlProcessor(image_processor=image_processor, tokenizer=tokenizer)

    def policy(self, match: NameMatch, shape: tuple[int, ...], requested_bits: int) -> TensorPolicy:
        policy = super().policy(match, shape, requested_bits)
        if match.source_name.endswith("vision_model.embeddings.patch_embedding.weight"):
            return replace(policy, use_gptq=False, fallback_reason=policy.fallback_reason or "vision patch embedding has no linear Hessian target")
        return policy

    def module_target_name(self, source_name: str, _model: Any) -> str | None:
        if source_name.startswith("model.vision_tower.vision_model."):
            source_name = source_name.replace("model.vision_tower.vision_model.", "model.vision_tower.", 1)
        return super().module_target_name(source_name, _model)


ADAPTERS: dict[str, FamilyAdapter] = {
    "generic": FamilyAdapter(),
    "gemma4": Gemma4Adapter(),
    "qwen": QwenAdapter(),
    "lfm2": Lfm2Adapter(),
    "moonshine": FamilyAdapter("moonshine"),
    "nomic": NomicAdapter(),
    "whisper": WhisperAdapter(),
    "parakeet": ParakeetAdapter(),
    "parakeet_tdt": ParakeetTDTAdapter(),
}


def adapter_for_family(family: str) -> FamilyAdapter:
    return ADAPTERS.get(family, ADAPTERS["generic"])


def _cfg_get(c, key, default=None):
    if c is None:
        return default
    if isinstance(c, dict):
        return c.get(key, default)
    return getattr(c, key, default)


def _tensor_shape(tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def _nomic_layer_suffix(name: str, suffix: str) -> int | None:
    prefix = "encoder.layers."
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix) : -len(suffix)]
    if not middle.isdigit():
        return None
    return int(middle)


def _transpose_last_two(tensor):
    if len(_tensor_shape(tensor)) < 2:
        return tensor
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.transpose(-1, -2).contiguous()
    return np.swapaxes(np.asarray(tensor), -1, -2).copy()


def split_channel_tensor(
    tensor,
    output_template: str,
    channel_names: list[str] | None,
    transform: str,
    transpose: bool = False,
) -> list[TensorEmission]:
    shape = _tensor_shape(tensor)
    if len(shape) not in {1, 2}:
        raise ValueError(f"channel split expects rank 1 or 2 tensor, got {shape}")
    channels = len(channel_names) if channel_names is not None else None
    if channels is None:
        raise ValueError("channel_names are required unless an adapter overrides split behavior")
    if shape[0] % channels != 0:
        raise ValueError(f"first tensor dimension {shape[0]} is not divisible by {channels}")
    if torch is not None and isinstance(tensor, torch.Tensor):
        pieces = tensor.reshape(channels, -1, *shape[1:])
    else:
        pieces = np.asarray(tensor).reshape(channels, -1, *shape[1:])
    emissions = []
    for idx, channel in enumerate(channel_names):
        piece = pieces[idx]
        if transpose:
            piece = _transpose_last_two(piece)
        out = output_template.replace("{channel}", str(channel))
        emissions.append(TensorEmission(out, piece, transform if not transpose else f"{transform}+transpose", qdq_restore="runtime_key"))
    return emissions


def _load_audio_16k(path):
    import soundfile as sf

    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != 16000:
        old_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        new_len = max(1, int(round(audio.shape[0] * 16000 / sr)))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(new_x, old_x, audio).astype(np.float32)
    return audio
