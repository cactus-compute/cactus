from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import math
import struct
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from cactus.transpile.component_pipeline import ComponentModuleSpec
from cactus.convert.cactus_adapters.tensor_io import CACTUS_MAGIC
from cactus.convert.cactus_adapters.tensor_io import align_offset
from cactus.convert.model_adapters.naming import gemma4_scale_factor


_GEMMA4_SAFE_TEXT_MLP_PRODUCT_SCALE = 1.0 / 64.0
_UNSET = object()
try:
    from transformers.models.gemma4.modeling_gemma4 import create_bidirectional_mask as _GEMMA4_CREATE_BIDIRECTIONAL_MASK  # type: ignore
except Exception:
    try:
        from transformers.masking_utils import create_bidirectional_mask as _GEMMA4_CREATE_BIDIRECTIONAL_MASK  # type: ignore
    except Exception:
        _GEMMA4_CREATE_BIDIRECTIONAL_MASK = None


class UnsupportedComponentsError(RuntimeError):
    pass


@dataclass
class CanonicalizedModel:
    module: torch.nn.Module
    task: str
    family: str
    input_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Gemma4NativeMergeSegment:
    kind: str
    input_start: int
    length: int
    feature_start: int = 0


@dataclass(frozen=True)
class _Gemma4NativeMergePlan:
    segments: tuple[_Gemma4NativeMergeSegment, ...]
    pli_token_ids: tuple[int, ...]


def _model_name_or_path(model: torch.nn.Module) -> str:
    value = getattr(model, "name_or_path", None)
    if isinstance(value, str) and value:
        return value
    config = getattr(model, "config", None)
    value = getattr(config, "_name_or_path", None)
    if isinstance(value, str) and value:
        return value
    return ""


def _transpile_graph_meta(model: torch.nn.Module, *, adapter_family: str, adapter_type: str, input_names: tuple[str, ...]) -> dict[str, object]:
    meta: dict[str, object] = {
        "adapter_family": adapter_family,
        "adapter_type": adapter_type,
        "model_name_or_path": _model_name_or_path(model),
        "input_names": input_names,
    }
    # Rope-table precompute falls back to this on components that don't reserve a KV cache.
    for candidate in _config_candidates_for_context_length(model):
        for key in ("max_position_embeddings", "context_length", "model_max_length", "n_positions"):
            value = _config_int_value(candidate, key)
            if value is not None:
                meta["max_position_embeddings"] = value
                return meta
    return meta


def _extract_tensor_output(output: object, *, preferred_field: str | None = None) -> torch.Tensor:
    if preferred_field is not None:
        value = getattr(output, preferred_field, None)
        if isinstance(value, torch.Tensor):
            return value

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]

    for field_name in ("last_hidden_state", "logits"):
        value = getattr(output, field_name, None)
        if isinstance(value, torch.Tensor):
            return value

    raise TypeError(f"could not extract tensor output from {type(output).__name__}")


def _gemma4_get_placeholder_masks(
    get_placeholder_mask: Callable[..., object],
    *,
    token_type_ids: torch.Tensor | None,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
    }
    try:
        signature = inspect.signature(get_placeholder_mask)
    except (TypeError, ValueError):
        signature = None
    if token_type_ids is not None and signature is not None:
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "token_type_ids" in parameters:
            kwargs["token_type_ids"] = token_type_ids

    accepts_token_type_ids = "token_type_ids" in kwargs
    result = get_placeholder_mask(**kwargs)
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError(
            "Gemma4 get_placeholder_mask must return "
            "three placeholder masks"
        )
    first_mask, second_mask, third_mask = result
    if not (
        isinstance(first_mask, torch.Tensor)
        and isinstance(second_mask, torch.Tensor)
        and isinstance(third_mask, torch.Tensor)
    ):
        raise TypeError("Gemma4 get_placeholder_mask returned non-tensor masks")
    if accepts_token_type_ids:
        text_mask, image_mask, audio_mask = first_mask, second_mask, third_mask
    else:
        image_mask, video_mask, audio_mask = first_mask, second_mask, third_mask
        text_mask = ~(image_mask | video_mask | audio_mask)
    return text_mask, image_mask, audio_mask


def _module_or_config_attr(module: object, name: str, default: object | None = None) -> object:
    value = getattr(module, name, None)
    if value is not None:
        return value
    config = getattr(module, "config", None)
    value = getattr(config, name, None)
    if value is not None:
        return value
    return default


def _gemma4_get_per_layer_inputs(
    backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    get_per_layer_inputs = getattr(backbone, "get_per_layer_inputs")
    try:
        signature = inspect.signature(get_per_layer_inputs)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "inputs_embeds" not in signature.parameters:
        return get_per_layer_inputs(input_ids)
    return get_per_layer_inputs(input_ids, inputs_embeds)


def _select_last_active_token(hidden_or_logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None or hidden_or_logits.ndim < 3:
        return hidden_or_logits[:, -1:, ...]

    if attention_mask.ndim != 2:
        raise ValueError(f"expected 2D attention mask, got shape {tuple(attention_mask.shape)}")
    if attention_mask.shape[1] != hidden_or_logits.shape[1]:
        raise ValueError(
            "attention mask / hidden sequence length mismatch: "
            f"{tuple(attention_mask.shape)} vs {tuple(hidden_or_logits.shape)}"
        )

    trailing_zero = attention_mask[:, :1] - attention_mask[:, :1]
    shifted_mask = torch.cat((attention_mask[:, 1:], trailing_zero), dim=1)
    last_active_mask = torch.logical_and(attention_mask != 0, shifted_mask == 0)
    expanded_mask = last_active_mask.to(dtype=hidden_or_logits.dtype)
    for _ in range(hidden_or_logits.ndim - 2):
        expanded_mask = expanded_mask.unsqueeze(-1)
    return (hidden_or_logits * expanded_mask).sum(dim=1, keepdim=True)


def _select_last_non_pad_token(
    hidden_or_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    *,
    pad_token_id: int | None,
) -> torch.Tensor:
    if hidden_or_logits.ndim < 3:
        return hidden_or_logits
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return hidden_or_logits[:, -1:, ...]
    if input_ids.shape[1] != hidden_or_logits.shape[1]:
        raise ValueError(
            "input id / hidden sequence length mismatch: "
            f"{tuple(input_ids.shape)} vs {tuple(hidden_or_logits.shape)}"
        )
    if pad_token_id is None:
        return hidden_or_logits[:, -1:, ...]
    attention_mask = (input_ids != int(pad_token_id)).to(dtype=torch.int64)
    return _select_last_active_token(hidden_or_logits, attention_mask)


def _tile_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
    reps = -(-length // int(tensor.shape[1]))
    return torch.cat([tensor] * reps, dim=1)[:, :length].contiguous()


def _resolve_model_pad_token_id(model: torch.nn.Module) -> int | None:
    def _coerce(value: object) -> int | None:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    config = getattr(model, "config", None)
    for attr_name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        resolved = _coerce(getattr(config, attr_name, None))
        if resolved is not None:
            return resolved
    generation_config = getattr(model, "generation_config", None)
    for attr_name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        resolved = _coerce(getattr(generation_config, attr_name, None))
        if resolved is not None:
            return resolved
    return None


def _filter_supported_kwargs(module: torch.nn.Module, kwargs: dict[str, object]) -> dict[str, object]:
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return kwargs
    accepted = signature.parameters
    supports_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in accepted.values()
    )
    if supports_var_kwargs:
        return kwargs
    return {name: value for name, value in kwargs.items() if name in accepted}


def _module_floating_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return None


def _module_device(module: torch.nn.Module) -> torch.device | None:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return None


def _module_has_meta_tensors(module: torch.nn.Module) -> bool:
    return any(parameter.is_meta for parameter in module.parameters()) or any(
        buffer.is_meta for buffer in module.buffers()
    )


def _to_meta_example_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or tensor.is_meta:
        return tensor
    return torch.empty_strided(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        dtype=tensor.dtype,
        device="meta",
    )


def _unique_modules(*candidates: object) -> tuple[torch.nn.Module, ...]:
    modules: list[torch.nn.Module] = []
    seen: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, torch.nn.Module):
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        modules.append(candidate)
    return tuple(modules)


def _gemma4_vision_modules(multimodal_backbone: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    return _unique_modules(
        getattr(multimodal_backbone, "vision_tower", None),
        getattr(multimodal_backbone, "embed_vision", None),
    )


def _gemma4_audio_modules(multimodal_backbone: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    return _unique_modules(
        getattr(multimodal_backbone, "audio_tower", None),
        getattr(multimodal_backbone, "embed_audio", None),
    )


def _gemma4_text_config(multimodal_backbone: torch.nn.Module) -> object:
    config = getattr(multimodal_backbone, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config()
        except Exception:
            pass
    return config


def _parse_cache_context_length(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto"}:
            return None
        parsed = int(normalized)
    else:
        parsed = int(value)
    if parsed <= 0:
        raise ValueError("--cache-context-length must be a positive integer or auto")
    return parsed


def _config_int_value(config: object, key: str) -> int | None:
    if config is None:
        return None
    if isinstance(config, dict):
        raw_value = config.get(key)
    else:
        raw_value = getattr(config, key, None)
    if raw_value is None:
        return None
    parsed = int(raw_value)
    return parsed if parsed > 0 else None


def _config_candidates_for_context_length(model: torch.nn.Module) -> tuple[object, ...]:
    candidates: list[object] = []
    seen: set[int] = set()

    def _add(candidate: object) -> None:
        if candidate is None:
            return
        candidate_id = id(candidate)
        if candidate_id in seen:
            return
        seen.add(candidate_id)
        candidates.append(candidate)

    for owner in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ):
        config = getattr(owner, "config", None)
        _add(config)
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            _add(get_text_config())
        for attr in ("text_config", "llm_config", "language_config", "decoder_config"):
            _add(getattr(config, attr, None))

    return tuple(candidates)


def _cache_context_length(
    model: torch.nn.Module,
    *,
    input_seq_len: int,
    cache_context_length: str | int | None,
    fallback_extra_tokens: int,
) -> int:
    explicit_length = _parse_cache_context_length(cache_context_length)
    if explicit_length is not None:
        return explicit_length

    for candidate in _config_candidates_for_context_length(model):
        for key in ("max_position_embeddings", "context_length", "model_max_length", "n_positions"):
            value = _config_int_value(candidate, key)
            if value is not None:
                return value
    return input_seq_len + int(fallback_extra_tokens)


def _max_cache_seq_len(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache_context_length: str | int | None,
    *,
    fallback_extra_tokens: int,
) -> int:
    return max(1024, _cache_context_length(
        model,
        input_seq_len=int(input_ids.shape[1]),
        cache_context_length=cache_context_length,
        fallback_extra_tokens=fallback_extra_tokens,
    ))


def _gemma4_special_token_ids(multimodal_backbone: torch.nn.Module) -> tuple[int, int, int]:
    config = getattr(multimodal_backbone, "config", None)
    text_config = _gemma4_text_config(multimodal_backbone)
    image_token_id = int(getattr(config, "image_token_id", 0) or 0)
    audio_token_id = int(getattr(config, "audio_token_id", 0) or 0)
    pad_token_id = getattr(text_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(config, "pad_token_id", 0)
    return image_token_id, audio_token_id, int(pad_token_id or 0)


@contextmanager
def _temporary_cpu_float32_modules(modules: tuple[torch.nn.Module, ...]):
    promoted: list[tuple[torch.nn.Module, torch.dtype]] = []
    for module in modules:
        device = _module_device(module)
        dtype = _module_floating_dtype(module)
        if device is None or device.type != "cpu" or dtype is None or dtype == torch.float32:
            continue
        promoted.append((module, dtype))
        module.to(dtype=torch.float32)
    try:
        yield
    finally:
        for module, dtype in reversed(promoted):
            module.to(dtype=dtype)


def _torch_is_compiling() -> bool:
    dynamo = getattr(torch, "_dynamo", None)
    is_compiling = getattr(dynamo, "is_compiling", None)
    if callable(is_compiling):
        try:
            return bool(is_compiling())
        except Exception:
            return False
    return False


_CACTUS_TORCH_LIBRARIES: list[object] = []
_LFM2_MOE_CUSTOM_OP_REGISTERED = False
_LFM2_MOE_CUSTOM_OP = "cactus_transpile::lfm2_moe_layer_gated"
_QWEN2_MOE_CUSTOM_OP_REGISTERED = False
_QWEN2_MOE_CUSTOM_OP = "cactus_transpile::qwen2_moe_layer_gated"
_GEMMA4_MOE_CUSTOM_OP_REGISTERED = False
_GEMMA4_MOE_CUSTOM_OP = "cactus_transpile::gemma4_moe_layer_gated"


def _ignore_duplicate_torch_registration(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "already" in text or "duplicate" in text or "previously" in text


def _moe_normalize_and_scale(routing_weights, normalize_routing, epsilon, routed_scaling_factor):
    if normalize_routing:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + float(epsilon))
    return routing_weights * float(routed_scaling_factor)


def _moe_scatter_experts(hidden, selected_experts, routing_weights, w1_weights, w3_weights, w2_weights, num_experts, activation):
    output = torch.zeros_like(hidden)
    for expert_idx in range(int(num_experts)):
        selected = selected_experts == expert_idx
        if not bool(selected.any()):
            continue
        token_idx, topk_pos = torch.where(selected)
        current = hidden[token_idx]
        gate = F.linear(current, w1_weights[expert_idx])
        up = F.linear(current, w3_weights[expert_idx])
        expert_out = F.linear(activation(gate) * up, w2_weights[expert_idx])
        expert_out = expert_out * routing_weights[token_idx, topk_pos, None]
        output.index_add_(0, token_idx, expert_out.to(dtype=output.dtype))
    return output


def _gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def _register_moe_custom_op(schema, op_name, impl, fake):
    try:
        library = torch.library.Library("cactus_transpile", "FRAGMENT")
        library.define(schema)
        _CACTUS_TORCH_LIBRARIES.append(library)
    except RuntimeError as exc:
        if not _ignore_duplicate_torch_registration(exc):
            raise
    try:
        torch.library.impl(op_name, "CompositeExplicitAutograd")(impl)
    except RuntimeError as exc:
        if not _ignore_duplicate_torch_registration(exc):
            raise
    try:
        torch.library.register_fake(op_name)(fake)
    except RuntimeError as exc:
        if not _ignore_duplicate_torch_registration(exc):
            raise


def _ensure_lfm2_moe_custom_op_registered() -> None:
    global _LFM2_MOE_CUSTOM_OP_REGISTERED
    if _LFM2_MOE_CUSTOM_OP_REGISTERED:
        return

    schema = (
        "lfm2_moe_layer_gated("
        "Tensor hidden, Tensor router_logits, Tensor expert_bias, "
        "Tensor[] w1_weights, Tensor[] w3_weights, Tensor[] w2_weights, "
        "int num_experts, int num_experts_per_tok, bool use_expert_bias, "
        "bool normalize_routing, float epsilon, float routed_scaling_factor"
        ") -> Tensor"
    )

    def _impl(hidden, router_logits, expert_bias, w1_weights, w3_weights, w2_weights,
              num_experts, num_experts_per_tok, use_expert_bias, normalize_routing,
              epsilon, routed_scaling_factor):
        routing_probs = router_logits.sigmoid()
        if use_expert_bias:
            scores = routing_probs + expert_bias.to(device=routing_probs.device, dtype=routing_probs.dtype)
            _, selected_experts = torch.topk(scores, k=int(num_experts_per_tok), dim=-1)
            routing_weights = torch.gather(routing_probs, dim=1, index=selected_experts).type_as(router_logits)
        else:
            routing_weights, selected_experts = torch.topk(routing_probs, k=int(num_experts_per_tok), dim=-1)
        routing_weights = _moe_normalize_and_scale(routing_weights, normalize_routing, epsilon, routed_scaling_factor)
        return _moe_scatter_experts(hidden, selected_experts, routing_weights, w1_weights, w3_weights, w2_weights, num_experts, F.silu)

    def _fake(hidden, router_logits, expert_bias, w1_weights, w3_weights, w2_weights,
              num_experts, num_experts_per_tok, use_expert_bias, normalize_routing,
              epsilon, routed_scaling_factor):
        return hidden.new_empty(hidden.shape)

    _register_moe_custom_op(schema, _LFM2_MOE_CUSTOM_OP, _impl, _fake)
    _LFM2_MOE_CUSTOM_OP_REGISTERED = True


def _ensure_qwen2_moe_custom_op_registered() -> None:
    global _QWEN2_MOE_CUSTOM_OP_REGISTERED
    if _QWEN2_MOE_CUSTOM_OP_REGISTERED:
        return

    schema = (
        "qwen2_moe_layer_gated("
        "Tensor hidden, Tensor router_logits, "
        "Tensor[] w1_weights, Tensor[] w3_weights, Tensor[] w2_weights, "
        "int num_experts, int num_experts_per_tok, bool normalize_routing, "
        "float epsilon, float routed_scaling_factor"
        ") -> Tensor"
    )

    def _impl(hidden, router_logits, w1_weights, w3_weights, w2_weights,
              num_experts, num_experts_per_tok, normalize_routing, epsilon, routed_scaling_factor):
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(dtype=router_logits.dtype)
        routing_weights, selected_experts = torch.topk(routing_probs, k=int(num_experts_per_tok), dim=-1)
        routing_weights = _moe_normalize_and_scale(routing_weights, normalize_routing, epsilon, routed_scaling_factor)
        return _moe_scatter_experts(hidden, selected_experts, routing_weights, w1_weights, w3_weights, w2_weights, num_experts, F.silu)

    def _fake(hidden, router_logits, w1_weights, w3_weights, w2_weights,
              num_experts, num_experts_per_tok, normalize_routing, epsilon, routed_scaling_factor):
        return hidden.new_empty(hidden.shape)

    _register_moe_custom_op(schema, _QWEN2_MOE_CUSTOM_OP, _impl, _fake)
    _QWEN2_MOE_CUSTOM_OP_REGISTERED = True


def _ensure_gemma4_moe_custom_op_registered() -> None:
    global _GEMMA4_MOE_CUSTOM_OP_REGISTERED
    if _GEMMA4_MOE_CUSTOM_OP_REGISTERED:
        return

    schema = (
        "gemma4_moe_layer_gated("
        "Tensor hidden, Tensor router_logits, "
        "Tensor[] w1_weights, Tensor[] w3_weights, Tensor[] w2_weights, "
        "int num_experts, int num_experts_per_tok, bool normalize_routing, "
        "float epsilon, float routed_scaling_factor"
        ") -> Tensor"
    )

    def _impl(hidden, router_logits, w1_weights, w3_weights, w2_weights,
              num_experts, num_experts_per_tok, normalize_routing, epsilon, routed_scaling_factor):
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(dtype=router_logits.dtype)
        selected_experts = torch.topk(router_logits, k=int(num_experts_per_tok), dim=-1).indices
        routing_weights = torch.gather(routing_probs, -1, selected_experts)
        routing_weights = _moe_normalize_and_scale(routing_weights, normalize_routing, epsilon, routed_scaling_factor)
        return _moe_scatter_experts(hidden, selected_experts, routing_weights, w1_weights, w3_weights, w2_weights, num_experts, _gelu_tanh)

    def _fake(hidden, router_logits, w1_weights, w3_weights, w2_weights,
              num_experts, num_experts_per_tok, normalize_routing, epsilon, routed_scaling_factor):
        return hidden.new_empty(hidden.shape)

    _register_moe_custom_op(schema, _GEMMA4_MOE_CUSTOM_OP, _impl, _fake)
    _GEMMA4_MOE_CUSTOM_OP_REGISTERED = True


class _Lfm2MoeRuntimeMoeBlock(torch.nn.Module):

    def __init__(self, block: torch.nn.Module, *, layer_index: int):
        super().__init__()
        _ensure_lfm2_moe_custom_op_registered()
        experts = getattr(block, "experts", None)
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        gate = getattr(block, "gate", None)
        if not isinstance(gate, torch.nn.Module) or not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
            raise TypeError("LFM2-MoE runtime block requires gate, gate_up_proj, and down_proj")

        self.layer_index = int(layer_index)
        self.gate = gate
        self.num_experts = int(getattr(block, "num_experts", int(gate_up.shape[0])) or int(gate_up.shape[0]))
        self.top_k = int(getattr(block, "top_k", 0) or 0)
        self.use_expert_bias = bool(getattr(block, "use_expert_bias", False))
        self.normalize_routing = bool(getattr(block, "norm_topk_prob", False))
        self.routed_scaling_factor = float(getattr(block, "routed_scaling_factor", 1.0) or 1.0)
        self.epsilon = 1.0e-6
        self.hidden_dim = int(gate_up.shape[2])
        self.intermediate_dim = int(gate_up.shape[1]) // 2

        if self.num_experts <= 0 or self.top_k <= 0:
            raise ValueError("LFM2-MoE runtime block requires positive num_experts and top_k")
        if int(gate_up.shape[0]) != self.num_experts or int(down.shape[0]) != self.num_experts:
            raise ValueError("LFM2-MoE expert tensor count mismatch")
        if int(gate_up.shape[1]) != 2 * self.intermediate_dim:
            raise ValueError("LFM2-MoE gate_up_proj must have an even expert dimension")

        self.w1_weights = torch.nn.ParameterList()
        self.w3_weights = torch.nn.ParameterList()
        self.w2_weights = torch.nn.ParameterList()
        for expert_idx in range(self.num_experts):
            self.w1_weights.append(torch.nn.Parameter(
                gate_up[expert_idx, : self.intermediate_dim, :].detach().contiguous(),
                requires_grad=False,
            ))
            self.w3_weights.append(torch.nn.Parameter(
                gate_up[expert_idx, self.intermediate_dim :, :].detach().contiguous(),
                requires_grad=False,
            ))
            self.w2_weights.append(torch.nn.Parameter(down[expert_idx].detach().contiguous(), requires_grad=False))

        expert_bias = getattr(block, "expert_bias", None)
        if self.use_expert_bias and isinstance(expert_bias, torch.Tensor):
            bias_value = expert_bias.detach().to(dtype=torch.float32)
        else:
            bias_value = torch.zeros(self.num_experts, dtype=torch.float32, device=gate_up.device)
        self.register_buffer("expert_bias", bias_value, persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(hidden_flat)
        output = torch.ops.cactus_transpile.lfm2_moe_layer_gated(
            hidden_flat,
            router_logits,
            self.expert_bias,
            list(self.w1_weights),
            list(self.w3_weights),
            list(self.w2_weights),
            self.num_experts,
            self.top_k,
            self.use_expert_bias,
            self.normalize_routing,
            self.epsilon,
            self.routed_scaling_factor,
        )
        return output.reshape(batch_size, sequence_length, hidden_dim)


def _is_lfm2_moe_sparse_block(block: object) -> bool:
    experts = getattr(block, "experts", None)
    return (
        isinstance(block, torch.nn.Module)
        and isinstance(getattr(block, "gate", None), torch.nn.Module)
        and isinstance(getattr(experts, "gate_up_proj", None), torch.Tensor)
        and isinstance(getattr(experts, "down_proj", None), torch.Tensor)
    )


def _is_lfm2_moe_model(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type == "lfm2_moe":
        return True
    return type(model).__module__.startswith("transformers.models.lfm2_moe.")


def _lfm2_backbone_from_model(model: torch.nn.Module) -> torch.nn.Module | None:
    model_root = getattr(model, "model", None)
    language_model = getattr(model_root, "language_model", None)
    backbone = language_model if isinstance(language_model, torch.nn.Module) else model_root
    return backbone if isinstance(backbone, torch.nn.Module) else None


def _lfm2_position_embeddings(
    backbone: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor,
) -> object:
    rotary_emb = getattr(backbone, "rotary_emb", None)
    if callable(rotary_emb):
        return rotary_emb(hidden_states, position_ids=position_ids)
    pos_emb = getattr(backbone, "pos_emb", None)
    if callable(pos_emb):
        return pos_emb(hidden_states, position_ids=position_ids)
    raise AttributeError(f"{type(backbone).__name__} has neither rotary_emb nor pos_emb")


def _lfm2_causal_mask_for_capture(
    create_causal_mask: Callable[..., object],
    *,
    backbone: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> object | None:
    if inputs_embeds.is_meta:
        return None
    return create_causal_mask(
        config=backbone.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )


def _patch_lfm2_moe_feed_forwards(model: torch.nn.Module) -> None:
    if not _is_lfm2_moe_model(model):
        return
    backbone = _lfm2_backbone_from_model(model)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        return
    for layer_index, layer in enumerate(layers):
        feed_forward = getattr(layer, "feed_forward", None)
        if isinstance(feed_forward, _Lfm2MoeRuntimeMoeBlock):
            continue
        if _is_lfm2_moe_sparse_block(feed_forward):
            setattr(layer, "feed_forward", _Lfm2MoeRuntimeMoeBlock(feed_forward, layer_index=layer_index))


class _Qwen2MoeRuntimeMoeBlock(torch.nn.Module):

    def __init__(self, block: torch.nn.Module, *, layer_index: int):
        super().__init__()
        _ensure_qwen2_moe_custom_op_registered()
        experts = getattr(block, "experts", None)
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        gate = getattr(block, "gate", None)
        gate_weight = getattr(gate, "weight", None)
        shared_expert = getattr(block, "shared_expert", None)
        shared_expert_gate = getattr(block, "shared_expert_gate", None)
        if (
            not isinstance(gate, torch.nn.Module)
            or not isinstance(gate_weight, torch.Tensor)
            or not isinstance(gate_up, torch.Tensor)
            or not isinstance(down, torch.Tensor)
            or not isinstance(shared_expert, torch.nn.Module)
            or not isinstance(shared_expert_gate, torch.nn.Module)
        ):
            raise TypeError(
                "Qwen2-MoE runtime block requires gate, packed experts, shared_expert, and shared_expert_gate"
            )

        self.layer_index = int(layer_index)
        self.gate = gate
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        config = getattr(block, "config", None)
        def _route_attr(*names, default=None):
            for obj in (block, config):
                if obj is None:
                    continue
                for name in names:
                    val = getattr(obj, name, None)
                    if val is not None:
                        return val
            return default
        self.num_experts = int(_route_attr("num_experts", default=int(gate_up.shape[0])) or int(gate_up.shape[0]))
        self.top_k = int(_route_attr("top_k", "num_experts_per_tok", default=0) or 0)
        self.normalize_routing = bool(_route_attr("norm_topk_prob", default=False))
        self.routed_scaling_factor = float(getattr(block, "routed_scaling_factor", 1.0) or 1.0)
        self.epsilon = 1.0e-6
        self.hidden_dim = int(gate_up.shape[2])
        self.intermediate_dim = int(gate_up.shape[1]) // 2

        if self.num_experts <= 0 or self.top_k <= 0:
            raise ValueError("Qwen2-MoE runtime block requires positive num_experts and top_k")
        if int(gate_up.shape[0]) != self.num_experts or int(down.shape[0]) != self.num_experts:
            raise ValueError("Qwen2-MoE expert tensor count mismatch")
        if int(gate_up.shape[1]) != 2 * self.intermediate_dim:
            raise ValueError("Qwen2-MoE gate_up_proj must have an even expert dimension")

        self.w1_weights = torch.nn.ParameterList()
        self.w3_weights = torch.nn.ParameterList()
        self.w2_weights = torch.nn.ParameterList()
        for expert_idx in range(self.num_experts):
            self.w1_weights.append(torch.nn.Parameter(
                gate_up[expert_idx, : self.intermediate_dim, :].detach().contiguous(),
                requires_grad=False,
            ))
            self.w3_weights.append(torch.nn.Parameter(
                gate_up[expert_idx, self.intermediate_dim :, :].detach().contiguous(),
                requires_grad=False,
            ))
            self.w2_weights.append(torch.nn.Parameter(down[expert_idx].detach().contiguous(), requires_grad=False))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = F.linear(hidden_flat, self.gate.weight)
        routed_output = torch.ops.cactus_transpile.qwen2_moe_layer_gated(
            hidden_flat,
            router_logits,
            list(self.w1_weights),
            list(self.w3_weights),
            list(self.w2_weights),
            self.num_experts,
            self.top_k,
            self.normalize_routing,
            self.epsilon,
            self.routed_scaling_factor,
        )
        shared_output = self.shared_expert(hidden_flat)
        shared_output = torch.sigmoid(self.shared_expert_gate(hidden_flat)) * shared_output
        output = routed_output + shared_output
        return output.reshape(batch_size, sequence_length, hidden_dim)


def _is_qwen2_moe_sparse_block(block: object) -> bool:
    experts = getattr(block, "experts", None)
    gate = getattr(block, "gate", None)
    return (
        isinstance(block, torch.nn.Module)
        and isinstance(gate, torch.nn.Module)
        and isinstance(getattr(gate, "weight", None), torch.Tensor)
        and isinstance(getattr(experts, "gate_up_proj", None), torch.Tensor)
        and isinstance(getattr(experts, "down_proj", None), torch.Tensor)
        and isinstance(getattr(block, "shared_expert", None), torch.nn.Module)
        and isinstance(getattr(block, "shared_expert_gate", None), torch.nn.Module)
    )


def _is_qwen2_moe_model(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type == "qwen2_moe":
        return True
    return type(model).__module__.startswith("transformers.models.qwen2_moe.")


def _qwen2_moe_backbone_from_model(model: torch.nn.Module) -> torch.nn.Module | None:
    backbone = getattr(model, "model", None)
    return backbone if isinstance(backbone, torch.nn.Module) else None


def _patch_qwen2_moe_mlps(model: torch.nn.Module) -> None:
    if not _is_qwen2_moe_model(model):
        return
    backbone = _qwen2_moe_backbone_from_model(model)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        return
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, _Qwen2MoeRuntimeMoeBlock):
            continue
        if _is_qwen2_moe_sparse_block(mlp):
            setattr(layer, "mlp", _Qwen2MoeRuntimeMoeBlock(mlp, layer_index=layer_index))


class _Gemma4RuntimeMoeBlock(torch.nn.Module):

    def __init__(
        self,
        block: torch.nn.Module,
        *,
        layer_index: int,
        per_expert_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        _ensure_gemma4_moe_custom_op_registered()
        gate_up = getattr(block, "gate_up_proj", None)
        down = getattr(block, "down_proj", None)
        if per_expert_scale is None:
            per_expert_scale = getattr(block, "per_expert_scale", None)
        if (
            not isinstance(gate_up, torch.Tensor)
            or not isinstance(down, torch.Tensor)
            or not isinstance(per_expert_scale, torch.Tensor)
        ):
            raise TypeError("Gemma4 MoE runtime block requires gate_up_proj, down_proj, and per_expert_scale")

        self.layer_index = int(layer_index)
        self.num_experts = int(getattr(block, "num_experts", int(gate_up.shape[0])) or int(gate_up.shape[0]))
        self.hidden_dim = int(gate_up.shape[2])
        self.intermediate_dim = int(gate_up.shape[1]) // 2
        if self.num_experts <= 0:
            raise ValueError("Gemma4 MoE runtime block requires positive num_experts")
        if int(gate_up.shape[0]) != self.num_experts or int(down.shape[0]) != self.num_experts:
            raise ValueError("Gemma4 MoE expert tensor count mismatch")
        if int(gate_up.shape[1]) != 2 * self.intermediate_dim:
            raise ValueError("Gemma4 MoE gate_up_proj must have an even expert dimension")

        self.w1_weights = torch.nn.ParameterList()
        self.w3_weights = torch.nn.ParameterList()
        self.w2_weights = torch.nn.ParameterList()
        for expert_idx in range(self.num_experts):
            self.w1_weights.append(torch.nn.Parameter(
                gate_up[expert_idx, : self.intermediate_dim, :].detach().contiguous(),
                requires_grad=False,
            ))
            self.w3_weights.append(torch.nn.Parameter(
                gate_up[expert_idx, self.intermediate_dim :, :].detach().contiguous(),
                requires_grad=False,
            ))
            scale = per_expert_scale[expert_idx].detach().to(device=down.device, dtype=down.dtype)
            self.w2_weights.append(torch.nn.Parameter(
                (down[expert_idx].detach() * scale).contiguous(),
                requires_grad=False,
            ))

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        top_k: int,
        normalize_routing: bool = True,
        epsilon: float = 1.0e-6,
    ) -> torch.Tensor:
        return torch.ops.cactus_transpile.gemma4_moe_layer_gated(
            hidden_states,
            router_logits,
            list(self.w1_weights),
            list(self.w3_weights),
            list(self.w2_weights),
            self.num_experts,
            int(top_k),
            bool(normalize_routing),
            float(epsilon),
            1.0,
        )


class _Gemma4MoeRuntimeDecoderLayer(torch.nn.Module):
    """Gemma4 decoder layer that preserves HF math while exporting MoE as one semantic op."""

    def __init__(self, layer: torch.nn.Module, *, layer_index: int):
        super().__init__()
        self.config = layer.config
        self.hidden_size = layer.hidden_size
        self.layer_idx = layer.layer_idx
        self.self_attn = layer.self_attn
        config_layer_types = tuple(getattr(self.config, "layer_types", ()))
        self.attention_type = getattr(layer, "attention_type", getattr(self.self_attn, "attention_type", None))
        if self.attention_type is None:
            self.attention_type = (
                config_layer_types[layer_index]
                if layer_index < len(config_layer_types)
                else "full_attention"
            )
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.register_buffer("layer_scalar", layer.layer_scalar.detach().clone(), persistent=True)
        self.hidden_size_per_layer_input = getattr(layer, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            self.act_fn = layer.act_fn
            self.per_layer_input_gate = layer.per_layer_input_gate
            self.per_layer_projection = layer.per_layer_projection
            self.post_per_layer_input_norm = layer.post_per_layer_input_norm
        self.enable_moe_block = True
        self.router = layer.router
        self.moe = _Gemma4RuntimeMoeBlock(
            layer.experts,
            layer_index=layer_index,
            per_expert_scale=layer.router.per_expert_scale,
        )
        self.post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
        self.post_feedforward_layernorm_2 = layer.post_feedforward_layernorm_2
        self.pre_feedforward_layernorm_2 = layer.pre_feedforward_layernorm_2
        self.top_k_experts = int(getattr(self.config, "top_k_experts", 0) or 0)
        self.router_epsilon = float(getattr(self.config, "rms_norm_eps", 1.0e-6) or 1.0e-6)

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_states = self.router.norm(hidden_states)
        scalar_root_size = getattr(self.router, "scalar_root_size", None)
        if scalar_root_size is None:
            root_size = getattr(self.router, "root_size", 1.0)
            scalar_root_size = root_size.to(router_states.dtype) if isinstance(root_size, torch.Tensor) else float(root_size)
        router_states = router_states * self.router.scale.to(router_states.dtype)
        router_states = router_states * scalar_root_size
        return self.router.proj(router_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        per_layer_input: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = False,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states_1 = self.pre_feedforward_layernorm(hidden_states)
        hidden_states_1 = self.mlp(hidden_states_1)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_1)

        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = self._router_logits(hidden_states_flat)
        hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.moe(
            hidden_states_2,
            router_logits,
            top_k=self.top_k_experts,
            normalize_routing=True,
            epsilon=self.router_epsilon,
        )
        hidden_states_2 = hidden_states_2.reshape(hidden_states.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = hidden_states_1 + hidden_states_2
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = torch.multiply(hidden_states, per_layer_input)
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


def _is_gemma4_moe_decoder_layer(layer: object) -> bool:
    return (
        isinstance(layer, torch.nn.Module)
        and bool(getattr(layer, "enable_moe_block", False))
        and hasattr(layer, "router")
        and hasattr(layer, "experts")
        and not isinstance(layer, _Gemma4MoeRuntimeDecoderLayer)
    )


def _gemma4_backbone_from_model(model: torch.nn.Module) -> torch.nn.Module | None:
    model_root = getattr(model, "model", None)
    backbone = getattr(model_root, "language_model", model_root)
    return backbone if isinstance(backbone, torch.nn.Module) else None


def _patch_gemma4_moe_layers(model: torch.nn.Module) -> None:
    if _family_key(model) != "gemma4":
        return
    backbone = _gemma4_backbone_from_model(model)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        return
    for layer_index, layer in enumerate(layers):
        if isinstance(layer, _Gemma4MoeRuntimeDecoderLayer):
            continue
        if _is_gemma4_moe_decoder_layer(layer):
            layers[layer_index] = _Gemma4MoeRuntimeDecoderLayer(layer, layer_index=layer_index)


def _gemma4_cpu_safe_text_mlp_enabled(layer: torch.nn.Module, hidden_states: torch.Tensor) -> bool:
    if hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float16:
        return False
    return _gemma4_text_mlp_manual_available(layer)


def _gemma4_text_mlp_manual_available(layer: torch.nn.Module) -> bool:
    if bool(getattr(layer, "enable_moe_block", False)):
        return False
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return False
    for attr_name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, attr_name):
            return False
    return True


def _gemma4_cpu_safe_text_mlp_forward(mlp: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    gate = F.linear(
        hidden_states,
        mlp.gate_proj.weight,
        mlp.gate_proj.bias,
    )
    up = F.linear(
        hidden_states,
        mlp.up_proj.weight,
        mlp.up_proj.bias,
    )
    # Gemma4 text MLPs can overflow in FP16 at the gated product.
    # Scaling one branch here keeps the product and down-projection finite.
    # The following post-feedforward RMSNorm is scale-invariant, so this
    # constant factor cancels back out in the normalized output.
    activated = mlp.act_fn(gate) * _GEMMA4_SAFE_TEXT_MLP_PRODUCT_SCALE
    return F.linear(
        activated * up,
        mlp.down_proj.weight,
        mlp.down_proj.bias,
    )


def _gemma4_text_attention_forward(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    past_key_values: object | None,
    use_cache: bool,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
) -> torch.Tensor:
    if shared_kv_states is None:
        attn_out, _ = attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states={},
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return attn_out

    from transformers.models.gemma4.modeling_gemma4 import ALL_ATTENTION_FUNCTIONS  # type: ignore
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb  # type: ignore
    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward  # type: ignore

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    if position_embeddings is None:
        raise ValueError("Gemma4 text attention requires position embeddings")
    cos, sin = position_embeddings

    query_states = attn.q_proj(hidden_states).view(hidden_shape)
    query_states = attn.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    kv_shared_key = getattr(attn, "layer_type", getattr(attn, "kv_shared_layer_index", None))
    if bool(getattr(attn, "is_kv_shared_layer", False)) and kv_shared_key in shared_kv_states:
        key_states, value_states = shared_kv_states[kv_shared_key]
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = attn.k_proj(hidden_states).view(hidden_shape)
        value_states = attn.v_proj(hidden_states).view(hidden_shape) if attn.v_proj is not None else key_states

        key_states = attn.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = attn.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        layer_idx = getattr(attn, "layer_idx", None)
        if bool(getattr(attn, "store_full_length_kv", False)):
            shared_kv_states[kv_shared_key] = (key_states, value_states)
        elif layer_idx is not None and not bool(getattr(attn, "is_kv_shared_layer", False)):
            shared_kv_states[int(layer_idx)] = (key_states, value_states)

    attention_interface = eager_attention_forward
    if getattr(attn.config, "_attn_implementation", "eager") != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

    attn_output, _ = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=attn.attention_dropout if attn.training else 0.0,
        scaling=attn.scaling,
        sliding_window=attn.sliding_window,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return attn.o_proj(attn_output)


def _gemma4_text_decoder_layer_forward(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    per_layer_input: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: object | None = None,
    use_cache: bool = False,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    if isinstance(layer, _Gemma4MoeRuntimeDecoderLayer):
        return layer(
            hidden_states,
            position_embeddings=position_embeddings,
            per_layer_input=per_layer_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            shared_kv_states=shared_kv_states,
        )
    manual_attention_required = shared_kv_states is not None
    if not manual_attention_required and not _gemma4_cpu_safe_text_mlp_enabled(layer, hidden_states):
        return layer(
            hidden_states,
            position_embeddings=position_embeddings,
            per_layer_input=per_layer_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
    if not _gemma4_text_mlp_manual_available(layer):
        raise RuntimeError("Gemma4 KV sharing requires a supported text MLP structure")

    residual = hidden_states

    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = _gemma4_text_attention_forward(
        layer.self_attn,
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        shared_kv_states=shared_kv_states,
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    mlp_out = _gemma4_cpu_safe_text_mlp_forward(layer.mlp, hidden_states)
    hidden_states = layer.post_feedforward_layernorm(mlp_out)
    hidden_states = residual.float() + hidden_states
    hidden_states = hidden_states.to(dtype=residual.dtype)

    if getattr(layer, "hidden_size_per_layer_input", 0):
        residual = hidden_states
        hidden_states = layer.per_layer_input_gate(hidden_states)
        hidden_states = layer.act_fn(hidden_states)
        if per_layer_input is None:
            raise ValueError("Gemma4 layer expected per_layer_input but none was provided")
        hidden_states = torch.multiply(hidden_states, per_layer_input)
        hidden_states = layer.per_layer_projection(hidden_states)
        hidden_states = layer.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = hidden_states * layer.layer_scalar
    return hidden_states


def _gemma4_text_backbone_forward(
    backbone: torch.nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    per_layer_inputs: torch.Tensor | None,
    causal_mask_mapping: dict[str, torch.Tensor],
    position_ids: torch.LongTensor,
    layer_start: int = 0,
    layer_end: int | None = None,
    apply_norm: bool = True,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None | object = _UNSET,
    capture_layer_index: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    hidden_states = inputs_embeds
    captured_hidden: torch.Tensor | None = None
    layer_types = tuple(dict.fromkeys(getattr(backbone.config, "layer_types", ())))
    position_embeddings = {
        layer_type: backbone.rotary_emb(hidden_states, position_ids, layer_type)
        for layer_type in layer_types
    }
    if shared_kv_states is _UNSET:
        num_shared_layers = int(getattr(backbone.config, "num_kv_shared_layers", 0) or 0)
        has_shared_kv_layers = num_shared_layers > 0 or any(
            bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False))
            or bool(getattr(getattr(layer, "self_attn", None), "store_full_length_kv", False))
            for layer in getattr(backbone, "layers", ())
        )
        shared_kv_states = {} if has_shared_kv_layers else None

    config_layer_types = tuple(getattr(backbone.config, "layer_types", ()))
    end = backbone.config.num_hidden_layers if layer_end is None else min(int(layer_end), int(backbone.config.num_hidden_layers))
    for layer_index in range(max(0, int(layer_start)), end):
        decoder_layer = backbone.layers[layer_index]
        layer_per_input = None
        if per_layer_inputs is not None:
            layer_per_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]
        attention_type = getattr(
            decoder_layer,
            "attention_type",
            config_layer_types[layer_index] if layer_index < len(config_layer_types) else "full_attention",
        )
        layer_shared_kv_states = shared_kv_states
        if shared_kv_states is not None:
            layer_attn = getattr(decoder_layer, "self_attn", None)
            layer_uses_shared_kv = (
                bool(getattr(layer_attn, "is_kv_shared_layer", False))
                or bool(getattr(layer_attn, "store_full_length_kv", False))
            )
            if not layer_uses_shared_kv:
                layer_shared_kv_states = None
        hidden_states = _gemma4_text_decoder_layer_forward(
            decoder_layer,
            hidden_states,
            per_layer_input=layer_per_input,
            attention_mask=causal_mask_mapping[attention_type],
            position_embeddings=position_embeddings[attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            shared_kv_states=layer_shared_kv_states,
        )
        if capture_layer_index is not None and layer_index == int(capture_layer_index):
            captured_hidden = hidden_states

    output_hidden = backbone.norm(hidden_states) if apply_norm else hidden_states
    if capture_layer_index is None:
        return output_hidden
    if captured_hidden is None:
        captured_hidden = hidden_states
    return output_hidden, captured_hidden


def _gemma4_strip_audio_padding(audio_output: object) -> torch.Tensor:
    audio_features = getattr(audio_output, "pooler_output", None)
    if not isinstance(audio_features, torch.Tensor):
        raise TypeError("Gemma4 audio output did not expose tensor pooler_output")

    audio_mask_from_encoder = getattr(audio_output, "attention_mask", None)
    if isinstance(audio_mask_from_encoder, torch.Tensor):
        return audio_features[audio_mask_from_encoder].unsqueeze(0)

    audio_mask_from_encoder = getattr(audio_output, "audio_mel_mask", None)
    if not isinstance(audio_mask_from_encoder, torch.Tensor):
        return audio_features
    all_real_tokens: list[torch.Tensor] = []
    for encodings, padding_mask in zip(audio_features, audio_mask_from_encoder, strict=True):
        all_real_tokens.append(encodings[~padding_mask])
    return torch.cat(all_real_tokens, dim=0).unsqueeze(0)


def _gemma4_rms_norm_no_scale(hidden_states: torch.Tensor, *, eps: float) -> torch.Tensor:
    hidden_states_f32 = hidden_states.float()
    mean_squared = hidden_states_f32.pow(2).mean(-1, keepdim=True) + eps
    return hidden_states_f32 * torch.pow(mean_squared, -0.5)


def _gemma4_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    normed = _gemma4_rms_norm_no_scale(hidden_states, eps=eps)
    return normed * weight.to(device=normed.device, dtype=normed.dtype)


def _gemma4_layer_norm_eps(multimodal_backbone: torch.nn.Module) -> float:
    config = getattr(multimodal_backbone, "config", None)
    text_config = None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except Exception:
            text_config = None
    for candidate in (text_config, config):
        value = getattr(candidate, "rms_norm_eps", None)
        if value is not None:
            return float(value)
        value = getattr(candidate, "layer_norm_eps", None)
        if value is not None:
            return float(value)
    return 1e-6


def _gemma4_load_cactus_fp_tensor(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        header = handle.read(84)
    if len(header) < 84 or header[:4] != CACTUS_MAGIC:
        raise RuntimeError(f"Gemma4 Cactus tensor is missing a valid header: {path}")

    alignment = max(1, int(struct.unpack_from("<I", header, 8)[0]))
    ndim = int(struct.unpack_from("<I", header, 12)[0])
    dims = struct.unpack_from("<QQQQ", header, 16)
    shape = tuple(int(dim) for dim in dims[:ndim] if int(dim) > 0)
    precision = int(struct.unpack_from("<I", header, 48)[0])
    byte_size = int(struct.unpack_from("<Q", header, 52)[0])
    scales_bytes = int(struct.unpack_from("<Q", header, 60)[0])
    dtype = {1: np.float16, 2: np.float32}.get(precision)
    if dtype is None:
        raise RuntimeError(f"Gemma4 Cactus tensor must be FP16/FP32, got precision={precision}: {path}")

    aligned_header = align_offset(84, alignment)
    scales_offset = aligned_header if scales_bytes > 0 else 0
    data_offset = (
        align_offset(scales_offset + scales_bytes, alignment)
        if scales_bytes > 0
        else aligned_header
    )
    data_count = byte_size // np.dtype(dtype).itemsize
    array = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=(data_count,))
    tensor = torch.from_numpy(np.array(array, copy=True))
    if shape:
        tensor = tensor.reshape(shape)
    return tensor


def _gemma4_load_vision_post_proj_norm(weights_dir: str | Path | None) -> torch.Tensor | None:
    if weights_dir is None:
        return None
    path = Path(weights_dir).expanduser() / "embed_vision_post_proj_norm.weights"
    if not path.exists():
        return None
    tensor = _gemma4_load_cactus_fp_tensor(path)
    if tensor.ndim != 1:
        raise RuntimeError(f"Gemma4 vision post-projection norm must be 1D, got shape={tuple(tensor.shape)}")
    return tensor.float()


def _gemma4_can_use_native_like_vision_features(multimodal_backbone: torch.nn.Module) -> bool:
    vision_tower = getattr(multimodal_backbone, "vision_tower", None)
    embed_vision = getattr(multimodal_backbone, "embed_vision", None)
    return (
        isinstance(vision_tower, torch.nn.Module)
        and isinstance(embed_vision, torch.nn.Module)
        and hasattr(vision_tower, "patch_embedder")
        and hasattr(vision_tower, "encoder")
        and hasattr(embed_vision, "embedding_projection")
    )


def _gemma4_can_use_native_like_audio_features(multimodal_backbone: torch.nn.Module) -> bool:
    audio_tower = getattr(multimodal_backbone, "audio_tower", None)
    embed_audio = getattr(multimodal_backbone, "embed_audio", None)
    layers = getattr(audio_tower, "layers", None) if isinstance(audio_tower, torch.nn.Module) else None
    conformer = getattr(audio_tower, "conformer", None) if isinstance(audio_tower, torch.nn.Module) else None
    return (
        isinstance(audio_tower, torch.nn.Module)
        and isinstance(embed_audio, torch.nn.Module)
        and hasattr(audio_tower, "subsample_conv_projection")
        and (isinstance(layers, torch.nn.ModuleList) or isinstance(conformer, torch.nn.ModuleList))
        and hasattr(embed_audio, "embedding_projection")
    )


def _gemma4_pool_vision_hidden_native_like(
    vision_tower: torch.nn.Module,
    hidden_states: torch.Tensor,
    pixel_position_ids: torch.Tensor,
    *,
    image_soft_token_counts: tuple[int, ...] | None = None,
    image_pool_shapes: tuple[tuple[int, int, int], ...] | None = None,
) -> torch.Tensor:
    output_length = int(_module_or_config_attr(vision_tower, "default_output_length", 280) or 280)
    pooling_kernel_size = int(_module_or_config_attr(vision_tower, "pooling_kernel_size", 3) or 3)
    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    pooled_batches: list[torch.Tensor] = []
    for row_idx in range(int(hidden_states.shape[0])):
        hidden_row = hidden_states[row_idx]
        position_row = pixel_position_ids[row_idx]
        padding_row = padding_positions[row_idx]
        if image_pool_shapes is not None and row_idx < len(image_pool_shapes):
            grid_h, grid_w, pooled_count = image_pool_shapes[row_idx]
            valid_patch_count = int(grid_h) * int(grid_w)
            channels = int(hidden_row.shape[-1])
            valid_hidden = hidden_row[:valid_patch_count].float()
            pooled = valid_hidden.reshape(
                int(grid_h) // pooling_kernel_size,
                pooling_kernel_size,
                int(grid_w) // pooling_kernel_size,
                pooling_kernel_size,
                channels,
            ).mean(dim=(1, 3))
            pooled_batches.append(pooled.reshape(int(pooled_count), channels))
            continue

        # Avoid boolean advanced indexing here. Gemma4 image padding is not
        # guaranteed to be prefix-shaped, and the generic lowerer optimizes
        # some masks as prefix slices. Zero-weighting padded patches preserves
        # native pooling semantics while staying easy to lower.
        clamped_positions = position_row.clamp(min=0)
        max_x = clamped_positions[:, 0].max() + 1
        kernel_positions = torch.div(clamped_positions, pooling_kernel_size, rounding_mode="floor")
        kernel_indices = kernel_positions[:, 0] + torch.div(
            max_x,
            pooling_kernel_size,
            rounding_mode="floor",
        ) * kernel_positions[:, 1]
        valid_patch_weights = torch.logical_not(padding_row).float().unsqueeze(-1)
        weights = (
            F.one_hot(kernel_indices.long(), output_length).float()
            * valid_patch_weights
            / float(pooling_kernel_size**2)
        )
        pooled_full = weights.transpose(0, 1) @ hidden_row.float()
        if image_soft_token_counts is not None and row_idx < len(image_soft_token_counts):
            pooled_batches.append(pooled_full[: int(image_soft_token_counts[row_idx])])
        else:
            valid_bins = torch.logical_not((weights == 0).all(dim=0))
            pooled_batches.append(pooled_full[valid_bins])
    pooled_hidden = pooled_batches[0] if len(pooled_batches) == 1 else torch.cat(pooled_batches, dim=0)
    hidden_size = int(_module_or_config_attr(vision_tower, "hidden_size", pooled_hidden.shape[-1]) or pooled_hidden.shape[-1])
    pooled_hidden = pooled_hidden * math.sqrt(float(hidden_size))
    if bool(_module_or_config_attr(vision_tower, "standardize", False)):
        std_bias = getattr(vision_tower, "std_bias", None)
        std_scale = getattr(vision_tower, "std_scale", None)
        if std_bias is not None and std_scale is not None:
            pooled_hidden = (pooled_hidden - std_bias) * std_scale
    return pooled_hidden


def _gemma4_vision_encoder_hidden_states(
    vision_encoder: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_position_ids: torch.Tensor | None,
) -> torch.Tensor:
    config = getattr(vision_encoder, "config", None)
    hidden_states = inputs_embeds
    rotary_emb = getattr(vision_encoder, "rotary_emb")
    layers = getattr(vision_encoder, "layers")
    num_layers = int(getattr(config, "num_hidden_layers", len(layers)))
    layer_types = tuple(str(value) for value in (getattr(config, "layer_types", ()) or ()))
    if layer_types and all(hasattr(rotary_emb, f"{layer_type}_inv_freq") for layer_type in layer_types):
        # Newer Gemma4 builds use per-layer vision RoPE tables keyed by the
        # layer attention type instead of a single vision RoPE table.
        attention_mask_4d = (attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)).unsqueeze(1)
        position_embeddings_by_type = {
            layer_type: rotary_emb(hidden_states, pixel_position_ids, layer_type)
            for layer_type in layer_types
        }
        for decoder_layer in layers[:num_layers]:
            layer_type = str(getattr(decoder_layer, "attention_type", layer_types[0]))
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings_by_type[layer_type],
                position_ids=pixel_position_ids,
            )
        return hidden_states

    if _GEMMA4_CREATE_BIDIRECTIONAL_MASK is None:
        raise RuntimeError("Gemma4 bidirectional mask helper is unavailable in this transformers install")
    if inputs_embeds.is_meta:
        attention_mask = (attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)).unsqueeze(1)
    else:
        attention_mask = _GEMMA4_CREATE_BIDIRECTIONAL_MASK(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    position_embeddings = rotary_emb(hidden_states, pixel_position_ids)
    for decoder_layer in layers[:num_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=pixel_position_ids,
        )
    return hidden_states


def _gemma4_convert_audio_mask_to_blocked_static(
    mask_4d: torch.Tensor,
    *,
    chunk_size: int,
    left_context: int,
    right_context: int,
) -> torch.Tensor:
    """Build Gemma4's blocked local audio mask without tensor index expand.

    Hugging Face's helper uses an expanded gather-index tensor. The v2 graph
    runtime does not have a first-class expand op, so we express the same fixed
    chunk extraction as static slices and cats.
    """
    _, _, seq_len, _ = mask_4d.shape
    seq_len_int = int(seq_len)
    num_blocks = (seq_len_int + int(chunk_size) - 1) // int(chunk_size)
    padded_seq_len = num_blocks * int(chunk_size)
    pad_amount = padded_seq_len - seq_len_int
    if pad_amount:
        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
    mask_4d = F.pad(mask_4d, (int(left_context), int(right_context)), value=False)

    context_size = int(chunk_size) + int(left_context) + int(right_context)
    blocks: list[torch.Tensor] = []
    for block_idx in range(num_blocks):
        start = block_idx * int(chunk_size)
        block = mask_4d[
            :,
            :,
            start : start + int(chunk_size),
            start : start + context_size,
        ]
        blocks.append(block.unsqueeze(2))
    return torch.cat(blocks, dim=2)


def _gemma4_compute_native_like_image_features(
    multimodal_backbone: torch.nn.Module,
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor | None,
    *,
    post_proj_norm_weight: torch.Tensor | None = None,
    image_soft_token_counts: tuple[int, ...] | None = None,
    image_pool_shapes: tuple[tuple[int, int, int], ...] | None = None,
) -> torch.Tensor:
    if pixel_position_ids is None:
        raise TypeError("Gemma4 native-like vision feature path requires pixel_position_ids")
    vision_tower = getattr(multimodal_backbone, "vision_tower", None)
    embed_vision = getattr(multimodal_backbone, "embed_vision", None)
    if not isinstance(vision_tower, torch.nn.Module) or not isinstance(embed_vision, torch.nn.Module):
        raise TypeError("Gemma4 multimodal backbone is missing native-like vision modules")

    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    vision_inputs = vision_tower.patch_embedder(
        pixel_values,
        pixel_position_ids,
        padding_positions,
    )
    vision_hidden = _gemma4_vision_encoder_hidden_states(
        vision_tower.encoder,
        vision_inputs,
        ~padding_positions,
        pixel_position_ids,
    )
    pooled_hidden = _gemma4_pool_vision_hidden_native_like(
        vision_tower,
        vision_hidden,
        pixel_position_ids,
        image_soft_token_counts=image_soft_token_counts,
        image_pool_shapes=image_pool_shapes,
    )
    projection = getattr(embed_vision, "embedding_projection", None)
    if not isinstance(projection, torch.nn.Linear):
        raise TypeError("Gemma4 vision embedder is missing embedding_projection")
    pre_projection_norm = getattr(embed_vision, "embedding_pre_projection_norm", None)
    if isinstance(pre_projection_norm, torch.nn.Module):
        pooled_hidden = pre_projection_norm(pooled_hidden)
    pooled_hidden = pooled_hidden.to(dtype=projection.weight.dtype)
    projected = F.linear(
        pooled_hidden,
        projection.weight,
        projection.bias,
    )
    if post_proj_norm_weight is not None and post_proj_norm_weight.numel() > 0:
        return _gemma4_rms_norm(
            projected,
            post_proj_norm_weight,
            eps=_gemma4_layer_norm_eps(multimodal_backbone),
        )
    return projected


_GEMMA4_NPU_PROJ_WEIGHTS = (
    ("embed_audio", "embed_audio_proj.weights"),
    ("embed_vision", "embed_vision_proj.weights"),
)


@contextmanager
def _gemma4_npu_projection_reparam(module: torch.nn.Module):
    backbone = getattr(module, "multimodal_backbone", None)
    scaled: list[tuple[torch.nn.Linear, float]] = []
    for attr, cactus_name in _GEMMA4_NPU_PROJ_WEIGHTS:
        projection = getattr(getattr(backbone, attr, None), "embedding_projection", None)
        factor = gemma4_scale_factor(cactus_name)
        if isinstance(projection, torch.nn.Linear) and factor != 1.0:
            scaled.append((projection, factor))
    for projection, factor in scaled:
        projection.weight.data.mul_(factor)
    try:
        yield
    finally:
        for projection, factor in scaled:
            projection.weight.data.div_(factor)


def _gemma4_compute_native_like_audio_features(
    multimodal_backbone: torch.nn.Module,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
) -> torch.Tensor:
    audio_tower = getattr(multimodal_backbone, "audio_tower", None)
    embed_audio = getattr(multimodal_backbone, "embed_audio", None)
    if not isinstance(audio_tower, torch.nn.Module) or not isinstance(embed_audio, torch.nn.Module):
        raise TypeError("Gemma4 multimodal backbone is missing native-like audio modules")

    config = getattr(audio_tower, "config", None)
    if config is None:
        raise TypeError("Gemma4 audio tower is missing config")
    hidden_states, output_mask = audio_tower.subsample_conv_projection(input_features, input_features_mask)

    if not hasattr(audio_tower, "rel_pos_enc"):
        # Newer Gemma4 audio towers keep relative-position handling inside each
        # conformer block. Follow that native path with fixed-shape masks.
        chunk_size = int(getattr(config, "conf_attention_chunk_size", getattr(config, "attention_chunk_size", 12)))
        right_context = int(getattr(config, "conf_attention_context_right", getattr(config, "attention_context_right", 0)))
        left_context = int(getattr(config, "conf_attention_context_left", getattr(config, "attention_context_left", 13)))
        max_past_horizon = max(0, left_context - 1)
        upper_diagonal = max_past_horizon + right_context
        context_size = chunk_size + max_past_horizon + right_context
        lower_causal_mask = torch.tril(
            torch.ones((context_size, chunk_size), dtype=torch.bool, device=hidden_states.device),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((chunk_size, context_size), dtype=torch.bool, device=hidden_states.device),
            diagonal=upper_diagonal,
        )
        causal_valid_mask = (
            torch.ones((chunk_size, context_size), dtype=torch.bool, device=hidden_states.device)
            * lower_causal_mask
            * upper_causal_mask
        )

        layers = getattr(audio_tower, "conformer", None)
        if not isinstance(layers, torch.nn.ModuleList):
            raise TypeError("Gemma4 audio tower is missing conformer layers")
        num_layers = int(getattr(config, "num_hidden_layers", len(layers)))
        for encoder_layer in layers[:num_layers]:
            hidden_states = encoder_layer(hidden_states, output_mask, causal_valid_mask)

        reduction_factor = int(getattr(config, "conf_reduction_factor", 1) or 1)
        if reduction_factor > 1:
            hidden_states = hidden_states[:, ::reduction_factor]
            if output_mask is not None:
                output_mask = output_mask[:, ::reduction_factor]

        output_proj = getattr(audio_tower, "output_proj", None)
        audio_encodings = output_proj(hidden_states) if isinstance(output_proj, torch.nn.Linear) else hidden_states
    else:
        position_embeddings = audio_tower.rel_pos_enc(hidden_states)
        seq_len = hidden_states.shape[1]
        query_positions = torch.arange(seq_len, device=hidden_states.device)[:, None]
        key_positions = torch.arange(seq_len, device=hidden_states.device)[None, :]
        distance = query_positions - key_positions
        left_context = int(getattr(config, "attention_context_left", 13)) - 1
        right_context = int(getattr(config, "attention_context_right", 0))
        local_mask = ((distance >= 0) & (distance < left_context)) | ((distance < 0) & ((-distance) < right_context))
        attention_mask = local_mask.unsqueeze(0).unsqueeze(0)
        if output_mask is not None:
            attention_mask = attention_mask & output_mask[:, None, None, :].to(dtype=torch.bool)
        attention_mask = _gemma4_convert_audio_mask_to_blocked_static(
            attention_mask,
            chunk_size=int(getattr(config, "attention_chunk_size", 12)),
            left_context=left_context,
            right_context=right_context,
        )

        layers = getattr(audio_tower, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList):
            layers = getattr(audio_tower, "conformer", None)
        if not isinstance(layers, torch.nn.ModuleList):
            raise TypeError("Gemma4 audio tower is missing layers/conformer")
        num_layers = int(getattr(config, "num_hidden_layers", len(layers)))
        for encoder_layer in layers[:num_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        output_proj = getattr(audio_tower, "output_proj", None)
        if not isinstance(output_proj, torch.nn.Linear):
            raise TypeError("Gemma4 audio tower is missing output_proj")
        audio_encodings = output_proj(hidden_states)

    if not callable(getattr(embed_audio, "forward", None)):
        raise TypeError("Gemma4 audio embedder is not callable")
    projected = embed_audio(inputs_embeds=audio_encodings)
    # The transpiled bundle is shape-specialized from representative media.
    # Native Gemma4 audio preprocessing emits an unpadded feature tensor for
    # that media, so the post-subsampling sequence is already the real token
    # sequence. Returning the dense sequence avoids dynamic boolean indexing in
    # torch.export and matches the prompt's static audio soft-token count.
    return projected


def _gemma4_feature_token_count(features: torch.Tensor | None) -> int:
    if features is None or features.numel() == 0:
        return 0
    if features.ndim == 3:
        return int(features.shape[1])
    if features.ndim == 2:
        return int(features.shape[0])
    if features.ndim >= 1:
        return int(features.reshape(-1, features.shape[-1]).shape[0])
    return 0


def _gemma4_static_image_soft_token_counts(
    pixel_position_ids: torch.Tensor | None,
    *,
    pooling_kernel_size: int,
) -> tuple[int, ...] | None:
    if pixel_position_ids is None or pixel_position_ids.ndim != 3:
        return None
    counts: list[int] = []
    for positions in pixel_position_ids.detach().cpu():
        valid = positions[(positions != -1).any(dim=-1)]
        if valid.numel() == 0:
            counts.append(0)
            continue
        max_x = int(valid[:, 0].max().item()) + 1
        max_y = int(valid[:, 1].max().item()) + 1
        counts.append((max_x // int(pooling_kernel_size)) * (max_y // int(pooling_kernel_size)))
    return tuple(counts)


def _gemma4_static_image_pool_shapes(
    pixel_position_ids: torch.Tensor | None,
    *,
    pooling_kernel_size: int,
) -> tuple[tuple[int, int, int], ...] | None:
    if pixel_position_ids is None or pixel_position_ids.ndim != 3:
        return None
    shapes: list[tuple[int, int, int]] = []
    for positions in pixel_position_ids.detach().cpu():
        valid = positions[(positions != -1).any(dim=-1)]
        if valid.numel() == 0:
            shapes.append((0, 0, 0))
            continue
        grid_w = int(valid[:, 0].max().item()) + 1
        grid_h = int(valid[:, 1].max().item()) + 1
        pooled_count = (grid_w // int(pooling_kernel_size)) * (grid_h // int(pooling_kernel_size))
        shapes.append((grid_h, grid_w, pooled_count))
    return tuple(shapes)


def _gemma4_build_native_merge_plan(
    multimodal_backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    image_feature_count: int,
    audio_feature_count: int,
) -> _Gemma4NativeMergePlan:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError(
            "Gemma4 native multimodal merge currently expects a static batch-1 input_ids tensor, "
            f"got shape {tuple(input_ids.shape)}"
        )

    image_token_id, audio_token_id, pad_token_id = _gemma4_special_token_ids(multimodal_backbone)
    token_ids = [int(token) for token in input_ids.detach().cpu().reshape(-1).tolist()]
    segments: list[_Gemma4NativeMergeSegment] = []
    pli_token_ids: list[int] = []
    image_offset = 0
    audio_offset = 0
    index = 0

    while index < len(token_ids):
        token_id = token_ids[index]
        is_image = image_token_id != 0 and token_id == image_token_id
        is_audio = audio_token_id != 0 and token_id == audio_token_id

        if is_image or is_audio:
            region_start = index
            while index < len(token_ids) and token_ids[index] == token_id:
                index += 1
            placeholder_count = index - region_start
            if is_image:
                insert_count = min(placeholder_count, max(0, image_feature_count - image_offset))
                if insert_count > 0:
                    segments.append(
                        _Gemma4NativeMergeSegment(
                            kind="image",
                            input_start=region_start,
                            length=insert_count,
                            feature_start=image_offset,
                        )
                    )
                    pli_token_ids.extend([pad_token_id] * insert_count)
                    image_offset += insert_count
            else:
                insert_count = min(placeholder_count, max(0, audio_feature_count - audio_offset))
                if insert_count > 0:
                    segments.append(
                        _Gemma4NativeMergeSegment(
                            kind="audio",
                            input_start=region_start,
                            length=insert_count,
                            feature_start=audio_offset,
                        )
                    )
                    pli_token_ids.extend([pad_token_id] * insert_count)
                    audio_offset += insert_count
            continue

        text_start = index
        while index < len(token_ids):
            next_token = token_ids[index]
            if image_token_id != 0 and next_token == image_token_id:
                break
            if audio_token_id != 0 and next_token == audio_token_id:
                break
            index += 1
        text_tokens = token_ids[text_start:index]
        if text_tokens:
            segments.append(
                _Gemma4NativeMergeSegment(
                    kind="text",
                    input_start=text_start,
                    length=len(text_tokens),
                )
            )
            pli_token_ids.extend(text_tokens)

    if not segments:
        raise RuntimeError("Gemma4 native multimodal merge built no input segments")
    return _Gemma4NativeMergePlan(
        segments=tuple(segments),
        pli_token_ids=tuple(pli_token_ids),
    )


def _gemma4_feature_sequence(
    features: torch.Tensor,
    *,
    batch_size: int,
    feature_dim: int,
) -> torch.Tensor:
    if features.ndim == 3:
        if int(features.shape[0]) == batch_size:
            return features
        if int(features.shape[0]) == 1:
            return features.expand(batch_size, -1, -1)
    if features.ndim == 2:
        if int(features.shape[-1]) != feature_dim:
            raise ValueError(f"Gemma4 feature dim mismatch: expected {feature_dim}, got {tuple(features.shape)}")
        return features.unsqueeze(0).expand(batch_size, -1, -1)
    if features.ndim >= 1 and int(features.shape[-1]) == feature_dim:
        flattened = features.reshape(-1, feature_dim)
        return flattened.unsqueeze(0).expand(batch_size, -1, -1)
    raise ValueError(f"unsupported Gemma4 feature tensor shape: {tuple(features.shape)}")


def _gemma4_text_embedding_scale(embedding: torch.nn.Module, fallback_scale: float) -> float:
    """Return the extra scale needed after calling the HF embedding module.

    Native Cactus scales raw token embedding weights by sqrt(hidden_dim).  The
    HF Gemma4 embedding module already applies that scale in forward(), so the
    transpiler must not multiply a second time when it traces the HF module.
    """
    for attr_name in ("scalar_embed_scale", "embed_scale"):
        value = getattr(embedding, attr_name, None)
        if isinstance(value, torch.Tensor):
            try:
                if value.numel() > 0 and abs(float(value.reshape(-1)[0].item()) - float(fallback_scale)) < 1e-3:
                    return 1.0
            except Exception:
                continue
        elif isinstance(value, (float, int)) and abs(float(value) - float(fallback_scale)) < 1e-3:
            return 1.0
    return float(fallback_scale)


def _gemma4_apply_native_merge_plan(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    image_features: torch.Tensor,
    audio_features: torch.Tensor,
    merge_plan: _Gemma4NativeMergePlan,
    pli_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = model.get_input_embeddings()
    if not isinstance(embedding, torch.nn.Module):
        raise TypeError("Gemma4 model is missing input embeddings")
    batch_size = int(input_ids.shape[0])
    feature_dim = int(getattr(embedding, "embedding_dim", 0) or image_features.shape[-1] or audio_features.shape[-1])
    target_dtype = _module_floating_dtype(embedding) or image_features.dtype
    image_sequence = _gemma4_feature_sequence(
        image_features,
        batch_size=batch_size,
        feature_dim=feature_dim,
    ).to(device=input_ids.device, dtype=target_dtype)
    audio_sequence = _gemma4_feature_sequence(
        audio_features,
        batch_size=batch_size,
        feature_dim=feature_dim,
    ).to(device=input_ids.device, dtype=target_dtype)

    embedded_segments: list[torch.Tensor] = []
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    hidden_scale = float(getattr(model_config, "hidden_size", 0) or 0)
    if hidden_scale <= 0.0:
        hidden_scale = float(getattr(text_config, "hidden_size", 0) or 0)
    if hidden_scale <= 0.0:
        hidden_scale = float(feature_dim)
    hidden_scale = float(hidden_scale) ** 0.5
    text_extra_scale = _gemma4_text_embedding_scale(embedding, hidden_scale)
    for segment in merge_plan.segments:
        if segment.kind == "text":
            text_tokens = input_ids[:, segment.input_start : segment.input_start + segment.length]
            text_embeds = embedding(text_tokens)
            if text_extra_scale != 1.0:
                text_embeds = text_embeds * text_extra_scale
            embedded_segments.append(text_embeds)
        elif segment.kind == "image":
            embedded_segments.append(
                image_sequence[:, segment.feature_start : segment.feature_start + segment.length, :]
            )
        elif segment.kind == "audio":
            embedded_segments.append(
                audio_sequence[:, segment.feature_start : segment.feature_start + segment.length, :]
            )
        else:
            raise RuntimeError(f"unknown Gemma4 merge segment kind: {segment.kind!r}")

    inputs_embeds = torch.cat(embedded_segments, dim=1)
    if pli_token_ids.numel() != inputs_embeds.shape[1]:
        raise RuntimeError(
            "Gemma4 native merge PLI token count mismatch: "
            f"{int(pli_token_ids.numel())} vs {int(inputs_embeds.shape[1])}"
        )
    pli_tokens = pli_token_ids.to(device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
    if batch_size != 1:
        pli_tokens = pli_tokens.expand(batch_size, -1)
    return inputs_embeds, pli_tokens


def _gemma4_remap_sequence_tensor(
    tensor: torch.Tensor | None,
    *,
    merge_plan: _Gemma4NativeMergePlan,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim != 2:
        raise ValueError(f"Gemma4 merge remap expects a rank-2 tensor, got shape {tuple(tensor.shape)}")
    remapped_segments: list[torch.Tensor] = []
    for segment in merge_plan.segments:
        remapped_segments.append(
            tensor[:, segment.input_start : segment.input_start + segment.length]
        )
    if not remapped_segments:
        return tensor[:, :0]
    return torch.cat(remapped_segments, dim=1)


def _gemma4_build_standard_causal_mask_mapping(
    *,
    create_causal_mask: Callable[..., torch.Tensor] | None,
    create_sliding_window_causal_mask: Callable[..., torch.Tensor] | None,
    config: object,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    if not callable(create_causal_mask) or not callable(create_sliding_window_causal_mask):
        raise RuntimeError("Gemma4 standard causal mask helpers are unavailable")
    if inputs_embeds.is_meta or position_ids.is_meta or (attention_mask is not None and attention_mask.is_meta):
        return {
            "full_attention": None,
            "sliding_attention": None,
        }
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }


def _infer_input_names(module: torch.nn.Module, *, preferred: tuple[str, ...]) -> tuple[str, ...]:
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return preferred[:1]

    control_names = {
        "self",
        "return_dict",
        "use_cache",
        "past_key_values",
        "cache_position",
        "position_ids",
        "labels",
        "decoder_input_ids",
        "decoder_attention_mask",
        "output_attentions",
        "output_hidden_states",
    }
    available = [
        name
        for name, parameter in signature.parameters.items()
        if name not in control_names
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    matched = [name for name in preferred if name in available]
    if matched:
        return tuple(matched)
    if available:
        return tuple(available[: min(2, len(available))])
    return preferred[:1]


class BoundInputAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], family: str, metadata_task: str):
        super().__init__()
        self.model = model
        self.input_names = tuple(input_names)
        self.family = family
        self.metadata_task = metadata_task

    def _kwargs_from_bound_inputs(self, *bound_inputs: torch.Tensor | None) -> dict[str, torch.Tensor]:
        provided = tuple(bound_inputs)
        if len(self.input_names) > len(provided):
            raise ValueError(
                f"adapter expected at most {len(provided)} bound inputs, got {len(self.input_names)} names"
            )
        kwargs: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.input_names):
            value = provided[index]
            if value is None:
                raise ValueError(f"missing required bound input {index} for {name}")
            kwargs[name] = value
        return kwargs

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family=self.family,
                    adapter_type=type(self).__name__,
                    input_names=self.input_names,
                ),
                "task": self.metadata_task,
            }
        }


class CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = getattr(model, "model", None)
        self.lm_head = getattr(model, "lm_head", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)

    def forward(self, input_ids: torch.Tensor):
        backbone = self.backbone
        lm_head = self.lm_head
        if isinstance(backbone, torch.nn.Module) and isinstance(lm_head, torch.nn.Module):
            backbone_kwargs: dict[str, object] = {
                "input_ids": input_ids,
                "attention_mask": (input_ids != int(self.pad_token_id)).long()
                if self.pad_token_id is not None
                else None,
                "use_cache": False,
                "return_dict": False,
            }
            outputs = backbone(**_filter_supported_kwargs(backbone, backbone_kwargs))
            hidden_states = _extract_tensor_output(outputs, preferred_field="last_hidden_state")
            hidden_states = _select_last_non_pad_token(
                hidden_states,
                input_ids,
                pad_token_id=self.pad_token_id,
            )
            return lm_head(hidden_states)

        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )
        logits = _extract_tensor_output(outputs, preferred_field="logits")
        return _select_last_non_pad_token(
            logits,
            input_ids,
            pad_token_id=self.pad_token_id,
        )

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="generic",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
            }
        }


class Lfm2CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        _patch_lfm2_moe_feed_forwards(model)
        self.model = model
        model_root = getattr(model, "model", None)
        language_model = getattr(model_root, "language_model", None)
        self.backbone = language_model if isinstance(language_model, torch.nn.Module) else model_root
        self.lm_head = getattr(model, "lm_head", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        self.adapter_family = _family_key(model)
        from transformers.models.lfm2.modeling_lfm2 import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(self, input_ids: torch.Tensor):
        backbone = self.backbone
        lm_head = self.lm_head
        if not isinstance(backbone, torch.nn.Module) or not isinstance(lm_head, torch.nn.Module):
            raise TypeError("LFM2 causal logits adapter requires backbone and lm_head modules")

        inputs_embeds = backbone.embed_tokens(input_ids)
        attention_mask = (
            (input_ids != int(self.pad_token_id)).to(dtype=torch.int64)
            if self.pad_token_id is not None
            else None
        )
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = _lfm2_causal_mask_for_capture(
            self._create_causal_mask,
            backbone=backbone,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = _lfm2_position_embeddings(backbone, hidden_states, position_ids=position_ids)
        layer_types = tuple(getattr(backbone.config, "layer_types", ()))
        linear_attention = attention_mask if inputs_embeds.shape[1] != 1 else None

        for layer_index, decoder_layer in enumerate(backbone.layers[: backbone.config.num_hidden_layers]):
            layer_type = layer_types[layer_index] if layer_index < len(layer_types) else "full_attention"
            layer_mask = causal_mask if layer_type == "full_attention" else linear_attention
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )

        hidden_states = backbone.embedding_norm(hidden_states)
        hidden_states = _select_last_non_pad_token(
            hidden_states,
            input_ids,
            pad_token_id=self.pad_token_id,
        )
        return lm_head(hidden_states)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family=self.adapter_family,
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
            }
        }


class Lfm2CausalLMStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        _patch_lfm2_moe_feed_forwards(model)
        self.model = model
        model_root = getattr(model, "model", None)
        language_model = getattr(model_root, "language_model", None)
        self.backbone = language_model if isinstance(language_model, torch.nn.Module) else model_root
        self.lm_head = getattr(model, "lm_head", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        self.adapter_family = _family_key(model)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        lm_head = self.lm_head
        if not isinstance(backbone, torch.nn.Module) or not isinstance(lm_head, torch.nn.Module):
            raise TypeError("LFM2 causal step adapter requires backbone and lm_head modules")

        inputs_embeds = backbone.embed_tokens(input_ids)
        text_position_ids = position_ids.to(dtype=torch.int64)
        causal_mask = None

        hidden_states = inputs_embeds
        position_embeddings = _lfm2_position_embeddings(backbone, hidden_states, position_ids=text_position_ids)
        layer_types = tuple(getattr(backbone.config, "layer_types", ()))

        for layer_index, decoder_layer in enumerate(backbone.layers[: backbone.config.num_hidden_layers]):
            layer_type = layer_types[layer_index] if layer_index < len(layer_types) else "full_attention"
            layer_mask = causal_mask if layer_type == "full_attention" else None
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )

        hidden_states = backbone.embedding_norm(hidden_states)
        return lm_head(hidden_states[:, -1:, :])

    def get_transpile_metadata(self):
        layer_types = tuple(str(value) for value in (getattr(self.backbone.config, "layer_types", ()) or ()))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family=self.adapter_family,
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "position_ids"),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": layer_types,
            }
        }


def _lfm2_vl_model_root(model: torch.nn.Module) -> torch.nn.Module:
    root = getattr(model, "model", None)
    if not isinstance(root, torch.nn.Module):
        raise TypeError("LFM2-VL adapter requires a model.model module")
    return root


def _lfm2_language_backbone(model: torch.nn.Module) -> torch.nn.Module:
    _patch_lfm2_moe_feed_forwards(model)
    root = getattr(model, "model", None)
    language_model = getattr(root, "language_model", None)
    backbone = language_model if isinstance(language_model, torch.nn.Module) else root
    if not isinstance(backbone, torch.nn.Module):
        raise TypeError("LFM2 adapter requires a language model backbone")
    return backbone


class Lfm2VlVisionEncoderAdapter(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        weights_dir: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.weights_dir = weights_dir
        root = _lfm2_vl_model_root(model)
        vision_tower = getattr(root, "vision_tower", None)
        if not isinstance(vision_tower, torch.nn.Module):
            raise TypeError("LFM2-VL model is missing a vision_tower module")
        vision_model = getattr(vision_tower, "vision_model", None)
        if not isinstance(vision_model, torch.nn.Module):
            raise TypeError("LFM2-VL vision_tower is missing vision_model")

        self.embeddings = vision_model.embeddings
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        positional_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.embeddings.patch_embedding.weight.dtype
        hidden_states = self.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))
        hidden_states = hidden_states + positional_embeddings.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        masked_bias = -30000.0
        seq_len = hidden_states.shape[1]
        key_valid = pixel_attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        encoder_attention_mask = (key_valid * (-masked_bias) + masked_bias).expand(
            hidden_states.shape[0], 1, seq_len, seq_len
        )
        for encoder_layer in self.encoder.layers:
            hidden_states = encoder_layer(hidden_states, encoder_attention_mask)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="lfm2_vl",
                    adapter_type=type(self).__name__,
                    input_names=("pixel_values", "pixel_attention_mask", "positional_embeddings"),
                ),
                "weights_dir": self.weights_dir,
            }
        }


class Lfm2VlVisionProjectorAdapter(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        weights_dir: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.weights_dir = weights_dir
        root = _lfm2_vl_model_root(model)
        projector = getattr(root, "multi_modal_projector", None)
        if not isinstance(projector, torch.nn.Module):
            raise TypeError("LFM2-VL model is missing multi_modal_projector")
        self.projector = projector

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        hidden_states = vision_features
        if self.projector.use_layer_norm:
            hidden_states = self.projector.layer_norm(hidden_states)
        hidden_states = self.projector.linear_1(hidden_states)
        hidden_states = self.projector.act(hidden_states)
        hidden_states = self.projector.linear_2(hidden_states)
        return hidden_states

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="lfm2_vl",
                    adapter_type=type(self).__name__,
                    input_names=("vision_features",),
                ),
                "weights_dir": self.weights_dir,
            }
        }


class Lfm2VlLMEncoderAdapter(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        weights_dir: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.weights_dir = weights_dir
        self.backbone = _lfm2_language_backbone(model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        return inputs_embeds, attention_mask.to(dtype=torch.int64), position_ids

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="lfm2_vl",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "attention_mask"),
                ),
                "weights_dir": self.weights_dir,
            }
        }


class Lfm2VlLMEncoderStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__()
        self.model = model
        self.weights_dir = weights_dir
        self.backbone = _lfm2_language_backbone(model)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        attention_mask = input_ids.to(dtype=torch.int64) * 0 + 1
        return inputs_embeds, attention_mask, position_ids.to(dtype=torch.int64)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="lfm2_vl",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "position_ids"),
                ),
                "weights_dir": self.weights_dir,
            }
        }


class Lfm2VlLMEncoderTextChunkAdapter(Lfm2VlLMEncoderStepAdapter):
    pass


class Lfm2VlDecoderAdapter(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        weights_dir: str | None = None,
        last_token_only: bool = True,
        return_hidden: bool = False,
    ):
        super().__init__()
        self.model = model
        self.weights_dir = weights_dir
        self.last_token_only = bool(last_token_only)
        self.return_hidden = bool(return_hidden)
        self.backbone = _lfm2_language_backbone(model)
        self.lm_head = getattr(model, "lm_head", None)
        if not isinstance(self.lm_head, torch.nn.Module):
            raise TypeError("LFM2-VL decoder adapter requires an lm_head module")
        from transformers.models.lfm2.modeling_lfm2 import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        causal_mask = _lfm2_causal_mask_for_capture(
            self._create_causal_mask,
            backbone=self.backbone,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = _lfm2_position_embeddings(self.backbone, hidden_states, position_ids=position_ids)
        layer_types = tuple(getattr(self.backbone.config, "layer_types", ()))
        linear_attention = attention_mask if inputs_embeds.shape[1] != 1 else None

        for layer_index, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            layer_type = layer_types[layer_index] if layer_index < len(layer_types) else "full_attention"
            layer_mask = causal_mask if layer_type == "full_attention" else linear_attention
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )

        hidden_states = self.backbone.embedding_norm(hidden_states)
        if self.return_hidden:
            return hidden_states
        if self.last_token_only:
            hidden_states = hidden_states[:, -1:, :]
        return self.lm_head(hidden_states)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="lfm2_vl",
                    adapter_type=type(self).__name__,
                    input_names=("inputs_embeds", "attention_mask", "position_ids"),
                ),
                "weights_dir": self.weights_dir,
            }
        }


class Lfm2VlMultimodalCausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...]):
        super().__init__()
        self.model = model
        self.input_names = tuple(input_names)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        kwargs = {
            name: value
            for name, value in zip(self.input_names, args, strict=True)
        }
        outputs = self.model(
            **kwargs,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        return _extract_tensor_output(outputs, preferred_field="logits")

    def get_transpile_metadata(self):
        return {
            "graph": _transpile_graph_meta(
                self.model,
                adapter_family="lfm2_vl",
                adapter_type=type(self).__name__,
                input_names=self.input_names,
            ),
        }


class GemmaCausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = model.model
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        from transformers.models.gemma.modeling_gemma import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = self._create_causal_mask(
            self.backbone.config,
            inputs_embeds,
            None,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.backbone.rotary_emb(hidden_states, position_ids=position_ids)
        checkpoints: list[torch.Tensor] = []

        for decoder_layer in self.backbone.layers[: self.backbone.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        hidden_states = _select_last_non_pad_token(
            hidden_states,
            input_ids,
            pad_token_id=self.pad_token_id,
        )
        checkpoints.append(hidden_states)
        return _gemma4_apply_final_logit_softcapping(self.model, self.model.lm_head(hidden_states)), checkpoints

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="gemma",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
            }
        }


def _gemma3_text_backbone_forward(backbone, inputs_embeds, causal_mask_mapping, position_ids,
                                  checkpoints=None):
    hidden_states = inputs_embeds
    layer_types = backbone.config.layer_types
    position_embeddings = {
        layer_type: backbone.rotary_emb(hidden_states, position_ids, layer_type)
        for layer_type in set(layer_types)
    }
    for i, decoder_layer in enumerate(backbone.layers[: backbone.config.num_hidden_layers]):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[layer_types[i]],
            position_embeddings=position_embeddings[layer_types[i]],
            position_ids=position_ids,
            past_key_values=None,
        )
        if checkpoints is not None:
            checkpoints.append(hidden_states)
    return backbone.norm(hidden_states)


def _gemma3_decoder_graph_meta(model, backbone, adapter_type, input_names):
    sliding_window = getattr(backbone.config, "sliding_window", None)
    return {
        "graph": {
            **_transpile_graph_meta(
                model,
                adapter_family="gemma3",
                adapter_type=adapter_type,
                input_names=input_names,
            ),
            "num_hidden_layers": int(backbone.config.num_hidden_layers),
            "layer_types": tuple(getattr(backbone.config, "layer_types", [])),
            "sliding_window": None if sliding_window is None else int(sliding_window),
        }
    }


class Gemma3CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = model.model
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        from transformers.models.gemma3.modeling_gemma3 import create_causal_mask  # type: ignore
        from transformers.models.gemma3.modeling_gemma3 import create_sliding_window_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask
        self._create_sliding_window_causal_mask = create_sliding_window_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask_mapping = {
            "full_attention": self._create_causal_mask(
                self.backbone.config,
                inputs_embeds,
                None,
                None,
                past_key_values=None,
                position_ids=position_ids,
            ),
            "sliding_attention": self._create_sliding_window_causal_mask(
                self.backbone.config,
                inputs_embeds,
                None,
                None,
                past_key_values=None,
                position_ids=position_ids,
            ),
        }

        checkpoints: list[torch.Tensor] = []
        hidden_states = _gemma3_text_backbone_forward(
            self.backbone, inputs_embeds, causal_mask_mapping, position_ids, checkpoints=checkpoints,
        )
        checkpoints.append(hidden_states)
        return self.model.lm_head(hidden_states), checkpoints

    def get_transpile_metadata(self):
        return _gemma3_decoder_graph_meta(self.model, self.backbone, type(self).__name__, ("input_ids",))


class Gemma3LMEncoderStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.model

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        return inputs_embeds.to(torch.float16), position_ids.to(dtype=torch.int64)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.backbone,
                    adapter_family="gemma3",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "position_ids"),
                ),
            }
        }


class Gemma3EmbedsCausalLMStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = model.model

    def _additive_mask_mapping(self, inputs_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = int(inputs_embeds.shape[1])
        zero_mask = torch.zeros(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        return {layer_type: zero_mask for layer_type in set(self.backbone.config.layer_types)}

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds.to(self.backbone.norm.weight.dtype)
        causal_mask_mapping = self._additive_mask_mapping(hidden_states)
        text_position_ids = position_ids.to(dtype=torch.int64)
        hidden_states = _gemma3_text_backbone_forward(
            self.backbone, hidden_states, causal_mask_mapping, text_position_ids,
        )
        logits = self.model.lm_head(hidden_states[:, -1:, :])
        return _gemma4_apply_final_logit_softcapping(self.model, logits)

    def get_transpile_metadata(self):
        return _gemma3_decoder_graph_meta(self.model, self.backbone, type(self).__name__, ("inputs_embeds", "position_ids"))


class Gemma3EmbedsCausalLMPrefillChunkAdapter(Gemma3EmbedsCausalLMStepAdapter):
    def _additive_mask_mapping(self, inputs_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = int(inputs_embeds.shape[1])
        blocked_values = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ) * torch.finfo(inputs_embeds.dtype).min
        allowed_values = torch.zeros(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        causal = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=inputs_embeds.device),
        ).view(1, 1, seq_len, seq_len)
        mapping = {"full_attention": torch.where(causal, allowed_values, blocked_values)}
        if "sliding_attention" in set(self.backbone.config.layer_types):
            window = int(self.backbone.config.sliding_window)
            below_window = torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=inputs_embeds.device),
                diagonal=-window,
            ).view(1, 1, seq_len, seq_len)
            mapping["sliding_attention"] = torch.where(
                below_window, blocked_values, mapping["full_attention"],
            )
        return mapping


class Gemma4CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        _patch_gemma4_moe_layers(model)
        self.model = model
        model_backbone = model.model
        self.backbone = getattr(model_backbone, "language_model", model_backbone)
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        from transformers.models.gemma4.modeling_gemma4 import create_causal_mask  # type: ignore
        from transformers.models.gemma4.modeling_gemma4 import create_sliding_window_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask
        self._create_sliding_window_causal_mask = create_sliding_window_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        per_layer_inputs = None
        if self.backbone.hidden_size_per_layer_input:
            per_layer_inputs = _gemma4_get_per_layer_inputs(self.backbone, input_ids, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        mask_kwargs = {
            "config": self.backbone.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": self._create_causal_mask(**mask_kwargs),
            "sliding_attention": self._create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        checkpoints: list[torch.Tensor] = []
        position_embeddings = {
            layer_type: self.backbone.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.backbone.unique_layer_types
        }
        shared_kv_states: dict[str, torch.Tensor] = {}

        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.backbone.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.backbone.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=None,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        checkpoints.append(hidden_states)
        return self.model.lm_head(hidden_states), checkpoints

    def debug_first_block(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        per_layer_inputs = None
        if self.backbone.hidden_size_per_layer_input:
            per_layer_inputs = _gemma4_get_per_layer_inputs(self.backbone, input_ids, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        mask_kwargs = {
            "config": self.backbone.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": self._create_causal_mask(**mask_kwargs),
            "sliding_attention": self._create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.backbone.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.backbone.unique_layer_types
        }
        shared_kv_states: dict[str, torch.Tensor] = {}
        layer = self.backbone.layers[0]
        layer_type = self.backbone.config.layer_types[0]
        per_layer_input = per_layer_inputs[:, :, 0, :] if per_layer_inputs is not None else None

        checkpoints: dict[str, torch.Tensor] = {}

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        checkpoints["pre_attn_norm"] = normed

        attn_out = layer.self_attn(
            normed,
            position_embeddings=position_embeddings[layer_type],
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids,
            past_key_values=None,
            shared_kv_states=shared_kv_states,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        checkpoints["attn_o_proj"] = attn_out

        post_attn_norm = layer.post_attention_layernorm(attn_out)
        checkpoints["post_attn_norm"] = post_attn_norm

        after_attention = residual + post_attn_norm
        checkpoints["after_attention_residual"] = after_attention

        pre_ffn_norm = layer.pre_feedforward_layernorm(after_attention)
        checkpoints["pre_ffn_norm"] = pre_ffn_norm

        mlp_out = layer.mlp(pre_ffn_norm)
        checkpoints["mlp_down"] = mlp_out

        post_ffn_norm = layer.post_feedforward_layernorm(mlp_out)
        checkpoints["post_ffn_norm"] = post_ffn_norm

        after_ffn = after_attention + post_ffn_norm
        checkpoints["after_ffn_residual"] = after_ffn

        if per_layer_input is not None:
            gated = layer.per_layer_input_gate(after_ffn)
            gated = layer.act_fn(gated)
            projected = gated * per_layer_input
            per_layer_proj = layer.per_layer_projection(projected)
            checkpoints["per_layer_input_proj"] = per_layer_proj
            post_per_layer_input_norm = layer.post_per_layer_input_norm(per_layer_proj)
            checkpoints["post_per_layer_input_norm"] = post_per_layer_input_norm
            after_ffn = after_ffn + post_per_layer_input_norm

        layer_scalar = getattr(layer, "layer_scalar", None)
        if layer_scalar is not None:
            after_ffn = after_ffn * layer_scalar
        checkpoints["layer_scalar_out"] = after_ffn
        return checkpoints

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="gemma4",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            }
        }


def _gemma4_apply_final_logit_softcapping(model: torch.nn.Module, logits: torch.Tensor) -> torch.Tensor:
    config = getattr(model, "config", None)
    text_config = None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except Exception:
            text_config = None
    cap = getattr(text_config, "final_logit_softcapping", None)
    if cap is None:
        cap = getattr(config, "final_logit_softcapping", None)
    if cap is None:
        return logits
    cap_value = float(cap)
    if cap_value <= 0.0:
        return logits
    return torch.tanh(logits / cap_value) * cap_value


class Gemma4MultimodalCausalLMLogitsAdapter(BoundInputAdapter):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], weights_dir: str | None = None):
        _patch_gemma4_moe_layers(model)
        super().__init__(
            model,
            input_names=input_names,
            family="gemma4",
            metadata_task="multimodal_causal_lm_logits",
        )
        model_backbone = model.model
        self.multimodal_backbone = model_backbone
        self.backbone = getattr(model_backbone, "language_model", model_backbone)
        self.last_token_logits_only = False
        self._use_cached_multimodal_features = False
        self._native_merge_plan: _Gemma4NativeMergePlan | None = None
        self.register_buffer("_cached_image_features", torch.empty(0), persistent=False)
        self.register_buffer("_cached_audio_features", torch.empty(0), persistent=False)
        self.register_buffer("_native_merge_pli_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        vision_post_proj_norm = _gemma4_load_vision_post_proj_norm(weights_dir)
        if vision_post_proj_norm is None:
            vision_post_proj_norm = torch.empty(0)
        self.register_buffer("_cactus_vision_post_proj_norm", vision_post_proj_norm, persistent=False)
        self._capture_cpu_float32_text_modules: list[tuple[torch.nn.Module, torch.dtype]] = []
        self._create_causal_mask_mapping = None
        self._create_masks_for_generate = None
        self._create_causal_mask = None
        self._create_sliding_window_causal_mask = None
        try:
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_sliding_window_causal_mask  # type: ignore

            self._create_causal_mask = create_causal_mask
            self._create_causal_mask_mapping = create_causal_mask_mapping
            self._create_masks_for_generate = create_masks_for_generate
            self._create_sliding_window_causal_mask = create_sliding_window_causal_mask
        except Exception:
            pass

    def _apply_final_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        return _gemma4_apply_final_logit_softcapping(self.model, logits)

    def _capture_text_modules(self) -> tuple[torch.nn.Module, ...]:
        return _unique_modules(
            self.backbone,
            getattr(self.model, "get_input_embeddings", lambda: None)(),
            getattr(self.model, "lm_head", None),
        )

    def _compute_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor | None,
        get_image_features: Callable[..., object],
    ) -> torch.Tensor:
        multimodal_backbone = self.multimodal_backbone
        if _gemma4_can_use_native_like_vision_features(multimodal_backbone):
            return _gemma4_compute_native_like_image_features(
                multimodal_backbone,
                pixel_values,
                pixel_position_ids,
                post_proj_norm_weight=self._cactus_vision_post_proj_norm,
            )
        vision_tower = getattr(multimodal_backbone, "vision_tower", None)
        embed_vision = getattr(multimodal_backbone, "embed_vision", None)
        vision_modules = _gemma4_vision_modules(multimodal_backbone)
        if (
            pixel_values.device.type != "cpu"
            or pixel_values.dtype != torch.float16
            or len(vision_modules) != 2
            or _torch_is_compiling()
        ):
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        vision_dtype = _module_floating_dtype(vision_tower)
        embed_dtype = _module_floating_dtype(embed_vision)
        if vision_dtype != torch.float16 and embed_dtype != torch.float16:
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        # Gemma4's CPU float16 vision path can emit non-finite soft tokens; upcast only
        # the static image feature extraction path and restore the original module dtypes.
        with _temporary_cpu_float32_modules(vision_modules):
            return get_image_features(
                pixel_values.float(),
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

    def _compute_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        get_audio_features: Callable[..., object],
    ) -> torch.Tensor:
        multimodal_backbone = self.multimodal_backbone
        if _gemma4_can_use_native_like_audio_features(multimodal_backbone):
            return _gemma4_compute_native_like_audio_features(
                multimodal_backbone,
                input_features,
                input_features_mask,
            )
        audio_modules = _gemma4_audio_modules(multimodal_backbone)
        if (
            input_features.device.type != "cpu"
            or input_features.dtype != torch.float16
            or len(audio_modules) != 2
            or _torch_is_compiling()
        ):
            audio_output = get_audio_features(input_features, input_features_mask, return_dict=True)
            return _gemma4_strip_audio_padding(audio_output)

        if all(_module_floating_dtype(module) != torch.float16 for module in audio_modules):
            audio_output = get_audio_features(input_features, input_features_mask, return_dict=True)
            return _gemma4_strip_audio_padding(audio_output)

        with _temporary_cpu_float32_modules(audio_modules):
            audio_output = get_audio_features(input_features.float(), input_features_mask, return_dict=True)
            return _gemma4_strip_audio_padding(audio_output)

    def prepare_cpu_float32_capture(self) -> None:
        self._capture_cpu_float32_text_modules.clear()
        if os.environ.get("CACTUS_GEMMA4_CAPTURE_FP32") != "1":
            return
        for module in self._capture_text_modules():
            device = _module_device(module)
            dtype = _module_floating_dtype(module)
            if device is None or device.type != "cpu" or dtype is None or dtype == torch.float32:
                continue
            self._capture_cpu_float32_text_modules.append((module, dtype))
            module.to(dtype=torch.float32)

    def restore_cpu_float32_capture(self) -> None:
        for module, dtype in reversed(self._capture_cpu_float32_text_modules):
            module.to(dtype=dtype)
        self._capture_cpu_float32_text_modules.clear()

    def _resolve_image_features(
        self,
        *,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor | None,
        pixel_position_ids: torch.Tensor | None,
        get_image_features: Callable[..., object],
    ) -> torch.Tensor | None:
        if pixel_values is None:
            return None
        if self._use_cached_multimodal_features and self._cached_image_features.numel() > 0:
            image_features = self._cached_image_features
        else:
            image_features = self._compute_image_features(
                pixel_values,
                pixel_position_ids,
                get_image_features,
            )
        return image_features.to(inputs_embeds.device, inputs_embeds.dtype)

    def _resolve_audio_features(
        self,
        *,
        inputs_embeds: torch.Tensor,
        input_features: torch.Tensor | None,
        input_features_mask: torch.Tensor | None,
        get_audio_features: Callable[..., object],
    ) -> torch.Tensor | None:
        if input_features is None or input_features_mask is None:
            return None
        if self._use_cached_multimodal_features and self._cached_audio_features.numel() > 0:
            audio_features = self._cached_audio_features
        else:
            audio_features = self._compute_audio_features(
                input_features,
                input_features_mask,
                get_audio_features,
            )
        return audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

    def _prepare_text_backbone_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        pixel_position_ids: torch.Tensor | None,
        input_features: torch.Tensor | None,
        input_features_mask: torch.Tensor | None,
        get_placeholder_mask: Callable[..., object],
        get_image_features: Callable[..., object],
        get_audio_features: Callable[..., object],
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor], torch.LongTensor]:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        image_features = self._resolve_image_features(
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            get_image_features=get_image_features,
        )
        audio_features = self._resolve_audio_features(
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            input_features_mask=input_features_mask,
            get_audio_features=get_audio_features,
        )

        if image_features is not None and audio_features is not None and self._native_merge_plan is not None:
            inputs_embeds, per_layer_inputs_tokens = _gemma4_apply_native_merge_plan(
                self.model,
                input_ids=input_ids,
                image_features=image_features,
                audio_features=audio_features,
                merge_plan=self._native_merge_plan,
                pli_token_ids=self._native_merge_pli_token_ids,
            )
            attention_mask = _gemma4_remap_sequence_tensor(
                attention_mask,
                merge_plan=self._native_merge_plan,
            )
            token_type_ids = _gemma4_remap_sequence_tensor(
                token_type_ids,
                merge_plan=self._native_merge_plan,
            )
            if token_type_ids is None:
                raise RuntimeError("Gemma4 native merge requires token_type_ids for multimodal attention masking")
        else:
            text_mask, image_mask, audio_mask = _gemma4_get_placeholder_masks(
                get_placeholder_mask,
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )
            if image_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    image_features,
                )
            if audio_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    audio_features,
                )
            per_layer_inputs_tokens = input_ids * text_mask.to(dtype=input_ids.dtype)

        per_layer_inputs = None
        text_config = _gemma4_text_config(self.multimodal_backbone)
        if getattr(text_config, "hidden_size_per_layer_input", None):
            per_layer_inputs = _gemma4_get_per_layer_inputs(self.backbone, per_layer_inputs_tokens, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        if self._native_merge_plan is not None:
            causal_mask_mapping = _gemma4_build_standard_causal_mask_mapping(
                create_causal_mask=self._create_causal_mask,
                create_sliding_window_causal_mask=self._create_sliding_window_causal_mask,
                config=self.backbone.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        elif getattr(text_config, "use_bidirectional_attention", None) == "vision":
            causal_mask_mapping = self._create_causal_mask_mapping(
                self.multimodal_backbone.config,
                inputs_embeds,
                attention_mask,
                None,
                position_ids,
                token_type_ids,
                pixel_values,
                is_training=self.training,
            )
        else:
            causal_mask_mapping = self._create_masks_for_generate(
                self.multimodal_backbone.config,
                inputs_embeds,
                attention_mask,
                None,
                position_ids,
            )
        return inputs_embeds, per_layer_inputs, causal_mask_mapping, position_ids

    def _forward_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        pixel_position_ids: torch.Tensor | None,
        input_features: torch.Tensor | None,
        input_features_mask: torch.Tensor | None,
        get_placeholder_mask: Callable[..., object],
        get_image_features: Callable[..., object],
        get_audio_features: Callable[..., object],
    ) -> torch.Tensor:
        inputs_embeds, per_layer_inputs, causal_mask_mapping, position_ids = self._prepare_text_backbone_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            get_placeholder_mask=get_placeholder_mask,
            get_image_features=get_image_features,
            get_audio_features=get_audio_features,
        )
        return _gemma4_text_backbone_forward(
            self.backbone,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
        )

    def prime_static_multimodal_features(self, *bound_inputs: torch.Tensor | None) -> None:
        kwargs = self._kwargs_from_bound_inputs(*bound_inputs)
        input_ids = kwargs["input_ids"]
        token_type_ids = kwargs.get("token_type_ids")
        pixel_values = kwargs.get("pixel_values")
        pixel_position_ids = kwargs.get("pixel_position_ids")
        input_features = kwargs.get("input_features")
        input_features_mask = kwargs.get("input_features_mask")

        multimodal_backbone = self.multimodal_backbone
        get_placeholder_mask = getattr(multimodal_backbone, "get_placeholder_mask", None)
        get_image_features = getattr(multimodal_backbone, "get_image_features", None)
        get_audio_features = getattr(multimodal_backbone, "get_audio_features", None)
        if not callable(get_placeholder_mask):
            return

        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            _, image_mask, audio_mask = _gemma4_get_placeholder_masks(
                get_placeholder_mask,
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )

            if pixel_values is not None and callable(get_image_features) and image_mask.any():
                image_features = self._compute_image_features(
                    pixel_values,
                    pixel_position_ids,
                    get_image_features,
                )
                self._cached_image_features = image_features.detach()
            else:
                self._cached_image_features = self._cached_image_features.new_empty(0)

            if (
                input_features is not None
                and input_features_mask is not None
                and callable(get_audio_features)
                and audio_mask.any()
            ):
                self._cached_audio_features = self._compute_audio_features(
                    input_features,
                    input_features_mask,
                    get_audio_features,
                ).detach()
            else:
                self._cached_audio_features = self._cached_audio_features.new_empty(0)

            if self._cached_image_features.numel() > 0 and self._cached_audio_features.numel() > 0:
                plan = _gemma4_build_native_merge_plan(
                    self.multimodal_backbone,
                    input_ids,
                    image_feature_count=_gemma4_feature_token_count(self._cached_image_features),
                    audio_feature_count=_gemma4_feature_token_count(self._cached_audio_features),
                )
                self._native_merge_plan = plan
                self._native_merge_pli_token_ids = torch.tensor(
                    plan.pli_token_ids,
                    dtype=torch.long,
                )
            else:
                self._native_merge_plan = None
                self._native_merge_pli_token_ids = self._native_merge_pli_token_ids.new_empty(0)

        self._use_cached_multimodal_features = True

    def forward(self, *bound_inputs: torch.Tensor | None) -> torch.Tensor:
        kwargs = self._kwargs_from_bound_inputs(*bound_inputs)
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask")
        token_type_ids = kwargs.get("token_type_ids")
        pixel_values = kwargs.get("pixel_values")
        pixel_position_ids = kwargs.get("pixel_position_ids")
        input_features = kwargs.get("input_features")
        input_features_mask = kwargs.get("input_features_mask")

        multimodal_backbone = self.multimodal_backbone
        get_placeholder_mask = getattr(multimodal_backbone, "get_placeholder_mask", None)
        get_image_features = getattr(multimodal_backbone, "get_image_features", None)
        get_audio_features = getattr(multimodal_backbone, "get_audio_features", None)
        lm_head = getattr(self.model, "lm_head", None)
        if (
            not callable(get_placeholder_mask)
            or not callable(get_image_features)
            or not callable(get_audio_features)
            or not callable(self._create_causal_mask_mapping)
            or not callable(self._create_masks_for_generate)
            or not isinstance(lm_head, torch.nn.Module)
        ):
            outputs = self.model(
                return_dict=True,
                use_cache=False,
                **kwargs,
            )
            logits = _extract_tensor_output(outputs, preferred_field="logits")
            if self.last_token_logits_only and logits.ndim >= 3:
                return _select_last_active_token(logits, attention_mask)
            return logits

        hidden_states = self._forward_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            get_placeholder_mask=get_placeholder_mask,
            get_image_features=get_image_features,
            get_audio_features=get_audio_features,
        )
        if self.last_token_logits_only:
            hidden_states = _select_last_active_token(hidden_states, attention_mask)
        return self._apply_final_logit_softcapping(lm_head(hidden_states))

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="gemma4",
                    adapter_type=type(self).__name__,
                    input_names=self.input_names,
                ),
                "task": self.metadata_task,
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
                "last_token_logits_only": bool(self.last_token_logits_only),
            }
        }


_GEMMA4_DECODER_PIPELINE_IO_KEYS = (
    "inputs_embeds",
    "per_layer_inputs",
    "position_ids",
)


class _Gemma4MultimodalComponentBase(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        input_names: tuple[str, ...],
        weights_dir: str | None = None,
        native_merge_plan: _Gemma4NativeMergePlan | None = None,
        native_image_soft_token_counts: tuple[int, ...] | None = None,
        native_image_pool_shapes: tuple[tuple[int, int, int], ...] | None = None,
    ):
        super().__init__()
        _patch_gemma4_moe_layers(model)
        self.model = model
        self.input_names = input_names
        self.multimodal_backbone = model.model
        model_backbone = model.model
        self.backbone = getattr(model_backbone, "language_model", model_backbone)
        self._native_merge_plan = native_merge_plan
        self._native_image_soft_token_counts = native_image_soft_token_counts
        self._native_image_pool_shapes = native_image_pool_shapes
        self.register_buffer(
            "_native_merge_pli_token_ids",
            torch.tensor(native_merge_plan.pli_token_ids, dtype=torch.long)
            if native_merge_plan is not None
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        vision_post_proj_norm = _gemma4_load_vision_post_proj_norm(weights_dir)
        if vision_post_proj_norm is None:
            vision_post_proj_norm = torch.empty(0)
        self.register_buffer("_cactus_vision_post_proj_norm", vision_post_proj_norm, persistent=False)
        self._create_causal_mask_mapping = None
        self._create_masks_for_generate = None
        self._create_causal_mask = None
        self._create_sliding_window_causal_mask = None
        try:
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_sliding_window_causal_mask  # type: ignore

            self._create_causal_mask = create_causal_mask
            self._create_causal_mask_mapping = create_causal_mask_mapping
            self._create_masks_for_generate = create_masks_for_generate
            self._create_sliding_window_causal_mask = create_sliding_window_causal_mask
        except Exception:
            pass
        self._capture_modules: list[tuple[torch.nn.Module, torch.dtype]] = []

    def _base_graph_meta(self, *, adapter_type: str, input_names: tuple[str, ...]) -> dict[str, object]:
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            **_transpile_graph_meta(
                self.model,
                adapter_family="gemma4",
                adapter_type=adapter_type,
                input_names=input_names,
            ),
            "task": "multimodal_causal_lm_logits",
            "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
            "layer_types": tuple(layer_types),
            "sliding_window": None if sliding_window is None else int(sliding_window),
        }

    def _compute_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if _gemma4_can_use_native_like_vision_features(self.multimodal_backbone):
            return _gemma4_compute_native_like_image_features(
                self.multimodal_backbone,
                pixel_values,
                pixel_position_ids,
                post_proj_norm_weight=self._cactus_vision_post_proj_norm,
                image_soft_token_counts=self._native_image_soft_token_counts,
                image_pool_shapes=self._native_image_pool_shapes,
            )
        get_image_features = getattr(self.multimodal_backbone, "get_image_features", None)
        vision_tower = getattr(self.multimodal_backbone, "vision_tower", None)
        embed_vision = getattr(self.multimodal_backbone, "embed_vision", None)
        vision_modules = _gemma4_vision_modules(self.multimodal_backbone)
        if not callable(get_image_features):
            raise TypeError("Gemma4 multimodal backbone is missing get_image_features")
        if (
            pixel_values.device.type != "cpu"
            or pixel_values.dtype != torch.float16
            or len(vision_modules) != 2
            or _torch_is_compiling()
        ):
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        vision_dtype = _module_floating_dtype(vision_tower)
        embed_dtype = _module_floating_dtype(embed_vision)
        if vision_dtype != torch.float16 and embed_dtype != torch.float16:
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        with _temporary_cpu_float32_modules(vision_modules):
            return get_image_features(
                pixel_values.float(),
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

    def _compute_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> torch.Tensor:
        if _gemma4_can_use_native_like_audio_features(self.multimodal_backbone):
            return _gemma4_compute_native_like_audio_features(
                self.multimodal_backbone,
                input_features,
                input_features_mask,
            )
        get_audio_features = getattr(self.multimodal_backbone, "get_audio_features", None)
        if not callable(get_audio_features):
            raise TypeError("Gemma4 multimodal backbone is missing get_audio_features")
        audio_modules = _gemma4_audio_modules(self.multimodal_backbone)
        if (
            input_features.device.type == "cpu"
            and input_features.dtype == torch.float16
            and len(audio_modules) == 2
            and not _torch_is_compiling()
            and any(_module_floating_dtype(module) == torch.float16 for module in audio_modules)
        ):
            with _temporary_cpu_float32_modules(audio_modules):
                audio_output = get_audio_features(input_features.float(), input_features_mask, return_dict=True)
                return _gemma4_strip_audio_padding(audio_output)

        audio_output = get_audio_features(input_features, input_features_mask, return_dict=True)
        return _gemma4_strip_audio_padding(audio_output)

    def _modules_to_prepare_for_capture(self) -> tuple[torch.nn.Module, ...]:
        return ()

    def prepare_for_capture(self, **_: object) -> None:
        self.restore_after_capture()
        modules = self._modules_to_prepare_for_capture()
        if not modules:
            return
        promoted: list[tuple[torch.nn.Module, torch.dtype]] = []
        for module in modules:
            device = _module_device(module)
            dtype = _module_floating_dtype(module)
            if device is None or device.type != "cpu" or dtype is None or dtype == torch.float32:
                continue
            promoted.append((module, dtype))
            module.to(dtype=torch.float32)
        self._capture_modules = promoted

    def restore_after_capture(self, **_: object) -> None:
        for module, dtype in reversed(self._capture_modules):
            module.to(dtype=dtype)
        self._capture_modules.clear()

    def _prepare_decoder_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self._native_merge_plan is not None:
            audio_features_for_merge = audio_features
            if audio_features_for_merge is None:
                audio_features_for_merge = image_features.new_empty(
                    (int(input_ids.shape[0]), 0, int(image_features.shape[-1]))
                )
            inputs_embeds, per_layer_inputs_tokens = _gemma4_apply_native_merge_plan(
                self.model,
                input_ids=input_ids,
                image_features=image_features,
                audio_features=audio_features_for_merge,
                merge_plan=self._native_merge_plan,
                pli_token_ids=self._native_merge_pli_token_ids,
            )
            attention_mask = _gemma4_remap_sequence_tensor(
                attention_mask,
                merge_plan=self._native_merge_plan,
            )
            token_type_ids = _gemma4_remap_sequence_tensor(
                token_type_ids,
                merge_plan=self._native_merge_plan,
            )
            if token_type_ids is None:
                raise RuntimeError("Gemma4 native merge requires token_type_ids for multimodal attention masking")
        else:
            get_placeholder_mask = getattr(self.multimodal_backbone, "get_placeholder_mask", None)
            if not callable(get_placeholder_mask):
                raise TypeError("Gemma4 multimodal backbone is missing get_placeholder_mask")

            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            text_mask, image_mask, audio_mask = _gemma4_get_placeholder_masks(
                get_placeholder_mask,
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_features.to(inputs_embeds.device, inputs_embeds.dtype),
            )
            if audio_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    audio_features.to(inputs_embeds.device, inputs_embeds.dtype),
                )
            per_layer_inputs_tokens = input_ids * text_mask.to(dtype=input_ids.dtype)

        per_layer_inputs: torch.Tensor
        text_config = _gemma4_text_config(self.multimodal_backbone)
        if getattr(text_config, "hidden_size_per_layer_input", None):
            per_layer_inputs = _gemma4_get_per_layer_inputs(self.backbone, per_layer_inputs_tokens, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        else:
            per_layer_inputs = inputs_embeds.new_empty((inputs_embeds.shape[0], inputs_embeds.shape[1], 0, 0))

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        return (
            inputs_embeds,
            per_layer_inputs,
            position_ids,
        )

    def _prepare_text_decoder_step_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        embedding = self.model.get_input_embeddings()
        if not isinstance(embedding, torch.nn.Module):
            raise TypeError("Gemma4 model is missing input embeddings")
        inputs_embeds = embedding(input_ids)

        model_config = getattr(self.model, "config", None)
        text_config = getattr(model_config, "text_config", None)
        hidden_scale = float(getattr(model_config, "hidden_size", 0) or 0)
        if hidden_scale <= 0.0:
            hidden_scale = float(getattr(text_config, "hidden_size", 0) or 0)
        if hidden_scale <= 0.0:
            hidden_scale = float(inputs_embeds.shape[-1])
        text_extra_scale = _gemma4_text_embedding_scale(embedding, hidden_scale ** 0.5)
        if text_extra_scale != 1.0:
            inputs_embeds = inputs_embeds * text_extra_scale

        text_config = _gemma4_text_config(self.multimodal_backbone)
        if getattr(text_config, "hidden_size_per_layer_input", None):
            per_layer_inputs = _gemma4_get_per_layer_inputs(self.backbone, input_ids, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        else:
            per_layer_inputs = inputs_embeds.new_empty((inputs_embeds.shape[0], inputs_embeds.shape[1], 0, 0))
        return inputs_embeds, per_layer_inputs, position_ids


class Gemma4VisionEncoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        weights_dir: str | None = None,
        native_image_soft_token_counts: tuple[int, ...] | None = None,
        native_image_pool_shapes: tuple[tuple[int, int, int], ...] | None = None,
    ):
        super().__init__(
            model,
            input_names=("pixel_values", "pixel_position_ids"),
            weights_dir=weights_dir,
            native_image_soft_token_counts=native_image_soft_token_counts,
            native_image_pool_shapes=native_image_pool_shapes,
        )

    def _modules_to_prepare_for_capture(self) -> tuple[torch.nn.Module, ...]:
        return ()

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor | None) -> torch.Tensor:
        return self._compute_image_features(pixel_values, pixel_position_ids)

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=("pixel_values", "pixel_position_ids"),
            ),
        }


class Gemma4AudioEncoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__(model, input_names=("input_features", "input_features_mask"), weights_dir=weights_dir)

    def _modules_to_prepare_for_capture(self) -> tuple[torch.nn.Module, ...]:
        return ()

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        return self._compute_audio_features(input_features, input_features_mask)

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=("input_features", "input_features_mask"),
            ),
        }


class Gemma4LMEncoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        weights_dir: str | None = None,
        native_merge_plan: _Gemma4NativeMergePlan | None = None,
        include_audio: bool = True,
    ):
        self.include_audio = bool(include_audio)
        input_names = ("input_ids", "attention_mask", "token_type_ids", "image_features")
        if self.include_audio:
            input_names = (*input_names, "audio_features")
        super().__init__(
            model,
            input_names=input_names,
            weights_dir=weights_dir,
            native_merge_plan=native_merge_plan,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        return self._prepare_decoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_features=image_features,
            audio_features=audio_features,
        )

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=self.input_names,
            ),
        }


class Gemma4LMEncoderStepAdapter(_Gemma4MultimodalComponentBase):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__(model, input_names=("input_ids", "position_ids"), weights_dir=weights_dir)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        return self._prepare_text_decoder_step_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
        )

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=("input_ids", "position_ids"),
            ),
        }


class Gemma4LMEncoderTextChunkAdapter(Gemma4LMEncoderStepAdapter):
    pass


class Gemma4LMEncoderMediaStepAdapter(_Gemma4MultimodalComponentBase):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__(
            model,
            input_names=("inputs_embeds", "input_ids", "position_ids"),
            weights_dir=weights_dir,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        text_config = _gemma4_text_config(self.multimodal_backbone)
        if getattr(text_config, "hidden_size_per_layer_input", None):
            per_layer_inputs = _gemma4_get_per_layer_inputs(self.backbone, input_ids, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        else:
            per_layer_inputs = inputs_embeds.new_empty((inputs_embeds.shape[0], inputs_embeds.shape[1], 0, 0))
        return inputs_embeds, per_layer_inputs, position_ids

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=("inputs_embeds", "input_ids", "position_ids"),
            ),
        }


class Gemma4LMEncoderMediaChunkAdapter(Gemma4LMEncoderMediaStepAdapter):
    pass


class Gemma4DecoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__(model, input_names=_GEMMA4_DECODER_PIPELINE_IO_KEYS, weights_dir=weights_dir)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        normalized_per_layer_inputs = per_layer_inputs
        if normalized_per_layer_inputs.numel() == 0:
            normalized_per_layer_inputs = None
        attention_mask = torch.ones(
            position_ids.shape,
            dtype=torch.long,
            device=position_ids.device,
        )
        causal_mask_mapping = _gemma4_build_standard_causal_mask_mapping(
            create_causal_mask=self._create_causal_mask,
            create_sliding_window_causal_mask=self._create_sliding_window_causal_mask,
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = _gemma4_text_backbone_forward(
            self.backbone,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=normalized_per_layer_inputs,
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
        )
        logits = self.model.lm_head(hidden_states[:, -1:, :])
        return _gemma4_apply_final_logit_softcapping(self.model, logits)

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            )
        }


class Gemma4DecoderStepAdapter(Gemma4DecoderAdapter):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_per_layer_inputs = per_layer_inputs
        if normalized_per_layer_inputs.numel() == 0:
            normalized_per_layer_inputs = None
        attention_mask = torch.ones(
            position_ids.shape,
            dtype=torch.long,
            device=position_ids.device,
        )
        causal_mask_mapping = _gemma4_build_standard_causal_mask_mapping(
            create_causal_mask=self._create_causal_mask,
            create_sliding_window_causal_mask=self._create_sliding_window_causal_mask,
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states, probe_hidden = _gemma4_text_backbone_forward(
            self.backbone,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=normalized_per_layer_inputs,
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
            capture_layer_index=28,
        )
        logits = self.model.lm_head(hidden_states[:, -1:, :])
        logits = _gemma4_apply_final_logit_softcapping(self.model, logits)
        return logits, probe_hidden[:, -1:, :]


class Gemma4DecoderPrefillChunkAdapter(Gemma4DecoderAdapter):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        normalized_per_layer_inputs = per_layer_inputs
        if normalized_per_layer_inputs.numel() == 0:
            normalized_per_layer_inputs = None
        attention_mask = torch.ones(
            position_ids.shape,
            dtype=torch.long,
            device=position_ids.device,
        )
        causal_mask_mapping = _gemma4_build_standard_causal_mask_mapping(
            create_causal_mask=self._create_causal_mask,
            create_sliding_window_causal_mask=self._create_sliding_window_causal_mask,
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        num_layers = int(getattr(self.backbone.config, "num_hidden_layers", len(self.backbone.layers)))
        num_shared_layers = int(getattr(self.backbone.config, "num_kv_shared_layers", 0) or 0)
        first_shared_layer = max(0, num_layers - num_shared_layers)
        if first_shared_layer <= 0 or first_shared_layer >= num_layers or int(inputs_embeds.shape[1]) <= 1:
            hidden_states = _gemma4_text_backbone_forward(
                self.backbone,
                inputs_embeds=inputs_embeds,
                per_layer_inputs=normalized_per_layer_inputs,
                causal_mask_mapping=causal_mask_mapping,
                position_ids=position_ids,
            )
        else:
            shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            hidden_states = _gemma4_text_backbone_forward(
                self.backbone,
                inputs_embeds=inputs_embeds,
                per_layer_inputs=normalized_per_layer_inputs,
                causal_mask_mapping=causal_mask_mapping,
                position_ids=position_ids,
                layer_end=first_shared_layer,
                apply_norm=False,
                shared_kv_states=shared_kv_states,
            )
            tail_masks = {
                key: value[:, :, -1:, :] if value is not None and value.ndim == 4 else value
                for key, value in causal_mask_mapping.items()
            }
            tail_per_layer_inputs = (
                None
                if normalized_per_layer_inputs is None
                else normalized_per_layer_inputs[:, -1:, :, :]
            )
            hidden_states = _gemma4_text_backbone_forward(
                self.backbone,
                inputs_embeds=hidden_states[:, -1:, :],
                per_layer_inputs=tail_per_layer_inputs,
                causal_mask_mapping=tail_masks,
                position_ids=position_ids[:, -1:],
                layer_start=first_shared_layer,
                apply_norm=True,
                shared_kv_states=shared_kv_states,
            )
        logits = self.model.lm_head(hidden_states[:, -1:, :])
        return _gemma4_apply_final_logit_softcapping(self.model, logits)


class Gemma4DecoderEmbedChunkAdapter(Gemma4DecoderAdapter):
    """Embedding-readout variant of the prefill chunk: runs the full backbone over
    ALL tokens with final norm (no shared-KV-tail last-token shortcut) and returns
    the full-sequence last hidden state (no lm_head)."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        normalized_per_layer_inputs = per_layer_inputs
        if normalized_per_layer_inputs.numel() == 0:
            normalized_per_layer_inputs = None
        attention_mask = torch.ones(
            position_ids.shape,
            dtype=torch.long,
            device=position_ids.device,
        )
        causal_mask_mapping = _gemma4_build_standard_causal_mask_mapping(
            create_causal_mask=self._create_causal_mask,
            create_sliding_window_causal_mask=self._create_sliding_window_causal_mask,
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return _gemma4_text_backbone_forward(
            self.backbone,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=normalized_per_layer_inputs,
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
        )


def _build_gemma3_causal_lm_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
    dynamic_batch: bool = True,
    max_slots: int = 1,
) -> list[ComponentModuleSpec] | None:
    input_ids = named_tensors.get("input_ids")
    if input_ids is None:
        return None
    pad_token_id = _resolve_model_pad_token_id(model)
    requested = tuple(components or (
        "decoder",
        "lm_encoder_step",
        "lm_encoder_text_chunk",
        "decoder_prefill_chunk",
        "decoder_step",
    ))
    requested_set = set(requested)
    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "causal_lm_logits",
        "adapter_family": "gemma3",
    }
    metadata = {"family": "gemma3", "task": "causal_lm_logits"}

    specs: list[ComponentModuleSpec] = []
    if "decoder" in requested_set:
        decoder = Gemma3CausalLMLogitsAdapter(model, pad_token_id=pad_token_id).eval()
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=(input_ids,),
            input_keys=("input_ids",),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata=metadata,
        ))

    cached_components = {"lm_encoder_step", "lm_encoder_text_chunk", "decoder_prefill_chunk", "decoder_step"}
    if not (cached_components & requested_set):
        return specs

    lm_encoder_step = Gemma3LMEncoderStepAdapter(model).eval()
    lm_encoder_text_chunk = Gemma3LMEncoderStepAdapter(model).eval()
    decoder_step = Gemma3EmbedsCausalLMStepAdapter(model).eval()
    decoder_prefill_chunk = Gemma3EmbedsCausalLMPrefillChunkAdapter(model).eval()
    max_cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512)
    cached_graph_meta = {
        "use_internal_kv_cache": True,
        "max_cache_seq_len": max_cache_seq_len,
        "cache_sink_size": 4,
    }
    prefill_chunk_size = max(2, int(os.environ.get("CACTUS_GEMMA3_PREFILL_CHUNK", "128") or "128"))

    step_input_ids = input_ids[:, :1]
    step_position_ids = torch.zeros_like(step_input_ids, dtype=torch.int64)
    chunk_input_ids = _tile_to_length(input_ids, prefill_chunk_size)
    chunk_position_ids = torch.arange(
        prefill_chunk_size, dtype=torch.int64, device=input_ids.device,
    ).unsqueeze(0)
    with torch.no_grad():
        step_embeds, step_pos_out = lm_encoder_step(step_input_ids, step_position_ids)
        chunk_embeds, chunk_pos_out = lm_encoder_text_chunk(chunk_input_ids, chunk_position_ids)

    if "lm_encoder_step" in requested_set:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_step",
            module=lm_encoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("inputs_embeds", "position_ids"),
            graph_meta={**common_graph_meta, "component": "lm_encoder_step"},
            metadata=metadata,
            dynamic_batch_axis=0 if dynamic_batch else None,
        ))
    if "lm_encoder_text_chunk" in requested_set:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_text_chunk",
            module=lm_encoder_text_chunk,
            example_inputs=(chunk_input_ids, chunk_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("inputs_embeds", "position_ids"),
            graph_meta={
                **common_graph_meta,
                "component": "lm_encoder_text_chunk",
                "encoder_chunk_size": prefill_chunk_size,
            },
            metadata=metadata,
        ))
    if "decoder_prefill_chunk" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder_prefill_chunk",
            module=decoder_prefill_chunk,
            example_inputs=(chunk_embeds, chunk_pos_out),
            input_keys=("inputs_embeds", "position_ids"),
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                **cached_graph_meta,
                "component": "decoder_prefill_chunk",
                "prefill_chunk_size": prefill_chunk_size,
            },
            metadata=metadata,
        ))
    if "decoder_step" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder_step",
            module=decoder_step,
            example_inputs=(step_embeds, step_pos_out),
            input_keys=("inputs_embeds", "position_ids"),
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                **cached_graph_meta,
                "component": "decoder_step",
                "cache_num_slots": (max_slots if dynamic_batch else 1),
            },
            metadata=metadata,
            dynamic_batch_axis=0 if dynamic_batch else None,
        ))
    return specs


def _build_gemma4_causal_lm_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
    dynamic_batch: bool = True,
    max_slots: int = 1,
) -> list[ComponentModuleSpec] | None:
    input_ids = named_tensors.get("input_ids")
    if input_ids is None:
        return None
    pad_token_id = _resolve_model_pad_token_id(model)
    requested_set = set(components or ())
    if not requested_set or "decoder" in requested_set:
        requested_set |= {
            "decoder",
            "lm_encoder_step",
            "lm_encoder_text_chunk",
            "decoder_prefill_chunk",
            "decoder_step",
        }
    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "causal_lm_logits",
        "adapter_family": "gemma4",
    }
    metadata = {"family": "gemma4", "task": "causal_lm_logits"}

    specs: list[ComponentModuleSpec] = []
    if "decoder" in requested_set:
        decoder = Gemma4CausalLMLogitsAdapter(model, pad_token_id=pad_token_id).eval()
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=(input_ids,),
            input_keys=("input_ids",),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata=metadata,
        ))

    cached_components = {"lm_encoder_step", "lm_encoder_text_chunk", "decoder_prefill_chunk", "decoder_step"}
    if not (cached_components & requested_set):
        return specs

    lm_encoder_step = Gemma4LMEncoderStepAdapter(model, weights_dir=weights_dir).eval()
    lm_encoder_text_chunk = Gemma4LMEncoderTextChunkAdapter(model, weights_dir=weights_dir).eval()
    decoder_prefill_chunk = Gemma4DecoderPrefillChunkAdapter(model, weights_dir=weights_dir).eval()
    decoder_step = Gemma4DecoderStepAdapter(model, weights_dir=weights_dir).eval()
    cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512)
    prefill_chunk_size = max(2, int(os.environ.get("CACTUS_GEMMA4_PREFILL_CHUNK", "128") or "128"))

    step_input_ids = input_ids[:, -1:].contiguous()
    step_position_ids = torch.full(
        (int(input_ids.shape[0]), 1),
        max(0, int(input_ids.shape[1]) - 1),
        dtype=torch.long,
        device=input_ids.device,
    )
    chunk_input_ids = _tile_to_length(input_ids, prefill_chunk_size)
    chunk_position_ids = torch.arange(
        prefill_chunk_size, dtype=torch.long, device=input_ids.device,
    ).unsqueeze(0).expand(int(input_ids.shape[0]), -1).contiguous()
    with torch.no_grad():
        decoder_step_inputs = lm_encoder_step(step_input_ids, step_position_ids)
        prefill_decoder_inputs = lm_encoder_text_chunk(chunk_input_ids, chunk_position_ids)

    if "lm_encoder_step" in requested_set:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_step",
            module=lm_encoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            graph_meta={**common_graph_meta, "component": "lm_encoder_step"},
            metadata=metadata,
            dynamic_batch_axis=0 if dynamic_batch else None,
        ))
    if "lm_encoder_text_chunk" in requested_set:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_text_chunk",
            module=lm_encoder_text_chunk,
            example_inputs=(chunk_input_ids, chunk_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            graph_meta={
                **common_graph_meta,
                "component": "lm_encoder_text_chunk",
                "encoder_chunk_size": prefill_chunk_size,
            },
            metadata=metadata,
        ))
    if "decoder_prefill_chunk" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder_prefill_chunk",
            module=decoder_prefill_chunk,
            example_inputs=tuple(prefill_decoder_inputs),
            input_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_prefill_chunk",
                "use_internal_kv_cache": True,
                "max_cache_seq_len": cache_seq_len,
                "cache_sink_size": 4,
                "prefill_chunk_size": prefill_chunk_size,
            },
            metadata=metadata,
        ))
    if "decoder_step" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder_step",
            module=decoder_step,
            example_inputs=tuple(decoder_step_inputs),
            input_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            output_keys=("logits", "probe_hidden"),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_step",
                "use_internal_kv_cache": True,
                "max_cache_seq_len": cache_seq_len,
                "cache_sink_size": 4,
                "cache_num_slots": (max_slots if dynamic_batch else 1),
            },
            metadata=metadata,
            dynamic_batch_axis=0 if dynamic_batch else None,
        ))
    return specs


def _gemma4_make_audio_mask_fp16_safe(model: torch.nn.Module) -> None:
    backbone = getattr(model, "model", model)
    audio_tower = getattr(backbone, "audio_tower", None)
    if audio_tower is None:
        return
    seen: set[int] = set()
    for mod in audio_tower.modules():
        cfg = getattr(mod, "config", None)
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        value = getattr(cfg, "attention_invalid_logits_value", None)
        if isinstance(value, (int, float)) and value < -1e4:
            cfg.attention_invalid_logits_value = -1e4


def _build_gemma4_multimodal_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
    dynamic_batch: bool = True,
    max_slots: int = 1,
) -> list[ComponentModuleSpec]:
    pixel_values = named_tensors.get("pixel_values")
    pixel_position_ids = named_tensors.get("pixel_position_ids")
    input_features = named_tensors.get("input_features")
    input_features_mask = named_tensors.get("input_features_mask")
    input_ids = named_tensors["input_ids"]
    attention_mask = named_tensors.get("attention_mask")
    token_type_ids = named_tensors["token_type_ids"]
    planning_input_ids = input_ids
    low_memory_meta = _module_has_meta_tensors(model)
    multimodal_backbone = getattr(model, "model", model)
    model_has_audio = len(_gemma4_audio_modules(multimodal_backbone)) >= 2

    has_vision_inputs = pixel_values is not None
    has_audio_inputs = model_has_audio and input_features is not None and input_features_mask is not None
    default_components = ["lm_encoder", "decoder"]
    if has_vision_inputs:
        default_components.insert(0, "vision_encoder")
    if has_audio_inputs:
        insert_at = 1 if has_vision_inputs else 0
        default_components.insert(insert_at, "audio_encoder")
    requested_components = tuple(components or tuple(default_components))
    requested_set = set(requested_components)
    if not requested_set:
        return []
    include_audio = "audio_encoder" in requested_set or (components is None and has_audio_inputs)

    expanded_components: list[str] = []

    def _require(component: str) -> None:
        if component not in expanded_components:
            expanded_components.append(component)

    if "decoder" in requested_set:
        if has_vision_inputs or "vision_encoder" in requested_set:
            _require("vision_encoder")
        if include_audio:
            _require("audio_encoder")
        _require("lm_encoder")
        _require("decoder_prefill_chunk")
        _require("decoder_embed_chunk")
        _require("lm_encoder_text_chunk")
        _require("lm_encoder_media_chunk")
        _require("lm_encoder_step")
        _require("lm_encoder_media_step")
        _require("decoder_step")
    if "lm_encoder" in requested_set:
        if has_vision_inputs or "vision_encoder" in requested_set:
            _require("vision_encoder")
        if include_audio:
            _require("audio_encoder")
        _require("lm_encoder")
    if "decoder_prefill_chunk" in requested_set:
        _require("decoder_prefill_chunk")
    if "vision_encoder" in requested_set:
        _require("vision_encoder")
    if "audio_encoder" in requested_set:
        _require("audio_encoder")
    if "lm_encoder_step" in requested_set:
        _require("lm_encoder_step")
    if "lm_encoder_text_chunk" in requested_set:
        _require("lm_encoder_text_chunk")
    if "lm_encoder_media_step" in requested_set:
        _require("lm_encoder_media_step")
    if "lm_encoder_media_chunk" in requested_set:
        _require("lm_encoder_media_chunk")
    if "decoder_step" in requested_set:
        _require("lm_encoder_step")
        _require("decoder_step")

    if "vision_encoder" in expanded_components and pixel_values is None:
        raise RuntimeError("Gemma4 vision component requested but pixel_values were not prepared")
    if "audio_encoder" in expanded_components and (input_features is None or input_features_mask is None):
        raise RuntimeError("Gemma4 audio component requested but input_features/input_features_mask were not prepared")

    vision_encoder: Gemma4VisionEncoderAdapter | None = None
    vision_encoder_npu: Gemma4VisionEncoderAdapter | None = None
    audio_encoder: Gemma4AudioEncoderAdapter | None = None
    native_image_soft_token_counts: tuple[int, ...] | None = None
    native_image_pool_shapes: tuple[tuple[int, int, int], ...] | None = None
    if "vision_encoder" in expanded_components:
        vision_tower = getattr(getattr(model, "model", model), "vision_tower", None)
        pooling_kernel_size = int(_module_or_config_attr(vision_tower, "pooling_kernel_size", 3) or 3)
        native_image_soft_token_counts = _gemma4_static_image_soft_token_counts(
            pixel_position_ids,
            pooling_kernel_size=pooling_kernel_size,
        )
        native_image_pool_shapes = _gemma4_static_image_pool_shapes(
            pixel_position_ids,
            pooling_kernel_size=pooling_kernel_size,
        )
        vision_encoder = Gemma4VisionEncoderAdapter(
            model,
            weights_dir=weights_dir,
            native_image_soft_token_counts=native_image_soft_token_counts,
            native_image_pool_shapes=native_image_pool_shapes,
        ).eval()
        vision_encoder_npu = Gemma4VisionEncoderAdapter(
            model,
            weights_dir=weights_dir,
            native_image_soft_token_counts=native_image_soft_token_counts,
            native_image_pool_shapes=None,
        ).eval()
    if "audio_encoder" in expanded_components:
        _gemma4_make_audio_mask_fp16_safe(model)
        audio_encoder = Gemma4AudioEncoderAdapter(model, weights_dir=weights_dir).eval()

    if low_memory_meta:
        pixel_values = _to_meta_example_tensor(pixel_values)
        pixel_position_ids = _to_meta_example_tensor(pixel_position_ids)
        input_features = _to_meta_example_tensor(input_features)
        input_features_mask = _to_meta_example_tensor(input_features_mask)
        input_ids = _to_meta_example_tensor(input_ids)
        attention_mask = _to_meta_example_tensor(attention_mask)
        token_type_ids = _to_meta_example_tensor(token_type_ids)

    image_features: torch.Tensor | None = None
    audio_features: torch.Tensor | None = None
    decoder_inputs: tuple[torch.Tensor, ...] | None = None
    prefill_decoder_inputs: tuple[torch.Tensor, ...] | None = None
    decoder_step_inputs: tuple[torch.Tensor, ...] | None = None
    step_input_ids: torch.Tensor | None = None
    step_position_ids: torch.Tensor | None = None
    media_step_embeds: torch.Tensor | None = None
    native_merge_plan: _Gemma4NativeMergePlan | None = None

    with torch.no_grad():
        if "vision_encoder" in expanded_components and (
            "lm_encoder" in expanded_components
            or "decoder" in expanded_components
        ):
            if vision_encoder is None or pixel_values is None:
                raise RuntimeError("Gemma4 vision features requested but vision encoder inputs are unavailable")
            image_features = vision_encoder(pixel_values, pixel_position_ids)
        if "audio_encoder" in expanded_components and (
            "lm_encoder" in expanded_components
            or "decoder" in expanded_components
        ):
            if audio_encoder is None or input_features is None or input_features_mask is None:
                raise RuntimeError("Gemma4 audio features requested but audio encoder inputs are unavailable")
            audio_features = audio_encoder(input_features, input_features_mask)
        if "lm_encoder" in expanded_components or "decoder" in expanded_components:
            if image_features is None and vision_encoder is not None and pixel_values is not None:
                image_features = vision_encoder(pixel_values, pixel_position_ids)
            if include_audio and audio_features is None and audio_encoder is not None and input_features is not None and input_features_mask is not None:
                audio_features = audio_encoder(input_features, input_features_mask)
            if image_features is not None:
                native_merge_plan = _gemma4_build_native_merge_plan(
                    getattr(model, "model", model),
                    planning_input_ids,
                    image_feature_count=_gemma4_feature_token_count(image_features),
                    audio_feature_count=_gemma4_feature_token_count(audio_features),
                )
    lm_encoder = Gemma4LMEncoderAdapter(
        model,
        weights_dir=weights_dir,
        native_merge_plan=native_merge_plan,
        include_audio=include_audio,
    ).eval()
    encoder_chunk_size = max(1, int(os.environ.get("CACTUS_GEMMA4_ENCODER_CHUNK", "128") or "128"))
    encoder_chunk_size = min(encoder_chunk_size, int(input_ids.shape[1]))
    lm_encoder_step = Gemma4LMEncoderStepAdapter(model, weights_dir=weights_dir).eval()
    lm_encoder_text_chunk = Gemma4LMEncoderTextChunkAdapter(model, weights_dir=weights_dir).eval()
    lm_encoder_media_step = Gemma4LMEncoderMediaStepAdapter(model, weights_dir=weights_dir).eval()
    lm_encoder_media_chunk = Gemma4LMEncoderMediaChunkAdapter(model, weights_dir=weights_dir).eval()
    decoder = Gemma4DecoderAdapter(model, weights_dir=weights_dir).eval()
    decoder_prefill = Gemma4DecoderPrefillChunkAdapter(model, weights_dir=weights_dir).eval()
    decoder_embed = Gemma4DecoderEmbedChunkAdapter(model, weights_dir=weights_dir).eval()
    decoder_step = Gemma4DecoderStepAdapter(model, weights_dir=weights_dir).eval()

    with torch.no_grad():
        if "decoder" in expanded_components:
            if image_features is None:
                raise RuntimeError("Gemma4 decoder spec requires precomputed image features")
            if include_audio and audio_features is None:
                raise RuntimeError("Gemma4 decoder spec requires precomputed audio features")
            lm_encoder_inputs: tuple[torch.Tensor | None, ...]
            if include_audio:
                lm_encoder_inputs = (
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    image_features,
                    audio_features,
                )
            else:
                lm_encoder_inputs = (
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    image_features,
                )
            decoder_inputs = lm_encoder(*lm_encoder_inputs)
        if "lm_encoder_step" in expanded_components or "decoder_step" in expanded_components:
            step_input_ids = input_ids[:, -1:].contiguous()
            step_position_ids = torch.full(
                (int(input_ids.shape[0]), 1),
                max(0, int(input_ids.shape[1]) - 1),
                dtype=torch.long,
                device=input_ids.device,
            )
            decoder_step_inputs = lm_encoder_step(step_input_ids, step_position_ids)
        if "lm_encoder_media_step" in expanded_components:
            if audio_features is not None and _gemma4_feature_token_count(audio_features) > 0:
                media_source = audio_features if audio_features.ndim == 3 else audio_features.unsqueeze(0)
                media_step_embeds = media_source[:, :1, :].contiguous()
            elif image_features is not None and _gemma4_feature_token_count(image_features) > 0:
                media_source = image_features if image_features.ndim == 3 else image_features.unsqueeze(0)
                media_step_embeds = media_source[:, :1, :].contiguous()
            else:
                text_config = _gemma4_text_config(getattr(model, "model", model))
                hidden_size = int(getattr(text_config, "hidden_size", 0) or 0)
                if hidden_size <= 0:
                    raise RuntimeError("Gemma4 lm_encoder_media_step spec could not infer hidden size")
                dtype = _module_floating_dtype(model) or torch.float16
                media_step_embeds = torch.zeros(
                    (int(input_ids.shape[0]), 1, hidden_size),
                    dtype=dtype,
                    device=input_ids.device,
                )
        chunk_input_ids = input_ids[:, :encoder_chunk_size].contiguous()
        chunk_position_ids = torch.arange(
            encoder_chunk_size,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0).expand(int(input_ids.shape[0]), -1).contiguous()
        if "decoder_prefill_chunk" in expanded_components and decoder_inputs is None:
            prefill_decoder_inputs = lm_encoder_text_chunk(chunk_input_ids, chunk_position_ids)
        if media_step_embeds is not None:
            hidden_size = int(media_step_embeds.shape[-1])
            media_chunk_embeds = torch.zeros(
                (int(input_ids.shape[0]), encoder_chunk_size, hidden_size),
                dtype=media_step_embeds.dtype,
                device=media_step_embeds.device,
            )
            media_chunk_embeds[:, :1, :] = media_step_embeds
        else:
            text_config = _gemma4_text_config(getattr(model, "model", model))
            hidden_size = int(getattr(text_config, "hidden_size", 0) or 0)
            if hidden_size <= 0:
                hidden_size = int(chunk_input_ids.shape[-1])
            dtype = _module_floating_dtype(model) or torch.float16
            media_chunk_embeds = torch.zeros(
                (int(input_ids.shape[0]), encoder_chunk_size, hidden_size),
                dtype=dtype,
                device=input_ids.device,
            )

    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "multimodal_causal_lm_logits",
        "adapter_family": "gemma4",
    }
    specs: list[ComponentModuleSpec] = []
    if "vision_encoder" in expanded_components:
        if vision_encoder is None or vision_encoder_npu is None or pixel_values is None:
            raise RuntimeError("Gemma4 vision_encoder spec requires prepared pixel_values")
        specs.append(ComponentModuleSpec(
            component="vision_encoder",
            module=vision_encoder,
            example_inputs=(pixel_values, pixel_position_ids),
            input_keys=("pixel_values", "pixel_position_ids"),
            output_keys=("image_features",),
            graph_meta={**common_graph_meta, "component": "vision_encoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
            npu_module=vision_encoder_npu,
            npu_runtime_input_count=2,
            npu_reparam=_gemma4_npu_projection_reparam,
        ))
    if "audio_encoder" in expanded_components:
        if audio_encoder is None or input_features is None or input_features_mask is None:
            raise RuntimeError("Gemma4 audio_encoder spec requires prepared audio features")
        specs.append(ComponentModuleSpec(
            component="audio_encoder",
            module=audio_encoder,
            example_inputs=(input_features, input_features_mask),
            input_keys=("input_features", "input_features_mask"),
            output_keys=("audio_features",),
            graph_meta={**common_graph_meta, "component": "audio_encoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
            npu_reparam=_gemma4_npu_projection_reparam,
        ))
    if "lm_encoder" in expanded_components:
        if image_features is None:
            raise RuntimeError("Gemma4 lm_encoder spec requires precomputed image features")
        if include_audio and audio_features is None:
            raise RuntimeError("Gemma4 lm_encoder spec requires precomputed audio features")
        lm_encoder_example_inputs: tuple[torch.Tensor | None, ...]
        lm_encoder_input_keys = ("input_ids", "attention_mask", "token_type_ids", "image_features")
        if include_audio:
            lm_encoder_example_inputs = (input_ids, attention_mask, token_type_ids, image_features, audio_features)
            lm_encoder_input_keys = (*lm_encoder_input_keys, "audio_features")
        else:
            lm_encoder_example_inputs = (input_ids, attention_mask, token_type_ids, image_features)
        specs.append(ComponentModuleSpec(
            component="lm_encoder",
            module=lm_encoder,
            example_inputs=lm_encoder_example_inputs,
            input_keys=lm_encoder_input_keys,
            output_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            graph_meta={**common_graph_meta, "component": "lm_encoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=256)
    prefill_chunk_size = max(1, int(os.environ.get("CACTUS_GEMMA4_PREFILL_CHUNK", "128") or "128"))
    prefill_chunk_size = min(prefill_chunk_size, int(input_ids.shape[1]))
    if "decoder" in expanded_components:
        if decoder_inputs is None:
            raise RuntimeError("Gemma4 decoder spec requires precomputed decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=tuple(decoder_inputs),
            input_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder_prefill_chunk" in expanded_components:
        if decoder_inputs is not None:
            chunk_inputs = tuple(
                value[:, :prefill_chunk_size, ...].contiguous()
                if value.ndim >= 2
                else value
                for value in decoder_inputs
            )
        elif prefill_decoder_inputs is not None:
            prefill_chunk_size = int(prefill_decoder_inputs[0].shape[1])
            chunk_inputs = tuple(prefill_decoder_inputs)
        else:
            raise RuntimeError("Gemma4 decoder_prefill_chunk spec requires precomputed decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder_prefill_chunk",
            module=decoder_prefill,
            example_inputs=chunk_inputs,
            input_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_prefill_chunk",
                "use_internal_kv_cache": True,
                "max_cache_seq_len": cache_seq_len,
                "cache_sink_size": 4,
                "prefill_chunk_size": prefill_chunk_size,
            },
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder_embed_chunk" in expanded_components:
        if decoder_inputs is not None:
            embed_chunk_inputs = tuple(
                value[:, :prefill_chunk_size, ...].contiguous()
                if value.ndim >= 2
                else value
                for value in decoder_inputs
            )
        elif prefill_decoder_inputs is not None:
            embed_chunk_inputs = tuple(prefill_decoder_inputs)
        else:
            raise RuntimeError("Gemma4 decoder_embed_chunk spec requires precomputed decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder_embed_chunk",
            module=decoder_embed,
            example_inputs=embed_chunk_inputs,
            input_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            output_keys=("last_hidden_state",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_embed_chunk",
                "use_internal_kv_cache": True,
                "max_cache_seq_len": cache_seq_len,
                "cache_sink_size": 4,
                "prefill_chunk_size": prefill_chunk_size,
            },
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "lm_encoder_step" in expanded_components:
        if step_input_ids is None or step_position_ids is None:
            raise RuntimeError("Gemma4 lm_encoder_step spec requires step token inputs")
        specs.append(ComponentModuleSpec(
            component="lm_encoder_step",
            module=lm_encoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            graph_meta={**common_graph_meta, "component": "lm_encoder_step"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
            dynamic_batch_axis=0 if dynamic_batch else None,
        ))
    if "lm_encoder_text_chunk" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_text_chunk",
            module=lm_encoder_text_chunk,
            example_inputs=(chunk_input_ids, chunk_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            graph_meta={
                **common_graph_meta,
                "component": "lm_encoder_text_chunk",
                "encoder_chunk_size": encoder_chunk_size,
            },
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "lm_encoder_media_step" in expanded_components:
        if media_step_embeds is None or step_input_ids is None or step_position_ids is None:
            raise RuntimeError("Gemma4 lm_encoder_media_step spec requires step token inputs")
        specs.append(ComponentModuleSpec(
            component="lm_encoder_media_step",
            module=lm_encoder_media_step,
            example_inputs=(media_step_embeds, torch.zeros_like(step_input_ids), step_position_ids),
            input_keys=("inputs_embeds", "input_ids", "position_ids"),
            output_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            graph_meta={**common_graph_meta, "component": "lm_encoder_media_step"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "lm_encoder_media_chunk" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_media_chunk",
            module=lm_encoder_media_chunk,
            example_inputs=(media_chunk_embeds, torch.zeros_like(chunk_input_ids), chunk_position_ids),
            input_keys=("inputs_embeds", "input_ids", "position_ids"),
            output_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            graph_meta={
                **common_graph_meta,
                "component": "lm_encoder_media_chunk",
                "encoder_chunk_size": encoder_chunk_size,
            },
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder_step" in expanded_components:
        if decoder_step_inputs is None:
            raise RuntimeError("Gemma4 decoder_step spec requires precomputed step decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder_step",
            module=decoder_step,
            example_inputs=tuple(decoder_step_inputs),
            input_keys=_GEMMA4_DECODER_PIPELINE_IO_KEYS,
            output_keys=("logits", "probe_hidden"),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_step",
                "use_internal_kv_cache": True,
                "max_cache_seq_len": cache_seq_len,
                "cache_sink_size": 4,
                "cache_num_slots": (max_slots if dynamic_batch else 1),
            },
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
            dynamic_batch_axis=0 if dynamic_batch else None,
        ))
    return specs


class Qwen35CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = _resolve_qwen35_text_backbone(model)
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        attention_mask = (
            (input_ids != int(self.pad_token_id)).to(dtype=torch.int64)
            if self.pad_token_id is not None
            else None
        )
        base_position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).view(1, 1, -1)
        position_ids = torch.cat(
            (base_position_ids, base_position_ids, base_position_ids, base_position_ids),
            dim=0,
        )
        text_position_ids = position_ids[0]
        multimodal_position_ids = position_ids[1:]

        causal_mask = self._create_causal_mask(
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = attention_mask

        hidden_states = inputs_embeds
        checkpoints: list[torch.Tensor] = []
        position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids)

        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=_qwen35_attention_mask_for_layer(
                    self.backbone.config,
                    i,
                    causal_mask=causal_mask,
                    linear_attention_mask=linear_attn_mask,
                ),
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        hidden_states = _select_last_non_pad_token(
            hidden_states,
            input_ids,
            pad_token_id=self.pad_token_id,
        )
        checkpoints.append(hidden_states)
        return self.model.lm_head(hidden_states), checkpoints

    def get_transpile_metadata(self):
        layer_types = _qwen35_layer_types(self.backbone.config)
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": layer_types,
            }
        }


class Qwen35CausalLMStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = _resolve_qwen35_text_backbone(model)
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        text_position_ids = position_ids.to(dtype=torch.int64)
        base_position_ids = text_position_ids.view(1, text_position_ids.shape[0], -1)
        multimodal_position_ids = torch.cat(
            (base_position_ids, base_position_ids, base_position_ids),
            dim=0,
        )
        causal_mask = self._create_causal_mask(
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = None

        hidden_states = inputs_embeds
        position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids)
        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=_qwen35_attention_mask_for_layer(
                    self.backbone.config,
                    i,
                    causal_mask=causal_mask,
                    linear_attention_mask=linear_attn_mask,
                ),
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )

        hidden_states = self.backbone.norm(hidden_states)
        return self.model.lm_head(hidden_states[:, -1:, :])

    def get_transpile_metadata(self):
        layer_types = _qwen35_layer_types(self.backbone.config)
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "position_ids"),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": layer_types,
            }
        }


def _resolve_qwen35_text_backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    language_model = getattr(backbone, "language_model", None)
    if isinstance(language_model, torch.nn.Module):
        return language_model
    if isinstance(backbone, torch.nn.Module):
        return backbone
    raise AttributeError(f"{type(model).__name__} does not expose a Qwen3.5 text backbone")


def _resolve_qwen35_multimodal_backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    if isinstance(backbone, torch.nn.Module) and hasattr(backbone, "visual"):
        return backbone
    raise AttributeError(f"{type(model).__name__} does not expose a Qwen3.5 multimodal backbone")


def _qwen35_layer_types(config) -> tuple[str, ...]:
    layer_types = tuple(str(value) for value in (getattr(config, "layer_types", ()) or ()))
    if layer_types:
        return layer_types
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    return tuple("full_attention" for _ in range(num_layers))


def _qwen35_attention_mask_for_layer(
    config,
    layer_index: int,
    *,
    causal_mask,
    linear_attention_mask,
):
    layer_types = _qwen35_layer_types(config)
    layer_type = layer_types[layer_index] if layer_index < len(layer_types) else "full_attention"
    return linear_attention_mask if layer_type == "linear_attention" else causal_mask


class Qwen35VisionEncoderAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, image_grid_thw: torch.Tensor):
        super().__init__()
        self.backbone = _resolve_qwen35_multimodal_backbone(model)
        self.visual = self.backbone.visual
        self.register_buffer("_image_grid_thw", image_grid_thw.detach().clone().to(dtype=torch.long), persistent=False)
        with torch.no_grad():
            rotary_pos_emb = self.visual.rot_pos_emb(self._image_grid_thw)
            emb = torch.cat((rotary_pos_emb.reshape(rotary_pos_emb.shape[0], -1),) * 2, dim=-1)
            self.register_buffer("_vision_cos", emb.cos(), persistent=False)
            self.register_buffer("_vision_sin", emb.sin(), persistent=False)
            self.register_buffer(
                "_vision_pos_embed",
                self.visual.fast_pos_embed_interpolate(self._image_grid_thw),
                persistent=False,
            )

    def _vision_attention(self, attn: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision  # type: ignore

        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            self._vision_cos,
            self._vision_sin,
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * float(attn.scaling)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        return attn.proj(attn_output)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        proj = self.visual.patch_embed.proj
        patch_weight = proj.weight.reshape(int(proj.weight.shape[0]), -1)
        hidden_states = torch.nn.functional.linear(
            pixel_values.to(dtype=patch_weight.dtype),
            patch_weight,
            proj.bias,
        )
        hidden_states = hidden_states + self._vision_pos_embed.to(dtype=hidden_states.dtype)
        for block in self.visual.blocks:
            hidden_states = hidden_states + self._vision_attention(block.attn, block.norm1(hidden_states))
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
        return self.visual.merger(hidden_states)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.backbone,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("pixel_values",),
                ),
            }
        }


class Qwen35LMEncoderAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, input_ids: torch.Tensor):
        super().__init__()
        self.backbone = _resolve_qwen35_multimodal_backbone(model)
        self.language_model = self.backbone.language_model
        self.image_token_id = int(self.backbone.config.image_token_id)
        image_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=False).flatten()
        if int(image_positions.numel()) <= 0:
            raise ValueError("Qwen3.5 multimodal LM encoder requires image placeholder tokens")
        self.image_token_start = int(image_positions[0].item())
        self.image_token_count = int(image_positions.numel())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        image_embeds = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).reshape(
            1,
            self.image_token_count,
            int(inputs_embeds.shape[-1]),
        )
        prefix = inputs_embeds[:, : self.image_token_start, :]
        suffix = inputs_embeds[:, self.image_token_start + self.image_token_count :, :]
        inputs_embeds = torch.cat((prefix, image_embeds, suffix), dim=1)
        return inputs_embeds, attention_mask, position_ids

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.backbone,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "attention_mask", "position_ids", "image_features"),
                ),
            }
        }


class Qwen35EmbedsCausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = _resolve_qwen35_text_backbone(model)
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if position_ids.ndim == 2:
            multimodal_position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            text_position_ids = position_ids
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            multimodal_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            multimodal_position_ids = position_ids
        causal_mask = self._create_causal_mask(
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids)
        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=_qwen35_attention_mask_for_layer(
                    self.backbone.config,
                    i,
                    causal_mask=causal_mask,
                    linear_attention_mask=attention_mask,
                ),
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
        hidden_states = self.backbone.norm(hidden_states)
        return self.model.lm_head(hidden_states)

    def get_transpile_metadata(self):
        layer_types = _qwen35_layer_types(self.backbone.config)
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("inputs_embeds", "attention_mask", "position_ids"),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": layer_types,
            }
        }


class Qwen35LMEncoderStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = _resolve_qwen35_text_backbone(model)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        base_position_ids = position_ids.to(dtype=torch.int64).view(1, position_ids.shape[0], -1)
        multimodal_position_ids = torch.cat(
            (base_position_ids, base_position_ids, base_position_ids),
            dim=0,
        )
        return inputs_embeds, multimodal_position_ids

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.backbone,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "position_ids"),
                ),
            }
        }


class Qwen35LMEncoderTextChunkAdapter(Qwen35LMEncoderStepAdapter):
    pass


class Qwen35EmbedsCausalLMStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = _resolve_qwen35_text_backbone(model)
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.ndim == 2:
            text_position_ids = position_ids.to(dtype=torch.int64)
            base_position_ids = text_position_ids.view(1, text_position_ids.shape[0], -1)
            multimodal_position_ids = torch.cat(
                (base_position_ids, base_position_ids, base_position_ids),
                dim=0,
            )
        else:
            text_position_ids = None
            multimodal_position_ids = position_ids.to(dtype=torch.int64)
        causal_mask = self._create_causal_mask(
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids)
        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=_qwen35_attention_mask_for_layer(
                    self.backbone.config,
                    i,
                    causal_mask=causal_mask,
                    linear_attention_mask=None,
                ),
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
        hidden_states = self.backbone.norm(hidden_states)
        return self.model.lm_head(hidden_states[:, -1:, :])

    def get_transpile_metadata(self):
        layer_types = _qwen35_layer_types(self.backbone.config)
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("inputs_embeds", "position_ids"),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": layer_types,
            }
        }


class Qwen35EmbedsCausalLMPrefillChunkAdapter(Qwen35EmbedsCausalLMStepAdapter):
    pass


class Qwen3CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = model.model
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        self.adapter_family = _family_key(model)

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        seq_len = int(inputs_embeds.shape[1])
        allowed_positions = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=inputs_embeds.device),
        ).view(1, 1, seq_len, seq_len)
        if self.pad_token_id is not None:
            key_mask = (input_ids != int(self.pad_token_id)).view(input_ids.shape[0], 1, 1, seq_len)
            allowed_positions = torch.logical_and(allowed_positions, key_mask)
        allowed_values = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ) * 0.0
        blocked_values = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ) * torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.where(
            allowed_positions,
            allowed_values,
            blocked_values,
        )

        hidden_states = inputs_embeds
        checkpoints: list[torch.Tensor] = []
        position_embeddings = self.backbone.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        hidden_states = _select_last_non_pad_token(
            hidden_states,
            input_ids,
            pad_token_id=self.pad_token_id,
        )
        checkpoints.append(hidden_states)
        return self.model.lm_head(hidden_states), checkpoints

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family=self.adapter_family,
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            }
        }


class Qwen3CausalLMStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = model.model
        self.pad_token_id = pad_token_id if pad_token_id is not None else _resolve_model_pad_token_id(model)
        self.adapter_family = _family_key(model)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        seq_len = int(inputs_embeds.shape[1])
        allowed_positions = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        allowed_values = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ) * 0.0
        blocked_values = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ) * torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.where(allowed_positions, allowed_values, blocked_values)

        hidden_states = inputs_embeds
        text_position_ids = position_ids.to(dtype=torch.int64)
        position_embeddings = self.backbone.rotary_emb(hidden_states, text_position_ids)
        for decoder_layer in self.backbone.layers[: self.backbone.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )

        hidden_states = self.backbone.norm(hidden_states)
        return self.model.lm_head(hidden_states[:, -1:, :])

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family=self.adapter_family,
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "position_ids"),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            }
        }


class Qwen2MoeCausalLMLogitsAdapter(Qwen3CausalLMLogitsAdapter):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        _patch_qwen2_moe_mlps(model)
        super().__init__(model, pad_token_id=pad_token_id)


class Qwen2MoeCausalLMStepAdapter(Qwen3CausalLMStepAdapter):
    def __init__(self, model: torch.nn.Module, *, pad_token_id: int | None = None):
        _patch_qwen2_moe_mlps(model)
        super().__init__(model, pad_token_id=pad_token_id)


class Qwen3LMEncoderStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.model

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        return inputs_embeds, position_ids.to(dtype=torch.int64)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.backbone,
                    adapter_family="qwen3",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "position_ids"),
                ),
            }
        }


class Qwen3LMEncoderTextChunkAdapter(Qwen3LMEncoderStepAdapter):
    pass


class Qwen3EmbedsCausalLMStepAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = model.model

    def _encode_hidden_states(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(inputs_embeds.shape[1])
        allowed_positions = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=inputs_embeds.device),
        ).view(1, 1, seq_len, seq_len)
        allowed_values = torch.zeros(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        blocked_values = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ) * torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.where(allowed_positions, allowed_values, blocked_values)

        hidden_states = inputs_embeds
        text_position_ids = position_ids.to(dtype=torch.int64)
        position_embeddings = self.backbone.rotary_emb(hidden_states, text_position_ids)
        for decoder_layer in self.backbone.layers[: self.backbone.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return self.backbone.norm(hidden_states)

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self._encode_hidden_states(inputs_embeds, position_ids)
        return self.model.lm_head(hidden_states[:, -1:, :])

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="qwen3",
                    adapter_type=type(self).__name__,
                    input_names=("inputs_embeds", "position_ids"),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            }
        }


class Qwen3EmbedsCausalLMPrefillChunkAdapter(Qwen3EmbedsCausalLMStepAdapter):
    pass


class Qwen3EmbedsCausalLMEmbedChunkAdapter(Qwen3EmbedsCausalLMStepAdapter):
    """Like the prefill chunk, but emits the full-sequence last hidden state
    (no lm_head, no last-token slice) for embedding readout."""

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self._encode_hidden_states(inputs_embeds, position_ids)


def _build_qwen_causal_lm_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
    dynamic_batch: bool = True,
    max_slots: int = 1,
) -> list[ComponentModuleSpec] | None:
    input_ids = named_tensors.get("input_ids")
    if input_ids is None:
        return None
    family = _family_key(model)
    pad_token_id = _resolve_model_pad_token_id(model)
    if family == "qwen3_5":
        decoder = Qwen35CausalLMLogitsAdapter(model, pad_token_id=pad_token_id).eval()
        decoder_step = Qwen35CausalLMStepAdapter(model, pad_token_id=pad_token_id).eval()
    elif family == "qwen2_moe":
        decoder = Qwen2MoeCausalLMLogitsAdapter(model, pad_token_id=pad_token_id).eval()
        decoder_step = Qwen2MoeCausalLMStepAdapter(model, pad_token_id=pad_token_id).eval()
    elif family == "qwen3":
        decoder = Qwen3CausalLMLogitsAdapter(model, pad_token_id=pad_token_id).eval()
        decoder_step = Qwen3CausalLMStepAdapter(model, pad_token_id=pad_token_id).eval()
    else:
        return None

    requested = tuple(components or ("decoder", "decoder_step"))
    requested_set = set(requested)
    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "causal_lm_logits",
        "adapter_family": family,
    }

    chunk_components = {"lm_encoder_step", "lm_encoder_text_chunk", "decoder_media_step", "decoder_prefill_chunk", "decoder_embed_chunk"}
    wants_chunked = bool(chunk_components & requested_set)
    if wants_chunked and family != "qwen3":
        raise UnsupportedComponentsError(
            f"text chunked-prefill components are only supported for the qwen3 family, got {family}"
        )

    specs: list[ComponentModuleSpec] = []
    if "decoder" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=(input_ids,),
            input_keys=("input_ids",),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata={"family": family, "task": "causal_lm_logits"},
        ))
    if "decoder_step" in requested_set:
        step_input_ids = input_ids[:, :1]
        step_position_ids = torch.zeros_like(step_input_ids, dtype=torch.int64)
        max_cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512)
        specs.append(ComponentModuleSpec(
            component="decoder_step",
            module=decoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_step",
                "use_internal_kv_cache": True,
                "use_internal_conv_cache": True,
                "use_internal_gated_deltanet_cache": True,
                "max_cache_seq_len": max_cache_seq_len,
                "cache_sink_size": 4,
                "cache_num_slots": (max_slots if dynamic_batch else 1),
            },
            metadata={"family": family, "task": "causal_lm_logits"},
            dynamic_batch_axis=0 if dynamic_batch else None,
        ))

    if wants_chunked:
        lm_encoder_step = Qwen3LMEncoderStepAdapter(model).eval()
        lm_encoder_text_chunk = Qwen3LMEncoderTextChunkAdapter(model).eval()
        decoder_media_step = Qwen3EmbedsCausalLMStepAdapter(model).eval()
        decoder_prefill_chunk = Qwen3EmbedsCausalLMPrefillChunkAdapter(model).eval()
        decoder_embed_chunk = Qwen3EmbedsCausalLMEmbedChunkAdapter(model).eval()
        max_cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512)
        prefill_chunk_size = max(1, int(os.environ.get("CACTUS_QWEN_PREFILL_CHUNK", "128") or "128"))
        chunk_input_ids = _tile_to_length(input_ids, prefill_chunk_size)
        chunk_position_ids = torch.arange(
            prefill_chunk_size,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0).expand(int(input_ids.shape[0]), -1).contiguous()
        with torch.no_grad():
            chunk_embeds, chunk_pos_out = lm_encoder_text_chunk(chunk_input_ids, chunk_position_ids)
        step_input_ids = input_ids[:, :1]
        step_position_ids = torch.zeros_like(step_input_ids, dtype=torch.int64)
        with torch.no_grad():
            step_embeds, step_pos_out = lm_encoder_step(step_input_ids, step_position_ids)

        if "lm_encoder_step" in requested_set:
            specs.append(ComponentModuleSpec(
                component="lm_encoder_step",
                module=lm_encoder_step,
                example_inputs=(step_input_ids, step_position_ids),
                input_keys=("input_ids", "position_ids"),
                output_keys=("inputs_embeds", "position_ids"),
                graph_meta={**common_graph_meta, "component": "lm_encoder_step"},
                metadata={"family": family, "task": "causal_lm_logits"},
                dynamic_batch_axis=0 if dynamic_batch else None,
            ))
        if "lm_encoder_text_chunk" in requested_set:
            specs.append(ComponentModuleSpec(
                component="lm_encoder_text_chunk",
                module=lm_encoder_text_chunk,
                example_inputs=(chunk_input_ids, chunk_position_ids),
                input_keys=("input_ids", "position_ids"),
                output_keys=("inputs_embeds", "position_ids"),
                graph_meta={**common_graph_meta, "component": "lm_encoder_text_chunk"},
                metadata={"family": family, "task": "causal_lm_logits"},
            ))
        if "decoder_prefill_chunk" in requested_set:
            specs.append(ComponentModuleSpec(
                component="decoder_prefill_chunk",
                module=decoder_prefill_chunk,
                example_inputs=(chunk_embeds, chunk_pos_out),
                input_keys=("inputs_embeds", "position_ids"),
                output_keys=("logits",),
                graph_meta={
                    **common_graph_meta,
                    "component": "decoder_prefill_chunk",
                    "use_internal_kv_cache": True,
                    "use_internal_conv_cache": True,
                    "use_internal_gated_deltanet_cache": True,
                    "max_cache_seq_len": max_cache_seq_len,
                    "cache_sink_size": 4,
                    "prefill_chunk_size": prefill_chunk_size,
                },
                metadata={"family": family, "task": "causal_lm_logits"},
            ))
        if "decoder_embed_chunk" in requested_set:
            specs.append(ComponentModuleSpec(
                component="decoder_embed_chunk",
                module=decoder_embed_chunk,
                example_inputs=(chunk_embeds, chunk_pos_out),
                input_keys=("inputs_embeds", "position_ids"),
                output_keys=("last_hidden_state",),
                graph_meta={
                    **common_graph_meta,
                    "component": "decoder_embed_chunk",
                    "use_internal_kv_cache": True,
                    "use_internal_conv_cache": True,
                    "use_internal_gated_deltanet_cache": True,
                    "max_cache_seq_len": max_cache_seq_len,
                    "cache_sink_size": 4,
                    "prefill_chunk_size": prefill_chunk_size,
                },
                metadata={"family": family, "task": "causal_lm_logits"},
            ))
        if "decoder_media_step" in requested_set:
            specs.append(ComponentModuleSpec(
                component="decoder_media_step",
                module=decoder_media_step,
                example_inputs=(step_embeds, step_pos_out),
                input_keys=("inputs_embeds", "position_ids"),
                output_keys=("logits",),
                graph_meta={
                    **common_graph_meta,
                    "component": "decoder_media_step",
                    "use_internal_kv_cache": True,
                    "use_internal_conv_cache": True,
                    "use_internal_gated_deltanet_cache": True,
                    "max_cache_seq_len": max_cache_seq_len,
                    "cache_sink_size": 4,
                    "cache_num_slots": (max_slots if dynamic_batch else 1),
                },
                metadata={"family": family, "task": "causal_lm_logits"},
                dynamic_batch_axis=0 if dynamic_batch else None,
            ))
    return specs


def _build_qwen3_5_multimodal_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
) -> list[ComponentModuleSpec] | None:
    input_ids = named_tensors.get("input_ids")
    attention_mask = named_tensors.get("attention_mask")
    position_ids = named_tensors.get("position_ids")
    pixel_values = named_tensors.get("pixel_values")
    image_grid_thw = named_tensors.get("image_grid_thw")
    if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, position_ids, pixel_values, image_grid_thw)):
        return None

    requested = tuple(components or (
        "vision_encoder",
        "lm_encoder",
        "decoder",
        "lm_encoder_text_chunk",
        "decoder_prefill_chunk",
        "lm_encoder_step",
        "decoder_media_step",
        "decoder_step",
    ))
    requested_set = set(requested)
    expanded_components: list[str] = []

    def _require(component: str) -> None:
        if component not in expanded_components:
            expanded_components.append(component)

    if "decoder" in requested_set:
        _require("vision_encoder")
        _require("lm_encoder")
        _require("decoder")
        _require("lm_encoder_text_chunk")
        _require("decoder_prefill_chunk")
    if "lm_encoder" in requested_set:
        _require("vision_encoder")
        _require("lm_encoder")
    if "lm_encoder_text_chunk" in requested_set:
        _require("lm_encoder_text_chunk")
    if "decoder_prefill_chunk" in requested_set:
        _require("lm_encoder_text_chunk")
        _require("decoder_prefill_chunk")
    if "vision_encoder" in requested_set:
        _require("vision_encoder")
    if "decoder_step" in requested_set:
        _require("decoder_step")
    if "decoder_media_step" in requested_set:
        _require("decoder_media_step")
    if "lm_encoder_step" in requested_set or "decoder_media_step" in requested_set:
        _require("lm_encoder_step")

    vision_encoder = Qwen35VisionEncoderAdapter(model, image_grid_thw=image_grid_thw).eval()
    lm_encoder = Qwen35LMEncoderAdapter(model, input_ids=input_ids).eval()
    decoder = Qwen35EmbedsCausalLMLogitsAdapter(model).eval()
    lm_encoder_step = Qwen35LMEncoderStepAdapter(model).eval()
    lm_encoder_text_chunk = Qwen35LMEncoderTextChunkAdapter(model).eval()
    decoder_media_step = Qwen35EmbedsCausalLMStepAdapter(model).eval()
    decoder_prefill_chunk = Qwen35EmbedsCausalLMPrefillChunkAdapter(model).eval()
    decoder_step = Qwen35CausalLMStepAdapter(model, pad_token_id=_resolve_model_pad_token_id(model)).eval()

    image_features: torch.Tensor | None = None
    decoder_inputs: tuple[torch.Tensor, ...] | None = None
    prefill_decoder_inputs: tuple[torch.Tensor, ...] | None = None
    with torch.no_grad():
        if "vision_encoder" in expanded_components and ("lm_encoder" in expanded_components or "decoder" in expanded_components):
            image_features = vision_encoder(pixel_values)
        if "lm_encoder" in expanded_components or "decoder" in expanded_components:
            if image_features is None:
                image_features = vision_encoder(pixel_values)
            decoder_inputs = lm_encoder(input_ids, attention_mask, position_ids, image_features)
        prefill_chunk_size = max(1, int(os.environ.get("CACTUS_QWEN_PREFILL_CHUNK", "128") or "128"))
        chunk_input_ids = _tile_to_length(input_ids, prefill_chunk_size)
        chunk_position_ids = torch.arange(
            prefill_chunk_size,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0).expand(int(input_ids.shape[0]), -1).contiguous()
        if "decoder_prefill_chunk" in expanded_components:
            prefill_decoder_inputs = lm_encoder_text_chunk(chunk_input_ids, chunk_position_ids)

    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "multimodal_causal_lm_logits",
        "adapter_family": "qwen3_5",
    }
    max_cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512)
    specs: list[ComponentModuleSpec] = []
    if "vision_encoder" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="vision_encoder",
            module=vision_encoder,
            example_inputs=(pixel_values,),
            input_keys=("pixel_values",),
            output_keys=("image_features",),
            graph_meta={**common_graph_meta, "component": "vision_encoder"},
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    if "lm_encoder" in expanded_components:
        if image_features is None:
            raise RuntimeError("Qwen3.5 lm_encoder spec requires precomputed image features")
        specs.append(ComponentModuleSpec(
            component="lm_encoder",
            module=lm_encoder,
            example_inputs=(input_ids, attention_mask, position_ids, image_features),
            input_keys=("input_ids", "attention_mask", "position_ids", "image_features"),
            output_keys=("inputs_embeds", "attention_mask", "position_ids"),
            graph_meta={**common_graph_meta, "component": "lm_encoder"},
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder" in expanded_components:
        if decoder_inputs is None:
            raise RuntimeError("Qwen3.5 decoder spec requires precomputed decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=decoder_inputs,
            input_keys=("inputs_embeds", "attention_mask", "position_ids"),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    if "lm_encoder_step" in expanded_components:
        step_input_ids = input_ids[:, :1]
        step_position_ids = torch.zeros_like(step_input_ids, dtype=torch.int64)
        specs.append(ComponentModuleSpec(
            component="lm_encoder_step",
            module=lm_encoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("inputs_embeds", "position_ids"),
            graph_meta={**common_graph_meta, "component": "lm_encoder_step"},
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    if "lm_encoder_text_chunk" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_text_chunk",
            module=lm_encoder_text_chunk,
            example_inputs=(chunk_input_ids, chunk_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("inputs_embeds", "position_ids"),
            graph_meta={**common_graph_meta, "component": "lm_encoder_text_chunk"},
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder_prefill_chunk" in expanded_components:
        if prefill_decoder_inputs is None:
            raise RuntimeError("Qwen3.5 decoder_prefill_chunk spec requires precomputed text chunk decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder_prefill_chunk",
            module=decoder_prefill_chunk,
            example_inputs=prefill_decoder_inputs,
            input_keys=("inputs_embeds", "position_ids"),
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_prefill_chunk",
                "use_internal_kv_cache": True,
                "use_internal_conv_cache": True,
                "use_internal_gated_deltanet_cache": True,
                "max_cache_seq_len": max_cache_seq_len,
                "cache_sink_size": 4,
                "prefill_chunk_size": prefill_chunk_size,
            },
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder_media_step" in expanded_components:
        if decoder_inputs is None:
            raise RuntimeError("Qwen3.5 decoder_media_step spec requires precomputed decoder inputs")
        step_inputs_embeds = decoder_inputs[0][:, :1, :]
        step_position_ids = decoder_inputs[2][:, :, :1]
        specs.append(ComponentModuleSpec(
            component="decoder_media_step",
            module=decoder_media_step,
            example_inputs=(step_inputs_embeds, step_position_ids),
            input_keys=("inputs_embeds", "position_ids"),
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_media_step",
                "use_internal_kv_cache": True,
                "use_internal_conv_cache": True,
                "use_internal_gated_deltanet_cache": True,
                "max_cache_seq_len": max_cache_seq_len,
                "cache_sink_size": 4,
            },
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder_step" in expanded_components:
        step_input_ids = input_ids[:, :1]
        step_position_ids = torch.zeros_like(step_input_ids, dtype=torch.int64)
        specs.append(ComponentModuleSpec(
            component="decoder_step",
            module=decoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_step",
                "use_internal_kv_cache": True,
                "use_internal_conv_cache": True,
                "use_internal_gated_deltanet_cache": True,
                "max_cache_seq_len": max_cache_seq_len,
                "cache_sink_size": 4,
            },
            metadata={"family": "qwen3_5", "task": "multimodal_causal_lm_logits"},
        ))
    return specs


class CTCLogitsAdapter(BoundInputAdapter):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], family: str):
        super().__init__(model, input_names=input_names, family=family, metadata_task="ctc_logits")

    def forward(self, *bound_inputs: torch.Tensor | None) -> torch.Tensor:
        outputs = self.model(return_dict=True, **self._kwargs_from_bound_inputs(*bound_inputs))
        return _extract_tensor_output(outputs, preferred_field="logits")


class EncoderHiddenStatesAdapter(BoundInputAdapter):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], family: str):
        encoder = None
        get_encoder = getattr(model, "get_encoder", None)
        if callable(get_encoder):
            encoder = get_encoder()
        if encoder is None:
            encoder = getattr(model, "encoder", None)
        if encoder is None:
            model_attr = getattr(model, "model", None)
            if model_attr is not None:
                encoder = getattr(model_attr, "encoder", None)
        if not isinstance(encoder, torch.nn.Module):
            raise NotImplementedError(f"{type(model).__name__} does not expose an encoder module")
        super().__init__(model, input_names=input_names, family=family, metadata_task="encoder_hidden_states")
        self.encoder = encoder

    def forward(self, *bound_inputs: torch.Tensor | None) -> torch.Tensor:
        outputs = self.encoder(return_dict=True, **self._kwargs_from_bound_inputs(*bound_inputs))
        return _extract_tensor_output(outputs, preferred_field="last_hidden_state")


def _nomic_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class NomicTextEmbeddingAdapter(torch.nn.Module):
    """Export-friendly nomic-bert encoder that returns last_hidden_state.

    Reuses the HF submodules (embeddings, layernorms, attention/MLP weights) but
    reimplements the forward to avoid the HF rotary cache and the data-dependent
    MoE dispatch (`torch.where`/`index_add_`) which do not survive torch.export.
    Pooling and L2 normalization are applied downstream in the C++ engine.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        for name in ("embeddings", "emb_ln", "encoder"):
            if not isinstance(getattr(model, name, None), torch.nn.Module):
                raise NotImplementedError(f"{type(model).__name__} is not a nomic-bert model")
        self.embeddings = model.embeddings
        self.emb_ln = model.emb_ln
        self.encoder = model.encoder
        config = model.config
        self.n_heads = int(config.n_head)
        self.head_dim = int(config.n_embd) // int(config.n_head)
        self.hidden_dim = int(config.n_embd)
        self.n_experts = int(getattr(config, "num_experts", 0) or 0)
        self.top_k = int(getattr(config, "moe_top_k", 0) or 0)
        self.ffn_dim = int(config.n_inner)
        self.rope_base = float(getattr(config, "rotary_emb_base", 10000.0) or 10000.0)
        # Pre-transpose each MoE expert w2 to [hidden, n_experts*ffn] so the second
        # expert matmul consumes it as a direct linear weight — transposing a packed
        # quantized weight inside the graph is not lowerable. Idempotent by shape so
        # repeated adapter construction on the same model is safe.
        packed_rows = self.n_experts * self.ffn_dim
        for layer in self.encoder.layers:
            if getattr(layer, "moe", False):
                w2 = layer.mlp.experts.mlp.w2
                if w2.shape[0] == packed_rows:
                    layer.mlp.experts.mlp.w2 = torch.nn.Parameter(
                        w2.detach().t().contiguous(), requires_grad=False
                    )

    def _rope_cos_sin(self, seq_len: int, device, dtype):
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
        return cos, sin

    def _attention(self, layer, hidden_states, additive_mask, cos, sin):
        batch, seq, _ = hidden_states.shape
        qkv = layer.attn.Wqkv(hidden_states).view(batch, seq, 3, self.n_heads, self.head_dim)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        cos_b = cos[None, :, None, :]
        sin_b = sin[None, :, None, :]
        query = query * cos_b + _nomic_rotate_half(query) * sin_b
        key = key * cos_b + _nomic_rotate_half(key) * sin_b
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + additive_mask
        context = torch.matmul(F.softmax(scores, dim=-1), value)
        context = context.permute(0, 2, 1, 3).reshape(batch, seq, self.hidden_dim)
        return layer.attn.out_proj(context)

    def _topk_gate(self, weights):
        # torch.topk does not lower; build the top-k softmax gate via iterative
        # max + masking (top_k is small). gate[n, e] = softmax weight when expert e
        # is among the top-k for token n, else 0.
        neg = torch.finfo(weights.dtype).min
        remaining = weights
        gate = weights * 0.0
        for _ in range(self.top_k):
            current_max = torch.amax(remaining, dim=-1, keepdim=True)
            selected = (remaining >= current_max).to(weights.dtype)
            gate = gate + selected * weights
            remaining = remaining + selected * neg
        return gate

    def _moe_mlp(self, moe, hidden_states):
        # Dense MoE: run all experts via two fused matmuls against the packed
        # expert weights (no per-expert indexing — slicing a quantized weight is
        # not lowerable), then gate by scaling each expert's activations. The
        # gated sum is identical to top-k routing because non-selected experts
        # have gate 0.
        batch, seq, _ = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_dim)
        weights = moe.router.layer(flat).softmax(dim=-1, dtype=torch.float32)
        gate = self._topk_gate(weights).to(hidden_states.dtype)
        all_hidden = F.linear(flat, moe.experts.mlp.w1)  # [tokens, n_experts * ffn]
        activated = F.gelu(all_hidden).reshape(-1, self.n_experts, self.ffn_dim)
        activated = activated * gate.unsqueeze(-1)
        activated = activated.reshape(-1, self.n_experts * self.ffn_dim)
        # w2 is pre-transposed to [hidden, n_experts*ffn] in __init__.
        out = F.linear(activated, moe.experts.mlp.w2)
        out = out + moe.experts.bias
        return out.reshape(batch, seq, self.hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.emb_ln(self.embeddings.word_embeddings(input_ids))
        seq_len = input_ids.shape[1]
        cos, sin = self._rope_cos_sin(seq_len, input_ids.device, hidden_states.dtype)
        mask_float = attention_mask[:, None, None, :].to(hidden_states.dtype)
        additive_mask = (mask_float - 1.0) * (-torch.finfo(hidden_states.dtype).min)
        for layer in self.encoder.layers:
            hidden_states = layer.norm1(self._attention(layer, hidden_states, additive_mask, cos, sin) + hidden_states)
            mlp_out = self._moe_mlp(layer.mlp, hidden_states) if layer.moe else layer.mlp(hidden_states)
            hidden_states = layer.norm2(mlp_out + hidden_states)
        return hidden_states


class WhisperEncoderComponentAdapter(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.encoder.conv1(input_features))
        hidden_states = F.gelu(self.encoder.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        positions = self.encoder.embed_positions(position_ids)
        hidden_states = hidden_states + positions.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).unsqueeze(0)

        for layer in self.encoder.layers:
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.activation_fn(layer.fc1(hidden_states))
            hidden_states = layer.fc2(hidden_states)
            hidden_states = residual + hidden_states

        return self.encoder.layer_norm(hidden_states)


class WhisperDecoderCrossKVComponentAdapter(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        batch = int(encoder_hidden_states.shape[0])
        for layer in self.decoder.layers:
            attn = layer.encoder_attn
            kv_shape = (batch, -1, int(attn.num_heads), int(attn.head_dim))
            outputs.append(attn.k_proj(encoder_hidden_states).view(kv_shape).contiguous())
            outputs.append(attn.v_proj(encoder_hidden_states).view(kv_shape).contiguous())
        return tuple(outputs)


class WhisperDecoderStepWithCrossKVComponentAdapter(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, proj_out: torch.nn.Module, *, pad_token_id: int | None):
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out
        self.pad_token_id = pad_token_id

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *cross_kv: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.decoder.embed_tokens(decoder_input_ids)
        positions = self.decoder.embed_positions(
            decoder_input_ids,
            past_key_values_length=0,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + positions.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for layer_index, layer in enumerate(self.decoder.layers):
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states, _ = layer.self_attn(
                hidden_states,
                attention_mask=None,
                past_key_values=None,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.encoder_attn_layer_norm(hidden_states)
            attn = layer.encoder_attn
            query_shape = (*hidden_states.shape[:-1], int(attn.num_heads), int(attn.head_dim))
            query_states = (attn.q_proj(hidden_states) * attn.scaling).view(query_shape).transpose(1, 2).contiguous()
            key_states = cross_kv[layer_index * 2].transpose(1, 2).contiguous()
            value_states = cross_kv[layer_index * 2 + 1].transpose(1, 2).contiguous()
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                scale=1.0,
            )
            attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
            hidden_states = residual + attn.out_proj(attn_output)

            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.activation_fn(layer.fc1(hidden_states))
            hidden_states = layer.fc2(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.decoder.layer_norm(hidden_states)
        hidden_states = _select_last_non_pad_token(
            hidden_states,
            decoder_input_ids,
            pad_token_id=self.pad_token_id,
        )
        return self.proj_out(hidden_states)


class NeedleSourceEncoderComponentAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encode = getattr(self.model, "cactus_source_encode", None)
        if not callable(encode):
            raise TypeError(f"{type(self.model).__name__} does not expose cactus_source_encode")
        return encode(input_ids, attention_mask)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="needle",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids", "attention_mask"),
                ),
                "task": "causal_lm_logits",
            }
        }


class NeedleDecoderCrossKVComponentAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cross_kv = getattr(self.model, "cactus_decoder_cross_kv", None)
        if not callable(cross_kv):
            raise TypeError(f"{type(self.model).__name__} does not expose cactus_decoder_cross_kv")
        return cross_kv(encoder_hidden_states, encoder_attention_mask)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="needle",
                    adapter_type=type(self).__name__,
                    input_names=("encoder_hidden_states", "encoder_attention_mask"),
                ),
                "task": "causal_lm_logits",
            }
        }


class NeedleDecoderStepComponentAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        *cross_kv: torch.Tensor,
    ) -> torch.Tensor:
        step = getattr(self.model, "cactus_decoder_step", None)
        if not callable(step):
            raise TypeError(f"{type(self.model).__name__} does not expose cactus_decoder_step")
        return step(decoder_input_ids, position_ids, encoder_attention_mask, *cross_kv)

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="needle",
                    adapter_type=type(self).__name__,
                    input_names=("decoder_input_ids", "position_ids", "encoder_attention_mask"),
                ),
                "task": "causal_lm_logits",
            }
        }


def _family_key(model: torch.nn.Module) -> str:
    explicit_family = getattr(model, "family", None)
    if isinstance(explicit_family, str) and explicit_family:
        return explicit_family.lower()
    module_name = type(model).__module__
    if module_name.startswith("transformers.models.whisper."):
        return "whisper"
    if module_name.startswith("transformers.models.gemma4."):
        return "gemma4"
    if module_name.startswith("transformers.models.gemma3."):
        return "gemma3"
    if module_name.startswith("transformers.models.gemma."):
        return "gemma"
    if module_name.startswith("transformers.models.qwen3_5."):
        return "qwen3_5"
    if module_name.startswith("transformers.models.qwen3_vl."):
        return "qwen3_5"
    if module_name.startswith("transformers.models.qwen2_moe."):
        return "qwen2_moe"
    if module_name.startswith("transformers.models.qwen3."):
        return "qwen3"
    if module_name.startswith("transformers.models.lfm2_vl."):
        return "lfm2_vl"
    if module_name.startswith("transformers.models.lfm2_moe."):
        return "lfm2_moe"
    if module_name.startswith("transformers.models.lfm2."):
        return "lfm2"
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    if model_type == "needle":
        return "needle"
    if model_type == "qwen2_moe":
        return "qwen2_moe"
    if "nomic" in module_name.lower() or "nomic" in model_type:
        return "nomic"
    return "generic"


_ENCODER_CROSS_KV_ROUTE = "encoder_cross_kv_decoder_step"


def _cross_kv_output_keys(num_layers: int) -> tuple[str, ...]:
    return tuple(
        name
        for layer_index in range(int(num_layers))
        for name in (f"cross_k_{layer_index}", f"cross_v_{layer_index}")
    )


def _component_spec(
    *,
    component: str,
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_keys: tuple[str, ...],
    output_keys: tuple[str, ...],
    common_graph_meta: dict[str, object],
    family: str,
    task: str,
    graph_meta: dict[str, object] | None = None,
    runtime_role: str | None = None,
    source_kind: str | None = None,
) -> ComponentModuleSpec:
    component_metadata: dict[str, object] = {"family": family, "task": task}
    if runtime_role is not None:
        component_metadata["runtime_route"] = _ENCODER_CROSS_KV_ROUTE
        component_metadata["runtime_role"] = runtime_role
    if source_kind is not None:
        component_metadata["source_kind"] = source_kind
    return ComponentModuleSpec(
        component=component,
        module=module,
        example_inputs=example_inputs,
        input_keys=input_keys,
        output_keys=output_keys,
        graph_meta={**common_graph_meta, "component": component, **(graph_meta or {})},
        metadata=component_metadata,
    )


def _encoder_cross_kv_step_specs(
    *,
    source_component: str,
    source_module: torch.nn.Module,
    source_inputs: tuple[torch.Tensor, ...],
    source_input_keys: tuple[str, ...],
    source_output_keys: tuple[str, ...],
    source_kind: str,
    cross_kv_module: torch.nn.Module,
    cross_kv_inputs: tuple[torch.Tensor, ...],
    cross_kv_input_keys: tuple[str, ...],
    cross_kv_output_keys: tuple[str, ...],
    decoder_step_module: torch.nn.Module,
    decoder_step_inputs: tuple[torch.Tensor, ...],
    decoder_step_input_keys: tuple[str, ...],
    common_graph_meta: dict[str, object],
    family: str,
    task: str,
    max_cache_seq_len: int,
    decoder_step_graph_meta: dict[str, object] | None = None,
) -> list[ComponentModuleSpec]:
    return [
        _component_spec(
            component=source_component,
            module=source_module,
            example_inputs=source_inputs,
            input_keys=source_input_keys,
            output_keys=source_output_keys,
            common_graph_meta=common_graph_meta,
            family=family,
            task=task,
            runtime_role="source_encoder",
            source_kind=source_kind,
        ),
        _component_spec(
            component="decoder_cross_kv",
            module=cross_kv_module,
            example_inputs=cross_kv_inputs,
            input_keys=cross_kv_input_keys,
            output_keys=cross_kv_output_keys,
            common_graph_meta=common_graph_meta,
            family=family,
            task=task,
            runtime_role="decoder_cross_kv",
        ),
        _component_spec(
            component="decoder_step",
            module=decoder_step_module,
            example_inputs=decoder_step_inputs,
            input_keys=decoder_step_input_keys,
            output_keys=("logits",),
            common_graph_meta=common_graph_meta,
            family=family,
            task=task,
            graph_meta={
                "use_internal_kv_cache": True,
                "max_cache_seq_len": max_cache_seq_len,
                "cache_sink_size": 0,
                **(decoder_step_graph_meta or {}),
            },
            runtime_role="decoder_step",
        ),
    ]


def _build_whisper_seq2seq_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    inputs_metadata: dict[str, object] | None = None,
    weights_dir: str | None = None,
) -> list[ComponentModuleSpec]:
    input_features = named_tensors.get("input_features")
    if not isinstance(input_features, torch.Tensor):
        raise RuntimeError("Whisper component transpile requires input_features")

    get_encoder = getattr(model, "get_encoder", None)
    encoder = get_encoder() if callable(get_encoder) else None
    if not isinstance(encoder, torch.nn.Module):
        encoder = getattr(getattr(model, "model", None), "encoder", None)
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    proj_out = getattr(model, "proj_out", None)
    if not isinstance(encoder, torch.nn.Module) or not isinstance(decoder, torch.nn.Module):
        raise RuntimeError(f"{type(model).__name__} does not expose Whisper encoder/decoder modules")
    if not isinstance(proj_out, torch.nn.Module):
        raise RuntimeError(f"{type(model).__name__} does not expose a Whisper projection head")

    metadata = dict(inputs_metadata or {})
    decoder_prompt_ids = metadata.get("decoder_input_ids")
    if not isinstance(decoder_prompt_ids, list) or not decoder_prompt_ids:
        decoder_start_token_id = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
        if not isinstance(decoder_start_token_id, int):
            raise RuntimeError("Whisper component transpile requires decoder_input_ids metadata")
        decoder_prompt_ids = [int(decoder_start_token_id)]
    else:
        decoder_prompt_ids = [int(value) for value in decoder_prompt_ids]

    target_token_count = int(metadata.get("target_token_count", len(decoder_prompt_ids)) or len(decoder_prompt_ids))
    target_token_count = max(target_token_count, len(decoder_prompt_ids))
    pad_token_id = int(metadata.get("pad_token_id", getattr(getattr(model, "config", None), "pad_token_id", 0)) or 0)
    encoder_adapter = WhisperEncoderComponentAdapter(encoder).eval()
    decoder_cross_kv_adapter = WhisperDecoderCrossKVComponentAdapter(decoder).eval()
    decoder_step_adapter = WhisperDecoderStepWithCrossKVComponentAdapter(
        decoder,
        proj_out,
        pad_token_id=pad_token_id,
    ).eval()

    with torch.no_grad():
        encoder_hidden_states = encoder_adapter(input_features)
        decoder_cross_kv = decoder_cross_kv_adapter(encoder_hidden_states)

    decoder_input_ids = torch.tensor([[decoder_prompt_ids[0]]], dtype=torch.int64, device=input_features.device)
    step_position_ids = torch.zeros_like(decoder_input_ids, dtype=torch.int64)

    common_graph_meta = {
        **_transpile_graph_meta(
            model,
            adapter_family="whisper",
            adapter_type="component_pipeline",
            input_names=("input_features",),
        ),
        "task": "seq2seq_transcription",
        "adapter_family": "whisper",
    }
    if weights_dir:
        common_graph_meta["weights_dir"] = weights_dir
    cross_kv_output_keys = _cross_kv_output_keys(len(decoder.layers))
    return _encoder_cross_kv_step_specs(
        source_component="audio_encoder",
        source_module=encoder_adapter,
        source_inputs=(input_features,),
        source_input_keys=("input_features",),
        source_output_keys=("encoder_hidden_states",),
        source_kind="audio_features",
        cross_kv_module=decoder_cross_kv_adapter,
        cross_kv_inputs=(encoder_hidden_states,),
        cross_kv_input_keys=("encoder_hidden_states",),
        cross_kv_output_keys=cross_kv_output_keys,
        decoder_step_module=decoder_step_adapter,
        decoder_step_inputs=(decoder_input_ids, step_position_ids, *decoder_cross_kv),
        decoder_step_input_keys=("decoder_input_ids", "position_ids", *cross_kv_output_keys),
        common_graph_meta=common_graph_meta,
        family="whisper",
        task="seq2seq_transcription",
        max_cache_seq_len=max(16, int(target_token_count)),
    )


def _resolve_decoder_start_token_id(model: torch.nn.Module, *, fallback: int = 0) -> int:
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    for owner in (config, generation_config):
        for attr_name in ("decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(owner, attr_name, None) if owner is not None else None
            if value is not None:
                return int(value)
    return int(fallback)

NEEDLE_CONTEXT_LENGTH = 1024

def _build_needle_causal_lm_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
) -> list[ComponentModuleSpec] | None:
    input_ids = named_tensors.get("input_ids")
    if input_ids is None:
        return None
    for method_name in ("cactus_source_encode", "cactus_decoder_cross_kv", "cactus_decoder_step"):
        if not callable(getattr(model, method_name, None)):
            raise RuntimeError(f"{type(model).__name__} does not expose {method_name}")

    pad_token_id = _resolve_model_pad_token_id(model)
    current_len = int(input_ids.shape[1])
    if current_len < NEEDLE_CONTEXT_LENGTH:
        pad_value = int(pad_token_id) if pad_token_id is not None else 0
        padding = torch.full(
            (int(input_ids.shape[0]), NEEDLE_CONTEXT_LENGTH - current_len),
            pad_value,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, padding], dim=1)
    if pad_token_id is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    else:
        attention_mask = (input_ids != int(pad_token_id)).to(dtype=torch.int64)
    source_encoder = NeedleSourceEncoderComponentAdapter(model).eval()
    decoder_cross_kv = NeedleDecoderCrossKVComponentAdapter(model).eval()
    decoder_step = NeedleDecoderStepComponentAdapter(model).eval()

    with torch.no_grad():
        encoder_hidden_states, encoder_attention_mask = source_encoder(input_ids, attention_mask)
        cross_kv = decoder_cross_kv(encoder_hidden_states, encoder_attention_mask)

    decoder_start_token_id = _resolve_decoder_start_token_id(model, fallback=int(input_ids[0, 0].item()))
    decoder_input_ids = torch.full(
        (int(input_ids.shape[0]), 1),
        decoder_start_token_id,
        dtype=torch.int64,
        device=input_ids.device,
    )
    step_position_ids = torch.zeros_like(decoder_input_ids, dtype=torch.int64)
    cross_kv_output_keys = _cross_kv_output_keys(len(cross_kv) // 2)

    common_graph_meta = {
        **_transpile_graph_meta(
            model,
            adapter_family="needle",
            adapter_type="component_pipeline",
            input_names=("input_ids",),
        ),
        "task": "causal_lm_logits",
        "adapter_family": "needle",
    }
    if weights_dir:
        common_graph_meta["weights_dir"] = weights_dir

    return _encoder_cross_kv_step_specs(
        source_component="source_encoder",
        source_module=source_encoder,
        source_inputs=(input_ids, attention_mask),
        source_input_keys=("input_ids", "attention_mask"),
        source_output_keys=("encoder_hidden_states", "encoder_attention_mask"),
        source_kind="text_tokens",
        cross_kv_module=decoder_cross_kv,
        cross_kv_inputs=(encoder_hidden_states, encoder_attention_mask),
        cross_kv_input_keys=("encoder_hidden_states", "encoder_attention_mask"),
        cross_kv_output_keys=cross_kv_output_keys,
        decoder_step_module=decoder_step,
        decoder_step_inputs=(decoder_input_ids, step_position_ids, encoder_attention_mask, *cross_kv),
        decoder_step_input_keys=("decoder_input_ids", "position_ids", "encoder_attention_mask", *cross_kv_output_keys),
        common_graph_meta=common_graph_meta,
        family="needle",
        task="causal_lm_logits",
        max_cache_seq_len=max(16, int(input_ids.shape[1])),
        decoder_step_graph_meta={
            "external_cross_kv_cache_inputs": True,
            "cross_kv_input_start_index": 3,
            "cross_kv_input_count": len(cross_kv_output_keys),
            "cross_kv_input_layout": "bthd",
        },
    )


def _build_nomic_text_embedding_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
) -> list[ComponentModuleSpec]:
    input_ids = named_tensors.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("nomic text_embedding transpile requires input_ids")
    attention_mask = named_tensors.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.ones_like(input_ids)

    adapter = NomicTextEmbeddingAdapter(model).eval()

    common_graph_meta = {
        **_transpile_graph_meta(
            model,
            adapter_family="nomic",
            adapter_type="component_pipeline",
            input_names=("input_ids", "attention_mask"),
        ),
        "task": "text_embedding",
        "adapter_family": "nomic",
    }
    if weights_dir:
        common_graph_meta["weights_dir"] = weights_dir

    return [
        ComponentModuleSpec(
            component="text_embedding",
            module=adapter,
            example_inputs=(input_ids, attention_mask),
            input_keys=("input_ids", "attention_mask"),
            output_keys=("last_hidden_state",),
            graph_meta={**common_graph_meta, "component": "text_embedding"},
            metadata={"family": "nomic", "task": "text_embedding"},
        )
    ]


def _build_lfm2_causal_lm_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
) -> list[ComponentModuleSpec] | None:
    input_ids = named_tensors.get("input_ids")
    if input_ids is None:
        return None
    requested = tuple(components or ("decoder", "decoder_step"))
    requested_set = set(requested)
    family = _family_key(model)
    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "causal_lm_logits",
        "adapter_family": family,
    }
    metadata = {"family": family, "task": "causal_lm_logits"}
    chunk_components = {"lm_encoder_step", "lm_encoder_text_chunk", "decoder_prefill_chunk"}
    if chunk_components & requested_set:
        return _lfm2_chunked_pipeline_specs(
            model,
            requested_set=requested_set,
            input_ids=input_ids,
            weights_dir=weights_dir,
            cache_context_length=cache_context_length,
            common_graph_meta=common_graph_meta,
            metadata=metadata,
        )

    pad_token_id = _resolve_model_pad_token_id(model)
    decoder = Lfm2CausalLMLogitsAdapter(model, pad_token_id=pad_token_id).eval()
    decoder_step = Lfm2CausalLMStepAdapter(model, pad_token_id=pad_token_id).eval()
    specs: list[ComponentModuleSpec] = []
    if "decoder" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=(input_ids,),
            input_keys=("input_ids",),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata=metadata,
        ))
    if "decoder_step" in requested_set:
        step_input_ids = input_ids[:, :1]
        step_position_ids = torch.zeros_like(step_input_ids, dtype=torch.int64)
        max_cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512)
        specs.append(ComponentModuleSpec(
            component="decoder_step",
            module=decoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("logits",),
            graph_meta={
                **common_graph_meta,
                "component": "decoder_step",
                "use_internal_kv_cache": True,
                "use_internal_conv_cache": True,
                "use_internal_gated_deltanet_cache": True,
                "max_cache_seq_len": max_cache_seq_len,
                "cache_sink_size": 4,
            },
            metadata=metadata,
        ))
    return specs


def _lfm2_chunked_pipeline_specs(
    model: torch.nn.Module,
    *,
    requested_set: set[str],
    input_ids: torch.Tensor,
    weights_dir: str | None,
    cache_context_length: str | int | None,
    common_graph_meta: dict[str, object],
    metadata: dict[str, str],
    decoder_inputs_fn: Callable[[], tuple[torch.Tensor, ...]] | None = None,
) -> list[ComponentModuleSpec]:
    lm_encoder_step = Lfm2VlLMEncoderStepAdapter(model, weights_dir=weights_dir).eval()
    lm_encoder_text_chunk = Lfm2VlLMEncoderTextChunkAdapter(model, weights_dir=weights_dir).eval()
    decoder_last_token = Lfm2VlDecoderAdapter(model, weights_dir=weights_dir, last_token_only=True).eval()
    decoder_embed = Lfm2VlDecoderAdapter(model, weights_dir=weights_dir, last_token_only=False, return_hidden=True).eval()

    prefill_chunk = max(2, int(os.environ.get("CACTUS_LFM2_PREFILL_CHUNK", "128") or "128"))
    chunk_input_ids = _tile_to_length(input_ids, prefill_chunk)
    chunk_position_ids = torch.arange(
        prefill_chunk, dtype=torch.long, device=input_ids.device,
    ).unsqueeze(0).expand(int(input_ids.shape[0]), -1).contiguous()
    step_input_ids = input_ids[:, :1]
    step_position_ids = torch.zeros_like(step_input_ids, dtype=torch.int64)

    decoder_inputs: tuple[torch.Tensor, ...] | None = None
    if {"decoder_prefill_chunk", "decoder_embed_chunk", "decoder_step"} & requested_set:
        if decoder_inputs_fn is not None:
            decoder_inputs = decoder_inputs_fn()
        else:
            with torch.no_grad():
                decoder_inputs = lm_encoder_text_chunk(chunk_input_ids, chunk_position_ids)

    cache_graph_meta = {
        "use_internal_kv_cache": True,
        "use_internal_conv_cache": True,
        "max_cache_seq_len": _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512),
        "cache_sink_size": 4,
    }

    specs: list[ComponentModuleSpec] = []
    if "lm_encoder_step" in requested_set:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_step",
            module=lm_encoder_step,
            example_inputs=(step_input_ids, step_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("inputs_embeds", "attention_mask", "position_ids"),
            graph_meta={**common_graph_meta, "component": "lm_encoder_step"},
            metadata=metadata,
        ))
    if "lm_encoder_text_chunk" in requested_set:
        specs.append(ComponentModuleSpec(
            component="lm_encoder_text_chunk",
            module=lm_encoder_text_chunk,
            example_inputs=(chunk_input_ids, chunk_position_ids),
            input_keys=("input_ids", "position_ids"),
            output_keys=("inputs_embeds", "attention_mask", "position_ids"),
            graph_meta={
                **common_graph_meta,
                "component": "lm_encoder_text_chunk",
                "encoder_chunk_size": prefill_chunk,
            },
            metadata=metadata,
        ))
    if "decoder_prefill_chunk" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder_prefill_chunk",
            module=decoder_last_token,
            example_inputs=tuple(tensor[:, :prefill_chunk, ...] for tensor in decoder_inputs),
            input_keys=("inputs_embeds", "attention_mask", "position_ids"),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder_prefill_chunk", **cache_graph_meta},
            metadata=metadata,
        ))
    if "decoder_embed_chunk" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder_embed_chunk",
            module=decoder_embed,
            example_inputs=tuple(tensor[:, :prefill_chunk, ...] for tensor in decoder_inputs),
            input_keys=("inputs_embeds", "attention_mask", "position_ids"),
            output_keys=("last_hidden_state",),
            graph_meta={**common_graph_meta, "component": "decoder_embed_chunk", **cache_graph_meta},
            metadata=metadata,
        ))
    if "decoder_step" in requested_set:
        specs.append(ComponentModuleSpec(
            component="decoder_step",
            module=decoder_last_token,
            example_inputs=tuple(tensor[:, :1, ...] for tensor in decoder_inputs),
            input_keys=("inputs_embeds", "attention_mask", "position_ids"),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder_step", **cache_graph_meta},
            metadata=metadata,
        ))
    return specs


def _build_lfm2_vl_multimodal_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
) -> list[ComponentModuleSpec]:
    input_ids = named_tensors["input_ids"]
    attention_mask = named_tensors["attention_mask"]
    pixel_values = named_tensors["pixel_values"]
    spatial_shapes = named_tensors["spatial_shapes"]
    pixel_attention_mask = named_tensors["pixel_attention_mask"]
    text_context = max(8, int(os.environ.get("CACTUS_LFM2_TEXT_CONTEXT", "128") or "128"))
    text_context = min(text_context, int(input_ids.shape[1]))
    text_input_ids = torch.zeros(
        (int(input_ids.shape[0]), text_context),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    text_attention_mask = torch.ones(
        (int(attention_mask.shape[0]), text_context),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    requested_components = tuple(components or ("vision_encoder", "lm_encoder", "text_lm_encoder", "decoder"))
    requested_set = set(requested_components)
    if not requested_set:
        return []

    expanded_components: list[str] = []

    def _require(component: str) -> None:
        if component not in expanded_components:
            expanded_components.append(component)

    if "decoder" in requested_set:
        _require("vision_encoder")
        _require("vision_projector")
        _require("lm_encoder")
        _require("text_lm_encoder")
        _require("decoder")
        _require("text_decoder")
        _require("lm_encoder_step")
        _require("lm_encoder_text_chunk")
        _require("decoder_prefill_chunk")
        _require("decoder_embed_chunk")
        _require("decoder_step")
    if "lm_encoder" in requested_set:
        _require("vision_encoder")
        _require("vision_projector")
        _require("lm_encoder")
    if "lm_encoder_step" in requested_set:
        _require("lm_encoder_step")
    if "lm_encoder_text_chunk" in requested_set:
        _require("lm_encoder_text_chunk")
    if "decoder_step" in requested_set:
        _require("lm_encoder_step")
        _require("decoder_step")
    if "decoder_prefill_chunk" in requested_set:
        _require("lm_encoder_text_chunk")
        _require("decoder_prefill_chunk")
    if "text_lm_encoder" in requested_set:
        _require("text_lm_encoder")
    if "text_decoder" in requested_set:
        _require("text_lm_encoder")
        _require("text_decoder")
    if "vision_encoder" in requested_set:
        _require("vision_encoder")
        _require("vision_projector")

    vision_encoder = Lfm2VlVisionEncoderAdapter(model, weights_dir=weights_dir).eval()
    vision_projector = Lfm2VlVisionProjectorAdapter(model, weights_dir=weights_dir).eval()

    _vemb = vision_encoder.embeddings
    with torch.no_grad():
        _pos_grid = _vemb.position_embedding.weight.reshape(
            _vemb.position_embedding_size, _vemb.position_embedding_size, -1
        )
        example_positional_embeddings = _vemb.resize_positional_embeddings(
            _pos_grid, spatial_shapes.detach().cpu(), max_length=int(pixel_attention_mask.shape[1])
        ).to(pixel_values.dtype)
    _root = _lfm2_vl_model_root(model)
    _factor = int(_root.multi_modal_projector.factor)
    _proj_in = int(_vemb.embed_dim) * _factor * _factor
    _max_subimage_tokens = int(getattr(_root.config, "max_image_tokens", 256) or 256)
    example_vision_features = torch.zeros((_max_subimage_tokens, _proj_in), dtype=pixel_values.dtype)

    lm_encoder = Lfm2VlLMEncoderAdapter(model, weights_dir=weights_dir).eval()
    text_lm_encoder = Lfm2VlLMEncoderAdapter(model, weights_dir=weights_dir).eval()
    decoder = Lfm2VlDecoderAdapter(model, weights_dir=weights_dir, last_token_only=False).eval()

    decoder_inputs: tuple[torch.Tensor, ...] | None = None
    text_decoder_inputs: tuple[torch.Tensor, ...] | None = None
    with torch.no_grad():
        if "lm_encoder" in expanded_components or "decoder" in expanded_components:
            decoder_inputs = lm_encoder(input_ids, attention_mask)
        if "text_lm_encoder" in expanded_components:
            text_decoder_inputs = text_lm_encoder(text_input_ids, text_attention_mask)

    def _ensure_decoder_inputs() -> tuple[torch.Tensor, ...]:
        nonlocal decoder_inputs
        if decoder_inputs is None:
            decoder_inputs = lm_encoder(input_ids, attention_mask)
        return decoder_inputs

    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "multimodal_causal_lm_logits",
        "adapter_family": "lfm2_vl",
    }
    metadata = {"family": "lfm2_vl", "task": "multimodal_causal_lm_logits"}
    max_cache_seq_len = _max_cache_seq_len(model, input_ids, cache_context_length, fallback_extra_tokens=512)
    lm_seq = max(int(input_ids.shape[1]), int(max_cache_seq_len))
    lm_example_input_ids = torch.zeros((int(input_ids.shape[0]), lm_seq), dtype=input_ids.dtype, device=input_ids.device)
    lm_example_attention_mask = torch.ones((int(attention_mask.shape[0]), lm_seq), dtype=attention_mask.dtype, device=attention_mask.device)
    specs: list[ComponentModuleSpec] = []
    if "vision_encoder" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="vision_encoder",
            module=vision_encoder,
            example_inputs=(pixel_values, pixel_attention_mask, example_positional_embeddings),
            input_keys=("pixel_values", "pixel_attention_mask", "positional_embeddings"),
            output_keys=("last_hidden_state",),
            graph_meta={**common_graph_meta, "component": "vision_encoder"},
            metadata=metadata,
            npu_runtime_input_count=3,
        ))
    if "vision_projector" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="vision_projector",
            module=vision_projector,
            example_inputs=(example_vision_features,),
            input_keys=("vision_features",),
            output_keys=("image_features",),
            graph_meta={**common_graph_meta, "component": "vision_projector"},
            metadata=metadata,
        ))
    if "lm_encoder" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="lm_encoder",
            module=lm_encoder,
            example_inputs=(lm_example_input_ids, lm_example_attention_mask),
            input_keys=("input_ids", "attention_mask"),
            output_keys=("inputs_embeds", "attention_mask", "position_ids"),
            graph_meta={**common_graph_meta, "component": "lm_encoder"},
            metadata=metadata,
        ))
    if "text_lm_encoder" in expanded_components:
        if text_decoder_inputs is None:
            raise RuntimeError("LFM2-VL text_lm_encoder spec requires precomputed text decoder inputs")
        specs.append(ComponentModuleSpec(
            component="text_lm_encoder",
            module=text_lm_encoder,
            example_inputs=(text_input_ids, text_attention_mask),
            input_keys=("input_ids", "attention_mask"),
            output_keys=("inputs_embeds", "attention_mask", "position_ids"),
            graph_meta={**common_graph_meta, "component": "text_lm_encoder"},
            metadata=metadata,
        ))
    if "decoder" in expanded_components:
        if decoder_inputs is None:
            decoder_inputs = text_decoder_inputs
        if decoder_inputs is None:
            raise RuntimeError("LFM2-VL decoder spec requires precomputed decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=decoder_inputs,
            input_keys=("inputs_embeds", "attention_mask", "position_ids"),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata=metadata,
        ))
    if "text_decoder" in expanded_components:
        if text_decoder_inputs is None:
            raise RuntimeError("LFM2-VL text_decoder spec requires precomputed text decoder inputs")
        specs.append(ComponentModuleSpec(
            component="text_decoder",
            module=decoder,
            example_inputs=text_decoder_inputs,
            input_keys=("inputs_embeds", "attention_mask", "position_ids"),
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "text_decoder"},
            metadata=metadata,
        ))
    specs.extend(_lfm2_chunked_pipeline_specs(
        model,
        requested_set=set(expanded_components),
        input_ids=input_ids,
        weights_dir=weights_dir,
        cache_context_length=cache_context_length,
        common_graph_meta=common_graph_meta,
        metadata=metadata,
        decoder_inputs_fn=_ensure_decoder_inputs,
    ))
    return specs


def build_component_module_specs(
    model: torch.nn.Module,
    *,
    task: str,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
    inputs_metadata: dict[str, object] | None = None,
    components: tuple[str, ...] | None = None,
    cache_context_length: str | int | None = None,
    dynamic_batch: bool = True,
    max_slots: int = 1,
) -> list[ComponentModuleSpec] | None:
    family = _family_key(model)
    if family == "qwen3_5" and task == "multimodal_causal_lm_logits":
        return _build_qwen3_5_multimodal_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
            cache_context_length=cache_context_length,
        )
    if family in {"qwen2_moe", "qwen3", "qwen3_5"} and task == "causal_lm_logits":
        return _build_qwen_causal_lm_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
            cache_context_length=cache_context_length,
            dynamic_batch=dynamic_batch,
            max_slots=max_slots,
        )
    if family == "gemma3" and task == "causal_lm_logits":
        return _build_gemma3_causal_lm_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
            cache_context_length=cache_context_length,
            dynamic_batch=dynamic_batch,
            max_slots=max_slots,
        )
    if family == "gemma4" and task == "causal_lm_logits":
        return _build_gemma4_causal_lm_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
            cache_context_length=cache_context_length,
            dynamic_batch=dynamic_batch,
            max_slots=max_slots,
        )
    if family == "gemma4" and task == "multimodal_causal_lm_logits":
        return _build_gemma4_multimodal_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
            cache_context_length=cache_context_length,
            dynamic_batch=dynamic_batch,
            max_slots=max_slots,
        )
    if family == "lfm2_vl" and task == "multimodal_causal_lm_logits":
        return _build_lfm2_vl_multimodal_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
            cache_context_length=cache_context_length,
        )
    if family in {"lfm2", "lfm2_moe"} and task == "causal_lm_logits":
        return _build_lfm2_causal_lm_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
            cache_context_length=cache_context_length,
        )
    if family == "needle" and task == "causal_lm_logits":
        return _build_needle_causal_lm_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
        )
    if family == "parakeet_tdt" and task == "tdt_transcription":
        from cactus.transpile.tdt_runtime import build_parakeet_tdt_component_specs

        return build_parakeet_tdt_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
        )
    if family == "whisper" and task == "seq2seq_transcription":
        return _build_whisper_seq2seq_component_specs(
            model,
            named_tensors=named_tensors,
            inputs_metadata=inputs_metadata,
            weights_dir=weights_dir,
        )
    if family == "nomic" and task == "text_embedding":
        return _build_nomic_text_embedding_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
        )
    return None


def canonicalize_model_interface(
    model: torch.nn.Module,
    task: str = "causal_lm_logits",
    *,
    input_names: tuple[str, ...] | None = None,
    weights_dir: str | None = None,
    inputs_metadata: dict[str, object] | None = None,
) -> CanonicalizedModel:
    family = _family_key(model)
    adapter_factory: Callable[[torch.nn.Module], torch.nn.Module]
    resolved_input_names = tuple(input_names or ())
    padding_token_id_value = None
    if isinstance(inputs_metadata, dict):
        raw_padding_token_id = inputs_metadata.get("padding_token_id")
        if isinstance(raw_padding_token_id, int):
            padding_token_id_value = int(raw_padding_token_id)

    if task == "causal_lm_logits":
        if family == "gemma":
            adapter_factory = lambda inner_model: GemmaCausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        elif family == "gemma4":
            adapter_factory = lambda inner_model: Gemma4CausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        elif family == "gemma3":
            adapter_factory = lambda inner_model: Gemma3CausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        elif family == "qwen3_5":
            adapter_factory = lambda inner_model: Qwen35CausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        elif family == "qwen2_moe":
            adapter_factory = lambda inner_model: Qwen2MoeCausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        elif family == "qwen3":
            adapter_factory = lambda inner_model: Qwen3CausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        elif family in {"lfm2", "lfm2_vl", "lfm2_moe"}:
            adapter_factory = lambda inner_model: Lfm2CausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        else:
            adapter_factory = lambda inner_model: CausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                pad_token_id=padding_token_id_value,
            )
        resolved_input_names = ("input_ids",)
    elif task == "multimodal_causal_lm_logits":
        if family not in {"gemma4", "lfm2_vl", "qwen3_5"}:
            raise NotImplementedError(f"{type(model).__name__} does not support task={task}")
        if family == "qwen3_5":
            if not resolved_input_names:
                resolved_input_names = (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "pixel_values",
                    "image_grid_thw",
                )
            adapter_factory = lambda inner_model: BoundInputAdapter(  # type: ignore[assignment]
                inner_model,
                input_names=resolved_input_names,
                family="qwen3_5",
                metadata_task="multimodal_causal_lm_logits",
            )
        elif family == "lfm2_vl":
            if not resolved_input_names:
                resolved_input_names = (
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "spatial_shapes",
                    "pixel_attention_mask",
                )
            adapter_factory = lambda inner_model: Lfm2VlMultimodalCausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                input_names=resolved_input_names,
            )
        else:
            if not resolved_input_names:
                resolved_input_names = (
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "pixel_values",
                    "pixel_position_ids",
                    "input_features",
                    "input_features_mask",
                )
            adapter_factory = lambda inner_model: Gemma4MultimodalCausalLMLogitsAdapter(  # type: ignore[assignment]
                inner_model,
                input_names=resolved_input_names,
                weights_dir=weights_dir,
            )
    elif task == "ctc_logits":
        if not resolved_input_names:
            resolved_input_names = _infer_input_names(
                model,
                preferred=("input_values", "input_features", "attention_mask"),
            )
        adapter_factory = lambda inner_model: CTCLogitsAdapter(  # type: ignore[assignment]
            inner_model,
            input_names=resolved_input_names,
            family=family,
        )
    elif task == "encoder_hidden_states":
        if not resolved_input_names:
            resolved_input_names = _infer_input_names(
                model,
                preferred=("input_features", "input_values", "attention_mask"),
            )
        adapter_factory = lambda inner_model: EncoderHiddenStatesAdapter(  # type: ignore[assignment]
            inner_model,
            input_names=resolved_input_names,
            family=family,
        )
    elif task == "text_embedding":
        if not resolved_input_names:
            resolved_input_names = ("input_ids", "attention_mask")
        adapter_factory = NomicTextEmbeddingAdapter
    else:
        raise NotImplementedError(f"unsupported task={task}")

    return CanonicalizedModel(
        module=adapter_factory(model).eval(),
        task=task,
        family=family,
        input_names=resolved_input_names,
    )
