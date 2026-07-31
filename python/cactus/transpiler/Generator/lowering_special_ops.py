from __future__ import annotations

from typing import Any

from . import models
from .errors import UnsupportedLoweringError
from .lowering_utils import *
from ..IR import models as IRModels


def lower_moe(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)

    if node.target == "cactus.dense_mlp_tq_fused":
        require_len(node, inputs, 4)
        hidden = cast_to_precision(context, inputs[0], context.graph.FP16)
        return context.graph.dense_mlp_tq_fused(
            hidden,
            inputs[1],
            inputs[2],
            inputs[3],
            product_scale=float(node.attrs.get("product_scale", 1.0)),
        )

    if node.target == "cactus.moe_layer_gated":
        packed = lower_packed_lfm_moe_layer_gated(context, node, inputs)
        if packed is not None:
            return packed

        if len(inputs) < 6:
            raise UnsupportedLoweringError(
                f"{node.name}: cactus.moe_layer_gated requires hidden, routing_probs, topk_indices, w1, w3, and w2 inputs; "
                "packed/grouped MoE lowering still needs a weight-unpacking policy"
            )

        return context.graph.moe_layer_gated(
            inputs[0],
            inputs[1],
            cast_to_precision(context, inputs[2], context.graph.FP32),
            ensure_tensor_sequence(inputs[3]),
            ensure_tensor_sequence(inputs[4]),
            ensure_tensor_sequence(inputs[5]),
            required_int_attr(node, "num_experts"),
            required_int_attr(node, "num_experts_per_tok"),
            normalize_routing=bool(node.attrs.get("normalize_routing", True)),
            epsilon=float(node.attrs.get("epsilon", 1e-6)),
            routed_scaling_factor=float(node.attrs.get("routed_scaling_factor", 1.0)),
        )

    if node.target == "cactus.moe_layer_ungated":
        if len(inputs) < 5:
            raise unsupported_arity(node, len(inputs), "hidden, routing_probs, topk_indices, w1, and w2")

        return context.graph.moe_layer_ungated(
            inputs[0],
            inputs[1],
            cast_to_precision(context, inputs[2], context.graph.FP32),
            ensure_tensor_sequence(inputs[3]),
            ensure_tensor_sequence(inputs[4]),
            required_int_attr(node, "num_experts"),
            required_int_attr(node, "num_experts_per_tok"),
            normalize_routing=bool(node.attrs.get("normalize_routing", True)),
            epsilon=float(node.attrs.get("epsilon", 1e-6)),
            routed_scaling_factor=float(node.attrs.get("routed_scaling_factor", 1.0)),
        )

    raise UnsupportedLoweringError(f"{node.name}: unsupported MoE target {node.target}")


def lower_packed_lfm_moe_layer_gated(
    context: models.GenerationContext,
    node: IRModels.Node,
    inputs: tuple[Any, ...],
) -> Any | None:
    if len(inputs) != 5 or len(node.parents) != 5:
        return None

    hidden_node, router_node, bias_node, gate_up_node, down_node = node.parents
    gate_up_shape = concrete_shape(meta_shape(gate_up_node))
    down_shape = concrete_shape(meta_shape(down_node))
    num_experts = required_int_attr(node, "num_experts")
    top_k = required_int_attr(node, "num_experts_per_tok")

    if gate_up_shape is None or down_shape is None:
        return None

    if len(gate_up_shape) != 3 or len(down_shape) != 3:
        return None

    if gate_up_shape[0] != num_experts or down_shape[0] != num_experts:
        return None

    packed_intermediate, hidden_dim = gate_up_shape[1], gate_up_shape[2]
    if packed_intermediate % 2 != 0:
        raise UnsupportedLoweringError(f"{node.name}: packed gated MoE gate/up dimension must be even")

    intermediate_dim = packed_intermediate // 2

    if down_shape[1:] != (hidden_dim, intermediate_dim):
        raise UnsupportedLoweringError(
            f"{node.name}: expected LFM packed down_proj shape "
            f"[experts, hidden, intermediate], got {down_shape}"
        )

    hidden = moe_hidden_2d(context, node, inputs[0], hidden_dim)
    router_weight = inputs[1]
    expert_bias = cast_to_precision(context, inputs[2], context.graph.FP16)
    gate_up_weight = inputs[3]
    down_weight = inputs[4]

    router_logits = context.graph.matmul(hidden, router_weight, pretransposed_rhs=True)
    router_logits = context.graph.add(router_logits, expert_bias)
    routing_probs = context.graph.softmax(cast_to_precision(context, router_logits, context.graph.FP16), axis=-1)
    topk_indices = context.graph.index(context.graph.topk(routing_probs, top_k), 0, axis=0)
    topk_indices = cast_to_precision(context, topk_indices, context.graph.FP32)
    bundled_weights = lfm_moe_weight_bundle_parts(gate_up_weight, down_weight, num_experts)

    if bundled_weights is None:
        w1_weights, w3_weights, w2_weights = split_lfm_packed_moe_weights(
            context,
            gate_up_weight,
            down_weight,
            num_experts,
            intermediate_dim,
            hidden_dim,
        )
    else:
        w1_weights, w3_weights, w2_weights = bundled_weights

    return context.graph.moe_layer_gated(
        hidden,
        routing_probs,
        topk_indices,
        w1_weights,
        w3_weights,
        w2_weights,
        num_experts,
        top_k,
        normalize_routing=bool(node.attrs.get("normalize_routing", True)),
        epsilon=float(node.attrs.get("epsilon", 1e-6)),
        routed_scaling_factor=float(node.attrs.get("routed_scaling_factor", 1.0)),
    )


def moe_hidden_2d(context: models.GenerationContext, node: IRModels.Node, hidden: Any, hidden_dim: int) -> Any:
    hidden_shape = concrete_shape(meta_shape(node.parents[0])) if node.parents else None

    if hidden_shape is None:
        return cast_to_precision(context, hidden, context.graph.FP16)

    if len(hidden_shape) == 2:
        return cast_to_precision(context, hidden, context.graph.FP16)

    if len(hidden_shape) == 3 and hidden_shape[0] == 1 and hidden_shape[2] == hidden_dim:
        return cast_to_precision(context, context.graph.reshape(hidden, (hidden_shape[1], hidden_dim)), context.graph.FP16)

    raise UnsupportedLoweringError(f"{node.name}: packed MoE hidden input must be [tokens, hidden] or [1, tokens, hidden]")


def lfm_moe_weight_bundle_parts(
    gate_up_weight: Any,
    down_weight: Any,
    num_experts: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]] | None:
    if not isinstance(gate_up_weight, models.LfmMoeWeightBundle):
        return None

    if not isinstance(down_weight, models.LfmMoeWeightBundle):
        return None

    if (
        len(gate_up_weight.w1_weights) != num_experts
        or len(gate_up_weight.w3_weights) != num_experts
        or len(down_weight.w2_weights) != num_experts
    ):
        return None

    return gate_up_weight.w1_weights, gate_up_weight.w3_weights, down_weight.w2_weights


def split_lfm_packed_moe_weights(
    context: models.GenerationContext,
    gate_up_weight: Any,
    down_weight: Any,
    num_experts: int,
    intermediate_dim: int,
    hidden_dim: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    w1_weights = []
    w3_weights = []
    w2_weights = []

    for expert_index in range(num_experts):
        expert_gate_up = context.graph.reshape(
            context.graph.slice(gate_up_weight, 0, expert_index, length=1),
            (2 * intermediate_dim, hidden_dim),
        )
        w1_weights.append(context.graph.slice(expert_gate_up, 0, 0, length=intermediate_dim))
        w3_weights.append(context.graph.slice(expert_gate_up, 0, intermediate_dim, length=intermediate_dim))

        expert_down = context.graph.reshape(
            context.graph.slice(down_weight, 0, expert_index, length=1),
            (hidden_dim, intermediate_dim),
        )
        w2_weights.append(context.graph.transpose(expert_down))

    return tuple(w1_weights), tuple(w3_weights), tuple(w2_weights)


def lower_special_cactus(context: models.GenerationContext, node: IRModels.Node) -> Any:
    inputs = context.inputs_for(node)
    target = node.target

    if target == "cactus.rope":
        require_len(node, inputs, 1)
        return context.graph.rope(
            inputs[0],
            theta=float(attr_value(node, "theta", 10_000.0)),
            position_offset=int(attr_value(node, "position_offset", 0)),
        )

    if target == "cactus.rope_gptj":
        require_len(node, inputs, 1)
        return context.graph.rope_gptj(
            inputs[0],
            theta=float(attr_value(node, "theta", 10_000.0)),
            position_offset=int(attr_value(node, "position_offset", 0)),
            rot_dim=int(attr_value(node, "rot_dim", 0)),
        )

    if target == "cactus.glu":
        require_len(node, inputs, 1)
        if meta_shape(node) == meta_shape(node.parents[0]):
            return inputs[0]
        return context.graph.glu(inputs[0], axis=axis_attr(node, default=-1))

    if target == "cactus.lstm_cell":
        require_len(node, inputs, 7)
        return context.graph.lstm_cell(*inputs[:7])

    if target == "cactus.gated_deltanet_decode":
        require_len(node, inputs, 6)
        return context.graph.gated_deltanet_decode(*inputs[:6], scale=float(node.attrs.get("scale", 1.0)))

    if target == "cactus.gated_deltanet_prefill":
        require_len(node, inputs, 6)
        return context.graph.gated_deltanet_prefill(*inputs[:6], chunk_size=int(node.attrs.get("chunk_size", 1)), scale=float(node.attrs.get("scale", 1.0)))

    if target == "cactus.lfm_short_conv_decode":
        require_len(node, inputs, 2)
        weight = lfm_short_conv_decode_weight(context, node, inputs[1])
        weight = cast_to_precision(context, weight, getattr(inputs[0], "dtype", context.graph.FP16))
        product = context.graph.multiply(inputs[0], weight)
        return context.graph.sum(product, axis=normalize_dim(-1, len(meta_shape(node.parents[0]))))

    if target == "cactus.rel_pos_bias":
        require_len(node, inputs, 2)
        return context.graph.rel_pos_bias(inputs[0], inputs[1], scale=float(node.attrs.get("scale") or 1.0))

    if target == "cactus.sample":
        require_len(node, inputs, 1)
        return context.graph.sample(inputs[0], temperature=float(node.attrs.get("temperature", 0.6)), top_p=float(node.attrs.get("top_p", 0.95)), top_k=int(node.attrs.get("top_k", 20)))

    if target == "cactus.scatter_topk":
        require_len(node, inputs, 2)
        return context.graph.scatter_topk(inputs[0], inputs[1], num_classes=required_int_attr(node, "num_classes"))

    if target == "cactus.gaussian_topk":
        require_len(node, inputs, 1)
        return context.graph.gaussian_topk(inputs[0], ppf=float(node.attrs.get("ppf", 0.0)))

    if target == "cactus.altup_predict":
        require_len(node, inputs, 2)
        return context.graph.altup_predict(inputs[0], list(inputs[1:]))

    if target == "cactus.altup_correct":
        require_len(node, inputs, 3)
        return context.graph.altup_correct(inputs[0], inputs[1], list(inputs[2:]))

    if target == "cactus.stft":
        require_len(node, inputs, 2)
        return context.graph.stft(inputs[0], inputs[1], stride=required_int_attr(node, "stride"), num_fft_bins=required_int_attr(node, "num_fft_bins"))

    if target == "cactus.rfft":
        require_len(node, inputs, 1)
        return context.graph.rfft(inputs[0])

    if target == "cactus.irfft":
        require_len(node, inputs, 1)
        return context.graph.irfft(inputs[0], output_length=required_int_attr(node, "output_length"))

    if target == "cactus.spectrogram":
        require_len(node, inputs, 2)
        return context.graph.spectrogram(
            inputs[0],
            inputs[1],
            frame_length=required_int_attr(node, "frame_length"),
            hop_length=required_int_attr(node, "hop_length"),
            fft_length=required_int_attr(node, "fft_length"),
            power=float(node.attrs.get("power", 2.0)),
        )

    if target == "cactus.image_preprocess":
        require_len(node, inputs, 1)
        if not has_attrs(node, ("src_width", "src_height", "target_width", "target_height", "patch_size", "channels")):
            return context.graph.reshape(inputs[0], output_shape(node))

        return context.graph.image_preprocess(
            inputs[0],
            src_width=required_int_attr(node, "src_width"),
            src_height=required_int_attr(node, "src_height"),
            target_width=required_int_attr(node, "target_width"),
            target_height=required_int_attr(node, "target_height"),
            patch_size=required_int_attr(node, "patch_size"),
            channels=required_int_attr(node, "channels"),
            rescale_factor=float(node.attrs.get("rescale_factor", 1.0)),
            mean=node.attrs.get("mean", (0.0, 0.0, 0.0)),
            std_dev=node.attrs.get("std_dev", (1.0, 1.0, 1.0)),
        )

    if target == "cactus.bilinear_interpolation":
        require_len(node, inputs, 1)
        return context.graph.bilinear_interpolation(inputs[0], required_int_attr(node, "dst_height"), required_int_attr(node, "dst_width"))

    raise UnsupportedLoweringError(f"{node.name}: unsupported special Cactus target {target}")


def lfm_short_conv_decode_weight(context: models.GenerationContext, node: IRModels.Node, weight: Any) -> Any:
    cache_window_shape = meta_shape(node.parents[0]) if node.parents else ()
    weight_shape = meta_shape(node.parents[1]) if len(node.parents) > 1 else ()

    if len(cache_window_shape) != 3:
        return weight

    batch_size, hidden_dim, window_size = cache_window_shape

    if not all(isinstance(dim, int) for dim in cache_window_shape):
        return weight

    if tuple(weight_shape) == tuple(cache_window_shape):
        return weight

    if len(weight_shape) == 2 and tuple(weight_shape) == (hidden_dim, window_size):
        return context.graph.reshape(weight, (batch_size, hidden_dim, window_size))

    if len(weight_shape) == 3 and tuple(weight_shape) in {
        (1, hidden_dim, window_size),
        (hidden_dim, 1, window_size),
    }:
        return context.graph.reshape(weight, (batch_size, hidden_dim, window_size))

    raise UnsupportedLoweringError(
        f"{node.name}: lfm short conv decode weight shape {weight_shape} is incompatible with cache window {cache_window_shape}"
    )


def lower_unsupported_semantic(context: models.GenerationContext, node: IRModels.Node) -> Any:
    raise UnsupportedLoweringError(f"{node.name}: {node.target} has no safe Cactus lowering yet")
