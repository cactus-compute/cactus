from __future__ import annotations

import torch

from cactus.convert.model_adapters.adapters import adapter_for_family


def test_gemma4_moe_down_proj_folds_per_expert_scale_before_split():
    adapter = adapter_for_family("gemma4")
    down = torch.arange(2 * 4 * 3, dtype=torch.float16).reshape(2, 4, 3)
    scale = torch.tensor([2.0, 0.5], dtype=torch.float16)

    normalized = adapter.normalize_state_dict({
        "model.language_model.layers.5.moe.down_proj": down,
        "model.language_model.layers.5.moe.per_expert_scale": scale,
    })
    scaled = normalized.state_dict["model.language_model.layers.5.moe.down_proj"]
    match = adapter.name_tensor(
        "model.language_model.layers.5.moe.down_proj",
        scaled,
        num_layers=30,
    )
    emissions = adapter.expand_tensor(match, scaled)

    assert torch.equal(emissions[0].tensor, down[0] * scale[0])
    assert torch.equal(emissions[1].tensor, down[1] * scale[1])
    assert [emission.output_name for emission in emissions] == [
        "layer_5_moe_expert_0_w2.weights",
        "layer_5_moe_expert_1_w2.weights",
    ]
    assert "model.language_model.layers.5.moe.w2_weights.0" in emissions[0].source_names
    assert normalized.provenance["model.language_model.layers.5.moe.down_proj"].source_names == [
        "model.language_model.layers.5.moe.down_proj",
        "model.language_model.layers.5.moe.per_expert_scale",
    ]


def test_gemma4_a4b_experts_down_proj_folds_router_per_expert_scale():
    adapter = adapter_for_family("gemma4")
    down = torch.arange(2 * 4 * 3, dtype=torch.float16).reshape(2, 4, 3)
    scale = torch.tensor([2.0, 0.5], dtype=torch.float16)

    normalized = adapter.normalize_state_dict({
        "model.language_model.layers.5.experts.down_proj": down,
        "model.language_model.layers.5.router.per_expert_scale": scale,
    })
    scaled = normalized.state_dict["model.language_model.layers.5.experts.down_proj"]
    match = adapter.name_tensor(
        "model.language_model.layers.5.experts.down_proj",
        scaled,
        num_layers=30,
    )
    emissions = adapter.expand_tensor(match, scaled)

    assert torch.equal(emissions[0].tensor, down[0] * scale[0])
    assert torch.equal(emissions[1].tensor, down[1] * scale[1])
    assert [emission.output_name for emission in emissions] == [
        "layer_5_moe_expert_0_w2.weights",
        "layer_5_moe_expert_1_w2.weights",
    ]
    assert "model.language_model.layers.5.moe.w2_weights.0" in emissions[0].source_names


def test_gemma4_moe_gate_up_splits_into_runtime_experts():
    adapter = adapter_for_family("gemma4")
    tensor = torch.arange(2 * 6 * 4, dtype=torch.float16).reshape(2, 6, 4)

    match = adapter.name_tensor(
        "model.language_model.layers.5.moe.gate_up_proj",
        tensor,
        num_layers=30,
    )
    emissions = adapter.expand_tensor(match, tensor)

    assert [emission.output_name for emission in emissions] == [
        "layer_5_moe_expert_0_w1.weights",
        "layer_5_moe_expert_0_w3.weights",
        "layer_5_moe_expert_1_w1.weights",
        "layer_5_moe_expert_1_w3.weights",
    ]
    assert torch.equal(emissions[0].tensor, tensor[0, :3, :])
    assert torch.equal(emissions[1].tensor, tensor[0, 3:, :])
    assert "model.language_model.layers.5.moe.w1_weights.0" in emissions[0].source_names
    assert "model.language_model.layers.5.moe.w3_weights.1" in emissions[3].source_names


def test_gemma4_a4b_experts_gate_up_splits_into_runtime_experts():
    adapter = adapter_for_family("gemma4")
    tensor = torch.arange(2 * 6 * 4, dtype=torch.float16).reshape(2, 6, 4)

    match = adapter.name_tensor(
        "model.language_model.layers.5.experts.gate_up_proj",
        tensor,
        num_layers=30,
    )
    emissions = adapter.expand_tensor(match, tensor)

    assert [emission.output_name for emission in emissions] == [
        "layer_5_moe_expert_0_w1.weights",
        "layer_5_moe_expert_0_w3.weights",
        "layer_5_moe_expert_1_w1.weights",
        "layer_5_moe_expert_1_w3.weights",
    ]
    assert torch.equal(emissions[0].tensor, tensor[0, :3, :])
    assert torch.equal(emissions[3].tensor, tensor[1, 3:, :])


def test_gemma4_moe_router_policy_stays_fp16():
    adapter = adapter_for_family("gemma4")
    tensor = torch.randn(128, 2816, dtype=torch.float16)
    match = adapter.name_tensor(
        "model.language_model.layers.5.router.proj.weight",
        tensor,
        num_layers=30,
    )
    policy = adapter.policy(match, tuple(tensor.shape), requested_bits=4)

    assert match.output_name == "layer_5_router_proj.weights"
    assert policy.action == "fallback"
    assert policy.precision == "FP16"
