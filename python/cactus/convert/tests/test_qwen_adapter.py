from __future__ import annotations

import torch

from cactus.convert.model_adapters.adapters import adapter_for_family


def test_qwen2_moe_packed_gate_up_splits_into_runtime_experts():
    adapter = adapter_for_family("qwen")
    tensor = torch.arange(2 * 6 * 4, dtype=torch.float16).reshape(2, 6, 4)

    match = adapter.name_tensor(
        "model.layers.5.mlp.experts.gate_up_proj",
        tensor,
        num_layers=24,
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
    assert "model.layers.5.mlp.w1_weights.0" in emissions[0].source_names
    assert "model.layers.5.mlp.w3_weights.1" in emissions[3].source_names


def test_qwen2_moe_packed_down_splits_into_runtime_experts():
    adapter = adapter_for_family("qwen")
    tensor = torch.arange(2 * 4 * 3, dtype=torch.float16).reshape(2, 4, 3)

    match = adapter.name_tensor(
        "model.layers.5.mlp.experts.down_proj",
        tensor,
        num_layers=24,
    )
    emissions = adapter.expand_tensor(match, tensor)

    assert [emission.output_name for emission in emissions] == [
        "layer_5_moe_expert_0_w2.weights",
        "layer_5_moe_expert_1_w2.weights",
    ]
    assert torch.equal(emissions[0].tensor, tensor[0])
    assert torch.equal(emissions[1].tensor, tensor[1])
    assert "model.layers.5.mlp.w2_weights.0" in emissions[0].source_names


def test_qwen2_moe_router_uses_runtime_router_name_and_fp16_policy():
    adapter = adapter_for_family("qwen")
    tensor = torch.randn(60, 2048, dtype=torch.float16)

    match = adapter.name_tensor(
        "model.layers.5.mlp.gate.weight",
        tensor,
        num_layers=24,
    )
    policy = adapter.policy(match, tuple(tensor.shape), requested_bits=4)

    assert match.output_name == "layer_5_moe_router.weights"
    assert policy.action == "fallback"
    assert policy.precision == "FP16"
