from __future__ import annotations

from cactus.transpile.component_partition import COMPONENT_LM_ENCODER
from cactus.transpile.component_partition import COMPONENT_VISION_ENCODER
from cactus.transpile.component_partition import classify_node_component
from cactus.transpile.component_partition import classify_value_component
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue


def test_component_partition_ignores_torch_layer_names() -> None:
    value = IRValue(
        id="weight",
        meta={
            "source_name": "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            "torch_name": "model.layers.0.mlp.gate_proj.weight",
        },
    )

    assert classify_value_component(
        "weight",
        value,
        family="",
        task="multimodal_causal_lm_logits",
    ) is None


def test_component_partition_uses_semantic_input_names() -> None:
    assert (
        classify_value_component(
            "pixel_values",
            IRValue(id="pixel_values"),
            family="",
            task="multimodal_causal_lm_logits",
        )
        == COMPONENT_VISION_ENCODER
    )


def test_component_partition_uses_ops_for_multimodal_merge() -> None:
    node = IRNode(
        id="embedding_0",
        op="embedding",
        inputs=["input_ids", "weight"],
        outputs=["inputs_embeds"],
    )

    assert (
        classify_node_component(
            node,
            family="",
            task="multimodal_causal_lm_logits",
        )
        == COMPONENT_LM_ENCODER
    )
