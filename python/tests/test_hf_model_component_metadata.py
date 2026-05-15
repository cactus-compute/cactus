from __future__ import annotations

from cactus.transpile import hf_model
from cactus.transpile.component_partition import extract_component_subgraphs
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue


def test_non_pipeline_causal_lm_graph_is_stamped_for_decoder_component() -> None:
    graph = IRGraph(
        values={
            "input_ids": IRValue(id="input_ids", shape=(1, 8), dtype="int64", users=["n_logits"]),
            "logits": IRValue(id="logits", shape=(1, 8, 3), dtype="fp16", producer="n_logits"),
        },
        nodes={
            "n_logits": IRNode(
                id="n_logits",
                op="reshape",
                inputs=["input_ids"],
                outputs=["logits"],
                meta={},
            )
        },
        order=["n_logits"],
        inputs=["input_ids"],
        outputs=["logits"],
        meta={"adapter_family": "generic"},
    )

    hf_model._stamp_transpile_graph_metadata(graph, task="causal_lm_logits", family="generic")

    component_graphs = extract_component_subgraphs(graph)
    assert list(component_graphs) == ["decoder"]
