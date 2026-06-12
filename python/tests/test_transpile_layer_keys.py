from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.lower import _kv_cache_layer_key
from cactus.transpile.optimize_graph import normalize_cached_decoder_attention_hints


def _decoder_graph(component: str) -> IRGraph:
    values: dict[str, IRValue] = {}
    nodes: dict[str, IRNode] = {}
    for index in range(2):
        node_id = "n_scaled_dot_product_attention" + ("" if index == 0 else f"_{index}")
        for name in ("query", "key", "value"):
            values[f"{name}_{index}"] = IRValue(id=f"{name}_{index}", shape=(1, 8, 1, 64), dtype="fp16")
        values[f"out_{index}"] = IRValue(id=f"out_{index}", shape=(1, 8, 1, 64), dtype="fp16", producer=node_id)
        nodes[node_id] = IRNode(
            id=node_id,
            op="scaled_dot_product_attention",
            inputs=[f"query_{index}", f"key_{index}", f"value_{index}"],
            outputs=[f"out_{index}"],
        )
    return IRGraph(
        values=values,
        nodes=nodes,
        order=list(nodes),
        inputs=[value_id for value_id in values if not value_id.startswith("out_")],
        outputs=["out_1"],
        constants={},
        meta={
            "component": component,
            "use_internal_kv_cache": True,
            "layer_types": ["full_attention"] * 2,
        },
    )


def test_kv_cache_layer_key_keeps_zero_layer_index() -> None:
    node = IRNode(
        id="n_scaled_dot_product_attention",
        op="scaled_dot_product_attention",
        inputs=["query_0", "key_0", "value_0"],
        outputs=["out_0"],
    )
    assert _kv_cache_layer_key(node) == "n_scaled_dot_product_attention"
    node.meta["layer_index"] = 5
    assert _kv_cache_layer_key(node) == "5"
    node.meta["attention_layer_index"] = 0
    assert _kv_cache_layer_key(node) == "0"


def test_media_step_receives_attention_layer_indices() -> None:
    graph = _decoder_graph("decoder_media_step")

    changed = normalize_cached_decoder_attention_hints(graph)

    assert changed is True
    assert graph.nodes["n_scaled_dot_product_attention"].meta["attention_layer_index"] == 0
    assert graph.nodes["n_scaled_dot_product_attention_1"].meta["attention_layer_index"] == 1


def test_step_and_media_step_layer_keys_match() -> None:
    keys = []
    for component in ("decoder_step", "decoder_media_step"):
        graph = _decoder_graph(component)
        normalize_cached_decoder_attention_hints(graph)
        keys.append([_kv_cache_layer_key(graph.nodes[node_id]) for node_id in graph.order])

    assert keys[0] == keys[1] == ["0", "1"]
