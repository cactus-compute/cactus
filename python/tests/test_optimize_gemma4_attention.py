from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.optimize_graph import normalize_gemma4_decoder_attention_semantics


def test_normalize_gemma4_full_attention_uses_sequence_window_compat() -> None:
    graph = IRGraph(
        values={
            "query": IRValue(id="query", shape=(1, 8, 800, 512), dtype="fp16"),
            "key": IRValue(id="key", shape=(1, 1, 800, 512), dtype="fp16"),
            "value": IRValue(id="value", shape=(1, 1, 800, 512), dtype="fp16"),
            "full_attention_mask": IRValue(
                id="full_attention_mask",
                shape=(1, 1, 800, 800),
                dtype="bool",
            ),
            "proj": IRValue(id="proj", shape=(1536, 4096), dtype="fp16"),
            "out": IRValue(id="out", shape=(1, 800, 1536), dtype="fp16", producer="attn"),
        },
        nodes={
            "attn": IRNode(
                id="attn",
                op="attention_block",
                inputs=["query", "key", "value", "full_attention_mask", "proj"],
                outputs=["out"],
                attrs={
                    "has_mask": True,
                    "is_causal": False,
                    "window_size": 0,
                    "attention_output_shape": (1, 800, 4096),
                },
                meta={"attention_layer_type": "full_attention"},
            )
        },
        order=["attn"],
        inputs=["query", "key", "value", "full_attention_mask"],
        outputs=["out"],
        constants={},
        meta={
            "adapter_family": "gemma4",
            "component": "decoder",
            "input_names": ("query", "key", "value", "full_attention_mask"),
        },
    )

    changed = normalize_gemma4_decoder_attention_semantics(graph)

    assert changed is True
    assert graph.nodes["attn"].inputs == ["query", "key", "value", "proj"]
    assert graph.nodes["attn"].attrs["has_mask"] is False
    assert graph.nodes["attn"].attrs["is_causal"] is True
    assert graph.nodes["attn"].attrs["window_size"] == 800
    assert graph.nodes["attn"].meta["gemma4_full_attention_window_compat"] is True


def test_normalize_gemma4_sliding_attention_elides_runtime_mask() -> None:
    graph = IRGraph(
        values={
            "query": IRValue(id="query", shape=(1, 8, 800, 256), dtype="fp16"),
            "key": IRValue(id="key", shape=(1, 8, 800, 256), dtype="fp16"),
            "value": IRValue(id="value", shape=(1, 8, 800, 256), dtype="fp16"),
            "sliding_attention_mask": IRValue(
                id="sliding_attention_mask",
                shape=(1, 1, 800, 800),
                dtype="bool",
            ),
            "proj": IRValue(id="proj", shape=(1536, 2048), dtype="fp16"),
            "out": IRValue(id="out", shape=(1, 800, 1536), dtype="fp16", producer="attn"),
        },
        nodes={
            "attn": IRNode(
                id="attn",
                op="attention_block",
                inputs=["query", "key", "value", "sliding_attention_mask", "proj"],
                outputs=["out"],
                attrs={
                    "has_mask": True,
                    "is_causal": False,
                    "window_size": 0,
                    "attention_output_shape": (1, 800, 2048),
                },
                meta={"attention_layer_type": "sliding_attention"},
            )
        },
        order=["attn"],
        inputs=["query", "key", "value", "sliding_attention_mask"],
        outputs=["out"],
        constants={},
        meta={
            "adapter_family": "gemma4",
            "component": "decoder",
            "sliding_window": 512,
            "input_names": ("query", "key", "value", "sliding_attention_mask"),
        },
    )

    changed = normalize_gemma4_decoder_attention_semantics(graph)

    assert changed is True
    assert graph.nodes["attn"].inputs == ["query", "key", "value", "proj"]
    assert graph.nodes["attn"].attrs["has_mask"] is False
    assert graph.nodes["attn"].attrs["is_causal"] is True
    assert graph.nodes["attn"].attrs["window_size"] == 512


def test_normalize_gemma4_attention_recovers_layer_hints_from_graph_meta() -> None:
    graph = IRGraph(
        values={
            "query_0": IRValue(id="query_0", shape=(1, 8, 800, 256), dtype="fp16"),
            "key_0": IRValue(id="key_0", shape=(1, 8, 800, 256), dtype="fp16"),
            "value_0": IRValue(id="value_0", shape=(1, 8, 800, 256), dtype="fp16"),
            "sliding_attention_mask": IRValue(
                id="sliding_attention_mask",
                shape=(1, 1, 800, 800),
                dtype="bool",
            ),
            "out_0": IRValue(id="out_0", shape=(1, 8, 800, 256), dtype="fp16", producer="attn_0"),
            "query_1": IRValue(id="query_1", shape=(1, 8, 800, 256), dtype="fp16"),
            "key_1": IRValue(id="key_1", shape=(1, 8, 800, 256), dtype="fp16"),
            "value_1": IRValue(id="value_1", shape=(1, 8, 800, 256), dtype="fp16"),
            "full_attention_mask": IRValue(
                id="full_attention_mask",
                shape=(1, 1, 800, 800),
                dtype="bool",
            ),
            "out_1": IRValue(id="out_1", shape=(1, 8, 800, 256), dtype="fp16", producer="attn_1"),
        },
        nodes={
            "attn_0": IRNode(
                id="attn_0",
                op="attention",
                inputs=["query_0", "key_0", "value_0", "sliding_attention_mask"],
                outputs=["out_0"],
                attrs={"is_causal": False, "window_size": 0},
            ),
            "attn_1": IRNode(
                id="attn_1",
                op="attention",
                inputs=["query_1", "key_1", "value_1", "full_attention_mask"],
                outputs=["out_1"],
                attrs={"is_causal": False, "window_size": 0},
            ),
        },
        order=["attn_0", "attn_1"],
        inputs=[
            "query_0",
            "key_0",
            "value_0",
            "sliding_attention_mask",
            "query_1",
            "key_1",
            "value_1",
            "full_attention_mask",
        ],
        outputs=["out_1"],
        constants={},
        meta={
            "adapter_family": "gemma4",
            "component": "decoder",
            "sliding_window": 512,
            "layer_types": ("sliding_attention", "full_attention"),
        },
    )

    changed = normalize_gemma4_decoder_attention_semantics(graph)

    assert changed is True
    assert graph.nodes["attn_0"].meta["attention_layer_type"] == "sliding_attention"
    assert graph.nodes["attn_0"].meta["attention_layer_index"] == 0
    assert graph.nodes["attn_0"].inputs == ["query_0", "key_0", "value_0"]
    assert graph.nodes["attn_0"].attrs["is_causal"] is True
    assert graph.nodes["attn_0"].attrs["window_size"] == 512
    assert graph.nodes["attn_1"].meta["attention_layer_type"] == "full_attention"
    assert graph.nodes["attn_1"].meta["attention_layer_index"] == 1
    assert graph.nodes["attn_1"].inputs == ["query_1", "key_1", "value_1"]
    assert graph.nodes["attn_1"].attrs["is_causal"] is True
    assert graph.nodes["attn_1"].attrs["window_size"] == 800


def test_normalize_gemma4_attention_elides_logical_fp16_mask() -> None:
    graph = IRGraph(
        values={
            "query": IRValue(id="query", shape=(1, 8, 800, 256), dtype="fp16"),
            "key": IRValue(id="key", shape=(1, 8, 800, 256), dtype="fp16"),
            "value": IRValue(id="value", shape=(1, 8, 800, 256), dtype="fp16"),
            "mask_a": IRValue(id="mask_a", shape=(1, 1, 800, 800), dtype="bool"),
            "mask_b": IRValue(id="mask_b", shape=(1, 1, 800, 800), dtype="bool"),
            "mask_fp16": IRValue(id="mask_fp16", shape=(1, 1, 800, 800), dtype="fp16", producer="mask_and"),
            "out": IRValue(id="out", shape=(1, 8, 800, 256), dtype="fp16", producer="attn"),
        },
        nodes={
            "mask_and": IRNode(
                id="mask_and",
                op="logical_and",
                inputs=["mask_a", "mask_b"],
                outputs=["mask_fp16"],
            ),
            "attn": IRNode(
                id="attn",
                op="attention",
                inputs=["query", "key", "value", "mask_fp16"],
                outputs=["out"],
                attrs={"is_causal": False, "window_size": 0},
                meta={"attention_layer_type": "sliding_attention"},
            ),
        },
        order=["mask_and", "attn"],
        inputs=["query", "key", "value", "mask_a", "mask_b"],
        outputs=["out"],
        constants={},
        meta={
            "adapter_family": "gemma4",
            "component": "decoder",
            "sliding_window": 512,
        },
    )

    changed = normalize_gemma4_decoder_attention_semantics(graph)

    assert changed is True
    assert graph.nodes["attn"].inputs == ["query", "key", "value"]
    assert graph.nodes["attn"].attrs["is_causal"] is True
    assert graph.nodes["attn"].attrs["window_size"] == 512
