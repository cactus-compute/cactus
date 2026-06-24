from __future__ import annotations

from collections import Counter

import torch

from cactus.transpile.aten_ops import canonical_torch_op
from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.normalize import normalize_target
from cactus.transpile.optimize_graph import fuse_dense_mlp_tq


class OddlyNamedLinearBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pear_tree = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.pear_tree(x))


class SameOpsDifferentNames(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.not_a_transformer_layer = torch.nn.ModuleDict(
            {"banana": torch.nn.Linear(4, 3, bias=False)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.not_a_transformer_layer["banana"](x))


def _op_counts(module: torch.nn.Module) -> Counter[str]:
    example = torch.randn(2, 4)
    captured = capture_model(module, (example,))
    return Counter(captured.ir_graph.nodes[node_id].op for node_id in captured.ir_graph.order)


def test_aten_targets_normalize_from_real_op_overloads() -> None:
    assert canonical_torch_op(torch.ops.aten.add.Tensor) == "aten.add.Tensor"
    assert normalize_target(torch.ops.aten.add.Tensor) == "add"
    assert normalize_target(torch.ops.aten.linear.default) == "linear"
    assert normalize_target("torch.ops.aten.silu.default") == "silu"


def test_import_is_based_on_aten_ops_not_module_names() -> None:
    first = _op_counts(OddlyNamedLinearBlock())
    second = _op_counts(SameOpsDifferentNames())

    assert first == second
    assert first["linear"] == 1
    assert first["silu"] == 1


def test_import_records_canonical_aten_op_metadata() -> None:
    captured = capture_model(OddlyNamedLinearBlock(), (torch.randn(2, 4),))
    node_meta = [captured.ir_graph.nodes[node_id].meta for node_id in captured.ir_graph.order]

    assert {meta["aten_op"] for meta in node_meta} >= {"aten.linear.default", "aten.silu.default"}
    assert all("torch_name" in meta for meta in node_meta)
    assert all("module_paths" not in meta for meta in node_meta)


def test_dense_mlp_fusion_uses_topology_not_layer_names() -> None:
    graph = IRGraph(
        values={
            "x": IRValue(id="x", shape=(1, 4), dtype="fp16"),
            "pear_weight": IRValue(id="pear_weight", shape=(8, 4), dtype="fp16", meta={"path": "pear.cq4.weights"}),
            "banana_weight": IRValue(id="banana_weight", shape=(8, 4), dtype="fp16", meta={"path": "banana.cq4.weights"}),
            "plum_weight": IRValue(id="plum_weight", shape=(4, 8), dtype="fp16", meta={"path": "plum.cq4.weights"}),
            "gate": IRValue(id="gate", shape=(1, 8), dtype="fp16", producer="gate_linear"),
            "activated": IRValue(id="activated", shape=(1, 8), dtype="fp16", producer="gate_gelu"),
            "up": IRValue(id="up", shape=(1, 8), dtype="fp16", producer="up_linear"),
            "product": IRValue(id="product", shape=(1, 8), dtype="fp16", producer="product"),
            "out": IRValue(id="out", shape=(1, 4), dtype="fp16", producer="down_linear"),
        },
        nodes={
            "gate_linear": IRNode(
                id="gate_linear",
                op="linear",
                inputs=["x", "pear_weight"],
                outputs=["gate"],
                attrs={"has_bias": False},
            ),
            "gate_gelu": IRNode(
                id="gate_gelu",
                op="gelu",
                inputs=["gate"],
                outputs=["activated"],
            ),
            "up_linear": IRNode(
                id="up_linear",
                op="linear",
                inputs=["x", "banana_weight"],
                outputs=["up"],
                attrs={"has_bias": False},
            ),
            "product": IRNode(
                id="product",
                op="multiply",
                inputs=["activated", "up"],
                outputs=["product"],
            ),
            "down_linear": IRNode(
                id="down_linear",
                op="linear",
                inputs=["product", "plum_weight"],
                outputs=["out"],
                attrs={"has_bias": False},
            ),
        },
        order=["gate_linear", "gate_gelu", "up_linear", "product", "down_linear"],
        inputs=["x"],
        outputs=["out"],
        constants={},
        meta={},
    )

    assert fuse_dense_mlp_tq(graph) is True
    assert graph.nodes["down_linear"].op == "dense_mlp_tq_fused"
    assert graph.nodes["down_linear"].inputs == ["x", "pear_weight", "banana_weight", "plum_weight"]






def _ensure_test_lfm2_moe_op_registered() -> None:
    from cactus.transpile.model_adapters import _ensure_lfm2_moe_custom_op_registered
    _ensure_lfm2_moe_custom_op_registered()


def _ensure_test_qwen2_moe_op_registered() -> None:
    from cactus.transpile.model_adapters import _ensure_qwen2_moe_custom_op_registered
    _ensure_qwen2_moe_custom_op_registered()


def _ensure_test_gemma4_moe_op_registered() -> None:
    from cactus.transpile.model_adapters import _ensure_gemma4_moe_custom_op_registered
    _ensure_gemma4_moe_custom_op_registered()


class _FakeLfm2MoeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(4, 2, bias=False)
        self.w1 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)])
        self.w3 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)])
        self.w2 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(4, 3)) for _ in range(2)])
        self.register_buffer("expert_bias", torch.zeros(2, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return torch.ops.cactus_transpile.lfm2_moe_layer_gated(
            hidden_flat,
            self.gate(hidden_flat),
            self.expert_bias,
            list(self.w1),
            list(self.w3),
            list(self.w2),
            2,
            1,
            True,
            True,
            1.0e-6,
            1.0,
        ).reshape(hidden_states.shape)


class _FakeQwen2MoeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(4, 2, bias=False)
        self.w1 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)])
        self.w3 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)])
        self.w2 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(4, 3)) for _ in range(2)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return torch.ops.cactus_transpile.qwen2_moe_layer_gated(
            hidden_flat,
            self.gate(hidden_flat),
            list(self.w1),
            list(self.w3),
            list(self.w2),
            2,
            1,
            False,
            1.0e-6,
            1.0,
        ).reshape(hidden_states.shape)


class _FakeGemma4MoeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = torch.nn.Linear(4, 2, bias=False)
        self.w1 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)])
        self.w3 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)])
        self.w2 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(4, 3)) for _ in range(2)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return torch.ops.cactus_transpile.gemma4_moe_layer_gated(
            hidden_flat,
            self.router(hidden_flat),
            list(self.w1),
            list(self.w3),
            list(self.w2),
            2,
            1,
            True,
            1.0e-6,
            1.0,
        ).reshape(hidden_states.shape)


def test_lfm2_moe_runtime_block_exports_single_semantic_op() -> None:
    _ensure_test_lfm2_moe_op_registered()
    block = _FakeLfm2MoeBlock().eval()
    captured = capture_model(block, (torch.randn(1, 2, 4),), strict=False)

    moe_nodes = [
        captured.ir_graph.nodes[node_id]
        for node_id in captured.ir_graph.order
        if captured.ir_graph.nodes[node_id].op == "lfm2_moe_layer_gated"
    ]

    assert len(moe_nodes) == 1
    assert moe_nodes[0].attrs["num_experts"] == 2
    assert moe_nodes[0].attrs["num_experts_per_tok"] == 1
    assert moe_nodes[0].attrs["use_expert_bias"] is True
    assert len(moe_nodes[0].inputs) == 3 + 3 * 2


def test_qwen2_moe_runtime_block_exports_single_semantic_op() -> None:
    _ensure_test_qwen2_moe_op_registered()
    block = _FakeQwen2MoeBlock().eval()
    captured = capture_model(block, (torch.randn(1, 2, 4),), strict=False)

    moe_nodes = [
        captured.ir_graph.nodes[node_id]
        for node_id in captured.ir_graph.order
        if captured.ir_graph.nodes[node_id].op == "qwen2_moe_layer_gated"
    ]

    assert len(moe_nodes) == 1
    assert moe_nodes[0].attrs["num_experts"] == 2
    assert moe_nodes[0].attrs["num_experts_per_tok"] == 1
    assert moe_nodes[0].attrs["normalize_routing"] is False
    assert len(moe_nodes[0].inputs) == 2 + 3 * 2


def test_gemma4_moe_runtime_block_exports_single_semantic_op() -> None:
    _ensure_test_gemma4_moe_op_registered()
    block = _FakeGemma4MoeBlock().eval()
    captured = capture_model(block, (torch.randn(1, 2, 4),), strict=False)

    moe_nodes = [
        captured.ir_graph.nodes[node_id]
        for node_id in captured.ir_graph.order
        if captured.ir_graph.nodes[node_id].op == "gemma4_moe_layer_gated"
    ]

    assert len(moe_nodes) == 1
    assert moe_nodes[0].attrs["num_experts"] == 2
    assert moe_nodes[0].attrs["num_experts_per_tok"] == 1
    assert moe_nodes[0].attrs["normalize_routing"] is True
    assert len(moe_nodes[0].inputs) == 2 + 3 * 2
