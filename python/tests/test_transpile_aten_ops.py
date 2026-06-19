from __future__ import annotations

from collections import Counter

import torch

from cactus.transpile.aten_ops import canonical_torch_op
from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.lower import transpile_preoptimized_ir
from cactus.transpile.normalize import normalize_target
from cactus.transpile.optimize_graph import fuse_dense_mlp_tq
from cactus.transpile.optimize_graph import optimize_graph


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


class ManualLSTMCell(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_ih = torch.nn.Parameter(torch.randn(16, 4) * 0.2)
        self.weight_hh = torch.nn.Parameter(torch.randn(16, 4) * 0.2)
        self.bias = torch.nn.Parameter(torch.randn(16) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.nn.functional.linear(x, self.weight_ih) + torch.nn.functional.linear(h, self.weight_hh) + self.bias
        i, f, g, o = torch.chunk(gates, 4, dim=-1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class ReusedLSTMStateOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell0 = ManualLSTMCell()
        self.cell1 = ManualLSTMCell()

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        h1: torch.Tensor,
        c1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0_next, c0_next = self.cell0(x, h0, c0)
        h1_next, _ = self.cell1(h0_next, h1, c1)
        return h1_next, h0_next, c0_next


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


def test_reused_lstm_state_output_is_materialized() -> None:
    torch.manual_seed(3)
    module = ReusedLSTMStateOutput().eval().to(dtype=torch.float16)
    inputs = tuple(torch.randn(1, 4, dtype=torch.float16) for _ in range(5))
    with torch.no_grad():
        expected = module(*inputs)

    captured = capture_model(module, inputs)
    optimized = optimize_graph(captured.ir_graph)
    lowered = transpile_preoptimized_ir(optimized)
    lowered.set_inputs([value.numpy() for value in inputs])
    actual = [value.numpy() for value in lowered.execute()]

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            torch.from_numpy(got),
            want.detach().cpu(),
            atol=2e-3,
            rtol=2e-3,
        )


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
