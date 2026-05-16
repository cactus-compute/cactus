import importlib
import sys
import types

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue


class FakeTensor:
    def __init__(self, tensor_id: int, shape: tuple[int, ...], dtype: int):
        self.id = tensor_id
        self.shape = shape
        self.dtype = dtype


class FakeGraph:
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT4 = 3
    CQ1 = 3
    CQ2 = 4
    CQ3 = 5
    CQ4 = 6

    def __init__(self):
        self.next_id = 0
        self.add_clipped_input_dtypes: list[tuple[int, int]] = []

    def _tensor(self, shape: tuple[int, ...], dtype: int) -> FakeTensor:
        tensor = FakeTensor(self.next_id, shape, dtype)
        self.next_id += 1
        return tensor

    def input(self, shape: tuple[int, ...], dtype: int) -> FakeTensor:
        return self._tensor(tuple(shape), dtype)

    def set_input(self, tensor: FakeTensor, value, dtype: int | None = None) -> None:
        return None

    def precision_cast(self, tensor: FakeTensor, dtype: int) -> FakeTensor:
        return self._tensor(tensor.shape, dtype)

    def reshape(self, tensor: FakeTensor, shape: tuple[int, ...]) -> FakeTensor:
        return self._tensor(tuple(shape), tensor.dtype)

    def slice(self, tensor: FakeTensor, axis: int, start: int, length: int) -> FakeTensor:
        shape = list(tensor.shape)
        shape[int(axis)] = int(length)
        return self._tensor(tuple(shape), tensor.dtype)

    def add_clipped(self, lhs: FakeTensor, rhs: FakeTensor) -> FakeTensor:
        self.add_clipped_input_dtypes.append((lhs.dtype, rhs.dtype))
        return self._tensor(lhs.shape, lhs.dtype)

    def add(self, lhs: FakeTensor, rhs: FakeTensor) -> FakeTensor:
        return self._tensor(lhs.shape, lhs.dtype)

    def rms_norm(self, tensor: FakeTensor, weight: FakeTensor, eps: float) -> FakeTensor:
        return self._tensor(tensor.shape, tensor.dtype)

    def gelu(self, tensor: FakeTensor) -> FakeTensor:
        return self._tensor(tensor.shape, tensor.dtype)

    def topk(self, tensor: FakeTensor, k: int) -> FakeTensor:
        return self._tensor((2, tensor.shape[0], int(k)), self.FP32)

    def gather(self, tensor: FakeTensor, indices: FakeTensor) -> FakeTensor:
        return self._tensor((indices.shape[0], *tensor.shape[1:]), tensor.dtype)

    def permute(self, tensor: FakeTensor, permutation: tuple[int, ...]) -> FakeTensor:
        return self._tensor(tuple(tensor.shape[index] for index in permutation), tensor.dtype)


def _import_lower_with_fake_graph(monkeypatch):
    fake_runtime = types.ModuleType("cactus.transpile.runtime_compat")
    fake_runtime.Graph = FakeGraph
    fake_runtime.Tensor = FakeTensor
    monkeypatch.setitem(sys.modules, "cactus.transpile.runtime_compat", fake_runtime)
    sys.modules.pop("cactus.transpile.lower", None)
    return importlib.import_module("cactus.transpile.lower")


def test_bf16_add_clipped_lowers_to_supported_runtime_precision(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    ir = IRGraph(
        values={
            "lhs": IRValue(id="lhs", shape=(1, 4), dtype="bf16", users=["add"]),
            "rhs": IRValue(id="rhs", shape=(1, 4), dtype="bf16", users=["add"]),
            "out": IRValue(id="out", shape=(1, 4), dtype="bf16", producer="add", users=[]),
        },
        nodes={
            "add": IRNode(id="add", op="add_clipped", inputs=["lhs", "rhs"], outputs=["out"]),
        },
        order=["add"],
        inputs=["lhs", "rhs"],
        outputs=["out"],
    )

    transpiled = lower.transpile_preoptimized_ir(ir)

    assert transpiled.graph.add_clipped_input_dtypes == [(FakeGraph.FP16, FakeGraph.FP16)]
    sys.modules.pop("cactus.transpile.lower", None)


def test_bf16_rms_norm_lowers_to_supported_runtime_precision(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    ir = IRGraph(
        values={
            "x": IRValue(id="x", shape=(1, 4), dtype="bf16", users=["rms"]),
            "weight": IRValue(id="weight", shape=(4,), dtype="bf16", users=["rms"]),
            "out": IRValue(id="out", shape=(1, 4), dtype="bf16", producer="rms", users=[]),
        },
        nodes={
            "rms": IRNode(id="rms", op="rms_norm", inputs=["x", "weight"], outputs=["out"], attrs={"eps": 1e-6}),
        },
        order=["rms"],
        inputs=["x", "weight"],
        outputs=["out"],
    )

    transpiled = lower.transpile_preoptimized_ir(ir)

    assert transpiled.outputs[0].dtype == FakeGraph.FP16
    sys.modules.pop("cactus.transpile.lower", None)


def test_bf16_activation_lowers_to_supported_runtime_precision(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    ir = IRGraph(
        values={
            "x": IRValue(id="x", shape=(1, 4), dtype="bf16", users=["gelu"]),
            "out": IRValue(id="out", shape=(1, 4), dtype="bf16", producer="gelu", users=[]),
        },
        nodes={
            "gelu": IRNode(id="gelu", op="gelu", inputs=["x"], outputs=["out"]),
        },
        order=["gelu"],
        inputs=["x"],
        outputs=["out"],
    )

    transpiled = lower.transpile_preoptimized_ir(ir)

    assert transpiled.outputs[0].dtype == FakeGraph.FP16
    sys.modules.pop("cactus.transpile.lower", None)


def test_scalar_binary_lowering_handles_python_values(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)

    graph = FakeGraph()

    assert lower._lower_binary_op(graph, 3, 2, "add") == 5
    assert lower._lower_binary_op(graph, 3, 2, "subtract") == 1
    assert lower._lower_binary_op(graph, 3, 2, "multiply") == 6
    assert lower._lower_binary_op(graph, 3, 2, "divide") == 1.5
    sys.modules.pop("cactus.transpile.lower", None)


def test_mixed_tensor_binary_lowering_uses_supported_runtime_precision(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    graph = FakeGraph()
    lhs = graph.input((1, 4), FakeGraph.FP16)
    rhs = graph.input((1, 4), FakeGraph.FP32)

    out = lower._lower_binary_op(graph, lhs, rhs, "add")

    assert out.dtype == FakeGraph.FP16
    sys.modules.pop("cactus.transpile.lower", None)


def test_scalar_abs_lowering_handles_python_value(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    node = IRNode(id="abs", op="abs", inputs=["value"], outputs=["out"])
    ir = IRGraph(values={}, nodes={"abs": node}, order=["abs"], inputs=[], outputs=[])

    assert lower._lower_ir_node(FakeGraph(), node, {"value": -3.0}, ir) == [3.0]
    sys.modules.pop("cactus.transpile.lower", None)


def test_scalar_compare_lowering_handles_python_value(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    node = IRNode(
        id="less_equal",
        op="scalar_less_equal",
        inputs=["value"],
        outputs=["out"],
        attrs={"value": 3.0},
    )
    ir = IRGraph(values={}, nodes={"less_equal": node}, order=["less_equal"], inputs=[], outputs=[])

    assert lower._lower_ir_node(FakeGraph(), node, {"value": 3.0}, ir) == [1.0]
    assert lower._lower_ir_node(FakeGraph(), node, {"value": 4.0}, ir) == [0.0]
    sys.modules.pop("cactus.transpile.lower", None)


def test_flip_lowering_uses_reversed_indices(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    node = IRNode(
        id="flip",
        op="aten.flip.default",
        inputs=["value"],
        outputs=["out"],
        attrs={"args": [None, [1]]},
    )
    ir = IRGraph(values={}, nodes={"flip": node}, order=["flip"], inputs=[], outputs=[])
    tensor = FakeTensor(0, (2, 3, 4), FakeGraph.FP16)

    (out,) = lower._lower_ir_node(FakeGraph(), node, {"value": tensor}, ir)

    assert out.shape == (2, 3, 4)
    assert out.dtype == FakeGraph.FP16
    assert lower._lower_ir_node(FakeGraph(), node, {"value": 1.0}, ir) == [1.0]
    sys.modules.pop("cactus.transpile.lower", None)


def test_scalar_slice_lowering_preserves_python_value(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    node = IRNode(
        id="slice",
        op="slice",
        inputs=["value"],
        outputs=["out"],
        attrs={"axis": 0, "start": 0, "end": 1, "step": 1},
    )
    ir = IRGraph(values={}, nodes={"slice": node}, order=["slice"], inputs=[], outputs=[])

    assert lower._lower_ir_node(FakeGraph(), node, {"value": 1.0}, ir) == [1.0]
    sys.modules.pop("cactus.transpile.lower", None)


def test_topk_lowering_returns_values_then_indices(monkeypatch):
    lower = _import_lower_with_fake_graph(monkeypatch)
    node = IRNode(
        id="topk",
        op="aten.topk.default",
        inputs=["value"],
        outputs=["out"],
        attrs={"args": [None, 2, -1]},
    )
    ir = IRGraph(values={}, nodes={"topk": node}, order=["topk"], inputs=[], outputs=[])
    tensor = FakeTensor(0, (3, 4, 5), FakeGraph.FP16)

    values, indices = lower._lower_ir_node(FakeGraph(), node, {"value": tensor}, ir)[0]

    assert values.shape == (3, 4, 2)
    assert indices.shape == (3, 4, 2)
    assert values.dtype == FakeGraph.FP32
    assert indices.dtype == FakeGraph.FP32
    sys.modules.pop("cactus.transpile.lower", None)
