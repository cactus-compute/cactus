from __future__ import annotations

from typing import Any

from . import models as M
from .fusion_builders import _graph, _single_node_graph, _variadic_input


def _capture(name: str, node: str, source_attr: str, *, default: Any = None, required: bool = False) -> M.AttrCapture:
    return M.AttrCapture(name, node, source_attr, default=default, required=required)


def _same_name_graphs(specs: dict[str, tuple[str, ...]]) -> dict[str, M.FusionGraph]:
    return {name: _single_node_graph(name, name, inputs) for name, inputs in specs.items()}


def _scalar_graph(name: str) -> M.FusionGraph:
    return _single_node_graph(name, name, ("x",), attr_captures=(_capture("value", name, "other"),))


PLAIN_DIRECT_SPECS = {
    "add": ("a", "b"),
    "subtract": ("a", "b"),
    "multiply": ("a", "b"),
    "divide": ("a", "b"),
    "abs": ("x",),
    "sqrt": ("x",),
    "scalar_exp": ("x",),
    "scalar_sqrt": ("x",),
    "scalar_rsqrt": ("x",),
    "scalar_cos": ("x",),
    "scalar_sin": ("x",),
    "scalar_log": ("x",),
    "where": ("condition", "x", "y"),
    "not_equal": ("a", "b"),
    "equal": ("a", "b"),
    "less": ("a", "b"),
    "less_equal": ("a", "b"),
    "greater": ("a", "b"),
    "greater_equal": ("a", "b"),
    "bitwise_and": ("a", "b"),
    "bitwise_or": ("a", "b"),
    "logical_not": ("x",),
    "bitwise_not": ("x",),
    "relu": ("x",),
    "silu": ("x",),
    "gelu": ("x",),
    "gelu_erf": ("x",),
    "sigmoid": ("x",),
    "tanh": ("x",),
}

SCALAR_OPS = (
    "scalar_add",
    "scalar_subtract",
    "scalar_multiply",
    "scalar_divide",
    "scalar_floor_divide",
    "scalar_not_equal",
    "scalar_equal",
    "scalar_less",
    "scalar_less_equal",
    "scalar_greater",
    "scalar_greater_equal",
)

DIRECT_GRAPHS: dict[str, M.FusionGraph] = {
    **_same_name_graphs(PLAIN_DIRECT_SPECS),
    **{name: _scalar_graph(name) for name in SCALAR_OPS},
    "pow": _single_node_graph("pow", "pow", ("x",), attr_captures=(_capture("exponent", "pow", "exponent"),)),
    **{
        name: _single_node_graph(name, name, ("x",), attr_captures=(_capture("axis", name, "dim", default=-1),))
        for name in ("mean", "sum", "variance", "min", "max", "cumsum", "softmax")
    },
    "topk": _single_node_graph("topk", "topk_direct", ("x",), attr_captures=(M.AttrCapture("k", "topk_direct", "k"),)),
    **{
        name: _single_node_graph(name, name, ("x",), attr_captures=(_capture("shape", name, "shape"),))
        for name in ("view", "reshape", "expand")
    },
    "flatten": _single_node_graph(
        "flatten",
        "flatten",
        ("x",),
        attr_captures=(
            _capture("start_dim", "flatten", "start_dim", default=0),
            _capture("end_dim", "flatten", "end_dim", default=-1),
        ),
    ),
    "transpose": _single_node_graph("transpose", "transpose", ("x",), attr_captures=(_capture("permutation", "transpose", "permutation"),)),
    "slice": _single_node_graph(
        "slice",
        "slice",
        ("x",),
        attr_captures=tuple(_capture(name, "slice", attr) for name, attr in (("axis", "dim"), ("start", "start"), ("end", "end"), ("step", "step"))),
    ),
    "unfold": _single_node_graph(
        "unfold",
        "unfold",
        ("x",),
        attr_captures=(
            _capture("dimension", "unfold", "dim"),
            _capture("size", "unfold", "size"),
            _capture("step", "unfold", "step", default=1),
        ),
    ),
    "index": _single_node_graph(
        "index",
        "index",
        ("x",),
        attr_captures=(
            _capture("index_value", "index", "index"),
            _capture("axis", "index", "dim", default=0),
        ),
    ),
    "cat": _graph("cat", "cat", ("cat",), inputs=(_variadic_input("values", "cat", 0, min_count=2),), attr_captures=(_capture("axis", "cat", "dim", default=0),)),
    "gather": _single_node_graph("gather", "gather", ("tensor", "indices"), attr_captures=(_capture("axis", "gather", "dim", default=0),)),
    "precision_cast": _single_node_graph("precision_cast", "precision_cast", ("x",), attr_captures=(_capture("dtype", "precision_cast", "dtype"),)),
    "pad": _single_node_graph(
        "pad",
        "pad",
        ("x",),
        attr_captures=(
            _capture("pad", "pad", "pad"),
            _capture("value", "pad", "value", default=0.0),
        ),
    ),
    "clamp": _single_node_graph(
        "clamp",
        "clamp",
        ("x",),
        attr_captures=(
            _capture("lo", "clamp", "min"),
            _capture("hi", "clamp", "max"),
        ),
    ),
    "groupnorm": _single_node_graph(
        "groupnorm",
        "groupnorm",
        ("x", "weight", "bias"),
        attr_captures=(
            _capture("num_groups", "groupnorm", "num_groups"),
            _capture("epsilon", "groupnorm", "eps", default=1e-5),
        ),
    ),
    "batchnorm": _single_node_graph(
        "batchnorm",
        "batchnorm",
        ("x", "weight", "bias", "running_mean", "running_var"),
        attr_captures=(
            _capture("axis", "batchnorm", "axis", default=1),
            _capture("epsilon", "batchnorm", "eps", default=1e-5),
        ),
    ),
    "embedding_from_tensor": _single_node_graph("embedding_from_tensor", "embedding", ("embedding_tensor", "indices")),
}
