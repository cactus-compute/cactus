from __future__ import annotations

from types import SimpleNamespace

import torch

from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.importers import ImportContext
from cactus.transpile.importers import import_get_attr


def test_import_get_attr_preserves_meta_tensor_constant() -> None:
    ir = IRGraph(values={}, nodes={}, order=[], inputs=[], outputs=[])
    ctx = ImportContext()
    node = SimpleNamespace(name="p_weight", op="get_attr")
    value = torch.empty((3, 4), device="meta", dtype=torch.float16)

    import_get_attr(
        ir,
        node,
        ctx,
        value,
        shape=(3, 4),
        dtype="float16",
        source_name="linear.weight",
    )

    constant = ir.constants["v_p_weight"]
    assert isinstance(constant, torch.Tensor)
    assert constant.is_meta
    assert tuple(constant.shape) == (3, 4)
    assert ir.values["v_p_weight"].meta["source_name"] == "linear.weight"
