from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cactus.transpile.capture_pytorch import (
    capture_model_with_fallback,
    dump_graph,
    get_dtype,
    get_shape,
)
from cactus.transpile.lower import transpile_captured


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)
        y = self.fc2(h)
        return y


def main():
    model = TinyBlock().eval()
    x = torch.randn(2, 8)

    captured = capture_model_with_fallback(model, args=(x,))

    print("strict:", captured.strict)
    print("graph_module type:", type(captured.graph_module).__name__)
    print("state_dict keys:", list(captured.state_dict.keys()))
    print()

    print("=== FX Graph ===")
    print(captured.graph)
    print()

    print("=== Node Walk ===")
    for i, node in enumerate(captured.graph.nodes):
        print(f"[{i}] op={node.op} name={node.name} target={node.target}")
        print(f"    args={node.args}")
        print(f"    kwargs={node.kwargs}")
        print(f"    shape={get_shape(node)} dtype={get_dtype(node)}")
        print()

    print("=== Full Dump ===")
    print(dump_graph(captured))
    print()

    print("=== Lower To Cactus ===")
    try:
        transpiled = transpile_captured(captured)
        print("lowering succeeded")
        print("lowered graph type:", type(transpiled.graph).__name__)
        print("runtime inputs:", transpiled.runtime_inputs)
        print("bound constants:", transpiled.bound_constants)
        print("outputs:", transpiled.outputs)
        print()
        print("binding runtime input 0 from example tensor")
        transpiled.set_input(0, x)
    except NotImplementedError as exc:
        print("lowering stopped on unimplemented case:")
        print(f"  {exc}")


if __name__ == "__main__":
    main()
