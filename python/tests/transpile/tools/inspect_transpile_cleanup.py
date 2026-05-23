from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.capture_pytorch import dump_graph
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.graph_ir import IRGraph


class Toy(nn.Module):
    def forward(self, x, y):
        z = x + y
        z = z * 0.5
        z = torch.nn.functional.gelu(z)
        z = torch.softmax(z, dim=-1)
        return z


class AttentionBlockToy(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float16)

    def forward(self, q, k, v, gate):
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        attn = attn.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        attn = attn * torch.sigmoid(gate)
        attn = attn.to(self.out_proj.weight.dtype)
        return self.out_proj(attn)


def build_case(name: str) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    if name == "toy":
        model = Toy().eval()
        args = (
            torch.randn(2, 4, dtype=torch.float16),
            torch.randn(2, 4, dtype=torch.float16),
        )
        return model, args

    if name == "attention_block":
        model = AttentionBlockToy().eval()
        args = (
            torch.randn(1, 2, 3, 4, dtype=torch.float16),
            torch.randn(1, 2, 3, 4, dtype=torch.float16),
            torch.randn(1, 2, 3, 4, dtype=torch.float16),
            torch.randn(1, 3, 8, dtype=torch.float16),
        )
        return model, args

    raise ValueError(f"unknown case: {name}")


def format_ir_graph(graph: IRGraph) -> str:
    lines: list[str] = []
    lines.append("IR Summary")
    lines.append(f"  inputs={graph.inputs}")
    lines.append(f"  outputs={graph.outputs}")
    lines.append(f"  nodes={len(graph.order)}")
    if graph.meta:
        lines.append(f"  meta={graph.meta}")
    lines.append("")
    lines.append("IR Nodes")

    for index, node_id in enumerate(graph.order):
        node = graph.nodes[node_id]
        lines.append(f"[{index}] {node.id} op={node.op}")
        lines.append(f"  inputs={node.inputs}")
        lines.append(f"  outputs={node.outputs}")
        if node.attrs:
            lines.append(f"  attrs={node.attrs}")
        if node.kind != "generic":
            lines.append(f"  kind={node.kind}")
        if node.meta:
            lines.append(f"  meta={node.meta}")
        for output_id in node.outputs:
            value = graph.values.get(output_id)
            if value is None:
                continue
            lines.append(
                f"  value[{output_id}] shape={value.shape} dtype={value.dtype} users={value.users}"
            )

    if graph.constants:
        lines.append("")
        lines.append("IR Constants")
        for value_id, value in graph.constants.items():
            value_meta = graph.values.get(value_id)
            shape = None if value_meta is None else value_meta.shape
            dtype = None if value_meta is None else value_meta.dtype
            typename = type(value).__name__
            lines.append(f"- {value_id}: type={typename} shape={shape} dtype={dtype}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a small PyTorch model, dump the exported graph, and print IR before/after cleanup.",
    )
    parser.add_argument(
        "--case",
        choices=("toy", "attention_block"),
        default="toy",
        help="Which built-in model to inspect.",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Hide exported graph metadata to keep the dump shorter.",
    )
    args = parser.parse_args()

    model, example_args = build_case(args.case)
    captured = capture_model(model, example_args)

    print(f"Case: {args.case}")
    print(f"Model: {type(model).__name__}")
    print("")
    print("=== Exported Graph ===")
    print(dump_graph(captured, include_meta=not args.no_meta))
    print("")
    print("=== IR Before Cleanup ===")
    print(format_ir_graph(captured.ir_graph))
    print("")

    canonicalize_exported_graph(captured.ir_graph)

    print("=== IR After Cleanup ===")
    print(format_ir_graph(captured.ir_graph))


if __name__ == "__main__":
    main()
