from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.canonicalize.cleanup import canonicalize_exported_graph
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.model_adapters import canonicalize_model_interface


class Gemma4FullModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = canonicalize_model_interface(model, task="causal_lm_logits").module

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


class Gemma4FirstBlockCheckpointWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, checkpoint_name: str):
        super().__init__()
        adapter = canonicalize_model_interface(model, task="causal_lm_logits")
        if adapter.family != "gemma4":
            raise ValueError(f"expected gemma4 adapter, got family={adapter.family}")
        if not hasattr(adapter.module, "debug_first_block"):
            raise ValueError("Gemma4 adapter does not expose debug_first_block()")
        self.adapter = adapter.module
        self.checkpoint_name = checkpoint_name

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        checkpoints = self.adapter.debug_first_block(input_ids)
        if self.checkpoint_name not in checkpoints:
            available = ", ".join(sorted(checkpoints.keys()))
            raise KeyError(f"unknown checkpoint {self.checkpoint_name!r}; available: {available}")
        return checkpoints[self.checkpoint_name]


def format_ir_graph(graph: IRGraph, *, max_nodes: int | None = None) -> str:
    lines: list[str] = []
    lines.append("IR Summary")
    lines.append(f"  inputs={graph.inputs}")
    lines.append(f"  outputs={graph.outputs}")
    lines.append(f"  nodes={len(graph.order)}")
    if graph.meta:
        lines.append(f"  meta={graph.meta}")
    lines.append("")
    lines.append("IR Nodes")

    node_ids = graph.order if max_nodes is None else graph.order[:max_nodes]
    for index, node_id in enumerate(node_ids):
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

    if max_nodes is not None and len(graph.order) > max_nodes:
        lines.append(f"... ({len(graph.order) - max_nodes} more nodes omitted)")

    if graph.constants:
        lines.append("")
        lines.append("IR Constants")
        constant_ids = sorted(graph.constants.keys())
        preview_ids = constant_ids if max_nodes is None else constant_ids[: max(8, min(len(constant_ids), max_nodes))]
        for value_id in preview_ids:
            value = graph.constants[value_id]
            value_meta = graph.values.get(value_id)
            shape = None if value_meta is None else value_meta.shape
            dtype = None if value_meta is None else value_meta.dtype
            lines.append(f"- {value_id}: type={type(value).__name__} shape={shape} dtype={dtype}")
        if len(preview_ids) < len(constant_ids):
            lines.append(f"... ({len(constant_ids) - len(preview_ids)} more constants omitted)")

    return "\n".join(lines)


def format_unsupported_ops(graph: IRGraph) -> str:
    counts = graph.meta.get("canonical_unsupported_op_counts", {})
    if not isinstance(counts, dict):
        counts = {}

    examples: dict[str, list[str]] = {}
    for node_id in graph.order:
        node = graph.nodes[node_id]
        print(f"node {node_id} op={node.op}")
        if node.op not in counts:
            continue
        bucket = examples.setdefault(node.op, [])
        if len(bucket) < 5:
            bucket.append(node.id)

    total = sum(int(count) for count in counts.values())
    lines = ["Unsupported Ops", f"  unique={len(counts)}", f"  total_nodes={total}"]
    if not counts:
        lines.append("  none")
        return "\n".join(lines)

    for op_name, count in counts.items():
        sample_nodes = ", ".join(examples.get(op_name, ()))
        lines.append(f"- {op_name}: {count}")
        if sample_nodes:
            lines.append(f"  examples={sample_nodes}")
    return "\n".join(lines)


def resolve_local_model_path(model_id: str) -> str:
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"could not find local snapshot for {model_id!r} under {snapshots_dir}"
        )
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"no snapshots found for {model_id!r} under {snapshots_dir}")
    return str(snapshots[-1])


def load_model(model_id: str) -> torch.nn.Module:
    token = os.environ.get("HF_TOKEN")
    common_kwargs: dict[str, object] = {
        "local_files_only": True,
    }
    if token:
        common_kwargs["token"] = token

    local_model_path = resolve_local_model_path(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        **common_kwargs,
    ).eval()
    return model


def build_module(
    model: torch.nn.Module,
    *,
    mode: str,
    checkpoint: str,
) -> torch.nn.Module:
    if mode == "full":
        return Gemma4FullModelWrapper(model).eval()
    if mode == "first_block":
        return Gemma4FirstBlockCheckpointWrapper(model, checkpoint).eval()
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Gemma4, run canonicalize/cleanup.py, and report unsupported ops.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-4-E2B"),
        help="Hugging Face model id. Uses local cache only.",
    )
    parser.add_argument(
        "--mode",
        choices=("first_block", "full"),
        default="first_block",
        help="Capture a first-block checkpoint or the full Gemma4 adapter graph.",
    )
    parser.add_argument(
        "--checkpoint",
        default="after_ffn_residual",
        help="First-block checkpoint name when --mode=first_block.",
    )
    parser.add_argument(
        "--input-ids",
        default="2,818,5279,529,7001,563",
        help="Comma-separated integer token ids to capture.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=120,
        help="Maximum IR nodes to print when --print-ir is enabled.",
    )
    parser.add_argument(
        "--print-ir",
        action="store_true",
        help="Also print the IR before and after canonical cleanup.",
    )
    args = parser.parse_args()

    model = load_model(args.model_id)
    module = build_module(model, mode=args.mode, checkpoint=args.checkpoint)
    token_ids = [int(token.strip()) for token in args.input_ids.split(",") if token.strip()]
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    print(f"Model: {args.model_id}")
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint if args.mode == 'first_block' else '(full model)'}")
    print(f"Input IDs shape: {tuple(input_ids.shape)}")
    print("")

    captured = capture_model(module, (input_ids,))

    if args.print_ir:
        print("=== IR Before Canonical Cleanup ===")
        print(format_ir_graph(captured.ir_graph, max_nodes=args.max_nodes))
        print("")

    canonicalize_exported_graph(captured.ir_graph)

    print("=== Unsupported Ops After Canonical Cleanup ===")
    print(format_unsupported_ops(captured.ir_graph))

    if args.print_ir:
        print("")
        print("=== IR After Canonical Cleanup ===")
        print(format_ir_graph(captured.ir_graph, max_nodes=args.max_nodes))


if __name__ == "__main__":
    main()
