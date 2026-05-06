from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from src.transpile.capture.capture_pytorch import capture_model_with_fallback
from src.transpile.capture.capture_pytorch import dump_graph
from src.transpile.capture.graph_ir import verify_ir


MODELS = (
    "google/gemma-4-E2B",
    "google/gemma-3-270m-it",
    "google/gemma-2b-it",
    "google/gemma-2-2b",
    "Qwen/Qwen3.5-2B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)

PROMPT = "The capital of France is"
ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "python_capture_tests"


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "model"


def resolve_local_snapshot(model_id: str) -> str | None:
    explicit = Path(model_id)
    if explicit.exists():
        return str(explicit)

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id.replace("/", "--"))
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return str(snapshots[-1])


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if hasattr(value, "__dataclass_fields__"):
        return jsonify(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def graph_to_dict(graph) -> dict[str, Any]:
    return {
        "meta": jsonify(graph.meta),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "constants": {value_id: jsonify(constant) for value_id, constant in graph.constants.items()},
        "values": {value_id: jsonify(asdict(value)) for value_id, value in graph.values.items()},
        "nodes": [jsonify(asdict(graph.nodes[node_id])) for node_id in graph.order],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class CausalLMCaptureWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits
        raise TypeError(f"unsupported model output type: {type(outputs).__name__}")


def capture_one_model(model_id: str) -> dict[str, Any]:
    snapshot = resolve_local_snapshot(model_id)
    if snapshot is None:
        raise FileNotFoundError(f"no local snapshot found for {model_id}")

    artifact_dir = ARTIFACTS_DIR / slugify(model_id)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"]

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype="auto",
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()

    captured = capture_model_with_fallback(
        CausalLMCaptureWrapper(model),
        (input_ids,),
    )
    verify_ir(captured.ir_graph)

    write_json(
        artifact_dir / "graph_ir.json",
        {
            "model_id": model_id,
            "snapshot": snapshot,
            "prompt": PROMPT,
            "strict": captured.strict,
            "graph": graph_to_dict(captured.ir_graph),
        },
    )
    (artifact_dir / "fx_graph.txt").write_text(dump_graph(captured) + "\n")

    return {
        "model_id": model_id,
        "snapshot": snapshot,
        "artifact_dir": str(artifact_dir),
        "strict": captured.strict,
        "ir_nodes": len(captured.ir_graph.order),
        "ir_values": len(captured.ir_graph.values),
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    for model_id in MODELS:
        print(f"capturing {model_id}")
        try:
            result = capture_one_model(model_id)
        except Exception as exc:
            result = {
                "model_id": model_id,
                "error": str(exc),
            }
        results.append(result)
        print(json.dumps(result, indent=2))

    write_json(
        ARTIFACTS_DIR / "capture_manifest.json",
        {
            "prompt": PROMPT,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
