from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).parent.parent))

if torch is not None and AutoModelForCausalLM is not None:
    from src.transpile.capture_pytorch import capture_model
    from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
    from src.transpile.model_adapters import canonicalize_model_interface
    from src.transpile.optimize_graph import optimize_graph
    from src.transpile.lower import transpile_preoptimized_ir


_RUN_QWEN35_LAYER_DIFF = os.environ.get("CACTUS_RUN_LOCAL_QWEN35_LAYER_DIFF_TEST") == "1"
_DEFAULT_MODEL_ID = os.environ.get("CACTUS_TEST_QWEN35_MODEL_ID", "Qwen/Qwen3.5-2B")
_DEFAULT_INPUT_IDS = os.environ.get("CACTUS_TEST_QWEN35_INPUT_IDS", "151644,8948,198,9707")
_DEFAULT_WEIGHTS_DIR = os.environ.get(
    "CACTUS_TEST_QWEN35_WEIGHTS_DIR",
    str((Path(__file__).resolve().parents[2] / "weights" / "qwen3.5-2b").resolve()),
)


def _resolve_local_model_path(model_id: str) -> str:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id.replace("/", "--"))
    )
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"could not find local snapshot for {model_id!r} under {snapshots_dir}")
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"no snapshots found for {model_id!r} under {snapshots_dir}")
    return str(snapshots[-1])


def _parse_input_ids(raw: str) -> list[int]:
    token_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not token_ids:
        raise ValueError("CACTUS_TEST_QWEN35_INPUT_IDS produced no token ids")
    return token_ids


def _summarize_ops(ir) -> str:
    counts: dict[str, int] = {}
    for node_id in ir.order:
        op_name = str(ir.nodes[node_id].op)
        counts[op_name] = counts.get(op_name, 0) + 1
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    preview = items[:8]
    return ",".join(f"{name}:{count}" for name, count in preview)


@dataclass
class DiffStats:
    label: str
    shape: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    argmax_ref: int | None
    argmax_transpiled: int | None
    raw_ir_nodes: int
    optimized_ir_nodes: int
    top_ops: str


if torch is not None and AutoModelForCausalLM is not None:
    class _WeightsBoundTranspileWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module, *, weights_dir: str | None):
            super().__init__()
            self.module = module
            self.weights_dir = weights_dir

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.module(*args)

        def get_transpile_metadata(self) -> dict[str, object]:
            metadata: dict[str, object] = {}
            provider = getattr(self.module, "get_transpile_metadata", None)
            if callable(provider):
                provided = provider()
                if isinstance(provided, dict):
                    metadata.update(provided)
            graph_meta: dict[str, object] = {}
            base_graph = metadata.get("graph", {})
            if isinstance(base_graph, dict):
                graph_meta.update(base_graph)
            if self.weights_dir:
                graph_meta["weights_dir"] = self.weights_dir
            metadata["graph"] = graph_meta
            return metadata


    class Qwen35CheckpointWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module, checkpoint_index: int, *, weights_dir: str | None = None):
            super().__init__()
            adapter = canonicalize_model_interface(model, task="causal_lm_logits", weights_dir=weights_dir)
            if adapter.family != "qwen3_5":
                raise ValueError(f"expected qwen3_5 adapter, got family={adapter.family}")
            self.adapter = adapter.module
            self.checkpoint_index = int(checkpoint_index)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            _, checkpoints = self.adapter.debug_forward(input_ids)
            return checkpoints[self.checkpoint_index]


    class Qwen35LogitsWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
            super().__init__()
            adapter = canonicalize_model_interface(model, task="causal_lm_logits", weights_dir=weights_dir)
            if adapter.family != "qwen3_5":
                raise ValueError(f"expected qwen3_5 adapter, got family={adapter.family}")
            self.adapter = adapter.module

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            logits, _ = self.adapter.debug_forward(input_ids)
            return logits


class TestTranspileQwen35LayerDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _RUN_QWEN35_LAYER_DIFF:
            raise unittest.SkipTest("set CACTUS_RUN_LOCAL_QWEN35_LAYER_DIFF_TEST=1 to run")
        if torch is None:
            raise unittest.SkipTest("torch is not available")
        if AutoModelForCausalLM is None:
            raise unittest.SkipTest("transformers is not available")

        local_model_path = _resolve_local_model_path(_DEFAULT_MODEL_ID)
        cls.model_id = _DEFAULT_MODEL_ID
        cls.model_path = local_model_path
        cls.input_ids_list = _parse_input_ids(_DEFAULT_INPUT_IDS)
        cls.input_ids = torch.tensor([cls.input_ids_list], dtype=torch.long)
        cls.weights_dir = None
        candidate_weights_dir = Path(_DEFAULT_WEIGHTS_DIR).expanduser().resolve()
        if candidate_weights_dir.exists():
            cls.weights_dir = str(candidate_weights_dir)

        common_kwargs: dict[str, object] = {
            "local_files_only": True,
            "torch_dtype": torch.float16,
            "device_map": None,
            "low_cpu_mem_usage": True,
        }
        token = os.environ.get("HF_TOKEN")
        if token:
            common_kwargs["token"] = token

        cls.model = AutoModelForCausalLM.from_pretrained(local_model_path, **common_kwargs).eval()

        adapter = canonicalize_model_interface(cls.model, task="causal_lm_logits", weights_dir=cls.weights_dir)
        if adapter.family != "qwen3_5":
            raise unittest.SkipTest(f"expected qwen3_5 adapter, got {adapter.family}")
        cls.num_layers = int(adapter.module.backbone.config.num_hidden_layers)

        print("")
        print(f"[qwen35-layer-diff] model_id={cls.model_id}")
        print(f"[qwen35-layer-diff] model_path={cls.model_path}")
        print(f"[qwen35-layer-diff] input_ids={cls.input_ids_list}")
        print(f"[qwen35-layer-diff] num_layers={cls.num_layers}")
        print(f"[qwen35-layer-diff] weights_dir={cls.weights_dir}")

    @classmethod
    def tearDownClass(cls) -> None:
        model = getattr(cls, "model", None)
        if model is not None:
            del cls.model

    def _run_wrapper_diff(self, label: str, wrapper: torch.nn.Module) -> DiffStats:
        wrapper = wrapper.eval()
        with torch.no_grad():
            reference = wrapper(self.input_ids).detach().float().cpu().numpy()

        transpile_wrapper = _WeightsBoundTranspileWrapper(wrapper, weights_dir=self.weights_dir).eval()
        captured = capture_model(transpile_wrapper, (self.input_ids,))
        raw_ir_nodes = len(captured.ir_graph.order)
        canonicalize_exported_graph(captured.ir_graph)
        optimize_graph(captured.ir_graph)
        optimized_ir_nodes = len(captured.ir_graph.order)
        top_ops = _summarize_ops(captured.ir_graph)
        transpiled = transpile_preoptimized_ir(captured.ir_graph)
        transpiled.set_inputs([self.input_ids.cpu().numpy()])
        actual = transpiled.execute()[0].numpy().astype(np.float32)

        diff = np.abs(reference.astype(np.float32) - actual)
        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))

        argmax_ref: int | None = None
        argmax_transpiled: int | None = None
        if reference.ndim == 3:
            argmax_ref = int(np.argmax(reference[0, -1]))
            argmax_transpiled = int(np.argmax(actual[0, -1]))

        print(
            "[qwen35-layer-diff] "
            f"{label} "
            f"shape={tuple(reference.shape)} "
            f"raw_ir_nodes={raw_ir_nodes} "
            f"optimized_ir_nodes={optimized_ir_nodes} "
            f"max_abs_diff={max_abs_diff:.6f} "
            f"mean_abs_diff={mean_abs_diff:.6f} "
            f"top_ops={top_ops}"
        )
        if argmax_ref is not None and argmax_transpiled is not None:
            print(
                "[qwen35-layer-diff] "
                f"{label} "
                f"argmax_ref={argmax_ref} "
                f"argmax_transpiled={argmax_transpiled}"
            )

        return DiffStats(
            label=label,
            shape=tuple(reference.shape),
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            argmax_ref=argmax_ref,
            argmax_transpiled=argmax_transpiled,
            raw_ir_nodes=raw_ir_nodes,
            optimized_ir_nodes=optimized_ir_nodes,
            top_ops=top_ops,
        )

    def test_qwen35_layer_by_layer_debug_diff(self) -> None:
        layer_stats: list[DiffStats] = []
        for layer_index in range(self.num_layers):
            wrapper = Qwen35CheckpointWrapper(self.model, checkpoint_index=layer_index, weights_dir=self.weights_dir)
            layer_stats.append(self._run_wrapper_diff(f"layer_{layer_index}", wrapper))

        norm_wrapper = Qwen35CheckpointWrapper(self.model, checkpoint_index=self.num_layers, weights_dir=self.weights_dir)
        norm_stats = self._run_wrapper_diff("final_norm", norm_wrapper)
        logits_stats = self._run_wrapper_diff("logits", Qwen35LogitsWrapper(self.model, weights_dir=self.weights_dir))

        worst_layer = max(layer_stats, key=lambda item: item.max_abs_diff)
        print(
            "[qwen35-layer-diff] "
            f"worst_layer={worst_layer.label} "
            f"worst_layer_max_abs_diff={worst_layer.max_abs_diff:.6f} "
            f"worst_layer_mean_abs_diff={worst_layer.mean_abs_diff:.6f}"
        )
        print(
            "[qwen35-layer-diff] "
            f"final_norm_max_abs_diff={norm_stats.max_abs_diff:.6f} "
            f"logits_max_abs_diff={logits_stats.max_abs_diff:.6f}"
        )

        self.assertEqual(len(layer_stats), self.num_layers)
        self.assertTrue(np.isfinite(logits_stats.max_abs_diff))
        self.assertTrue(np.isfinite(logits_stats.mean_abs_diff))
