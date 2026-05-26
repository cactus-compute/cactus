from __future__ import annotations

from collections.abc import Callable
from collections import Counter
import importlib.util
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cactus.transpile.capture_pytorch import capture_model
from cactus.transpile.capture_jax import capture_jax_function
from cactus.transpile.capture_jax import capture_jax_function_with_params
from cactus.transpile.capture_jax import capture_jax_generation_graphs
from cactus.transpile.capture_jax import capture_jax_graphs
from cactus.transpile.capture_jax import capture_jax_sequence_model
from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.capture_jax import _flatten_named_leaves
from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.transpile.lower import transpile_ir


def _time_ms(fn: Callable[[], None], *, warmup: int = 10, iterations: int = 100) -> float:
    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(iterations):
        fn()
    return (perf_counter() - start) * 1000.0 / iterations


def _print_timings(
    label: str,
    *,
    fn: Callable[[object], object],
    x: object,
    x_np: np.ndarray,
    graph: object,
    warmup: int = 10,
    iterations: int = 100,
) -> None:
    jit_fn = jax.jit(fn)
    jit_fn(x).block_until_ready()

    jax_eager_ms = _time_ms(lambda: fn(x).block_until_ready(), warmup=warmup, iterations=iterations)
    jax_jit_ms = _time_ms(lambda: jit_fn(x).block_until_ready(), warmup=warmup, iterations=iterations)
    cactus_execute_ms = _time_ms(lambda: graph.execute(), warmup=warmup, iterations=iterations)
    cactus_set_input_execute_ms = _time_ms(
        lambda: (graph.set_inputs([x_np]), graph.execute()),
        warmup=warmup,
        iterations=iterations,
    )

    print(
        f"\n{label}:\n"
        f"  jax eager:                 {jax_eager_ms:.4f} ms\n"
        f"  jax jit execute-only:      {jax_jit_ms:.4f} ms\n"
        f"  cactus execute-only:       {cactus_execute_ms:.4f} ms\n"
        f"  cactus set_input+execute:  {cactus_set_input_execute_ms:.4f} ms"
    )


def _print_torch_timings(
    label: str,
    *,
    module: torch.nn.Module,
    x: torch.Tensor,
    x_np: np.ndarray,
    graph: object,
    warmup: int = 10,
    iterations: int = 100,
) -> None:
    with torch.no_grad():
        _time_ms(lambda: module(x), warmup=warmup, iterations=warmup)
        torch_eager_ms = _time_ms(lambda: module(x), warmup=warmup, iterations=iterations)
    cactus_execute_ms = _time_ms(lambda: graph.execute(), warmup=warmup, iterations=iterations)
    cactus_set_input_execute_ms = _time_ms(
        lambda: (graph.set_inputs([x_np]), graph.execute()),
        warmup=warmup,
        iterations=iterations,
    )

    print(
        f"\n{label}:\n"
        f"  pytorch eager:             {torch_eager_ms:.4f} ms\n"
        f"  cactus execute-only:       {cactus_execute_ms:.4f} ms\n"
        f"  cactus set_input+execute:  {cactus_set_input_execute_ms:.4f} ms"
    )


def _diff_stats(got: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.abs(got.astype(np.float32) - expected.astype(np.float32))
    return float(np.max(diff)), float(np.mean(diff))


def _print_diff(label: str, got: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    max_diff, mean_diff = _diff_stats(got, expected)
    print(f"{label}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    return max_diff, mean_diff


def _op_counts(ir) -> Counter[str]:
    return Counter(ir.nodes[node_id].op for node_id in ir.order)


def test_jax_layer_norm_outlier_matches_jax() -> None:
    width = 768
    x = np.zeros((1, 6, width), dtype=np.float16)
    x[:, :, :] = np.linspace(-3.0, 3.0, width, dtype=np.float16)
    x[0, 0, 447] = np.float16(1615.0)
    params = {
        "weight": jnp.linspace(0.75, 1.25, width, dtype=jnp.float16),
        "bias": jnp.linspace(-0.1, 0.1, width, dtype=jnp.float16),
    }

    def layer_norm(p, values):
        values_fp32 = values.astype(jnp.float32)
        mean = jnp.mean(values_fp32, axis=-1, keepdims=True)
        variance = jnp.mean((values_fp32 - mean) * (values_fp32 - mean), axis=-1, keepdims=True)
        return ((values_fp32 - mean) * jax.lax.rsqrt(variance + 1.0e-5) * p["weight"] + p["bias"]).astype(jnp.float16)

    expected = np.asarray(layer_norm(params, jnp.asarray(x)))
    bundle = capture_jax_graphs(
        params,
        [
            JaxGraphSpec(
                name="layer_norm",
                fn=layer_norm,
                example_args=(jnp.asarray(x),),
                input_names=("x",),
                output_names=("y",),
            )
        ],
        weights_dir=None,
    )
    graph = bundle.graphs["layer_norm"]
    graph.graph.set_inputs([x])
    actual = graph.graph.execute()[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_capture_jax_tiny_mlp_matches_jax() -> None:
    key_w1, key_w2, key_x = jax.random.split(jax.random.PRNGKey(0), 3)
    w1 = jax.random.normal(key_w1, (8, 16), dtype=jnp.float16) * jnp.float16(0.1)
    b1 = jnp.zeros((16,), dtype=jnp.float16)
    w2 = jax.random.normal(key_w2, (16, 4), dtype=jnp.float16) * jnp.float16(0.1)
    b2 = jnp.zeros((4,), dtype=jnp.float16)

    def fn(x):
        h = jnp.dot(x, w1) + b1
        h = h * jax.nn.sigmoid(h)
        return jnp.dot(h, w2) + b2

    x = jax.random.normal(key_x, (2, 8), dtype=jnp.float16)
    x_np = np.asarray(x)
    ir = capture_jax_function(fn, (x,), constant_names=("w1", "b1", "w2", "b2"))

    graph = transpile_ir(ir)
    graph.set_inputs([x_np])
    got = graph.execute()[0].numpy().astype(np.float32)
    expected = np.asarray(fn(x)).astype(np.float32)

    assert ir.meta["frontend"] == "jax"
    assert ir.meta["adapter_family"] == "generic"
    assert np.max(np.abs(got - expected)) < 5e-2

    _print_timings(
        "JAX generic tiny MLP timings, embedded constants (batch=2, fp16)",
        fn=fn,
        x=x,
        x_np=x_np,
        graph=graph,
    )


def _write_fp16_weight(weights_dir: Path, name: str, value: object) -> None:
    save_tensor_with_header(np.asarray(value), weights_dir / f"{name}.weights", precision="FP16")


def _write_weights_manifest(weights_dir: Path, params: dict[str, object]) -> None:
    for name, value in params.items():
        _write_fp16_weight(weights_dir, name, value)
    (weights_dir / "weights_manifest.json").write_text(
        json.dumps(
            {
                name: {
                    "filename": f"{name}.weights",
                    "kind": "weight",
                }
                for name in params
            }
        )
        + "\n"
    )


def _flatten_jax_params(params: object) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in _flatten_named_leaves(jax.tree_util, params)}


def test_capture_jax_params_resolve_mmap_weights(tmp_path: Path) -> None:
    key_w1, key_b1, key_w2, key_b2, key_x = jax.random.split(jax.random.PRNGKey(1), 5)
    params = {
        "w1": jax.random.normal(key_w1, (8, 16), dtype=jnp.float16) * jnp.float16(0.1),
        "b1": jax.random.normal(key_b1, (16,), dtype=jnp.float16) * jnp.float16(0.01),
        "w2": jax.random.normal(key_w2, (16, 4), dtype=jnp.float16) * jnp.float16(0.1),
        "b2": jax.random.normal(key_b2, (4,), dtype=jnp.float16) * jnp.float16(0.01),
    }
    _write_weights_manifest(tmp_path, params)

    def fn(model_params, x):
        h = jnp.dot(x, model_params["w1"]) + model_params["b1"]
        h = h * jax.nn.sigmoid(h)
        return jnp.dot(h, model_params["w2"]) + model_params["b2"]

    x = jax.random.normal(key_x, (2, 8), dtype=jnp.float16)
    x_np = np.asarray(x)
    ir = capture_jax_function_with_params(fn, params, (x,), weights_dir=str(tmp_path))

    graph = transpile_ir(ir)
    graph.set_inputs([x_np])
    got = graph.execute()[0].numpy().astype(np.float32)
    expected = np.asarray(fn(params, x)).astype(np.float32)

    assert np.max(np.abs(got - expected)) < 5e-2
    assert len(graph.bound_constant_bindings) == len(params)
    assert {binding["source_name"] for binding in graph.bound_constant_bindings} == set(params)

    _print_timings(
        "JAX generic tiny MLP timings, mmap params (batch=2, fp16)",
        fn=lambda value: fn(params, value),
        x=x,
        x_np=x_np,
        graph=graph,
    )


def test_capture_jax_broadcast_in_dim_preserves_non_trailing_dimensions() -> None:
    x = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float16)

    def fn(value):
        return jnp.broadcast_to(value[:, None, None, :], (1, 2, 4, 3))

    ir = capture_jax_function(fn, (x,))
    graph = transpile_ir(ir)
    graph.set_inputs([np.asarray(x)])
    got = graph.execute()[0].numpy().astype(np.float32)
    expected = np.asarray(fn(x)).astype(np.float32)

    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_capture_jax_generation_graphs_tags_decoder_step_cache() -> None:
    params = {"scale": jnp.asarray(2.0, dtype=jnp.float16)}
    token = jnp.asarray([[3]], dtype=jnp.float16)
    position = jnp.asarray([[4]], dtype=jnp.float16)

    def step_fn(model_params, token_arg, position_arg):
        return token_arg * model_params["scale"] + position_arg

    bundle = capture_jax_generation_graphs(
        params,
        decoder_step=JaxGraphSpec(
            name="decoder_step",
            fn=step_fn,
            example_args=(token, position),
            input_names=("token", "position"),
            output_names=("logits",),
        ),
        max_cache_seq_len=16,
        cache_sink_size=0,
        enable_attention_fusion=False,
    )
    captured = bundle.graphs["decoder_step"]
    out = captured.execute(token, position)[0].numpy()

    assert np.allclose(out, np.asarray([[10.0]], dtype=np.float16))
    assert captured.ir_graph.meta["jax_graph_role"] == "decoder_step"
    assert captured.ir_graph.meta["component"] == "decoder_step"
    assert captured.ir_graph.meta["use_internal_kv_cache"] is True
    assert captured.ir_graph.meta["max_cache_seq_len"] == 16


def test_capture_jax_complex_mlp_mmap_weights_timings(tmp_path: Path) -> None:
    keys = jax.random.split(jax.random.PRNGKey(2), 10)
    dims = (128, 256, 256, 128, 64)
    params = {
        "w1": jax.random.normal(keys[0], (dims[0], dims[1]), dtype=jnp.float16) * jnp.float16(0.04),
        "b1": jax.random.normal(keys[1], (dims[1],), dtype=jnp.float16) * jnp.float16(0.01),
        "w2": jax.random.normal(keys[2], (dims[1], dims[2]), dtype=jnp.float16) * jnp.float16(0.04),
        "b2": jax.random.normal(keys[3], (dims[2],), dtype=jnp.float16) * jnp.float16(0.01),
        "w3": jax.random.normal(keys[4], (dims[2], dims[3]), dtype=jnp.float16) * jnp.float16(0.04),
        "b3": jax.random.normal(keys[5], (dims[3],), dtype=jnp.float16) * jnp.float16(0.01),
        "w4": jax.random.normal(keys[6], (dims[3], dims[4]), dtype=jnp.float16) * jnp.float16(0.04),
        "b4": jax.random.normal(keys[7], (dims[4],), dtype=jnp.float16) * jnp.float16(0.01),
    }
    _write_weights_manifest(tmp_path, params)

    def fn(model_params, x):
        h = jnp.dot(x, model_params["w1"]) + model_params["b1"]
        h = h * jax.nn.sigmoid(h)
        h = jnp.dot(h, model_params["w2"]) + model_params["b2"]
        h = h * jax.nn.sigmoid(h)
        h = jnp.dot(h, model_params["w3"]) + model_params["b3"]
        h = h * jax.nn.sigmoid(h)
        return jnp.dot(h, model_params["w4"]) + model_params["b4"]

    x = jax.random.normal(keys[8], (16, dims[0]), dtype=jnp.float16)
    x_np = np.asarray(x)
    ir = capture_jax_function_with_params(fn, params, (x,), weights_dir=str(tmp_path))

    graph = transpile_ir(ir)
    graph.set_inputs([x_np])
    got = graph.execute()[0].numpy().astype(np.float32)
    expected = np.asarray(fn(params, x)).astype(np.float32)

    assert np.max(np.abs(got - expected)) < 1e-1
    assert len(graph.bound_constant_bindings) == len(params)
    assert {binding["source_name"] for binding in graph.bound_constant_bindings} == set(params)

    _print_timings(
        "JAX generic larger MLP timings, mmap params (batch=16, fp16)",
        fn=lambda value: fn(params, value),
        x=x,
        x_np=x_np,
        graph=graph,
        warmup=5,
        iterations=30,
    )


class TorchComplexMLP(torch.nn.Module):
    def __init__(self, params: dict[str, object]) -> None:
        super().__init__()
        for name, value in params.items():
            tensor = torch.from_numpy(np.array(value)).to(torch.float16)
            self.register_parameter(name, torch.nn.Parameter(tensor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(x, self.w1) + self.b1
        h = h * torch.sigmoid(h)
        h = torch.matmul(h, self.w2) + self.b2
        h = h * torch.sigmoid(h)
        h = torch.matmul(h, self.w3) + self.b3
        h = h * torch.sigmoid(h)
        return torch.matmul(h, self.w4) + self.b4


def test_capture_pytorch_complex_mlp_timings_against_same_shape() -> None:
    rng = np.random.default_rng(2)
    dims = (128, 256, 256, 128, 64)
    params = {
        "w1": (rng.standard_normal((dims[0], dims[1])) * 0.04).astype(np.float16),
        "b1": (rng.standard_normal((dims[1],)) * 0.01).astype(np.float16),
        "w2": (rng.standard_normal((dims[1], dims[2])) * 0.04).astype(np.float16),
        "b2": (rng.standard_normal((dims[2],)) * 0.01).astype(np.float16),
        "w3": (rng.standard_normal((dims[2], dims[3])) * 0.04).astype(np.float16),
        "b3": (rng.standard_normal((dims[3],)) * 0.01).astype(np.float16),
        "w4": (rng.standard_normal((dims[3], dims[4])) * 0.04).astype(np.float16),
        "b4": (rng.standard_normal((dims[4],)) * 0.01).astype(np.float16),
    }
    module = TorchComplexMLP(params).eval()
    x_np = rng.standard_normal((16, dims[0])).astype(np.float16)
    x = torch.from_numpy(x_np)

    captured = capture_model(module, (x,))
    graph = transpile_ir(captured.ir_graph)
    graph.set_inputs([x_np])

    with torch.no_grad():
        expected = module(x).numpy().astype(np.float32)
    got = graph.execute()[0].numpy().astype(np.float32)

    assert np.max(np.abs(got - expected)) < 1e-1

    _print_torch_timings(
        "PyTorch generic larger MLP timings, embedded params (batch=16, fp16)",
        module=module,
        x=x,
        x_np=x_np,
        graph=graph,
        warmup=5,
        iterations=30,
    )


def _gemma_like_params(*, blocks: int, dim: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for block in range(blocks):
        prefix = f"b{block}"
        params[f"{prefix}.norm1"] = (1.0 + rng.standard_normal((dim,)) * 0.01).astype(np.float16)
        params[f"{prefix}.q"] = (rng.standard_normal((dim, dim)) * 0.035).astype(np.float16)
        params[f"{prefix}.k"] = (rng.standard_normal((dim, dim)) * 0.035).astype(np.float16)
        params[f"{prefix}.v"] = (rng.standard_normal((dim, dim)) * 0.035).astype(np.float16)
        params[f"{prefix}.o"] = (rng.standard_normal((dim, dim)) * 0.035).astype(np.float16)
        params[f"{prefix}.norm2"] = (1.0 + rng.standard_normal((dim,)) * 0.01).astype(np.float16)
        params[f"{prefix}.gate"] = (rng.standard_normal((dim, hidden)) * 0.03).astype(np.float16)
        params[f"{prefix}.up"] = (rng.standard_normal((dim, hidden)) * 0.03).astype(np.float16)
        params[f"{prefix}.down"] = (rng.standard_normal((hidden, dim)) * 0.03).astype(np.float16)
    return params


def _jax_rms_norm(x, weight, eps=jnp.float16(1e-3)):
    variance = jnp.sum(x * x, axis=-1, keepdims=True) / jnp.float16(x.shape[-1])
    return (x / jnp.sqrt(variance + eps)) * weight


def _jax_gemma_like(params, x, *, blocks: int):
    for block in range(blocks):
        prefix = f"b{block}"
        h = _jax_rms_norm(x, params[f"{prefix}.norm1"])
        q = jnp.dot(h, params[f"{prefix}.q"])
        k = jnp.dot(h, params[f"{prefix}.k"])
        v = jnp.dot(h, params[f"{prefix}.v"])
        x = x + jnp.dot((q * jax.nn.sigmoid(k)) * v, params[f"{prefix}.o"])

        h = _jax_rms_norm(x, params[f"{prefix}.norm2"])
        gate = jnp.dot(h, params[f"{prefix}.gate"])
        up = jnp.dot(h, params[f"{prefix}.up"])
        x = x + jnp.dot((gate * jax.nn.sigmoid(gate)) * up, params[f"{prefix}.down"])
    return x


class TorchGemmaLikeBlockStack(torch.nn.Module):
    def __init__(self, params: dict[str, np.ndarray], *, blocks: int) -> None:
        super().__init__()
        self.blocks = blocks
        for name, value in params.items():
            safe_name = name.replace(".", "_")
            tensor = torch.from_numpy(np.array(value)).to(torch.float16)
            self.register_parameter(safe_name, torch.nn.Parameter(tensor))

    def _param(self, block: int, suffix: str) -> torch.Tensor:
        return getattr(self, f"b{block}_{suffix}")

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = torch.sum(x * x, dim=-1, keepdim=True) / x.shape[-1]
        return (x / torch.sqrt(variance + 1e-3)) * weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in range(self.blocks):
            h = self._rms_norm(x, self._param(block, "norm1"))
            q = torch.matmul(h, self._param(block, "q"))
            k = torch.matmul(h, self._param(block, "k"))
            v = torch.matmul(h, self._param(block, "v"))
            x = x + torch.matmul((q * torch.sigmoid(k)) * v, self._param(block, "o"))

            h = self._rms_norm(x, self._param(block, "norm2"))
            gate = torch.matmul(h, self._param(block, "gate"))
            up = torch.matmul(h, self._param(block, "up"))
            x = x + torch.matmul((gate * torch.sigmoid(gate)) * up, self._param(block, "down"))
        return x


def test_capture_gemma_like_jax_and_pytorch_timings_and_diffs(tmp_path: Path) -> None:
    blocks = 3
    dim = 128
    hidden = 384
    params_np = _gemma_like_params(blocks=blocks, dim=dim, hidden=hidden, seed=11)
    _write_weights_manifest(tmp_path, params_np)
    params_jax = {name: jnp.asarray(value) for name, value in params_np.items()}

    rng = np.random.default_rng(12)
    x_np = (rng.standard_normal((4, 16, dim)) * 0.2).astype(np.float16)
    x_jax = jnp.asarray(x_np)

    jax_ir = capture_jax_function_with_params(
        lambda model_params, value: _jax_gemma_like(model_params, value, blocks=blocks),
        params_jax,
        (x_jax,),
        weights_dir=str(tmp_path),
    )
    jax_graph = transpile_ir(jax_ir)
    jax_graph.set_inputs([x_np])
    cactus_from_jax = jax_graph.execute()[0].numpy().astype(np.float32)
    jax_expected = np.asarray(_jax_gemma_like(params_jax, x_jax, blocks=blocks)).astype(np.float32)
    jax_max_diff, jax_mean_diff = _print_diff("Gemma-like JAX Cactus vs JAX", cactus_from_jax, jax_expected)

    torch_module = TorchGemmaLikeBlockStack(params_np, blocks=blocks).eval()
    x_torch = torch.from_numpy(x_np)
    captured = capture_model(torch_module, (x_torch,))
    torch_graph = transpile_ir(captured.ir_graph)
    torch_graph.set_inputs([x_np])
    cactus_from_torch = torch_graph.execute()[0].numpy().astype(np.float32)
    with torch.no_grad():
        torch_expected = torch_module(x_torch).numpy().astype(np.float32)
    torch_max_diff, torch_mean_diff = _print_diff("Gemma-like PyTorch Cactus vs PyTorch", cactus_from_torch, torch_expected)
    cross_max_diff, cross_mean_diff = _print_diff("Gemma-like Cactus JAX vs Cactus PyTorch", cactus_from_jax, cactus_from_torch)

    print(
        "\nGemma-like IR op counts:\n"
        f"  jax:     nodes={len(jax_ir.order)} ops={dict(_op_counts(jax_ir))}\n"
        f"  pytorch: nodes={len(captured.ir_graph.order)} ops={dict(_op_counts(captured.ir_graph))}"
    )

    _print_timings(
        "JAX generic Gemma-like block stack timings, mmap params (batch=4, seq=16, fp16)",
        fn=lambda value: _jax_gemma_like(params_jax, value, blocks=blocks),
        x=x_jax,
        x_np=x_np,
        graph=jax_graph,
        warmup=3,
        iterations=15,
    )
    _print_torch_timings(
        "PyTorch generic Gemma-like block stack timings, embedded params (batch=4, seq=16, fp16)",
        module=torch_module,
        x=x_torch,
        x_np=x_np,
        graph=torch_graph,
        warmup=3,
        iterations=15,
    )

    assert jax_max_diff < 2e-1
    assert jax_mean_diff < 3e-2
    assert torch_max_diff < 2e-1
    assert torch_mean_diff < 3e-2
    assert cross_max_diff < 2e-1
    assert cross_mean_diff < 3e-2


def _attention_like_params(*, blocks: int, dim: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    return _gemma_like_params(blocks=blocks, dim=dim, hidden=hidden, seed=seed)


def _jax_attention_like(params, x, *, blocks: int):
    scale = jnp.float16(1.0 / np.sqrt(float(x.shape[-1])))
    for block in range(blocks):
        prefix = f"b{block}"
        h = _jax_rms_norm(x, params[f"{prefix}.norm1"])
        q = jnp.dot(h, params[f"{prefix}.q"])
        k = jnp.dot(h, params[f"{prefix}.k"])
        v = jnp.dot(h, params[f"{prefix}.v"])
        scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) * scale
        probs = jnp.exp(scores)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        attn = jnp.matmul(probs, v)
        x = x + jnp.dot(attn, params[f"{prefix}.o"])

        h = _jax_rms_norm(x, params[f"{prefix}.norm2"])
        gate = jnp.dot(h, params[f"{prefix}.gate"])
        up = jnp.dot(h, params[f"{prefix}.up"])
        x = x + jnp.dot((gate * jax.nn.sigmoid(gate)) * up, params[f"{prefix}.down"])
    return x


class TorchAttentionLikeBlockStack(torch.nn.Module):
    def __init__(self, params: dict[str, np.ndarray], *, blocks: int) -> None:
        super().__init__()
        self.blocks = blocks
        for name, value in params.items():
            safe_name = name.replace(".", "_")
            tensor = torch.from_numpy(np.array(value)).to(torch.float16)
            self.register_parameter(safe_name, torch.nn.Parameter(tensor))

    def _param(self, block: int, suffix: str) -> torch.Tensor:
        return getattr(self, f"b{block}_{suffix}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / float(x.shape[-1]) ** 0.5
        for block in range(self.blocks):
            h = TorchGemmaLikeBlockStack._rms_norm(x, self._param(block, "norm1"))
            q = torch.matmul(h, self._param(block, "q"))
            k = torch.matmul(h, self._param(block, "k"))
            v = torch.matmul(h, self._param(block, "v"))
            scores = torch.matmul(q, torch.transpose(k, 1, 2)) * scale
            probs = torch.exp(scores)
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            attn = torch.matmul(probs, v)
            x = x + torch.matmul(attn, self._param(block, "o"))

            h = TorchGemmaLikeBlockStack._rms_norm(x, self._param(block, "norm2"))
            gate = torch.matmul(h, self._param(block, "gate"))
            up = torch.matmul(h, self._param(block, "up"))
            x = x + torch.matmul((gate * torch.sigmoid(gate)) * up, self._param(block, "down"))
        return x


def test_capture_attention_like_jax_and_pytorch_timings_and_diffs(tmp_path: Path) -> None:
    blocks = 2
    dim = 96
    hidden = 256
    params_np = _attention_like_params(blocks=blocks, dim=dim, hidden=hidden, seed=21)
    _write_weights_manifest(tmp_path, params_np)
    params_jax = {name: jnp.asarray(value) for name, value in params_np.items()}

    rng = np.random.default_rng(22)
    x_np = (rng.standard_normal((2, 12, dim)) * 0.15).astype(np.float16)
    x_jax = jnp.asarray(x_np)

    jax_ir = capture_jax_function_with_params(
        lambda model_params, value: _jax_attention_like(model_params, value, blocks=blocks),
        params_jax,
        (x_jax,),
        weights_dir=str(tmp_path),
    )
    jax_graph = transpile_ir(jax_ir)
    jax_graph.set_inputs([x_np])
    cactus_from_jax = jax_graph.execute()[0].numpy().astype(np.float32)
    jax_expected = np.asarray(_jax_attention_like(params_jax, x_jax, blocks=blocks)).astype(np.float32)
    jax_max_diff, jax_mean_diff = _print_diff("Attention-like JAX Cactus vs JAX", cactus_from_jax, jax_expected)

    torch_module = TorchAttentionLikeBlockStack(params_np, blocks=blocks).eval()
    x_torch = torch.from_numpy(x_np)
    captured = capture_model(torch_module, (x_torch,))
    torch_graph = transpile_ir(captured.ir_graph)
    torch_graph.set_inputs([x_np])
    cactus_from_torch = torch_graph.execute()[0].numpy().astype(np.float32)
    with torch.no_grad():
        torch_expected = torch_module(x_torch).numpy().astype(np.float32)
    torch_max_diff, torch_mean_diff = _print_diff("Attention-like PyTorch Cactus vs PyTorch", cactus_from_torch, torch_expected)
    cross_max_diff, cross_mean_diff = _print_diff("Attention-like Cactus JAX vs Cactus PyTorch", cactus_from_jax, cactus_from_torch)

    print(
        "\nAttention-like IR op counts:\n"
        f"  jax:     nodes={len(jax_ir.order)} ops={dict(_op_counts(jax_ir))}\n"
        f"  pytorch: nodes={len(captured.ir_graph.order)} ops={dict(_op_counts(captured.ir_graph))}"
    )

    _print_timings(
        "JAX generic attention-like block stack timings, mmap params (batch=2, seq=12, fp16)",
        fn=lambda value: _jax_attention_like(params_jax, value, blocks=blocks),
        x=x_jax,
        x_np=x_np,
        graph=jax_graph,
        warmup=3,
        iterations=15,
    )
    _print_torch_timings(
        "PyTorch generic attention-like block stack timings, embedded params (batch=2, seq=12, fp16)",
        module=torch_module,
        x=x_torch,
        x_np=x_np,
        graph=torch_graph,
        warmup=3,
        iterations=15,
    )

    assert jax_max_diff < 2e-1
    assert jax_mean_diff < 3e-2
    assert torch_max_diff < 2e-1
    assert torch_mean_diff < 3e-2
    assert cross_max_diff < 2e-1
    assert cross_mean_diff < 3e-2


def _monster_params(
    *,
    blocks: int,
    dim: int,
    heads: int,
    seq: int,
    hidden: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for block in range(blocks):
        prefix = f"m{block}"
        params[f"{prefix}.norm1"] = (1.0 + rng.standard_normal((dim,)) * 0.01).astype(np.float16)
        params[f"{prefix}.q"] = (rng.standard_normal((dim, dim)) * 0.03).astype(np.float16)
        params[f"{prefix}.k"] = (rng.standard_normal((dim, dim)) * 0.03).astype(np.float16)
        params[f"{prefix}.v"] = (rng.standard_normal((dim, dim)) * 0.03).astype(np.float16)
        params[f"{prefix}.o"] = (rng.standard_normal((dim, dim)) * 0.03).astype(np.float16)
        params[f"{prefix}.attn_bias"] = (rng.standard_normal((1, heads, seq, seq)) * 0.01).astype(np.float16)
        params[f"{prefix}.norm2"] = (1.0 + rng.standard_normal((dim,)) * 0.01).astype(np.float16)
        params[f"{prefix}.mix"] = (rng.standard_normal((dim, dim)) * 0.025).astype(np.float16)
        params[f"{prefix}.gate0"] = (rng.standard_normal((dim, hidden)) * 0.025).astype(np.float16)
        params[f"{prefix}.up0"] = (rng.standard_normal((dim, hidden)) * 0.025).astype(np.float16)
        params[f"{prefix}.down0"] = (rng.standard_normal((hidden, dim)) * 0.025).astype(np.float16)
        params[f"{prefix}.gate1"] = (rng.standard_normal((dim, hidden)) * 0.025).astype(np.float16)
        params[f"{prefix}.up1"] = (rng.standard_normal((dim, hidden)) * 0.025).astype(np.float16)
        params[f"{prefix}.down1"] = (rng.standard_normal((hidden, dim)) * 0.025).astype(np.float16)
    return params


def _jax_monster(params, x, *, blocks: int, heads: int):
    head_dim = x.shape[-1] // heads
    scale = jnp.float16(1.0 / np.sqrt(float(head_dim)))
    one = jnp.float16(1.0)
    for block in range(blocks):
        prefix = f"m{block}"
        h = _jax_rms_norm(x, params[f"{prefix}.norm1"])
        q = jnp.dot(h, params[f"{prefix}.q"]).reshape(x.shape[0], x.shape[1], heads, head_dim)
        k = jnp.dot(h, params[f"{prefix}.k"]).reshape(x.shape[0], x.shape[1], heads, head_dim)
        v = jnp.dot(h, params[f"{prefix}.v"]).reshape(x.shape[0], x.shape[1], heads, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale + params[f"{prefix}.attn_bias"]
        probs = jnp.exp(scores)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        context = jnp.matmul(probs, v)
        context = jnp.transpose(context, (0, 2, 1, 3)).reshape(x.shape[0], x.shape[1], x.shape[2])
        x = x + jnp.dot(context, params[f"{prefix}.o"])

        h = _jax_rms_norm(x, params[f"{prefix}.norm2"])
        mix = jax.nn.sigmoid(jnp.dot(h, params[f"{prefix}.mix"]))
        gate0 = jnp.dot(h, params[f"{prefix}.gate0"])
        up0 = jnp.dot(h, params[f"{prefix}.up0"])
        expert0 = jnp.dot((gate0 * jax.nn.sigmoid(gate0)) * up0, params[f"{prefix}.down0"])
        gate1 = jnp.dot(h, params[f"{prefix}.gate1"])
        up1 = jnp.dot(h, params[f"{prefix}.up1"])
        expert1 = jnp.dot((gate1 * jnp.tanh(gate1)) * up1, params[f"{prefix}.down1"])
        x = x + mix * expert0 + (mix * jnp.float16(-1.0) + one) * expert1
    return x


class TorchMonsterBlockStack(torch.nn.Module):
    def __init__(self, params: dict[str, np.ndarray], *, blocks: int, heads: int) -> None:
        super().__init__()
        self.blocks = blocks
        self.heads = heads
        for name, value in params.items():
            safe_name = name.replace(".", "_")
            tensor = torch.from_numpy(np.array(value)).to(torch.float16)
            self.register_parameter(safe_name, torch.nn.Parameter(tensor))

    def _param(self, block: int, suffix: str) -> torch.Tensor:
        return getattr(self, f"m{block}_{suffix}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_dim = x.shape[-1] // self.heads
        scale = 1.0 / float(head_dim) ** 0.5
        for block in range(self.blocks):
            h = TorchGemmaLikeBlockStack._rms_norm(x, self._param(block, "norm1"))
            q = torch.matmul(h, self._param(block, "q")).reshape(x.shape[0], x.shape[1], self.heads, head_dim)
            k = torch.matmul(h, self._param(block, "k")).reshape(x.shape[0], x.shape[1], self.heads, head_dim)
            v = torch.matmul(h, self._param(block, "v")).reshape(x.shape[0], x.shape[1], self.heads, head_dim)
            q = torch.permute(q, (0, 2, 1, 3))
            k = torch.permute(k, (0, 2, 1, 3))
            v = torch.permute(v, (0, 2, 1, 3))
            scores = torch.matmul(q, torch.permute(k, (0, 1, 3, 2))) * scale + self._param(block, "attn_bias")
            probs = torch.exp(scores)
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            context = torch.matmul(probs, v)
            context = torch.permute(context, (0, 2, 1, 3)).reshape(x.shape[0], x.shape[1], x.shape[2])
            x = x + torch.matmul(context, self._param(block, "o"))

            h = TorchGemmaLikeBlockStack._rms_norm(x, self._param(block, "norm2"))
            mix = torch.sigmoid(torch.matmul(h, self._param(block, "mix")))
            gate0 = torch.matmul(h, self._param(block, "gate0"))
            up0 = torch.matmul(h, self._param(block, "up0"))
            expert0 = torch.matmul((gate0 * torch.sigmoid(gate0)) * up0, self._param(block, "down0"))
            gate1 = torch.matmul(h, self._param(block, "gate1"))
            up1 = torch.matmul(h, self._param(block, "up1"))
            expert1 = torch.matmul((gate1 * torch.tanh(gate1)) * up1, self._param(block, "down1"))
            x = x + mix * expert0 + (mix * -1.0 + 1.0) * expert1
        return x


def test_capture_monster_jax_and_pytorch_timings_and_diffs(tmp_path: Path) -> None:
    blocks = 3
    heads = 4
    seq = 16
    dim = 128
    hidden = 384
    params_np = _monster_params(blocks=blocks, dim=dim, heads=heads, seq=seq, hidden=hidden, seed=31)
    _write_weights_manifest(tmp_path, params_np)
    params_jax = {name: jnp.asarray(value) for name, value in params_np.items()}

    rng = np.random.default_rng(32)
    x_np = (rng.standard_normal((2, seq, dim)) * 0.12).astype(np.float16)
    x_jax = jnp.asarray(x_np)

    jax_ir = capture_jax_function_with_params(
        lambda model_params, value: _jax_monster(model_params, value, blocks=blocks, heads=heads),
        params_jax,
        (x_jax,),
        weights_dir=str(tmp_path),
    )
    jax_graph = transpile_ir(jax_ir)
    jax_graph.set_inputs([x_np])
    cactus_from_jax = jax_graph.execute()[0].numpy().astype(np.float32)
    jax_expected = np.asarray(_jax_monster(params_jax, x_jax, blocks=blocks, heads=heads)).astype(np.float32)
    jax_max_diff, jax_mean_diff = _print_diff("Monster JAX Cactus vs JAX", cactus_from_jax, jax_expected)

    torch_module = TorchMonsterBlockStack(params_np, blocks=blocks, heads=heads).eval()
    x_torch = torch.from_numpy(x_np)
    captured = capture_model(torch_module, (x_torch,))
    torch_graph = transpile_ir(captured.ir_graph)
    torch_graph.set_inputs([x_np])
    cactus_from_torch = torch_graph.execute()[0].numpy().astype(np.float32)
    with torch.no_grad():
        torch_expected = torch_module(x_torch).numpy().astype(np.float32)
    torch_max_diff, torch_mean_diff = _print_diff("Monster PyTorch Cactus vs PyTorch", cactus_from_torch, torch_expected)
    cross_max_diff, cross_mean_diff = _print_diff("Monster Cactus JAX vs Cactus PyTorch", cactus_from_jax, cactus_from_torch)

    print(
        "\nMonster IR op counts:\n"
        f"  jax:     nodes={len(jax_ir.order)} ops={dict(_op_counts(jax_ir))}\n"
        f"  pytorch: nodes={len(captured.ir_graph.order)} ops={dict(_op_counts(captured.ir_graph))}"
    )

    _print_timings(
        "JAX generic monster block stack timings, mmap params (batch=2, seq=16, fp16)",
        fn=lambda value: _jax_monster(params_jax, value, blocks=blocks, heads=heads),
        x=x_jax,
        x_np=x_np,
        graph=jax_graph,
        warmup=2,
        iterations=8,
    )
    _print_torch_timings(
        "PyTorch generic monster block stack timings, embedded params (batch=2, seq=16, fp16)",
        module=torch_module,
        x=x_torch,
        x_np=x_np,
        graph=torch_graph,
        warmup=2,
        iterations=8,
    )

    assert jax_max_diff < 3e-1
    assert jax_mean_diff < 4e-2
    assert torch_max_diff < 3e-1
    assert torch_mean_diff < 4e-2
    assert cross_max_diff < 3e-1
    assert cross_mean_diff < 4e-2


def test_capture_real_needle_reduced_forward_timings_and_diffs(tmp_path: Path) -> None:
    flax = pytest.importorskip("flax")
    assert flax is not None
    needle_arch_path = Path("/private/tmp/cactus-needle/needle/model/architecture.py")
    if not needle_arch_path.exists():
        pytest.skip("clone https://github.com/cactus-compute/needle to /private/tmp/cactus-needle")

    spec = importlib.util.spec_from_file_location("needle_architecture", needle_arch_path)
    assert spec is not None and spec.loader is not None
    needle_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(needle_arch)

    config = needle_arch.TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        num_kv_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        max_seq_len=16,
        dtype="float16",
        dropout_rate=0.0,
        no_feedforward=True,
    )
    model = needle_arch.SimpleAttentionNetwork(config)
    src = jnp.asarray([[1, 2, 3, 4, 0, 0, 0, 0]], dtype=jnp.int32)
    tgt = jnp.asarray([[1, 4, 5, 0]], dtype=jnp.int32)
    source_mask = src != config.pad_token_id
    target_mask = (
        jnp.tril(jnp.ones((tgt.shape[1], tgt.shape[1]), dtype=jnp.bool_))[None, :, :]
        & (tgt != config.pad_token_id)[:, None, :]
    )
    params = model.init(jax.random.PRNGKey(0), src, tgt, source_mask, target_mask)["params"]

    flat_params = _flatten_jax_params(params)
    _write_weights_manifest(tmp_path, flat_params)

    def fn(model_params, source_tokens, target_tokens, src_mask, tgt_mask):
        return model.apply({"params": model_params}, source_tokens, target_tokens, src_mask, tgt_mask)

    captured_needle = capture_jax_sequence_model(
        fn,
        params,
        src,
        tgt,
        pad_token_id=config.pad_token_id,
        mask_style="compact",
        weights_dir=str(tmp_path),
        graph_meta={"model": "needle_reduced"},
    )
    needle_ir = captured_needle.ir_graph
    needle_graph = captured_needle.graph

    cactus_out = captured_needle.execute(src, tgt)[0].numpy().astype(np.float32)
    jax_out = np.asarray(fn(params, src, tgt, source_mask, target_mask)).astype(np.float32)
    max_diff, mean_diff = _print_diff("Needle reduced JAX Cactus vs JAX", cactus_out, jax_out)

    runtime_src = jnp.asarray([[8, 7, 6, 5, 4, 0, 0, 0]], dtype=jnp.int32)
    runtime_tgt = jnp.asarray([[1, 6, 7, 0]], dtype=jnp.int32)
    runtime_prepared = captured_needle.prepare_inputs(runtime_src, runtime_tgt)
    runtime_cactus_out = captured_needle.execute(runtime_src, runtime_tgt)[0].numpy().astype(np.float32)
    runtime_jax_out = np.asarray(fn(params, *runtime_prepared.args)).astype(np.float32)
    runtime_max_diff, runtime_mean_diff = _print_diff(
        "Needle reduced runtime input Cactus vs JAX",
        runtime_cactus_out,
        runtime_jax_out,
    )

    prepared = captured_needle.prepare_inputs(src, tgt)

    def encoder_fn(model_params, source_tokens, src_mask):
        encoder_out, _ = model.apply(
            {"params": model_params},
            source_tokens,
            src_mask=src_mask,
            method=model.encode_text,
        )
        return encoder_out

    encoder_out_jax = encoder_fn(params, prepared.source_tokens, prepared.source_mask)

    def decoder_prefill_fn(model_params, target_tokens, encoder_out, self_mask, cross_mask):
        return model.apply(
            {"params": model_params},
            target_tokens,
            encoder_out,
            self_mask=self_mask,
            cross_mask=cross_mask,
            method=model.decode,
        )

    bundle = capture_jax_graphs(
        params,
        (
            JaxGraphSpec(
                name="encoder",
                role="encoder",
                fn=encoder_fn,
                example_args=(prepared.source_tokens, prepared.source_mask),
                input_names=("source_tokens", "source_mask"),
                output_names=("encoder_out",),
            ),
            JaxGraphSpec(
                name="decoder_prefill",
                role="decoder_prefill",
                fn=decoder_prefill_fn,
                example_args=(prepared.target_tokens, encoder_out_jax, prepared.target_mask, prepared.source_mask),
                input_names=("target_tokens", "encoder_out", "self_mask", "cross_mask"),
                output_names=("logits",),
            ),
        ),
        weights_dir=str(tmp_path),
        graph_meta={"model": "needle_reduced", "graph_family": "sequence_split"},
    )
    encoder_cactus = bundle.execute("encoder", prepared.source_tokens, prepared.source_mask)[0].numpy()
    decoder_cactus = bundle.execute(
        "decoder_prefill",
        prepared.target_tokens,
        encoder_cactus,
        prepared.target_mask,
        prepared.source_mask,
    )[0].numpy().astype(np.float32)
    split_max_diff, split_mean_diff = _print_diff(
        "Needle reduced split encoder+decoder Cactus vs JAX",
        decoder_cactus,
        jax_out,
    )

    print(
        "\nNeedle reduced IR op counts:\n"
        f"  jax: nodes={len(needle_ir.order)} ops={dict(_op_counts(needle_ir))}\n"
        f"  mmap_bound_constants={len(needle_graph.bound_constant_bindings)} flat_params={len(flat_params)}\n"
        f"  split_graphs={list(bundle.graphs)} "
        f"encoder_nodes={len(bundle.graphs['encoder'].ir_graph.order)} "
        f"decoder_nodes={len(bundle.graphs['decoder_prefill'].ir_graph.order)}"
    )

    jit_fn = jax.jit(lambda source_tokens, target_tokens, src_mask, tgt_mask: fn(params, source_tokens, target_tokens, src_mask, tgt_mask))
    jit_fn(src, tgt, source_mask, target_mask).block_until_ready()
    jax_jit_ms = _time_ms(
        lambda: jit_fn(src, tgt, source_mask, target_mask).block_until_ready(),
        warmup=2,
        iterations=8,
    )
    needle_graph.set_inputs(prepared.numpy_args())
    cactus_execute_ms = _time_ms(lambda: needle_graph.execute(), warmup=2, iterations=8)
    cactus_set_input_execute_ms = _time_ms(
        lambda: (needle_graph.set_inputs(prepared.numpy_args()), needle_graph.execute()),
        warmup=2,
        iterations=8,
    )
    print(
        "\nNeedle reduced scanned full forward timings (batch=1, src=8, tgt=4, layers=2+2, fp16):\n"
        f"  jax jit execute-only:      {jax_jit_ms:.4f} ms\n"
        f"  cactus execute-only:       {cactus_execute_ms:.4f} ms\n"
        f"  cactus set_input+execute:  {cactus_set_input_execute_ms:.4f} ms"
    )

    assert max_diff < 5e-1
    assert mean_diff < 5e-2
    assert runtime_max_diff < 5e-1
    assert runtime_mean_diff < 5e-2
    assert split_max_diff < 5e-1
    assert split_mean_diff < 5e-2
    assert bundle.graphs["encoder"].ir_graph.meta["jax_graph_role"] == "encoder"
    assert bundle.graphs["decoder_prefill"].ir_graph.meta["jax_graph_role"] == "decoder_prefill"
    assert len(needle_graph.bound_constant_bindings) == len(flat_params) - 1
    assert {binding["source_name"] for binding in needle_graph.bound_constant_bindings} == set(flat_params) - {"log_temp"}


def test_real_needle_checkpoint_generates_weather_tool_call_when_available(tmp_path: Path) -> None:
    checkpoint = Path("checkpoints/needle.pkl")
    tokenizer_model = Path("checkpoints/tokenizer/needle.model")
    needle_root = Path("/private/tmp/cactus-needle")
    needle_arch_path = needle_root / "needle/model/architecture.py"
    needle_tokenizer_path = needle_root / "needle/dataset/tokenizer.py"
    if not checkpoint.exists() or not tokenizer_model.exists():
        pytest.skip("download Cactus-Compute/needle needle.pkl and tokenizer/needle.model to python/checkpoints")
    if not needle_arch_path.exists() or not needle_tokenizer_path.exists():
        pytest.skip("clone https://github.com/cactus-compute/needle to /private/tmp/cactus-needle")

    arch_spec = importlib.util.spec_from_file_location("needle_architecture_real", needle_arch_path)
    tok_spec = importlib.util.spec_from_file_location("needle_tokenizer_real", needle_tokenizer_path)
    assert arch_spec is not None and arch_spec.loader is not None
    assert tok_spec is not None and tok_spec.loader is not None
    needle_arch = importlib.util.module_from_spec(arch_spec)
    needle_tok = importlib.util.module_from_spec(tok_spec)
    arch_spec.loader.exec_module(needle_arch)
    tok_spec.loader.exec_module(needle_tok)

    with checkpoint.open("rb") as f:
        data = pickle.load(f)
    params = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.bfloat16), data["params"])
    config = needle_arch.TransformerConfig(**data["config"])
    model = needle_arch.SimpleAttentionNetwork(config)
    tokenizer = needle_tok.NeedleTokenizer(str(tokenizer_model))

    query = "What's the weather in San Francisco?"
    tools = (
        '[{"name":"get_weather","description":"Get current weather for a city.",'
        '"parameters":{"location":{"type":"string","description":"City name.","required":true}}}]'
    )
    max_enc_len = 256
    max_gen_len = 48
    query_tokens = tokenizer.encode(query)[: max_enc_len - 2]
    tool_tokens = tokenizer.encode(tools)
    enc_tokens = query_tokens + [tokenizer.tools_token_id] + tool_tokens[: max_enc_len - len(query_tokens) - 1]
    enc_input = jnp.asarray([enc_tokens], dtype=jnp.int32)
    src_mask = needle_arch.make_padding_mask(enc_input, tokenizer.pad_token_id)

    start = perf_counter()
    encoder_out, enc_mask = model.apply({"params": params}, enc_input, src_mask=src_mask, method="encode")
    encoder_out.block_until_ready()
    encode_ms = (perf_counter() - start) * 1000.0

    tgt_mask = needle_arch.make_causal_mask(max_gen_len)

    flat_params = _flatten_jax_params(params)
    _write_weights_manifest(
        tmp_path,
        {name: value.astype(np.float16) for name, value in flat_params.items() if name != "log_temp"},
    )

    def encoder_fn(model_params, source_tokens, src_mask_arg):
        return model.apply({"params": model_params}, source_tokens, src_mask=src_mask_arg, method="encode")

    def decoder_prefill_fn(model_params, target_tokens, encoder_state, self_mask, cross_mask):
        return model.apply(
            {"params": model_params},
            target_tokens,
            encoder_state,
            self_mask=self_mask,
            cross_mask=cross_mask,
            method="decode",
        )

    seed_buffer = jnp.full((1, max_gen_len), tokenizer.pad_token_id, dtype=jnp.int32)
    seed_buffer = seed_buffer.at[0, 0].set(tokenizer.eos_token_id)
    bundle = capture_jax_graphs(
        params,
        (
            JaxGraphSpec(
                name="encoder",
                role="encoder",
                fn=encoder_fn,
                example_args=(enc_input, src_mask),
                input_names=("source_tokens", "src_mask"),
                output_names=("encoder_out", "enc_mask"),
            ),
            JaxGraphSpec(
                name="decoder_prefill",
                role="decoder_prefill",
                fn=decoder_prefill_fn,
                example_args=(seed_buffer, encoder_out, tgt_mask, enc_mask),
                input_names=("target_tokens", "encoder_out", "self_mask", "cross_mask"),
                output_names=("logits",),
            ),
        ),
        weights_dir=str(tmp_path),
        graph_meta={"model": "needle_real", "graph_family": "sequence_split"},
    )
    encoder_cactus, mask_cactus = [
        output.numpy() for output in bundle.execute("encoder", enc_input, src_mask)
    ]
    seed_logits_cactus = bundle.execute(
        "decoder_prefill",
        seed_buffer,
        encoder_cactus,
        tgt_mask,
        mask_cactus,
    )[0].numpy().astype(np.float32)
    seed_logits_jax = np.asarray(
        decoder_prefill_fn(params, seed_buffer, encoder_out, tgt_mask, enc_mask)
    ).astype(np.float32)
    seed_logit_max_diff = float(np.nanmax(np.abs(seed_logits_cactus - seed_logits_jax)))
    seed_logit_mean_diff = float(np.nanmean(np.abs(seed_logits_cactus - seed_logits_jax)))

    encoder_graph = bundle.graphs["encoder"].graph
    decoder_graph = bundle.graphs["decoder_prefill"].graph
    encoder_graph.set_inputs([np.asarray(enc_input), np.asarray(src_mask)])
    cactus_encoder_ms = _time_ms(lambda: encoder_graph.execute(), warmup=2, iterations=8)
    decoder_graph.set_inputs([np.asarray(seed_buffer), encoder_cactus, np.asarray(tgt_mask), mask_cactus])
    cactus_prefill_decoder_ms = _time_ms(lambda: decoder_graph.execute(), warmup=2, iterations=8)
    cactus_prefill_ms = cactus_encoder_ms + cactus_prefill_decoder_ms

    @jax.jit
    def decode_step(dec_buffer):
        return model.apply(
            {"params": params},
            dec_buffer,
            encoder_out,
            self_mask=tgt_mask,
            cross_mask=enc_mask,
            method="decode",
        )

    dec_buffer = jnp.full((1, max_gen_len), tokenizer.pad_token_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(tokenizer.eos_token_id)
    decode_step(dec_buffer).block_until_ready()
    generated: list[int] = []
    start = perf_counter()
    for index in range(max_gen_len - 1):
        logits = decode_step(dec_buffer)
        logits.block_until_ready()
        next_token = int(jnp.argmax(logits[0, index]))
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        dec_buffer = dec_buffer.at[0, index + 1].set(next_token)
    decode_ms = (perf_counter() - start) * 1000.0

    result = tokenizer.decode(generated)
    if result.startswith("<tool_call>"):
        result = result[len("<tool_call>"):]
    parsed = json.loads(result)

    cactus_buffer = np.full((1, max_gen_len), tokenizer.pad_token_id, dtype=np.int32)
    cactus_buffer[0, 0] = tokenizer.eos_token_id
    cactus_generated: list[int] = []
    start = perf_counter()
    for index in range(max_gen_len - 1):
        logits = bundle.execute(
            "decoder_prefill",
            cactus_buffer,
            encoder_cactus,
            tgt_mask,
            mask_cactus,
        )[0].numpy().astype(np.float32)
        next_token = int(np.nanargmax(logits[0, index]))
        if next_token == tokenizer.eos_token_id:
            break
        cactus_generated.append(next_token)
        cactus_buffer[0, index + 1] = next_token
    cactus_decode_ms = (perf_counter() - start) * 1000.0

    cactus_result = tokenizer.decode(cactus_generated)
    if cactus_result.startswith("<tool_call>"):
        cactus_result = cactus_result[len("<tool_call>"):]
    cactus_parsed = json.loads(cactus_result)

    print(
        "\nNeedle real checkpoint English prompt:\n"
        f"  query={query!r}\n"
        f"  jax_result={result}\n"
        f"  cactus_result={cactus_result}\n"
        f"  seed_logit_diff=max {seed_logit_max_diff:.4f}, mean {seed_logit_mean_diff:.4f}\n"
        f"  jax_encode={encode_ms:.2f} ms jax_decode_loop={decode_ms:.2f} ms "
        f"jax_decode_tps={len(generated) / (decode_ms / 1000.0):.2f}\n"
        f"  cactus_prefill={cactus_prefill_ms:.2f} ms "
        f"cactus_prefill_tps={len(enc_tokens) / (cactus_prefill_ms / 1000.0):.2f} "
        f"(encoder={cactus_encoder_ms:.2f} ms decoder={cactus_prefill_decoder_ms:.2f} ms)\n"
        f"  cactus_decode_loop={cactus_decode_ms:.2f} ms tokens={len(cactus_generated)} "
        f"cactus_decode_tps={len(cactus_generated) / (cactus_decode_ms / 1000.0):.2f}"
    )
    assert parsed == [{"name": "get_weather", "arguments": {"location": "San Francisco"}}]
    assert cactus_parsed == [{"name": "get_weather", "arguments": {"location": "San Francisco"}}]
    assert not np.isnan(seed_logits_cactus).any()
