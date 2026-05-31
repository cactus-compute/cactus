from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.transpile.capture_jax import capture_jax_function
from cactus.transpile.capture_jax import capture_jax_function_with_params
from cactus.transpile.lower import transpile_ir


def _execute_ir(ir, *inputs: object) -> list[np.ndarray]:
    graph = transpile_ir(ir)
    graph.set_inputs([np.asarray(value) for value in inputs])
    return [output.numpy() for output in graph.execute()]


def _assert_close(actual: object, expected: object, *, atol: float = 8e-2, rtol: float = 8e-2) -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=atol, rtol=rtol)


def _write_fp16_manifest(weights_dir: Path, params: dict[str, object]) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, value in params.items():
        filename = f"{name}.weights"
        save_tensor_with_header(np.asarray(value), weights_dir / filename, precision="FP16")
        manifest[name] = {"filename": filename, "kind": "weight"}
    (weights_dir / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def test_capture_jax_function_imports_common_tensor_ops() -> None:
    x = jnp.asarray([[0.2, -0.4, 0.7], [1.0, -1.2, 0.3]], dtype=jnp.float16)
    w = jnp.asarray([[0.3, -0.5], [0.2, 0.4], [-0.7, 0.1]], dtype=jnp.float16)
    b = jnp.asarray([0.05, -0.02], dtype=jnp.float16)

    def fn(values):
        hidden = jnp.matmul(values, w) + b
        gated = hidden * jax.nn.sigmoid(hidden)
        return jnp.concatenate([gated, jnp.tanh(gated)], axis=-1)

    ir = capture_jax_function(fn, (x,), constant_names=("w", "b"))
    got = _execute_ir(ir, x)[0]

    assert ir.meta["frontend"] == "jax"
    assert ir.meta["adapter_family"] == "generic"
    _assert_close(got, fn(x))


def test_capture_jax_params_resolve_mmap_weights(tmp_path: Path) -> None:
    params = {
        "w1": jnp.asarray([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]], dtype=jnp.float16),
        "b1": jnp.asarray([0.01, -0.02, 0.03], dtype=jnp.float16),
        "w2": jnp.asarray([[0.2], [-0.3], [0.4]], dtype=jnp.float16),
    }
    x = jnp.asarray([[1.0, -2.0]], dtype=jnp.float16)
    _write_fp16_manifest(tmp_path, params)

    def fn(model_params, values):
        hidden = jax.nn.gelu(values @ model_params["w1"] + model_params["b1"])
        return hidden @ model_params["w2"]

    ir = capture_jax_function_with_params(fn, params, (x,), weights_dir=str(tmp_path))
    graph = transpile_ir(ir)
    graph.set_inputs([np.asarray(x)])
    got = graph.execute()[0].numpy()

    bound_sources = {binding["source_name"] for binding in graph.bound_constant_bindings}
    assert bound_sources == set(params)
    _assert_close(got, fn(params, x))


def test_capture_jax_handles_scalar_and_broadcast_boundaries() -> None:
    scalar = jnp.asarray(2.0, dtype=jnp.float16)
    x = jnp.asarray([[1.0, -2.0, 3.0]], dtype=jnp.float16)

    def fn(runtime_scalar, values):
        constant = jnp.asarray(0.25, dtype=jnp.float16)
        return values * runtime_scalar + jnp.broadcast_to(constant, values.shape)

    ir = capture_jax_function(fn, (scalar, x))
    got = _execute_ir(ir, scalar, x)[0]

    _assert_close(got, fn(scalar, x))


def test_capture_jax_gqa_repeat_then_matmul_matches_jax() -> None:
    q = jnp.ones((1, 4, 3, 2), dtype=jnp.float16)
    k = jnp.asarray(np.arange(12, dtype=np.float16).reshape(1, 2, 3, 2) / 10.0)

    def fn(query, key):
        repeated = jnp.repeat(key, 2, axis=1)
        return jnp.matmul(query, jnp.swapaxes(repeated, -1, -2))

    ir = capture_jax_function(fn, (q, k))
    got = _execute_ir(ir, q, k)[0]

    _assert_close(got, fn(q, k))


def test_capture_jax_rms_norm_pattern_is_visible_to_optimizer() -> None:
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float16).reshape(1, 2, 8))
    weight = jnp.asarray(np.linspace(0.8, 1.2, 8, dtype=np.float16))

    def fn(values):
        rms = jnp.sqrt(jnp.mean(values.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1.0e-6)
        return ((values / rms) * weight).astype(jnp.float16)

    ir = capture_jax_function(fn, (x,), constant_names=("weight",))
    got = _execute_ir(ir, x)[0]

    assert any(ir.nodes[node_id].op == "rms_norm" for node_id in ir.order)
    _assert_close(got, fn(x))


def test_capture_jax_rms_norm_sum_div_pattern_avoids_fp16_overflow() -> None:
    x = jnp.asarray(np.linspace(-650.0, 640.0, 16, dtype=np.float16).reshape(1, 2, 8))
    weight = jnp.asarray(np.linspace(0.8, 1.2, 8, dtype=np.float16))

    def fn(values):
        squared = values.astype(jnp.float32) * values.astype(jnp.float32)
        var = jnp.sum(squared, axis=-1, keepdims=True) / values.shape[-1]
        return (values * jax.lax.rsqrt(var + 1.0e-5)) * weight

    ir = capture_jax_function(fn, (x,), constant_names=("weight",))
    got = _execute_ir(ir, x)[0]

    assert any(ir.nodes[node_id].op == "rms_norm" for node_id in ir.order)
    _assert_close(got, fn(x))


@pytest.mark.parametrize("variant", ["mean_rsqrt", "sum_div_rsqrt", "pow_neg_half", "inv_sqrt", "weight_first"])
def test_capture_jax_rms_norm_common_spellings_lower_to_rms_norm(variant: str) -> None:
    x = jnp.asarray(np.linspace(-650.0, 640.0, 16, dtype=np.float16).reshape(1, 2, 8))
    weight = jnp.asarray(np.linspace(0.8, 1.2, 8, dtype=np.float16))

    def fn(values):
        squared = values.astype(jnp.float32) * values.astype(jnp.float32)
        mean_square = jnp.mean(squared, axis=-1, keepdims=True)
        if variant == "sum_div_rsqrt":
            inv = jax.lax.rsqrt(jnp.sum(squared, axis=-1, keepdims=True) / values.shape[-1] + 1.0e-5)
            return values * inv * weight
        if variant == "pow_neg_half":
            return values * ((mean_square + 1.0e-5) ** -0.5) * weight
        if variant == "inv_sqrt":
            return values * (1.0 / jnp.sqrt(mean_square + 1.0e-5)) * weight
        if variant == "weight_first":
            return weight * (values * jax.lax.rsqrt(mean_square + 1.0e-5))
        return values * jax.lax.rsqrt(mean_square + 1.0e-5) * weight

    ir = capture_jax_function(fn, (x,), constant_names=("weight",))
    got = _execute_ir(ir, x)[0]

    assert any(ir.nodes[node_id].op == "rms_norm" for node_id in ir.order)
    _assert_close(got, fn(x))


def test_capture_jax_zero_centered_rms_norm_lowers_to_rms_norm() -> None:
    x = jnp.asarray(np.linspace(-73.0, 73.0, 512, dtype=np.float16).reshape(1, 1, 512))
    scale = jnp.zeros((512,), dtype=jnp.float16)

    def fn(values):
        rms = jnp.sqrt(jnp.mean(values.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1.0e-6)
        return ((1.0 + scale) * values / rms).astype(jnp.float16)

    ir = capture_jax_function(fn, (x,), constant_names=("scale",))
    got = _execute_ir(ir, x)[0]

    assert any(ir.nodes[node_id].op == "rms_norm" for node_id in ir.order)
    _assert_close(got, fn(x))


@pytest.mark.parametrize("variant", ["mean_rsqrt", "sum_div_rsqrt", "sqrt_div", "no_bias"])
def test_capture_jax_layer_norm_common_spellings_lower_to_layer_norm(variant: str) -> None:
    x = jnp.asarray(np.linspace(-650.0, 640.0, 16, dtype=np.float16).reshape(1, 2, 8))
    weight = jnp.asarray(np.linspace(0.8, 1.2, 8, dtype=np.float16))
    bias = jnp.asarray(np.linspace(-0.2, 0.2, 8, dtype=np.float16))

    def fn(values):
        centered = values - jnp.mean(values.astype(jnp.float32), axis=-1, keepdims=True)
        if variant == "sum_div_rsqrt":
            var = jnp.sum(centered.astype(jnp.float32) * centered.astype(jnp.float32), axis=-1, keepdims=True) / values.shape[-1]
            normed = centered * jax.lax.rsqrt(var + 1.0e-5)
        elif variant == "sqrt_div":
            var = jnp.mean(centered.astype(jnp.float32) * centered.astype(jnp.float32), axis=-1, keepdims=True)
            normed = centered / jnp.sqrt(var + 1.0e-5)
        else:
            var = jnp.mean(centered.astype(jnp.float32) * centered.astype(jnp.float32), axis=-1, keepdims=True)
            normed = centered * jax.lax.rsqrt(var + 1.0e-5)
        if variant == "no_bias":
            return normed * weight
        return normed * weight + bias

    ir = capture_jax_function(fn, (x,), constant_names=("weight", "bias"))
    got = _execute_ir(ir, x)[0]

    assert any(ir.nodes[node_id].op == "layer_norm" for node_id in ir.order)
    _assert_close(got, fn(x))


def test_capture_jax_batch_norm_lowers_to_batch_norm() -> None:
    x = jnp.asarray(np.linspace(-1.0, 1.0, 24, dtype=np.float16).reshape(2, 3, 4))
    scale = jnp.asarray([0.8, 1.1, 0.9, 1.2], dtype=jnp.float16)
    bias = jnp.asarray([0.05, -0.02, 0.03, -0.04], dtype=jnp.float16)
    running_mean = jnp.asarray([0.1, -0.2, 0.05, 0.0], dtype=jnp.float16)
    running_var = jnp.asarray([0.9, 1.2, 0.7, 1.5], dtype=jnp.float16)

    def fn(values):
        normalized = (values - running_mean) * (scale * ((running_var + 1.0e-5) ** -0.5))
        return normalized + bias

    ir = capture_jax_function(fn, (x,), constant_names=("bias", "running_mean", "running_var", "scale"))
    got = _execute_ir(ir, x)[0]

    assert any(ir.nodes[node_id].op == "batch_norm" for node_id in ir.order)
    _assert_close(got, fn(x))


def test_capture_jax_prenorm_residual_add_lowers_to_add_clipped() -> None:
    x = jnp.asarray(np.linspace(-0.5, 0.5, 16, dtype=np.float16).reshape(1, 2, 8))
    weight = jnp.asarray(np.linspace(0.8, 1.2, 8, dtype=np.float16))

    def fn(values):
        rms = jnp.sqrt(jnp.mean(values.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1.0e-6)
        branch = (values / rms) * weight
        return values + branch

    ir = capture_jax_function(fn, (x,), constant_names=("weight",))
    got = _execute_ir(ir, x)[0]

    assert any(ir.nodes[node_id].op == "add_clipped" for node_id in ir.order)
    _assert_close(got, fn(x))


def test_capture_jax_prefill_attention_pattern_lowers_to_attention() -> None:
    q = jnp.asarray(np.linspace(-0.5, 0.5, 1 * 2 * 3 * 4, dtype=np.float16).reshape(1, 2, 3, 4))
    k = jnp.asarray(np.linspace(0.25, -0.25, 1 * 2 * 3 * 4, dtype=np.float16).reshape(1, 2, 3, 4))
    v = jnp.asarray(np.linspace(-0.75, 0.75, 1 * 2 * 3 * 4, dtype=np.float16).reshape(1, 2, 3, 4))
    mask = jnp.asarray(np.where(np.tril(np.ones((1, 2, 3, 3), dtype=np.bool_)), 0.0, -1.0e4), dtype=jnp.float16)

    def fn(query, key, value, additive_mask):
        scores = (query @ key.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.asarray(4.0, dtype=jnp.float32))
        probs = jax.nn.softmax(scores + additive_mask, axis=-1)
        return (probs @ value).transpose(0, 2, 1, 3).reshape(1, 3, 8)

    ir = capture_jax_function(fn, (q, k, v, mask))
    got = _execute_ir(ir, q, k, v, mask)[0]

    assert any(ir.nodes[node_id].op == "attention" for node_id in ir.order)
    _assert_close(got, fn(q, k, v, mask))


def test_capture_jax_attention_keeps_broadcast_mask_decomposed() -> None:
    q = jnp.asarray(np.linspace(-0.5, 0.5, 2 * 4 * 5 * 8, dtype=np.float16).reshape(2, 4, 5, 8))
    k = jnp.asarray(np.linspace(0.25, -0.25, 2 * 4 * 5 * 8, dtype=np.float16).reshape(2, 4, 5, 8))
    v = jnp.asarray(np.linspace(-0.75, 0.75, 2 * 4 * 5 * 8, dtype=np.float16).reshape(2, 4, 5, 8))
    mask = jnp.asarray(np.where(np.tril(np.ones((1, 1, 5, 5), dtype=np.bool_)), 0.0, -1.0e4), dtype=jnp.float16)

    def fn(query, key, value, additive_mask):
        scores = (query @ key.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.asarray(8.0, dtype=jnp.float32))
        probs = jax.nn.softmax(scores + additive_mask, axis=-1)
        return (probs @ value).transpose(0, 2, 1, 3).reshape(2, 5, 32)

    ir = capture_jax_function(fn, (q, k, v, mask))
    got = _execute_ir(ir, q, k, v, mask)[0]

    assert not any(ir.nodes[node_id].op == "attention" for node_id in ir.order)
    _assert_close(got, fn(q, k, v, mask))


def test_capture_flax_module_smoke_when_available() -> None:
    pytest.importorskip("flax")
    from flax import linen as nn

    class TinyFlaxModule(nn.Module):
        @nn.compact
        def __call__(self, token_ids, features):
            embedded = nn.Embed(8, 4)(token_ids)
            projected = nn.Dense(4)(features)
            hidden = nn.LayerNorm()(embedded + projected)
            return nn.Dense(3)(jax.nn.silu(hidden))

    model = TinyFlaxModule()
    token_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)
    features = jnp.asarray([[[0.2, -0.1], [0.4, 0.3]]], dtype=jnp.float16)
    params = model.init(jax.random.PRNGKey(0), token_ids, features)["params"]

    def fn(model_params, ids, feats):
        return model.apply({"params": model_params}, ids, feats)

    ir = capture_jax_function_with_params(fn, params, (token_ids, features))
    got = _execute_ir(ir, token_ids, features)[0]

    assert any(ir.nodes[node_id].op == "layer_norm" for node_id in ir.order)
    _assert_close(got, fn(params, token_ids, features), atol=1.2e-1, rtol=1.2e-1)
