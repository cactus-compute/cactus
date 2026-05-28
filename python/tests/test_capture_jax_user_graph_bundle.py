from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_user_graph_bundle
from cactus.transpile.jax_user_graph_bundle import load_jax_user_graph_bundle


def _assert_close(actual: object, expected: object, *, atol: float = 8e-2, rtol: float = 8e-2) -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=atol, rtol=rtol)


def _toy_export():
    params = {
        "encoder_w": jnp.asarray([[0.2, -0.1, 0.4], [0.3, 0.5, -0.2]], dtype=jnp.float16),
        "decoder_w": jnp.asarray([[0.1, -0.3], [0.4, 0.2], [-0.2, 0.6]], dtype=jnp.float16),
        "out_b": jnp.asarray([0.01, -0.02], dtype=jnp.float16),
    }
    source = jnp.asarray([[[1.0, -0.5], [0.25, 0.75]]], dtype=jnp.float16)
    target = jnp.asarray([[[0.4, -0.1, 0.2]]], dtype=jnp.float16)

    def encoder_fn(model_params, source_features):
        encoded = source_features @ model_params["encoder_w"]
        return encoded * jax.nn.sigmoid(encoded)

    def decoder_fn(model_params, target_features, encoder_out):
        context = jnp.mean(encoder_out, axis=1, keepdims=True)
        return (target_features + context) @ model_params["decoder_w"] + model_params["out_b"]

    encoder_out = encoder_fn(params, source)
    specs = (
        JaxGraphSpec(
            name="encoder",
            role="encoder",
            fn=encoder_fn,
            example_args=(source,),
            input_names=("source_features",),
            output_names=("encoder_out",),
        ),
        JaxGraphSpec(
            name="decoder",
            role="generic",
            fn=decoder_fn,
            example_args=(target, encoder_out),
            input_names=("target_features", "encoder_out"),
            output_names=("logits",),
        ),
    )
    return params, specs, source, target, encoder_fn, decoder_fn


def test_jax_user_graph_bundle_writes_manifest_graphs_and_weights(tmp_path: Path) -> None:
    params, specs, source, target, encoder_fn, decoder_fn = _toy_export()
    result = build_jax_user_graph_bundle(
        params=params,
        specs=specs,
        output_dir=tmp_path / "bundle",
        model_id="toy-jax",
        task="generic",
        inputs_metadata={"owner": "client"},
    )

    encoder_out = result.bundle.execute("encoder", source)[0].numpy()
    logits = result.bundle.execute("decoder", target, encoder_out)[0].numpy()
    manifest = json.loads(result.components_manifest_path.read_text())

    assert manifest["model_source"] == "jax_user_graph"
    assert manifest["component_order"] == ["encoder", "decoder"]
    assert (tmp_path / "bundle/components/encoder/graph.cactus").exists()
    assert (tmp_path / "bundle/components/decoder/graph.cactus").exists()
    for component in manifest["components"]:
        raw_ir_path = tmp_path / "bundle" / component["raw_ir"]
        optimized_ir_path = tmp_path / "bundle" / component["optimized_ir"]
        assert raw_ir_path == tmp_path / "bundle" / component["directory"] / "raw_ir.json"
        assert optimized_ir_path == tmp_path / "bundle" / component["directory"] / "optimized_ir.json"
        assert raw_ir_path.exists()
        assert optimized_ir_path.exists()
        assert json.loads(raw_ir_path.read_text())["graph"]["meta"]["frontend"] == "jax"
        assert json.loads(optimized_ir_path.read_text())["graph"]["outputs"] == component["outputs"]
    assert (tmp_path / "bundle/weights_manifest.json").exists()
    _assert_close(encoder_out, encoder_fn(params, source))
    _assert_close(logits, decoder_fn(params, target, encoder_fn(params, source)))


def test_jax_user_graph_bundle_loads_saved_graphs_and_mmap_weights(tmp_path: Path) -> None:
    params, specs, source, target, encoder_fn, decoder_fn = _toy_export()
    result = build_jax_user_graph_bundle(
        params=params,
        specs=specs,
        output_dir=tmp_path / "bundle",
        model_id="toy-jax",
    )
    loaded = load_jax_user_graph_bundle(result.output_dir)

    encoder_out = loaded.execute("encoder", source)[0].numpy()
    logits = loaded.execute("decoder", target, encoder_out)[0].numpy()

    assert set(loaded.graphs) == {"encoder", "decoder"}
    assert loaded.graphs["encoder"].logical_inputs == ["source_features"]
    assert loaded.graphs["decoder"].logical_outputs == ["logits"]
    _assert_close(encoder_out, encoder_fn(params, source))
    _assert_close(logits, decoder_fn(params, target, encoder_fn(params, source)))


def test_jax_user_graph_bundle_supports_external_weights_dir(tmp_path: Path) -> None:
    params = {
        "w": jnp.asarray([[0.5], [-0.25]], dtype=jnp.float16),
        "b": jnp.asarray([0.1], dtype=jnp.float16),
    }
    x = jnp.asarray([[2.0, -1.0]], dtype=jnp.float16)

    def fn(model_params, values):
        return values @ model_params["w"] + model_params["b"]

    result = build_jax_user_graph_bundle(
        params=params,
        specs=(JaxGraphSpec(name="project", fn=fn, example_args=(x,), input_names=("x",), output_names=("y",)),),
        output_dir=tmp_path / "bundle",
        weights_dir=tmp_path / "weights",
        model_id="external-weights",
    )
    loaded = load_jax_user_graph_bundle(result.components_manifest_path)
    manifest = json.loads(result.components_manifest_path.read_text())

    assert result.weights_dir == tmp_path / "weights"
    assert (tmp_path / "weights/weights_manifest.json").exists()
    assert manifest["weights_dir"] == str(tmp_path / "weights")
    _assert_close(loaded.execute("project", x)[0].numpy(), fn(params, x))


def test_jax_user_graph_bundle_binds_duplicate_equal_params_by_path(tmp_path: Path) -> None:
    params = {
        "a": jnp.zeros((2,), dtype=jnp.float16),
        "b": jnp.zeros((2,), dtype=jnp.float16),
    }
    weight_arrays = {
        "a": np.asarray([1.0, 1.0], dtype=np.float16),
        "b": np.asarray([2.0, 2.0], dtype=np.float16),
    }
    x = jnp.asarray([10.0, 20.0], dtype=jnp.float16)

    def fn(model_params, values):
        return values + model_params["b"]

    result = build_jax_user_graph_bundle(
        params=params,
        weight_arrays=weight_arrays,
        specs=(JaxGraphSpec(name="forward", fn=fn, example_args=(x,)),),
        output_dir=tmp_path / "bundle",
        model_id="duplicate-param-bindings",
    )
    loaded = load_jax_user_graph_bundle(result.output_dir)
    bindings = loaded.manifest["components"][0]["bound_constant_bindings"]

    assert [binding["source_name"] for binding in bindings] == ["b"]
    _assert_close(loaded.execute("forward", x)[0].numpy(), np.asarray(x) + weight_arrays["b"])


def test_jax_user_graph_execute_coerces_inputs_to_example_dtype(tmp_path: Path) -> None:
    params = {"w": jnp.asarray([[0.5], [-0.25]], dtype=jnp.float16)}
    x = jnp.asarray([[2.0, -1.0]], dtype=jnp.float16)

    def fn(model_params, values):
        return values @ model_params["w"]

    result = build_jax_user_graph_bundle(
        params=params,
        specs=(JaxGraphSpec(name="project", fn=fn, example_args=(x,)),),
        output_dir=tmp_path / "bundle",
        model_id="dtype-coercion",
    )

    got = result.bundle.execute("project", np.asarray(x, dtype=np.float32))[0].numpy()
    _assert_close(got, fn(params, x))


def test_jax_user_graph_rejects_wrong_input_count(tmp_path: Path) -> None:
    params = {"w": jnp.asarray([[1.0]], dtype=jnp.float16)}
    x = jnp.asarray([[2.0]], dtype=jnp.float16)

    def fn(model_params, values):
        return values @ model_params["w"]

    result = build_jax_user_graph_bundle(
        params=params,
        specs=(JaxGraphSpec(name="project", fn=fn, example_args=(x,)),),
        output_dir=tmp_path / "bundle",
        model_id="errors",
    )

    with pytest.raises(ValueError, match="expected 1 inputs"):
        result.bundle.execute("project")
