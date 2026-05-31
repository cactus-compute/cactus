from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_generation_graph_bundle
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
            name="encoder", role="encoder", fn=encoder_fn, example_args=(source,),
            input_names=("source_features",), output_names=("encoder_out",),
        ),
        JaxGraphSpec(
            name="decoder", role="generic", fn=decoder_fn, example_args=(target, encoder_out),
            input_names=("target_features", "encoder_out"), output_names=("logits",),
        ),
    )
    return params, specs, source, target, encoder_fn, decoder_fn


def _build_toy_bundle(tmp_path: Path, **kwargs):
    params, specs, source, target, encoder_fn, decoder_fn = _toy_export()
    result = build_jax_user_graph_bundle(
        params=params,
        specs=specs,
        output_dir=tmp_path / "bundle",
        model_id="toy-jax",
        **kwargs,
    )
    return result, params, source, target, encoder_fn, decoder_fn


def test_jax_user_graph_bundle_writes_manifest_graphs_and_weights(tmp_path: Path) -> None:
    result, params, source, target, encoder_fn, decoder_fn = _build_toy_bundle(
        tmp_path,
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
        raw_ir_path, optimized_ir_path = (tmp_path / "bundle" / component[key] for key in ("raw_ir", "optimized_ir"))
        assert raw_ir_path.exists() and optimized_ir_path.exists()
        assert json.loads(raw_ir_path.read_text())["graph"]["meta"]["frontend"] == "jax"
        assert json.loads(optimized_ir_path.read_text())["graph"]["outputs"] == component["outputs"]
    assert (tmp_path / "bundle/weights_manifest.json").exists()
    _assert_close(encoder_out, encoder_fn(params, source))
    _assert_close(logits, decoder_fn(params, target, encoder_fn(params, source)))


def test_jax_user_graph_bundle_loads_saved_graphs_and_mmap_weights(tmp_path: Path) -> None:
    result, params, source, target, encoder_fn, decoder_fn = _build_toy_bundle(tmp_path)
    loaded = load_jax_user_graph_bundle(result.output_dir)

    encoder_out = loaded.execute("encoder", source)[0]
    logits = loaded.execute("decoder", target, encoder_out)[0].numpy()

    assert set(loaded.graphs) == {"encoder", "decoder"}
    assert loaded.graphs["encoder"].logical_inputs == ["source_features"]
    assert loaded.graphs["decoder"].logical_outputs == ["logits"]
    _assert_close(encoder_out.numpy(), encoder_fn(params, source))
    _assert_close(logits, decoder_fn(params, target, encoder_fn(params, source)))
    loaded.reset()


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
    _assert_close(loaded.execute("project", np.asarray(x, dtype=np.float32))[0].numpy(), fn(params, x))


def test_jax_generation_decoder_step_fuses_attention_without_internal_cache(tmp_path: Path) -> None:
    params = {
        "wq": jnp.eye(16, dtype=jnp.float16),
        "wk": jnp.eye(16, dtype=jnp.float16),
        "wv": jnp.eye(16, dtype=jnp.float16),
    }
    x = jnp.asarray(np.linspace(-0.75, 0.75, 16, dtype=np.float16).reshape(1, 1, 16))

    def decoder_step(model_params, values):
        q = (values @ model_params["wq"]).reshape(1, 1, 2, 8).transpose(0, 2, 1, 3)
        k = (values @ model_params["wk"]).reshape(1, 1, 2, 8).transpose(0, 2, 1, 3)
        v = (values @ model_params["wv"]).reshape(1, 1, 2, 8).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.asarray(8.0, dtype=jnp.float32))
        probs = jax.nn.softmax(scores, axis=-1).astype(jnp.float16)
        return (probs @ v).transpose(0, 2, 1, 3).reshape(1, 1, 16)

    result = build_jax_generation_graph_bundle(
        params=params,
        decoder_step=JaxGraphSpec(name="decoder_step", fn=decoder_step, example_args=(x,), output_names=("hidden",)),
        output_dir=tmp_path / "bundle",
        model_id="decoder-step-attention",
    )
    graph = result.bundle.graphs["decoder_step"].ir_graph
    attention_nodes = [node for node in graph.nodes.values() if node.op == "attention"]

    assert "use_internal_kv_cache" not in graph.meta
    assert len(attention_nodes) == 1
    assert result.bundle.graphs["decoder_step"].graph.cache_state_tensors == []
    _assert_close(result.bundle.execute("decoder_step", x)[0].numpy(), decoder_step(params, x))


def test_jax_generation_decoder_step_fuses_cross_attention(tmp_path: Path) -> None:
    params = {
        "wq": jnp.eye(16, dtype=jnp.float16),
    }
    x = jnp.asarray(np.linspace(-0.75, 0.75, 16, dtype=np.float16).reshape(1, 1, 16))
    key = jnp.asarray(np.linspace(-0.5, 0.5, num=1 * 2 * 4 * 8).reshape(1, 2, 4, 8), dtype=jnp.float16)
    value = jnp.asarray(np.linspace(0.75, -0.25, num=1 * 2 * 4 * 8).reshape(1, 2, 4, 8), dtype=jnp.float16)
    mask = jnp.asarray([[[[True, True, True, False]]]], dtype=jnp.bool_)

    def decoder_step(model_params, values, cross_k, cross_v, cross_mask):
        q = (values @ model_params["wq"]).reshape(1, 1, 2, 8).transpose(0, 2, 1, 3)
        scores = (q @ cross_k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.asarray(8.0, dtype=jnp.float32))
        scores = jnp.where(cross_mask, scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores, axis=-1).astype(jnp.float16)
        return (probs @ cross_v).transpose(0, 2, 1, 3).reshape(1, 1, 16)

    result = build_jax_generation_graph_bundle(
        params=params,
        decoder_step=JaxGraphSpec(
            name="decoder_step", fn=decoder_step, example_args=(x, key, value, mask), output_names=("hidden",),
        ),
        output_dir=tmp_path / "bundle",
        model_id="cross-attention-step",
    )
    graph = result.bundle.graphs["decoder_step"].ir_graph
    attention_nodes = [node for node in graph.nodes.values() if node.op == "attention"]

    assert len(attention_nodes) == 1
    assert attention_nodes[0].meta["rewritten_from"] == "jax_decoder_step_cross_attention"
    assert len(attention_nodes[0].inputs) == 4
    assert result.bundle.graphs["decoder_step"].graph.cache_state_tensors == []
    _assert_close(
        result.bundle.execute("decoder_step", x, key, value, mask)[0].numpy(),
        decoder_step(params, x, key, value, mask),
    )
