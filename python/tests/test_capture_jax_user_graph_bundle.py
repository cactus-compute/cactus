from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.transpile.capture_jax import capture_jax_graphs
from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_user_graph_bundle


def _write_fp16_weights(weights_dir: Path, params: dict[str, object]) -> None:
    for name, value in params.items():
        save_tensor_with_header(np.asarray(value), weights_dir / f"{name}.weights", precision="FP16")
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


def _user_supplied_jax_export():
    key = jax.random.PRNGKey(11)
    keys = jax.random.split(key, 7)
    params = {
        "encoder_w": jax.random.normal(keys[0], (6, 8), dtype=jnp.float16) * jnp.float16(0.05),
        "encoder_b": jax.random.normal(keys[1], (8,), dtype=jnp.float16) * jnp.float16(0.01),
        "decoder_w": jax.random.normal(keys[2], (5, 8), dtype=jnp.float16) * jnp.float16(0.05),
        "position_w": jax.random.normal(keys[3], (1, 8), dtype=jnp.float16) * jnp.float16(0.01),
        "out_w": jax.random.normal(keys[4], (8, 7), dtype=jnp.float16) * jnp.float16(0.05),
        "out_b": jax.random.normal(keys[5], (7,), dtype=jnp.float16) * jnp.float16(0.01),
    }

    source_features = jax.random.normal(keys[6], (1, 3, 6), dtype=jnp.float16)
    prefill_features = jnp.asarray(
        [[[0.2, -0.1, 0.4, 0.0, 0.3], [0.0, 0.1, -0.2, 0.5, -0.3]]],
        dtype=jnp.float16,
    )
    step_feature = prefill_features[:, :1, :]
    position = jnp.asarray([[1.0]], dtype=jnp.float16)

    def encoder_fn(model_params, source):
        encoded = jnp.matmul(source, model_params["encoder_w"]) + model_params["encoder_b"]
        return encoded * jax.nn.sigmoid(encoded)

    def decoder_prefill_fn(model_params, target_features, encoder_out):
        context = jnp.mean(encoder_out, axis=1, keepdims=True)
        hidden = jnp.matmul(target_features, model_params["decoder_w"]) + context
        hidden = hidden * jax.nn.sigmoid(hidden)
        return jnp.matmul(hidden, model_params["out_w"]) + model_params["out_b"]

    def decoder_step_fn(model_params, target_feature, step_position, encoder_out):
        context = jnp.mean(encoder_out, axis=1, keepdims=True)
        position_embed = step_position[:, :, None] * model_params["position_w"]
        hidden = jnp.matmul(target_feature, model_params["decoder_w"]) + context + position_embed
        hidden = hidden * jax.nn.sigmoid(hidden)
        return jnp.matmul(hidden, model_params["out_w"]) + model_params["out_b"]

    specs = (
        JaxGraphSpec(
            name="encoder",
            role="encoder",
            fn=encoder_fn,
            example_args=(source_features,),
            input_names=("source_features",),
            output_names=("encoder_out",),
        ),
        JaxGraphSpec(
            name="decoder_prefill",
            role="decoder_prefill",
            fn=decoder_prefill_fn,
            example_args=(prefill_features, encoder_fn(params, source_features)),
            input_names=("target_features", "encoder_out"),
            output_names=("logits",),
        ),
        JaxGraphSpec(
            name="decoder_step",
            role="decoder_step",
            fn=decoder_step_fn,
            example_args=(step_feature, position, encoder_fn(params, source_features)),
            input_names=("target_feature", "position", "encoder_out"),
            output_names=("logits",),
            graph_meta={
                "component": "decoder_step",
                "use_internal_kv_cache": True,
                "max_cache_seq_len": 16,
            },
        ),
    )
    examples = {
        "source_features": source_features,
        "prefill_features": prefill_features,
        "step_feature": step_feature,
        "position": position,
    }
    fns = {
        "encoder": encoder_fn,
        "decoder_prefill": decoder_prefill_fn,
        "decoder_step": decoder_step_fn,
    }
    return params, specs, examples, fns


def test_user_supplied_jax_graph_bundle_with_shared_weights(tmp_path: Path) -> None:
    params, specs, examples, fns = _user_supplied_jax_export()
    _write_fp16_weights(tmp_path, params)

    bundle = capture_jax_graphs(
        params,
        specs,
        weights_dir=str(tmp_path),
        graph_meta={
            "frontend": "jax",
            "adapter_family": "generic",
            "graph_family": "user_supplied",
        },
    )

    encoder_out = bundle.execute("encoder", examples["source_features"])[0].numpy().astype(np.float32)
    expected_encoder = np.asarray(fns["encoder"](params, examples["source_features"])).astype(np.float32)

    prefill_logits = bundle.execute(
        "decoder_prefill",
        examples["prefill_features"],
        encoder_out,
    )[0].numpy().astype(np.float32)
    expected_prefill = np.asarray(
        fns["decoder_prefill"](params, examples["prefill_features"], expected_encoder)
    ).astype(np.float32)

    step_logits = bundle.execute(
        "decoder_step",
        examples["step_feature"],
        examples["position"],
        encoder_out,
    )[0].numpy().astype(np.float32)
    expected_step = np.asarray(
        fns["decoder_step"](params, examples["step_feature"], examples["position"], expected_encoder)
    ).astype(np.float32)

    assert set(bundle.graphs) == {"encoder", "decoder_prefill", "decoder_step"}
    assert np.max(np.abs(encoder_out - expected_encoder)) < 8e-2
    assert np.max(np.abs(prefill_logits - expected_prefill)) < 8e-2
    assert np.max(np.abs(step_logits - expected_step)) < 8e-2
    assert bundle.graphs["decoder_step"].ir_graph.meta["use_internal_kv_cache"] is True
    assert bundle.graphs["decoder_step"].ir_graph.meta["max_cache_seq_len"] == 16

    bound_sources = {
        binding["source_name"]
        for graph in bundle.graphs.values()
        for binding in graph.graph.bound_constant_bindings
    }
    assert bound_sources == set(params)


def test_generic_jax_user_graph_bundle_writer_supports_unrelated_model(tmp_path: Path) -> None:
    params = {
        "proj_w": jnp.asarray([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.6]], dtype=jnp.float16),
        "proj_b": jnp.asarray([0.01, -0.02], dtype=jnp.float16),
    }
    x = jnp.asarray([[[1.0, -2.0, 0.5], [0.25, 0.5, -0.75]]], dtype=jnp.float16)

    def tiny_feature_graph(model_params, features):
        return jnp.tanh(jnp.matmul(features, model_params["proj_w"]) + model_params["proj_b"])

    result = build_jax_user_graph_bundle(
        params=params,
        output_dir=tmp_path / "bundle",
        model_id="tiny_unrelated_jax_model",
        task="feature-projection",
        family="tiny",
        specs=(
            JaxGraphSpec(
                name="feature_projector",
                role="encoder",
                fn=tiny_feature_graph,
                example_args=(x,),
                input_names=("features",),
                output_names=("projected",),
            ),
        ),
    )

    got = result.bundle.execute("feature_projector", x)[0].numpy().astype(np.float32)
    expected = np.asarray(tiny_feature_graph(params, x)).astype(np.float32)
    manifest = json.loads(result.components_manifest_path.read_text())

    assert np.max(np.abs(got - expected)) < 5e-2
    assert (tmp_path / "bundle/components/feature_projector/graph.cactus").exists()
    assert (tmp_path / "bundle/weights/weights_manifest.json").exists()
    assert manifest["model_source"] == "jax_user_graph"
    assert manifest["component_order"] == ["feature_projector"]


def test_official_tiny_gemma_jax_user_graph_bundle_when_available(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", "/private/tmp")
    if not hasattr(np, "float128"):
        monkeypatch.setattr(np, "float128", np.longdouble, raising=False)

    pytest.importorskip("gemma")
    from gemma.gm.nn import _config
    from gemma.gm.nn import _modules
    from gemma.gm.nn import _transformer

    config = _config.TransformerConfig(
        num_embed=64,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        final_logit_softcap=None,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attention_types=(_modules.AttentionType.GLOBAL,),
        use_qk_norm=False,
    )
    model = _transformer.Transformer(config=config, dtype=jnp.float16)
    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
    attention_mask = jnp.tril(jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_))[None, :, :]
    params = model.init(jax.random.PRNGKey(0), tokens, positions=positions, attention_mask=attention_mask)["params"]

    def logits_fn(model_params, token_ids, pos, attn_mask):
        return model.apply(
            {"params": model_params},
            token_ids,
            positions=pos,
            attention_mask=attn_mask,
        ).logits

    result = build_jax_user_graph_bundle(
        params=params,
        output_dir=tmp_path / "gemma_bundle",
        model_id="official_gemma_tiny_random",
        task="causal-lm-logits",
        family="gemma",
        specs=(
            JaxGraphSpec(
                name="decoder",
                role="decoder",
                fn=logits_fn,
                example_args=(tokens, positions, attention_mask),
                input_names=("tokens", "positions", "attention_mask"),
                output_names=("logits",),
            ),
        ),
    )

    got = result.bundle.execute("decoder", tokens, positions, attention_mask)[0].numpy().astype(np.float32)
    expected = np.asarray(logits_fn(params, tokens, positions, attention_mask)).astype(np.float32)
    manifest = json.loads(result.components_manifest_path.read_text())

    assert np.max(np.abs(got - expected)) < 2e-3
    assert np.mean(np.abs(got - expected)) < 2e-4
    assert manifest["family"] == "gemma"
    assert manifest["component_order"] == ["decoder"]
