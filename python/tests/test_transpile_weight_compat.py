from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.transpile.weight_binding import WeightBinding
from cactus.transpile.weight_binding import resolve_weight_binding
from cactus.transpile.weight_compat import _open_cactus_tensor_file
from cactus.transpile.weight_compat import ensure_binding_compatible
from cactus.transpile.weight_compat import ensure_embedding_binding_compatible


def _write_grouped_int8_embedding(path: Path, rows: int, cols: int) -> None:
    rng = np.random.default_rng(1234)
    tensor = rng.standard_normal((rows, cols), dtype=np.float32)
    save_tensor_with_header(tensor, path, precision="INT8", model_type="generic")


def test_token_embedding_binding_upgrades_to_cq4(tmp_path: Path) -> None:
    source = tmp_path / "token_embeddings.weights"
    _write_grouped_int8_embedding(source, rows=8, cols=128)

    legacy_fp16 = tmp_path / "token_embeddings.fp16.weights"
    legacy_fp16.write_bytes(b"legacy")

    binding = WeightBinding(
        path=str(source),
        kind="embedding",
        source_name="model.embed_tokens.weight",
    )
    compat = ensure_embedding_binding_compatible(binding)

    assert compat.path.endswith(".cq4.weights")
    assert compat.kind == "embedding"
    assert not legacy_fp16.exists()

    opened = _open_cactus_tensor_file(compat.path)
    assert opened.precision == 6
    assert opened.shape == (8, 128)
    assert opened.scales is not None


def test_per_layer_embedding_binding_upgrades_to_cq2(tmp_path: Path) -> None:
    source = tmp_path / "embed_tokens_per_layer.weights"
    _write_grouped_int8_embedding(source, rows=8, cols=256)

    binding = WeightBinding(
        path=str(source),
        kind="embedding",
        source_name="model.language_model.embed_tokens_per_layer.weight",
    )
    compat = ensure_embedding_binding_compatible(binding)

    assert compat.path.endswith(".cq2.weights")

    opened = _open_cactus_tensor_file(compat.path)
    assert opened.precision == 4
    assert opened.shape == (8, 256)
    assert opened.scales is not None


def test_gemma4_per_layer_projection_binding_upgrades_legacy_int4(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    source = tmp_path / "per_layer_model_proj.weights"
    tensor = rng.standard_normal((8, 128), dtype=np.float32)
    save_tensor_with_header(tensor, source, precision="INT4", model_type="generic")

    binding = WeightBinding(
        path=str(source),
        kind="weight",
        source_name="module.model.model.language_model.per_layer_model_projection.weight",
    )
    compat = ensure_binding_compatible(binding, source_tensor=tensor)

    assert compat.path.endswith(".cq4.weights")
    opened = _open_cactus_tensor_file(compat.path)
    assert opened.precision == 6
    assert opened.shape == (8, 128)
    assert opened.scales is not None


def test_decoder_binding_prefers_existing_cq4_companion(tmp_path: Path) -> None:
    rng = np.random.default_rng(4321)
    source = tmp_path / "layer_0_attn_q.weights"
    tensor = rng.standard_normal((8, 128), dtype=np.float32)
    save_tensor_with_header(tensor, source, precision="INT4", model_type="generic")

    companion = tmp_path / "layer_0_attn_q.cq4.weights"
    save_tensor_with_header(tensor, companion, precision="FP16", model_type="generic")

    binding = WeightBinding(
        path=str(source),
        kind="weight",
        source_name="module.backbone.layers.0.self_attn.q_proj.weight",
    )
    compat = ensure_binding_compatible(binding, source_tensor=0)

    assert compat.path == str(companion)


def test_resolve_weight_binding_does_not_guess_legacy_filenames(tmp_path: Path) -> None:
    (tmp_path / "token_embeddings.weights").write_bytes(b"")
    (tmp_path / "layer_0_attn_q.weights").write_bytes(b"")
    (tmp_path / "embed_tokens_per_layer.weights").write_bytes(b"")
    (tmp_path / "per_layer_model_proj.weights").write_bytes(b"")

    assert (
        resolve_weight_binding(
            weights_dir=str(tmp_path),
            source_name="model.embed_tokens.weight",
        )
        is None
    )
    assert (
        resolve_weight_binding(
            weights_dir=str(tmp_path),
            source_name="model.layers.0.self_attn.q_proj.weight",
        )
        is None
    )
    assert (
        resolve_weight_binding(
            weights_dir=str(tmp_path),
            source_name="model.language_model.embed_tokens_per_layer.weight",
        )
        is None
    )
    assert (
        resolve_weight_binding(
            weights_dir=str(tmp_path),
            source_name="model.language_model.per_layer_model_projection.weight",
        )
        is None
    )


def test_resolve_weight_binding_prefers_exact_manifest_without_name_guessing(tmp_path: Path) -> None:
    (tmp_path / "exact.weights").write_bytes(b"")
    (tmp_path / "token_embeddings.weights").write_bytes(b"")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps(
            {
                "v_adapter_named_anything": {
                    "filename": "exact.weights",
                    "kind": "weight",
                }
            }
        )
    )

    exact = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="v_adapter_named_anything",
    )
    assert exact is not None
    assert exact.path.endswith("exact.weights")

    guessed = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="model.embed_tokens.weight",
    )
    assert guessed is None


def test_resolve_weight_binding_reads_convert_manifest_aliases(tmp_path: Path) -> None:
    (tmp_path / "decoder_attn_q.cq4.weights").write_bytes(b"")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps(
            {
                "weights": [
                    {
                        "source_name": "model.layers.0.self_attn.q_proj.weight",
                        "hf_name": "model.layers.0.self_attn.q_proj.weight",
                        "adapter_name": "backbone.layers.0.self_attn.q_proj.weight",
                        "source_names": ["adapter.backbone.layers.0.self_attn.q_proj.weight"],
                        "output_name": "decoder_attn_q.cq4.weights",
                        "component": "language",
                    }
                ]
            }
        )
    )

    binding = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="adapter.backbone.layers.0.self_attn.q_proj.weight",
    )

    assert binding is not None
    assert binding.path.endswith("decoder_attn_q.cq4.weights")


def test_resolve_weight_binding_expands_manifest_wrapper_prefixes(tmp_path: Path) -> None:
    (tmp_path / "layer_0_q.cq4.weights").write_bytes(b"")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps(
            {
                "model.layers.0.self_attn.q_proj.weight": {
                    "filename": "layer_0_q.cq4.weights",
                    "kind": "weight",
                }
            }
        )
    )

    binding = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="adapter.model.model.layers.0.self_attn.q_proj.weight",
    )

    assert binding is not None
    assert binding.path.endswith("layer_0_q.cq4.weights")


def test_resolve_weight_binding_expands_component_local_manifest_aliases(tmp_path: Path) -> None:
    (tmp_path / "encoder_layer_0_q.cq4.weights").write_bytes(b"")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps(
            {
                "encoder.layers.0.self_attn.q_proj.weight": {
                    "filename": "encoder_layer_0_q.cq4.weights",
                    "kind": "weight",
                }
            }
        )
    )

    binding = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="module.layers.0.self_attn.q_proj.weight",
    )

    assert binding is not None
    assert binding.path.endswith("encoder_layer_0_q.cq4.weights")


def test_resolve_weight_binding_expands_language_model_backbone_aliases(tmp_path: Path) -> None:
    (tmp_path / "layer_0_scalar.weights").write_bytes(b"")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps(
            {
                "model.language_model.layers.0.layer_scalar": {
                    "filename": "layer_0_scalar.weights",
                    "kind": "weight",
                }
            }
        )
    )

    binding = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="module.backbone.layers.0.layer_scalar",
    )

    assert binding is not None
    assert binding.path.endswith("layer_0_scalar.weights")


def test_resolve_weight_binding_expands_nested_model_tower_aliases(tmp_path: Path) -> None:
    (tmp_path / "vision_q.weights").write_bytes(b"")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps(
            {
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight": {
                    "filename": "vision_q.weights",
                    "kind": "weight",
                }
            }
        )
    )

    binding = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="module.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
    )

    assert binding is not None
    assert binding.path.endswith("vision_q.weights")


def test_rank5_materialized_constant_uses_cactus_tensor_file(tmp_path: Path) -> None:
    from cactus.transpile.hf_model import _write_cactus_constant_tensor
    from cactus.transpile.runtime_compat import Graph

    tensor = np.zeros((1, 2, 3, 4, 5), dtype=np.float16)
    path = tmp_path / "rank5.weights"

    _write_cactus_constant_tensor(output_path=path, value=tensor, precision=int(Graph.FP16))

    opened = _open_cactus_tensor_file(path)
    assert opened.shape == tensor.shape
    assert opened.precision == int(Graph.FP16)
