from __future__ import annotations

from cactus.transpile.weight_binding import resolve_weight_binding


def test_resolve_weight_binding_falls_back_to_tied_token_embeddings_for_lm_head(tmp_path):
    (tmp_path / "token_embeddings.weights").write_bytes(b"")

    binding = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="module.model.lm_head.weight",
    )

    assert binding is not None
    assert binding.path.endswith("token_embeddings.weights")
    assert binding.kind == "weight"
