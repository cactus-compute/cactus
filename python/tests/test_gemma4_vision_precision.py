from __future__ import annotations

import torch

import cactus.transpile.model_adapters as model_adapters
from cactus.transpile.capture_pytorch import capture_model


class _PatchEmbedder(torch.nn.Module):
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        return pixel_values


class _VisionTower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedder = _PatchEmbedder()
        self.encoder = torch.nn.Identity()


class _EmbedVision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_projection = torch.nn.Linear(3, 2).half()
        self.eps = 1e-6


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = _VisionTower()
        self.embed_vision = _EmbedVision()


class _NativeLikeVisionWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor) -> torch.Tensor:
        return model_adapters._gemma4_compute_native_like_image_features(
            self.backbone,
            pixel_values,
            pixel_position_ids,
        )


def test_gemma4_native_like_vision_projection_keeps_weight_precision(monkeypatch) -> None:
    def fake_hidden_states(
        _vision_encoder: torch.nn.Module,
        inputs_embeds: torch.Tensor,
        _attention_mask: torch.Tensor,
        _pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return inputs_embeds

    def fake_pool(
        _vision_tower: torch.nn.Module,
        vision_hidden: torch.Tensor,
        _pixel_position_ids: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        return vision_hidden.reshape(-1, vision_hidden.shape[-1]).float()

    monkeypatch.setattr(model_adapters, "_gemma4_vision_encoder_hidden_states", fake_hidden_states)
    monkeypatch.setattr(model_adapters, "_gemma4_pool_vision_hidden_native_like", fake_pool)

    captured = capture_model(
        _NativeLikeVisionWrapper().eval(),
        (
            torch.randn(1, 2, 3, dtype=torch.float16),
            torch.zeros(1, 2, 2, dtype=torch.long),
        ),
        strict=False,
    )

    bad_casts = []
    for node_id in captured.ir_graph.order:
        node = captured.ir_graph.nodes[node_id]
        if node.op != "precision_cast":
            continue
        if any("embed_vision_embedding_projection" in input_id for input_id in node.inputs):
            bad_casts.append(node.id)

    assert bad_casts == []
