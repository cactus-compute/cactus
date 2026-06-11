from __future__ import annotations

from types import SimpleNamespace

import torch

from cactus.transpile.model_adapters import build_component_module_specs


class ToyNeedle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="needle",
            decoder_start_token_id=1,
            pad_token_id=0,
        )

    def cactus_source_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 2).contiguous()
        return hidden, attention_mask[:, None, None, :].to(dtype=torch.bool)

    def cactus_decoder_cross_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del encoder_attention_mask
        batch, seq, _ = encoder_hidden_states.shape
        return tuple(torch.zeros((batch, seq, 1, 2), dtype=torch.float32) for _ in range(4))

    def cactus_decoder_step(
        self,
        decoder_input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        *cross_kv: torch.Tensor,
    ) -> torch.Tensor:
        del position_ids, encoder_attention_mask, cross_kv
        return torch.zeros((decoder_input_ids.shape[0], decoder_input_ids.shape[1], 8), dtype=torch.float32)


def test_needle_encoder_cross_kv_route_specs_are_metadata_driven() -> None:
    specs = build_component_module_specs(
        ToyNeedle(),
        task="causal_lm_logits",
        named_tensors={"input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long)},
    )

    assert specs is not None
    assert [spec.component for spec in specs] == ["source_encoder", "decoder_cross_kv", "decoder_step"]

    source, cross_kv, decoder_step = specs
    assert source.input_keys == ("input_ids", "attention_mask")
    assert source.output_keys == ("encoder_hidden_states", "encoder_attention_mask")
    assert source.metadata["runtime_route"] == "encoder_cross_kv_decoder_step"
    assert source.metadata["runtime_role"] == "source_encoder"
    assert source.metadata["source_kind"] == "text_tokens"

    assert cross_kv.input_keys == ("encoder_hidden_states", "encoder_attention_mask")
    assert cross_kv.output_keys == ("cross_k_0", "cross_v_0", "cross_k_1", "cross_v_1")
    assert cross_kv.metadata["runtime_role"] == "decoder_cross_kv"

    assert decoder_step.input_keys == (
        "decoder_input_ids",
        "position_ids",
        "encoder_attention_mask",
        "cross_k_0",
        "cross_v_0",
        "cross_k_1",
        "cross_v_1",
    )
    assert decoder_step.output_keys == ("logits",)
    assert decoder_step.metadata["runtime_role"] == "decoder_step"
    assert decoder_step.graph_meta["use_internal_kv_cache"] is True
