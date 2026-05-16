from __future__ import annotations

import torch

from cactus.transpile.model_adapters import build_component_module_specs


class _FakeGemma4(torch.nn.Module):
    family = "gemma4"

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()


def test_gemma4_target_cached_verifier_components_are_explicit() -> None:
    specs = build_component_module_specs(
        _FakeGemma4(),
        task="causal_lm_logits",
        named_tensors={
            "input_ids": torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
        },
        weights_dir="/tmp/cactus-weights",
        components=("decoder_prefill_chunk", "decoder_step", "decoder_verify_m2", "decoder_verify_m3", "decoder_verify_m4"),
    )

    assert specs is not None
    by_name = {spec.component: spec for spec in specs}
    assert tuple(by_name) == ("decoder_prefill_chunk", "decoder_step", "decoder_verify_m2", "decoder_verify_m3", "decoder_verify_m4")
    assert by_name["decoder_prefill_chunk"].input_keys == ("input_ids", "position_ids")
    assert by_name["decoder_prefill_chunk"].output_keys == ("verifier_logits",)
    assert by_name["decoder_prefill_chunk"].graph_meta["use_internal_kv_cache"] is True
    assert by_name["decoder_prefill_chunk"].graph_meta["prefill_chunk_size"] == 4
    assert by_name["decoder_step"].input_keys == ("input_ids", "position_ids")
    assert by_name["decoder_step"].output_keys == ("verifier_logits",)
    assert by_name["decoder_step"].graph_meta["use_internal_kv_cache"] is True
    assert by_name["decoder_step"].graph_meta["verifier_width"] == 1
    assert by_name["decoder_step"].example_inputs[0].shape == (1, 1)
    assert by_name["decoder_step"].example_inputs[1].shape == (1, 1)

    for verifier_width in (2, 3, 4):
        component = by_name[f"decoder_verify_m{verifier_width}"]
        assert component.input_keys == ("input_ids", "position_ids")
        assert component.output_keys == ("verifier_logits",)
        assert component.graph_meta["use_internal_kv_cache"] is True
        assert component.graph_meta["verifier_width"] == verifier_width
        assert component.example_inputs[0].shape == (1, verifier_width)
        assert component.example_inputs[1].shape == (1, verifier_width)
