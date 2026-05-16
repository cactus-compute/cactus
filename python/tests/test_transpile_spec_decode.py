from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cactus.transpile.model_adapters import Gemma4AssistantMTPAdapter
from cactus.transpile.model_adapters import Gemma4CausalLMLogitsAdapter
from cactus.transpile.model_adapters import Gemma4SpecDecodeDecoderAdapter
from cactus.transpile.model_adapters import build_component_module_specs
from cactus.transpile.model_adapters import _resolve_model_pad_token_id
from cactus.transpile import hf_model
from cactus.transpile.spec_decode import AssistantSpecDecodeAdapter
from cactus.transpile.spec_decode import TargetSpecDecodeAdapter
from cactus.transpile.spec_decode import validate_spec_decode_manifest


def _manifest() -> dict[str, object]:
    return {
        "spec_decode": {
            "version": 1,
            "method": "single_position_mtp",
            "target": {
                "verifier_logits": "target_logits",
                "target_hidden_state": "target_hidden",
                "target_token_embedding": "target_embedding",
                "shared_kv": {
                    "full_attention": {
                        "key": "target_full_key",
                        "value": "target_full_value",
                    },
                    "sliding_attention": {
                        "key": "target_sliding_key",
                        "value": "target_sliding_value",
                    },
                },
            },
            "assistant": {
                "current_token_embedding": "assistant_token_embedding",
                "previous_target_hidden": "assistant_prev_hidden",
                "position": "assistant_position",
                "shared_kv": {
                    "full_attention": {
                        "key": "assistant_full_key",
                        "value": "assistant_full_value",
                    },
                    "sliding_attention": {
                        "key": "assistant_sliding_key",
                        "value": "assistant_sliding_value",
                    },
                },
                "logits_output": "assistant_logits_out",
                "next_hidden_output": "assistant_hidden_out",
            },
        }
    }


def test_valid_spec_decode_manifest_loads() -> None:
    validate_spec_decode_manifest(_manifest())


def test_missing_target_role_fails_clearly() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["target"]["target_token_embedding"] = ""
    with pytest.raises(ValueError, match="target.target_token_embedding"):
        validate_spec_decode_manifest(manifest)


def test_missing_assistant_role_fails_clearly() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["assistant"]["logits_output"] = ""
    with pytest.raises(ValueError, match="assistant.logits_output"):
        validate_spec_decode_manifest(manifest)


def test_unsupported_method_fails_clearly() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["method"] = "tree"
    with pytest.raises(ValueError, match="unsupported spec_decode method"):
        validate_spec_decode_manifest(manifest)


def test_unsupported_version_fails_clearly() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["version"] = 2
    with pytest.raises(ValueError, match="unsupported spec_decode version"):
        validate_spec_decode_manifest(manifest)


def test_token_only_assistant_manifest_is_rejected_for_mtp() -> None:
    manifest = _manifest()
    assistant = manifest["spec_decode"]["assistant"]
    assistant.pop("current_token_embedding")
    assistant["current_token"] = "assistant_token"
    with pytest.raises(ValueError, match="assistant.current_token_embedding"):
        validate_spec_decode_manifest(manifest)


def test_shared_kv_roles_must_be_named() -> None:
    manifest = _manifest()
    manifest["spec_decode"]["target"].pop("shared_kv")
    manifest["spec_decode"]["target"]["assistant_shared_state_tensors"] = ["shared_a"]
    with pytest.raises(ValueError, match="target.shared_kv.full_attention.key"):
        validate_spec_decode_manifest(manifest)


def test_target_adapter_emits_expected_role_names_and_shapes() -> None:
    logits = np.zeros((1, 3, 5), dtype=np.float32)
    hidden = np.zeros((1, 3, 7), dtype=np.float32)
    embedding = np.zeros((1, 7), dtype=np.float32)
    full_key = np.zeros((1, 2), dtype=np.float32)
    full_value = np.zeros((1, 2), dtype=np.float32)
    sliding_key = np.zeros((1, 3), dtype=np.float32)
    sliding_value = np.zeros((1, 3), dtype=np.float32)
    roles = TargetSpecDecodeAdapter(_manifest()).emit_roles(
        {
            "target_logits": logits,
            "target_hidden": hidden,
            "target_embedding": embedding,
            "target_full_key": full_key,
            "target_full_value": full_value,
            "target_sliding_key": sliding_key,
            "target_sliding_value": sliding_value,
        }
    )
    assert roles["verifier_logits"].shape == (1, 3, 5)
    assert roles["target_hidden_state"].shape == (1, 3, 7)
    assert roles["target_token_embedding"].shape == (1, 7)
    assert roles["shared_kv"]["full_attention"]["key"].shape == (1, 2)
    assert roles["shared_kv"]["full_attention"]["value"].shape == (1, 2)
    assert roles["shared_kv"]["sliding_attention"]["key"].shape == (1, 3)
    assert roles["shared_kv"]["sliding_attention"]["value"].shape == (1, 3)


def test_assistant_adapter_is_manifest_driven_not_positional() -> None:
    adapter = AssistantSpecDecodeAdapter(_manifest())
    prepared = adapter.prepare_inputs(
        {
            "position": np.array([4]),
            "shared_kv": {
                "full_attention": {
                    "key": np.array([[1.0]], dtype=np.float32),
                    "value": np.array([[2.0]], dtype=np.float32),
                },
                "sliding_attention": {
                    "key": np.array([[3.0]], dtype=np.float32),
                    "value": np.array([[4.0]], dtype=np.float32),
                },
            },
            "previous_target_hidden": np.zeros((1, 7), dtype=np.float32),
            "current_token_embedding": np.zeros((1, 7), dtype=np.float32),
        }
    )
    assert tuple(prepared) == (
        "assistant_token_embedding",
        "assistant_prev_hidden",
        "assistant_position",
        "assistant_full_key",
        "assistant_full_value",
        "assistant_sliding_key",
        "assistant_sliding_value",
    )

    roles = adapter.emit_roles(
        {
            "assistant_hidden_out": np.zeros((1, 7), dtype=np.float32),
            "assistant_logits_out": np.zeros((1, 5), dtype=np.float32),
        }
    )
    assert roles["assistant_logits"].shape == (1, 5)
    assert roles["next_assistant_hidden"].shape == (1, 7)


class _FakeGemma4(torch.nn.Module):
    family = "gemma4"

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()


class _FakeGemma4AssistantFamily(torch.nn.Module):
    __module__ = "transformers.models.gemma4_assistant.modeling_gemma4_assistant"

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            backbone_hidden_size=1536,
            text_config=SimpleNamespace(global_head_dim=512, head_dim=256),
        )


def test_gemma4_target_decoder_adapter_emits_real_shared_kv(monkeypatch) -> None:
    logits = torch.ones((1, 2, 5))
    hidden = torch.full((1, 2, 4), 2.0)
    full_key = torch.full((1, 1, 2, 4), 3.0)
    full_value = torch.full((1, 1, 2, 4), 4.0)
    sliding_key = torch.full((1, 1, 1, 4), 5.0)
    sliding_value = torch.full((1, 1, 1, 4), 6.0)

    def _fake_forward(self, input_ids):
        return (
            logits,
            [hidden],
            {
                "full_attention": (full_key, full_value),
                "sliding_attention": (sliding_key, sliding_value),
            },
        )

    monkeypatch.setattr(Gemma4CausalLMLogitsAdapter, "debug_forward_with_shared_kv", _fake_forward)

    outputs = Gemma4SpecDecodeDecoderAdapter(_FakeGemma4())(torch.zeros((1, 2), dtype=torch.long))

    assert outputs == (logits, hidden, full_key, full_value, sliding_key, sliding_value)


class _FakeAssistantOutput:
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor) -> None:
        self.logits = logits
        self.last_hidden_state = last_hidden_state


class _FakeGemma4Assistant(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        inputs_embeds = kwargs["inputs_embeds"]
        return _FakeAssistantOutput(
            logits=torch.ones((*inputs_embeds.shape[:2], 5), dtype=inputs_embeds.dtype),
            last_hidden_state=inputs_embeds[..., : inputs_embeds.shape[-1] // 2],
        )


def test_gemma4_assistant_adapter_matches_hf_single_position_contract() -> None:
    model = _FakeGemma4Assistant()
    adapter = Gemma4AssistantMTPAdapter(model)
    current_embedding = torch.full((1, 1, 4), 2.0)
    previous_hidden = torch.full((1, 1, 4), 3.0)
    position = torch.tensor([[7]])
    full_key = torch.full((1, 1, 2, 4), 4.0)
    full_value = torch.full((1, 1, 2, 4), 5.0)
    sliding_key = torch.full((1, 1, 1, 4), 6.0)
    sliding_value = torch.full((1, 1, 1, 4), 7.0)

    logits, next_hidden = adapter(
        current_embedding,
        previous_hidden,
        position,
        full_key,
        full_value,
        sliding_key,
        sliding_value,
    )

    assert logits.shape == (1, 1, 5)
    assert next_hidden.shape == (1, 1, 4)
    call = model.calls[0]
    assert torch.equal(call["inputs_embeds"], torch.cat([current_embedding, previous_hidden], dim=-1))
    assert call["position_ids"] is position
    assert call["use_cache"] is False
    assert call["attention_mask"] is None
    shared_kv = call["shared_kv_states"]
    assert torch.equal(shared_kv["full_attention"][0], full_key)
    assert torch.equal(shared_kv["full_attention"][1], full_value)
    assert torch.equal(shared_kv["sliding_attention"][0], sliding_key)
    assert torch.equal(shared_kv["sliding_attention"][1], sliding_value)


def test_resolve_model_pad_token_id_accepts_list_valued_fallbacks() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=None, eos_token_id=[1, 106], bos_token_id=None),
        generation_config=None,
    )

    assert _resolve_model_pad_token_id(model) == 1


def test_gemma4_target_spec_decode_components_expose_logical_roles() -> None:
    specs = build_component_module_specs(
        _FakeGemma4(),
        task="causal_lm_logits",
        named_tensors={
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "current_token_ids": torch.zeros((1, 1), dtype=torch.long),
        },
        weights_dir="/tmp/cactus-weights",
        components=("decoder", "target_embedding"),
    )

    assert specs is not None
    by_name = {spec.component: spec for spec in specs}
    assert tuple(by_name) == ("decoder", "target_embedding")
    assert by_name["decoder"].output_keys == (
        "verifier_logits",
        "target_hidden_state",
        "shared_kv.full_attention.key",
        "shared_kv.full_attention.value",
        "shared_kv.sliding_attention.key",
        "shared_kv.sliding_attention.value",
    )
    assert by_name["target_embedding"].input_keys == ("current_token_ids",)
    assert by_name["target_embedding"].output_keys == ("target_token_embedding",)
    assert by_name["decoder"].graph_meta["weights_dir"] == "/tmp/cactus-weights"
    assert by_name["target_embedding"].graph_meta["weights_dir"] == "/tmp/cactus-weights"


def test_gemma4_assistant_spec_decode_component_exposes_hf_mtp_roles() -> None:
    specs = build_component_module_specs(
        _FakeGemma4(),
        task="causal_lm_logits",
        named_tensors={
            "current_token_embedding": torch.zeros((1, 1, 4)),
            "previous_target_hidden": torch.zeros((1, 1, 4)),
            "position": torch.zeros((1, 1), dtype=torch.long),
            "shared_kv.full_attention.key": torch.zeros((1, 1, 2, 4)),
            "shared_kv.full_attention.value": torch.zeros((1, 1, 2, 4)),
            "shared_kv.sliding_attention.key": torch.zeros((1, 1, 2, 4)),
            "shared_kv.sliding_attention.value": torch.zeros((1, 1, 2, 4)),
        },
        components=("assistant",),
    )

    assert specs is not None
    assert len(specs) == 1
    assistant = specs[0]
    assert assistant.component == "assistant"
    assert assistant.input_keys == (
        "current_token_embedding",
        "previous_target_hidden",
        "position",
        "shared_kv.full_attention.key",
        "shared_kv.full_attention.value",
        "shared_kv.sliding_attention.key",
        "shared_kv.sliding_attention.value",
    )
    assert assistant.output_keys == ("logits_output", "next_hidden_output")


def test_gemma4_assistant_transformers_family_uses_mtp_component_specs() -> None:
    specs = build_component_module_specs(
        _FakeGemma4AssistantFamily(),
        task="causal_lm_logits",
        named_tensors={
            "current_token_embedding": torch.zeros((1, 1, 4)),
            "previous_target_hidden": torch.zeros((1, 1, 4)),
            "position": torch.zeros((1, 1), dtype=torch.long),
            "shared_kv.full_attention.key": torch.zeros((1, 1, 2, 4)),
            "shared_kv.full_attention.value": torch.zeros((1, 1, 2, 4)),
            "shared_kv.sliding_attention.key": torch.zeros((1, 1, 2, 4)),
            "shared_kv.sliding_attention.value": torch.zeros((1, 1, 2, 4)),
        },
        components=("assistant",),
    )

    assert specs is not None
    assert [spec.component for spec in specs] == ["assistant"]


def test_gemma4_assistant_default_examples_match_hf_projection_width() -> None:
    specs = build_component_module_specs(
        _FakeGemma4AssistantFamily(),
        task="causal_lm_logits",
        named_tensors={},
        components=("assistant",),
    )

    assert specs is not None
    inputs = specs[0].example_inputs
    assert inputs[0].shape == (1, 1, 1536)
    assert inputs[1].shape == (1, 1, 1536)
    assert inputs[3].shape == (1, 1, 1, 512)
    assert inputs[5].shape == (1, 1, 1, 256)


def test_component_io_roles_are_logical_not_output_order_dependent() -> None:
    specs = build_component_module_specs(
        _FakeGemma4(),
        task="causal_lm_logits",
        named_tensors={
            "current_token_embedding": torch.zeros((1, 1, 4)),
            "previous_target_hidden": torch.zeros((1, 1, 4)),
            "position": torch.zeros((1, 1), dtype=torch.long),
            "shared_kv.full_attention.key": torch.zeros((1, 1, 2, 4)),
            "shared_kv.full_attention.value": torch.zeros((1, 1, 2, 4)),
            "shared_kv.sliding_attention.key": torch.zeros((1, 1, 2, 4)),
            "shared_kv.sliding_attention.value": torch.zeros((1, 1, 2, 4)),
        },
        components=("assistant",),
    )

    assert specs is not None
    assistant = specs[0]
    assert assistant.output_keys[0] == "logits_output"
    assert assistant.output_keys[1] == "next_hidden_output"


def test_gemma4_causal_component_pipeline_auto_requires_explicit_components() -> None:
    assert not hf_model._should_use_component_pipeline(
        mode="auto",
        task="causal_lm_logits",
        has_component_specs=True,
        requested_components=None,
    )
    assert hf_model._should_use_component_pipeline(
        mode="auto",
        task="causal_lm_logits",
        has_component_specs=True,
        requested_components=("decoder", "target_embedding"),
    )
