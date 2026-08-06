from __future__ import annotations

import torch

from cactus.convert.model_adapters.adapters import adapter_for_family


def test_lfm2_vl_adapter_selects_runtime_safe_model_class():
    from transformers import Lfm2VlForConditionalGeneration

    adapter = adapter_for_family("lfm2")
    cfg = {"model_type": "lfm2", "architectures": ["Lfm2VlForConditionalGeneration"]}
    assert adapter.model_class(cfg) is Lfm2VlForConditionalGeneration


def test_lfm2_processor_fallback_handles_tokenizers_backend(tmp_path):
    import json

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Lfm2VlProcessor

    tokenizer = Tokenizer(WordLevel({"<|pad|>": 0, "<|startoftext|>": 1, "<|im_end|>": 2, "<image>": 3, "hello": 4}, unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "bos_token": "<|startoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|pad|>",
                "image_token": "<image>",
                "image_start_token": "<image>",
                "image_end_token": "<image>",
                "image_thumbnail": "<image>",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "Lfm2VlImageProcessorFast",
                "do_resize": True,
                "size": {"height": 512, "width": 512},
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "do_pad": True,
                "data_format": "channels_first",
            }
        ),
        encoding="utf-8",
    )

    processor = adapter_for_family("lfm2").load_processor(str(tmp_path))
    assert isinstance(processor, Lfm2VlProcessor)
    assert processor.image_token == "<image>"
    assert processor.image_token_id == 3


def test_lfm2_moe_packed_gate_up_splits_into_runtime_experts():
    adapter = adapter_for_family("lfm2")
    tensor = torch.arange(2 * 6 * 4, dtype=torch.float16).reshape(2, 6, 4)

    match = adapter.name_tensor(
        "model.layers.5.feed_forward.experts.gate_up_proj",
        tensor,
        num_layers=24,
    )
    emissions = adapter.expand_tensor(match, tensor)

    assert [emission.output_name for emission in emissions] == [
        "layer_5_moe_expert_0_w1.weights",
        "layer_5_moe_expert_0_w3.weights",
        "layer_5_moe_expert_1_w1.weights",
        "layer_5_moe_expert_1_w3.weights",
    ]
    assert torch.equal(emissions[0].tensor, tensor[0, :3, :])
    assert torch.equal(emissions[1].tensor, tensor[0, 3:, :])
    assert "model.layers.5.feed_forward.w1_weights.0" in emissions[0].source_names
    assert "model.layers.5.feed_forward.w3_weights.1" in emissions[3].source_names


def test_lfm2_moe_packed_down_splits_into_runtime_experts():
    adapter = adapter_for_family("lfm2")
    tensor = torch.arange(2 * 4 * 3, dtype=torch.float16).reshape(2, 4, 3)

    match = adapter.name_tensor(
        "model.layers.5.feed_forward.experts.down_proj",
        tensor,
        num_layers=24,
    )
    emissions = adapter.expand_tensor(match, tensor)

    assert [emission.output_name for emission in emissions] == [
        "layer_5_moe_expert_0_w2.weights",
        "layer_5_moe_expert_1_w2.weights",
    ]
    assert torch.equal(emissions[0].tensor, tensor[0])
    assert torch.equal(emissions[1].tensor, tensor[1])
    assert "model.layers.5.feed_forward.w2_weights.0" in emissions[0].source_names
