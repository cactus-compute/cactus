from __future__ import annotations

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
