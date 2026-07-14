"""Needle encoder-decoder model (mirrors the files published on Cactus-Compute/needle)."""


def register_with_transformers() -> None:
    """Register the needle architecture with transformers' Auto* factories.

    Idempotent; imports torch/sentencepiece lazily so importing cactus stays light.
    """
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    from .configuration_needle import NeedleConfig
    from .modeling_needle import NeedleForCausalLM, NeedleModel
    from .tokenization_needle import NeedleTokenizer

    AutoConfig.register("needle", NeedleConfig, exist_ok=True)
    AutoModel.register(NeedleConfig, NeedleModel, exist_ok=True)
    AutoModelForCausalLM.register(NeedleConfig, NeedleForCausalLM, exist_ok=True)
    AutoModelForSeq2SeqLM.register(NeedleConfig, NeedleForCausalLM, exist_ok=True)
    AutoTokenizer.register(NeedleConfig, slow_tokenizer_class=NeedleTokenizer, exist_ok=True)
