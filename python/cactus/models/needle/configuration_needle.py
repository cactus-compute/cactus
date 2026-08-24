"""Hugging Face configuration for Needle."""

from __future__ import annotations

from transformers import PretrainedConfig


class NeedleConfig(PretrainedConfig):
    model_type = "needle"

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int | None = None,
        d_model: int = 512,
        num_attention_heads: int | None = None,
        num_heads: int = 8,
        num_key_value_heads: int | None = None,
        num_kv_heads: int = 4,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 8,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        unk_token_id: int = 3,
        decoder_start_token_id: int | None = None,
        tie_word_embeddings: bool = True,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        kwargs.pop("is_encoder_decoder", None)
        hidden_size = int(hidden_size if hidden_size is not None else d_model)
        num_attention_heads = int(num_attention_heads if num_attention_heads is not None else num_heads)
        num_key_value_heads = int(num_key_value_heads if num_key_value_heads is not None else num_kv_heads)
        decoder_start_token_id = eos_token_id if decoder_start_token_id is None else decoder_start_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            unk_token_id=unk_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=True,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.vocab_size = int(vocab_size)
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_heads = num_key_value_heads
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_hidden_layers = int(num_decoder_layers)
        self.rope_theta = float(rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)
        self.attention_head_dim = hidden_size // max(1, num_attention_heads)
