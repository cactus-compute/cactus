"""Minimal PyTorch Needle model for Cactus conversion."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from .configuration_needle import NeedleConfig


class NeedleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return x.to(dtype=dtype) * (1.0 + self.weight.to(dtype=dtype))


def _padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return (input_ids != int(pad_token_id))[:, None, None, :]


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()[None, None, :, :]


def _build_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (float(theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


def _rotary_tables(
    inv_freq: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    inv_freq = inv_freq[None, :, None].float().expand(batch_size, -1, 1).to(device)
    freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _rotary_tables_for_position_ids(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(position_ids.device)
    freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _add_clipped(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b, min=-65500.0, max=65500.0)


class NeedleAttention(nn.Module):
    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        kv_size = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = NeedleRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = NeedleRMSNorm(self.head_dim, config.rms_norm_eps)
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        batch, q_len, _ = hidden_states.shape
        kv_len = key_value_states.shape[1]
        q = self.q_proj(hidden_states).view(batch, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value_states).view(batch, kv_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(key_value_states).view(batch, kv_len, self.num_key_value_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
            enable_gqa=self.num_heads != self.num_key_value_heads,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(batch, q_len, self.hidden_size))

    def project_kv(self, key_value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, kv_len, _ = key_value_states.shape
        k = self.k_proj(key_value_states).view(batch, kv_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(key_value_states).view(batch, kv_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k)
        return k.contiguous(), v.contiguous()

    def forward_with_kv(
        self,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        batch, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        if rope is not None:
            cos, sin = rope
            q = _apply_rope(q, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
            enable_gqa=self.num_heads != self.num_key_value_heads,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(batch, q_len, self.hidden_size))


class NeedleEncoderLayer(nn.Module):
    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.input_layernorm = NeedleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = NeedleAttention(config)
        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normed = self.input_layernorm(hidden_states)
        attn = self.self_attn(normed, normed, attention_mask, rope)
        return _add_clipped(hidden_states, torch.sigmoid(self.attn_gate).to(dtype=attn.dtype) * attn)


class NeedleDecoderLayer(nn.Module):
    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.input_layernorm = NeedleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = NeedleAttention(config)
        self.self_attn_gate = nn.Parameter(torch.zeros(1))
        self.encoder_attn_layer_norm = NeedleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.encoder_attn = NeedleAttention(config)
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_mask: torch.Tensor,
        encoder_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normed = self.input_layernorm(hidden_states)
        attn = self.self_attn(normed, normed, self_mask, rope)
        hidden_states = _add_clipped(hidden_states, torch.sigmoid(self.self_attn_gate).to(dtype=attn.dtype) * attn)
        attn = self.encoder_attn(self.encoder_attn_layer_norm(hidden_states), encoder_hidden_states, encoder_mask, None)
        return _add_clipped(hidden_states, torch.sigmoid(self.cross_attn_gate).to(dtype=attn.dtype) * attn)


class NeedleEncoder(nn.Module):
    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([NeedleEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.final_norm = NeedleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = float(config.rope_theta)
        self.register_buffer("inv_freq", _build_inv_freq(self.head_dim, self.rope_theta), persistent=False)

    def reset_rope(self) -> None:
        self.inv_freq = _build_inv_freq(self.head_dim, self.rope_theta).to(device=self.inv_freq.device)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inv_freq = _build_inv_freq(self.head_dim, self.rope_theta).to(device=hidden_states.device)
        rope = _rotary_tables(
            inv_freq,
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.device,
            hidden_states.dtype,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rope)
        return self.final_norm(hidden_states)


class NeedleDecoder(nn.Module):
    def __init__(self, config: NeedleConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([NeedleDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = NeedleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = float(config.rope_theta)
        self.register_buffer("inv_freq", _build_inv_freq(self.head_dim, self.rope_theta), persistent=False)

    def reset_rope(self) -> None:
        self.inv_freq = _build_inv_freq(self.head_dim, self.rope_theta).to(device=self.inv_freq.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_mask: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        inv_freq = _build_inv_freq(self.head_dim, self.rope_theta).to(device=hidden_states.device)
        rope = _rotary_tables(
            inv_freq,
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.device,
            hidden_states.dtype,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, self_mask, encoder_mask, rope)
        return self.norm(hidden_states)


class NeedleModel(PreTrainedModel):
    config_class = NeedleConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = math.sqrt(float(config.hidden_size))
        self.encoder = NeedleEncoder(config)
        self.decoder = NeedleDecoder(config)
        self.post_init()
        self.reset_rope()

    def reset_rope(self) -> None:
        self.encoder.reset_rope()
        self.decoder.reset_rope()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> BaseModelOutput:
        decoder_input_ids = input_ids if decoder_input_ids is None else decoder_input_ids
        encoder_mask = _padding_mask(input_ids, self.config.pad_token_id)
        if attention_mask is not None:
            encoder_mask = encoder_mask & attention_mask[:, None, None, :].to(dtype=torch.bool)
        encoder_hidden = self.embed_tokens(input_ids) * self.embed_scale
        encoder_hidden = self.encoder(encoder_hidden, encoder_mask)

        self_mask = _causal_mask(decoder_input_ids.shape[1], decoder_input_ids.device)
        decoder_hidden = self.embed_tokens(decoder_input_ids) * self.embed_scale
        decoder_hidden = self.decoder(decoder_hidden, encoder_hidden, self_mask, encoder_mask)
        return BaseModelOutput(last_hidden_state=decoder_hidden)


class NeedleForCausalLM(PreTrainedModel):
    config_class = NeedleConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: NeedleConfig) -> None:
        super().__init__(config)
        self.model = NeedleModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.model.reset_rope()
        self.tie_weights()

    def get_encoder(self) -> NeedleEncoder:
        return self.model.encoder

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs,
        ).last_hidden_state
        return Seq2SeqLMOutput(logits=self.lm_head(hidden_states))

    def cactus_source_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_mask = _padding_mask(input_ids, self.config.pad_token_id)
        base_mask = base_mask & attention_mask[:, None, None, :].to(dtype=torch.bool)
        encoder_mask = base_mask.expand(
            -1,
            int(self.config.num_attention_heads),
            input_ids.shape[1],
            -1,
        ).contiguous()
        encoder_hidden = self.model.embed_tokens(input_ids) * self.model.embed_scale
        encoder_hidden = self.model.encoder(encoder_hidden, encoder_mask)
        decoder_mask = base_mask.expand(
            -1,
            int(self.config.num_attention_heads),
            1,
            -1,
        ).contiguous()
        return encoder_hidden, decoder_mask.to(dtype=encoder_hidden.dtype)

    def cactus_decoder_cross_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del encoder_attention_mask
        outputs: list[torch.Tensor] = []
        for layer in self.model.decoder.layers:
            k, v = layer.encoder_attn.project_kv(encoder_hidden_states)
            outputs.extend((k, v))
        return tuple(outputs)

    def cactus_decoder_step(
        self,
        decoder_input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        *cross_kv: torch.Tensor,
    ) -> torch.Tensor:
        encoder_attention_mask = encoder_attention_mask != 0
        hidden_states = self.model.embed_tokens(decoder_input_ids) * self.model.embed_scale
        inv_freq = _build_inv_freq(
            self.model.decoder.head_dim,
            self.model.decoder.rope_theta,
        ).to(device=hidden_states.device)
        rope = _rotary_tables_for_position_ids(
            inv_freq,
            position_ids.to(dtype=torch.long),
            hidden_states.dtype,
        )
        for layer_index, layer in enumerate(self.model.decoder.layers):
            normed = layer.input_layernorm(hidden_states)
            attn = layer.self_attn(normed, normed, None, rope)
            hidden_states = _add_clipped(hidden_states, torch.sigmoid(layer.self_attn_gate).to(dtype=attn.dtype) * attn)

            cross_attn = layer.encoder_attn.forward_with_kv(
                layer.encoder_attn_layer_norm(hidden_states),
                cross_kv[layer_index * 2],
                cross_kv[layer_index * 2 + 1],
                encoder_attention_mask,
                None,
            )
            hidden_states = _add_clipped(hidden_states, torch.sigmoid(layer.cross_attn_gate).to(dtype=cross_attn.dtype) * cross_attn)
        hidden_states = self.model.decoder.norm(hidden_states)
        return self.lm_head(hidden_states)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, NeedleRMSNorm):
            nn.init.zeros_(module.weight)
