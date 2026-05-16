from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cactus.transpile.component_pipeline import ComponentModuleSpec
from cactus.transpile.audio_preprocess import prepare_native_parakeet_audio_features
from cactus.transpile.model_profiles import add_tensor_aliases
from cactus.transpile.model_profiles import PARAKEET_TDT_PROFILE


def _cfg_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _load_tensor_state_dict(model_source: str) -> dict[str, torch.Tensor]:
    root = Path(model_source)
    safetensors_path = root / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return _with_parakeet_tdt_aliases(dict(load_file(str(safetensors_path))))

    bin_path = root / "pytorch_model.bin"
    if bin_path.exists():
        loaded = torch.load(bin_path, map_location="cpu")
        if isinstance(loaded, dict):
            return _with_parakeet_tdt_aliases({
                str(key): value
                for key, value in loaded.items()
                if isinstance(value, torch.Tensor)
            })
    raise RuntimeError(f"unsupported Parakeet TDT checkpoint format in {model_source}")


def _add_tdt_derived_aliases(state_dict: dict[str, torch.Tensor]) -> None:
    def alias(target: str, source: str) -> None:
        if target not in state_dict and source in state_dict:
            state_dict[target] = state_dict[source]

    for index in range(4):
        alias(f"decoder.prediction.dec_rnn.lstm.{index}.Wx", f"decoder.lstm.weight_ih_l{index}")
        alias(f"decoder.prediction.dec_rnn.lstm.{index}.Wh", f"decoder.lstm.weight_hh_l{index}")
        bias_key = f"decoder.prediction.dec_rnn.lstm.{index}.bias"
        bias_ih = state_dict.get(f"decoder.lstm.bias_ih_l{index}")
        bias_hh = state_dict.get(f"decoder.lstm.bias_hh_l{index}")
        if bias_key not in state_dict and bias_ih is not None and bias_hh is not None:
            state_dict[bias_key] = bias_ih + bias_hh

    for source_index, target_index in ((0, 0), (2, 2), (3, 3), (5, 5), (6, 6)):
        alias(
            f"encoder.pre_encode.conv.{target_index}.weight",
            f"encoder.subsampling.layers.{source_index}.weight",
        )
        alias(
            f"encoder.pre_encode.conv.{target_index}.bias",
            f"encoder.subsampling.layers.{source_index}.bias",
        )
    alias("encoder.pre_encode.out.weight", "encoder.subsampling.linear.weight")
    alias("encoder.pre_encode.out.bias", "encoder.subsampling.linear.bias")



def _with_parakeet_tdt_aliases(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply profile-declared aliases needed by the local TDT fallback runtime."""

    add_tensor_aliases(
        state_dict,
        PARAKEET_TDT_PROFILE,
        derived_aliases=_add_tdt_derived_aliases,
    )
    return state_dict


@dataclass
class ParakeetTDTConfig:
    model_source: str
    sample_rate: int
    num_mel_bins: int
    hidden_dim: int
    num_layers: int
    attention_heads: int
    attention_head_dim: int
    attention_scale: float
    ff_intermediate_dim: int
    conv_kernel_size: int
    subsampling_factor: int
    subsampling_conv_channels: int
    predictor_hidden_dim: int
    predictor_num_layers: int
    joint_dim: int
    num_tdt_durations: int
    tdt_durations: tuple[int, ...]
    blank_id: int
    vocabulary: tuple[str, ...]
    encoder_hidden_act: str


def prepare_parakeet_tdt_audio_features(
    audio_file: str | Path,
    *,
    expected_frames: int | None,
    expected_mels: int,
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    return prepare_native_parakeet_audio_features(
        audio_file,
        expected_frames=expected_frames,
        expected_mels=expected_mels,
        torch_dtype=torch_dtype,
    )


def greedy_decode_parakeet_tdt_token_ids(
    *,
    config: ParakeetTDTConfig,
    encoder_hidden_states: np.ndarray,
    initial_states: tuple[np.ndarray, ...],
    step,
) -> list[int]:
    hidden = np.ascontiguousarray(np.asarray(encoder_hidden_states))
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(
            f"expected encoder_hidden_states with shape [1, T, D], got {tuple(hidden.shape)}"
        )
    predictor_layers = int(config.predictor_num_layers)
    if len(initial_states) != predictor_layers * 2:
        raise ValueError(
            f"expected {predictor_layers * 2} initial state tensors, got {len(initial_states)}"
        )

    states = tuple(np.ascontiguousarray(np.asarray(state)) for state in initial_states)
    durations = [int(value) for value in config.tdt_durations]
    if not durations and int(config.num_tdt_durations) > 0:
        durations = list(range(int(config.num_tdt_durations)))
    duration_class_count = max(len(durations), int(config.num_tdt_durations))
    last_token = _initial_parakeet_tdt_blank_id(config)
    emitted: list[int] = []
    time_index = 0

    while time_index < int(hidden.shape[1]):
        frame = np.ascontiguousarray(hidden[:, time_index, :])
        advanced = False
        symbols_added = 0

        while symbols_added < 10:
            logits, next_states = step(frame, last_token, states)
            logits_array = np.asarray(logits, dtype=np.float32)
            total_classes = int(logits_array.shape[-1])
            active_duration_count = duration_class_count
            if active_duration_count <= 0 or active_duration_count >= total_classes:
                active_duration_count = max(1, min(total_classes - 1, int(config.num_tdt_durations) or len(durations) or 1))
                if len(durations) != active_duration_count:
                    durations = list(range(active_duration_count))
            token_class_count = total_classes - active_duration_count
            blank_id = _effective_parakeet_tdt_blank_id(
                config,
                token_class_count=token_class_count,
            )
            if last_token < 0 or last_token >= token_class_count:
                last_token = blank_id
            token_scores = logits_array[:, :token_class_count]
            duration_scores = logits_array[:, token_class_count:]
            next_token = int(np.argmax(token_scores[0]))
            duration_index = int(np.argmax(duration_scores[0]))
            skip = int(durations[min(duration_index, len(durations) - 1)]) if durations else 1

            if next_token != blank_id:
                emitted.append(next_token)
                last_token = next_token
                states = tuple(np.ascontiguousarray(np.asarray(state).copy()) for state in next_states)

            symbols_added += 1

            if skip > 0:
                time_index += skip
                advanced = True
                break
            if next_token == blank_id:
                time_index += 1
                advanced = True
                break

        if not advanced:
            time_index += 1

    return emitted


def _initial_parakeet_tdt_blank_id(config: ParakeetTDTConfig) -> int:
    blank_id = int(config.blank_id)
    vocab_size = len(config.vocabulary)
    if vocab_size > 0 and blank_id == vocab_size - 1:
        return vocab_size
    return blank_id


def _effective_parakeet_tdt_blank_id(
    config: ParakeetTDTConfig,
    *,
    token_class_count: int,
) -> int:
    blank_id = int(config.blank_id)
    vocab_size = len(config.vocabulary)
    if vocab_size > 0 and blank_id == vocab_size - 1 and token_class_count == vocab_size + 1:
        return token_class_count - 1
    if 0 <= blank_id < token_class_count:
        return blank_id
    if vocab_size > 0 and token_class_count == vocab_size + 1:
        return token_class_count - 1
    return max(0, token_class_count - 1)


def load_parakeet_tdt_config(model_source: str) -> ParakeetTDTConfig:
    root_path = Path(model_source)
    config_path = root_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json for Parakeet TDT: {config_path}")
    root = json.loads(config_path.read_text())
    encoder = root.get("encoder") or root.get("encoder_config") or {}
    decoder = root.get("decoder") or {}
    prediction = decoder.get("prediction") or decoder.get("prednet") or {}
    joint = root.get("joint") or {}
    jointnet = joint.get("jointnet") or {}
    model_defaults = root.get("model_defaults") or {}
    preprocessor = root.get("preprocessor") or {}

    hidden_dim = int(_cfg_get(root, "hidden_dim", _cfg_get(encoder, "d_model", _cfg_get(encoder, "hidden_size", 0))))
    attention_heads = int(_cfg_get(encoder, "n_heads", _cfg_get(encoder, "num_attention_heads", 0)))
    attention_head_dim = hidden_dim // max(attention_heads, 1)
    tdt_durations = tuple(
        int(value)
        for value in _cfg_get(root, "tdt_durations", _cfg_get(root, "durations", _cfg_get(model_defaults, "tdt_durations", (0, 1, 2, 3, 4))))
    )
    vocabulary = tuple(str(value) for value in _cfg_get(joint, "vocabulary", ()))
    if not vocabulary:
        vocabulary = _load_parakeet_vocabulary(root_path)
    decoder_vocab_size = int(_cfg_get(root, "vocab_size", _cfg_get(decoder, "vocab_size", len(vocabulary))))
    decoding = root.get("decoding") or {}
    blank_id_raw = _cfg_get(
        root,
        "tdt_blank_id",
        _cfg_get(root, "blank_token_id", _cfg_get(decoding, "blank_id", _cfg_get(decoder, "blank_id", None))),
    )
    if blank_id_raw is None:
        blank_id = decoder_vocab_size
    else:
        try:
            blank_id = int(blank_id_raw)
        except (TypeError, ValueError):
            blank_id = decoder_vocab_size
        if blank_id < 0:
            blank_id = decoder_vocab_size

    return ParakeetTDTConfig(
        model_source=model_source,
        sample_rate=int(_cfg_get(preprocessor, "sample_rate", 16000)),
        num_mel_bins=int(_cfg_get(root, "num_mel_bins", _cfg_get(preprocessor, "features", _cfg_get(encoder, "feat_in", _cfg_get(encoder, "num_mel_bins", 128))))),
        hidden_dim=hidden_dim,
        num_layers=int(_cfg_get(root, "num_layers", _cfg_get(encoder, "n_layers", _cfg_get(encoder, "num_hidden_layers", 0)))),
        attention_heads=attention_heads,
        attention_head_dim=attention_head_dim,
        attention_scale=float(attention_head_dim ** -0.5 if attention_head_dim > 0 else 1.0),
        ff_intermediate_dim=int(
            _cfg_get(
                root,
                "ffn_intermediate_dim",
                _cfg_get(
                    encoder,
                    "ffn_hidden_size",
                    _cfg_get(encoder, "intermediate_size", round(hidden_dim * float(_cfg_get(encoder, "ff_expansion_factor", 4.0)))),
                ),
            )
        ),
        conv_kernel_size=int(_cfg_get(root, "conv_kernel_size", _cfg_get(encoder, "conv_kernel_size", 9))),
        subsampling_factor=int(_cfg_get(root, "subsampling_factor", _cfg_get(encoder, "subsampling_factor", 8))),
        subsampling_conv_channels=int(
            _cfg_get(root, "subsampling_conv_channels", _cfg_get(encoder, "subsampling_conv_channels", 256))
        ),
        predictor_hidden_dim=int(
            _cfg_get(root, "predictor_hidden_dim", _cfg_get(root, "decoder_hidden_size", _cfg_get(prediction, "pred_hidden", _cfg_get(model_defaults, "pred_hidden", 640))))
        ),
        predictor_num_layers=int(
            _cfg_get(root, "predictor_num_layers", _cfg_get(root, "num_decoder_layers", _cfg_get(prediction, "pred_rnn_layers", 1)))
        ),
        joint_dim=int(_cfg_get(root, "tdt_joint_dim", _cfg_get(jointnet, "joint_hidden", _cfg_get(model_defaults, "joint_hidden", 640)))),
        num_tdt_durations=int(_cfg_get(root, "tdt_num_durations", _cfg_get(model_defaults, "num_tdt_durations", len(tdt_durations)))),
        tdt_durations=tdt_durations,
        blank_id=blank_id,
        vocabulary=vocabulary,
        encoder_hidden_act=str(_cfg_get(root, "encoder_hidden_act", _cfg_get(encoder, "activation", _cfg_get(encoder, "hidden_act", "silu")))).lower(),
    )


def _load_parakeet_vocabulary(root: Path) -> tuple[str, ...]:
    vocab_txt = root / "vocab.txt"
    if vocab_txt.exists():
        pieces: list[str] = []
        for line in vocab_txt.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                _, token = line.split("\t", 1)
                pieces.append(token)
            elif line:
                pieces.append(line)
        if pieces:
            return tuple(pieces)

    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            loaded = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        except Exception:
            return ()
        model = loaded.get("model") if isinstance(loaded, dict) else None
        vocab = model.get("vocab") if isinstance(model, dict) else None
        if isinstance(vocab, dict):
            ordered = sorted(
                ((int(index), str(token)) for token, index in vocab.items()),
                key=lambda item: item[0],
            )
            return tuple(token for _, token in ordered)
    return ()


def _copy_linear_weight(linear: nn.Linear, weight: torch.Tensor, *, bias: torch.Tensor | None = None) -> None:
    linear.weight.data.copy_(weight.to(dtype=linear.weight.dtype, device=linear.weight.device))
    if linear.bias is not None:
        if bias is None:
            linear.bias.data.zero_()
        else:
            linear.bias.data.copy_(bias.to(dtype=linear.bias.dtype, device=linear.bias.device))


def _copy_conv2d_weight(conv: nn.Conv2d, weight: torch.Tensor, *, bias: torch.Tensor | None = None) -> None:
    tensor = weight
    if tuple(tensor.shape) != tuple(conv.weight.shape):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    conv.weight.data.copy_(tensor.to(dtype=conv.weight.dtype, device=conv.weight.device))
    if conv.bias is not None:
        if bias is None:
            conv.bias.data.zero_()
        else:
            conv.bias.data.copy_(bias.to(dtype=conv.bias.dtype, device=conv.bias.device))


def _copy_conv1d_weight(conv: nn.Conv1d, weight: torch.Tensor, *, bias: torch.Tensor | None = None) -> None:
    tensor = weight
    if tensor.ndim == 3 and tensor.shape[1] != conv.in_channels // conv.groups:
        tensor = tensor.permute(0, 2, 1).contiguous()
    conv.weight.data.copy_(tensor.to(dtype=conv.weight.dtype, device=conv.weight.device))
    if conv.bias is not None:
        if bias is None:
            conv.bias.data.zero_()
        else:
            conv.bias.data.copy_(bias.to(dtype=conv.bias.dtype, device=conv.bias.device))


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if "gelu" in activation:
        return F.gelu(x)
    if activation == "relu":
        return F.relu(x)
    return F.silu(x)


def _relative_position_embeddings(*, seq_len: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rel_positions = torch.arange(seq_len - 1, -seq_len, -1, device=device, dtype=torch.float32)
    half_dim = hidden_dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) * 2.0 / float(max(hidden_dim, 1)))
    )
    angles = rel_positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    embeddings = torch.stack((sin, cos), dim=-1).reshape(2 * seq_len - 1, half_dim * 2)
    if embeddings.shape[1] < hidden_dim:
        pad = torch.zeros(
            (embeddings.shape[0], hidden_dim - embeddings.shape[1]),
            device=device,
            dtype=embeddings.dtype,
        )
        embeddings = torch.cat((embeddings, pad), dim=1)
    elif embeddings.shape[1] > hidden_dim:
        embeddings = embeddings[:, :hidden_dim]
    return embeddings.to(dtype=dtype)


def _relative_position_bias(query: torch.Tensor, relative_key: torch.Tensor, *, scale: float) -> torch.Tensor:
    batch, heads, seq_len, _ = query.shape
    scores = torch.matmul(query, relative_key.transpose(-1, -2))
    rel_index = (
        torch.arange(seq_len, device=query.device).view(seq_len, 1)
        - torch.arange(seq_len, device=query.device).view(1, seq_len)
        + (seq_len - 1)
    )
    rel_index = rel_index.view(1, 1, seq_len, seq_len).expand(batch, heads, seq_len, seq_len)
    gathered = scores.gather(-1, rel_index)
    return gathered * float(scale)


class ParakeetTDTFeedForward(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, prefix: str, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.ff_intermediate_dim, bias=False)
        self.linear2 = nn.Linear(config.ff_intermediate_dim, config.hidden_dim, bias=False)
        _copy_linear_weight(self.linear1, state_dict[f"{prefix}.linear1.weight"])
        _copy_linear_weight(self.linear2, state_dict[f"{prefix}.linear2.weight"])

    def forward(self, x: torch.Tensor, *, activation: str) -> torch.Tensor:
        return self.linear2(_apply_activation(self.linear1(x), activation))


class ParakeetTDTSelfAttention(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, prefix: str, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.linear_q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.linear_k = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.linear_v = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.linear_out = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.linear_pos = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        _copy_linear_weight(self.linear_q, state_dict[f"{prefix}.linear_q.weight"])
        _copy_linear_weight(self.linear_k, state_dict[f"{prefix}.linear_k.weight"])
        _copy_linear_weight(self.linear_v, state_dict[f"{prefix}.linear_v.weight"])
        _copy_linear_weight(self.linear_out, state_dict[f"{prefix}.linear_out.weight"])
        _copy_linear_weight(self.linear_pos, state_dict[f"{prefix}.linear_pos.weight"])
        self.pos_bias_u = nn.Parameter(state_dict[f"{prefix}.pos_bias_u"].clone())
        self.pos_bias_v = nn.Parameter(state_dict[f"{prefix}.pos_bias_v"].clone())
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        num_heads = self.config.attention_heads
        head_dim = self.config.attention_head_dim

        q = self.linear_q(x).view(batch, seq_len, num_heads, head_dim)
        k = self.linear_k(x).view(batch, seq_len, num_heads, head_dim)
        v = self.linear_v(x).view(batch, seq_len, num_heads, head_dim)

        rel_pos = _relative_position_embeddings(
            seq_len=seq_len,
            hidden_dim=self.config.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )
        rel_k = self.linear_pos(rel_pos).view(1, 2 * seq_len - 1, num_heads, head_dim)

        q_u = q + self.pos_bias_u.view(1, 1, num_heads, head_dim).to(dtype=x.dtype, device=x.device)
        q_v = q + self.pos_bias_v.view(1, 1, num_heads, head_dim).to(dtype=x.dtype, device=x.device)

        q_u_heads = q_u.permute(0, 2, 1, 3)
        q_v_heads = q_v.permute(0, 2, 1, 3)
        k_heads = k.permute(0, 2, 1, 3)
        v_heads = v.permute(0, 2, 1, 3)
        rel_k_heads = rel_k.permute(0, 2, 1, 3)

        rel_bias = _relative_position_bias(
            q_v_heads,
            rel_k_heads,
            scale=self.config.attention_scale,
        )
        attn = F.scaled_dot_product_attention(
            q_u_heads,
            k_heads,
            v_heads,
            attn_mask=rel_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.permute(0, 2, 1, 3).reshape(batch, seq_len, self.config.hidden_dim)
        return self.linear_out(attn)


class ParakeetTDTConformerConv(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, prefix: str, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        hidden_dim = config.hidden_dim
        kernel = config.conv_kernel_size
        self.pointwise_conv1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=hidden_dim,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True, track_running_stats=True)
        self.pointwise_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        _copy_conv1d_weight(self.pointwise_conv1, state_dict[f"{prefix}.pointwise_conv1.weight"])
        _copy_conv1d_weight(self.depthwise_conv, state_dict[f"{prefix}.depthwise_conv.weight"])
        _copy_conv1d_weight(self.pointwise_conv2, state_dict[f"{prefix}.pointwise_conv2.weight"])
        self.batch_norm.weight.data.copy_(state_dict[f"{prefix}.batch_norm.weight"].to(dtype=self.batch_norm.weight.dtype))
        self.batch_norm.bias.data.copy_(state_dict[f"{prefix}.batch_norm.bias"].to(dtype=self.batch_norm.bias.dtype))
        self.batch_norm.running_mean.data.copy_(
            state_dict[f"{prefix}.batch_norm.running_mean"].to(dtype=self.batch_norm.running_mean.dtype)
        )
        self.batch_norm.running_var.data.copy_(
            state_dict[f"{prefix}.batch_norm.running_var"].to(dtype=self.batch_norm.running_var.dtype)
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = _apply_activation(x, self.config.encoder_hidden_act)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ParakeetTDTEncoderLayer(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, layer_index: int, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        prefix = f"encoder.layers.{layer_index}"
        self.feed_forward1 = ParakeetTDTFeedForward(config, f"{prefix}.feed_forward1", state_dict)
        self.feed_forward2 = ParakeetTDTFeedForward(config, f"{prefix}.feed_forward2", state_dict)
        self.self_attn = ParakeetTDTSelfAttention(config, f"{prefix}.self_attn", state_dict)
        self.conv = ParakeetTDTConformerConv(config, f"{prefix}.conv", state_dict)
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_dim)
        self.norm_self_att = nn.LayerNorm(config.hidden_dim)
        self.norm_conv = nn.LayerNorm(config.hidden_dim)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_dim)
        self.norm_out = nn.LayerNorm(config.hidden_dim)
        for name in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            module = getattr(self, name)
            module.weight.data.copy_(state_dict[f"{prefix}.{name}.weight"].to(dtype=module.weight.dtype))
            module.bias.data.copy_(state_dict[f"{prefix}.{name}.bias"].to(dtype=module.bias.dtype))
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x), activation=self.config.encoder_hidden_act)
        x = x + self.self_attn(self.norm_self_att(x))
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x), activation=self.config.encoder_hidden_act)
        return self.norm_out(x)


class ParakeetTDTPreEncode(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        channels = config.subsampling_conv_channels
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
                nn.Identity(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels),
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
                nn.Identity(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels),
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            ]
        )
        _copy_conv2d_weight(self.conv[0], state_dict["encoder.pre_encode.conv.0.weight"], bias=state_dict["encoder.pre_encode.conv.0.bias"])
        _copy_conv2d_weight(self.conv[2], state_dict["encoder.pre_encode.conv.2.weight"], bias=state_dict["encoder.pre_encode.conv.2.bias"])
        _copy_conv2d_weight(self.conv[3], state_dict["encoder.pre_encode.conv.3.weight"], bias=state_dict["encoder.pre_encode.conv.3.bias"])
        _copy_conv2d_weight(self.conv[5], state_dict["encoder.pre_encode.conv.5.weight"], bias=state_dict["encoder.pre_encode.conv.5.bias"])
        _copy_conv2d_weight(self.conv[6], state_dict["encoder.pre_encode.conv.6.weight"], bias=state_dict["encoder.pre_encode.conv.6.bias"])
        projected_width = config.num_mel_bins
        for _ in range(3):
            projected_width = (projected_width + 2 - 3) // 2 + 1
        self.out = nn.Linear(channels * projected_width, config.hidden_dim)
        _copy_linear_weight(
            self.out,
            state_dict["encoder.pre_encode.out.weight"],
            bias=state_dict["encoder.pre_encode.out.bias"],
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        if input_features.ndim != 3:
            raise ValueError(f"expected input_features [batch, frames, mels], got {tuple(input_features.shape)}")
        x = input_features.unsqueeze(1)
        x = F.relu(self.conv[0](x))
        x = F.relu(self.conv[3](self.conv[2](x)))
        x = F.relu(self.conv[6](self.conv[5](x)))
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
        return self.out(x)


class ParakeetTDTEncoder(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.pre_encode = ParakeetTDTPreEncode(config, state_dict)
        self.layers = nn.ModuleList(
            [ParakeetTDTEncoderLayer(config, index, state_dict) for index in range(config.num_layers)]
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        x = self.pre_encode(input_features)
        for layer in self.layers:
            x = layer(x)
        return x


class ParakeetTDTDecoderCell(nn.Module):
    def __init__(self, hidden_dim: int, prefix: str, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.Wx = nn.Parameter(state_dict[f"{prefix}.Wx"].clone())
        self.Wh = nn.Parameter(state_dict[f"{prefix}.Wh"].clone())
        self.bias = nn.Parameter(state_dict[f"{prefix}.bias"].clone())
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = F.linear(x, self.Wx) + F.linear(h, self.Wh) + self.bias
        i, f, g, o = torch.chunk(gates, 4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ParakeetTDTDecoderPrediction(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        vocab_size = max(config.blank_id + 1, state_dict["decoder.prediction.embed.weight"].shape[0])
        self.embed = nn.Embedding(vocab_size, config.predictor_hidden_dim)
        self.embed.weight.data.copy_(state_dict["decoder.prediction.embed.weight"].to(dtype=self.embed.weight.dtype))
        self.dec_rnn = nn.Module()
        self.dec_rnn.lstm = nn.ModuleList(
            [
                ParakeetTDTDecoderCell(
                    config.predictor_hidden_dim,
                    f"decoder.prediction.dec_rnn.lstm.{index}",
                    state_dict,
                )
                for index in range(config.predictor_num_layers)
            ]
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        state_h: tuple[torch.Tensor, ...],
        state_c: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x = self.embed(token_ids)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0, :]
        next_h: list[torch.Tensor] = []
        next_c: list[torch.Tensor] = []
        for index, cell in enumerate(self.dec_rnn.lstm):
            h_new, c_new = cell(x, state_h[index], state_c[index])
            next_h.append(h_new)
            next_c.append(c_new)
            x = h_new
        return x, tuple(next_h), tuple(next_c)


class ParakeetTDTJoint(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.enc = nn.Linear(config.hidden_dim, config.joint_dim)
        self.pred = nn.Linear(config.predictor_hidden_dim, config.joint_dim)
        output_classes = int(state_dict["joint.joint_net.2.weight"].shape[0])
        self.joint_net = nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
            nn.Linear(
                config.joint_dim,
                output_classes,
            ),
        )
        _copy_linear_weight(self.enc, state_dict["joint.enc.weight"], bias=state_dict["joint.enc.bias"])
        _copy_linear_weight(self.pred, state_dict["joint.pred.weight"], bias=state_dict["joint.pred.bias"])
        _copy_linear_weight(
            self.joint_net[2],
            state_dict["joint.joint_net.2.weight"],
            bias=state_dict["joint.joint_net.2.bias"],
        )

    def forward(self, encoder_frame: torch.Tensor, predictor_hidden: torch.Tensor) -> torch.Tensor:
        return self.joint_net(F.relu(self.enc(encoder_frame) + self.pred(predictor_hidden)))


class ParakeetTDTDecoderStep(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.prediction = ParakeetTDTDecoderPrediction(config, state_dict)
        self.joint = ParakeetTDTJoint(config, state_dict)
        self.config = config

    def forward(self, encoder_frame: torch.Tensor, token_ids: torch.Tensor, *state_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(state_tensors) != self.config.predictor_num_layers * 2:
            raise ValueError(
                f"expected {self.config.predictor_num_layers * 2} predictor state tensors, got {len(state_tensors)}"
            )
        state_h = tuple(state_tensors[0::2])
        state_c = tuple(state_tensors[1::2])
        predictor_hidden, next_h, next_c = self.prediction(token_ids, state_h, state_c)
        logits = self.joint(encoder_frame, predictor_hidden)
        outputs: list[torch.Tensor] = [logits]
        for h, c in zip(next_h, next_c, strict=True):
            outputs.append(h)
            outputs.append(c)
        return tuple(outputs)


class ParakeetTDTLocalModel(nn.Module):
    def __init__(self, config: ParakeetTDTConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.name_or_path = config.model_source
        self.family = "parakeet_tdt"
        self.config = config
        self.encoder = ParakeetTDTEncoder(config, state_dict)
        self.decoder = nn.Module()
        self.decoder.prediction = ParakeetTDTDecoderPrediction(config, state_dict)
        self.joint = ParakeetTDTJoint(config, state_dict)
        self.decoder_step = ParakeetTDTDecoderStep(config, state_dict)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features)

    def initial_decoder_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        state: list[torch.Tensor] = []
        for _ in range(self.config.predictor_num_layers):
            state.append(torch.zeros((batch_size, self.config.predictor_hidden_dim), device=device, dtype=dtype))
            state.append(torch.zeros((batch_size, self.config.predictor_hidden_dim), device=device, dtype=dtype))
        return tuple(state)

    def greedy_decode_token_ids(self, input_features: torch.Tensor) -> list[int]:
        with torch.no_grad():
            encoder_hidden = self.encoder(input_features)
            batch = int(encoder_hidden.shape[0])
            if batch != 1:
                raise ValueError("Parakeet TDT local greedy decode currently expects batch size 1")
            states = self.initial_decoder_state(
                batch_size=batch,
                device=encoder_hidden.device,
                dtype=encoder_hidden.dtype,
            )
            state_arrays = tuple(state.detach().cpu().numpy() for state in states)
            encoder_hidden_np = encoder_hidden.detach().cpu().numpy()

            def _step(
                frame: np.ndarray,
                token_id: int,
                state_values: tuple[np.ndarray, ...],
            ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
                frame_tensor = torch.from_numpy(frame).to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)
                token_tensor = torch.tensor([[token_id]], device=encoder_hidden.device, dtype=torch.long)
                state_tensors = tuple(
                    torch.from_numpy(value).to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)
                    for value in state_values
                )
                outputs = self.decoder_step(frame_tensor, token_tensor, *state_tensors)
                logits = outputs[0].detach().cpu().numpy()
                next_states = tuple(output.detach().cpu().numpy() for output in outputs[1:])
                return logits, next_states

            return greedy_decode_parakeet_tdt_token_ids(
                config=self.config,
                encoder_hidden_states=encoder_hidden_np,
                initial_states=state_arrays,
                step=_step,
            )

    def decode_token_ids(self, token_ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.config.vocabulary):
                continue
            piece = self.config.vocabulary[token_id]
            if piece.startswith("<|") and piece.endswith("|>"):
                continue
            pieces.append(piece)
        text = "".join(pieces).replace("▁", " ")
        return re.sub(r"\s+", " ", text).strip()


def load_tdt_local_model(model_source: str, *, torch_dtype: torch.dtype) -> ParakeetTDTLocalModel:
    config = load_parakeet_tdt_config(model_source)
    state_dict = _load_tensor_state_dict(model_source)
    predictor_vocab_size = int(state_dict["decoder.prediction.embed.weight"].shape[0])
    if config.blank_id < 0 or config.blank_id >= predictor_vocab_size:
        config = replace(config, blank_id=predictor_vocab_size - 1)
    elif config.blank_id == len(config.vocabulary) - 1 and predictor_vocab_size == len(config.vocabulary) + 1:
        config = replace(config, blank_id=predictor_vocab_size - 1)
    model = ParakeetTDTLocalModel(config, state_dict).eval()
    model.to(dtype=torch_dtype)
    return model


def build_parakeet_tdt_component_specs(
    model: ParakeetTDTLocalModel,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
) -> list[ComponentModuleSpec]:
    input_features = named_tensors["input_features"]
    example_hidden = model.encoder(input_features)
    batch_size = int(example_hidden.shape[0])
    initial_states = model.initial_decoder_state(
        batch_size=batch_size,
        device=example_hidden.device,
        dtype=example_hidden.dtype,
    )
    example_frame = example_hidden[:, :1, :].reshape(batch_size, example_hidden.shape[-1])
    example_token_id = torch.full(
        (batch_size,),
        int(model.config.blank_id),
        device=example_hidden.device,
        dtype=torch.long,
    )

    state_input_keys: list[str] = []
    state_output_keys: list[str] = []
    for index in range(model.config.predictor_num_layers):
        state_input_keys.extend((f"state_h_{index}", f"state_c_{index}"))
        state_output_keys.extend((f"state_h_{index}", f"state_c_{index}"))

    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "tdt_transcription",
        "adapter_family": "parakeet_tdt",
    }
    return [
        ComponentModuleSpec(
            component="audio_encoder",
            module=model.encoder,
            example_inputs=(input_features,),
            input_keys=("input_features",),
            output_keys=("encoder_hidden_states",),
            graph_meta={**common_graph_meta, "component": "audio_encoder"},
            metadata={"family": "parakeet_tdt", "task": "tdt_transcription"},
        ),
        ComponentModuleSpec(
            component="decoder",
            module=model.decoder_step,
            example_inputs=(example_frame, example_token_id, *initial_states),
            input_keys=("encoder_frame", "token_ids", *state_input_keys),
            output_keys=("step_logits", *state_output_keys),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata={"family": "parakeet_tdt", "task": "tdt_transcription"},
        ),
    ]
