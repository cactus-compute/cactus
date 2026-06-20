from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cactus.transpile.component_pipeline import ComponentModuleSpec
from cactus.transpile.tdt_runtime import ParakeetTDTConfig
from cactus.transpile.tdt_runtime import ParakeetTDTANEEncoder
from cactus.transpile.tdt_runtime import ParakeetTDTDecoderPrediction
from cactus.transpile.tdt_runtime import ParakeetTDTEncoder
from cactus.transpile.tdt_runtime import ParakeetTDTJoint
from cactus.transpile.tdt_runtime import _add_tdt_derived_aliases
from cactus.transpile.tdt_runtime import _cfg_get
from cactus.transpile.tdt_runtime import _copy_linear_weight
from cactus.transpile.tdt_runtime import _load_parakeet_vocabulary
from cactus.transpile.tdt_runtime import _parse_attention_context_size
from cactus.transpile.audio_preprocess import _PARAKEET_FRAME_LENGTH
from cactus.transpile.audio_preprocess import _PARAKEET_HOP_LENGTH
from cactus.transpile.audio_preprocess import _PARAKEET_LOG_FLOOR
from cactus.transpile.audio_preprocess import _PARAKEET_N_FFT
from cactus.transpile.audio_preprocess import _PARAKEET_PREEMPHASIS
from cactus.transpile.audio_preprocess import _PARAKEET_SAMPLE_RATE
from cactus.transpile.audio_preprocess import generic_log_mel_features
from cactus.transpile.audio_preprocess import load_audio_waveform
from cactus.transpile.model_profiles import NEMOTRON_ASR_PROFILE
from cactus.transpile.model_profiles import add_tensor_aliases


def nemotron_asr_frame_values() -> tuple[int, ...]:
    raw = os.environ.get("CACTUS_NEMOTRON_ASR_FRAMES")
    if raw is None:
        raw = os.environ.get("CACTUS_NPU_NEMOTRON_FRAMES", "1024,2048,4096,6144")
    values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("CACTUS_NEMOTRON_ASR_FRAMES values must be positive frame counts")
    return values


def _valid_hidden_frame_count(active_frames: int | None, total_frames: int, hidden_frames: int) -> int:
    if active_frames is None or active_frames <= 0 or total_frames <= 0:
        return hidden_frames
    return min(hidden_frames, (int(active_frames) * hidden_frames + total_frames - 1) // total_frames)


@contextmanager
def _temporary_float32(module: torch.nn.Module):
    dtypes = {
        param.dtype
        for param in module.parameters()
        if param.is_floating_point()
    }
    dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
    if dtype is None or dtype == torch.float32:
        yield
        return
    module.to(dtype=torch.float32)
    try:
        yield
    finally:
        module.to(dtype=dtype)


@dataclass
class NemotronASRConfig:
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
    conv_norm_type: str
    conv_is_causal: bool
    subsampling_factor: int
    subsampling_conv_channels: int
    causal_downsampling: bool
    predictor_hidden_dim: int
    predictor_num_layers: int
    joint_dim: int
    blank_id: int
    vocabulary: tuple[str, ...]
    encoder_hidden_act: str
    prompt_dim: int
    prompt_dictionary: dict[str, int]
    default_prompt_id: int
    max_symbols_per_step: int
    attention_context_style: str
    attention_context_size: tuple[int, int]

    def as_parakeet_config(self) -> ParakeetTDTConfig:
        return ParakeetTDTConfig(
            model_source=self.model_source,
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            attention_heads=self.attention_heads,
            attention_head_dim=self.attention_head_dim,
            attention_scale=self.attention_scale,
            ff_intermediate_dim=self.ff_intermediate_dim,
            conv_kernel_size=self.conv_kernel_size,
            conv_norm_type=self.conv_norm_type,
            conv_is_causal=self.conv_is_causal,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            causal_downsampling=self.causal_downsampling,
            predictor_hidden_dim=self.predictor_hidden_dim,
            predictor_num_layers=self.predictor_num_layers,
            joint_dim=self.joint_dim,
            num_tdt_durations=0,
            tdt_durations=(),
            blank_id=self.blank_id,
            vocabulary=self.vocabulary,
            encoder_hidden_act=self.encoder_hidden_act,
            attention_context_style=self.attention_context_style,
            attention_context_size=self.attention_context_size,
        )


def _load_tensor_state_dict(model_source: str) -> dict[str, torch.Tensor]:
    root = Path(model_source)
    safetensors_path = root / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return _with_nemotron_asr_aliases(dict(load_file(str(safetensors_path))))

    bin_path = root / "pytorch_model.bin"
    if bin_path.exists():
        loaded = torch.load(bin_path, map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            loaded = loaded["state_dict"]
        if isinstance(loaded, dict):
            return _with_nemotron_asr_aliases({
                str(key): value
                for key, value in loaded.items()
                if isinstance(value, torch.Tensor)
            })
    raise RuntimeError(f"unsupported Nemotron ASR checkpoint format in {model_source}")


def _with_nemotron_asr_aliases(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    add_tensor_aliases(
        state_dict,
        NEMOTRON_ASR_PROFILE,
        derived_aliases=_add_tdt_derived_aliases,
    )
    return state_dict


def _prompt_dim_from_dictionary(prompt_dictionary: dict[str, int], *candidates: int) -> int:
    dim = max((int(value or 0) for value in candidates), default=0)
    if prompt_dictionary:
        dim = max(dim, max(int(value) for value in prompt_dictionary.values()) + 1)
    return max(1, dim)


def prepare_nemotron_asr_audio_features(
    audio_file: str | Path,
    *,
    expected_frames: int | None,
    expected_mels: int,
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    if expected_frames is None or expected_frames <= 0:
        expected_frames = max(nemotron_asr_frame_values())
    waveform = load_audio_waveform(audio_file, target_sample_rate=_PARAKEET_SAMPLE_RATE)
    features, feature_length = generic_log_mel_features(
        waveform,
        sample_rate=_PARAKEET_SAMPLE_RATE,
        num_mels=expected_mels,
        n_fft=_PARAKEET_N_FFT,
        hop_length=_PARAKEET_HOP_LENGTH,
        frame_length=_PARAKEET_FRAME_LENGTH,
        preemphasis=_PARAKEET_PREEMPHASIS,
        mel_floor=_PARAKEET_LOG_FLOOR,
        mel_floor_additive=True,
        normalize_active_frames_only=None,
    )
    active_frames = feature_length
    if isinstance(expected_frames, int) and expected_frames > 0:
        active_frames = min(active_frames, expected_frames)
    features = features[:active_frames, :]
    if features.shape[1] != expected_mels:
        raise ValueError(
            f"feature mel dimension mismatch: expected {expected_mels}, got {features.shape[1]}"
        )
    if isinstance(expected_frames, int) and expected_frames > active_frames:
        features = np.pad(features, ((0, expected_frames - active_frames), (0, 0)), mode="constant")
    tensor = torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0).to(dtype=torch_dtype)
    return tensor, active_frames


def load_nemotron_asr_config(model_source: str) -> NemotronASRConfig:
    root_path = Path(model_source)
    config_path = root_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json for Nemotron ASR: {config_path}")
    root = json.loads(config_path.read_text())
    encoder = root.get("encoder") or root.get("encoder_config") or {}
    decoder = root.get("decoder") or {}
    prediction = decoder.get("prediction") or decoder.get("prednet") or {}
    joint = root.get("joint") or {}
    jointnet = joint.get("jointnet") or {}
    model_defaults = root.get("model_defaults") or {}
    preprocessor = root.get("preprocessor") or {}
    decoding = root.get("decoding") or {}
    greedy = decoding.get("greedy") or {}

    hidden_dim = int(_cfg_get(root, "hidden_dim", _cfg_get(encoder, "d_model", _cfg_get(encoder, "hidden_size", 0))))
    attention_heads = int(_cfg_get(encoder, "n_heads", _cfg_get(encoder, "num_attention_heads", 0)))
    attention_head_dim = hidden_dim // max(attention_heads, 1)
    vocabulary = tuple(str(value) for value in _cfg_get(joint, "vocabulary", _cfg_get(root, "labels", ())))
    if not vocabulary:
        vocabulary = _load_parakeet_vocabulary(root_path)
    decoder_vocab_size = int(_cfg_get(root, "rnnt_blank_id", _cfg_get(decoder, "vocab_size", len(vocabulary))))
    prompt_dictionary = _cfg_get(root, "prompt_dictionary", _cfg_get(model_defaults, "prompt_dictionary", {}))
    if not isinstance(prompt_dictionary, dict):
        prompt_dictionary = {}
    prompt_dictionary = {str(key): int(value) for key, value in prompt_dictionary.items()}
    prompt_dim = _prompt_dim_from_dictionary(
        prompt_dictionary,
        int(_cfg_get(root, "prompt_dim", 0) or 0),
        int(_cfg_get(root, "num_prompts", 0) or 0),
        int(_cfg_get(model_defaults, "num_prompts", 0) or 0),
    )

    return NemotronASRConfig(
        model_source=model_source,
        sample_rate=int(_cfg_get(preprocessor, "sample_rate", _cfg_get(root, "sample_rate", 16000))),
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
        conv_norm_type=str(_cfg_get(root, "conv_norm_type", _cfg_get(encoder, "conv_norm_type", "batch_norm"))).lower(),
        conv_is_causal=str(_cfg_get(root, "conv_context_size", _cfg_get(encoder, "conv_context_size", ""))).lower() == "causal",
        subsampling_factor=int(_cfg_get(root, "subsampling_factor", _cfg_get(encoder, "subsampling_factor", 8))),
        subsampling_conv_channels=int(_cfg_get(root, "subsampling_conv_channels", _cfg_get(encoder, "subsampling_conv_channels", 256))),
        causal_downsampling=bool(_cfg_get(root, "causal_downsampling", _cfg_get(encoder, "causal_downsampling", False))),
        predictor_hidden_dim=int(_cfg_get(root, "predictor_hidden_dim", _cfg_get(root, "decoder_hidden_size", _cfg_get(prediction, "pred_hidden", _cfg_get(model_defaults, "pred_hidden", 640))))),
        predictor_num_layers=int(_cfg_get(root, "predictor_num_layers", _cfg_get(root, "num_decoder_layers", _cfg_get(prediction, "pred_rnn_layers", 1)))),
        joint_dim=int(_cfg_get(root, "rnnt_joint_dim", _cfg_get(jointnet, "joint_hidden", _cfg_get(model_defaults, "joint_hidden", 640)))),
        blank_id=int(_cfg_get(root, "rnnt_blank_id", decoder_vocab_size)),
        vocabulary=vocabulary,
        encoder_hidden_act=str(_cfg_get(root, "encoder_hidden_act", _cfg_get(encoder, "activation", _cfg_get(encoder, "hidden_act", "silu")))).lower(),
        prompt_dim=prompt_dim,
        prompt_dictionary=prompt_dictionary,
        default_prompt_id=int(_cfg_get(root, "default_prompt_id", prompt_dictionary.get("auto", 0))),
        max_symbols_per_step=int(_cfg_get(root, "max_symbols_per_step", _cfg_get(greedy, "max_symbols", 10))),
        attention_context_style=str(_cfg_get(encoder, "att_context_style", "regular")),
        attention_context_size=_parse_attention_context_size(_cfg_get(encoder, "att_context_size", None)),
    )


class NemotronASRPromptKernel(nn.Module):
    def __init__(self, config: NemotronASRConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        linear1_weight = state_dict["prompt_kernel.0.weight"]
        linear2_weight = state_dict["prompt_kernel.2.weight"]
        self.linear1 = nn.Linear(int(linear1_weight.shape[1]), int(linear1_weight.shape[0]))
        self.linear2 = nn.Linear(int(linear2_weight.shape[1]), int(linear2_weight.shape[0]))
        _copy_linear_weight(self.linear1, state_dict["prompt_kernel.0.weight"], bias=state_dict["prompt_kernel.0.bias"])
        _copy_linear_weight(self.linear2, state_dict["prompt_kernel.2.weight"], bias=state_dict["prompt_kernel.2.bias"])

    def forward(self, encoder_frame: torch.Tensor, language_prompt: torch.Tensor) -> torch.Tensor:
        if language_prompt.ndim == 1:
            language_prompt = language_prompt.unsqueeze(0)
        x = torch.cat((encoder_frame, language_prompt.to(dtype=encoder_frame.dtype, device=encoder_frame.device)), dim=-1)
        return self.linear2(F.relu(self.linear1(x)))


class NemotronASRDecoderStep(nn.Module):
    def __init__(self, config: NemotronASRConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        parakeet_config = config.as_parakeet_config()
        self.prompt_kernel = NemotronASRPromptKernel(config, state_dict)
        self.prediction = ParakeetTDTDecoderPrediction(parakeet_config, state_dict)
        self.joint = ParakeetTDTJoint(parakeet_config, state_dict)
        self.config = config

    def forward(self, encoder_frame: torch.Tensor, language_prompt: torch.Tensor, token_ids: torch.Tensor, *state_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(state_tensors) != self.config.predictor_num_layers * 2:
            raise ValueError(
                f"expected {self.config.predictor_num_layers * 2} predictor state tensors, got {len(state_tensors)}"
            )
        state_h = tuple(state_tensors[0::2])
        state_c = tuple(state_tensors[1::2])
        prompted_frame = self.prompt_kernel(encoder_frame, language_prompt)
        predictor_hidden, next_h, next_c = self.prediction(token_ids, state_h, state_c)
        logits = self.joint(prompted_frame, predictor_hidden)
        outputs: list[torch.Tensor] = [logits]
        for h, c in zip(next_h, next_c, strict=True):
            outputs.append(h)
            outputs.append(c)
        return tuple(outputs)


class NemotronASRLocalModel(nn.Module):
    def __init__(self, config: NemotronASRConfig, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.name_or_path = config.model_source
        self.family = "nemotron_asr"
        self.config = config
        self.encoder = ParakeetTDTEncoder(config.as_parakeet_config(), state_dict)
        self.decoder_step = NemotronASRDecoderStep(config, state_dict)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features)

    def initial_decoder_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        state: list[torch.Tensor] = []
        for _ in range(self.config.predictor_num_layers):
            state.append(torch.zeros((batch_size, self.config.predictor_hidden_dim), device=device, dtype=dtype))
            state.append(torch.zeros((batch_size, self.config.predictor_hidden_dim), device=device, dtype=dtype))
        return tuple(state)

    def language_prompt(self, language: str = "auto", *, batch_size: int = 1, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        prompt_id = int(self.config.prompt_dictionary.get(language, self.config.default_prompt_id))
        prompt_id = max(0, min(prompt_id, self.config.prompt_dim - 1))
        prompt = torch.zeros((batch_size, self.config.prompt_dim), device=device, dtype=dtype)
        prompt[:, prompt_id] = 1.0
        return prompt

    def greedy_decode_token_ids(self, input_features: torch.Tensor, *, language: str = "auto", active_frames: int | None = None) -> list[int]:
        with torch.no_grad():
            encoder_hidden = self.encoder(input_features)
            valid_hidden = _valid_hidden_frame_count(active_frames, int(input_features.shape[1]), int(encoder_hidden.shape[1]))
            encoder_hidden = encoder_hidden[:, :valid_hidden, :]
            batch = int(encoder_hidden.shape[0])
            if batch != 1:
                raise ValueError("Nemotron ASR local greedy decode currently expects batch size 1")
            states = self.initial_decoder_state(batch_size=batch, device=encoder_hidden.device, dtype=encoder_hidden.dtype)
            language_prompt = self.language_prompt(language, batch_size=batch, device=encoder_hidden.device, dtype=encoder_hidden.dtype)
            state_arrays = tuple(state.detach().cpu().numpy() for state in states)
            encoder_hidden_np = encoder_hidden.detach().cpu().numpy()
            prompt_np = language_prompt.detach().cpu().numpy()

            def _step(frame: np.ndarray, token_id: int, state_values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
                frame_tensor = torch.from_numpy(frame).to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)
                prompt_tensor = torch.from_numpy(prompt_np).to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)
                token_tensor = torch.tensor([token_id], device=encoder_hidden.device, dtype=torch.long)
                state_tensors = tuple(
                    torch.from_numpy(value).to(device=encoder_hidden.device, dtype=encoder_hidden.dtype)
                    for value in state_values
                )
                outputs = self.decoder_step(frame_tensor, prompt_tensor, token_tensor, *state_tensors)
                logits = outputs[0].detach().cpu().numpy()
                next_states = tuple(output.detach().cpu().numpy() for output in outputs[1:])
                return logits, next_states

            return greedy_decode_nemotron_asr_token_ids(
                config=self.config,
                encoder_hidden_states=encoder_hidden_np,
                initial_states=state_arrays,
                step=_step,
            )

    def decode_token_ids(self, token_ids: list[int], *, strip_lang_tags: bool = True) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.config.vocabulary):
                continue
            pieces.append(self.config.vocabulary[token_id])
        text = "".join(pieces).replace("▁", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if strip_lang_tags:
            text = re.sub(r"\s*<[a-z]{2}(?:-[A-Z]{2})?>\s*", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        return text


def greedy_decode_nemotron_asr_token_ids(
    *,
    config: NemotronASRConfig,
    encoder_hidden_states: np.ndarray,
    initial_states: tuple[np.ndarray, ...],
    step,
) -> list[int]:
    hidden = np.ascontiguousarray(np.asarray(encoder_hidden_states))
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"expected encoder_hidden_states with shape [1, T, D], got {tuple(hidden.shape)}")
    if len(initial_states) != int(config.predictor_num_layers) * 2:
        raise ValueError(f"expected {int(config.predictor_num_layers) * 2} initial state tensors, got {len(initial_states)}")

    states = tuple(np.ascontiguousarray(np.asarray(state)) for state in initial_states)
    last_token = int(config.blank_id)
    emitted: list[int] = []

    for time_index in range(int(hidden.shape[1])):
        frame = np.ascontiguousarray(hidden[:, time_index, :])
        symbols_added = 0
        while symbols_added < int(config.max_symbols_per_step):
            logits, next_states = step(frame, last_token, states)
            logits_array = np.asarray(logits, dtype=np.float32)
            next_token = int(np.argmax(logits_array[0]))
            symbols_added += 1
            if next_token == int(config.blank_id):
                break
            emitted.append(next_token)
            last_token = next_token
            states = tuple(np.ascontiguousarray(np.asarray(state).copy()) for state in next_states)
    return emitted


def load_nemotron_asr_local_model(model_source: str, *, torch_dtype: torch.dtype) -> NemotronASRLocalModel:
    config = load_nemotron_asr_config(model_source)
    state_dict = _load_tensor_state_dict(model_source)
    predictor_vocab_size = int(state_dict["decoder.prediction.embed.weight"].shape[0])
    if config.blank_id < 0 or config.blank_id >= predictor_vocab_size:
        config = replace(config, blank_id=predictor_vocab_size - 1)
    model = NemotronASRLocalModel(config, state_dict).eval()
    model.to(dtype=torch_dtype)
    return model


def build_nemotron_asr_component_specs(
    model: NemotronASRLocalModel,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
) -> list[ComponentModuleSpec]:
    input_features = named_tensors["input_features"]
    example_hidden = model.encoder(input_features)
    batch_size = int(example_hidden.shape[0])
    initial_states = model.initial_decoder_state(batch_size=batch_size, device=example_hidden.device, dtype=example_hidden.dtype)
    example_prompt = model.language_prompt("auto", batch_size=batch_size, device=example_hidden.device, dtype=example_hidden.dtype)
    example_frame = example_hidden[:, :1, :].reshape(batch_size, example_hidden.shape[-1])
    example_token_id = torch.full((batch_size,), int(model.config.blank_id), device=example_hidden.device, dtype=torch.long)

    state_input_keys: list[str] = []
    state_output_keys: list[str] = []
    for index in range(model.config.predictor_num_layers):
        state_input_keys.extend((f"state_h_{index}", f"state_c_{index}"))
        state_output_keys.extend((f"state_h_{index}", f"state_c_{index}"))

    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "rnnt_transcription",
        "adapter_family": "nemotron_asr",
    }

    factor = max(1, int(model.config.subsampling_factor))
    npu_frame_values = nemotron_asr_frame_values()
    npu_variants: list[dict[str, object]] = []
    for npu_frames in npu_frame_values:
        if npu_frames <= 0 or npu_frames % factor != 0:
            raise ValueError(f"CACTUS_NEMOTRON_ASR_FRAMES values must be positive multiples of {factor}, got {npu_frames}")
        npu_input_features = torch.zeros(
            (int(input_features.shape[0]), npu_frames, int(input_features.shape[-1])),
            device=input_features.device,
            dtype=input_features.dtype,
        )
        valid_frames = min(int(input_features.shape[1]), npu_frames)
        npu_input_features[:, :valid_frames, :] = input_features[:, :valid_frames, :]
        with torch.no_grad():
            ane_seq_len = int(model.encoder.pre_encode(npu_input_features).shape[1])
        npu_variants.append({
            "module": ParakeetTDTANEEncoder(model.encoder, ane_seq_len),
            "example_inputs": (npu_input_features.to(dtype=torch.float32),),
            "filename": f"audio_encoder_{npu_frames}.mlpackage",
        })

    return [
        ComponentModuleSpec(
            component="audio_encoder",
            module=model.encoder,
            example_inputs=(input_features,),
            input_keys=("input_features",),
            output_keys=("encoder_hidden_states",),
            graph_meta={**common_graph_meta, "component": "audio_encoder"},
            metadata={"family": "nemotron_asr", "task": "rnnt_transcription"},
            npu_runtime_input_count=1,
            npu_reparam=_temporary_float32,
            npu_variants=tuple(npu_variants),
        ),
        ComponentModuleSpec(
            component="decoder",
            module=model.decoder_step,
            example_inputs=(example_frame, example_prompt, example_token_id, *initial_states),
            input_keys=("encoder_frame", "language_prompt", "token_ids", *state_input_keys),
            output_keys=("step_logits", *state_output_keys),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata={"family": "nemotron_asr", "task": "rnnt_transcription"},
        ),
    ]
