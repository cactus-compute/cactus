from __future__ import annotations

import numpy as np
import torch

from cactus.transpile import hf_model
from cactus.transpile import component_bundle_runtime
from cactus.transpile import model_adapters


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array)

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeComponent:
    def __init__(self, output_fn):
        self._output_fn = output_fn
        self._input_names = ()
        self._output_names = ()
        self._inputs = ()

    def set_inputs(self, inputs):
        self._inputs = tuple(inputs)

    def execute(self):
        return [_FakeTensor(self._output_fn(self._inputs))]


class _FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        mapping = {
            "<|endoftext|>": [99],
            "<|endoftranscript|>": [99],
            "</s>": [99],
            "<pad>": [0],
            "override": [42],
        }
        return list(mapping.get(text, []))

    def decode(self, token_ids, skip_special_tokens: bool = False):
        pieces = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {0, 99}:
                continue
            pieces.append(
                {
                    5: "hello",
                    6: " world",
                    42: "override",
                    99: "<|endoftranscript|>",
                }.get(int(token_id), f"<{int(token_id)}>")
            )
        return "".join(pieces)


def test_infer_task_from_config_prefers_whisper_seq2seq(monkeypatch) -> None:
    monkeypatch.setattr(
        hf_model,
        "_load_config_json",
        lambda _: {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]},
    )
    assert hf_model._infer_task_from_config("openai/whisper-small") == "seq2seq_transcription"


def test_run_seq2seq_transcription_bundle_decodes_until_eos(monkeypatch) -> None:
    def _fake_prepare_features(*, audio_file, manifest, expected_shape, torch_dtype):
        _ = audio_file, manifest, expected_shape, torch_dtype
        return np.zeros((1, 80, 3000), dtype=np.float16), 123

    monkeypatch.setattr(
        component_bundle_runtime,
        "_prepare_generic_audio_encoder_features",
        _fake_prepare_features,
    )
    monkeypatch.setattr(
        component_bundle_runtime,
        "_load_bundle_tokenizer",
        lambda manifest: _FakeTokenizer(),
    )

    encoder = _FakeComponent(lambda inputs: np.zeros((1, 1500, 4), dtype=np.float16))

    def _decoder_outputs(inputs):
        decoder_input_ids, encoder_hidden_states = inputs
        _ = encoder_hidden_states
        logits = np.full((1, decoder_input_ids.shape[1], 256), -100.0, dtype=np.float32)
        active_tokens = int(np.count_nonzero(decoder_input_ids[0] != 0))
        step_index = max(0, active_tokens - 2)
        position = max(0, active_tokens - 1)
        if step_index == 0:
            logits[0, position, 220] = 200.0
            logits[0, position, 5] = 100.0
        elif step_index == 1:
            logits[0, position, 7] = 200.0
            logits[0, position, 6] = 100.0
        else:
            logits[0, position, 99] = 100.0
        return logits

    decoder = _FakeComponent(_decoder_outputs)
    manifest = {
        "model_id": "openai/whisper-small",
        "family": "whisper",
        "task": "seq2seq_transcription",
        "component_order": ["audio_encoder", "decoder"],
        "inputs": {
            "audio_file": "/tmp/fake.wav",
            "input_shapes": {"input_features": [1, 80, 3000]},
            "decoder_input_ids": [50258, 50364],
            "pad_token_id": 0,
            "eos_token_id": 99,
            "target_token_count": 8,
            "suppress_tokens": [7],
            "begin_suppress_tokens": [220],
        },
        "components": [
            {
                "component": "audio_encoder",
                "logical_inputs": ["input_features"],
                "logical_outputs": ["encoder_hidden_states"],
            },
            {
                "component": "decoder",
                "logical_inputs": ["decoder_input_ids", "encoder_hidden_states"],
                "logical_outputs": ["logits"],
            },
        ],
    }

    result = component_bundle_runtime._run_seq2seq_transcription_bundle(
        component_graphs={"audio_encoder": encoder, "decoder": decoder},
        manifest=manifest,
        audio_file="/tmp/fake.wav",
        prompt=None,
        torch_dtype=np.float16,  # type: ignore[arg-type]
        max_new_tokens=4,
        stop_sequences=(),
    )

    assert result["generated_token_ids"] == [5, 6, 99]
    assert result["transcript"] == "hello world"
    assert result["stop_reason"] == "eos_token"
    assert result["active_feature_frames"] == 123


def test_resolve_whisper_decoder_prompt_token_ids_prefers_forced_decoder_ids() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            mapping = {
                "<|startoftranscript|>": [50258] if not add_special_tokens else [50258, 99999],
                "prompt": [77, 88],
            }
            return list(mapping.get(text, []))

    prompt_ids = hf_model._resolve_whisper_decoder_prompt_token_ids(
        _Tokenizer(),
        prompt=None,
        decoder_start_token_id=50258,
        forced_decoder_ids=[[1, 50259], [2, 50359], [3, 50363]],
    )
    assert prompt_ids == [50258, 50259, 50359, 50363]


def test_select_last_non_pad_token_uses_last_real_position() -> None:
    hidden = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[11, 22, 0, 0]], dtype=torch.int64)
    selected = model_adapters._select_last_non_pad_token(
        hidden,
        input_ids,
        pad_token_id=0,
    )
    assert tuple(selected.shape) == (1, 1, 2)
    assert torch.equal(selected[0, 0], hidden[0, 1])
