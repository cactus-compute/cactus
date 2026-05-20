from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from cactus.transpile import audio_preprocess
from cactus.transpile import component_bundle_runtime
from cactus.transpile import multimodal_runtime
from cactus.transpile import hf_model


def test_multimodal_decoder_inputs_right_align_to_static_tail() -> None:
    class FakeComponent:
        _input_names = ("inputs_embeds", "per_layer_inputs", "position_ids")

    store = {
        "inputs_embeds": np.arange(1 * 6 * 2, dtype=np.float16).reshape(1, 6, 2),
        "per_layer_inputs": np.arange(1 * 6 * 1 * 2, dtype=np.float16).reshape(1, 6, 1, 2),
        "position_ids": np.arange(6, dtype=np.int64).reshape(1, 6),
    }
    original = {key: value.copy() for key, value in store.items()}

    component_bundle_runtime._right_align_decoder_inputs_to_static_tail(
        store,
        component=FakeComponent(),  # type: ignore[arg-type]
        prompt_token_count=4,
    )

    assert np.all(store["inputs_embeds"][:, :2, :] == 0)
    np.testing.assert_array_equal(store["inputs_embeds"][:, 2:, :], original["inputs_embeds"][:, :4, :])
    assert np.all(store["per_layer_inputs"][:, :2, :, :] == 0)
    np.testing.assert_array_equal(store["per_layer_inputs"][:, 2:, :, :], original["per_layer_inputs"][:, :4, :, :])
    assert np.all(store["position_ids"][:, :2] == 0)
    np.testing.assert_array_equal(store["position_ids"][:, 2:], original["position_ids"][:, :4])


def test_static_input_padding_left_trims_overlong_token_inputs() -> None:
    store = {
        "input_ids": np.arange(6, dtype=np.int64).reshape(1, 6),
        "attention_mask": np.ones((1, 6), dtype=np.int64),
    }

    component_bundle_runtime._pad_prepared_store_to_static_input_shapes(
        store,
        inputs_meta={"input_shapes": {"input_ids": [1, 4], "attention_mask": [1, 4]}},
        tokenizer=None,
    )

    np.testing.assert_array_equal(store["input_ids"], np.asarray([[2, 3, 4, 5]], dtype=np.int64))
    np.testing.assert_array_equal(store["attention_mask"], np.ones((1, 4), dtype=np.int64))


def test_static_input_padding_trims_overlong_audio_features() -> None:
    store = {
        "input_features": np.ones((1, 6, 2), dtype=np.float16),
        "input_features_mask": np.ones((1, 6), dtype=bool),
    }

    component_bundle_runtime._pad_prepared_store_to_static_input_shapes(
        store,
        inputs_meta={"input_shapes": {"input_features": [1, 4, 2], "input_features_mask": [1, 4]}},
        tokenizer=None,
    )

    assert store["input_features"].shape == (1, 4, 2)
    assert store["input_features_mask"].shape == (1, 4)


def test_gemma4_multimodal_headroom_uses_context_floor() -> None:
    prepared = hf_model.PreparedInputs(
        names=("input_ids", "attention_mask", "token_type_ids"),
        tensors=(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.ones((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
        ),
        metadata={"input_shapes": {"input_ids": [1, 3]}},
    )

    padded = hf_model._add_multimodal_generation_headroom(
        prepared,
        tokenizer=None,
        max_new_tokens=1,
        min_context_tokens=8,
    )

    assert padded.tensors[0].shape == (1, 8)
    assert padded.metadata["target_token_count"] == 8


def test_audio_waveform_loader_caps_duration(monkeypatch, tmp_path: Path) -> None:
    sample_rate = 16000
    audio_path = tmp_path / "long.wav"
    wavfile.write(audio_path, sample_rate, np.ones(sample_rate * 2, dtype=np.float32))

    monkeypatch.setenv("CACTUS_TRANSPILER_MAX_AUDIO_SECONDS", "0.5")
    waveform = audio_preprocess.load_audio_waveform(
        audio_path,
        target_sample_rate=sample_rate,
    )

    assert waveform.shape == (sample_rate // 2,)


def test_materialized_transpile_constants_embed_in_cactus_graph(tmp_path: Path) -> None:
    graph_path = tmp_path / "constant.cactus"
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)

    graph = hf_model.Graph()
    constant = graph.input(expected.shape, hf_model.Graph.FP16)
    graph.set_input(constant, expected)
    graph.mark_embedded_input(constant)
    graph.save(graph_path)

    assert {path.name for path in tmp_path.iterdir()} == {"constant.cactus"}
    loaded_graph = hf_model.Graph.load(graph_path)
    loaded_constant = component_bundle_runtime.Tensor(
        loaded_graph,
        constant.id,
        constant.shape,
        constant.dtype,
    )
    np.testing.assert_array_equal(loaded_constant.numpy(), expected)


def test_stateful_decode_graphs_are_reloaded_when_bundle_cache_hits(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "components": [
            {
                "component": "vision_encoder",
                "logical_inputs": ["pixel_values"],
                "logical_outputs": ["image_features"],
            },
            {
                "component": "decoder_prefill_chunk",
                "logical_inputs": ["inputs_embeds"],
                "logical_outputs": ["logits"],
            },
            {
                "component": "decoder_step",
                "logical_inputs": ["inputs_embeds"],
                "logical_outputs": ["logits"],
            },
        ],
    }
    calls: list[str] = []

    class FakeComponent:
        def __init__(self, component: str):
            self.component = component

    def fake_manifest(_bundle_dir_or_manifest):
        return tmp_path, manifest

    def fake_load_saved_component_graph(*, component_entry, **_kwargs):
        component = str(component_entry["component"])
        calls.append(component)
        return FakeComponent(component)

    component_bundle_runtime._COMPONENT_GRAPH_CACHE.clear()
    monkeypatch.setattr(component_bundle_runtime, "load_component_bundle_manifest", fake_manifest)
    monkeypatch.setattr(component_bundle_runtime, "load_saved_component_graph", fake_load_saved_component_graph)

    loaded, _ = component_bundle_runtime.load_saved_component_graphs(tmp_path)
    assert set(loaded) == {"vision_encoder", "decoder_prefill_chunk", "decoder_step"}
    assert calls == ["vision_encoder", "decoder_prefill_chunk", "decoder_step"]

    calls.clear()
    loaded, _ = component_bundle_runtime.load_saved_component_graphs(tmp_path)
    assert set(loaded) == {"vision_encoder", "decoder_prefill_chunk", "decoder_step"}
    assert calls == ["decoder_prefill_chunk", "decoder_step"]


def test_skipped_component_outputs_are_seeded_as_zeros() -> None:
    class FakeTensor:
        dtype = component_bundle_runtime.Graph.FP16
        shape = (1, 2, 3)

    class FakeComponent:
        component = "audio_encoder"
        outputs = [FakeTensor()]
        _output_names = ("audio_features",)

    store: dict[str, np.ndarray] = {}
    component_bundle_runtime._seed_skipped_component_outputs(
        store,
        component_graphs={"audio_encoder": FakeComponent()},  # type: ignore[dict-item]
        component_names=("audio_encoder",),
    )

    assert set(store) == {"audio_features"}
    assert store["audio_features"].shape == (1, 2, 3)
    assert store["audio_features"].dtype == np.float16
    assert np.all(store["audio_features"] == 0)


def test_gemma4_multimodal_bundle_uses_text_only_cached_path_without_media(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_text_only(**kwargs):
        calls.append(kwargs)
        return {"response": "ok", "decode_mode": "cached_step_text"}

    monkeypatch.setattr(component_bundle_runtime, "_run_gemma4_text_only_cached_bundle", fake_text_only)

    result = component_bundle_runtime._run_multimodal_causal_lm_bundle(
        component_graphs={
            "lm_encoder_step": object(),  # type: ignore[dict-item]
            "decoder_step": object(),  # type: ignore[dict-item]
        },
        manifest={"family": "gemma4", "task": "multimodal_causal_lm_logits", "inputs": {}},
        prompt="Hello",
        image_files=(),
        audio_file=None,
        torch_dtype=component_bundle_runtime.torch.float16,
        system_prompt=None,
        enable_thinking=False,
        max_new_tokens=1,
        stop_sequences=(),
    )

    assert result == {"response": "ok", "decode_mode": "cached_step_text"}
    assert calls and calls[0]["prompt"] == "Hello"


def test_gemma4_prompt_uses_chat_turn_format() -> None:
    class FakeTokenizer:
        def __call__(self, text, **_kwargs):
            return {"input_ids": [ord(char) for char in text]}

    ids = component_bundle_runtime._tokenize_bundle_prompt_for_manifest(
        {"family": "gemma4"},
        FakeTokenizer(),
        "Hello",
    )
    decoded = "".join(chr(value) for value in ids)

    assert decoded == "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"


def test_gemma4_stop_token_ids_include_turn_end() -> None:
    class FakeTokenizer:
        eos_token_id = None

        def convert_tokens_to_ids(self, token):
            return {"<turn|>": 106, "<eos>": 1}.get(token)

    assert component_bundle_runtime._bundle_stop_token_ids(
        manifest={"family": "gemma4"},
        tokenizer=FakeTokenizer(),
    ) == {1, 106}


def test_runtime_image_inputs_resize_to_static_square(tmp_path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return

    image_path = tmp_path / "tall.png"
    Image.new("RGB", (20, 40), color=(255, 0, 0)).save(image_path)

    images = multimodal_runtime._load_image_inputs((str(image_path),))
    lfm_images = component_bundle_runtime._load_image_inputs_for_runtime((str(image_path),))

    assert images[0].size == (256, 256)
    assert lfm_images[0].size == (256, 256)
