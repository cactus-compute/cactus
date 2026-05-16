from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from cactus import cli
from cactus.cli import run as run_cli
from cactus.cli import transpile as transpile_cli
from cactus.transpile import audio_preprocess
from cactus.transpile import component_bundle_runtime
from cactus.transpile import multimodal_runtime
from cactus.transpile import hf_model


def test_cactus_run_detects_transpiled_bundle_and_uses_main_style_audio(monkeypatch, tmp_path: Path, capsys) -> None:
    bundle_dir = tmp_path / "bundle"
    components_dir = bundle_dir / "components"
    components_dir.mkdir(parents=True)
    (components_dir / "manifest.json").write_text(
        '{"model_id":"example/model","family":"generic","task":"causal_lm_logits","components":[]}',
        encoding="utf-8",
    )
    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"RIFF")

    calls = []

    def _fake_run_transpiled(args):
        calls.append(args)
        print("hello from transpiled")
        return 0

    monkeypatch.setattr(transpile_cli, "cmd_run_transpiled", _fake_run_transpiled)

    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "run",
            str(bundle_dir),
            "--audio",
            str(audio_file),
            "--prompt",
            "Hello",
        ]
    )

    rc = run_cli.cmd_run(args)

    assert rc == 0
    assert len(calls) == 1
    forwarded = calls[0]
    assert forwarded.bundle_dir == str(bundle_dir)
    assert forwarded.audio == str(audio_file.resolve())
    assert forwarded.audio_file == str(audio_file.resolve())
    assert forwarded.prompt == "Hello"
    assert forwarded._transpiled_from_run is True
    captured = capsys.readouterr().out
    assert "Starting Cactus Chat with model:" in captured
    assert "hello from transpiled" in captured


def test_run_transpiled_human_result_prints_response(capsys) -> None:
    transpile_cli._print_transpiled_run_result(
        {
            "response": "  generated text  ",
            "transcript": "not used",
        }
    )

    assert capsys.readouterr().out == "generated text\n"


def test_run_transpiled_once_deduplicates_image_aliases(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"not really an image")
    calls: list[dict[str, object]] = []

    def fake_runner(bundle_dir, **kwargs):
        calls.append({"bundle_dir": bundle_dir, **kwargs})
        return {"response": "ok"}

    args = type(
        "Args",
        (),
        {
            "image": str(image_path),
            "image_file": [str(image_path)],
            "image_files": [str(image_path)],
            "audio": None,
            "audio_file": None,
            "prompt": "describe",
            "input_ids": None,
            "weights_dir": None,
            "system": None,
            "thinking": False,
            "max_new_tokens": 1,
            "stop_sequence": [],
        },
    )()

    result = transpile_cli._run_transpiled_once(args, fake_runner, bundle_dir="bundle")

    assert result == {"response": "ok"}
    assert calls[0]["image_files"] == (str(image_path.resolve()),)


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


def test_materialized_transpile_constants_are_cactus_tensor_files(tmp_path: Path) -> None:
    tensor_path = tmp_path / "constant.weights"
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)

    hf_model._write_cactus_constant_tensor(
        output_path=tensor_path,
        value=expected,
        precision=int(hf_model.Graph.FP16),
    )

    assert {path.name for path in tmp_path.iterdir()} == {"constant.weights"}
    loaded = component_bundle_runtime._open_cactus_tensor_file(tensor_path)
    assert loaded.precision == int(hf_model.Graph.FP16)
    assert tuple(loaded.shape) == (2, 3)
    np.testing.assert_array_equal(loaded.data.reshape(loaded.shape), expected)


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


def test_media_turns_do_not_replay_text_history() -> None:
    history = [
        ("user", "Tell me about mountains"),
        ("assistant", "Mountains are tall landforms." * 20),
    ]

    assert transpile_cli._history_for_transpiled_turn(history, has_media=False) is history
    assert transpile_cli._history_for_transpiled_turn(history, has_media=True) == []


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
