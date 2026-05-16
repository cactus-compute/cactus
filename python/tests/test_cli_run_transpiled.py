from __future__ import annotations

from pathlib import Path

import numpy as np

from cactus import cli
from cactus.cli import run as run_cli
from cactus.cli import transpile as transpile_cli
from cactus.transpile import component_bundle_runtime
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
