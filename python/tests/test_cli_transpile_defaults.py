from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from cactus import cli
from cactus.cli import assistant_bundle
from cactus.cli import convert as convert_cli
from cactus.transpile import hf_model


def _write_gemma4_multimodal_config(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hf_config.json").write_text(
        (
            '{"model_type":"gemma4",'
            '"architectures":["Gemma4ForConditionalGeneration"],'
            '"vision_config":{"model_type":"gemma4_vision"},'
            '"audio_config":{"model_type":"gemma4_audio"}}'
        ),
        encoding="utf-8",
    )


def _gemma4_multimodal_extra_args(model_dir: Path, artifact_dir: Path) -> list[str]:
    assets_dir = convert_cli.PROJECT_ROOT / "cactus-engine" / "tests" / "assets"
    return [
        "--weights-dir",
        str(model_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--task",
        "multimodal_causal_lm_logits",
        "--max-new-tokens",
        "32",
        "--torch-dtype",
        "bfloat16",
        "--component-pipeline",
        "on",
        "--prompt",
        convert_cli._DEFAULT_MULTIMODAL_PROMPT,
        "--components",
        "vision_encoder,audio_encoder,lm_encoder,decoder",
        "--image-file",
        str(assets_dir / "test_monkey.png"),
        "--audio-file",
        str(assets_dir / "test.wav"),
        "--trust-remote-code",
    ]


def test_public_convert_and_direct_transpile_dtype_defaults_are_intentional() -> None:
    parser = cli.create_parser()

    convert_args = parser.parse_args(["convert", "model"])

    assert convert_args.torch_dtype == "bfloat16"
    assert hf_model.DEFAULT_TORCH_DTYPE == "float16"


def test_cmd_convert_transpiles_into_same_weights_folder(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    args = parser.parse_args(["convert", "gemma4"])

    model_dir = tmp_path / "weights" / "gemma-4-e2b-it"
    calls: list[tuple[str, object]] = []

    def _fake_cq_main(command):
        calls.append(("cq", list(command)))
        _write_gemma4_multimodal_config(model_dir)
        return 0

    def _fake_cmd_transpile(transpile_args):
        calls.append(("transpile", transpile_args))
        assert transpile_args.model_id == "google/gemma-4-E2B-it"
        assert transpile_args.execute_after_transpile is False
        assert transpile_args.allow_unconverted_weights is False
        return 0

    monkeypatch.setattr(convert_cli, "get_weights_dir", lambda model_id: model_dir)
    monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)

    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert calls[0][0] == "cq"
    assert calls[0][1] == [
        "convert",
        "--model",
        "google/gemma-4-E2B-it",
        "--out",
        str(model_dir),
        "--bits",
        "4",
        "--device",
        "cpu",
        "--force",
    ]
    assert calls[1][0] == "transpile"
    assert calls[1][1].extra_args == _gemma4_multimodal_extra_args(model_dir, model_dir)
    assert not (model_dir / "transpile_entrypoints.json").exists()


def test_cmd_convert_honors_explicit_output_dir(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "custom"
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", str(output_dir)])

    cq_calls: list[list[str]] = []

    def _fake_cq_main(command):
        cq_calls.append(list(command))
        _write_gemma4_multimodal_config(output_dir)
        return 0

    transpile_calls: list[Namespace] = []

    def _fake_cmd_transpile(transpile_args):
        transpile_calls.append(transpile_args)
        return 0

    monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)

    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert cq_calls[0] == [
        "convert",
        "--model",
        "google/gemma-4-E2B-it",
        "--out",
        str(output_dir),
        "--bits",
        "4",
        "--device",
        "cpu",
        "--force",
    ]
    assert len(transpile_calls) == 1
    assert transpile_calls[0].extra_args == _gemma4_multimodal_extra_args(output_dir, output_dir)


def test_cmd_convert_supplies_default_audio_for_parakeet(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "parakeet"
    args = parser.parse_args(["convert", "nvidia/parakeet-tdt-0.6b-v3", str(output_dir)])

    def _fake_cq_main(command):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hf_config.json").write_text(
            '{"model_type":"parakeet_tdt","architectures":["ParakeetForTDT"]}',
            encoding="utf-8",
        )
        return 0

    transpile_calls: list[Namespace] = []

    def _fake_cmd_transpile(transpile_args):
        transpile_calls.append(transpile_args)
        return 0

    monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)

    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    extra_args = transpile_calls[0].extra_args
    assert extra_args[extra_args.index("--task") + 1] == "tdt_transcription"
    assert "--audio-file" in extra_args
    assert extra_args[extra_args.index("--audio-file") + 1].endswith("cactus-engine/tests/assets/test.wav")


def test_cmd_convert_supplies_default_audio_for_whisper(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "whisper"
    args = parser.parse_args(["convert", "whisper", str(output_dir)])

    def _fake_cq_main(command):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hf_config.json").write_text(
            '{"model_type":"whisper","architectures":["WhisperForConditionalGeneration"]}',
            encoding="utf-8",
        )
        return 0

    transpile_calls: list[Namespace] = []

    def _fake_cmd_transpile(transpile_args):
        transpile_calls.append(transpile_args)
        return 0

    monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)

    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    extra_args = transpile_calls[0].extra_args
    assert extra_args[extra_args.index("--task") + 1] == "seq2seq_transcription"
    assert "--audio-file" in extra_args


def test_cmd_convert_infers_text_tasks_for_qwen_and_lfm(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()

    for alias, model_type, arch in (
        ("qwen", "qwen3", "Qwen3ForCausalLM"),
        ("lfm", "lfm2_vl", "Lfm2VlForConditionalGeneration"),
    ):
        output_dir = tmp_path / alias
        args = parser.parse_args(["convert", alias, str(output_dir)])
        transpile_calls: list[Namespace] = []

        def _fake_cq_main(command, *, output_dir=output_dir, model_type=model_type, arch=arch):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "hf_config.json").write_text(
                f'{{"model_type":"{model_type}","architectures":["{arch}"]}}',
                encoding="utf-8",
            )
            return 0

        def _fake_cmd_transpile(transpile_args):
            transpile_calls.append(transpile_args)
            return 0

        monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)
        monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

        rc = convert_cli.cmd_convert(args)

        assert rc == 0
        extra_args = transpile_calls[0].extra_args
        assert extra_args[extra_args.index("--task") + 1] == "causal_lm_logits"
        assert "--audio-file" not in extra_args


def test_cmd_convert_keeps_gemma4_causal_component_pipeline_auto_without_assistant(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "gemma4"
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", str(output_dir)])
    transpile_calls: list[Namespace] = []

    def _fake_cq_main(command):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hf_config.json").write_text(
            '{"model_type":"gemma4","architectures":["Gemma4ForCausalLM"]}',
            encoding="utf-8",
        )
        return 0

    def _fake_cmd_transpile(transpile_args):
        transpile_calls.append(transpile_args)
        return 0

    monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)
    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    extra_args = transpile_calls[0].extra_args
    assert extra_args[extra_args.index("--task") + 1] == "causal_lm_logits"
    assert extra_args[extra_args.index("--component-pipeline") + 1] == "auto"


def test_cmd_convert_infers_multimodal_components_from_vision_config(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "lfm2-vl"
    args = parser.parse_args(["convert", "LiquidAI/LFM2-VL-450M", str(output_dir)])

    def _fake_cq_main(command):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hf_config.json").write_text(
            (
                '{"model_type":"lfm2_vl",'
                '"architectures":["Lfm2VlForConditionalGeneration"],'
                '"vision_config":{"model_type":"siglip2_vision_model"}}'
            ),
            encoding="utf-8",
        )
        return 0

    transpile_calls: list[Namespace] = []

    def _fake_cmd_transpile(transpile_args):
        transpile_calls.append(transpile_args)
        return 0

    monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)

    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert len(transpile_calls) == 1
    extra_args = transpile_calls[0].extra_args
    assert extra_args[extra_args.index("--task") + 1] == "multimodal_causal_lm_logits"
    assert extra_args[extra_args.index("--component-pipeline") + 1] == "on"
    assert extra_args[extra_args.index("--components") + 1] == "vision_encoder,lm_encoder,decoder"
    assert "--image-file" in extra_args
    assert "--audio-file" not in extra_args


def test_cmd_convert_transpiles_and_merges_assistant(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "main"
    args = parser.parse_args([
        "convert",
        "main/model",
        str(output_dir),
        "--assistant-model",
        "assistant/model",
        "--assistant-bits",
        "2",
        "--torch-dtype",
        "bfloat16",
        "--token",
        "hf_token",
        "--cache-dir",
        str(tmp_path / "hf-cache"),
    ])

    cq_calls: list[list[str]] = []

    def _fake_cq_main(command):
        cq_calls.append(list(command))
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_dir == output_dir:
            (out_dir / "hf_config.json").write_text('{"model_type":"gemma4"}', encoding="utf-8")
        (out_dir / "vocab.txt").write_text("0\tA\n", encoding="utf-8")
        (out_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        return 0

    transpile_calls: list[Namespace] = []

    def _fake_cmd_transpile(transpile_args):
        transpile_calls.append(transpile_args)
        artifact_dir = Path(transpile_args.extra_args[transpile_args.extra_args.index("--artifact-dir") + 1])
        component_name = "assistant" if "--components" in transpile_args.extra_args else "decoder"
        decoder_dir = artifact_dir / "components" / component_name
        decoder_dir.mkdir(parents=True, exist_ok=True)
        (decoder_dir / "graph.cactus").write_bytes(b"graph")
        (decoder_dir / "bound_constants").mkdir()
        (decoder_dir / "bound_constants" / "node_1.npy").write_bytes(b"npy")
        target_embedding_component = []
        component_order = [component_name]
        if component_name == "decoder":
            target_embedding_dir = artifact_dir / "components" / "target_embedding"
            target_embedding_dir.mkdir(parents=True, exist_ok=True)
            (target_embedding_dir / "graph.cactus").write_bytes(b"embedding")
            component_order.append("target_embedding")
            target_embedding_component.append(
                {
                    "component": "target_embedding",
                    "directory": "components/target_embedding",
                    "raw_ir": None,
                    "optimized_ir": None,
                    "graph": "components/target_embedding/graph.cactus",
                    "inputs": ["current_token_ids"],
                    "outputs": ["target_token_embedding"],
                    "logical_inputs": ["current_token_ids"],
                    "logical_outputs": ["target_token_embedding"],
                    "node_count": 1,
                    "weight_binding_count": 0,
                    "runtime_input_node_ids": [3],
                    "output_node_ids": [4],
                    "bound_constant_bindings": [],
                }
            )
        (artifact_dir / "components" / "manifest.json").write_text(
            json.dumps(
                {
                    "model_id": transpile_args.model_id,
                    "model_source": transpile_args.model_id,
                    "task": "causal_lm_logits",
                    "family": "generic",
                    "component_order": component_order,
                    "inputs": {},
                    "components": [
                        {
                            "component": component_name,
                            "directory": f"components/{component_name}",
                            "raw_ir": None,
                            "optimized_ir": None,
                            "graph": f"components/{component_name}/graph.cactus",
                            "inputs": ["input_ids"],
                            "outputs": ["logits"],
                            "logical_inputs": [],
                            "logical_outputs": [],
                            "node_count": 1,
                            "weight_binding_count": 0,
                            "runtime_input_node_ids": [0],
                            "output_node_ids": [2],
                            "bound_constant_bindings": [
                                {
                                    "node_id": 1,
                                    "value_id": "one",
                                    "path": f"components/{component_name}/bound_constants/node_1.npy",
                                    "kind": "saved_constant",
                                    "format": "npy",
                                    "precision": 2,
                                }
                            ],
                        }
                    ] + target_embedding_component,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(convert_cli, "cmd_transpile", _fake_cmd_transpile)

    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert cq_calls[0][cq_calls[0].index("--model") + 1] == "main/model"
    assert cq_calls[1][cq_calls[1].index("--model") + 1] == "assistant/model"
    assert cq_calls[1][cq_calls[1].index("--bits") + 1] == "2"
    assert cq_calls[0][cq_calls[0].index("--cache-dir") + 1] == str(tmp_path / "hf-cache")
    assert cq_calls[1][cq_calls[1].index("--cache-dir") + 1] == str(tmp_path / "hf-cache")
    assert transpile_calls[0].model_id == "main/model"
    assert transpile_calls[1].model_id == "assistant/model"
    assert transpile_calls[0].extra_args[transpile_calls[0].extra_args.index("--torch-dtype") + 1] == "bfloat16"
    assert transpile_calls[1].extra_args[transpile_calls[1].extra_args.index("--torch-dtype") + 1] == "bfloat16"
    assert transpile_calls[0].extra_args[transpile_calls[0].extra_args.index("--cache-dir") + 1] == str(tmp_path / "hf-cache")
    assert transpile_calls[1].extra_args[transpile_calls[1].extra_args.index("--cache-dir") + 1] == str(tmp_path / "hf-cache")
    assert transpile_calls[0].extra_args[transpile_calls[0].extra_args.index("--component-pipeline") + 1] == "on"
    assert transpile_calls[1].extra_args[transpile_calls[1].extra_args.index("--component-pipeline") + 1] == "on"
    assert transpile_calls[1].extra_args[transpile_calls[1].extra_args.index("--components") + 1] == "assistant"

    manifest = json.loads((output_dir / "components" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["component_order"] == ["decoder", "target_embedding", "assistant"]
    assert manifest["spec_decode"]["method"] == "single_position_mtp"
    assert manifest["spec_decode"]["target"]["verifier_logits"] == "verifier_logits"
    assert manifest["spec_decode"]["target"]["target_hidden_state"] == "target_hidden_state"
    assert manifest["spec_decode"]["target"]["target_token_embedding"] == "target_token_embedding"
    assert manifest["spec_decode"]["assistant"]["current_token_embedding"] == "current_token_embedding"
    assert manifest["spec_decode"]["assistant"]["logits_output"] == "logits_output"
    assert manifest["spec_decode"]["assistant"]["next_hidden_output"] == "next_hidden_output"
    assistant = [c for c in manifest["components"] if c["component"] == "assistant"][0]
    assert assistant["graph"] == "components/assistant/components/assistant/graph.cactus"
    assert assistant["bound_constant_bindings"][0]["path"] == (
        "components/assistant/components/assistant/bound_constants/node_1.npy"
    )
    assert (output_dir / "components" / "assistant" / "components" / "assistant" / "graph.cactus").exists()
    assert not (output_dir / "components" / "assistant" / "components" / "manifest.json").exists()


def test_cmd_convert_defaults_assistant_bits_to_main_bits(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "main"
    args = parser.parse_args([
        "convert",
        "main/model",
        str(output_dir),
        "--bits",
        "3",
        "--assistant-model",
        "assistant/model",
    ])

    def _fake_cq_main(command):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hf_config.json").write_text('{"model_type":"gemma4"}', encoding="utf-8")
        return 0

    monkeypatch.setattr(convert_cli, "cmd_transpile", lambda transpile_args: 0)

    captured: dict[str, object] = {}

    def _fake_package_assistant_for_convert(**kwargs):
        captured.update(kwargs)
        manifest_path = output_dir / "components" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        return manifest_path

    monkeypatch.setattr(convert_cli, "package_assistant_for_convert", _fake_package_assistant_for_convert)

    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert captured["assistant_bits"] == 3


def test_cmd_convert_rejects_assistant_for_non_gemma4(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "main"
    args = parser.parse_args([
        "convert",
        "main/model",
        str(output_dir),
        "--assistant-model",
        "assistant/model",
    ])

    def _fake_cq_main(command):
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hf_config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
        return 0

    transpile_calls: list[Namespace] = []
    monkeypatch.setattr(convert_cli, "cmd_transpile", lambda transpile_args: transpile_calls.append(transpile_args) or 0)
    monkeypatch.setattr(convert_cli, "run_cq_convert", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 1
    assert transpile_calls == []


def test_package_assistant_uses_default_prompt_when_unresolved_auto_task(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    assistant_dir = main_dir / "components" / "assistant"
    (main_dir / "components" / "decoder").mkdir(parents=True, exist_ok=True)
    (main_dir / "components" / "target_embedding").mkdir(parents=True, exist_ok=True)
    (main_dir / "components" / "decoder" / "graph.cactus").write_bytes(b"graph")
    (main_dir / "components" / "target_embedding" / "graph.cactus").write_bytes(b"graph")
    (main_dir / "components" / "manifest.json").write_text(
        json.dumps(
            {
                "task": "causal_lm_logits",
                "component_order": ["decoder", "target_embedding"],
                "components": [
                    {"component": "decoder", "graph": "components/decoder/graph.cactus", "bound_constant_bindings": []},
                    {"component": "target_embedding", "graph": "components/target_embedding/graph.cactus", "bound_constant_bindings": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    captured_extra_args: list[str] = []

    def _fake_transpile(namespace: Namespace) -> int:
        captured_extra_args.extend(namespace.extra_args)
        (assistant_dir / "components" / "assistant").mkdir(parents=True, exist_ok=True)
        (assistant_dir / "components" / "assistant" / "graph.cactus").write_bytes(b"graph")
        (assistant_dir / "components" / "manifest.json").write_text(
            json.dumps(
                {
                    "task": "causal_lm_logits",
                    "component_order": ["assistant"],
                    "components": [
                        {
                            "component": "assistant",
                            "directory": "components/assistant",
                            "graph": "components/assistant/graph.cactus",
                            "bound_constant_bindings": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return 0

    assistant_bundle.package_assistant_for_convert(
        cq_main=lambda _command: None,
        cmd_transpile=_fake_transpile,
        main_output_dir=main_dir,
        assistant_model_id="assistant/model",
        assistant_bits=4,
        device="cpu",
        prompt=None,  # type: ignore[arg-type]
        max_new_tokens=4,
        torch_dtype="bfloat16",
        token=None,
        cache_dir=None,
        trust_remote_code=False,
        local_files_only=False,
    )

    prompt_index = captured_extra_args.index("--prompt")
    assert captured_extra_args[prompt_index + 1] == "Hello"


def test_merge_assistant_bundle_rewrites_in_bundle_weight_paths(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    assistant_bundle_dir = main_dir / "components" / "assistant"
    assistant_weights = assistant_bundle_dir / "weights"
    assistant_weight = assistant_weights / "layer_0_attn.weights"

    for root in (main_dir, assistant_bundle_dir):
        decoder_dir = root / "components" / "decoder"
        decoder_dir.mkdir(parents=True, exist_ok=True)
        (decoder_dir / "graph.cactus").write_bytes(b"graph")
        (root / "vocab.txt").write_text("0\tA\n", encoding="utf-8")
        (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    assistant_weights.mkdir(parents=True, exist_ok=True)
    (assistant_weights / "vocab.txt").write_text("0\tA\n", encoding="utf-8")
    (assistant_weights / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    assistant_weight.write_bytes(b"weight")

    def _write_manifest(root: Path, *, binding_path: str | None = None) -> None:
        bindings = []
        is_main = root == main_dir
        if binding_path is not None:
            bindings.append(
                {
                    "node_id": 3,
                    "value_id": "w",
                    "path": binding_path,
                    "kind": "weight",
                    "source_name": "w",
                }
            )
        (root / "components" / "manifest.json").write_text(
            json.dumps(
                {
                    "model_id": root.name,
                    "model_source": root.name,
                    "task": "causal_lm_logits",
                    "family": "generic",
                    "component_order": ["decoder", "target_embedding"] if is_main else ["decoder"],
                    "inputs": {},
                    "components": [
                        {
                            "component": "decoder",
                            "directory": "components/decoder",
                            "raw_ir": None,
                            "optimized_ir": None,
                            "graph": "components/decoder/graph.cactus",
                            "inputs": ["input_ids"],
                            "outputs": ["logits"],
                            "logical_inputs": [],
                            "logical_outputs": [],
                            "node_count": 1,
                            "weight_binding_count": 0,
                            "runtime_input_node_ids": [0],
                            "output_node_ids": [2],
                            "bound_constant_bindings": bindings,
                        }
                    ] + ([
                        {
                            "component": "target_embedding",
                            "directory": "components/decoder",
                            "raw_ir": None,
                            "optimized_ir": None,
                            "graph": "components/decoder/graph.cactus",
                            "inputs": ["current_token_ids"],
                            "outputs": ["target_token_embedding"],
                            "logical_inputs": ["current_token_ids"],
                            "logical_outputs": ["target_token_embedding"],
                            "node_count": 1,
                            "weight_binding_count": 0,
                            "runtime_input_node_ids": [3],
                            "output_node_ids": [4],
                            "bound_constant_bindings": [],
                        }
                    ] if is_main else []),
                }
            ),
            encoding="utf-8",
        )

    _write_manifest(main_dir)
    _write_manifest(assistant_bundle_dir, binding_path=str(assistant_weight))

    assistant_bundle.merge_assistant_bundle(
        main_bundle_dir=main_dir,
        assistant_bundle_dir=assistant_bundle_dir,
        assistant_weights_dir=assistant_weights,
        assistant_model_id="assistant/model",
    )

    manifest = json.loads((main_dir / "components" / "manifest.json").read_text(encoding="utf-8"))
    assistant = [c for c in manifest["components"] if c["component"] == "assistant"][0]
    copied_path = assistant["bound_constant_bindings"][0]["path"]
    assert copied_path == "components/assistant/weights/layer_0_attn.weights"
    assert (main_dir / copied_path).read_bytes() == b"weight"
    assert not (assistant_bundle_dir / "components" / "manifest.json").exists()


def test_merge_assistant_bundle_reuses_target_embedding_binding(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    assistant_bundle_dir = main_dir / "components" / "assistant"
    assistant_weights = assistant_bundle_dir / "weights"
    target_embedding = main_dir / "token_embeddings.weights"
    assistant_embedding = assistant_weights / "token_embeddings.weights"

    for root in (main_dir, assistant_bundle_dir):
        decoder_dir = root / "components" / "decoder"
        decoder_dir.mkdir(parents=True, exist_ok=True)
        (decoder_dir / "graph.cactus").write_bytes(b"graph")
    (main_dir / "components" / "target_embedding").mkdir(parents=True, exist_ok=True)
    assistant_weights.mkdir(parents=True, exist_ok=True)
    target_embedding.write_bytes(b"target embedding")
    assistant_embedding.write_bytes(b"assistant duplicate")

    (main_dir / "components" / "manifest.json").write_text(
        json.dumps(
            {
                "task": "causal_lm_logits",
                "component_order": ["decoder", "target_embedding"],
                "components": [
                    {
                        "component": "decoder",
                        "directory": "components/decoder",
                        "graph": "components/decoder/graph.cactus",
                        "bound_constant_bindings": [],
                    },
                    {
                        "component": "target_embedding",
                        "directory": "components/target_embedding",
                        "graph": "components/target_embedding/graph.cactus",
                        "bound_constant_bindings": [
                            {
                                "node_id": 4,
                                "value_id": "embed",
                                "path": "token_embeddings.weights",
                                "kind": "embedding",
                                "source_name": "model.language_model.embed_tokens.weight",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (assistant_bundle_dir / "components" / "manifest.json").write_text(
        json.dumps(
            {
                "task": "causal_lm_logits",
                "component_order": ["assistant"],
                "components": [
                    {
                        "component": "assistant",
                        "directory": "components/decoder",
                        "graph": "components/decoder/graph.cactus",
                        "bound_constant_bindings": [
                            {
                                "node_id": 7,
                                "value_id": "embed",
                                "path": "token_embeddings.weights",
                                "kind": "embedding",
                                "source_name": "model.language_model.embed_tokens.weight",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assistant_bundle.merge_assistant_bundle(
        main_bundle_dir=main_dir,
        assistant_bundle_dir=assistant_bundle_dir,
        assistant_weights_dir=assistant_weights,
        assistant_model_id="assistant/model",
    )

    manifest = json.loads((main_dir / "components" / "manifest.json").read_text(encoding="utf-8"))
    assistant = [c for c in manifest["components"] if c["component"] == "assistant"][0]
    assert assistant["bound_constant_bindings"][0]["path"] == "token_embeddings.weights"
    assert (main_dir / assistant["bound_constant_bindings"][0]["path"]).read_bytes() == b"target embedding"


def test_merge_assistant_bundle_rewrites_weight_embedding_and_saved_constant_bindings(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    assistant_bundle_dir = main_dir / "components" / "assistant"
    assistant_weights = assistant_bundle_dir / "weights"
    assistant_decoder_dir = assistant_bundle_dir / "components" / "decoder"
    assistant_decoder_dir.mkdir(parents=True, exist_ok=True)
    assistant_weights.mkdir(parents=True, exist_ok=True)
    (main_dir / "components" / "decoder").mkdir(parents=True, exist_ok=True)
    (assistant_decoder_dir / "graph.cactus").write_bytes(b"graph")
    (main_dir / "components" / "decoder" / "graph.cactus").write_bytes(b"graph")
    for filename in ("layer.weights", "embedding.weights"):
        (assistant_weights / filename).write_bytes(filename.encode("utf-8"))
    constants_dir = assistant_decoder_dir / "bound_constants"
    constants_dir.mkdir()
    (constants_dir / "node_9.npy").write_bytes(b"constant")

    def _component_manifest(component: str, bindings: list[dict[str, object]], *, include_target_embedding: bool = False) -> dict[str, object]:
        return {
            "task": "causal_lm_logits",
            "component_order": [component, "target_embedding"] if include_target_embedding else [component],
            "components": [
                {
                    "component": component,
                    "directory": "components/decoder",
                    "graph": "components/decoder/graph.cactus",
                    "bound_constant_bindings": bindings,
                }
            ] + ([
                {
                    "component": "target_embedding",
                    "directory": "components/decoder",
                    "graph": "components/decoder/graph.cactus",
                    "bound_constant_bindings": [],
                }
            ] if include_target_embedding else []),
        }

    (main_dir / "components" / "manifest.json").write_text(
        json.dumps(_component_manifest("decoder", [], include_target_embedding=True)),
        encoding="utf-8",
    )
    (assistant_bundle_dir / "components" / "manifest.json").write_text(
        json.dumps(
            _component_manifest(
                "assistant",
                [
                    {"node_id": 1, "value_id": "w", "path": "layer.weights", "kind": "weight"},
                    {"node_id": 2, "value_id": "e", "path": "embedding.weights", "kind": "embedding"},
                    {
                        "node_id": 9,
                        "value_id": "c",
                        "path": "components/decoder/bound_constants/node_9.npy",
                        "kind": "saved_constant",
                    },
                ],
            )
        ),
        encoding="utf-8",
    )

    assistant_bundle.merge_assistant_bundle(
        main_bundle_dir=main_dir,
        assistant_bundle_dir=assistant_bundle_dir,
        assistant_weights_dir=assistant_weights,
        assistant_model_id="assistant/model",
    )

    manifest = json.loads((main_dir / "components" / "manifest.json").read_text(encoding="utf-8"))
    assistant = [c for c in manifest["components"] if c["component"] == "assistant"][0]
    paths_by_kind = {binding["kind"]: binding["path"] for binding in assistant["bound_constant_bindings"]}
    assert paths_by_kind == {
        "weight": "components/assistant/weights/layer.weights",
        "embedding": "components/assistant/weights/embedding.weights",
        "saved_constant": "components/assistant/components/decoder/bound_constants/node_9.npy",
    }


def test_merge_assistant_bundle_fails_on_missing_binding_path(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    assistant_bundle_dir = main_dir / "components" / "assistant"
    assistant_weights = assistant_bundle_dir / "weights"
    for root in (main_dir, assistant_bundle_dir):
        (root / "components" / "decoder").mkdir(parents=True, exist_ok=True)
        (root / "components" / "decoder" / "graph.cactus").write_bytes(b"graph")
    assistant_weights.mkdir(parents=True, exist_ok=True)
    (main_dir / "components" / "target_embedding").mkdir(parents=True, exist_ok=True)
    (main_dir / "components" / "target_embedding" / "graph.cactus").write_bytes(b"graph")
    (main_dir / "components" / "manifest.json").write_text(
        json.dumps(
            {
                "task": "causal_lm_logits",
                "component_order": ["decoder", "target_embedding"],
                "components": [
                    {"component": "decoder", "graph": "components/decoder/graph.cactus", "bound_constant_bindings": []},
                    {"component": "target_embedding", "graph": "components/target_embedding/graph.cactus", "bound_constant_bindings": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    (assistant_bundle_dir / "components" / "manifest.json").write_text(
        json.dumps(
            {
                "task": "causal_lm_logits",
                "component_order": ["assistant"],
                "components": [
                        {
                            "component": "assistant",
                            "directory": "components/decoder",
                            "graph": "components/decoder/graph.cactus",
                            "bound_constant_bindings": [
                            {"node_id": 1, "value_id": "w", "path": "missing.weights", "kind": "weight"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        assistant_bundle.merge_assistant_bundle(
            main_bundle_dir=main_dir,
            assistant_bundle_dir=assistant_bundle_dir,
            assistant_weights_dir=assistant_weights,
            assistant_model_id="assistant/model",
        )
    except RuntimeError as exc:
        assert "assistant binding path does not exist: missing.weights" in str(exc)
    else:
        raise AssertionError("missing assistant binding should fail packaging")


def test_merge_assistant_bundle_allows_assistant_without_tokenizer_files(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    assistant_bundle_dir = main_dir / "components" / "assistant"
    assistant_weights = assistant_bundle_dir / "weights"

    for root in (main_dir, assistant_bundle_dir):
        decoder_dir = root / "components" / "decoder"
        decoder_dir.mkdir(parents=True, exist_ok=True)
        (decoder_dir / "graph.cactus").write_bytes(b"graph")
        is_main = root == main_dir
        (root / "components" / "manifest.json").write_text(
            json.dumps(
                {
                    "task": "causal_lm_logits",
                    "component_order": ["decoder", "target_embedding"] if is_main else ["decoder"],
                    "components": [
                        {
                            "component": "decoder",
                            "directory": "components/decoder",
                            "graph": "components/decoder/graph.cactus",
                            "bound_constant_bindings": [],
                        }
                    ] + ([
                        {
                            "component": "target_embedding",
                            "directory": "components/decoder",
                            "graph": "components/decoder/graph.cactus",
                            "bound_constant_bindings": [],
                        }
                    ] if is_main else []),
                }
            ),
            encoding="utf-8",
        )
    main_dir.joinpath("vocab.txt").write_text("0\tA\n", encoding="utf-8")
    main_dir.joinpath("merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    assistant_weights.mkdir(parents=True, exist_ok=True)

    manifest_path = assistant_bundle.merge_assistant_bundle(
        main_bundle_dir=main_dir,
        assistant_bundle_dir=assistant_bundle_dir,
        assistant_weights_dir=assistant_weights,
        assistant_model_id="assistant/model",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["component_order"] == ["decoder", "target_embedding", "assistant"]


def test_assistant_tokenizer_compatibility_checks_json_sidecars(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    assistant_dir = tmp_path / "assistant"
    main_dir.mkdir()
    assistant_dir.mkdir()
    (main_dir / "tokenizer.json").write_text('{"model":"main"}', encoding="utf-8")
    (assistant_dir / "tokenizer.json").write_text('{"model":"assistant"}', encoding="utf-8")

    try:
        assistant_bundle._validate_assistant_tokenizer_compatibility(main_dir, assistant_dir)
    except RuntimeError as exc:
        assert "tokenizer.json differs" in str(exc)
    else:
        raise AssertionError("tokenizer.json mismatch should fail assistant packaging")


def test_transpile_config_detection_uses_custom_cache_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hf-cache"
    snapshot = cache_dir / "models--org--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"lfm2_vl"}', encoding="utf-8")

    config = hf_model._load_config_json(
        "org/model",
        cache_dir=str(cache_dir),
        local_files_only=True,
    )

    assert config["model_type"] == "lfm2_vl"


def test_cli_no_longer_registers_transpile_command() -> None:
    parser = cli.create_parser()
    try:
        parser.parse_args(["transpile", "gemma4"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("transpile should no longer be a public CLI command")
