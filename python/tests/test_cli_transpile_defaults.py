from __future__ import annotations

from pathlib import Path

from cactus import cli
from cactus.cli import convert as convert_cli
import cactus.cli.model as model_mod


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
    assets_dir = Path(model_mod.__file__).resolve().parent.parent / "assets"
    return [
        "--weights-dir",
        str(model_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--task",
        "multimodal_causal_lm_logits",
        "--max-new-tokens",
        "512",
        "--component-pipeline",
        "on",
        "--prompt",
        model_mod._DEFAULT_MULTIMODAL_PROMPT,
        "--components",
        "vision_encoder,audio_encoder,lm_encoder,decoder",
        "--image-file",
        str(assets_dir / "test_monkey.png"),
        "--audio-file",
        str(assets_dir / "test.wav"),
        "--trust-remote-code",
    ]


def _patch_transpile(monkeypatch, capture):
    import cactus.cli.transpile as transpile_mod

    def _fake_run_transpile(model_id, *, extra_args, execute_after_transpile=False,
                            allow_unconverted_weights=False):
        capture.append({
            "model_id": model_id,
            "extra_args": list(extra_args),
            "execute_after_transpile": execute_after_transpile,
            "allow_unconverted_weights": allow_unconverted_weights,
        })
        return 0

    monkeypatch.setattr(transpile_mod, "run_transpile", _fake_run_transpile)
    monkeypatch.setattr(model_mod, "run_transpile", _fake_run_transpile, raising=False)


def test_cmd_convert_transpiles_into_same_weights_folder(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", "--reconvert"])

    model_dir = tmp_path / "weights" / "gemma-4-e2b-it"
    cq_calls: list[list[str]] = []
    transpile_calls: list[dict] = []

    def _fake_cq_main(command):
        cq_calls.append(list(command))
        _write_gemma4_multimodal_config(model_dir)
        return 0

    monkeypatch.setattr(convert_cli, "get_weights_dir", lambda model_id: model_dir)
    _patch_transpile(monkeypatch, transpile_calls)

    import cactus.convert.cli as cq_cli
    monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert cq_calls[0] == [
        "convert",
        "--model",
        "google/gemma-4-E2B-it",
        "--out",
        str(model_dir),
        "--bits",
        "4",
        "--force",
    ]
    assert len(transpile_calls) == 1
    call = transpile_calls[0]
    assert call["model_id"] == "google/gemma-4-E2B-it"
    assert call["execute_after_transpile"] is False
    assert call["allow_unconverted_weights"] is False
    assert call["extra_args"] == _gemma4_multimodal_extra_args(model_dir, model_dir)
    assert not (model_dir / "transpile_entrypoints.json").exists()


def test_cmd_convert_honors_explicit_output_dir(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "custom"
    args = parser.parse_args(["convert", "google/gemma-4-E2B-it", str(output_dir), "--reconvert"])

    cq_calls: list[list[str]] = []
    transpile_calls: list[dict] = []

    def _fake_cq_main(command):
        cq_calls.append(list(command))
        _write_gemma4_multimodal_config(output_dir)
        return 0

    _patch_transpile(monkeypatch, transpile_calls)

    import cactus.convert.cli as cq_cli
    monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

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
        "--force",
    ]
    assert len(transpile_calls) == 1
    assert transpile_calls[0]["extra_args"] == _gemma4_multimodal_extra_args(output_dir, output_dir)


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

    transpile_calls: list[dict] = []
    _patch_transpile(monkeypatch, transpile_calls)

    import cactus.convert.cli as cq_cli
    monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    extra_args = transpile_calls[0]["extra_args"]
    assert extra_args[extra_args.index("--task") + 1] == "tdt_transcription"
    assert "--audio-file" in extra_args
    assert extra_args[extra_args.index("--audio-file") + 1].endswith("python/cactus/assets/test.wav")


def test_cmd_convert_supplies_default_audio_for_whisper(monkeypatch, tmp_path: Path) -> None:
    parser = cli.create_parser()
    output_dir = tmp_path / "whisper"
    args = parser.parse_args(["convert", "openai/whisper-small", str(output_dir)])

    def _fake_cq_main(command):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hf_config.json").write_text(
            '{"model_type":"whisper","architectures":["WhisperForConditionalGeneration"]}',
            encoding="utf-8",
        )
        return 0

    transpile_calls: list[dict] = []
    _patch_transpile(monkeypatch, transpile_calls)

    import cactus.convert.cli as cq_cli
    monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    extra_args = transpile_calls[0]["extra_args"]
    assert extra_args[extra_args.index("--task") + 1] == "seq2seq_transcription"
    assert "--audio-file" in extra_args


def test_cmd_convert_infers_multimodal_for_qwen_and_lfm(monkeypatch, tmp_path: Path) -> None:
    """Qwen/LFM models WITH vision_config get multimodal task."""
    parser = cli.create_parser()

    import cactus.convert.cli as cq_cli

    for model_id, model_type, arch, vision, expected_components in (
        (
            "Qwen/Qwen3-1.7B",
            "qwen3",
            "Qwen3ForCausalLM",
            "qwen3_vision",
            "vision_encoder,lm_encoder,decoder,lm_encoder_text_chunk,decoder_prefill_chunk,lm_encoder_step,decoder_media_step,decoder_step",
        ),
        (
            "LiquidAI/LFM2-VL-450M",
            "lfm2_vl",
            "Lfm2VlForConditionalGeneration",
            "siglip2_vision_model",
            "vision_encoder,lm_encoder,decoder",
        ),
    ):
        dir_name = model_id.split("/")[-1].lower() + "_mm"
        output_dir = tmp_path / dir_name
        args = parser.parse_args(["convert", model_id, str(output_dir), "--reconvert"])
        transpile_calls: list[dict] = []

        def _fake_cq_main(command, *, output_dir=output_dir, model_type=model_type, arch=arch, vision=vision):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "hf_config.json").write_text(
                f'{{"model_type":"{model_type}","architectures":["{arch}"],'
                f'"vision_config":{{"model_type":"{vision}"}}}}',
                encoding="utf-8",
            )
            return 0

        _patch_transpile(monkeypatch, transpile_calls)
        monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

        rc = convert_cli.cmd_convert(args)

        assert rc == 0
        extra_args = transpile_calls[0]["extra_args"]
        assert extra_args[extra_args.index("--task") + 1] == "multimodal_causal_lm_logits"
        assert extra_args[extra_args.index("--component-pipeline") + 1] == "on"
        assert extra_args[extra_args.index("--components") + 1] == expected_components
        assert "--image-file" in extra_args
        assert "--audio-file" not in extra_args


def test_cmd_convert_infers_causal_lm_for_qwen_and_lfm_without_vision(monkeypatch, tmp_path: Path) -> None:
    """Qwen/LFM models WITHOUT vision_config get causal_lm_logits."""
    parser = cli.create_parser()

    import cactus.convert.cli as cq_cli

    for model_id, model_type, arch in (
        ("Qwen/Qwen3-1.7B", "qwen3", "Qwen3ForCausalLM"),
        ("LiquidAI/LFM2-VL-450M", "lfm2_vl", "Lfm2VlForConditionalGeneration"),
    ):
        dir_name = model_id.split("/")[-1].lower()
        output_dir = tmp_path / dir_name
        args = parser.parse_args(["convert", model_id, str(output_dir), "--reconvert"])
        transpile_calls: list[dict] = []

        def _fake_cq_main(command, *, output_dir=output_dir, model_type=model_type, arch=arch):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "hf_config.json").write_text(
                f'{{"model_type":"{model_type}","architectures":["{arch}"]}}',
                encoding="utf-8",
            )
            return 0

        _patch_transpile(monkeypatch, transpile_calls)
        monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

        rc = convert_cli.cmd_convert(args)

        assert rc == 0
        extra_args = transpile_calls[0]["extra_args"]
        assert extra_args[extra_args.index("--task") + 1] == "causal_lm_logits"


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

    transpile_calls: list[dict] = []
    _patch_transpile(monkeypatch, transpile_calls)

    import cactus.convert.cli as cq_cli
    monkeypatch.setattr(cq_cli, "main", _fake_cq_main)

    rc = convert_cli.cmd_convert(args)

    assert rc == 0
    assert len(transpile_calls) == 1
    extra_args = transpile_calls[0]["extra_args"]
    assert extra_args[extra_args.index("--task") + 1] == "multimodal_causal_lm_logits"
    assert extra_args[extra_args.index("--component-pipeline") + 1] == "on"
    assert extra_args[extra_args.index("--components") + 1] == "vision_encoder,lm_encoder,decoder"
    assert "--image-file" in extra_args
    assert "--audio-file" not in extra_args


def test_cli_no_longer_registers_transpile_command() -> None:
    parser = cli.create_parser()
    try:
        parser.parse_args(["transpile", "gemma4"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("transpile should no longer be a public CLI command")
