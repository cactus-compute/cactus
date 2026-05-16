from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from cactus.cli import convert as convert_cli


def test_convert_with_assistant_requests_cached_target_verifier_components(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "main"
    args = Namespace(
        model_name="main/model",
        output_dir=str(output_dir),
        bits=4,
        token=None,
        cache_dir=None,
        device="cpu",
        local_files_only=False,
        task="auto",
        prompt=None,
        torch_dtype="bfloat16",
        image_file=[],
        audio_file=None,
        component_pipeline="auto",
        components=None,
        assistant_model="assistant/model",
        assistant_bits=None,
        max_new_tokens=32,
        trust_remote_code=False,
        system_prompt=None,
    )

    def fake_cq_convert(command):
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hf_config.json").write_text('{"model_type":"gemma4"}', encoding="utf-8")

    transpile_calls: list[Namespace] = []

    def fake_cmd_transpile(transpile_args):
        transpile_calls.append(transpile_args)
        return 0

    monkeypatch.setattr(convert_cli, "run_cq_convert", fake_cq_convert)
    monkeypatch.setattr(convert_cli, "cmd_transpile", fake_cmd_transpile)
    monkeypatch.setattr(
        convert_cli,
        "package_assistant_for_convert",
        lambda **_: output_dir / "components" / "manifest.json",
    )

    assert convert_cli.cmd_convert(args) == 0
    assert len(transpile_calls) == 1
    extra_args = transpile_calls[0].extra_args
    assert extra_args[extra_args.index("--component-pipeline") + 1] == "on"
    assert extra_args[extra_args.index("--components") + 1] == (
        "decoder,target_embedding,decoder_prefill_chunk,decoder_step,decoder_verify_m2,decoder_verify_m3,decoder_verify_m4"
    )
