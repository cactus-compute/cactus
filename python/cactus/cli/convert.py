from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .assistant_bundle import package_assistant_for_convert
from .common import (
    GREEN,
    PROJECT_ROOT,
    RED,
    YELLOW,
    get_weights_dir,
    print_color,
)
from .transpile import cmd_transpile, resolve_model_id_alias
from cactus.transpile.component_plan import infer_component_plan_from_output


_DEFAULT_MULTIMODAL_PROMPT = (
    "Respond with 2 lines. The first should be a description of the image, "
    "and the second should be a transcription of the audio"
)
_DEFAULT_TEXT_PROMPT = "Hello"
_ASSISTANT_TARGET_COMPONENTS = (
    "decoder",
    "target_embedding",
    "decoder_prefill_chunk",
    "decoder_step",
    "decoder_verify_m2",
    "decoder_verify_m3",
    "decoder_verify_m4",
)


@dataclass(frozen=True)
class _ConvertTranspileSpec:
    task: str
    components: tuple[str, ...] = ()
    needs_image: bool = False
    needs_audio: bool = False
    force_component_pipeline: bool = False


def _default_multimodal_asset_args() -> tuple[list[str], str | None]:
    assets_dir = PROJECT_ROOT / "cactus-engine" / "tests" / "assets"
    image_file = assets_dir / "test_monkey.png"
    audio_file = assets_dir / "test.wav"
    image_args = [str(image_file)] if image_file.exists() else []
    audio_arg = str(audio_file) if audio_file.exists() else None
    return image_args, audio_arg


def _default_audio_asset_arg() -> str | None:
    _, audio_file = _default_multimodal_asset_args()
    return audio_file


def _transpile_spec_for_convert(*, task: str, plan) -> _ConvertTranspileSpec:
    if task != "auto":
        if plan is not None and task == plan.task:
            return _ConvertTranspileSpec(
                task=task,
                components=tuple(plan.components or ()),
                needs_image=bool(plan.needs_image),
                needs_audio=bool(plan.needs_audio),
                force_component_pipeline=bool(plan.force_component_pipeline),
            )
        return _ConvertTranspileSpec(
            task=task,
            needs_image=task == "multimodal_causal_lm_logits",
            needs_audio=task
            in {"tdt_transcription", "seq2seq_transcription", "ctc_logits", "encoder_hidden_states", "multimodal_causal_lm_logits"},
            force_component_pipeline=task in {"tdt_transcription", "seq2seq_transcription", "multimodal_causal_lm_logits"},
        )

    if plan is None:
        return _ConvertTranspileSpec(task="causal_lm_logits")

    return _ConvertTranspileSpec(
        task=plan.task,
        components=tuple(plan.components or ()),
        needs_image=bool(plan.needs_image),
        needs_audio=bool(plan.needs_audio),
        force_component_pipeline=bool(plan.force_component_pipeline),
    )


def _remove_stale_transpile_artifacts(output_dir: str | Path) -> None:
    root = Path(output_dir)
    for relative in (
        "components",
        "transpile_entrypoints.json",
        "raw_ir.json",
        "optimized_ir.json",
        "graph.cactus",
        "graph_bindings.json",
        "result.json",
    ):
        path = root / relative
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for pattern in ("raw_ir_*.json", "optimized_ir_*.json"):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()


def _converted_model_type(output_dir: str | os.PathLike[str]) -> str:
    root = Path(output_dir)
    hf_config = root / "hf_config.json"
    if hf_config.exists():
        config = json.loads(hf_config.read_text(encoding="utf-8"))
        model_type = config.get("model_type")
        if model_type:
            return str(model_type).strip().lower()

    config_txt = root / "config.txt"
    if config_txt.exists():
        for line in config_txt.read_text(encoding="utf-8").splitlines():
            if line.startswith("model_type="):
                return line.split("=", 1)[1].strip().lower()
    return ""


def _supports_assistant_model(output_dir: str | os.PathLike[str], model_id: str) -> bool:
    model_type = _converted_model_type(output_dir)
    lowered_id = model_id.lower()
    return model_type == "gemma4" or "gemma-4" in lowered_id or "gemma4" in lowered_id


def run_cq_convert(command: list[str]) -> None:
    env = os.environ.copy()
    python_root = str(PROJECT_ROOT / "python")
    env["PYTHONPATH"] = python_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, "-c", "from cactus.convert.cli import main; main()", *command],
        cwd=PROJECT_ROOT,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CQ conversion failed with exit code {result.returncode}")


def cmd_convert(args):
    """Convert a HuggingFace model to CQ format and transpile it in place."""
    model_id = resolve_model_id_alias(args.model_name)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(get_weights_dir(model_id))

    bits = getattr(args, "bits", 4) or 4
    token = getattr(args, "token", None)
    cache_dir = getattr(args, "cache_dir", None)

    try:
        device = getattr(args, "device", "cpu") or "cpu"

        cq_args = [
            "convert",
            "--model",
            model_id,
            "--out",
            str(output_dir),
            "--bits",
            str(bits),
            "--device",
            str(device),
        ]
        if token:
            cq_args.extend(["--token", token])
        if cache_dir:
            cq_args.extend(["--cache-dir", cache_dir])
        if getattr(args, "local_files_only", False):
            cq_args.append("--local-files-only")
        cq_args.append("--force")

        run_cq_convert(cq_args)

        task = getattr(args, "task", "auto") or "auto"
        prompt = getattr(args, "prompt", None)
        torch_dtype = getattr(args, "torch_dtype", "bfloat16") or "bfloat16"
        image_files = [str(path) for path in (getattr(args, "image_file", None) or []) if str(path).strip()]
        audio_file = getattr(args, "audio_file", None)

        plan = infer_component_plan_from_output(str(output_dir), model_id=model_id)
        spec = _transpile_spec_for_convert(task=task, plan=plan)

        _remove_stale_transpile_artifacts(output_dir)

        spec_prompt = prompt
        spec_image_files = list(image_files)
        spec_audio_file = audio_file
        component_pipeline = getattr(args, "component_pipeline", "auto") or "auto"
        components = getattr(args, "components", None)

        if spec_prompt is None and spec.task == "multimodal_causal_lm_logits":
            spec_prompt = _DEFAULT_MULTIMODAL_PROMPT
        elif spec_prompt is None and spec.task == "causal_lm_logits":
            spec_prompt = _DEFAULT_TEXT_PROMPT

        needs_image = False
        needs_audio = False
        if spec.task == "multimodal_causal_lm_logits":
            needs_image = bool(spec.needs_image)
            needs_audio = bool(spec.needs_audio)
            if not needs_image and not needs_audio:
                needs_image = bool(spec_image_files)
                needs_audio = bool(spec_audio_file)
            if (needs_image and not spec_image_files) or (needs_audio and not spec_audio_file):
                default_images, default_audio = _default_multimodal_asset_args()
                if needs_image and not spec_image_files:
                    spec_image_files = default_images
                if needs_audio and not spec_audio_file:
                    spec_audio_file = default_audio
                print_color(
                    YELLOW,
                    "Multimodal transpile needs representative media shapes; "
                    "using bundled tiny test assets.",
                )
            if needs_image and not spec_image_files:
                print_color(
                    RED,
                    "Multimodal transpile requires --image-file for this model.",
                )
                return 1
            if needs_audio and not spec_audio_file:
                print_color(
                    RED,
                    "Multimodal transpile requires --audio-file for this model.",
                )
                return 1
        if component_pipeline == "auto" and spec.force_component_pipeline:
            component_pipeline = "on"
        if components is None and spec.components:
            components = ",".join(spec.components)

        used_default_audio = False
        if spec.task in {"tdt_transcription", "seq2seq_transcription", "ctc_logits", "encoder_hidden_states"} and not spec_audio_file:
            spec_audio_file = _default_audio_asset_arg()
            used_default_audio = spec_audio_file is not None
        if spec.task in {"tdt_transcription", "seq2seq_transcription", "ctc_logits", "encoder_hidden_states"} and used_default_audio:
            print_color(
                YELLOW,
                f"{spec.task} transpile needs a representative audio shape; "
                "using bundled tiny test audio asset.",
            )
        elif spec.task in {"tdt_transcription", "seq2seq_transcription", "ctc_logits", "encoder_hidden_states"} and not spec_audio_file:
            print_color(RED, f"{spec.task} transpile requires --audio-file.")
            return 1

        artifact_dir = Path(output_dir)

        assistant_model = getattr(args, "assistant_model", None)
        if assistant_model and spec.task != "causal_lm_logits":
            print_color(RED, "--assistant-model is currently supported only for causal_lm_logits bundles.")
            return 1
        if assistant_model and not _supports_assistant_model(output_dir, model_id):
            print_color(RED, "--assistant-model currently supports only Gemma4 causal_lm_logits bundles.")
            return 1
        if assistant_model and component_pipeline == "auto":
            component_pipeline = "on"
        if assistant_model and components is None:
            components = ",".join(_ASSISTANT_TARGET_COMPONENTS)

        extra_args = [
            "--weights-dir",
            str(output_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--task",
            spec.task,
            "--max-new-tokens",
            str(getattr(args, "max_new_tokens", 32) or 32),
            "--torch-dtype",
            str(torch_dtype),
            "--component-pipeline",
            component_pipeline,
        ]
        if spec_prompt is not None:
            extra_args.extend(["--prompt", spec_prompt])
        if components:
            extra_args.extend(["--components", str(components)])
        for image_file in spec_image_files:
            extra_args.extend(["--image-file", image_file])
        if spec_audio_file:
            extra_args.extend(["--audio-file", str(spec_audio_file)])
        if getattr(args, "system_prompt", None):
            extra_args.extend(["--system-prompt", str(args.system_prompt)])
        if token:
            extra_args.extend(["--token", token])
        if cache_dir:
            extra_args.extend(["--cache-dir", cache_dir])
        if getattr(args, "trust_remote_code", False) or spec.task == "multimodal_causal_lm_logits":
            extra_args.append("--trust-remote-code")
        if getattr(args, "local_files_only", False):
            extra_args.append("--local-files-only")

        transpile_args = argparse.Namespace(
            model_id=model_id,
            execute_after_transpile=False,
            allow_unconverted_weights=False,
            extra_args=extra_args,
        )
        rc = cmd_transpile(transpile_args)
        if rc != 0:
            return rc

        if assistant_model:
            assistant_model_id = resolve_model_id_alias(str(assistant_model))
            assistant_bits = getattr(args, "assistant_bits", None) or bits
            manifest_path = package_assistant_for_convert(
                cq_main=run_cq_convert,
                cmd_transpile=cmd_transpile,
                main_output_dir=output_dir,
                assistant_model_id=assistant_model_id,
                assistant_bits=int(assistant_bits),
                device=str(device),
                prompt=spec_prompt,
                max_new_tokens=int(getattr(args, "max_new_tokens", 32) or 32),
                torch_dtype=str(torch_dtype),
                token=token,
                cache_dir=cache_dir,
                trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
                local_files_only=bool(getattr(args, "local_files_only", False)),
            )
            print_color(GREEN, f"Assistant model integrated into {manifest_path}")

        print_color(GREEN, f"Model converted and transpiled to {output_dir}")
        return 0
    except SystemExit as e:
        return e.code if e.code else 0
    except Exception as e:
        print_color(RED, f"Conversion error: {e}")
        return 1
