"""Model resolution, weight management, and bundle preparation."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from .common import GREEN, PROJECT_ROOT, RED, YELLOW, print_color




def _convert_from_source(model_id, *, bits, token, weights_dir, skip_model_load=False):
    """Download from HuggingFace and run CQ conversion."""
    print_color(YELLOW, f"Converting {model_id} from HuggingFace source...")
    from ..convert.cli import main as cq_main

    cq_args = [
        "convert", "--model", model_id,
        "--out", str(weights_dir),
        "--bits", str(bits),
    ]
    if skip_model_load:
        cq_args.append("--skip-model-load")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    cq_main(cq_args)

    print_color(GREEN, f"Model converted and ready at {weights_dir}")
    return weights_dir


def ensure_weights(model_id, *, bits=4, platform=None, token=None, reconvert=False, output_dir=None, skip_model_load=False):
    from .download import get_bundle_dir

    weights_dir = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits, platform=platform)

    if reconvert and weights_dir.exists():
        print_color(YELLOW, "Removing cached weights for reconversion...")
        shutil.rmtree(weights_dir)

    if weights_dir.exists() and (weights_dir / "config.txt").exists():
        print_color(GREEN, f"Model weights found at {weights_dir}")
        return weights_dir

    if weights_dir.exists():
        print_color(YELLOW, "Removing incomplete weights from a previous run...")
        shutil.rmtree(weights_dir)

    return _convert_from_source(
        model_id,
        bits=bits,
        token=token,
        weights_dir=weights_dir,
        skip_model_load=skip_model_load,
    )



_DEFAULT_MULTIMODAL_PROMPT = (
    "Respond with 2 lines. The first should be a description of the image, "
    "and the second should be a transcription of the audio"
)
_DEFAULT_TEXT_PROMPT = "Hello"


@dataclass(frozen=True)
class _TranspileSpec:
    task: str
    components: tuple[str, ...] = ()
    default_max_new_tokens: int | None = None
    needs_image: bool = False
    needs_audio: bool = False
    force_component_pipeline: bool = False


def _spec_from_plan(plan):
    return _TranspileSpec(
        task=plan.task,
        components=tuple(plan.components or ()),
        default_max_new_tokens=plan.default_max_new_tokens,
        needs_image=bool(plan.needs_image),
        needs_audio=bool(plan.needs_audio),
        force_component_pipeline=bool(plan.force_component_pipeline),
    )


def _infer_transpile_spec(*, task, plan):
    if task != "auto":
        if plan is not None and task == plan.task:
            return _spec_from_plan(plan)
        return _TranspileSpec(
            task=task,
            needs_image=task == "multimodal_causal_lm_logits",
            needs_audio=task in {
                "tdt_transcription", "seq2seq_transcription",
                "ctc_logits", "encoder_hidden_states",
                "multimodal_causal_lm_logits",
            },
            force_component_pipeline=task in {
                "tdt_transcription", "seq2seq_transcription",
                "multimodal_causal_lm_logits",
            },
        )

    if plan is None:
        return _TranspileSpec(task="causal_lm_logits")

    return _spec_from_plan(plan)


def _default_max_new_tokens(spec):
    if spec.default_max_new_tokens is not None:
        return int(spec.default_max_new_tokens)
    return {
        "seq2seq_transcription": 128,
        "multimodal_causal_lm_logits": 512,
        "causal_lm_logits": 128,
    }.get(spec.task, 32)


def _default_multimodal_assets():
    """Return bundled test image/audio paths for multimodal shape capture."""
    candidates = (
        Path(__file__).resolve().parent.parent / "assets",
        PROJECT_ROOT / "cactus-engine" / "tests" / "assets",
    )
    def _find(name):
        return next((d / name for d in candidates if (d / name).exists()), None)
    image = _find("test_monkey.png")
    audio = _find("test.wav")
    return ([str(image)] if image else []), (str(audio) if audio else None)


def _default_audio_asset():
    _, audio = _default_multimodal_assets()
    return audio


def _remove_stale_transpile_artifacts(output_dir):
    for relative in (
        "components",
        "transpile_entrypoints.json",
        "raw_ir.json",
        "optimized_ir.json",
        "graph.cactus",
        "graph_bindings.json",
        "result.json",
    ):
        path = output_dir / relative
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for pattern in ("raw_ir_*.json", "optimized_ir_*.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _has_transpiled_bundle(path):
    return (path / "components" / "manifest.json").exists()


_AUDIO_TASKS = frozenset({
    "tdt_transcription", "seq2seq_transcription",
    "ctc_logits", "encoder_hidden_states",
})




def resolve_bundle_dir(model_id):
    path = Path(model_id).expanduser()
    if path.is_dir() and _has_transpiled_bundle(path):
        return path
    return None


@dataclass(frozen=True)
class TranspileOptions:
    task: str = "auto"
    prompt: str | None = None
    image_files: list[str] | None = None
    audio_file: str | None = None
    max_new_tokens: int | None = None
    component_pipeline: str = "auto"
    components: str | None = None
    system_prompt: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    npu: bool = False
    npu_quantize: int | None = None
    npu_audio_quantize: int | None = None
    npu_vision_quantize: int | None = None
    cache_context_length: str | int | None = None


def ensure_runnable_bundle(model_id, *, bits=4, platform=None, token=None,
                           reconvert=False, prebuilt=True, output_dir=None,
                           transpile=None):
    """Resolve a runnable bundle via the full fallback ladder, building if needed.

    Rungs, in order: (1) a local bundle path, (2) a cached prior build,
    (3) a prebuilt bundle on HuggingFace, (4) local convert + transpile.
    With prebuilt=False, rung (3) is skipped and the bundle is always built
    locally when not already present.
    Raises RuntimeError if every rung fails.
    """
    from .download import download_bundle, get_bundle_dir

    local = resolve_bundle_dir(model_id)
    if local is not None:
        return local

    if str(model_id).startswith(("/", "./", "../", "~")) and not Path(model_id).expanduser().exists():
        raise RuntimeError(f"path not found: {model_id}")

    cached = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits, platform=platform)
    if _has_transpiled_bundle(cached) and not reconvert:
        return cached

    if prebuilt and not reconvert:
        try:
            return download_bundle(model_id, bits=bits, platform=platform,
                                   token=token, output_dir=cached)
        except (RuntimeError, OSError) as exc:
            print_color(YELLOW, f"No prebuilt bundle ({exc}); building locally")

    opts = transpile or TranspileOptions()
    if platform == "apple" and not opts.npu:
        opts = replace(opts, npu=True)
    return ensure_bundle(model_id, bits=bits, platform=platform, token=token,
                         reconvert=reconvert, output_dir=cached, transpile=opts)


def prepare_bundle(args, *, model_id=None, transpile=None, prebuilt=True,
                   output_dir=None, fail_prefix="Model setup failed"):
    """Resolve the weights variant from args and return a runnable bundle, with uniform
    error handling shared by every model command. Returns the bundle Path, or
    None (after printing the error) on failure."""
    from .download import resolve_weights_variant
    try:
        return ensure_runnable_bundle(
            args.model_id if model_id is None else model_id,
            bits=getattr(args, "bits", 4),
            platform=resolve_weights_variant(getattr(args, "weights", "general")),
            token=getattr(args, "token", None),
            reconvert=getattr(args, "reconvert", False),
            prebuilt=prebuilt,
            output_dir=output_dir,
            transpile=transpile,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print_color(RED, f"{fail_prefix}: {exc}")
        return None


def ensure_bundle(model_id, *, bits=4, platform=None, token=None,
                  reconvert=False, output_dir=None, transpile=None,
                  skip_model_load=False):
    from .download import get_bundle_dir
    from .transpile import run_transpile
    from cactus.transpile.component_plan import infer_component_plan_from_output

    opts = transpile or TranspileOptions()

    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
    else:
        output_dir = get_bundle_dir(model_id, bits=bits, platform=platform)

    ensure_weights(
        model_id, bits=bits, platform=platform, token=token,
        reconvert=reconvert, output_dir=output_dir,
        skip_model_load=skip_model_load,
    )

    if _has_transpiled_bundle(output_dir):
        return output_dir

    plan = infer_component_plan_from_output(str(output_dir), model_id=model_id)
    spec = _infer_transpile_spec(task=opts.task, plan=plan)
    _remove_stale_transpile_artifacts(output_dir)

    spec_prompt = opts.prompt
    spec_image_files = list(opts.image_files or [])
    spec_audio_file = opts.audio_file

    if spec_prompt is None and spec.task == "multimodal_causal_lm_logits":
        spec_prompt = _DEFAULT_MULTIMODAL_PROMPT
    elif spec_prompt is None and spec.task == "causal_lm_logits":
        spec_prompt = _DEFAULT_TEXT_PROMPT

    effective_component_pipeline = opts.component_pipeline
    effective_components = opts.components

    if spec.task == "multimodal_causal_lm_logits":
        needs_image = spec.needs_image
        needs_audio = spec.needs_audio
        if not needs_image and not needs_audio:
            needs_image = bool(spec_image_files)
            needs_audio = bool(spec_audio_file)
        if (needs_image and not spec_image_files) or (needs_audio and not spec_audio_file):
            default_images, default_audio = _default_multimodal_assets()
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
            raise RuntimeError("Building this multimodal model requires --image-file.")
        if needs_audio and not spec_audio_file:
            raise RuntimeError("Building this multimodal model requires --audio-file.")

    if effective_component_pipeline == "auto" and spec.force_component_pipeline:
        effective_component_pipeline = "on"
    if effective_components is None and spec.components:
        effective_components = ",".join(spec.components)

    used_default_audio = False
    if spec.task in _AUDIO_TASKS and not spec_audio_file:
        spec_audio_file = _default_audio_asset()
        used_default_audio = spec_audio_file is not None
    if spec.task in _AUDIO_TASKS and used_default_audio:
        print_color(
            YELLOW,
            f"{spec.task} transpile needs a representative audio shape; "
            "using bundled tiny test audio asset.",
        )
    elif spec.task in _AUDIO_TASKS and not spec_audio_file:
        raise RuntimeError(f"Building a {spec.task} model requires --audio-file.")

    effective_max_new_tokens = opts.max_new_tokens or _default_max_new_tokens(spec)

    extra_args = [
        "--weights-dir", str(output_dir),
        "--artifact-dir", str(output_dir),
        "--task", spec.task,
        "--max-new-tokens", str(effective_max_new_tokens),
        "--component-pipeline", effective_component_pipeline,
    ]
    if spec_prompt is not None:
        extra_args.extend(["--prompt", spec_prompt])
    if effective_components:
        extra_args.extend(["--components", str(effective_components)])
    for img in spec_image_files:
        extra_args.extend(["--image-file", img])
    if spec_audio_file:
        extra_args.extend(["--audio-file", str(spec_audio_file)])
    if opts.system_prompt:
        extra_args.extend(["--system-prompt", str(opts.system_prompt)])
    if token:
        extra_args.extend(["--token", token])
    if opts.trust_remote_code or spec.task == "multimodal_causal_lm_logits":
        extra_args.append("--trust-remote-code")
    if opts.local_files_only:
        extra_args.append("--local-files-only")
    if opts.npu:
        extra_args.append("--npu")
        if opts.npu_quantize is not None:
            extra_args.extend(["--npu-quantize", str(int(opts.npu_quantize))])
        if opts.npu_audio_quantize is not None:
            extra_args.extend(["--npu-audio-quantize", str(int(opts.npu_audio_quantize))])
        if opts.npu_vision_quantize is not None:
            extra_args.extend(["--npu-vision-quantize", str(int(opts.npu_vision_quantize))])
    if opts.cache_context_length is not None:
        extra_args.extend(["--cache-context-length", str(opts.cache_context_length)])

    rc = run_transpile(model_id, extra_args=extra_args)
    if rc != 0:
        raise RuntimeError(f"Build failed for {model_id}")

    try:
        from cactus.convert.handoff_probe import (
            export_gemma4_handoff_probe,
            export_parakeet_handoff_probe,
        )

        if export_gemma4_handoff_probe(output_dir, model_id=model_id):
            print_color(GREEN, f"Gemma4 cloud handoff probe packaged into {output_dir}")
        if export_parakeet_handoff_probe(output_dir, model_id=model_id):
            print_color(GREEN, f"Parakeet cloud handoff probe packaged into {output_dir}")
    except Exception as e:
        print_color(YELLOW, f"Warning: failed to package cloud handoff probe: {e}")

    print_color(GREEN, f"Model built at {output_dir}")
    return output_dir
