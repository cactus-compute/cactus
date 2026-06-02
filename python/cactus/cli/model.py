"""Model resolution, weight management, and bundle preparation."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .common import GREEN, PROJECT_ROOT, YELLOW, print_color


# ── Weight download / conversion ──────────────────────────────────────


def _convert_from_source(model_id, *, bits, token, weights_dir):
    """Download from HuggingFace and run CQ conversion."""
    print_color(YELLOW, f"Converting {model_id} from HuggingFace source...")
    from ..convert.cli import main as cq_main

    cq_args = [
        "convert", "--model", model_id,
        "--out", str(weights_dir),
        "--bits", str(bits),
        "--force",
    ]
    if token:
        cq_args.extend(["--token", token])
    cq_main(cq_args)

    print_color(GREEN, f"Model converted and ready at {weights_dir}")
    return weights_dir


def ensure_weights(model_id, *, bits=4, token=None, reconvert=False, output_dir=None):
    """Return path to CQ weights dir, downloading or converting as needed.

    Fast path: pull a pre-converted CQ archive from huggingface.co/Cactus-Compute.
    Fallback: ``--reconvert`` or no archive available → build from source.
    """
    from .download import get_weights_dir, download_cq_weights

    weights_dir = Path(output_dir) if output_dir else get_weights_dir(model_id)

    if reconvert and weights_dir.exists():
        print_color(YELLOW, "Removing cached weights for reconversion...")
        shutil.rmtree(weights_dir)

    if weights_dir.exists() and (weights_dir / "config.txt").exists():
        print_color(GREEN, f"Model weights found at {weights_dir}")
        return weights_dir

    if not reconvert:
        try:
            return download_cq_weights(
                model_id, bits=bits, token=token, output_dir=weights_dir,
            )
        except (RuntimeError, OSError) as exc:
            print(f"  Pre-converted CQ not available ({exc})")

    return _convert_from_source(model_id, bits=bits, token=token, weights_dir=weights_dir)


# ── Transpile spec helpers ────────────────────────────────────────────

_DEFAULT_MULTIMODAL_PROMPT = (
    "Respond with 2 lines. The first should be a description of the image, "
    "and the second should be a transcription of the audio"
)
_DEFAULT_TEXT_PROMPT = "Hello"


@dataclass(frozen=True)
class _TranspileSpec:
    task: str
    components: tuple[str, ...] = ()
    needs_image: bool = False
    needs_audio: bool = False
    force_component_pipeline: bool = False


def _spec_from_plan(plan):
    """Convert a ComponentPlan into a _TranspileSpec."""
    return _TranspileSpec(
        task=plan.task,
        components=tuple(plan.components or ()),
        needs_image=bool(plan.needs_image),
        needs_audio=bool(plan.needs_audio),
        force_component_pipeline=bool(plan.force_component_pipeline),
    )


def _infer_transpile_spec(*, task, plan):
    """Determine transpile parameters from task + component plan."""
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


def _default_max_new_tokens(task):
    """Sensible token budget per task type."""
    return {
        "seq2seq_transcription": 128,
        "multimodal_causal_lm_logits": 512,
        "causal_lm_logits": 128,
    }.get(task, 32)


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
    """Clean old transpile outputs before re-transpiling."""
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
    """Check if path contains a transpiled bundle."""
    return (path / "components" / "manifest.json").exists()


_AUDIO_TASKS = frozenset({
    "tdt_transcription", "seq2seq_transcription",
    "ctc_logits", "encoder_hidden_states",
})


# ── Bundle preparation (weights + transpile) ──────────────────────────


def resolve_bundle_dir(model_id):
    path = Path(model_id).expanduser()
    if not path.is_dir():
        return None
    if (path / "components" / "manifest.json").exists():
        return path
    if path.name == "components" and (path / "manifest.json").exists():
        return path.parent
    return None


@dataclass(frozen=True)
class TranspileOptions:
    """Transpile-phase parameters for ensure_bundle."""
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


def ensure_bundle(model_id, *, bits=4, token=None,
                  reconvert=False, output_dir=None, transpile=None):
    """Return path to transpiled bundle, creating it if needed.
    """
    from .download import get_weights_dir
    from .transpile import run_transpile
    from cactus.transpile.component_plan import infer_component_plan_from_output

    opts = transpile or TranspileOptions()

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = get_weights_dir(model_id)

    # Step 1: ensure CQ weights exist
    ensure_weights(
        model_id, bits=bits, token=token,
        reconvert=reconvert, output_dir=output_dir,
    )

    # Step 2: skip if already transpiled
    if _has_transpiled_bundle(output_dir):
        return output_dir

    # Step 3: infer transpile spec from converted output
    plan = infer_component_plan_from_output(str(output_dir), model_id=model_id)
    spec = _infer_transpile_spec(task=opts.task, plan=plan)
    _remove_stale_transpile_artifacts(output_dir)

    # Step 4: resolve defaults for prompt, images, audio
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
            raise RuntimeError("Multimodal transpile requires --image-file for this model.")
        if needs_audio and not spec_audio_file:
            raise RuntimeError("Multimodal transpile requires --audio-file for this model.")

    if effective_component_pipeline == "auto" and spec.force_component_pipeline:
        effective_component_pipeline = "on"
    if effective_components is None and spec.components:
        effective_components = ",".join(spec.components)

    # Handle audio-only tasks
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
        raise RuntimeError(f"{spec.task} transpile requires --audio-file.")

    # Step 5: build transpile args and call run_transpile
    effective_max_new_tokens = opts.max_new_tokens or _default_max_new_tokens(spec.task)

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

    rc = run_transpile(model_id, extra_args=extra_args)
    if rc != 0:
        raise RuntimeError(f"Transpilation failed for {model_id}")

    try:
        from cactus.convert.handoff_probe import export_gemma4_handoff_probe

        if export_gemma4_handoff_probe(output_dir, model_id=model_id):
            print_color(GREEN, f"Gemma4 cloud handoff probe packaged into {output_dir}")
    except Exception as e:
        print_color(YELLOW, f"Warning: failed to package Gemma4 cloud handoff probe: {e}")

    print_color(GREEN, f"Model converted and transpiled to {output_dir}")
    return output_dir
