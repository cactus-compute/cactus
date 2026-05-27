"""Model resolution, weight management, and bundle preparation."""
from __future__ import annotations

import shutil
import json
from dataclasses import dataclass
from importlib.resources import as_file, files
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


def _is_gemma4_bundle(model_id, output_dir):
    config_path = Path(output_dir) / "config.txt"
    if config_path.exists():
        try:
            text = config_path.read_text(encoding="utf-8", errors="ignore")
            if "model_type=gemma4" in text or "Gemma4" in text:
                return True
        except OSError:
            pass
    return "gemma-4" in str(model_id).lower() or "gemma4" in str(model_id).lower()


def _install_gemma4_cloud_handoff_probe(model_id, output_dir):
    """Install the v10p6 probe artifact into Gemma4 bundles for native runtime use."""
    if not _is_gemma4_bundle(model_id, output_dir):
        return
    probe_dir = Path(output_dir) / "cloud_handoff"
    probe_dir.mkdir(parents=True, exist_ok=True)
    destination = probe_dir / "global_attn_probe_v10p6.bin"
    if destination.exists():
        return

    try:
        resource = files("cactus.cloud_handoff").joinpath(
            "models", "v10p6_probe_release", "global_attn_probe_v10p6.bin"
        )
        with as_file(resource) as source:
            if Path(source).exists():
                shutil.copy2(source, destination)
                return
    except Exception:
        pass

    from cactus.cloud_handoff import export_probe_binary

    export_probe_binary(destination)


def _gemma4_bundle_needs_probe_retranspile(model_id, output_dir):
    if not _is_gemma4_bundle(model_id, output_dir):
        return False
    manifest_path = Path(output_dir) / "components" / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for component in manifest.get("components", []):
        if component.get("component") != "decoder_step":
            continue
        outputs = set(component.get("logical_outputs") or [])
        return "cloud_handoff_hidden" not in outputs
    return False


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
        _install_gemma4_cloud_handoff_probe(model_id, output_dir)
        if not _gemma4_bundle_needs_probe_retranspile(model_id, output_dir):
            return output_dir
        print_color(
            YELLOW,
            "Existing Gemma4 bundle is missing cloud handoff probe outputs; "
            "refreshing transpiled graphs.",
        )

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

    rc = run_transpile(model_id, extra_args=extra_args)
    if rc != 0:
        raise RuntimeError(f"Transpilation failed for {model_id}")

    _install_gemma4_cloud_handoff_probe(model_id, output_dir)

    print_color(GREEN, f"Model converted and transpiled to {output_dir}")
    return output_dir
