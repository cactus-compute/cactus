import shutil
import tempfile
from pathlib import Path

from .common import GREEN, RED, YELLOW, print_color
from .download import get_bundle_dir, get_weights_dir


_GENERIC_TASKS = {"causal-lm", "speech-seq2seq"}


def _arg(args, name, default=None):
    return vars(args).get(name, default)


def _uses_legacy_graph_builder(args):
    task = _arg(args, "task", _arg(args, "generic_task"))
    if task not in (None, *_GENERIC_TASKS):
        return True
    fields = (
        "weights_dir", "artifact_dir", "prompt", "system_prompt", "input_ids", "audio_file",
        "max_new_tokens", "components", "torch_dtype", "graph_filename", "cache_context_length",
    )
    flags = (
        "enable_thinking", "trust_remote_code", "local_files_only", "allow_unconverted_weights",
        "execute_after_transpile", "skip_reference_compare", "low_memory_load", "no_fuse_rms_norm",
        "no_fuse_rope", "no_fuse_attention", "no_fuse_attention_block", "no_fuse_add_clipped",
        "no_fuse_gated_deltanet",
    )
    return (
        any(_arg(args, field) is not None for field in fields)
        or any(bool(_arg(args, flag, False)) for flag in flags)
        or bool(_arg(args, "image_file", ()))
        or _arg(args, "component_pipeline", "auto") != "auto"
    )


def _run_legacy_graph_builder(args):
    """Preserve main's explicit graph-capture path without changing the new default."""
    from .model import _default_multimodal_assets
    from .transpile import run_transpile
    from cactus.transpile.component_plan import infer_component_plan_from_output

    extra_args = ["--weights-dir", args.weights_dir]
    task = getattr(args, "task", None)
    if task and task != "auto":
        extra_args.extend(["--task", task])
    for name, flag in (("prompt", "--prompt"), ("system_prompt", "--system-prompt"),
                       ("input_ids", "--input-ids"), ("audio_file", "--audio-file"),
                       ("components", "--components"), ("torch_dtype", "--torch-dtype"),
                       ("graph_filename", "--graph-filename"),
                       ("cache_context_length", "--cache-context-length")):
        value = getattr(args, name, None)
        if value is not None:
            extra_args.extend([flag, str(value)])

    images = list(getattr(args, "image_file", ()) or ())
    plan = infer_component_plan_from_output(args.weights_dir, model_id=args.model_id)
    needs_image = bool(plan and plan.needs_image) or task == "multimodal_causal_lm_logits"
    needs_audio = bool(plan and plan.needs_audio) or task in {
        "ctc_logits", "encoder_hidden_states", "seq2seq_transcription", "tdt_transcription",
        "multimodal_causal_lm_logits",
    }
    audio = getattr(args, "audio_file", None)
    if (needs_image and not images) or (needs_audio and not audio):
        default_images, default_audio = _default_multimodal_assets()
        images = default_images if needs_image and not images else images
        audio = default_audio if needs_audio and not audio else audio
    for image in images:
        extra_args.extend(["--image-file", image])
    if audio and getattr(args, "audio_file", None) is None:
        extra_args.extend(["--audio-file", audio])

    if getattr(args, "max_new_tokens", None) is not None:
        extra_args.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if getattr(args, "component_pipeline", "auto") != "auto":
        extra_args.extend(["--component-pipeline", args.component_pipeline])
    if getattr(args, "artifact_dir", None):
        extra_args.extend(["--artifact-dir", args.artifact_dir])
    if args.token:
        extra_args.extend(["--token", args.token])
    for name, flag in (
        ("enable_thinking", "--enable-thinking"), ("trust_remote_code", "--trust-remote-code"),
        ("local_files_only", "--local-files-only"), ("low_memory_load", "--low-memory-load"),
        ("skip_reference_compare", "--skip-reference-compare"), ("no_fuse_rms_norm", "--no-fuse-rms-norm"),
        ("no_fuse_rope", "--no-fuse-rope"), ("no_fuse_attention", "--no-fuse-attention"),
        ("no_fuse_attention_block", "--no-fuse-attention-block"),
        ("no_fuse_add_clipped", "--no-fuse-add-clipped"),
        ("no_fuse_gated_deltanet", "--no-fuse-gated-deltanet"),
    ):
        if getattr(args, name, False):
            extra_args.append(flag)
    return run_transpile(
        args.model_id,
        extra_args=extra_args,
        execute_after_transpile=bool(getattr(args, "execute_after_transpile", False)),
        allow_unconverted_weights=bool(getattr(args, "allow_unconverted_weights", False)),
    )


def _merge_lora_adapter(base_model_id, lora_path, token=None):
    """Merge a LoRA/PEFT adapter into the base model, save to a temp dir, return path."""
    try:
        from peft import PeftModel
    except ImportError:
        print_color(RED, "Error: `peft` is required for LoRA merging.")
        print("Install with: pip install peft")
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print_color(YELLOW, f"Loading base model: {base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, token=token, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, token=token, trust_remote_code=True,
    )

    print_color(YELLOW, f"Loading LoRA adapter: {lora_path}")
    merged = PeftModel.from_pretrained(base, lora_path, token=token).merge_and_unload()

    out_dir = Path(tempfile.mkdtemp(prefix="cactus_lora_merged_"))
    print_color(YELLOW, f"Saving merged model to: {out_dir}")
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    lora_tok = Path(lora_path) / "tokenizer_config.json"
    if lora_tok.is_file():
        shutil.copy2(lora_tok, out_dir / "tokenizer_config.json")

    print_color(GREEN, "LoRA merge complete")
    return str(out_dir)


def cmd_convert(args):
    """Convert a HuggingFace model into a runnable Cactus bundle."""
    from .model import ensure_weights, package_handoff_probe
    from .transpiler import build_transpiled_bundle, parse_modalities, resolve_transpile_config

    source_model_id = args.model_id
    merged_dir = None

    if args.lora:
        merged_dir = _merge_lora_adapter(args.model_id, args.lora, token=args.token)
        if merged_dir is None:
            return 1
        source_model_id = merged_dir

    output_dir = args.output_dir or str(
        get_bundle_dir(args.model_id, bits=args.bits)
    )

    try:
        legacy_builder = _uses_legacy_graph_builder(args)
        generic_task = _arg(args, "task", _arg(args, "generic_task"))
        generic_task = generic_task if generic_task in _GENERIC_TASKS else None
        if not getattr(args, "weights_only", False) and not legacy_builder:
            resolve_transpile_config(
                args.model_id,
                input_modalities=getattr(args, "input_modalities", None),
                generic_task=generic_task,
                cache_style=getattr(args, "cache_style", None),
                fusion_groups=getattr(args, "fusion_groups", None),
            )
        weights_dir = ensure_weights(
            source_model_id,
            bits=args.bits,
            token=args.token,
            reconvert=args.reconvert,
            output_dir=output_dir,
            skip_model_load=bool(getattr(args, "skip_model_load", False)),
            calibration_manifest=getattr(args, "calibration_manifest", None),
        )
        if getattr(args, "weights_only", False):
            return 0

        if legacy_builder:
            args.weights_dir = args.weights_dir or str(weights_dir)
            args.artifact_dir = args.artifact_dir or str(weights_dir)
            result = _run_legacy_graph_builder(args)
            if result == 0:
                package_handoff_probe(args.artifact_dir, args.model_id)
            return result

        build_transpiled_bundle(
            source_model_id,
            weights_dir=weights_dir,
            output_dir=weights_dir,
            profile_model_id=args.model_id,
            input_modalities=parse_modalities(getattr(args, "input_modalities", None)),
            generic_task=generic_task,
            cache_style=getattr(args, "cache_style", None),
            fusion_groups=getattr(args, "fusion_groups", None),
            token=args.token,
        )
        package_handoff_probe(weights_dir, args.model_id)
        return 0
    except (RuntimeError, ValueError) as e:
        print_color(RED, f"Conversion error: {e}")
        return 1
    finally:
        if merged_dir:
            shutil.rmtree(merged_dir, ignore_errors=True)
