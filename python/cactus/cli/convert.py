import shutil
import tempfile
from pathlib import Path

from .common import GREEN, RED, YELLOW, print_color
from .download import get_weights_dir


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
    """Convert a HuggingFace model to CQ format. Does not transpile —
    use `cactus transpile` afterwards to build the runtime graph bundle.
    """
    from .model import ensure_weights

    source_model_id = args.model_id
    merged_dir = None

    if args.lora:
        merged_dir = _merge_lora_adapter(args.model_id, args.lora, token=args.token)
        if merged_dir is None:
            return 1
        source_model_id = merged_dir

    output_dir = args.output_dir or str(get_weights_dir(args.model_id))

    try:
        ensure_weights(
            source_model_id,
            bits=args.bits,
            token=args.token,
            reconvert=args.reconvert,
            output_dir=output_dir,
        )
        return 0
    except RuntimeError as e:
        print_color(RED, f"Conversion error: {e}")
        return 1
    finally:
        if merged_dir:
            shutil.rmtree(merged_dir, ignore_errors=True)


def cmd_transpile(args):
    """Build the runtime graph bundle from already-converted CQ weights."""
    from .transpile import run_transpile
    from .model import _default_multimodal_assets

    extra_args = []
    if args.weights_dir:
        extra_args.extend(["--weights-dir", args.weights_dir])
    if args.task and args.task != "auto":
        extra_args.extend(["--task", args.task])
    if args.prompt is not None:
        extra_args.extend(["--prompt", args.prompt])

    image_files = list(args.image_file or [])
    audio_file = args.audio_file
    if not image_files or not audio_file:
        default_images, default_audio = _default_multimodal_assets()
        if not image_files:
            image_files = default_images
        if not audio_file and default_audio:
            audio_file = default_audio

    for img in image_files:
        extra_args.extend(["--image-file", img])
    if audio_file:
        extra_args.extend(["--audio-file", audio_file])
    if args.max_new_tokens is not None:
        extra_args.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.component_pipeline and args.component_pipeline != "auto":
        extra_args.extend(["--component-pipeline", args.component_pipeline])
    if args.components:
        extra_args.extend(["--components", args.components])
    if args.trust_remote_code:
        extra_args.append("--trust-remote-code")
    if args.local_files_only:
        extra_args.append("--local-files-only")
    if args.artifact_dir:
        extra_args.extend(["--artifact-dir", args.artifact_dir])
    if args.skip_reference_compare:
        extra_args.append("--skip-reference-compare")
    if args.no_fuse_rms_norm:
        extra_args.append("--no-fuse-rms-norm")
    if args.no_fuse_rope:
        extra_args.append("--no-fuse-rope")
    if args.no_fuse_attention:
        extra_args.append("--no-fuse-attention")

    return run_transpile(
        args.model_id,
        extra_args=extra_args,
        execute_after_transpile=args.execute_after_transpile,
        allow_unconverted_weights=args.allow_unconverted_weights,
    )
