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
    """Convert a HuggingFace model into Cactus CQ weights.

    Runtime graph/bundle generation used to happen after this step, but that graph
    builder has been removed for a rewrite.
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
