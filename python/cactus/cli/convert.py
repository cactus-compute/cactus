from .common import RED, print_color
from .download import get_weights_dir


def cmd_convert(args):
    """Convert a HuggingFace model to CQ format and transpile it in place."""
    from .model import ensure_bundle, TranspileOptions

    output_dir = args.output_dir or str(get_weights_dir(args.model_id))

    try:
        ensure_bundle(
            args.model_id,
            bits=args.bits or 4,
            token=args.token,
            reconvert=args.reconvert,
            output_dir=output_dir,
            transpile=TranspileOptions(
                task=args.task or "auto",
                prompt=args.prompt,
                image_files=args.image_file or None,
                audio_file=args.audio_file,
                max_new_tokens=args.max_new_tokens,
                component_pipeline=args.component_pipeline or "auto",
                components=args.components,
                system_prompt=args.system_prompt,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                npu=getattr(args, "npu", False),
                npu_quantize=getattr(args, "npu_quantize", None),
                npu_audio_quantize=getattr(args, "npu_audio_quantize", None),
                npu_vision_quantize=getattr(args, "npu_vision_quantize", None),
            ),
        )
        return 0
    except RuntimeError as e:
        print_color(RED, f"Conversion error: {e}")
        return 1
