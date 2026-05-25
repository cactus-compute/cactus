from .common import RED, print_color
from .download import get_weights_dir


def cmd_convert(args):
    """Convert a HuggingFace model to CQ format and transpile it in place."""
    from .model import resolve_model_id, ensure_bundle, TranspileOptions

    model_id = resolve_model_id(args.model_id)
    output_dir = args.output_dir or str(get_weights_dir(model_id))

    try:
        ensure_bundle(
            model_id,
            bits=args.bits or 4,
            token=args.token,
            cache_dir=args.cache_dir,
            reconvert=args.reconvert,
            output_dir=output_dir,
            transpile=TranspileOptions(
                task=args.task or "auto",
                prompt=args.prompt,
                image_files=[p for p in map(str, args.image_file or []) if p.strip()],
                audio_file=args.audio_file,
                max_new_tokens=args.max_new_tokens,
                component_pipeline=args.component_pipeline or "auto",
                components=args.components,
                system_prompt=args.system_prompt,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            ),
        )
        return 0
    except SystemExit as e:
        return e.code if e.code else 0
    except RuntimeError as e:
        print_color(RED, f"Conversion error: {e}")
        return 1
