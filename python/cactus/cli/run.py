from pathlib import Path

from .common import GREEN, RED, apply_runtime_env, launch_binary, print_color


def cmd_run(args) -> int:
    apply_runtime_env(args)

    if args.image:
        args.image = str(Path(args.image).expanduser())
        if not Path(args.image).is_file():
            print_color(RED, f"Image not found: {args.image}")
            return 1
    if args.audio:
        args.audio = str(Path(args.audio).expanduser())
        if not Path(args.audio).is_file():
            print_color(RED, f"Audio not found: {args.audio}")
            return 1
    if args.result_json:
        args.result_json = str(Path(args.result_json).expanduser())

    from .model import TranspileOptions, prepare_bundle

    bundle_dir = prepare_bundle(args, transpile=TranspileOptions(
        image_files=[args.image] if args.image else None,
        audio_file=args.audio,
    ))
    if bundle_dir is None:
        return 1

    cmd = [str(bundle_dir)]
    for flag, value in (
        ("--system", args.system),
        ("--prompt", args.prompt),
        ("--image", args.image),
        ("--audio", args.audio),
        ("--input-ids", args.input_ids),
        ("--input-ids-file", args.input_ids_file),
        ("--max-new-tokens", args.max_new_tokens),
        ("--result-json", args.result_json),
        ("--confidence-threshold", args.confidence_threshold),
        ("--cloud-timeout-ms", args.cloud_timeout_ms),
        ("--backend", args.backend),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.thinking:
        cmd.append("--thinking")
    if args.no_cloud_handoff:
        cmd.append("--no-cloud-handoff")

    print_color(GREEN, f"Running: {bundle_dir}")
    print()
    return launch_binary("run", *cmd)
