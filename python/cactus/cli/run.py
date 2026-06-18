import os
import subprocess
from pathlib import Path

from .common import GREEN, apply_cloud_api_key_env, print_color, resolve_binary


def cmd_run(args) -> int:
    if args.no_cloud_tele:
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
    apply_cloud_api_key_env()

    if args.image:
        args.image = str(Path(args.image).expanduser())
    if args.audio:
        args.audio = str(Path(args.audio).expanduser())
    if args.result_json:
        args.result_json = str(Path(args.result_json).expanduser())

    from .model import TranspileOptions, prepare_bundle

    bundle_dir = prepare_bundle(args, transpile=TranspileOptions(
        image_files=[args.image] if args.image else None,
        audio_file=args.audio,
    ))
    if bundle_dir is None:
        return 1

    binary = resolve_binary("run")
    if binary is None:
        return 1

    cmd = [str(binary), str(bundle_dir)]
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
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.thinking:
        cmd.append("--thinking")
    if args.no_cloud_handoff:
        cmd.append("--no-cloud-handoff")

    print_color(GREEN, f"Running: {bundle_dir}")
    print()
    return subprocess.run(cmd).returncode
