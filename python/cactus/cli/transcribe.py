import os
import subprocess
from pathlib import Path

from .common import apply_cloud_api_key_env, print_color, resolve_binary, RED, GREEN


def cmd_transcribe(args):
    from .model import ensure_runnable_bundle, TranspileOptions
    from .download import resolve_platform

    if args.audio_file:
        audio_path = Path(args.audio_file).expanduser()
        if not audio_path.is_file():
            print_color(RED, f"Audio file not found: {audio_path}")
            return 1
        args.audio_file = str(audio_path)

    if args.no_cloud_tele:
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    if args.force_handoff:
        os.environ["CACTUS_FORCE_HANDOFF"] = "1"
    else:
        os.environ.pop("CACTUS_FORCE_HANDOFF", None)

    apply_cloud_api_key_env()

    try:
        bundle_dir = ensure_runnable_bundle(
            args.model_id,
            bits=args.bits,
            platform=resolve_platform(args.platform),
            token=args.token,
            reconvert=args.reconvert,
            transpile=TranspileOptions(audio_file=args.audio_file),
        )
    except RuntimeError as e:
        print_color(RED, f"Model setup failed: {e}")
        return 1

    binary = resolve_binary("transcribe")
    if binary is None:
        return 1

    cmd = [str(binary), str(bundle_dir)]
    if args.audio_file:
        cmd.append(args.audio_file)
    if args.language:
        cmd.extend(["--language", args.language])

    print_color(GREEN, f"Starting Cactus transcription with model: {args.model_id}")
    print()

    return subprocess.run(cmd).returncode
