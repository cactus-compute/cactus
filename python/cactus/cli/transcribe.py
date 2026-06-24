from pathlib import Path

from .common import apply_runtime_env, launch_binary, print_color, RED, GREEN


def cmd_transcribe(args):
    from .model import prepare_bundle, TranspileOptions

    if args.audio_file:
        audio_path = Path(args.audio_file).expanduser()
        if not audio_path.is_file():
            print_color(RED, f"Audio file not found: {audio_path}")
            return 1
        args.audio_file = str(audio_path)

    apply_runtime_env(args)

    bundle_dir = prepare_bundle(args, transpile=TranspileOptions(audio_file=args.audio_file))
    if bundle_dir is None:
        return 1

    cmd = [str(bundle_dir)]
    if args.audio_file:
        cmd.append(args.audio_file)
    if args.language:
        cmd.extend(["--language", args.language])

    print_color(GREEN, f"Starting Cactus transcription with model: {args.model_id}")
    print()

    return launch_binary("transcribe", *cmd)
