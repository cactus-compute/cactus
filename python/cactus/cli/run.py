import os
import subprocess
from pathlib import Path

from .common import print_color, is_repo_checkout, RED, GREEN


def cmd_run(args):
    from .model import ensure_bundle, resolve_bundle_dir, TranspileOptions
    from .config_utils import CactusConfig

    if args.no_cloud_tele:
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    api_key = CactusConfig().get_api_key()
    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key

    bundle_dir = resolve_bundle_dir(args.model_id)
    if bundle_dir is None:
        try:
            bundle_dir = ensure_bundle(
                args.model_id,
                token=args.token,
                reconvert=args.reconvert,
                transpile=TranspileOptions(
                    image_files=[args.image] if args.image else None,
                    audio_file=args.audio,
                ),
            )
        except RuntimeError as e:
            print_color(RED, f"Model setup failed: {e}")
            return 1

    chat = Path(__file__).resolve().parent.parent / "bin" / "chat"
    if is_repo_checkout() and not chat.exists():
        print_color(RED, "Chat binary not found. Run `cactus build` first.")
        return 1

    cmd = [str(chat), str(bundle_dir)]
    for flag, value in (
        ("--system", args.system),
        ("--prompt", args.prompt),
        ("--image", args.image),
        ("--audio", args.audio),
        ("--input-ids", args.input_ids),
        ("--max-new-tokens", args.max_new_tokens),
        ("--result-json", args.result_json),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.thinking:
        cmd.append("--thinking")

    print_color(GREEN, f"Starting Cactus Chat with model: {bundle_dir}")
    print()

    return subprocess.run(cmd).returncode
