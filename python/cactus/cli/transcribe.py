import os
import subprocess
from pathlib import Path

from .common import is_repo_checkout, print_color, RED, GREEN


def cmd_transcribe(args):
    from .model import ensure_bundle, resolve_bundle_dir, TranspileOptions
    from .config_utils import CactusConfig

    if args.no_cloud_tele:
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    if args.force_handoff:
        os.environ["CACTUS_FORCE_HANDOFF"] = "1"
    else:
        os.environ.pop("CACTUS_FORCE_HANDOFF", None)

    api_key = CactusConfig().get_api_key()
    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key

    bundle_dir = resolve_bundle_dir(args.model_id)
    if bundle_dir is not None:
        print_color(GREEN, f"Using local model: {bundle_dir}")
    else:
        try:
            bundle_dir = ensure_bundle(
                args.model_id,
                token=args.token,
                reconvert=args.reconvert,
                transpile=TranspileOptions(audio_file=args.audio_file),
            )
        except RuntimeError as e:
            print_color(RED, f"Model setup failed: {e}")
            return 1

    asr = Path(__file__).resolve().parent.parent / "bin" / "asr"
    if is_repo_checkout() and not asr.exists():
        print_color(RED, "ASR binary not found. Run `cactus build` first.")
        return 1

    cmd = [str(asr), str(bundle_dir), args.audio_file]
    if args.language:
        cmd.extend(["--language", args.language])

    print_color(GREEN, f"Starting Cactus ASR with model: {args.model_id}")
    print()

    return subprocess.run(cmd).returncode
