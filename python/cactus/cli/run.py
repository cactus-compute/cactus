import os
import platform
import subprocess
import sys
from pathlib import Path

from .common import PROJECT_ROOT, print_color, RED, GREEN


def _resolve_bundle_dir(model_id):
    """Return bundle root if model_id points to a local transpiled bundle."""
    path = Path(model_id).expanduser()
    if not path.exists() or not path.is_dir():
        return None
    if (path / "components" / "manifest.json").exists():
        return path
    if path.name == "components" and (path / "manifest.json").exists():
        return path.parent
    return None


def cmd_run(args):
    """Run a model — auto-converts if needed."""
    from .model import resolve_model_id, ensure_bundle

    if args.no_cloud_tele:
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    bundle_dir = _resolve_bundle_dir(args.model_id)
    if bundle_dir is None:
        model_id = resolve_model_id(args.model_id)
        try:
            bundle_dir = ensure_bundle(
                model_id,
                token=args.token,
                cache_dir=args.cache_dir,
                reconvert=args.reconvert,
            )
        except RuntimeError as e:
            print_color(RED, str(e))
            return 1

    chat = PROJECT_ROOT / "cactus-engine" / "tests" / "build" / "chat"
    if not chat.exists():
        print_color(RED, "Chat binary not found. Run `cactus build` first.")
        return 1

    cmd = [str(chat), str(bundle_dir)]
    for flag, value in (
        ("--system", args.system),
        ("--prompt", args.prompt),
        ("--image", args.image),
        ("--audio", args.audio or args.audio_file),
    ):
        if value:
            cmd.extend([
                flag,
                str(Path(value).expanduser().resolve())
                if flag in ("--image", "--audio")
                else str(value),
            ])
    if args.thinking:
        cmd.append("--thinking")

    if sys.stdout.isatty():
        subprocess.run(["clear" if platform.system() != "Windows" else "cls"],
                       shell=False, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print_color(GREEN, f"Starting Cactus Chat with model: {bundle_dir}")
    print()

    return subprocess.run(cmd).returncode
