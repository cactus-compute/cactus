import os
import platform
import subprocess
import sys
from pathlib import Path

from .common import PROJECT_ROOT, print_color, RED, GREEN


def _resolve_bundle_dir(model_id: str) -> Path | None:
    """Treat model_id as a local bundle dir; return its root if it is a transpiled bundle."""
    path = Path(model_id).expanduser()
    if not path.exists() or not path.is_dir():
        return None
    if (path / "components" / "manifest.json").exists():
        return path
    if path.name == "components" and (path / "manifest.json").exists():
        return path.parent
    return None


def _ensure_chat_binary() -> Path | None:
    chat = PROJECT_ROOT / "cactus-engine" / "tests" / "build" / "chat"
    if chat.exists():
        return chat
    print_color(RED, "Error: chat binary not found. Run `cactus build` first.")
    return None


def cmd_run(args):
    """Run a transpiled Cactus bundle through the libcactus-backed chat binary."""
    if getattr(args, 'no_cloud_tele', False):
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    bundle_dir = _resolve_bundle_dir(args.model_id)
    if bundle_dir is None:
        print_color(RED,
            f"Error: {args.model_id} is not a transpiled bundle. "
            "Run `cactus convert <hf_model>` to produce one.")
        return 1

    chat = _ensure_chat_binary()
    if chat is None:
        return 1

    cmd = [str(chat), str(bundle_dir)]
    for flag, value in (("--system", getattr(args, 'system', None)),
                         ("--prompt", getattr(args, 'prompt', None)),
                         ("--image", getattr(args, 'image', None)),
                         ("--audio", getattr(args, 'audio', None) or getattr(args, 'audio_file', None))):
        if value:
            cmd.extend([flag, str(Path(value).expanduser().resolve()) if flag in ("--image", "--audio") else str(value)])
    if getattr(args, 'thinking', False):
        cmd.append("--thinking")

    if sys.stdout.isatty():
        os.system('clear' if platform.system() != 'Windows' else 'cls')
    print_color(GREEN, f"Starting Cactus Chat with model: {bundle_dir}")
    print()

    return subprocess.run(cmd).returncode
