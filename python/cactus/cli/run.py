import os
import subprocess
from pathlib import Path

from .common import print_color, RED, GREEN


def _resolve_bundle_dir(model_id):
    path = Path(model_id).expanduser()
    if not path.exists() or not path.is_dir():
        return None
    if (path / "components" / "manifest.json").exists():
        return path
    if path.name == "components" and (path / "manifest.json").exists():
        return path.parent
    return None


def cmd_run(args):
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
            print_color(RED, f"Model setup failed: {e}")
            return 1

    chat = Path(__file__).resolve().parent.parent / "bin" / "chat"
    if not chat.exists():
        print_color(RED, "Chat binary not found. Run `cactus build` first.")
        return 1

    image_file = next((image for image in getattr(args, 'image_file', []) if image), None)

    cmd = [str(chat), str(bundle_dir)]
    for flag, value in (
        ("--system", getattr(args, 'system', None)),
        ("--prompt", getattr(args, 'prompt', None)),
        ("--image", getattr(args, 'image', None) or image_file),
        ("--audio", getattr(args, 'audio', None) or getattr(args, 'audio_file', None)),
    ):
        if value:
            cmd.extend([
                flag,
                str(Path(value).expanduser().resolve())
                if flag in ("--image", "--audio")
                else str(value),
            ])
    for flag, value in (
        ("--input-ids", getattr(args, 'input_ids', None)),
        ("--max-new-tokens", getattr(args, 'max_new_tokens', None)),
        ("--result-json", getattr(args, 'result_json', None)),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if getattr(args, 'thinking', False):
        cmd.append("--thinking")

    print_color(GREEN, f"Starting Cactus Chat with model: {bundle_dir}")
    print()

    return subprocess.run(cmd).returncode
