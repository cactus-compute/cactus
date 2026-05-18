import os
import platform
import subprocess
import sys
import json
from pathlib import Path

<<<<<<< HEAD
from .common import (
    PROJECT_ROOT,
    _ensure_chat_binary,
    get_effective_weights_dir,
    print_color,
    RED, GREEN, YELLOW,
)
from .download import cmd_download
from cactus.transpile.model_profiles import profile_for_model_type
=======
from .common import PROJECT_ROOT, print_color, RED, GREEN
>>>>>>> origin/v2


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


<<<<<<< HEAD
def _validate_audio_path(audio_path: str) -> str | None:
    resolved = str(Path(audio_path).expanduser().resolve())
    if not Path(resolved).exists():
        print_color(RED, f"Error: Audio file not found: {resolved}")
        return None
    return resolved


def _model_type_from_weights_dir(weights_dir: Path) -> str:
    config_json = weights_dir / "config.json"
    if config_json.exists():
        try:
            payload = json.loads(config_json.read_text())
            model_type = str(payload.get("model_type", "") or "").strip().lower()
            if model_type:
                return model_type
        except Exception:
            pass
    config_txt = weights_dir / "config.txt"
    if config_txt.exists():
        for line in config_txt.read_text(errors="ignore").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "model_type":
                return value.strip().lower()
    return ""


def _should_avoid_native_loader(weights_dir: Path) -> bool:
    model_type = _model_type_from_weights_dir(weights_dir)
    profile = profile_for_model_type(model_type)
    return bool(profile and profile.avoid_native_loader)


def _prepare_transpiled_run_args(args, *, manifest_path: Path) -> int:
    args.bundle_dir = str(_transpiled_bundle_root_from_manifest(manifest_path))
    args._transpiled_from_run = True

    image_path = getattr(args, 'image', None)
    if image_path:
        resolved = _validate_image_path(image_path)
        if resolved is None:
            return 1
        args.image = resolved

    image_files = []
    for image_file in getattr(args, 'image_file', []) or []:
        if not image_file:
            continue
        resolved = _validate_image_path(str(image_file))
        if resolved is None:
            return 1
        image_files.append(resolved)
    args.image_file = image_files

    audio_path = getattr(args, 'audio', None) or getattr(args, 'audio_file', None)
    if audio_path:
        resolved = _validate_audio_path(str(audio_path))
        if resolved is None:
            return 1
        args.audio = resolved
        args.audio_file = resolved

    return 0
=======
def _ensure_chat_binary() -> Path | None:
    chat = PROJECT_ROOT / "cactus-engine" / "tests" / "build" / "chat"
    if chat.exists():
        return chat
    print_color(RED, "Error: chat binary not found. Run `cactus build` first.")
    return None
>>>>>>> origin/v2


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

<<<<<<< HEAD
    local_path = Path(model_id)
    if local_path.exists() and (local_path / "config.txt").exists():
        weights_dir = local_path
        print_color(GREEN, f"Using local model: {weights_dir}")
    else:
        download_result = cmd_download(args)
        if download_result != 0:
            return download_result
        weights_dir = get_effective_weights_dir(model_id, args)
        manifest_path = _resolve_transpiled_manifest(weights_dir)
        if manifest_path is not None:
            if _prepare_transpiled_run_args(args, manifest_path=manifest_path) != 0:
                return 1
            _clear_terminal_for_chat()
            print_color(GREEN, f"Starting Cactus Chat with model: {model_id}")
            print()
            return cmd_run_transpiled(args)

    weights_path = Path(weights_dir)
    if _should_avoid_native_loader(weights_path):
        print_color(
            RED,
            "This weights folder contains CQ weights but no transpiled Cactus component bundle.",
        )
        print_color(
            YELLOW,
            "Native C++ model subclasses are not available in this build, so this model must run through "
            "the transpiled graph path. Build the bundle first, then rerun:\n"
            f"  cactus convert {model_id} {weights_path} --bits 4\n"
            f"  cactus run {weights_path}",
        )
        return 1

    image_path = getattr(args, 'image', None)
    if image_path:
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            print_color(RED, f"Error: Image file not found: {image_path}")
            return 1
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        if Path(image_path).suffix.lower() not in valid_exts:
            print_color(RED, f"Error: Unsupported image format. Supported: {', '.join(valid_exts)}")
            return 1

    try:
        chat_binary = _ensure_chat_binary(PROJECT_ROOT, lib_path)
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
=======
    chat = _ensure_chat_binary()
    if chat is None:
>>>>>>> origin/v2
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
