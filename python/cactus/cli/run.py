import os
import platform
import sys
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    _ensure_chat_binary,
    get_effective_weights_dir,
    print_color,
    RED, GREEN,
)
from .download import cmd_download


def _transpiled_bundle_root_from_manifest(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "components":
        return manifest_path.parent.parent
    return manifest_path.parent


def _clear_terminal_for_chat() -> None:
    if sys.stdout.isatty():
        os.system('clear' if platform.system() != 'Windows' else 'cls')


def _validate_image_path(image_path: str) -> str | None:
    resolved = str(Path(image_path).expanduser().resolve())
    path = Path(resolved)
    if not path.exists():
        print_color(RED, f"Error: Image file not found: {resolved}")
        return None
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    if path.suffix.lower() not in valid_exts:
        print_color(RED, f"Error: Unsupported image format. Supported: {', '.join(valid_exts)}")
        return None
    return resolved


def _validate_audio_path(audio_path: str) -> str | None:
    resolved = str(Path(audio_path).expanduser().resolve())
    if not Path(resolved).exists():
        print_color(RED, f"Error: Audio file not found: {resolved}")
        return None
    return resolved


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


def cmd_run(args):
    """Download model if needed and start interactive chat."""
    model_id = args.model_id

    from .transpile import _resolve_transpiled_manifest, cmd_run_transpiled

    manifest_path = _resolve_transpiled_manifest(model_id)
    if manifest_path is not None:
        if getattr(args, 'no_cloud_tele', False):
            os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
        if _prepare_transpiled_run_args(args, manifest_path=manifest_path) != 0:
            return 1
        _clear_terminal_for_chat()
        print_color(GREEN, f"Starting Cactus Chat with model: {model_id}")
        print()
        return cmd_run_transpiled(args)

    from .config_utils import CactusConfig
    from .common import prompt_for_api_key

    config = CactusConfig()
    api_key = prompt_for_api_key(config)

    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key

    if getattr(args, 'no_cloud_tele', False):
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    lib_path = PROJECT_ROOT / "cactus" / "build" / "libcactus.a"
    if not lib_path.exists():
        print_color(RED, "Error: Cactus library not built. Run 'cactus build' first.")
        return 1

    local_path = Path(model_id)
    if local_path.exists() and (local_path / "config.txt").exists():
        weights_dir = local_path
        print_color(GREEN, f"Using local model: {weights_dir}")
    else:
        download_result = cmd_download(args)
        if download_result != 0:
            return download_result
        weights_dir = get_effective_weights_dir(model_id, args)

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
        return 1

    _clear_terminal_for_chat()
    print_color(GREEN, f"Starting Cactus Chat with model: {model_id}")
    print()

    audio_path = getattr(args, 'audio', None)
    if audio_path:
        audio_path = str(Path(audio_path).resolve())
        if not Path(audio_path).exists():
            print_color(RED, f"Error: Audio file not found: {audio_path}")
            return 1

    cmd_args = [str(chat_binary), str(weights_dir)]
    if image_path:
        cmd_args.extend(['--image', image_path])
    if audio_path:
        cmd_args.extend(['--audio', audio_path])
    system_prompt = getattr(args, 'system', None)
    if system_prompt:
        cmd_args.extend(['--system', system_prompt])
    prompt = getattr(args, 'prompt', None)
    if prompt:
        cmd_args.extend(['--prompt', prompt])
    if getattr(args, 'thinking', False):
        cmd_args.append('--thinking')

    os.execv(str(chat_binary), cmd_args)
