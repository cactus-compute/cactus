import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    get_effective_weights_dir,
    print_color,
    RED,
    GREEN,
    YELLOW,
)
from .download import cmd_download
from cactus.transpile.model_profiles import profile_for_model_type


def _clear_terminal_for_chat() -> None:
    if sys.stdout.isatty():
        os.system("clear" if platform.system() != "Windows" else "cls")


def _ensure_chat_binary() -> Path | None:
    chat = PROJECT_ROOT / "cactus-engine" / "tests" / "build" / "chat"
    if chat.exists():
        return chat
    print_color(RED, "Error: chat binary not found. Run `cactus build` first.")
    return None


def _transpiled_bundle_root(path: Path) -> Path | None:
    path = path.expanduser().resolve()
    if (path / "components" / "manifest.json").exists():
        return path
    if path.name == "components" and (path / "manifest.json").exists():
        return path.parent
    return None


def _resolve_bundle_dir(model_id: str) -> Path | None:
    """Resolve a CLI model argument to a local transpiled bundle root."""
    candidates: list[Path] = []
    direct = Path(model_id).expanduser()
    if direct.exists():
        candidates.append(direct)

    try:
        weights_dir = Path(get_effective_weights_dir(model_id)).expanduser()
        if weights_dir.exists():
            candidates.append(weights_dir)
    except Exception:
        pass

    for candidate in candidates:
        bundle_root = _transpiled_bundle_root(candidate)
        if bundle_root is not None:
            return bundle_root
    return None


def _validate_image_path(image_path: str) -> str | None:
    resolved = str(Path(image_path).expanduser().resolve())
    path = Path(resolved)
    if not path.exists():
        print_color(RED, f"Error: Image file not found: {resolved}")
        return None
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if path.suffix.lower() not in valid_exts:
        print_color(RED, f"Error: Unsupported image format. Supported: {', '.join(sorted(valid_exts))}")
        return None
    return resolved


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


def _prepare_transpiled_run_args(args, *, bundle_dir: Path) -> int:
    args.bundle_dir = str(bundle_dir)
    args._transpiled_from_run = True

    image_path = getattr(args, "image", None)
    if image_path:
        resolved = _validate_image_path(str(image_path))
        if resolved is None:
            return 1
        args.image = resolved

    image_files = []
    for image_file in getattr(args, "image_file", []) or []:
        if not image_file:
            continue
        resolved = _validate_image_path(str(image_file))
        if resolved is None:
            return 1
        image_files.append(resolved)
    args.image_file = image_files

    audio_path = getattr(args, "audio", None) or getattr(args, "audio_file", None)
    if audio_path:
        resolved = _validate_audio_path(str(audio_path))
        if resolved is None:
            return 1
        args.audio = resolved
        args.audio_file = resolved

    return 0


def _run_transpiled_bundle(args, *, bundle_dir: Path, label: str) -> int:
    from .transpile import cmd_run_transpiled

    if _prepare_transpiled_run_args(args, bundle_dir=bundle_dir) != 0:
        return 1
    _clear_terminal_for_chat()
    print_color(GREEN, f"Starting Cactus Chat with model: {label}")
    print()
    return cmd_run_transpiled(args)


def _run_native_chat(args, *, weights_dir: Path, label: str) -> int:
    chat = _ensure_chat_binary()
    if chat is None:
        return 1

    cmd = [str(chat), str(weights_dir)]
    for flag, value in (
        ("--system", getattr(args, "system", None)),
        ("--prompt", getattr(args, "prompt", None)),
        ("--image", getattr(args, "image", None)),
        ("--audio", getattr(args, "audio", None) or getattr(args, "audio_file", None)),
    ):
        if not value:
            continue
        if flag in {"--image", "--audio"}:
            value = str(Path(str(value)).expanduser().resolve())
        cmd.extend([flag, str(value)])
    if getattr(args, "thinking", False):
        cmd.append("--thinking")

    _clear_terminal_for_chat()
    print_color(GREEN, f"Starting Cactus Chat with model: {label}")
    print()
    return subprocess.run(cmd).returncode


def cmd_run(args):
    """Download model if needed and start interactive chat."""
    model_id = args.model_id
    if getattr(args, "no_cloud_tele", False):
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

    bundle_dir = _resolve_bundle_dir(model_id)
    if bundle_dir is not None:
        return _run_transpiled_bundle(args, bundle_dir=bundle_dir, label=model_id)

    local_path = Path(model_id).expanduser()
    if local_path.exists() and (local_path / "config.txt").exists():
        weights_dir = local_path.resolve()
        print_color(GREEN, f"Using local model: {weights_dir}")
    else:
        download_result = cmd_download(args)
        if download_result != 0:
            return download_result
        weights_dir = Path(get_effective_weights_dir(model_id, args)).expanduser().resolve()

    bundle_dir = _resolve_bundle_dir(str(weights_dir))
    if bundle_dir is not None:
        return _run_transpiled_bundle(args, bundle_dir=bundle_dir, label=model_id)

    if _should_avoid_native_loader(weights_dir):
        print_color(
            RED,
            "This weights folder contains CQ weights but no transpiled Cactus component bundle.",
        )
        print_color(
            YELLOW,
            "Native C++ model subclasses are not available in this build, so this model must run "
            "through the transpiled graph path. Build the bundle first, then rerun:\n"
            f"  cactus convert {model_id} {weights_dir} --bits 4\n"
            f"  cactus run {weights_dir}",
        )
        return 1

    image_path = getattr(args, "image", None)
    if image_path and _validate_image_path(str(image_path)) is None:
        return 1
    audio_path = getattr(args, "audio", None) or getattr(args, "audio_file", None)
    if audio_path and _validate_audio_path(str(audio_path)) is None:
        return 1

    return _run_native_chat(args, weights_dir=weights_dir, label=model_id)
