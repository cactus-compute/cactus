#!/usr/bin/env python3
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _looks_like_project_root(path):
    return (
        (path / "python" / "cactus" / "cli").is_dir()
        and (path / "cactus-kernels").is_dir()
    )


PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent


def is_repo_checkout():
    return _looks_like_project_root(PROJECT_ROOT)


def weights_root() -> Path:
    if is_repo_checkout():
        return PROJECT_ROOT / "weights"
    return Path.home() / ".cache" / "cactus" / "weights"


def transpiled_root() -> Path:
    if is_repo_checkout():
        return PROJECT_ROOT / "transpiled"
    return Path.home() / ".cache" / "cactus" / "transpiled"


DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_TRANSCRIPTION_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_TEST_MODEL_ID = "LiquidAI/LFM2-VL-450M"
DEFAULT_TEST_TRANSCRIPTION_MODEL_ID = "openai/whisper-base"


# Add a new vendor accelerator by appending its name here.
SUPPORTED_PLATFORMS: tuple[str, ...] = ("apple",)


RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[1;36m'
NC = '\033[0m'


def _color_enabled():
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def print_color(color, message):
    if _color_enabled():
        print(f"{color}{message}{NC}")
    else:
        print(message)


def mask_key(key):
    return key[:4] + "..." + key[-4:] if len(key) >= 8 else "***"


BIN_DIR = SCRIPT_DIR.parent / "bin"


def apply_cloud_api_key_env() -> None:
    from .config_utils import CactusConfig
    api_key = CactusConfig().get_api_key()
    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key


def _auto_build_binaries() -> bool:
    """Build the native runtime and CLI binaries, equivalent to `cactus build`."""
    from argparse import Namespace
    from .compile import cmd_build

    return cmd_build(Namespace(apple=False, android=False, python=False)) == 0


def resolve_binary(name):
    path = BIN_DIR / name
    if path.exists():
        return path

    # First run in a source checkout: build automatically instead of erroring out.
    if is_repo_checkout():
        print_color(YELLOW, f"{name} binary not found; building Cactus (first run, this may take a minute)...")
        if _auto_build_binaries() and path.exists():
            return path

    print_color(RED, f"{name} binary not found at {path}.")
    if is_repo_checkout():
        print_color(RED, "Run `cactus build` first.")
    return None
