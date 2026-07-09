#!/usr/bin/env python3
import os
import subprocess
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
DEFAULT_TEST_MODEL_ID = DEFAULT_MODEL_ID
DEFAULT_TEST_TRANSCRIPTION_MODEL_ID = DEFAULT_TRANSCRIPTION_MODEL_ID


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


def convert_toolchain_error():
    """Message if the model-conversion toolchain (torch/transformers) is missing, else None."""
    import importlib.util
    missing = [m for m in ("torch", "transformers")
               if importlib.util.find_spec(m) is None]
    if not missing:
        return None
    return (
        "the model-conversion toolchain is not installed (missing: "
        + ", ".join(missing) + ").\n"
        "  Converting a model from source needs it. Either:\n"
        "    - use a prebuilt model:   cactus run <model>   or   cactus download <model>\n"
        "    - install the toolchain:  pip install \"cactus-compute[convert]\""
    )


BIN_DIR = SCRIPT_DIR.parent / "bin"


def apply_cloud_api_key_env() -> None:
    from .config_utils import CactusConfig
    try:
        api_key = CactusConfig().get_api_key()
    except (OSError, ValueError):
        return
    if api_key:
        os.environ["CACTUS_CLOUD_KEY"] = api_key


def apply_runtime_env(args) -> None:
    """Prepare the process env for an inference command: honour --no-cloud-tele
    and load the stored cloud API key."""
    if getattr(args, "no_cloud_tele", False):
        os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
    apply_cloud_api_key_env()


def is_valid_bundle(path) -> bool:
    """A runnable v2 bundle has both config.txt and the components manifest."""
    path = Path(path)
    return (path / "config.txt").exists() and (path / "components" / "manifest.json").exists()


def launch_binary(name, *args) -> int:
    """Resolve a bundled binary and exec it with str-coerced args. Returns its
    exit code, or 1 if the binary is unavailable."""
    binary = resolve_binary(name)
    if binary is None:
        return 1
    return subprocess.run([str(binary), *(str(a) for a in args)]).returncode


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
    return None
