#!/usr/bin/env python3
import sys
import os
import argparse
import json
import subprocess
import shutil
import platform
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / "python" / "cactus" / "cli").is_dir()
        and (path / "cactus-kernels").is_dir()
    )


def _resolve_project_root() -> Path:
    env_root = os.getenv("CACTUS_PROJECT_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _looks_like_project_root(candidate):
            return candidate

    module_root = SCRIPT_DIR.parent.parent.parent
    if _looks_like_project_root(module_root):
        return module_root

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if _looks_like_project_root(candidate):
            return candidate

    # Final fallback for unusual layouts.
    return module_root


PROJECT_ROOT = _resolve_project_root()
DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_TEST_MODEL_ID = "google/gemma-4-E2B-it"

try:
    with open(PROJECT_ROOT / "models.json", "r", encoding="utf-8") as f:
        MODEL_REGISTRY = json.load(f)
except Exception:
    MODEL_REGISTRY = {}

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def print_color(color, message):
    """Print a message with ANSI color codes."""
    print(f"{color}{message}{NC}")


from .download import (
    combo_label,
    download_cq_archive,
    get_model_dir_name,
    get_weights_dir,
    list_hf_cq_archives,
    resolve_archive,
    suggested_cq_repo,
)


def get_effective_weights_dir(model_id, args=None):
    return get_weights_dir(model_id)



def check_command(cmd):
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def run_command(cmd, cwd=None, check=True):
    """Run a script or command and optionally exit on failure.

    Args:
        cmd: Script path (str) or command list. String paths are executed
             directly without shell interpretation to handle spaces safely.
        cwd: Working directory for the command.
        check: If True, exit on non-zero return code.
    """
    if isinstance(cmd, str):
        cmd = [cmd]
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def _is_stale_binary(binary_path, dependency_paths):
    binary_path = Path(binary_path)
    if not binary_path.exists():
        return True

    try:
        binary_mtime = binary_path.stat().st_mtime
    except OSError:
        return True

    for dep in dependency_paths:
        dep_path = Path(dep)
        if not dep_path.exists():
            continue
        try:
            if dep_path.stat().st_mtime > binary_mtime:
                return True
        except OSError:
            continue

    return False


def _ensure_chat_binary(project_root, lib_path):
    from .compile import cmd_build
    tests_dir = project_root / "cactus-engine" / "tests"
    build_dir = tests_dir / "build"
    chat_binary = build_dir / "chat"
    chat_cpp = tests_dir / "chat.cpp"

    if not _is_stale_binary(chat_binary, [lib_path, chat_cpp]):
        return chat_binary

    print_color(YELLOW, "Refreshing chat binary for current Cactus library...")
    build_args = argparse.Namespace(
        apple=False,
        android=False,
        flutter=False,
        python=False,
    )
    result = cmd_build(build_args)
    if result != 0 or not chat_binary.exists():
        raise RuntimeError("Failed to rebuild chat binary")

    return chat_binary


def prompt_for_api_key(config):
    """Prompt user to set Cactus Cloud API key if not already configured. Returns the key or empty string."""
    api_key = config.get_api_key()
    if api_key:
        return api_key

    print("\n" + "="*50)
    print("  Cactus Cloud Setup (Optional)")
    print("="*50 + "\n")
    print("Get your cloud key at \033[1;36mhttps://www.cactuscompute.com/dashboard/api-keys\033[0m")
    print("to enable automatic cloud fallback.\n")

    api_key = input("Your Cactus Cloud key (press Enter to skip): ").strip()
    if api_key:
        config.set_api_key(api_key)
        masked = api_key[:4] + "..." + api_key[-4:]
        print_color(GREEN, f"API key saved: {masked}")
    print()
    return api_key
