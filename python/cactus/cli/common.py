#!/usr/bin/env python3
import os
import subprocess
import shutil
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


DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_TEST_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def print_color(color, message):
    """Print a message with ANSI color codes."""
    print(f"{color}{message}{NC}")


def mask_key(key):
    return key[:4] + "..." + key[-4:] if len(key) >= 8 else "***"


def check_command(cmd):
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def run_command(cmd, cwd=None):
    if isinstance(cmd, str):
        cmd = [cmd]
    return subprocess.run(cmd, cwd=cwd)


def prompt_for_api_key(config):
    """Prompt user to set Cactus Cloud API key if not already configured."""
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
        print_color(GREEN, f"API key saved: {mask_key(api_key)}")
    print()
    return api_key
