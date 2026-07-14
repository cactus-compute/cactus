from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .common import PROJECT_ROOT, RED, YELLOW, print_color
from .download import get_weights_dir


def _weights_dir_looks_transpile_ready(weights_dir):
    root = Path(weights_dir).expanduser().resolve()
    if not root.is_dir():
        return False
    if (root / "weights_manifest.json").exists():
        return True
    if any(root.glob("*.cq[1-4].weights")):
        return True
    return (root / "config.txt").exists() and any(root.glob("*.weights"))


def _extra_args_has_option(extra_args, option):
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in extra_args)


def _prepend_python_path(env):
    python_root = str(PROJECT_ROOT / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = python_root if not existing else f"{python_root}{os.pathsep}{existing}"


def run_transpile(model_id, *, extra_args, execute_after_transpile=False,
                  allow_unconverted_weights=False):
    from .common import convert_toolchain_error
    from .runtime import ensure_python_runtime_library

    err = convert_toolchain_error()
    if err:
        print_color(RED, f"Error: {err}")
        return 1

    extra_args = list(extra_args or [])
    command = [sys.executable, "-m", "cactus.transpile.hf_model", "--model-id", model_id]
    default_weights_dir = get_weights_dir(model_id)
    if not default_weights_dir.name:
        print_color(RED, f"Error: cannot derive a unique output dir from model_id {model_id!r}.")
        print_color(YELLOW, "Pass --weights-dir and --artifact-dir explicitly.")
        return 1
    if not _extra_args_has_option(extra_args, "--weights-dir"):
        if _weights_dir_looks_transpile_ready(default_weights_dir):
            command.extend(["--weights-dir", str(default_weights_dir)])
    if not _extra_args_has_option(extra_args, "--artifact-dir"):
        command.extend(["--artifact-dir", str(default_weights_dir)])

    if allow_unconverted_weights:
        command.append("--allow-unconverted-weights")
    if not execute_after_transpile and "--skip-execute" not in extra_args:
        command.append("--skip-execute")
    command.extend(extra_args)

    try:
        transpile_lib = ensure_python_runtime_library()
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        return 1

    env = os.environ.copy()
    env["CACTUS_LIB_PATH"] = str(transpile_lib)
    _prepend_python_path(env)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode
