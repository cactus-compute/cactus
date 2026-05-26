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
    from .runtime import ensure_python_runtime_library

    extra_args = list(extra_args or [])
    command = [sys.executable, "-m", "cactus.transpile.hf_model", "--model-id", model_id]
    if not _extra_args_has_option(extra_args, "--weights-dir"):
        default_weights_dir = get_weights_dir(model_id)
        if _weights_dir_looks_transpile_ready(default_weights_dir):
            command.extend(["--weights-dir", str(default_weights_dir)])
        elif not allow_unconverted_weights:
            print_color(RED, "Error: transpilation requires converted Cactus CQ weights.")
            print_color(YELLOW, f"Run conversion first: cactus convert {model_id}")
            return 1

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
