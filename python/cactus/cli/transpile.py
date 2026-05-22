from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .common import PROJECT_ROOT, RED, YELLOW, print_color
from .download import get_weights_dir
from .model import resolve_model_id


def _weights_dir_looks_transpile_ready(weights_dir):
    root = Path(weights_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
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


def cmd_transpile(args):
    """Invoke the saved-model transpiler entrypoint."""
    from .runtime import ensure_python_runtime_library

    model_id = resolve_model_id(args.model_id)
    extra_args = list(getattr(args, "extra_args", []) or [])
    allow_unconverted = bool(getattr(args, "allow_unconverted_weights", False))

    command = [sys.executable, "-m", "cactus.transpile.hf_model", "--model-id", model_id]
    if not _extra_args_has_option(extra_args, "--weights-dir"):
        default_weights_dir = get_weights_dir(model_id)
        if _weights_dir_looks_transpile_ready(default_weights_dir):
            command.extend(["--weights-dir", str(default_weights_dir)])
        elif not allow_unconverted:
            print_color(
                RED,
                "Error: transpilation requires converted Cactus CQ weights.",
            )
            print_color(
                YELLOW,
                "Run conversion first, then retry:\n"
                f"  cactus convert {model_id} {default_weights_dir} --bits 4\n"
                f"  cactus transpile {model_id} --weights-dir {default_weights_dir}",
            )
            return 1

    if allow_unconverted:
        command.append("--allow-unconverted-weights")
    if not getattr(args, "execute_after_transpile", False) and "--skip-execute" not in extra_args:
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
