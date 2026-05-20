from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .common import (
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    RED,
    YELLOW,
    print_color,
)
from .download import get_weights_dir


MODEL_ID_ALIASES = {
    "gemma4": DEFAULT_MODEL_ID,
    "gemma4-e2b": DEFAULT_MODEL_ID,
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "parakeet-tdt": "nvidia/parakeet-tdt-0.6b-v3",
    "whisper": "openai/whisper-small",
    "qwen": "Qwen/Qwen3-1.7B",
    "lfm": "LiquidAI/LFM2-VL-450M",
}


def resolve_model_id_alias(model_id: str) -> str:
    normalized = (model_id or "").strip()
    return MODEL_ID_ALIASES.get(normalized.lower(), normalized)


def _python_runtime_library_path() -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return PROJECT_ROOT / "cactus" / "build" / f"libcactus{suffix}"


def _static_cactus_library_path() -> Path:
    return PROJECT_ROOT / "cactus" / "build" / "libcactus.a"


def _build_static_cactus_library() -> Path:
    build_script = PROJECT_ROOT / "cactus" / "build.sh"
    if not build_script.exists():
        raise RuntimeError(f"The Cactus build script is missing: {build_script}")

    build = subprocess.run([str(build_script)], cwd=PROJECT_ROOT / "cactus")
    if build.returncode != 0:
        raise RuntimeError("Failed to build the Cactus static runtime")

    static_library_path = _static_cactus_library_path()
    if not static_library_path.exists():
        raise RuntimeError(
            "The Cactus build completed, but the static library was not produced.\n"
            f"Expected: {static_library_path}"
        )
    return static_library_path


def _public_cactus_api_symbols(static_library_path: Path) -> list[str]:
    if platform.system() == "Darwin":
        command = ["nm", "-gU", str(static_library_path)]
    else:
        command = ["nm", "-g", "--defined-only", str(static_library_path)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to inspect the Cactus static runtime symbols.\n"
            f"Command: {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )

    symbols: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[-1].strip()
        normalized = symbol[1:] if symbol.startswith("_") else symbol
        if normalized.startswith("cactus_") and symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise RuntimeError(f"Could not find any public cactus_* symbols in {static_library_path}")
    return symbols


def _link_python_runtime_library(*, static_library_path: Path, library_path: Path) -> None:
    build_dir = library_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        library_path.unlink()

    exported_symbols = _public_cactus_api_symbols(static_library_path)
    if platform.system() == "Darwin":
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.dylib")
        command = [
            compiler,
            "-dynamiclib",
            "-o",
            str(library_path),
            *[f"-Wl,-u,{symbol}" for symbol in exported_symbols],
            str(static_library_path),
            "-Wl,-install_name,@rpath/libcactus.dylib",
            "-lcurl",
            "-framework",
            "Accelerate",
            "-framework",
            "CoreML",
            "-framework",
            "Foundation",
            "-framework",
            "Security",
            "-framework",
            "SystemConfiguration",
            "-framework",
            "CFNetwork",
        ]
    else:
        compiler = shutil.which("g++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.so")
        command = [
            compiler,
            "-shared",
            "-o",
            str(library_path),
            *[f"-Wl,--undefined={symbol}" for symbol in exported_symbols],
            str(static_library_path),
            "-lcurl",
            "-pthread",
            "-ldl",
            "-lm",
        ]

    result = subprocess.run(command, cwd=build_dir)
    if result.returncode != 0 or not library_path.exists():
        raise RuntimeError(f"Failed to link the Cactus shared runtime: {library_path}")


def _ensure_python_runtime_library() -> Path:
    library_path = _python_runtime_library_path()
    static_library_path = _static_cactus_library_path()
    if (
        library_path.exists()
        and static_library_path.exists()
        and library_path.stat().st_mtime >= static_library_path.stat().st_mtime
    ):
        return library_path

    print_color(YELLOW, "Preparing Cactus shared runtime for transpiler...")
    if not static_library_path.exists():
        static_library_path = _build_static_cactus_library()
    _link_python_runtime_library(static_library_path=static_library_path, library_path=library_path)
    return library_path


def _weights_dir_looks_transpile_ready(weights_dir: Path) -> bool:
    root = Path(weights_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return False
    if (root / "weights_manifest.json").exists():
        return True
    if any(root.glob("*.cq[1-4].weights")):
        return True
    # Legacy converted Cactus weight folders may not have a manifest yet, but
    # they should still carry runtime config plus Cactus tensor files.
    return (root / "config.txt").exists() and any(root.glob("*.weights"))


def _extra_args_has_option(extra_args: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in extra_args)


def _prepend_python_path(env: dict[str, str]) -> None:
    python_root = str(PROJECT_ROOT / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = python_root if not existing else f"{python_root}{os.pathsep}{existing}"


def cmd_transpile(args) -> int:
    """Invoke the saved-model transpiler entrypoint."""
    model_id = resolve_model_id_alias(args.model_id)
    extra_args = list(getattr(args, "extra_args", []) or [])
    allow_unconverted = bool(getattr(args, "allow_unconverted_weights", False))

    command = [sys.executable, "-m", "cactus.transpile.hf_model", "--model-id", model_id]
    if _extra_args_has_option(extra_args, "--weights-dir"):
        pass
    else:
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
        transpile_lib = _ensure_python_runtime_library()
    except RuntimeError as exc:
        print_color(RED, f"Error: {exc}")
        return 1

    env = os.environ.copy()
    env["CACTUS_LIB_PATH"] = str(transpile_lib)
    _prepend_python_path(env)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return result.returncode
