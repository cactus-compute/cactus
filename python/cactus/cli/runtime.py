"""Build-system helpers for the C++ runtime."""
from __future__ import annotations

import platform
import shutil
import subprocess

from .common import PROJECT_ROOT, YELLOW, print_color


# ── Static library ────────────────────────────────────────────────────


def _static_library_path():
    return PROJECT_ROOT / "cactus" / "build" / "libcactus.a"


def ensure_library():
    """Build libcactus.a if it doesn't exist. Return its path."""
    lib = _static_library_path()
    if lib.exists():
        return lib

    build_script = PROJECT_ROOT / "cactus" / "build.sh"
    if not build_script.exists():
        raise RuntimeError(f"The Cactus build script is missing: {build_script}")

    result = subprocess.run([str(build_script)], cwd=PROJECT_ROOT / "cactus")
    if result.returncode != 0:
        raise RuntimeError("Failed to build the Cactus static runtime")

    if not lib.exists():
        raise RuntimeError(
            "The Cactus build completed, but the static library was not produced.\n"
            f"Expected: {lib}"
        )
    return lib


# ── Shared dynamic library (for transpiler Python FFI) ────────────────


def _python_runtime_library_path():
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return PROJECT_ROOT / "cactus" / "build" / f"libcactus{suffix}"


def _public_cactus_api_symbols(static_lib):
    """Extract cactus_* symbols from the static library."""
    if platform.system() == "Darwin":
        command = ["nm", "-gU", str(static_lib)]
    else:
        command = ["nm", "-g", "--defined-only", str(static_lib)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to inspect the Cactus static runtime symbols.\n"
            f"Command: {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )

    seen = set()
    symbols = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[-1].strip()
        normalized = symbol[1:] if symbol.startswith("_") else symbol
        if normalized.startswith("cactus_") and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    if not symbols:
        raise RuntimeError(
            f"Could not find any public cactus_* symbols in {static_lib}"
        )
    return symbols


def _link_python_runtime_library(*, static_lib, library_path):
    """Link static lib into a shared .dylib/.so with exported symbols."""
    build_dir = library_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        library_path.unlink()

    exported_symbols = _public_cactus_api_symbols(static_lib)
    if platform.system() == "Darwin":
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.dylib")
        command = [
            compiler,
            "-dynamiclib",
            "-o", str(library_path),
            *[f"-Wl,-u,{s}" for s in exported_symbols],
            str(static_lib),
            "-Wl,-install_name,@rpath/libcactus.dylib",
            "-lcurl",
            "-framework", "Accelerate",
            "-framework", "CoreML",
            "-framework", "Foundation",
            "-framework", "Security",
            "-framework", "SystemConfiguration",
            "-framework", "CFNetwork",
        ]
    else:
        compiler = shutil.which("g++") or shutil.which("c++")
        if not compiler:
            raise RuntimeError("Failed to find a C++ compiler for linking libcactus.so")
        command = [
            compiler,
            "-shared",
            "-o", str(library_path),
            *[f"-Wl,--undefined={s}" for s in exported_symbols],
            str(static_lib),
            "-lcurl", "-pthread", "-ldl", "-lm",
        ]

    result = subprocess.run(command, cwd=build_dir)
    if result.returncode != 0 or not library_path.exists():
        raise RuntimeError(f"Failed to link the Cactus shared runtime: {library_path}")


def ensure_python_runtime_library():
    """Build shared .dylib/.so for the transpiler. Return its path."""
    library_path = _python_runtime_library_path()
    static_lib = _static_library_path()

    if (
        library_path.exists()
        and static_lib.exists()
        and library_path.stat().st_mtime >= static_lib.stat().st_mtime
    ):
        return library_path

    print_color(YELLOW, "Preparing Cactus shared runtime for transpiler...")
    if not static_lib.exists():
        static_lib = ensure_library()
    _link_python_runtime_library(static_lib=static_lib, library_path=library_path)
    return library_path
