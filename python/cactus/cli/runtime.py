from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from .common import PROJECT_ROOT, YELLOW, print_color


def _static_library_path():
    return PROJECT_ROOT / "cactus-engine" / "build" / "libcactus_engine.a"


def ensure_library():
    lib = _static_library_path()
    if lib.exists():
        return lib

    build_script = PROJECT_ROOT / "cactus-engine" / "build.sh"
    if subprocess.run([str(build_script)], cwd=build_script.parent).returncode != 0:
        raise RuntimeError("Failed to build the Cactus static runtime")
    return lib


def _python_runtime_library_path():
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    bundled = Path(__file__).resolve().parent.parent / "bindings" / "lib" / f"libcactus_engine{suffix}"
    if bundled.exists():
        return bundled
    return PROJECT_ROOT / "cactus-engine" / "build" / f"libcactus_engine{suffix}"


def _public_cactus_api_symbols(static_lib):
    cmd = (
        ["nm", "-gU", str(static_lib)] if platform.system() == "Darwin"
        else ["nm", "-g", "--defined-only", str(static_lib)]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed: {result.stderr.strip()}")

    def is_cactus_api(symbol):
        name = symbol[1:] if symbol.startswith("_") else symbol
        return name.startswith("cactus_")

    symbols = sorted({
        parts[-1] for line in result.stdout.splitlines()
        if (parts := line.split()) and is_cactus_api(parts[-1])
    })
    if not symbols:
        raise RuntimeError(f"No public cactus_* symbols in {static_lib}")
    return symbols


def _link_python_runtime_library(*, static_lib, library_path):
    build_dir = library_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        library_path.unlink()

    compiler_name = "clang++" if platform.system() == "Darwin" else "g++"
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise RuntimeError(f"{compiler_name} is not installed; cannot link libcactus_engine")

    exported_symbols = _public_cactus_api_symbols(static_lib)
    if platform.system() == "Darwin":
        command = [
            compiler,
            "-dynamiclib",
            "-o", str(library_path),
            *[f"-Wl,-u,{s}" for s in exported_symbols],
            str(static_lib),
            "-Wl,-install_name,@rpath/libcactus_engine.dylib",
            "-lcurl",
            "-framework", "Accelerate",
            "-framework", "CoreML",
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "Security",
            "-framework", "SystemConfiguration",
            "-framework", "CFNetwork",
        ]
    else:
        command = [
            compiler,
            "-shared",
            "-o", str(library_path),
            *[f"-Wl,--undefined={s}" for s in exported_symbols],
            str(static_lib),
            "-lcurl", "-pthread", "-ldl", "-lm",
        ]

    if subprocess.run(command, cwd=build_dir).returncode != 0:
        raise RuntimeError(f"Failed to link the Cactus shared runtime: {library_path}")


def ensure_python_runtime_library():
    library_path = _python_runtime_library_path()
    static_lib = _static_library_path()

    if (
        library_path.exists()
        and (not static_lib.exists()
             or library_path.stat().st_mtime >= static_lib.stat().st_mtime)
    ):
        return library_path

    print_color(YELLOW, "Preparing Cactus shared runtime...")
    if not static_lib.exists():
        static_lib = ensure_library()
    _link_python_runtime_library(static_lib=static_lib, library_path=library_path)
    return library_path
