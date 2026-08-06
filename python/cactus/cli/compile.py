import shutil
import subprocess
import platform
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    print_color,
    RED, GREEN, YELLOW, BLUE,
)


def check_command(cmd):
    return shutil.which(cmd) is not None


def run_command(cmd, cwd=None):
    if isinstance(cmd, str):
        cmd = [cmd]
    return subprocess.run(cmd, cwd=cwd)


def check_libcurl():
    if platform.system() == 'Darwin':
        return True

    if check_command('pkg-config'):
        result = subprocess.run(['pkg-config', '--exists', 'libcurl'], capture_output=True)
        if result.returncode == 0:
            return True

    curl_paths = [
        '/usr/include/curl/curl.h',
        '/usr/include/x86_64-linux-gnu/curl/curl.h',
        '/usr/include/aarch64-linux-gnu/curl/curl.h',
        '/usr/local/include/curl/curl.h',
    ]
    for path in curl_paths:
        if Path(path).exists():
            return True

    return False


def _detect_sdl2() -> tuple[list[str], list[str]]:
    if not check_command("pkg-config"):
        return [], []

    if subprocess.run(["pkg-config", "--exists", "sdl2"], capture_output=True).returncode != 0:
        return [], []

    cflags = subprocess.run(["pkg-config", "--cflags", "sdl2"], capture_output=True, text=True)
    libs = subprocess.run(["pkg-config", "--libs", "sdl2"], capture_output=True, text=True)
    if cflags.returncode != 0 or libs.returncode != 0:
        return [], []

    return (
        ["-DHAVE_SDL2"] + cflags.stdout.strip().split(),
        libs.stdout.strip().split(),
    )


def build_binary(
    name: str,
    lib_path: Path,
    *,
    sdl2: tuple[list[str], list[str]] | None = None,
) -> int:
    tests_dir = PROJECT_ROOT / "cactus-engine" / "tests"
    source = tests_dir / f"{name}.cpp"
    build_dir = tests_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    is_darwin = platform.system() == "Darwin"
    sdl2_flags, sdl2_link = sdl2 if sdl2 is not None else _detect_sdl2()

    include_dirs = [
        PROJECT_ROOT,
        PROJECT_ROOT / "cactus-engine",
        PROJECT_ROOT / "cactus-graph",
        PROJECT_ROOT / "cactus-kernels",
        PROJECT_ROOT / "cactus-kernels" / "src",
    ]

    print(f"Compiling {name}.cpp...")

    if is_darwin:
        vendored_curl = PROJECT_ROOT / "cactus-engine" / "libs" / "curl" / "lib" / "libcurl.a"
        curl_link = [str(vendored_curl)] if vendored_curl.exists() else ["-lcurl"]
        compiler = "clang++"
        cmd = [
            compiler, "-std=c++20", "-O3",
            "-DACCELERATE_NEW_LAPACK",
            *[f"-I{d}" for d in include_dirs],
            *sdl2_flags,
            str(source), str(lib_path),
            "-o", name,
            *curl_link,
            "-framework", "Accelerate",
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "Security",
            "-framework", "SystemConfiguration",
            "-framework", "CFNetwork",
            *sdl2_link,
        ]
    else:
        compiler = "g++"
        cmd = [
            compiler, "-std=c++20", "-O3",
            *[f"-I{d}" for d in include_dirs],
            *sdl2_flags,
            str(source), str(lib_path),
            "-o", name,
            "-lcurl", "-pthread",
            *sdl2_link,
        ]

    if not check_command(compiler):
        print_color(RED, f"Error: {compiler} is not installed")
        return 1

    result = subprocess.run(cmd, cwd=build_dir)
    if result.returncode != 0:
        print_color(RED, f"{name} build failed")
        return 1

    bin_dir = Path(__file__).resolve().parent.parent / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build_dir / name, bin_dir / name)
    print_color(GREEN, f"Build complete: {bin_dir / name}")
    return 0


def cmd_build(args):
    if args.apple:
        return _build_with_script("apple", "Building Cactus for Apple platforms")
    if args.android:
        return _build_with_script("android", "Building Cactus for Android")
    if args.python:
        return cmd_build_python()

    print_color(BLUE, "Building Cactus library...")
    print("=" * 24)

    if not check_command('cmake'):
        print_color(RED, "Error: CMake is not installed")
        print("  macOS: brew install cmake")
        print("  Ubuntu: sudo apt-get install cmake build-essential")
        return 1

    if not check_libcurl():
        print_color(RED, "Error: libcurl development libraries not found")
        print("  macOS: brew install curl")
        print("  Ubuntu: sudo apt-get install libcurl4-openssl-dev")
        return 1

    cactus_dir = PROJECT_ROOT / "cactus-engine"
    lib_path = cactus_dir / "build" / "libcactus_engine.a"

    if run_command(str(cactus_dir / "build.sh"), cwd=cactus_dir).returncode != 0:
        print_color(RED, "Failed to build cactus library")
        return 1

    sdl2 = _detect_sdl2()
    if sdl2[0]:
        print_color(GREEN, "SDL2 found - voice input enabled")
    else:
        print_color(YELLOW, "SDL2 not found - voice input disabled")
        print_color(YELLOW, "Install SDL2 to enable voice input: brew install sdl2 (macOS)")

    rc = build_binary("run", lib_path, sdl2=([], []))
    if rc != 0:
        return rc
    rc = build_binary("transcribe", lib_path, sdl2=sdl2)
    if rc != 0:
        return rc

    print_color(GREEN, "Cactus library built successfully!")
    print(f"Library location: {lib_path}")

    return 0


def _build_with_script(subdir, title):
    print_color(BLUE, f"{title}...")

    if subdir == "apple" and platform.system() != "Darwin":
        print_color(RED, "Error: Apple builds require macOS")
        return 1

    if run_command(str(PROJECT_ROOT / subdir / "build.sh"), cwd=PROJECT_ROOT / subdir).returncode != 0:
        print_color(RED, f"{title} failed")
        return 1

    print_color(GREEN, f"{title} complete!")
    return 0


def cmd_build_python():
    print_color(BLUE, "Building Cactus for Python...")

    if not check_command('cmake'):
        print_color(RED, "Error: CMake is not installed")
        print("  macOS: brew install cmake")
        print("  Ubuntu: sudo apt-get install cmake")
        return 1

    cactus_dir = PROJECT_ROOT / "cactus-engine"
    if run_command(str(cactus_dir / "build.sh"), cwd=cactus_dir).returncode != 0:
        print_color(RED, "Build failed")
        return 1

    lib_name = "libcactus_engine.dylib" if platform.system() == "Darwin" else "libcactus_engine.so"
    lib_path = cactus_dir / "build" / lib_name
    print_color(GREEN, "Python build complete!")
    print(f"Library: {lib_path}")
    return 0
