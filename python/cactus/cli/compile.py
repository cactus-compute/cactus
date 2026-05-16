import os
import subprocess
import platform
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    check_command,
    run_command,
    print_color,
    RED, GREEN, YELLOW, BLUE,
)


def check_libcurl():
    """Check if libcurl development libraries are installed."""
    import platform

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


def cmd_build(args):
    """Build the Cactus library."""
    if getattr(args, 'apple', False):
        return cmd_build_apple(args)
    if getattr(args, 'android', False):
        return cmd_build_android(args)
    if getattr(args, 'flutter', False):
        return cmd_build_flutter(args)
    if getattr(args, 'python', False):
        return cmd_build_python(args)

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

    cactus_dir = PROJECT_ROOT / "cactus"
    lib_path = cactus_dir / "build" / "libcactus.a"

    print_color(YELLOW, "Building Cactus library...")
    build_script = cactus_dir / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1
    result = run_command(str(build_script), cwd=cactus_dir, check=False)
    if result.returncode != 0:
        print_color(RED, "Failed to build cactus library")
        return 1
    tests_dir = PROJECT_ROOT / "cactus-engine" / "tests"
    build_dir = tests_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling chat.cpp...")

    chat_cpp = tests_dir / "chat.cpp"
    if not chat_cpp.exists():
        print_color(RED, f"Error: chat.cpp not found at {chat_cpp}")
        return 1

    is_darwin = platform.system() == "Darwin"
    vendored_curl = PROJECT_ROOT / "cactus-engine" / "libs" / "curl" / "lib" / "libcurl.a"

    sdl2_available = False
    sdl2_flags = []
    sdl2_link = []
    if is_darwin:
        sdl2_check = subprocess.run(["brew", "list", "sdl2"], capture_output=True)
        if sdl2_check.returncode == 0:
            sdl2_prefix_result = subprocess.run(["brew", "--prefix", "sdl2"], capture_output=True, text=True)
            if sdl2_prefix_result.returncode == 0:
                sdl2_prefix = sdl2_prefix_result.stdout.strip()
                sdl2_flags = ["-DHAVE_SDL2", f"-I{sdl2_prefix}/include", f"-I{sdl2_prefix}/include/SDL2"]
                sdl2_link = [f"-L{sdl2_prefix}/lib", "-lSDL2"]
                sdl2_available = True
    else:
        sdl2_check = subprocess.run(["pkg-config", "--exists", "sdl2"], capture_output=True)
        if sdl2_check.returncode == 0:
            cflags = subprocess.run(["pkg-config", "--cflags", "sdl2"], capture_output=True, text=True)
            libs = subprocess.run(["pkg-config", "--libs", "sdl2"], capture_output=True, text=True)
            if cflags.returncode == 0 and libs.returncode == 0:
                sdl2_flags = ["-DHAVE_SDL2"] + cflags.stdout.strip().split()
                sdl2_link = libs.stdout.strip().split()
                sdl2_available = True

    if sdl2_available:
        print_color(GREEN, "SDL2 found - building with live audio support")
    else:
        print_color(YELLOW, "SDL2 not found - live mic recording will be disabled")
        print_color(YELLOW, "Install SDL2 for live mic support: brew install sdl2 (macOS)")
        print_color(YELLOW, "Then run `cactus build`")

    if is_darwin:
        curl_link = [str(vendored_curl)] if vendored_curl.exists() else ["-lcurl"]
        compiler = "clang++"
        cmd = [
            compiler, "-std=c++20", "-O3",
            "-DACCELERATE_NEW_LAPACK",
            f"-I{PROJECT_ROOT}",
            f"-I{PROJECT_ROOT / 'cactus-engine'}",
            f"-I{PROJECT_ROOT / 'cactus-graph'}",
            f"-I{PROJECT_ROOT / 'cactus-kernels'}",
            *sdl2_flags,
            str(chat_cpp),
            str(lib_path),
            "-o", "chat",
            *curl_link,
            "-framework", "Accelerate",
            "-framework", "CoreML",
            "-framework", "Foundation",
            "-framework", "Security",
            "-framework", "SystemConfiguration",
            "-framework", "CFNetwork",
            *sdl2_link,
        ]
    else:
        compiler = "g++"
        cmd = [
            compiler, "-std=c++20", "-O3",
            f"-I{PROJECT_ROOT}",
            f"-I{PROJECT_ROOT / 'cactus-engine'}",
            f"-I{PROJECT_ROOT / 'cactus-graph'}",
            f"-I{PROJECT_ROOT / 'cactus-kernels'}",
            *sdl2_flags,
            str(chat_cpp),
            str(lib_path),
            "-o", "chat",
            "-lcurl",
            "-pthread",
            *sdl2_link,
        ]

    if not check_command(compiler):
        print_color(RED, f"Error: {compiler} is not installed")
        return 1

    result = subprocess.run(cmd, cwd=build_dir)
    if result.returncode != 0:
        print_color(RED, "Build failed")
        return 1

    print_color(GREEN, f"Build complete: {build_dir / 'chat'}")

    asr_cpp = tests_dir / "asr.cpp"
    if asr_cpp.exists():
        print("Compiling asr.cpp...")

        if is_darwin:
            curl_link = [str(vendored_curl)] if vendored_curl.exists() else ["-lcurl"]
            cmd = [
                compiler, "-std=c++20", "-O3",
                "-DACCELERATE_NEW_LAPACK",
                f"-I{PROJECT_ROOT}",
                *sdl2_flags,
                str(asr_cpp),
                str(lib_path),
                "-o", "asr",
                *curl_link,
                "-framework", "Accelerate",
                "-framework", "CoreML",
                "-framework", "Foundation",
                "-framework", "Security",
                "-framework", "SystemConfiguration",
                "-framework", "CFNetwork",
                *sdl2_link,
            ]
        else:
            cmd = [
                compiler, "-std=c++20", "-O3",
                f"-I{PROJECT_ROOT}",
                *sdl2_flags,
                str(asr_cpp),
                str(lib_path),
                "-o", "asr",
                "-lcurl",
                "-pthread",
                *sdl2_link,
            ]

        result = subprocess.run(cmd, cwd=build_dir)
        if result.returncode != 0:
            print_color(RED, "ASR build failed")
            return 1

        print_color(GREEN, f"Build complete: {build_dir / 'asr'}")

    print_color(GREEN, "Cactus library built successfully!")
    print(f"Library location: {lib_path}")

    return 0


def cmd_build_apple(args):
    """Build Cactus for Apple platforms (iOS/macOS)."""
    print_color(BLUE, "Building Cactus for Apple platforms...")
    print("=" * 40)

    if platform.system() != "Darwin":
        print_color(RED, "Error: Apple builds require macOS")
        return 1

    build_script = PROJECT_ROOT / "apple" / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=PROJECT_ROOT / "apple", check=False)
    if result.returncode != 0:
        print_color(RED, "Apple build failed")
        return 1

    print_color(GREEN, "Apple build complete!")
    return 0


def cmd_build_android(args):
    """Build Cactus for Android."""
    print_color(BLUE, "Building Cactus for Android...")
    print("=" * 32)

    build_script = PROJECT_ROOT / "android" / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=PROJECT_ROOT / "android", check=False)
    if result.returncode != 0:
        print_color(RED, "Android build failed")
        return 1

    print_color(GREEN, "Android build complete!")
    return 0


def cmd_build_flutter(args):
    """Build Cactus for Flutter (iOS, macOS, Android)."""
    print_color(BLUE, "Building Cactus for Flutter...")
    print("=" * 32)

    build_script = PROJECT_ROOT / "flutter" / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=PROJECT_ROOT / "flutter", check=False)
    if result.returncode != 0:
        print_color(RED, "Flutter build failed")
        return 1

    print_color(GREEN, "Flutter build complete!")
    print()
    print("Output:")
    print(f"  flutter/libcactus.so")
    print(f"  flutter/cactus-ios.xcframework")
    print(f"  flutter/cactus-macos.xcframework")
    return 0


def cmd_build_python(args):
    """Build Cactus shared library for Python FFI."""
    print_color(BLUE, "Building Cactus for Python...")
    print("=" * 30)

    if not check_command('cmake'):
        print_color(RED, "Error: CMake is not installed")
        print("  macOS: brew install cmake")
        print("  Ubuntu: sudo apt-get install cmake")
        return 1

    cactus_dir = PROJECT_ROOT / "cactus"
    build_script = cactus_dir / "build.sh"
    if not build_script.exists():
        print_color(RED, f"Error: build.sh not found at {build_script}")
        return 1

    result = run_command(str(build_script), cwd=cactus_dir, check=False)
    if result.returncode != 0:
        print_color(RED, "Build failed")
        return 1

    if platform.system() == "Darwin":
        lib_name = "libcactus.dylib"
    else:
        lib_name = "libcactus.so"

    lib_path = cactus_dir / "build" / lib_name
    if not lib_path.exists():
        print_color(RED, f"Shared library not found at {lib_path}")
        return 1

    print_color(GREEN, "Python build complete!")
    print(f"Library: {lib_path}")
    return 0
