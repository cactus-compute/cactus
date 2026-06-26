import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    print_color,
    RED, GREEN, YELLOW, BLUE,
)


def cmd_clean(args):
    """Remove all build artifacts, caches, and downloaded weights."""
    print_color(BLUE, "Cleaning all build artifacts from Cactus project...")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    destructive_targets = [
        PROJECT_ROOT / "weights",
        PROJECT_ROOT / "transpiled",
        PROJECT_ROOT / "venv",
    ]
    present = [t for t in destructive_targets if t.exists()]
    if present and not args.yes:
        print_color(YELLOW, "This also deletes the following (not just build artifacts):")
        for t in present:
            print(f"  - {t}")
        print_color(YELLOW, "Downloaded weights are NOT re-fetched by setup and will be lost.")
        if not sys.stdin.isatty():
            print_color(RED, "Refusing to proceed without confirmation. Re-run with --yes to skip this prompt.")
            return 0
        if input("Continue? [y/N]: ").strip().lower() not in ("y", "yes"):
            print_color(YELLOW, "Aborted.")
            return 0

    print("Stopping any running Cactus server...")
    try:
        stopped = subprocess.run(["pkill", "-f", "cactus serve"], capture_output=True)
        print("Stopped running Cactus server(s)." if stopped.returncode == 0 else "No running Cactus server found.")
    except FileNotFoundError:
        print_color(YELLOW, "Could not stop the server automatically; stop it manually if one is running.")
    print()

    preserve_roots = [
        (PROJECT_ROOT / "cactus-engine" / "libs" / "curl").resolve(),
        (PROJECT_ROOT / "android" / "mbedtls").resolve(),
        (PROJECT_ROOT / "libs" / "mbedtls").resolve(),
    ]

    def should_preserve(path):
        resolved = path.resolve()
        return any(resolved.is_relative_to(root) for root in preserve_roots)

    def remove_if_exists(path):
        if path.is_dir():
            print(f"Removing: {path}")
            shutil.rmtree(path, ignore_errors=True)

    for tree in ("venv", "weights", "transpiled"):
        remove_if_exists(PROJECT_ROOT / tree)
    remove_if_exists(PROJECT_ROOT / "python" / "cactus" / "bin")
    remove_if_exists(PROJECT_ROOT / "python" / "cactus" / "code")

    print()
    print("Sweeping build, dependency, and cache artifacts across the tree...")

    def is_artifact_dir(name):
        return (
            name in {"node_modules", "dist", "__pycache__", ".pytest_cache"}
            or name == "build"
            or name.startswith(("build-", "build_"))
            or name.endswith((".egg-info", ".xcframework"))
        )

    artifact_file_suffixes = {".so", ".a", ".bin", ".dylib"}

    removed_dirs = 0
    removed_files = 0
    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=True):
        root_path = Path(root)
        kept = []
        for d in dirs:
            if d == ".git":
                continue
            full = root_path / d
            if is_artifact_dir(d) and not should_preserve(full):
                print(f"Removing: {full}")
                shutil.rmtree(full, ignore_errors=True)
                removed_dirs += 1
            else:
                kept.append(d)
        dirs[:] = kept
        for f in files:
            full = root_path / f
            if full.suffix in artifact_file_suffixes and not should_preserve(full):
                try:
                    full.unlink()
                    removed_files += 1
                except OSError:
                    pass
    print(f"Removed {removed_dirs} build/dependency directories and {removed_files} compiled artifacts")

    print()
    print_color(GREEN, "Clean complete!")
    print("All build artifacts, weights, and venv have been removed.")
    print()

    print_color(BLUE, "Re-running setup...")
    setup_script = PROJECT_ROOT / "setup"
    result = subprocess.run(
        ["bash", "-c", f"source {setup_script}"],
        cwd=PROJECT_ROOT
    )
    if result.returncode == 0:
        print_color(GREEN, "Setup complete!")
    else:
        print_color(YELLOW, "Setup had issues. Please run manually:")
        print("  source ./setup")
    return 0
