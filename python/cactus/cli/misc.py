import shutil
import subprocess
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    get_weights_dir,
    print_color,
    RED, GREEN, YELLOW, BLUE, NC,
)


def cmd_auth(args):
    """Manage Cactus Cloud API key."""
    from .config_utils import CactusConfig

    config = CactusConfig()

    if args.clear:
        config.clear_api_key()
        print_color(GREEN, "API key cleared.")
        return 0

    api_key = config.get_api_key()

    if api_key:
        masked = api_key[:4] + "..." + api_key[-4:]
        print(f"Current API key: {masked}")
    else:
        print("No API key set.")

    if args.status:
        return 0

    print()
    print("Get your cloud key at \033[1;36mhttps://www.cactuscompute.com/dashboard/api-keys\033[0m")
    new_key = input("Enter new API key (press Enter to skip): ").strip()
    if new_key:
        config.set_api_key(new_key)
        masked = new_key[:4] + "..." + new_key[-4:]
        print_color(GREEN, f"API key saved: {masked}")
    return 0


def cmd_clean(args):
    """Remove all build artifacts, caches, and downloaded weights."""
    print_color(BLUE, "Cleaning all build artifacts from Cactus project...")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    def remove_if_exists(path):
        if path.is_dir():
            print(f"Removing: {path}")
            shutil.rmtree(path)
        else:
            print(f"Not found: {path}")

    remove_if_exists(PROJECT_ROOT / "cactus" / "build")

    remove_if_exists(PROJECT_ROOT / "android" / "build")
    remove_if_exists(PROJECT_ROOT / "android" / "libs")
    remove_if_exists(PROJECT_ROOT / "android" / "arm64-v8a")

    remove_if_exists(PROJECT_ROOT / "apple" / "build")

    remove_if_exists(PROJECT_ROOT / "tests" / "build")

    remove_if_exists(PROJECT_ROOT / "venv")

    remove_if_exists(PROJECT_ROOT / "weights")

    # Clean telemetry cache
    telemetry_cache = Path.home() / "Library" / "Caches" / "cactus" / "telemetry"
    if telemetry_cache.exists():
        print(f"Removing telemetry cache: {telemetry_cache}")
        shutil.rmtree(telemetry_cache)
    else:
        print(f"Telemetry cache not found: {telemetry_cache}")

    # Re-cache API key from config so users don't need to run `cactus auth` again
    from .config_utils import CactusConfig
    config = CactusConfig()
    saved_key = config.load_config().get("api_key", "")
    if saved_key:
        config.cache_api_key(saved_key)
        masked = saved_key[:4] + "..." + saved_key[-4:]
        print(f"Restored cached API key: {masked}")

    print()
    print("Removing compiled libraries and frameworks...")

    preserve_roots = [
        PROJECT_ROOT / "cactus-engine" / "libs" / "curl",
        PROJECT_ROOT / "android" / "mbedtls",
        PROJECT_ROOT / "libs" / "mbedtls",
    ]

    def should_preserve_artifact(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            return False
        for root in preserve_roots:
            try:
                if resolved.is_relative_to(root.resolve()):
                    return True
            except FileNotFoundError:
                continue
        return False

    so_count = 0
    for so_file in PROJECT_ROOT.rglob("*.so"):
        so_file.unlink()
        so_count += 1
    print(f"Removed {so_count} .so files" if so_count else "No .so files found")

    a_count = 0
    a_preserved_count = 0
    for a_file in PROJECT_ROOT.rglob("*.a"):
        if should_preserve_artifact(a_file):
            a_preserved_count += 1
            continue
        a_file.unlink()
        a_count += 1
    if a_count or a_preserved_count:
        print(f"Removed {a_count} .a files (preserved {a_preserved_count} vendored static libs)")
    else:
        print("No .a files found")

    bin_count = 0
    for bin_file in PROJECT_ROOT.rglob("*.bin"):
        bin_file.unlink()
        bin_count += 1
    print(f"Removed {bin_count} .bin files" if bin_count else "No .bin files found")

    xcf_count = 0
    for xcf_dir in PROJECT_ROOT.rglob("*.xcframework"):
        if xcf_dir.is_dir():
            shutil.rmtree(xcf_dir)
            xcf_count += 1
    print(f"Removed {xcf_count} .xcframework directories" if xcf_count else "No .xcframework directories found")

    pycache_count = 0
    for pycache_dir in PROJECT_ROOT.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            pycache_count += 1
    print(f"Removed {pycache_count} __pycache__ directories" if pycache_count else "No __pycache__ directories found")

    egg_count = 0
    for egg_dir in PROJECT_ROOT.rglob("*.egg-info"):
        if egg_dir.is_dir():
            shutil.rmtree(egg_dir)
            egg_count += 1
    print(f"Removed {egg_count} .egg-info directories" if egg_count else "No .egg-info directories found")

    print()
    print_color(GREEN, "Clean complete!")
    print("All build artifacts have been removed.")
    print()

    # Re-run setup automatically
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
