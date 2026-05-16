import shutil
import subprocess
from pathlib import Path

from .common import (
    PROJECT_ROOT,
    MODEL_REGISTRY,
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
        PROJECT_ROOT / "libs" / "curl",
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


def cmd_list(args):
    """List all supported models and their download status."""
    PIPELINE_DISPLAY = {
        "text-generation": "Text Generation",
        "image-text-to-text": "Vision",
        "automatic-speech-recognition": "Speech Recognition",
        "feature-extraction": "Embeddings",
        "voice-activity-detection": "Voice Activity Detection",
    }
    PIPELINE_ORDER = list(PIPELINE_DISPLAY.keys())
    SHOW_TAGS = {"tools", "vision", "embed", "transcription"}
    EMBED_ALIASES = {"text-embed", "image-embed", "speech-embed"}

    DIM = '\033[2m'
    BOLD = '\033[1m'

    def filter_tags(tags):
        result = set()
        for t in tags:
            if t in SHOW_TAGS:
                result.add(t)
            elif t in EMBED_ALIASES:
                result.add("embed")
        return sorted(result)

    def get_dir_size(path):
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def format_size(size_bytes):
        if size_bytes >= 1_000_000_000:
            return f"{size_bytes / 1_073_741_824:.1f} GB"
        return f"{size_bytes / 1_048_576:.0f} MB"

    # Group models by pipeline_tag preserving order
    groups = {}
    for entry in MODEL_REGISTRY:
        tag = entry["pipeline_tag"]
        groups.setdefault(tag, []).append(entry)

    # Find max model name length for alignment
    max_name = max(len(e["model"]) for e in MODEL_REGISTRY)
    max_tags_len = 20

    only_downloaded = getattr(args, 'downloaded', False)

    if only_downloaded:
        print(f"\n {BOLD}Downloaded Models{NC}")
    else:
        print(f"\n {BOLD}Supported Models{NC}")
    print(f" {'─' * 66}")

    for ptag in PIPELINE_ORDER:
        models = groups.get(ptag)
        if not models:
            continue

        section = PIPELINE_DISPLAY[ptag]
        section_printed = False

        for entry in models:
            model_id = entry["model"]
            tags = filter_tags(entry["tags"])
            tags_str = ", ".join(tags)

            weights_dir = get_weights_dir(model_id)
            config_path = weights_dir / "config.txt"
            downloaded = config_path.exists()

            if only_downloaded and not downloaded:
                continue

            if not section_printed:
                print(f"\n {BOLD}{section}{NC}")
                section_printed = True

            if downloaded:
                prefix = f" {GREEN}\u2b07{NC}  "
                # Read quantization (weight quantization level, not compute precision)
                quantization = ""
                try:
                    for line in config_path.read_text().splitlines():
                        if line.startswith("quantization="):
                            quantization = line.split("=", 1)[1].strip()
                            break
                except OSError:
                    pass
                dir_size = get_dir_size(weights_dir)
                size_str = format_size(dir_size)
                if quantization:
                    info = f"{size_str} ({quantization})"
                else:
                    info = size_str
            else:
                prefix = "    "
                info = ""

            name_pad = model_id.ljust(max_name)
            tags_pad = tags_str.ljust(max_tags_len)

            if info:
                print(f"{prefix}{name_pad}  {DIM}{tags_pad}{NC}  {info}")
            else:
                print(f"{prefix}{name_pad}  {DIM}{tags_pad}{NC}")

    print()
    return 0
