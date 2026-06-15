"""Download pre-transpiled bundles from huggingface.co/Cactus-Compute."""
from pathlib import Path

from .common import (
    BLUE, GREEN, RED, YELLOW,
    SUPPORTED_PLATFORMS,
    print_color, weights_root,
)


def get_model_dir_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def get_weights_dir(model_id: str) -> Path:
    return weights_root() / get_model_dir_name(model_id)


def get_bundle_dir(model_id: str, *, bits: int = 4, platform: str | None = None) -> Path:
    from .utils import variant_suffix
    return weights_root() / f"{get_model_dir_name(model_id)}-{variant_suffix(bits, platform)}"


def _host_platform() -> str | None:
    """Best-fit accelerator for the current host: apple on macOS, else generic CPU."""
    import platform as _platform

    if _platform.system() == "Darwin" and "apple" in SUPPORTED_PLATFORMS:
        return "apple"
    return None


def resolve_platform(choice: str) -> str | None:
    if choice == "auto":
        return _host_platform()
    if choice == "cpu":
        return None
    if choice in SUPPORTED_PLATFORMS:
        return choice
    raise ValueError(f"unknown platform {choice!r}; supported: auto, cpu, {', '.join(SUPPORTED_PLATFORMS)}")


def ensure_model(model_id: str) -> Path:
    return download_bundle(model_id)


def download_bundle(model_id: str, *, bits: int = 4, platform: str | None = None,
                    token: str | None = None, output_dir: Path | None = None,
                    reconvert: bool = False) -> Path:
    from .utils import (
        download_cq_archive,
        list_hf_cq_archives,
        resolve_archive,
        suggested_cq_repo,
        variant_suffix,
    )

    repo_id = suggested_cq_repo(model_id)
    local_name = get_model_dir_name(model_id)
    bundle_dir = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits, platform=platform)

    label = variant_suffix(bits, platform)
    print()
    print_color(BLUE, f"Fetching {repo_id} [{label}]")

    archives = list_hf_cq_archives(repo_id, token=token)
    if not archives:
        raise RuntimeError(f"no bundles published at {repo_id}")

    resolution = resolve_archive(repo_id, local_name, archives, bits, platform=platform)
    download_cq_archive(resolution, bundle_dir, token=token, reconvert=reconvert)
    print_color(GREEN, f"Ready at {bundle_dir}")
    return bundle_dir


def cmd_download(args) -> int:
    try:
        platform = resolve_platform(args.platform)
        download_bundle(args.model_id, bits=args.bits, platform=platform, token=args.token)
        return 0
    except (RuntimeError, OSError, ValueError) as e:
        print_color(RED, f"Download failed: {e}")
        print_color(YELLOW, f"Try: cactus convert {args.model_id} --bits {args.bits} && cactus transpile {args.model_id}")
        return 1
