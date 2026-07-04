"""Download prebuilt bundles from huggingface.co/Cactus-Compute."""
from pathlib import Path

from .common import (
    BLUE, GREEN,
    SUPPORTED_WEIGHTS_VARIANTS,
    print_color, weights_root,
)


def get_model_dir_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def get_weights_dir(model_id: str) -> Path:
    return weights_root() / get_model_dir_name(model_id)


def get_bundle_dir(model_id: str, *, bits: int = 4, platform: str | None = None) -> Path:
    from .utils import variant_suffix
    return weights_root() / f"{get_model_dir_name(model_id)}-{variant_suffix(bits, platform)}"


def resolve_weights_variant(choice: str | None) -> str | None:
    """Map the --weights choice to the bundle variant suffix: general (the
    default on every platform) selects the portable bundle (no suffix)."""
    if choice in (None, "general"):
        return None
    if choice in SUPPORTED_WEIGHTS_VARIANTS:
        return choice
    raise ValueError(f"unknown weights variant {choice!r}; "
                     f"supported: general, {', '.join(SUPPORTED_WEIGHTS_VARIANTS)}")


def ensure_model(model_id: str) -> Path:
    return download_bundle(model_id)


def download_bundle(model_id: str, *, bits: int = 4, platform: str | None = None,
                    token: str | None = None, output_dir: Path | None = None) -> Path:
    from .utils import (
        download_cq_archive,
        list_hf_cq_archives,
        resolve_archive,
        resolve_weight_revision,
        suggested_cq_repo,
        variant_suffix,
    )

    repo_id = suggested_cq_repo(model_id)
    local_name = get_model_dir_name(model_id)
    bundle_dir = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits, platform=platform)

    revision = resolve_weight_revision(repo_id, token=token)
    label = variant_suffix(bits, platform)
    print()
    print_color(BLUE, f"Fetching {repo_id} [{label}] @ {revision or 'main'}")

    archives = list_hf_cq_archives(repo_id, token=token, revision=revision)
    if not archives:
        raise RuntimeError(f"no bundles published at {repo_id}")

    resolution = resolve_archive(repo_id, local_name, archives, bits, platform=platform)
    download_cq_archive(resolution, bundle_dir, token=token, revision=revision)
    print_color(GREEN, f"Ready at {bundle_dir}")
    return bundle_dir


def cmd_download(args) -> int:
    from .model import prepare_bundle

    bundle = prepare_bundle(args, fail_prefix=f"Failed to prepare {args.model_id}")
    return 1 if bundle is None else 0
