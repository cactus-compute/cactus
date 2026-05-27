"""Download command and weights path helpers."""
from pathlib import Path

from .common import (
    BLUE, GREEN, RED, YELLOW,
    PROJECT_ROOT, is_repo_checkout, print_color,
)


def get_model_dir_name(model_id):
    """Convert HuggingFace model ID to local directory name."""
    return model_id.split("/")[-1].lower()


def _weights_root():
    if is_repo_checkout():
        return PROJECT_ROOT / "weights"
    return Path.home() / ".cache" / "cactus" / "weights"


def get_weights_dir(model_id):
    """Return the weights directory for a model."""
    return _weights_root() / get_model_dir_name(model_id)


def ensure_model(model_id):
    """Return the weights directory, downloading or converting as needed.

    Public API — re-exported by ``cactus/__init__.py``.
    """
    from .model import ensure_weights
    return ensure_weights(model_id)


def download_cq_weights(model_id, *, bits=4, token=None, output_dir=None):
    """Download pre-converted CQ weights from huggingface.co/Cactus-Compute.

    Raises ``RuntimeError`` if no archive is published for the requested model
    or bit-width.
    """
    from .utils import (
        download_cq_archive,
        list_hf_cq_archives,
        resolve_archive,
        suggested_cq_repo,
    )

    weights_dir = Path(output_dir) if output_dir else get_weights_dir(model_id)
    cq_repo_id = suggested_cq_repo(model_id)
    local_name = get_model_dir_name(model_id)

    print()
    print_color(BLUE, f"Fetching {cq_repo_id} [cq{bits}]")

    archives = list_hf_cq_archives(cq_repo_id, token=token)
    if not archives:
        raise RuntimeError(f"no CQ archives published at {cq_repo_id}")

    resolution = resolve_archive(cq_repo_id, local_name, archives, bits)
    download_cq_archive(resolution, weights_dir, token=token)
    print_color(GREEN, f"Ready at {weights_dir}")
    return weights_dir


def cmd_download(args):
    """Download pre-converted CQ weights from huggingface.co/Cactus-Compute."""
    try:
        download_cq_weights(args.model_id, bits=args.bits, token=args.token)
        return 0
    except (RuntimeError, OSError) as e:
        print_color(RED, f"Download failed: {e}")
        print_color(YELLOW, f"Try: cactus convert {args.model_id} --bits {args.bits}")
        return 1
