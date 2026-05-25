"""Download command and path helpers."""
from pathlib import Path

from .common import PROJECT_ROOT, is_repo_checkout


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
    """Return the weights directory, downloading CQ weights if necessary.

    Public API — re-exported by ``cactus/__init__.py``.
    """
    from .model import ensure_weights
    return ensure_weights(model_id)


def cmd_download(args):
    """Download original model weights from HuggingFace."""
    from .model import resolve_model_id, download_model
    from .common import print_color, RED

    model_id = resolve_model_id(args.model_id)
    try:
        download_model(model_id, token=args.token, cache_dir=args.cache_dir)
        return 0
    except (RuntimeError, OSError) as e:
        print_color(RED, f"Download failed: {e}")
        return 1
