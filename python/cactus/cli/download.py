"""Download command and path helpers."""
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_model_dir_name(model_id):
    """Convert HuggingFace model ID to local directory name."""
    return model_id.split("/")[-1].lower()


def get_weights_dir(model_id):
    """Return ``<project>/weights/<model_name>``."""
    return _PROJECT_ROOT / "weights" / get_model_dir_name(model_id)


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
    except Exception as e:
        print_color(RED, f"Download failed: {e}")
        return 1
