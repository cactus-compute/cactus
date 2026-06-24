"""Model resolution, weight management, and bundle preparation."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import GREEN, RED, YELLOW, print_color


def _convert_from_source(model_id, *, bits, token, weights_dir):
    """Download from HuggingFace and run CQ conversion."""
    print_color(YELLOW, f"Converting {model_id} from HuggingFace source...")
    from ..convert.cli import main as cq_main

    cq_args = [
        "convert", "--model", model_id,
        "--out", str(weights_dir),
        "--bits", str(bits),
    ]
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    cq_main(cq_args)

    print_color(GREEN, f"Model converted and ready at {weights_dir}")
    return weights_dir


def ensure_weights(model_id, *, bits=4, platform=None, token=None, reconvert=False, output_dir=None):
    from .download import get_bundle_dir

    weights_dir = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits, platform=platform)

    if reconvert and weights_dir.exists():
        print_color(YELLOW, "Removing cached weights for reconversion...")
        shutil.rmtree(weights_dir)

    if weights_dir.exists() and (weights_dir / "config.txt").exists():
        print_color(GREEN, f"Model weights found at {weights_dir}")
        return weights_dir

    if weights_dir.exists():
        print_color(YELLOW, "Removing incomplete weights from a previous run...")
        shutil.rmtree(weights_dir)

    return _convert_from_source(model_id, bits=bits, token=token, weights_dir=weights_dir)


def _has_runnable_bundle(path):
    path = Path(path)
    return (path / "components" / "manifest.json").exists()


def resolve_bundle_dir(model_id):
    path = Path(model_id).expanduser()
    if path.is_dir() and _has_runnable_bundle(path):
        return path
    return None


def _local_build_unavailable(model_id) -> RuntimeError:
    return RuntimeError(
        f"No runnable prebuilt bundle is available for {model_id!r}. "
        "Local bundle builds are unavailable because the graph builder has been removed for rewrite."
    )


def ensure_runnable_bundle(model_id, *, bits=4, platform=None, token=None,
                           reconvert=False, prebuilt=True, output_dir=None):
    """Resolve a runnable bundle from a local path, cache, or prebuilt download.

    The old local source-to-bundle fallback was removed. Until the replacement
    compiler lands, commands that need runnable bundles can only use existing
    local bundles or published Cactus-Compute bundles.
    """
    from .download import download_bundle, get_bundle_dir

    local = resolve_bundle_dir(model_id)
    if local is not None:
        return local

    if str(model_id).startswith(("/", "./", "../", "~")) and not Path(model_id).expanduser().exists():
        raise RuntimeError(f"path not found: {model_id}")

    cached = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits, platform=platform)
    if reconvert and cached.exists():
        print_color(YELLOW, "Removing cached bundle before refresh...")
        shutil.rmtree(cached)
    elif _has_runnable_bundle(cached):
        return cached

    if not prebuilt:
        raise _local_build_unavailable(model_id)

    try:
        return download_bundle(model_id, bits=bits, platform=platform,
                               token=token, output_dir=cached)
    except (RuntimeError, OSError) as exc:
        raise _local_build_unavailable(model_id) from exc


def prepare_bundle(args, *, model_id=None, prebuilt=True,
                   output_dir=None, fail_prefix="Model setup failed"):
    """Resolve the platform from args and return a runnable bundle, with uniform
    error handling shared by every model command. Returns the bundle Path, or
    None (after printing the error) on failure."""
    from .download import resolve_platform
    try:
        return ensure_runnable_bundle(
            args.model_id if model_id is None else model_id,
            bits=getattr(args, "bits", 4),
            platform=resolve_platform(getattr(args, "platform", "auto")),
            token=getattr(args, "token", None),
            reconvert=getattr(args, "reconvert", False),
            prebuilt=prebuilt,
            output_dir=output_dir,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print_color(RED, f"{fail_prefix}: {exc}")
        return None


def ensure_bundle(model_id, *, bits=4, platform=None, token=None,
                  reconvert=False, output_dir=None):
    return ensure_runnable_bundle(
        model_id,
        bits=bits,
        platform=platform,
        token=token,
        reconvert=reconvert,
        output_dir=output_dir,
    )
