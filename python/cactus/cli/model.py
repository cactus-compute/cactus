"""Model resolution, weight management, and bundle preparation."""
from __future__ import annotations

import os
import json
import shutil
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TranspileOptions:
    input_modalities: tuple[str, ...] | None = None
    allow_unsupported_ops: bool = False


def _has_runnable_bundle(path):
    path = Path(path)
    return (path / "components" / "manifest.json").exists() or (path / "runtime_plan.json").exists()


def resolve_bundle_dir(model_id):
    path = Path(model_id).expanduser()
    if path.is_file() and path.name == "runtime_plan.json":
        path = path.parent

    if path.is_dir() and _has_runnable_bundle(path):
        materialize_engine_manifest_from_runtime_plan(path)
        return path
    return None


def materialize_engine_manifest_from_runtime_plan(bundle_dir):
    bundle_path = Path(bundle_dir)
    runtime_plan_path = bundle_path / "runtime_plan.json"
    engine_manifest_path = bundle_path / "components" / "manifest.json"

    if not runtime_plan_path.exists():
        return engine_manifest_path if engine_manifest_path.exists() else None

    if engine_manifest_path.exists() and engine_manifest_path.stat().st_mtime >= runtime_plan_path.stat().st_mtime:
        return engine_manifest_path

    plan = json.loads(runtime_plan_path.read_text(encoding="utf-8"))
    manifest = engine_manifest_from_runtime_plan(plan)
    engine_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    engine_manifest_path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    return engine_manifest_path


def engine_manifest_from_runtime_plan(plan):
    components = plan.get("components")

    if not isinstance(components, list):
        raise ValueError("runtime_plan.json must contain a components list")

    manifest = {"components": components}
    family = plan.get("family")

    if family:
        manifest["family"] = str(family)

    metadata = plan.get("metadata")

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if value is None:
                continue
            manifest[str(key)] = str(value)

    return manifest


def ensure_runnable_bundle(model_id, *, bits=4, platform=None, token=None,
                           reconvert=False, prebuilt=True, output_dir=None,
                           transpile: TranspileOptions | None = None):
    """Resolve a runnable bundle from a local path, cache, or prebuilt download.

    Resolution order is local bundle path, cached bundle, prebuilt download,
    then local CQ conversion plus transpilation.
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
        materialize_engine_manifest_from_runtime_plan(cached)
        return cached

    if not prebuilt:
        return ensure_bundle(
            model_id,
            bits=bits,
            platform=platform,
            token=token,
            reconvert=reconvert,
            output_dir=cached,
            transpile=transpile,
        )

    try:
        return download_bundle(model_id, bits=bits, platform=platform,
                               token=token, output_dir=cached)
    except (RuntimeError, OSError) as exc:
        print_color(YELLOW, f"No prebuilt bundle found for {model_id}; building locally...")
        try:
            return ensure_bundle(
                model_id,
                bits=bits,
                platform=platform,
                token=token,
                reconvert=reconvert,
                output_dir=cached,
                transpile=transpile,
            )
        except Exception as build_exc:
            raise RuntimeError(f"Could not prepare runnable bundle for {model_id!r}") from build_exc


def prepare_bundle(args, *, model_id=None, prebuilt=True,
                   output_dir=None, fail_prefix="Model setup failed",
                   transpile: TranspileOptions | None = None):
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
            transpile=transpile,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print_color(RED, f"{fail_prefix}: {exc}")
        return None


def ensure_bundle(model_id, *, bits=4, platform=None, token=None,
                  reconvert=False, output_dir=None,
                  transpile: TranspileOptions | None = None):
    from .transpiler import build_transpiled_bundle

    weights_dir = ensure_weights(
        model_id,
        bits=bits,
        platform=platform,
        token=token,
        reconvert=reconvert,
        output_dir=output_dir,
    )

    opts = transpile or TranspileOptions()
    return build_transpiled_bundle(
        model_id,
        weights_dir=weights_dir,
        output_dir=weights_dir,
        token=token,
        input_modalities=opts.input_modalities,
        allow_unsupported_ops=opts.allow_unsupported_ops,
    )
