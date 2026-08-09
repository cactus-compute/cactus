"""Model resolution, weight management, and bundle preparation."""
from __future__ import annotations

import os
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .common import GREEN, PROJECT_ROOT, RED, YELLOW, print_color


def _default_multimodal_assets():
    """Return bundled representative media used for multimodal graph capture."""
    candidates = (
        Path(__file__).resolve().parent.parent / "assets",
        PROJECT_ROOT / "cactus-engine" / "tests" / "assets",
    )

    def find(name):
        return next((directory / name for directory in candidates if (directory / name).exists()), None)

    image = find("test_monkey.png")
    audio = find("test.wav")
    return ([str(image)] if image else []), (str(audio) if audio else None)


def package_handoff_probe(output_dir, model_id):
    """Bundle the cloud-handoff probe for models that provide one."""
    try:
        from cactus.convert.handoff_probe import export_handoff_probe

        if export_handoff_probe(output_dir, model_id):
            print_color(GREEN, f"Cloud handoff probe packaged into {output_dir}")
    except Exception as exc:
        print_color(YELLOW, f"Warning: failed to package cloud handoff probe: {exc}")


def _convert_from_source(model_id, *, bits, token, weights_dir, skip_model_load=False):
    """Download from HuggingFace and run CQ conversion."""
    if bits not in (1, 2, 3, 4):
        raise SystemExit(
            f"CQ{bits} is a mixed-precision variant, available only as a prebuilt "
            f"download; local conversion supports uniform bits 1-4"
        )
    from .common import convert_toolchain_error
    err = convert_toolchain_error()
    if err:
        raise RuntimeError(err)
    from .transpiler import profile_for_model_id

    component_sources = tuple(getattr(profile_for_model_id(model_id), "component_sources", ()) or ())
    if component_sources:
        return _convert_component_sources(
            model_id, component_sources, bits=bits, token=token, weights_dir=weights_dir
        )
    print_color(YELLOW, f"Converting {model_id} from HuggingFace source...")
    from ..convert.cli import main as cq_main

    cq_args = [
        "convert", "--model", model_id,
        "--out", str(weights_dir),
        "--bits", str(bits),
    ]
    if skip_model_load:
        cq_args.append("--skip-model-load")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    cq_main(cq_args)

    print_color(GREEN, f"Model converted and ready at {weights_dir}")
    return weights_dir


def _convert_component_sources(model_id, component_sources, *, bits, token, weights_dir):
    """Convert a multi-repo pipeline (diffusion) component by component into one
    weights dir, merging each run's weights manifest."""
    import json
    import tempfile

    from huggingface_hub import snapshot_download

    from ..convert.cli import main as cq_main

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    merged_records = []
    converted = []
    for mode, spec in component_sources:
        _kind, _, source = spec.partition(":")
        if "/" in source:
            component_src = source
        else:
            snapshot = snapshot_download(model_id, allow_patterns=[f"{source}/*"], token=token)
            component_src = str(Path(snapshot) / source)
        print_color(YELLOW, f"Converting {mode} component from {component_src}...")
        with tempfile.TemporaryDirectory() as tmp:
            component_out = Path(tmp) / "out"
            cq_main(["convert", "--model", component_src, "--out", str(component_out), "--bits", str(bits)])
            manifest_path = component_out / "weights_manifest.json"
            if manifest_path.exists():
                merged_records.extend(json.loads(manifest_path.read_text(encoding="utf-8")).get("weights", []))
            for tensor_file in sorted(component_out.iterdir()):
                if tensor_file.suffix not in {".weights", ".bias"}:
                    continue
                destination = weights_dir / tensor_file.name
                if destination.exists():
                    raise RuntimeError(
                        f"components {converted} and {mode!r} both produce tensor file {tensor_file.name!r}"
                    )
                shutil.move(str(tensor_file), destination)
        converted.append(mode)
    (weights_dir / "weights_manifest.json").write_text(
        json.dumps({"weights": merged_records}, indent=2), encoding="utf-8"
    )
    (weights_dir / "config.txt").write_text(
        "\n".join(f"component={mode}" for mode in converted) + "\n", encoding="utf-8"
    )
    print_color(GREEN, f"Model converted and ready at {weights_dir}")
    return weights_dir


def ensure_weights(model_id, *, bits=4, token=None, reconvert=False, output_dir=None, skip_model_load=False):
    from .download import get_bundle_dir

    weights_dir = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits)

    if reconvert and weights_dir.exists():
        print_color(YELLOW, "Removing cached weights for reconversion...")
        shutil.rmtree(weights_dir)

    if weights_dir.exists() and (weights_dir / "config.txt").exists():
        print_color(GREEN, f"Model weights found at {weights_dir}")
        return weights_dir

    if weights_dir.exists():
        print_color(YELLOW, "Removing incomplete weights from a previous run...")
        shutil.rmtree(weights_dir)

    return _convert_from_source(
        model_id,
        bits=bits,
        token=token,
        weights_dir=weights_dir,
        skip_model_load=skip_model_load,
    )


@dataclass(frozen=True)
class TranspileOptions:
    input_modalities: tuple[str, ...] | None = None
    generic_task: str | None = None
    cache_style: str | None = None
    fusion_groups: tuple[str, ...] | None = None
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


def ensure_runnable_bundle(model_id, *, bits=4, token=None,
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

    cached = Path(output_dir) if output_dir else get_bundle_dir(model_id, bits=bits)
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
            token=token,
            reconvert=reconvert,
            output_dir=cached,
            transpile=transpile,
        )

    try:
        return download_bundle(model_id, bits=bits, token=token, output_dir=cached)
    except (RuntimeError, OSError) as exc:
        print_color(YELLOW, f"No prebuilt bundle found for {model_id}; building locally...")
        try:
            return ensure_bundle(
                model_id,
                bits=bits,
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
    """Resolve and return a runnable bundle, with uniform
    error handling shared by every model command. Returns the bundle Path, or
    None (after printing the error) on failure."""
    try:
        return ensure_runnable_bundle(
            args.model_id if model_id is None else model_id,
            bits=getattr(args, "bits", 4),
            token=getattr(args, "token", None),
            reconvert=getattr(args, "reconvert", False),
            prebuilt=prebuilt,
            output_dir=output_dir,
            transpile=transpile,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print_color(RED, f"{fail_prefix}: {exc}")
        return None


def ensure_bundle(model_id, *, bits=4, token=None,
                  reconvert=False, output_dir=None,
                  transpile: TranspileOptions | None = None):
    from .transpiler import build_transpiled_bundle

    weights_dir = ensure_weights(
        model_id,
        bits=bits,
        token=token,
        reconvert=reconvert,
        output_dir=output_dir,
    )

    # Embedding models currently use main's mature graph-capture adapter.  The
    # replacement transpiler intentionally handles generation/transcription
    # contracts; treating an embedding encoder as a generic causal LM is wrong.
    from cactus.transpile.component_plan import infer_component_plan_from_output
    plan = infer_component_plan_from_output(str(weights_dir), model_id=model_id)
    if plan is not None and plan.task == "text_embedding":
        from .transpile import run_transpile

        rc = run_transpile(model_id, extra_args=[
            "--weights-dir", str(weights_dir),
            "--artifact-dir", str(weights_dir),
        ])
        if rc != 0:
            raise RuntimeError(f"Build failed for {model_id}")
        package_handoff_probe(weights_dir, model_id)
        return weights_dir

    opts = transpile or TranspileOptions()
    return build_transpiled_bundle(
        model_id,
        weights_dir=weights_dir,
        output_dir=weights_dir,
        token=token,
        input_modalities=opts.input_modalities,
        generic_task=opts.generic_task,
        cache_style=opts.cache_style,
        fusion_groups=opts.fusion_groups,
        allow_unsupported_ops=opts.allow_unsupported_ops,
    )
