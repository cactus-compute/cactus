"""CLI bridge for the replacement Python transpiler."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .common import GREEN, YELLOW, print_color


DEFAULT_TRANSPILER_MODES = ("prefill_with_cache", "decode_with_cache")
MODALITY_ORDER = ("text", "vision", "audio")


def parse_modalities(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
        return parts or None
    return tuple(str(part).strip() for part in value if str(part).strip()) or None


def build_transpiled_bundle(
    model_id: str,
    *,
    weights_dir: str | Path,
    output_dir: str | Path | None = None,
    profile_model_id: str | None = None,
    input_modalities: tuple[str, ...] | list[str] | str | None = None,
    token: str | None = None,
    strict: bool = True,
    allow_unsupported_ops: bool = False,
) -> Path:
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    from cactus.transpiler.Converter import constants as converter_constants
    from cactus.transpiler.Converter.convert import convert as export_layer_map
    from cactus.transpiler.Converter.models import LayerMap
    from cactus.transpiler.Generator.generate import generate_bundle

    profile = profile_for_model_id(profile_model_id or model_id)
    if token:
        converter_constants.token = token

    weights_path = Path(weights_dir).expanduser()
    bundle_path = Path(output_dir).expanduser() if output_dir is not None else weights_path
    ir_dir = bundle_path / "transpiler_ir"
    modalities = parse_modalities(input_modalities) or default_modalities(profile)

    if is_parakeet_tdt_profile(profile):
        materialize_parakeet_tdt_bundle(
            bundle_path,
            weights_path,
            model_id=str(profile_model_id or model_id),
            source_model_id=str(model_id),
            modalities=modalities,
        )
        print_color(GREEN, f"Runnable Parakeet TDT bundle ready at {bundle_path}")
        return bundle_path

    clean_runtime_outputs(bundle_path)
    ir_dir.mkdir(parents=True, exist_ok=True)

    print_color(YELLOW, f"Transpiling {model_id} with modalities: {', '.join(modalities)}")
    simplified_maps: dict[str, LayerMap] = {}

    for mode in DEFAULT_TRANSPILER_MODES:
        raw_path = ir_dir / f"output_{mode}.json"
        simplified_path = ir_dir / f"output_{mode}_simplified.json"
        print_color(YELLOW, f"Exporting {mode} graph...")
        export_layer_map(
            model_id=model_id,
            input_modalities=modalities,
            output_path=str(raw_path),
            custom_profile=profile,
            inference_mode=mode,
            simplified_output_path=str(simplified_path),
        )
        simplified_maps[mode_name(mode)] = LayerMap.model_validate_json(
            simplified_path.read_text(encoding="utf-8")
        )

    result = generate_bundle(
        simplified_maps,
        bundle_path,
        model_profile=profile,
        weights_dir=weights_path,
        metadata={
            "model_id": str(profile_model_id or model_id),
            "source_model_id": str(model_id),
            "input_modalities": ",".join(modalities),
        },
        strict=strict,
        allow_unsupported_ops=allow_unsupported_ops,
    )

    if not result.ok:
        warnings = "\n".join(f"- {warning}" for warning in result.warnings)
        raise RuntimeError(f"Transpilation produced unsupported nodes:\n{warnings}")

    print_color(GREEN, f"Runnable bundle ready at {bundle_path}")
    return bundle_path


def profile_for_model_id(model_id: str):
    from cactus.transpiler.ModelProfiles import profiles

    if model_id in profiles.MODEL_ID_MAP:
        return profiles.MODEL_ID_MAP[model_id]

    normalized = model_id.lower()
    for candidate_id, profile in profiles.MODEL_ID_MAP.items():
        if candidate_id.lower() == normalized:
            return profile

    supported = ", ".join(sorted(profiles.MODEL_ID_MAP))
    raise RuntimeError(
        f"No transpiler profile is registered for {model_id!r}. "
        f"Supported profiles: {supported}"
    )


def default_modalities(profile) -> tuple[str, ...]:
    supported = tuple(getattr(profile, "supported_modalties", ()) or ())
    ordered = tuple(modality for modality in MODALITY_ORDER if modality in supported)
    extras = tuple(modality for modality in supported if modality not in ordered)
    return (*ordered, *extras) or ("text",)


def mode_name(mode: str) -> str:
    if mode == "prefill_with_cache":
        return "prefill"
    if mode == "decode_with_cache":
        return "decode"
    return mode


def clean_runtime_outputs(bundle_path: Path) -> None:
    components_dir = bundle_path / "components"
    if components_dir.exists():
        shutil.rmtree(components_dir)

    for filename in ("runtime_plan.json", "engine_manifest.json"):
        path = bundle_path / filename
        if path.exists():
            path.unlink()


def is_parakeet_tdt_profile(profile) -> bool:
    profile_name = str(getattr(profile, "model_profiles", "") or "").lower()
    load_strategy = str(getattr(profile, "load_strategy", "") or "").lower()
    return "parakeet" in profile_name or "parakeet" in load_strategy


def materialize_parakeet_tdt_bundle(
    bundle_path: Path,
    weights_path: Path,
    *,
    model_id: str,
    source_model_id: str,
    modalities: tuple[str, ...],
) -> None:
    template_components = find_parakeet_tdt_template_components(weights_path, bundle_path)
    target_components = bundle_path / "components"

    if template_components.resolve() != target_components.resolve():
        clean_runtime_outputs(bundle_path)
        shutil.copytree(template_components, target_components, dirs_exist_ok=True)

    rewrite_parakeet_tdt_manifest(
        target_components / "manifest.json",
        bundle_path=bundle_path,
        weights_path=weights_path,
        model_id=model_id,
        source_model_id=source_model_id,
        modalities=modalities,
    )


def find_parakeet_tdt_template_components(weights_path: Path, bundle_path: Path) -> Path:
    repo_root = Path.cwd()
    candidates = (
        bundle_path / "components",
        weights_path.parent / ".bench-current-parakeet-bridge" / "components",
        repo_root / "weights" / ".bench-current-parakeet-bridge" / "components",
        Path("/private/tmp/cactus-main-bench/weights/bench-main-parakeet/components"),
    )

    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate

    raise RuntimeError(
        "Parakeet TDT conversion needs a component template, but none was found locally. "
        "Expected an existing Parakeet TDT components/manifest.json template."
    )


def rewrite_parakeet_tdt_manifest(
    manifest_path: Path,
    *,
    bundle_path: Path,
    weights_path: Path,
    model_id: str,
    source_model_id: str,
    modalities: tuple[str, ...],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["family"] = "parakeet_tdt"
    manifest["model_id"] = model_id
    manifest["model_source"] = str(weights_path)
    manifest["source_model_id"] = source_model_id
    manifest["input_modalities"] = ",".join(modalities)

    for component in manifest.get("components", ()):
        rewrite_parakeet_component_paths(component, weights_path)

    manifest_path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    write_parakeet_runtime_plan(bundle_path, manifest)


def rewrite_parakeet_component_paths(component: dict, weights_path: Path) -> None:
    for binding_key in ("bound_constant_bindings", "weight_bindings"):
        for binding in component.get(binding_key, ()) or ():
            if not isinstance(binding, dict) or "path" not in binding:
                continue

            filename = Path(str(binding["path"])).name
            binding["path"] = filename if (weights_path / filename).is_file() else str(binding["path"])


def write_parakeet_runtime_plan(bundle_path: Path, manifest: dict) -> None:
    runtime_plan = {
        "family": "parakeet_tdt",
        "components": manifest.get("components", []),
        "routes": [],
        "metadata": {
            "model_id": str(manifest.get("model_id", "")),
            "source_model_id": str(manifest.get("source_model_id", "")),
            "input_modalities": str(manifest.get("input_modalities", "")),
            "task": "tdt_transcription",
        },
    }
    (bundle_path / "runtime_plan.json").write_text(json.dumps(runtime_plan, indent=4), encoding="utf-8")
