"""CLI bridge for the replacement Python transpiler."""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .common import GREEN, YELLOW, print_color
from .runtime import ensure_python_runtime_library


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

    clean_runtime_outputs(bundle_path)
    ir_dir.mkdir(parents=True, exist_ok=True)

    if getattr(profile, "model_profiles", "") == "parakeet":
        return build_parakeet_tdt_bundle(
            model_id,
            weights_dir=weights_path,
            output_dir=bundle_path,
            token=token,
        )

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

    return profiles.profile_for_model_id(model_id)


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


def build_parakeet_tdt_bundle(
    model_id: str,
    *,
    weights_dir: Path,
    output_dir: Path,
    token: str | None = None,
) -> Path:
    env = os.environ.copy()
    env["CACTUS_LIB_PATH"] = str(ensure_python_runtime_library())

    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token

    from cactus.transpiler.Converter import constants as converter_constants

    audio_path = converter_constants.MODALITY_INPUT_PATH["audio"]
    print_color(YELLOW, "Transpiling Parakeet TDT with the custom component exporter...")
    command = [
        sys.executable,
        "-m",
        "cactus.transpile.hf_model",
        "--model-id",
        model_id,
        "--task",
        "tdt_transcription",
        "--audio-file",
        str(audio_path),
        "--weights-dir",
        str(weights_dir),
        "--artifact-dir",
        str(output_dir),
        "--component-pipeline",
        "on",
        "--skip-execute",
        "--skip-reference-compare",
    ]

    result = subprocess.run(command, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Parakeet TDT transpilation failed with exit code {result.returncode}")

    write_runtime_plan_for_existing_manifest(output_dir, profile_for_model_id(model_id))
    print_color(GREEN, f"Runnable Parakeet TDT bundle ready at {output_dir}")
    return output_dir


def write_runtime_plan_for_existing_manifest(bundle_dir: Path, model_profile) -> None:
    from cactus.transpiler.RuntimePlan import models as RPModels

    manifest_path = bundle_dir / "components" / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = tuple(
        RPModels.runtime_component_from_engine_dict(component)
        for component in manifest.get("components", ())
        if isinstance(component, dict)
    )
    metadata = RPModels.string_dict({k: v for k, v in manifest.items() if isinstance(v, str)})
    metadata.update(RPModels.runtime_plan_metadata_from_model_profile(model_profile))
    metadata.update(RPModels.runtime_plan_metadata_from_components(components))
    plan = RPModels.RuntimePlan(
        family=manifest.get("family") or RPModels.runtime_family_from_model_profile(model_profile),
        components=components,
        routes=RPModels.runtime_routes_from_model_profile(model_profile),
        states=RPModels.runtime_states_from_model_profile(model_profile),
        aliases=RPModels.runtime_aliases_from_model_profile(model_profile),
        metadata=metadata,
    )
    plan.write(bundle_dir)
