from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


from . import component_split
from . import constants
from . import models
from . import lowerings as lowering_engine
from ..Converter import models as CModels
from ..IR import models as IRModels
from ..RuntimePlan import models as RPModels




GraphInput = CModels.LayerMap | IRModels.Graph
ComponentInput = GraphInput | Mapping[str, GraphInput]


def generate(
    ir_output: ComponentInput,
    output_dir: str | Path | None = None,
    *,
    model_profile: Any | None = None,
    component_name: str | None = None,
    weights_dir: str | Path | None = None,
    weights_manifest_path: str | Path | None = None,
    config: models.GeneratorConfig | None = None,
    lowerings: Mapping[str, models.LoweringRule | models.LoweringFn] | None = None,
    strict: bool = True,
    allow_unsupported_ops: bool = False,
) -> models.GenerationResult:
    if config is None and output_dir is None:
        raise ValueError("generate requires output_dir unless a GeneratorConfig is provided")

    generator_config = config or models.GeneratorConfig(
        output_dir=Path(output_dir),
        weights_dir=Path(weights_dir) if weights_dir is not None else None,
        weights_manifest_path=Path(weights_manifest_path) if weights_manifest_path is not None else None,
        model_profile=model_profile,
        graph_suffix=constants.DEFAULT_GRAPH_SUFFIX,
        strict=strict,
        allow_unsupported_ops=allow_unsupported_ops,
    )
    if config is not None and model_profile is not None and getattr(generator_config, "model_profile", None) is None:
        generator_config = replace(generator_config, model_profile=model_profile)
    prepare_generation_output_dir(generator_config.output_dir)
    components = component_graphs_from_input(ir_output, generator_config, component_name, model_profile)
    lowering_rules = lowering_engine.build_lowering_rules(lowerings)
    fp16_cache_components = fp16_kv_cache_components(model_profile)

    for component in components:
        lower_component_with_cache_contract(component, generator_config, lowering_rules, fp16_cache_components)

    return models.GenerationResult.from_components(components)


def generate_from_json(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_profile: Any | None = None,
    component_name: str | None = None,
    weights_dir: str | Path | None = None,
    weights_manifest_path: str | Path | None = None,
    strict: bool = True,
    allow_unsupported_ops: bool = False,
) -> models.GenerationResult:
    layer_map = read_layer_map(input_path)
    return generate(
        layer_map,
        output_dir,
        model_profile=model_profile,
        component_name=component_name,
        weights_dir=weights_dir,
        weights_manifest_path=weights_manifest_path,
        strict=strict,
        allow_unsupported_ops=allow_unsupported_ops,
    )


def generate_bundle(
    ir_output: ComponentInput,
    bundle_dir: str | Path,
    *,
    model_profile: Any | None = None,
    component_name: str | None = None,
    weights_dir: str | Path | None = None,
    weights_manifest_path: str | Path | None = None,
    metadata: dict[str, str] | None = None,
    strict: bool = True,
    allow_unsupported_ops: bool = False,
) -> models.GenerationResult:
    bundle_path = Path(bundle_dir)
    components_dir = bundle_path / "components"
    source_weights_dir = Path(weights_dir) if weights_dir is not None else None
    materialize_runtime_bundle_files(bundle_path, source_weights_dir, model_profile)
    result = generate(
        ir_output,
        output_dir=components_dir,
        model_profile=model_profile,
        component_name=component_name,
        weights_dir=weights_dir,
        weights_manifest_path=weights_manifest_path,
        strict=strict,
        allow_unsupported_ops=allow_unsupported_ops,
    )
    plan = RPModels.runtime_plan_from_generation_result(
        result,
        bundle_dir=bundle_path,
        model_profile=model_profile,
        metadata=metadata,
    )
    engine_manifest_path, runtime_plan_path = plan.write(bundle_path)
    return replace(result, engine_manifest_path=engine_manifest_path, runtime_plan_path=runtime_plan_path)


################################################# Generator Utils!!!!!!! #################################################


def fp16_kv_cache_components(model_profile: Any | None) -> frozenset[str]:
    cache_contract = getattr(model_profile, "cache_contract", None)
    return frozenset(str(value) for value in getattr(cache_contract, "fp16_kv_cache_components", ()) or ())


def lower_component_with_cache_contract(
    component: models.ComponentGraph,
    config: models.GeneratorConfig,
    lowering_rules: dict[str, models.LoweringRule],
    fp16_cache_components: frozenset[str],
) -> None:
    previous = os.environ.get("CACTUS_KV_CACHE_FP16")

    if component.name in fp16_cache_components:
        os.environ["CACTUS_KV_CACHE_FP16"] = "1"
        component.metadata["kv_cache_precision"] = "fp16"
    else:
        os.environ.pop("CACTUS_KV_CACHE_FP16", None)

    try:
        lowering_engine.lower_component(component, config, lowering_rules)
    finally:
        if previous is None:
            os.environ.pop("CACTUS_KV_CACHE_FP16", None)
        else:
            os.environ["CACTUS_KV_CACHE_FP16"] = previous


def materialize_runtime_bundle_files(bundle_dir: Path, weights_dir: Path | None, model_profile: Any | None = None) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if weights_dir is not None and weights_dir.exists():
        for source in weights_dir.iterdir():
            if source.is_file() and source.name not in constants.GENERATED_BUNDLE_METADATA_FILES:
                materialize_bundle_file(source, bundle_dir / source.name)

    materialize_lfm2_vl_position_grid(bundle_dir, model_profile)

    for source in model_profile_metadata_files(model_profile):
        materialize_bundle_file(source, bundle_dir / source.name, overwrite=False)

    ensure_tokenizer_sidecars(bundle_dir)


def materialize_lfm2_vl_position_grid(bundle_dir: Path, model_profile: Any | None) -> None:
    profile_name = str(getattr(model_profile, "model_profiles", "") or "").lower()

    if profile_name not in {"lfm_vlm", "lfm2_vl", "lfm-vlm"} and "lfm_vlm" not in profile_name and "lfm2_vl" not in profile_name:
        return

    source = bundle_dir / "vision_position_embedding.weights"
    target = bundle_dir / "lfm2_vl_position_embedding_grid.f32"

    if target.is_file():
        return

    tensor = read_fp16_cactus_tensor(source)

    if tensor is None or tensor.ndim != 2:
        return

    rows, hidden = tensor.shape
    grid = int(math.isqrt(int(rows)))

    if grid * grid != int(rows):
        return

    target.write_bytes(tensor.astype(np.float32, copy=False).reshape(grid, grid, hidden).tobytes())


def read_fp16_cactus_tensor(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None

    with path.open("rb") as file:
        header = file.read(84)

        if len(header) != 84 or header[:4] != b"CACT":
            return None

        _, alignment, ndim, d0, d1, d2, d3, precision, data_bytes, scales_bytes, _, _, _ = struct.unpack(
            "<IIIQQQQIQQIIQ",
            header[4:84],
        )

        if precision != 1:
            return None

        dims = [int(dim) for dim in (d0, d1, d2, d3)[:ndim]]
        element_count = math.prod(dims) if dims else 1

        if data_bytes < element_count * 2:
            return None

        scales_offset = aligned_offset(84, int(alignment))
        data_offset = aligned_offset(scales_offset + int(scales_bytes), int(alignment))
        file.seek(data_offset)
        raw = file.read(element_count * 2)

    if len(raw) != element_count * 2:
        return None

    return np.frombuffer(raw, dtype=np.float16).reshape(dims)


def aligned_offset(offset: int, alignment: int) -> int:
    if alignment <= 0:
        alignment = 32

    remainder = offset % alignment
    return offset if remainder == 0 else offset + alignment - remainder


def model_profile_metadata_files(model_profile: Any | None) -> tuple[Path, ...]:
    if model_profile is None:
        return ()

    try:
        from ..Converter import constants as converter_constants
    except Exception:
        return ()

    profile_name = getattr(model_profile, "model_profiles", None)

    if not profile_name:
        return ()

    metadata_dir = converter_constants.CONVERTER_JSON_DIR / str(profile_name)
    filenames = runtime_metadata_filenames(model_profile)
    return tuple(metadata_dir / filename for filename in filenames if (metadata_dir / filename).is_file())


def runtime_metadata_filenames(model_profile: Any | None) -> tuple[str, ...]:
    profile_files = tuple(str(filename) for filename in getattr(model_profile, "files", ()) or ())
    standard_files = (
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    )
    return tuple(models.unique_strings((*standard_files, *profile_files)))


def materialize_bundle_file(source: Path, target: Path, *, overwrite: bool = True) -> None:
    if not source.is_file():
        return

    if target.exists() or target.is_symlink():
        try:
            if target.samefile(source):
                return
        except OSError:
            pass

        if not overwrite:
            return

        target.unlink()

    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def ensure_tokenizer_sidecars(bundle_dir: Path) -> None:
    ensure_tokenizer_runtime_model_type(bundle_dir)

    if (bundle_dir / "vocab.txt").is_file():
        return

    if not (bundle_dir / "tokenizer.json").is_file():
        return

    try:
        from transformers import AutoTokenizer
        from cactus.convert.cactus_adapters import tokenizer as tokenizer_utils
    except Exception:
        return

    try:
        with tempfile.TemporaryDirectory(prefix="cactus_tokenizer_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)

            for filename in tokenizer_source_filenames():
                source = bundle_dir / filename

                if source.is_file():
                    shutil.copy2(source, temp_dir / filename)

            tokenizer = AutoTokenizer.from_pretrained(str(temp_dir), local_files_only=True, trust_remote_code=True)
            original_hf_download = tokenizer_utils.hf_hub_download
            tokenizer_utils.hf_hub_download = None

            try:
                tokenizer_utils.convert_hf_tokenizer(
                    tokenizer,
                    temp_dir,
                    model_id=str(bundle_dir),
                    model_type=bundle_model_type(bundle_dir),
                )
            finally:
                tokenizer_utils.hf_hub_download = original_hf_download

            for filename in tokenizer_sidecar_filenames():
                source = temp_dir / filename

                if source.is_file():
                    shutil.copy2(source, bundle_dir / filename)

            ensure_tokenizer_runtime_model_type(bundle_dir)
    except Exception:
        return


def ensure_tokenizer_runtime_model_type(bundle_dir: Path) -> None:
    model_type = bundle_model_type(bundle_dir)

    if not model_type:
        return

    config_path = bundle_dir / "tokenizer_config.txt"

    if not config_path.is_file():
        return

    lines = config_path.read_text(encoding="utf-8").splitlines()

    if any(line.split("=", 1)[0].strip() == "model_type" for line in lines if "=" in line):
        return

    lines.append(f"model_type={model_type}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tokenizer_source_filenames() -> tuple[str, ...]:
    return (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.jinja2",
    )


def tokenizer_sidecar_filenames() -> tuple[str, ...]:
    return (
        "vocab.txt",
        "merges.txt",
        "special_tokens.json",
        "tokenizer_config.txt",
        "chat_template.jinja2",
    )


def bundle_model_type(bundle_dir: Path) -> str | None:
    for filename in ("config.json", "hf_config.json"):
        config_path = bundle_dir / filename

        if not config_path.is_file():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        model_type = config.get("model_type")

        if model_type is not None:
            return str(model_type)

    return None


def prepare_generation_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    constants_dir = output_dir / "constants"

    if constants_dir.exists():
        shutil.rmtree(constants_dir)


def read_layer_map(input_path: str | Path) -> CModels.LayerMap:
    path = Path(input_path)
    return CModels.LayerMap.model_validate_json(path.read_text(encoding="utf-8"))


def component_graphs_from_input(
    ir_output: ComponentInput,
    config: models.GeneratorConfig,
    component_name: str | None,
    model_profile: Any | None,
) -> tuple[models.ComponentGraph, ...]:
    if isinstance(ir_output, Mapping):
        graphs = {name: graph_from_input(component_input) for name, component_input in ir_output.items()}
        split_graphs = component_split.split_component_graphs(graphs, model_profile)

        if split_graphs is not None:
            return tuple(
                models.ComponentGraph.from_ir(name, graph, config)
                for name, graph in split_graphs.items()
            )

        return tuple(
            models.ComponentGraph.from_ir(name, graph, config)
            for name, graph in graphs.items()
        )

    graph = graph_from_input(ir_output)
    name = component_name or default_component_name(graph, model_profile)
    return (models.ComponentGraph.from_ir(name, graph, config),)


def graph_from_input(ir_output: GraphInput) -> IRModels.Graph:
    if isinstance(ir_output, IRModels.Graph):
        return ir_output

    if isinstance(ir_output, CModels.LayerMap):
        return IRModels.Graph.from_map(ir_output)

    raise TypeError(f"Unsupported generator input: {type(ir_output).__name__}")


def default_component_name(graph: IRModels.Graph, model_profile: Any | None) -> str:
    profile_name = getattr(model_profile, "model_profiles", None)
    model_name = profile_name or graph.model_name or constants.DEFAULT_COMPONENT_NAME
    task = graph.task

    if task:
        return f"{model_name}_{task}"

    return model_name
