from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path


TOKENIZER_COMPAT_FILES = (
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "tokenizer.json",
    "special_tokens.json",
    "tokenizer_config.txt",
    "chat_template.jinja2",
)
CACTUS_WEIGHT_MAGIC = 0x54434143


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return loaded


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tokenizer_json_is_compatible(main_path: Path, assistant_path: Path) -> bool:
    main_tokenizer = _read_json(main_path)
    assistant_tokenizer = _read_json(assistant_path)
    main_added = main_tokenizer.pop("added_tokens", [])
    assistant_added = assistant_tokenizer.pop("added_tokens", [])
    if main_tokenizer != assistant_tokenizer:
        return False
    if not isinstance(main_added, list) or not isinstance(assistant_added, list):
        return main_added == assistant_added
    main_added_tokens = {
        json.dumps(token, sort_keys=True)
        for token in main_added
        if isinstance(token, dict)
    }
    return all(
        isinstance(token, dict) and json.dumps(token, sort_keys=True) in main_added_tokens
        for token in assistant_added
    )


def _json_entries_are_subset(main_values: object, assistant_values: object) -> bool:
    if not isinstance(main_values, list) or not isinstance(assistant_values, list):
        return main_values == assistant_values
    main_entries = {
        json.dumps(value, sort_keys=True)
        for value in main_values
        if isinstance(value, dict)
    }
    return all(
        isinstance(value, dict) and json.dumps(value, sort_keys=True) in main_entries
        for value in assistant_values
    )


def _json_mapping_is_subset(main_values: object, assistant_values: object) -> bool:
    if not isinstance(main_values, dict) or not isinstance(assistant_values, dict):
        return main_values == assistant_values
    return all(key in main_values and main_values[key] == value for key, value in assistant_values.items())


def _special_tokens_json_is_compatible(main_path: Path, assistant_path: Path) -> bool:
    main_tokens = _read_json(main_path)
    assistant_tokens = _read_json(assistant_path)
    if not _json_entries_are_subset(
        main_tokens.pop("additional_special_tokens", []),
        assistant_tokens.pop("additional_special_tokens", []),
    ):
        return False
    if not _json_mapping_is_subset(
        main_tokens.pop("special_tokens", {}),
        assistant_tokens.pop("special_tokens", {}),
    ):
        return False
    main_tokens.pop("chat_template", None)
    assistant_chat_template = assistant_tokens.pop("chat_template", None)
    if assistant_chat_template is not None:
        return False
    return main_tokens == assistant_tokens


def _tokenizer_config_txt_is_compatible(main_path: Path, assistant_path: Path) -> bool:
    def _parse(path: Path) -> dict[str, str]:
        entries: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            entries[key] = value
        return entries

    main_config = _parse(main_path)
    assistant_config = _parse(assistant_path)
    main_has_chat_template = main_config.pop("has_chat_template", None)
    assistant_has_chat_template = assistant_config.pop("has_chat_template", None)
    if main_config != assistant_config:
        return False
    if assistant_has_chat_template == "true" and main_has_chat_template != "true":
        return False
    return True


def _component_by_name(manifest: dict[str, object], component_name: str) -> dict[str, object]:
    components = manifest.get("components")
    if not isinstance(components, list):
        raise RuntimeError("component manifest is missing components array")
    for component in components:
        if isinstance(component, dict) and component.get("component") == component_name:
            return component
    raise RuntimeError(f"component manifest is missing {component_name} component")


def _component_by_first_name(manifest: dict[str, object], component_names: tuple[str, ...]) -> dict[str, object]:
    for component_name in component_names:
        try:
            return _component_by_name(manifest, component_name)
        except RuntimeError:
            pass
    raise RuntimeError(f"component manifest is missing one of: {', '.join(component_names)}")


def _manifest_path(bundle_dir: Path) -> Path:
    manifest_path = bundle_dir / "components" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"transpiled component manifest not found: {manifest_path}")
    return manifest_path


def _main_prompt_token_ids(bundle_dir: Path) -> list[int]:
    manifest = _read_json(_manifest_path(bundle_dir))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        return []
    prompt_input_ids = inputs.get("prompt_input_ids")
    if (
        isinstance(prompt_input_ids, list)
        and len(prompt_input_ids) == 1
        and isinstance(prompt_input_ids[0], list)
    ):
        return [int(value) for value in prompt_input_ids[0]]
    return []


def _validate_assistant_tokenizer_compatibility(main_dir: Path, assistant_dir: Path) -> None:
    for filename in TOKENIZER_COMPAT_FILES:
        main_path = main_dir / filename
        assistant_path = assistant_dir / filename
        if assistant_path.exists() and not main_path.exists():
            raise RuntimeError(
                f"assistant tokenizer is incompatible: {filename} exists only in assistant model"
            )
        if main_path.exists() and assistant_path.exists():
            if filename == "tokenizer.json":
                compatible = _tokenizer_json_is_compatible(main_path, assistant_path)
            elif filename == "special_tokens.json":
                compatible = _special_tokens_json_is_compatible(main_path, assistant_path)
            elif filename == "tokenizer_config.txt":
                compatible = _tokenizer_config_txt_is_compatible(main_path, assistant_path)
            else:
                compatible = main_path.read_bytes() == assistant_path.read_bytes()
            if not compatible:
                raise RuntimeError(f"assistant tokenizer is incompatible: {filename} differs")


def _resolve_assistant_path(path_value: str, assistant_bundle: Path, assistant_weights: Path) -> Path | None:
    path = Path(path_value)
    if path.is_absolute():
        return path if path.exists() else None
    for root in (assistant_bundle, assistant_weights):
        candidate = root / path
        if candidate.exists():
            return candidate
    return None


def _path_relative_to_main(path_value: object, *, assistant_bundle: Path, main_bundle: Path) -> object:
    if not isinstance(path_value, str) or not path_value:
        return path_value
    source = assistant_bundle / path_value
    try:
        return str(source.relative_to(main_bundle))
    except ValueError:
        return path_value


def _relocate_binding_path(
    path_value: str,
    *,
    assistant_bundle: Path,
    assistant_weights: Path,
    main_bundle: Path,
) -> str:
    source = _resolve_assistant_path(path_value, assistant_bundle, assistant_weights)
    if source is None:
        raise RuntimeError(f"assistant binding path does not exist: {path_value}")
    try:
        return str(source.relative_to(main_bundle))
    except ValueError:
        raise RuntimeError(f"assistant binding is outside the packaged model directory: {source}")


def _read_cactus_weight_shape(path: Path | None) -> tuple[int, ...] | None:
    if path is None or not path.exists():
        return None
    header = path.read_bytes()[:48]
    if len(header) < 48 or int.from_bytes(header[0:4], "little") != CACTUS_WEIGHT_MAGIC:
        return None
    ndim = int.from_bytes(header[12:16], "little")
    if ndim > 4:
        return None
    dims = []
    for index in range(ndim):
        offset = 16 + index * 8
        dims.append(int.from_bytes(header[offset:offset + 8], "little"))
    return tuple(dims)


def _main_embedding_paths_by_source(manifest: dict[str, object], main_bundle_dir: Path) -> dict[str, tuple[str, tuple[int, ...] | None]]:
    paths: dict[str, tuple[str, tuple[int, ...] | None]] = {}
    components = manifest.get("components")
    if not isinstance(components, list):
        return paths
    for component in components:
        if not isinstance(component, dict):
            continue
        bindings = component.get("bound_constant_bindings")
        if not isinstance(bindings, list):
            continue
        for binding in bindings:
            if not isinstance(binding, dict) or binding.get("kind") != "embedding":
                continue
            path = binding.get("path")
            if not isinstance(path, str) or not path:
                continue
            full_path = main_bundle_dir / path
            if not full_path.exists():
                continue
            entry = (path, _read_cactus_weight_shape(full_path))
            for key in ("source_name", "value_id"):
                value = binding.get(key)
                if isinstance(value, str) and value:
                    paths[value] = entry
    return paths


def _rewrite_assistant_component_paths(
    *,
    component: dict[str, object],
    assistant_bundle_dir: Path,
    assistant_weights_dir: Path,
    main_bundle_dir: Path,
    main_embedding_paths: dict[str, tuple[str, tuple[int, ...] | None]],
) -> dict[str, object]:
    assistant_component = dict(component)
    source_directory_value = assistant_component.get("directory")
    if not isinstance(source_directory_value, str) or not source_directory_value:
        raise RuntimeError("assistant decoder component is missing directory")
    source_component_dir = assistant_bundle_dir / source_directory_value
    if not source_component_dir.exists():
        raise RuntimeError(f"assistant decoder component directory not found: {source_component_dir}")

    assistant_component["component"] = "assistant"
    assistant_component["directory"] = str(source_component_dir.relative_to(main_bundle_dir))

    for field in ("raw_ir", "optimized_ir", "graph"):
        assistant_component[field] = _path_relative_to_main(
            assistant_component.get(field),
            assistant_bundle=assistant_bundle_dir,
            main_bundle=main_bundle_dir,
        )

    bindings = assistant_component.get("bound_constant_bindings")
    if isinstance(bindings, list):
        rewritten_bindings = []
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            rewritten = dict(binding)
            path_value = rewritten.get("path")
            shared_embedding_path = None
            if rewritten.get("kind") == "embedding":
                assistant_shape = (
                    _read_cactus_weight_shape(_resolve_assistant_path(path_value, assistant_bundle_dir, assistant_weights_dir))
                    if isinstance(path_value, str)
                    else None
                )
                for key in ("source_name", "value_id"):
                    source_value = rewritten.get(key)
                    if isinstance(source_value, str):
                        shared_embedding = main_embedding_paths.get(source_value)
                        if not shared_embedding:
                            continue
                        candidate_path, candidate_shape = shared_embedding
                        if assistant_shape is None or candidate_shape is None or assistant_shape == candidate_shape:
                            shared_embedding_path = candidate_path
                            break
            if shared_embedding_path:
                rewritten["path"] = shared_embedding_path
            elif isinstance(path_value, str):
                rewritten["path"] = _relocate_binding_path(
                    path_value,
                    assistant_bundle=assistant_bundle_dir,
                    assistant_weights=assistant_weights_dir,
                    main_bundle=main_bundle_dir,
                )
            rewritten_bindings.append(rewritten)
        assistant_component["bound_constant_bindings"] = rewritten_bindings

    return assistant_component


def _spec_decode_manifest(assistant_model_id: str) -> dict[str, object]:
    return {
        "version": 1,
        "method": "single_position_mtp",
        "assistant_model_id": assistant_model_id,
        "target": {
            "verifier_logits": "verifier_logits",
            "target_hidden_state": "target_hidden_state",
            "target_token_embedding": "target_token_embedding",
            "shared_kv": {
                "full_attention": {
                    "key": "shared_kv.full_attention.key",
                    "value": "shared_kv.full_attention.value",
                },
                "sliding_attention": {
                    "key": "shared_kv.sliding_attention.key",
                    "value": "shared_kv.sliding_attention.value",
                },
            },
        },
        "assistant": {
            "current_token_embedding": "current_token_embedding",
            "previous_target_hidden": "previous_target_hidden",
            "position": "position",
            "shared_kv": {
                "full_attention": {
                    "key": "shared_kv.full_attention.key",
                    "value": "shared_kv.full_attention.value",
                },
                "sliding_attention": {
                    "key": "shared_kv.sliding_attention.key",
                    "value": "shared_kv.sliding_attention.value",
                },
            },
            "logits_output": "logits_output",
            "next_hidden_output": "next_hidden_output",
        },
    }


def merge_assistant_bundle(
    *,
    main_bundle_dir: str | Path,
    assistant_bundle_dir: str | Path,
    assistant_weights_dir: str | Path,
    assistant_model_id: str,
) -> Path:
    main_bundle = Path(main_bundle_dir).resolve()
    assistant_bundle = Path(assistant_bundle_dir).resolve()
    assistant_weights = Path(assistant_weights_dir).resolve()
    main_manifest_path = _manifest_path(main_bundle)
    assistant_manifest_path = _manifest_path(assistant_bundle)
    main_manifest = _read_json(main_manifest_path)
    assistant_manifest = _read_json(assistant_manifest_path)

    if main_manifest.get("task") != "causal_lm_logits":
        raise RuntimeError("assistant MTP packaging currently requires a causal_lm_logits main bundle")
    if assistant_manifest.get("task") != "causal_lm_logits":
        raise RuntimeError("assistant MTP packaging currently requires a causal_lm_logits assistant bundle")

    _component_by_name(main_manifest, "decoder")
    _component_by_name(main_manifest, "target_embedding")
    assistant_decoder = _component_by_first_name(assistant_manifest, ("assistant", "decoder"))
    _validate_assistant_tokenizer_compatibility(main_bundle, assistant_weights)

    assistant_component = _rewrite_assistant_component_paths(
        component=assistant_decoder,
        assistant_bundle_dir=assistant_bundle,
        assistant_weights_dir=assistant_weights,
        main_bundle_dir=main_bundle,
        main_embedding_paths=_main_embedding_paths_by_source(main_manifest, main_bundle),
    )
    assistant_component["source_model_id"] = assistant_model_id

    manifest_components = main_manifest.get("components")
    if not isinstance(manifest_components, list):
        raise RuntimeError("main component manifest is missing components array")
    components = [
        component
        for component in manifest_components
        if isinstance(component, dict) and component.get("component") != "assistant"
    ]
    insert_at = len(components)
    for index, component in enumerate(components):
        if component.get("component") == "target_embedding":
            insert_at = index + 1
            break
    components.insert(insert_at, assistant_component)
    main_manifest["components"] = components

    raw_component_order = main_manifest.get("component_order")
    if not isinstance(raw_component_order, list):
        raw_component_order = []
    component_order = [
        component
        for component in raw_component_order
        if isinstance(component, str) and component != "assistant"
    ]
    try:
        target_embedding_index = component_order.index("target_embedding")
        component_order.insert(target_embedding_index + 1, "assistant")
    except ValueError:
        component_order.append("assistant")
    main_manifest["component_order"] = component_order
    main_manifest["spec_decode"] = _spec_decode_manifest(assistant_model_id)

    _write_json(main_manifest_path, main_manifest)
    assistant_manifest_path.unlink()
    return main_manifest_path


def _run_assistant_conversion(
    *,
    cq_main: Callable[[list[str]], object],
    assistant_model_id: str,
    assistant_weights_dir: Path,
    bits: int,
    device: str,
    token: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> None:
    cq_args = [
        "convert",
        "--model",
        assistant_model_id,
        "--out",
        str(assistant_weights_dir),
        "--bits",
        str(bits),
        "--device",
        str(device),
    ]
    if token:
        cq_args.extend(["--token", token])
    if cache_dir:
        cq_args.extend(["--cache-dir", cache_dir])
    if local_files_only:
        cq_args.append("--local-files-only")
    cq_args.append("--force")
    cq_main(cq_args)


def package_assistant_for_convert(
    *,
    cq_main: Callable[[list[str]], object],
    cmd_transpile: Callable[[argparse.Namespace], int],
    main_output_dir: str | Path,
    assistant_model_id: str,
    assistant_bits: int,
    device: str,
    prompt: str,
    max_new_tokens: int,
    torch_dtype: str,
    token: str | None,
    cache_dir: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Path:
    main_dir = Path(main_output_dir).resolve()
    assistant_dir = main_dir / "components" / "assistant"
    weights_dir = assistant_dir / "weights"
    if assistant_dir.exists():
        shutil.rmtree(assistant_dir)
    try:
        _run_assistant_conversion(
            cq_main=cq_main,
            assistant_model_id=assistant_model_id,
            assistant_weights_dir=weights_dir,
            bits=int(assistant_bits),
            device=device,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        prompt_text = prompt or "Hello"
        main_prompt_ids = _main_prompt_token_ids(main_dir)
        extra_args = [
            "--weights-dir",
            str(weights_dir),
            "--artifact-dir",
            str(assistant_dir),
            "--task",
            "causal_lm_logits",
            "--max-new-tokens",
            str(max_new_tokens),
            "--torch-dtype",
            str(torch_dtype),
            "--component-pipeline",
            "on",
            "--components",
            "assistant",
            "--tokenizer-source",
            str(main_dir),
        ]
        if main_prompt_ids:
            extra_args.extend(["--input-ids", ",".join(str(token_id) for token_id in main_prompt_ids)])
        else:
            extra_args.extend(["--prompt", prompt_text])
        if token:
            extra_args.extend(["--token", token])
        if cache_dir:
            extra_args.extend(["--cache-dir", cache_dir])
        if trust_remote_code:
            extra_args.append("--trust-remote-code")
        if local_files_only:
            extra_args.append("--local-files-only")

        rc = cmd_transpile(
            argparse.Namespace(
                model_id=assistant_model_id,
                execute_after_transpile=False,
                allow_unconverted_weights=False,
                extra_args=extra_args,
            )
        )
        if rc != 0:
            raise RuntimeError(f"assistant transpile failed with exit code {rc}")
    except Exception:
        if assistant_dir.exists():
            shutil.rmtree(assistant_dir)
        raise

    return merge_assistant_bundle(
        main_bundle_dir=main_dir,
        assistant_bundle_dir=assistant_dir,
        assistant_weights_dir=weights_dir,
        assistant_model_id=assistant_model_id,
    )
