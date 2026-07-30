from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True, frozen=True)
class ArrayComparison:
    name: str
    shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    max_abs_error: float
    mean_abs_error: float
    topk_match: bool | None = None
    passed: bool = False


@dataclass(slots=True, frozen=True)
class TextReference:
    prompt: str
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    text: str
    logits: np.ndarray | None = None


def run_cactus_component(
    bundle_dir: str | Path,
    component_name: str,
    inputs: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    bundle = Path(bundle_dir)
    component = load_component_manifest(bundle, component_name)
    graph = load_component_graph(bundle, component)
    bind_component_weights(graph, bundle, component)
    set_component_inputs(graph, component, inputs)
    graph.execute()
    return read_component_outputs(graph, component)


def compare_component_to_reference(
    bundle_dir: str | Path,
    component_name: str,
    inputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    topk: int = 5,
) -> dict[str, ArrayComparison]:
    cactus_outputs = run_cactus_component(bundle_dir, component_name, inputs)
    return {
        name: compare_arrays(
            name,
            cactus_outputs[name],
            reference,
            atol=atol,
            rtol=rtol,
            topk=topk,
        )
        for name, reference in reference_outputs.items()
        if name in cactus_outputs
    }


def compare_arrays(
    name: str,
    actual: Any,
    reference: Any,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    topk: int = 5,
) -> ArrayComparison:
    actual_array = np.asarray(actual)
    reference_array = np.asarray(reference)

    if actual_array.shape != reference_array.shape:
        return ArrayComparison(
            name=name,
            shape=tuple(int(dim) for dim in actual_array.shape),
            reference_shape=tuple(int(dim) for dim in reference_array.shape),
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            topk_match=None,
            passed=False,
        )

    diff = np.abs(actual_array.astype(np.float32) - reference_array.astype(np.float32))
    max_abs_error = float(diff.max()) if diff.size else 0.0
    mean_abs_error = float(diff.mean()) if diff.size else 0.0
    topk_match = compare_topk(actual_array, reference_array, topk)
    passed = bool(np.allclose(actual_array, reference_array, atol=atol, rtol=rtol))

    return ArrayComparison(
        name=name,
        shape=tuple(int(dim) for dim in actual_array.shape),
        reference_shape=tuple(int(dim) for dim in reference_array.shape),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        topk_match=topk_match,
        passed=passed,
    )


def run_cactus_text_smoke(
    bundle_dir: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 1,
    python_executable: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="cactus_golden_", suffix=".json", delete=False) as result_file:
        result_path = Path(result_file.name)

    command = [
        str(python_executable or sys.executable),
        "-m",
        "cactus",
        "run",
        str(bundle_dir),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--no-cloud-handoff",
        "--result-json",
        str(result_path),
    ]
    run_env = dict(os.environ)
    if env is not None:
        run_env.update({str(key): str(value) for key, value in env.items()})

    try:
        completed = subprocess.run(command, env=run_env, check=False, capture_output=True, text=True)
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        payload["returncode"] = completed.returncode
        payload["stdout"] = completed.stdout
        payload["stderr"] = completed.stderr
        return payload
    finally:
        result_path.unlink(missing_ok=True)


def run_hf_text_reference(
    model_id_or_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 1,
    apply_chat_template: bool = True,
    device: str = "cpu",
) -> TextReference:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model_path = str(model_id_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = getattr(processor, "tokenizer", processor)

    model = load_hf_generation_model(model_path).to(device)
    model.eval()

    text = prompt
    if apply_chat_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        generated = model.generate(**encoded, max_new_tokens=int(max_new_tokens), do_sample=False)

    input_ids = tuple(int(value) for value in encoded["input_ids"][0].detach().cpu().tolist())
    generated_ids = tuple(int(value) for value in generated[0].detach().cpu().tolist())
    decoded = tokenizer.decode(generated[0], skip_special_tokens=False)
    logits = outputs.logits[:, -1, :].detach().cpu().float().numpy()
    return TextReference(prompt=prompt, input_ids=input_ids, generated_ids=generated_ids, text=decoded, logits=logits)


def compare_hf_and_cactus_text(
    bundle_dir: str | Path,
    model_id_or_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 1,
    apply_chat_template: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    hf = run_hf_text_reference(
        model_id_or_path,
        prompt,
        max_new_tokens=max_new_tokens,
        apply_chat_template=apply_chat_template,
        device=device,
    )
    cactus = run_cactus_text_smoke(bundle_dir, prompt, max_new_tokens=max_new_tokens)
    return {
        "hf": {
            "input_ids": list(hf.input_ids),
            "generated_ids": list(hf.generated_ids),
            "text": hf.text,
        },
        "cactus": cactus,
        "tokenizer_parity_note": "Use hf.input_ids with `cactus run --input-ids` to isolate tokenizer/chat-template differences from graph math differences.",
    }


def audit_bundle_bindings(bundle_dir: str | Path) -> list[str]:
    bundle = Path(bundle_dir)
    manifest_path = bundle / "components" / "manifest.json"

    if not manifest_path.exists():
        return [f"Missing component manifest: {manifest_path}"]

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    gemma_scaled = 0
    gemma_per_layer = 0

    for component in data.get("components", ()):
        component_name = str(component.get("component") or component.get("name") or component.get("id") or "unknown")

        for index, binding in enumerate(component.get("bound_constant_bindings", ())):
            if not isinstance(binding, Mapping):
                issues.append(f"{component_name}[{index}]: binding is not an object")
                continue

            label = f"{component_name}[{index}] node_id={binding.get('node_id')}"
            source_name = binding.get("source_name")
            value_id = binding.get("value_id")
            scale_factor = float(binding.get("scale_factor", 1.0) or 1.0)

            if not source_name:
                issues.append(f"{label}: missing source_name")

            if not value_id:
                issues.append(f"{label}: missing value_id")

            if scale_factor != 1.0:
                gemma_scaled += 1

            source_text = str(source_name or binding.get("path") or "")
            if "per_layer" in source_text or "embed_tokens_per_layer" in source_text:
                gemma_per_layer += 1

    if gemma_scaled:
        issues.append(f"Gemma scaled bindings detected: {gemma_scaled}")

    if "gemma" in bundle.name.lower() and gemma_per_layer == 0:
        issues.append("Gemma bundle has no per-layer/PLI binding metadata; regenerate with source_name/value_id metadata before judging correctness")

    return issues


def load_hf_generation_model(model_path: str) -> Any:
    from transformers import AutoModelForCausalLM

    candidate_classes: list[Any] = [AutoModelForCausalLM]

    try:
        from transformers import AutoModelForImageTextToText
        candidate_classes.append(AutoModelForImageTextToText)
    except Exception:
        pass

    try:
        from transformers import AutoModelForSeq2SeqLM
        candidate_classes.append(AutoModelForSeq2SeqLM)
    except Exception:
        pass

    last_error: Exception | None = None

    for model_class in candidate_classes:
        try:
            return model_class.from_pretrained(model_path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to load an HF generation model from {model_path}") from last_error


def load_component_manifest(bundle_dir: Path, component_name: str) -> dict[str, Any]:
    manifest_path = bundle_dir / "components" / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    for component in data.get("components", ()):
        name = component.get("component") or component.get("name") or component.get("id")

        if name == component_name:
            return component

    raise KeyError(f"Component {component_name!r} was not found in {manifest_path}")


def load_component_graph(bundle_dir: Path, component: Mapping[str, Any]) -> Any:
    graph_path = bundle_dir / str(component.get("graph") or component.get("graph_path"))
    graph_cls = cactus_graph_class()
    return graph_cls.load(graph_path)


def bind_component_weights(graph: Any, bundle_dir: Path, component: Mapping[str, Any]) -> None:
    for binding in component.get("bound_constant_bindings", ()):
        if not isinstance(binding, Mapping):
            continue

        node_id = int(binding["node_id"])
        path = resolve_bundle_path(bundle_dir, str(binding["path"]))
        tensor = graph._tensor_from_node(node_id)
        graph.bind_mmap_weights(tensor, path)


def set_component_inputs(graph: Any, component: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    node_ids = tuple(int(value) for value in component.get("runtime_input_node_ids", ()))
    logical_inputs = tuple(str(value) for value in component.get("logical_inputs", ()))
    node_by_name = dict(zip(logical_inputs, node_ids))

    for name, value in inputs.items():
        if name not in node_by_name:
            raise KeyError(f"Input {name!r} is not declared by component {component.get('component')!r}")

        graph.set_input(graph._tensor_from_node(node_by_name[name]), value)


def read_component_outputs(graph: Any, component: Mapping[str, Any]) -> dict[str, np.ndarray]:
    node_ids = tuple(int(value) for value in component.get("output_node_ids", ()))
    logical_outputs = tuple(str(value) for value in component.get("logical_outputs", ()))
    outputs: dict[str, np.ndarray] = {}

    for name, node_id in zip(logical_outputs, node_ids):
        outputs[name] = graph._tensor_from_node(node_id).numpy()

    return outputs


def compare_topk(actual: np.ndarray, reference: np.ndarray, topk: int) -> bool | None:
    if topk <= 0 or actual.size == 0:
        return None

    actual_flat = actual.reshape(-1, actual.shape[-1]) if actual.ndim > 1 else actual.reshape(1, -1)
    reference_flat = reference.reshape(-1, reference.shape[-1]) if reference.ndim > 1 else reference.reshape(1, -1)

    if actual_flat.shape != reference_flat.shape or actual_flat.shape[-1] < topk:
        return None

    actual_top = np.argpartition(actual_flat, -topk, axis=-1)[:, -topk:]
    reference_top = np.argpartition(reference_flat, -topk, axis=-1)[:, -topk:]
    return all(set(map(int, left)) == set(map(int, right)) for left, right in zip(actual_top, reference_top))


def resolve_bundle_path(bundle_dir: Path, path: str) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    return bundle_dir / candidate


def cactus_graph_class() -> Any:
    try:
        from cactus.bindings.cactus import Graph
    except ModuleNotFoundError:
        from bindings.cactus import Graph

    return Graph
