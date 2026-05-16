from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WeightBinding:
    path: str
    kind: str  # "weight" | "embedding"
    source_name: str


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_EMBEDDING_OUTPUT_NAMES = {
    "token_embeddings.weights",
    "token_embeddings.cq2.weights",
    "token_embeddings.cq3.weights",
    "token_embeddings.cq4.weights",
    "decoder_token_embeddings.weights",
    "decoder_token_embeddings.cq2.weights",
    "decoder_token_embeddings.cq3.weights",
    "decoder_token_embeddings.cq4.weights",
}


def _candidate_model_dir_names(model_name_or_path: str) -> list[str]:
    candidates: list[str] = []

    def _add(name: str) -> None:
        name = name.strip().lower()
        if name and name not in candidates:
            candidates.append(name)

    raw = model_name_or_path.strip()
    if not raw:
        return candidates

    path = Path(raw)
    _add(path.name)
    if "/" in raw:
        _add(raw.split("/")[-1])

    slug = raw.replace("/", "--")
    _add(slug)
    _add(slug.replace("--", "-"))

    for part in path.parts:
        if not part.startswith("models--"):
            continue
        cache_name = part[len("models--") :]
        _add(cache_name)
        _add(cache_name.replace("--", "-"))
        _add(cache_name.split("--")[-1])
        break

    return candidates


def _default_weights_dir_for_model_name(model_name_or_path: str) -> str | None:
    if not model_name_or_path:
        return None
    for model_dir_name in _candidate_model_dir_names(model_name_or_path):
        candidate = _PROJECT_ROOT / "weights" / model_dir_name
        if candidate.exists():
            return str(candidate)
    return None


def resolve_transpile_weights_dir(graph_meta: dict[str, object]) -> str | None:
    explicit = graph_meta.get("weights_dir")
    if isinstance(explicit, str) and explicit:
        return explicit

    family = str(graph_meta.get("adapter_family", "")).upper()
    family_env = f"CACTUS_TRANSPILER_WEIGHTS_DIR_{family}"
    if family and family_env in os.environ and os.environ[family_env]:
        return os.environ[family_env]

    generic = os.environ.get("CACTUS_TRANSPILER_WEIGHTS_DIR")
    if generic:
        return generic

    model_name_or_path = graph_meta.get("model_name_or_path")
    if isinstance(model_name_or_path, str) and model_name_or_path:
        return _default_weights_dir_for_model_name(model_name_or_path)
    return None


def _manifest_source_aliases(name: str) -> tuple[str, ...]:
    aliases: list[str] = []

    def _add(candidate: str) -> None:
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    def _add_with_wrappers(candidate: str) -> None:
        _add(candidate)
        for prefix in (
            "module.",
            "module.model.",
            "adapter.",
            "adapter.model.",
        ):
            _add(f"{prefix}{candidate}")

    raw = name.strip()
    if not raw:
        return ()

    _add_with_wrappers(raw)

    for prefix in ("encoder.", "decoder.", "model.encoder.", "model.decoder."):
        if raw.startswith(prefix):
            _add_with_wrappers(raw[len(prefix) :])

    for prefix in ("model.language_model.", "language_model."):
        if raw.startswith(prefix):
            tail = raw[len(prefix) :]
            _add_with_wrappers(tail)
            _add_with_wrappers(f"backbone.{tail}")
            _add_with_wrappers(f"model.{tail}")

    if raw.startswith("model."):
        tail = raw[len("model.") :]
        _add_with_wrappers(tail)

    if raw.startswith("backbone."):
        tail = raw[len("backbone.") :]
        _add_with_wrappers(tail)
        _add_with_wrappers(f"model.{tail}")
        _add_with_wrappers(f"model.language_model.{tail}")

    return tuple(aliases)


def _manifest_entry_kind(output_name: str, explicit_kind: Any, explicit_names: list[object]) -> str:
    if isinstance(explicit_kind, str) and explicit_kind:
        return explicit_kind

    output_text = output_name.lower()
    explicit_text = " ".join(str(name).lower() for name in explicit_names if isinstance(name, str))
    if (
        output_name in _TOKEN_EMBEDDING_OUTPUT_NAMES
        or "embedding" in output_text
        or "token_embeddings" in output_text
        or "embed_tokens.weight" in explicit_text
        or "embed_positions.weight" in explicit_text
    ):
        return "embedding"
    return "weight"


def _flatten_convert_manifest(root_manifest: dict[str, Any]) -> dict[str, object]:
    flattened: dict[str, object] = {}
    rows = root_manifest.get("weights")
    if not isinstance(rows, list):
        for name, entry in root_manifest.items():
            if not isinstance(name, str):
                continue
            for alias in _manifest_source_aliases(name):
                flattened.setdefault(alias, entry)
        return flattened

    for row in rows:
        if not isinstance(row, dict):
            continue
        output_name = row.get("output_name") or row.get("filename")
        if not isinstance(output_name, str) or not output_name:
            continue

        explicit_names: list[object] = [
            row.get("source_name"),
            row.get("hf_name"),
            row.get("adapter_name"),
        ]
        source_names = row.get("source_names")
        if isinstance(source_names, list):
            explicit_names.extend(source_names)

        entry = {
            "filename": output_name,
            "kind": _manifest_entry_kind(output_name, row.get("kind"), explicit_names),
        }
        for source_name in explicit_names:
            if not isinstance(source_name, str) or not source_name:
                continue
            for alias in _manifest_source_aliases(source_name):
                flattened.setdefault(alias, entry)
    return flattened


def _load_weights_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "weights_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        loaded_manifest = json.loads(manifest_path.read_text())
    except Exception:
        return {}
    if not isinstance(loaded_manifest, dict):
        return {}
    return _flatten_convert_manifest(loaded_manifest)


def _binding_from_manifest_entry(root: Path, source_name: str, entry: object) -> WeightBinding | None:
    if not isinstance(entry, dict):
        return None
    filename = entry.get("filename")
    kind = entry.get("kind", "weight")
    if not isinstance(filename, str) or not isinstance(kind, str):
        return None
    candidate = root / filename
    if not candidate.exists():
        return None
    return WeightBinding(path=str(candidate), kind=kind, source_name=source_name)


def resolve_weight_binding(*, weights_dir: str | None, source_name: str) -> WeightBinding | None:
    """Resolve an exported parameter to a converted Cactus tensor file.

    The resolver intentionally trusts only `weights_manifest.json`. It may add
    wrapper-local aliases such as `module.` or `adapter.model.`, but it does not
    guess old model-specific filenames from layer names.
    """

    if not weights_dir:
        return None
    root = Path(weights_dir)
    if not root.exists():
        return None
    manifest = _load_weights_manifest(root)
    binding = _binding_from_manifest_entry(root, source_name, manifest.get(source_name))
    if binding is not None:
        return binding
    return None
