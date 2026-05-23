from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ..model_adapters.adapters import adapter_for_family
from ..model_adapters.detection import detect_family


def _dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _is_integer_dtype(dtype_name: str) -> bool:
    return any(x in dtype_name for x in ("int", "uint", "bool"))


def load_safetensor_index(root: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    out: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard in sorted(root.glob("*.safetensors")):
        tensors = load_file(shard)
        for key, tensor in tensors.items():
            out[key] = (tuple(int(x) for x in tensor.shape), _dtype_name(tensor.dtype))
        del tensors
    if out:
        return out
    raise ValueError(f"no safetensors found under {root}")


def _state_index(state: dict) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        key: (tuple(int(x) for x in tensor.shape), _dtype_name(tensor.dtype))
        for key, tensor in state.items()
    }


def load_source_state_index(model_path: str | Path, family: str = "auto") -> dict[str, tuple[tuple[int, ...], str]]:
    path = Path(model_path)
    if path.exists() and path.is_dir() and list(path.glob("*.safetensors")):
        source = load_file(sorted(path.glob("*.safetensors"))[0])
        for shard in sorted(path.glob("*.safetensors"))[1:]:
            source.update(load_file(shard))
        if family != "auto":
            source = adapter_for_family(family).normalize_state_dict(source).state_dict
        return _state_index(source)
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        resolved_family = family
        if resolved_family == "auto":
            try:
                cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=path.exists())
                resolved_family = detect_family(cfg, "auto")
            except Exception:
                resolved_family = "generic"

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)
        except Exception:
            model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)
        state = adapter_for_family(resolved_family).normalize_state_dict(model.state_dict()).state_dict
        return _state_index(state)
    except Exception as exc:
        raise RuntimeError(f"could not load source model state from {model_path}") from exc


def validate_qdq(args) -> dict[str, Any]:
    family = getattr(args, "model_family", "auto")
    source = load_source_state_index(args.source_model, family)
    qdq = load_safetensor_index(Path(args.qdq))
    source_keys = set(source)
    qdq_keys = set(qdq)
    missing = sorted(source_keys - qdq_keys)
    extra = sorted(qdq_keys - source_keys)
    shape_mismatches = []
    dtype_mismatches = []
    for key in sorted(source_keys & qdq_keys):
        source_shape, source_dtype = source[key]
        qdq_shape, qdq_dtype = qdq[key]
        if source_shape != qdq_shape:
            shape_mismatches.append({"key": key, "source": list(source_shape), "qdq": list(qdq_shape)})
        if _is_integer_dtype(source_dtype) and source_dtype != qdq_dtype:
            dtype_mismatches.append({"key": key, "source": source_dtype, "qdq": qdq_dtype})
    report = {
        "source_model": str(args.source_model),
        "qdq": str(args.qdq),
        "source_tensors": len(source),
        "qdq_tensors": len(qdq),
        "missing": missing,
        "extra": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "ok": not missing and not extra and not shape_mismatches and not dtype_mismatches,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and not report["ok"]:
        raise SystemExit(1)
    return report
