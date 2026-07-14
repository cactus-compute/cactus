from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .calibration.hessian import collect_manifest_hessians
from .calibration.loaders import language_text, load_manifest, read_jsonl
from .cactus_adapters.config_utils import (
    cfg_get,
)
from .cactus_adapters.tensor_io import (
    save_depthwise_conv_int8_with_header,
    save_pointwise_conv1d_int8_with_header,
    save_tensor_with_header,
)
from .export.files import copy_runtime_files, write_config_txt
from .export.qdq import convert_qdq
from .export.reports import print_summary, write_reports
from .export.validate import validate_qdq
from .model_adapters.detection import SUPPORTED_FAMILIES, detect_family
from .model_adapters.adapters import adapter_for_family
from .model_adapters.nemo import ensure_parakeet_tdt_nemo_source
from .quantization.cq import quantize_hadamard, quantize_orthogonal, write_cq_tensor
from .compat import patch_transformers_import_compat

ALPHA = 0.25


def _hf_cache_dir() -> str | None:
    import os

    return os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or None


def _load_hf(model_id_or_path: str, device: str, *, skip_model_load: bool = False):
    import logging, warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*You are using a model of type.*")
    for note in patch_transformers_import_compat():
        print(f"note={note}")
    from ..models.needle import register_with_transformers

    register_with_transformers()
    nemo_export = ensure_parakeet_tdt_nemo_source(model_id_or_path, cache_dir=_hf_cache_dir())
    if nemo_export is not None:
        model_id_or_path = nemo_export
    try:
        from transformers import AutoConfig, AutoModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for conversion") from exc
    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True, local_files_only=Path(model_id_or_path).exists())
    except Exception:
        try:
            if Path(model_id_or_path).exists():
                cfg_path = Path(model_id_or_path) / "config.json"
            else:
                from huggingface_hub import hf_hub_download
                cfg_path = Path(hf_hub_download(model_id_or_path, "config.json", cache_dir=_hf_cache_dir()))
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not load model {model_id_or_path!r}: {exc}") from exc
    family = detect_family(cfg, "auto")
    adapter = adapter_for_family(family)
    processor = adapter.load_processor(model_id_or_path)
    if skip_model_load:
        return cfg, processor, None
    model_cls = adapter.model_class(cfg)
    model_type = str(cfg_get(cfg, "model_type", "") or "").lower()
    if isinstance(cfg, dict) and model_type == "parakeet_tdt":
        return cfg, processor, None
    try:
        model = model_cls.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16 if torch is not None else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=Path(model_id_or_path).exists(),
        )
    except Exception as primary_exc:
        try:
            model = AutoModel.from_pretrained(
                model_id_or_path,
                torch_dtype=torch.float16 if torch is not None else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=Path(model_id_or_path).exists(),
            )
        except Exception as auto_exc:
            print(f"warning: could not load model object ({model_cls.__name__}: {primary_exc}; AutoModel: {auto_exc})")
            model = None
    if model is None:
        return cfg, processor, model
    model.eval()
    if torch is not None and device != "cpu":
        try:
            model.to(device)
        except Exception:
            model.to("cpu")
    return cfg, processor, model


def _module_name(param_name: str) -> str:
    return param_name[:-7] if param_name.endswith(".weight") else param_name


def _tensor_shape(tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def _tensor_bytes(path: Path | None) -> int:
    return path.stat().st_size if path and path.exists() else 0


def _bits_for_component(component: str, args: argparse.Namespace) -> int:
    if component == "language" and args.language_bits is not None:
        return int(args.language_bits)
    if component == "vision" and args.vision_bits is not None:
        return int(args.vision_bits)
    if component in {"audio", "transcription"} and args.audio_bits is not None:
        return int(args.audio_bits)
    if component == "embedding" and args.embedding_bits is not None:
        return int(args.embedding_bits)
    return int(args.bits)


def _load_checkpoint_state_dict(model_id_or_path: str) -> dict[str, Any] | None:
    nemo_export = ensure_parakeet_tdt_nemo_source(model_id_or_path, cache_dir=_hf_cache_dir())
    if nemo_export is not None:
        model_id_or_path = nemo_export
    root = Path(model_id_or_path)
    if not root.exists() or not root.is_dir():
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            try:
                model_file = hf_hub_download(model_id_or_path, "model.safetensors", cache_dir=_hf_cache_dir())
                return load_file(model_file)
            except Exception:
                index_file = hf_hub_download(model_id_or_path, "model.safetensors.index.json", cache_dir=_hf_cache_dir())
                index = json.loads(Path(index_file).read_text(encoding="utf-8"))
                shard_names = sorted(set(index.get("weight_map", {}).values()))
                if not shard_names:
                    return None
                state: dict[str, Any] = {}
                for shard in shard_names:
                    shard_path = hf_hub_download(model_id_or_path, shard, cache_dir=_hf_cache_dir())
                    for key, tensor in load_file(shard_path).items():
                        if key in state:
                            raise RuntimeError(f"duplicate tensor key {key!r} across checkpoint shards")
                        state[key] = tensor
                return state
        except RuntimeError:
            raise
        except Exception:
            return None
    safetensor_files = sorted(root.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors import safe_open
        except Exception:
            return None
        state: dict[str, Any] = {}
        for file in safetensor_files:
            with safe_open(file, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key in state:
                        raise RuntimeError(f"duplicate tensor key {key!r} across checkpoint shards")
                    state[key] = sf.get_tensor(key)
        return state
    if torch is None:
        return None
    bin_files = sorted(root.glob("pytorch_model*.bin"))
    if not bin_files:
        return None
    state = {}
    for file in bin_files:
        shard = torch.load(file, map_location="cpu")
        shard_state = shard.get("state_dict", shard) if isinstance(shard, dict) else shard
        for key, tensor in shard_state.items():
            if key in state:
                raise RuntimeError(f"duplicate tensor key {key!r} across checkpoint shards")
            state[key] = tensor
    return state


def _augment_state_dict_for_family(state_dict: dict[str, Any], family: str) -> dict[str, Any]:
    return adapter_for_family(family).normalize_state_dict(state_dict).state_dict


def _save_fallback_tensor(tensor, out_path: Path, precision: str, family: str) -> None:
    if len(_tensor_shape(tensor)) > 4:
        if torch is not None and isinstance(tensor, torch.Tensor):
            tensor = tensor.reshape(tensor.shape[0], -1)
        else:
            arr = np.asarray(tensor)
            tensor = arr.reshape(arr.shape[0], -1)
    if precision == "INT8":
        shape = _tensor_shape(tensor)
        if "conv_pointwise" in out_path.name and len(shape) == 3 and shape[2] == 1:
            save_pointwise_conv1d_int8_with_header(tensor, out_path)
            return
        if "conv_depthwise.weights" in out_path.name and len(shape) == 3 and shape[1] == 1:
            save_depthwise_conv_int8_with_header(tensor, out_path)
            return
        save_tensor_with_header(tensor, out_path, precision="INT8", model_type=family, allow_int8_bias=True)
        return
    save_tensor_with_header(tensor, out_path, precision="FP16", model_type=family)


def _scale_cq_norms(cq, factor: float):
    if factor == 1.0:
        return cq
    return replace(cq, norms=(cq.norms.astype(np.float32) * float(factor)).astype(np.float16))


def _validate_cq_layout(policy, shape: tuple[int, ...], source_name: str, output_name: str) -> None:
    if getattr(policy, "layout", "row_major") != "interleaved_4row":
        return
    if policy.rotation != "orthogonal" or int(policy.bits or 0) != 4:
        raise RuntimeError(f"{source_name}: INTERLEAVED_4ROW output {output_name} requires orthogonal CQ4")
    if len(shape) != 2:
        raise RuntimeError(f"{source_name}: INTERLEAVED_4ROW output {output_name} requires rank-2 tensor, got shape={shape}")
    n, k = int(shape[0]), int(shape[1])
    if n % 4 != 0 or k % 32 != 0:
        raise RuntimeError(
            f"{source_name}: INTERLEAVED_4ROW output {output_name} requires N % 4 == 0 and K % 32 == 0, got shape={shape}"
        )


def _adapt_tensor_for_cactus(tensor, output_name: str | None, family: str):
    match = type("_CompatMatch", (), {"output_name": output_name})()
    transformed, _name = adapter_for_family(family).transform_tensor(match, tensor)
    return transformed


def _input_scale_from_diag(diag: np.ndarray | None, tensor) -> np.ndarray | None:
    if diag is None:
        return None
    w = tensor.detach().float().cpu().numpy() if torch is not None and isinstance(tensor, torch.Tensor) else np.asarray(tensor, dtype=np.float32)
    d = np.asarray(diag, dtype=np.float32).reshape(-1)
    if d.shape[0] != w.shape[1]:
        return None
    x_abs = np.sqrt(np.clip(d, 1e-12, None)).clip(min=1e-6)
    w_abs = np.mean(np.abs(w), axis=0).clip(min=1e-6)
    raw = np.power(x_abs, ALPHA) / np.power(w_abs, 1.0 - ALPHA)
    raw = raw / np.exp(np.mean(np.log(np.clip(raw, 1e-6, None))))
    return np.clip(raw, 1.0 / 8.0, 8.0).astype(np.float32)


def _row_table_scale(tensor, token_ids: list[int]) -> np.ndarray | None:
    if not token_ids:
        return None
    table = tensor.detach().float().cpu().numpy() if torch is not None and isinstance(tensor, torch.Tensor) else np.asarray(tensor, dtype=np.float32)
    if table.ndim != 2:
        return None
    ids = np.asarray(token_ids, dtype=np.int64)
    ids = np.clip(ids, 0, table.shape[0] - 1)
    x_abs = np.mean(np.abs(table[ids]), axis=0).clip(min=1e-6)
    w_abs = np.mean(np.abs(table), axis=0).clip(min=1e-6)
    raw = np.power(x_abs, ALPHA) / np.power(w_abs, 1.0 - ALPHA)
    raw = raw / np.exp(np.mean(np.log(np.clip(raw, 1e-6, None))))
    return np.clip(raw, 1.0 / 8.0, 8.0).astype(np.float32)


def _collect_manifest_token_ids(
    processor,
    manifest: dict[str, Any],
    limits: dict[str, int | None],
    max_tokens: int = 32768,
) -> list[int]:
    section = manifest.get("language") or manifest.get("transcription")
    if not section:
        return []
    section_name = "language" if manifest.get("language") else "transcription"
    path = section["path"] if isinstance(section, dict) else section
    out: list[int] = []
    for row in read_jsonl(path, limits.get(section_name)):
        if processor is not None and section_name == "language":
            text = language_text(row)
            if text:
                try:
                    encoded = processor(text, return_tensors=None)
                    ids = encoded.get("input_ids", encoded) if isinstance(encoded, Mapping) else encoded
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                    out.extend(int(x) for x in ids)
                    if len(out) >= max_tokens:
                        return out[:max_tokens]
                    continue
                except Exception:
                    pass
        for key in ("full_token_ids", "input_ids", "prompt_token_ids", "completion_token_ids"):
            ids = row.get(key)
            if isinstance(ids, list):
                out.extend(int(x) for x in ids if isinstance(x, int))
                if len(out) >= max_tokens:
                    return out[:max_tokens]
        if len(out) >= max_tokens:
            return out[:max_tokens]
    return out[:max_tokens]


def _save_hessian_artifacts(out_dir: Path, stats) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"sample_counts": dict(stats.samples)}
    for attr in ("unresolved_targets", "errors", "error_samples", "rows", "timings"):
        if hasattr(stats, attr):
            metadata[attr] = getattr(stats, attr)
    if not stats.hessians:
        (out_dir / "hessian_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return dict(stats.samples)
    try:
        from safetensors.torch import save_file
        tensors = {}
        if torch is not None:
            for name, h in stats.hessians.items():
                tensors[name.replace("/", "__")] = h.detach().cpu()
            for name, d in stats.diag.items():
                tensors[name.replace("/", "__") + ".diag"] = d.detach().cpu()
            save_file(tensors, str(out_dir / "hessians.safetensors"))
    except Exception as exc:
        metadata["save_warning"] = str(exc)
    (out_dir / "hessian_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dict(stats.samples)


def _load_hessian_artifacts(cache_dir: Path):
    from .calibration.hessian import HessianStats

    metadata_path = cache_dir / "hessian_metadata.json"
    tensor_path = cache_dir / "hessians.safetensors"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{cache_dir} does not contain hessian_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats = HessianStats(samples={str(k): int(v) for k, v in metadata.get("sample_counts", {}).items()})
    stats.unresolved_targets = list(metadata.get("unresolved_targets", []))
    stats.errors = {str(k): int(v) for k, v in metadata.get("errors", {}).items()}
    stats.error_samples = list(metadata.get("error_samples", []))
    stats.rows = {str(k): int(v) for k, v in metadata.get("rows", {}).items()}
    stats.timings = {str(k): float(v) for k, v in metadata.get("timings", {}).items()}
    if tensor_path.exists():
        if torch is None:
            raise RuntimeError("torch is required to load Hessian cache")
        from safetensors.torch import load_file

        tensors = load_file(str(tensor_path), device="cpu")
        for key, tensor in tensors.items():
            name = key.replace("__", "/")
            if name.endswith(".diag"):
                stats.diag[name[:-5]] = tensor
            else:
                stats.hessians[name] = tensor
    return stats


def _config_dict(cfg) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return cfg
    return dict(getattr(cfg, "__dict__", {}))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def convert(args: argparse.Namespace) -> None:
    if args.bits not in {1, 2, 3, 4}:
        raise SystemExit("--bits must be one of 1,2,3,4")
    args.device = _resolve_device(args.device)
    out_dir = Path(args.out)
    if args.force and args.hessian_cache_in:
        try:
            if Path(args.hessian_cache_in).resolve() == out_dir.resolve():
                raise SystemExit("--hessian-cache-in cannot point at --out when --force is set; --force deletes --out before cache load")
        except FileNotFoundError:
            pass
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, processor, model = _load_hf(args.model, args.device, skip_model_load=bool(args.skip_model_load))
    runtime_source = ensure_parakeet_tdt_nemo_source(args.model, cache_dir=_hf_cache_dir()) or args.model
    family = detect_family(cfg, args.model_family)
    adapter = adapter_for_family(family)
    checkpoint_state = _load_checkpoint_state_dict(args.model)
    if checkpoint_state is None:
        if model is None:
            raise RuntimeError(f"could not load model object or checkpoint tensors from {args.model}")
        checkpoint_state = model.state_dict()
    normalized_state = adapter.normalize_state_dict(checkpoint_state)
    state_dict = normalized_state.state_dict
    model_config = adapter.runtime_config(cfg)
    model_config["model_type"] = adapter.runtime_model_type()
    write_config_txt({**_config_dict(cfg), **model_config}, out_dir)
    copy_runtime_files(runtime_source, out_dir, token=getattr(args, "token", None), cache_dir=_hf_cache_dir())
    try:
        from .cactus_adapters.tokenizer import convert_hf_tokenizer

        tokenizer = getattr(processor, "tokenizer", processor)
        if tokenizer is not None:
            convert_hf_tokenizer(tokenizer, out_dir, model_id=args.model, model_type=family)
    except Exception as exc:
        if args.strict:
            raise RuntimeError(f"failed to write Cactus tokenizer files: {exc}") from exc

    manifest = load_manifest(args.calibration_manifest)
    limits = {
        "language": args.max_language_examples,
        "vision": args.max_vision_examples,
        "audio": args.max_audio_examples,
        "transcription": args.max_transcription_examples,
        "embedding": args.max_embedding_examples,
    }
    rows: list[dict[str, Any]] = []
    pending: list[tuple[str, Any, Any, Any]] = []
    num_layers = max(
        int(model_config.get("num_layers", 0) or 0),
        int(cfg_get(cfg, "num_encoder_layers", 0) or 0),
        int(cfg_get(cfg, "num_decoder_layers", 0) or 0),
    ) or None
    target_modules: set[str] = set()
    for name, tensor in state_dict.items():
        match = adapter.name_tensor(name, tensor, num_layers)
        requested_bits = _bits_for_component(match.component, args)
        policy = adapter.policy(match, _tensor_shape(tensor), requested_bits)
        if policy.action == "convert" and policy.use_gptq:
            target = adapter.module_target_name(name, model)
            if target:
                target_modules.add(target)
        pending.append((name, tensor, match, policy))

    if target_modules and model is None and not args.hessian_cache_in:
        msg = "model object failed to load; GPTQ Hessian calibration is disabled and use_gptq tensors will fall back to RTN"
        if args.strict:
            raise RuntimeError(msg)
        print(f"warning: {msg}")

    if args.hessian_cache_in:
        hessian_stats = _load_hessian_artifacts(Path(args.hessian_cache_in))
    else:
        progress_path = out_dir / "hessian_progress.json" if args.hessian_progress else None
        hessian_stats = (
            collect_manifest_hessians(
                model,
                processor,
                manifest,
                limits,
                target_modules,
                args.device,
                adapter=adapter,
                batch_size=args.calibration_batch_size,
                gpu_flush_interval=args.hessian_gpu_flush_interval,
                progress_path=progress_path,
                preprocessed_cache_dir=args.preprocessed_cache_dir,
            )
            if processor is not None and model is not None and target_modules
            else None
        )
    hessian_samples = _save_hessian_artifacts(out_dir, hessian_stats) if hessian_stats is not None else {}
    if args.hessian_cache_out and hessian_stats is not None:
        _save_hessian_artifacts(Path(args.hessian_cache_out), hessian_stats)
    hessians_np = {}
    diag_np = {}
    if hessian_stats is not None and torch is not None:
        for name, h in hessian_stats.hessians.items():
            hessians_np[name] = h.detach().cpu().numpy().astype(np.float32)
        for name, d in hessian_stats.diag.items():
            diag_np[name] = d.detach().cpu().numpy().astype(np.float32)

    scale_token_ids = _collect_manifest_token_ids(processor, manifest, limits)

    for name, tensor, match, policy in pending:
        emissions = adapter.expand_tensor(match, tensor)
        if not emissions:
            emissions = [type("_IgnoredEmission", (), {"output_name": None, "tensor": tensor, "transform": "none", "source_names": None})()]
        provenance = normalized_state.provenance.get(name)
        source_names = provenance.source_names if provenance else [name]
        for emission in emissions:
            qdq_restore = getattr(emission, "qdq_restore", None) or (provenance.qdq_restore if provenance else "hf_key")
            out_path = out_dir / emission.output_name if emission.output_name else None
            emit_tensor = emission.tensor
            emit_match = replace(match, output_name=emission.output_name)
            requested_bits = _bits_for_component(emit_match.component, args)
            emit_policy = adapter.policy(emit_match, _tensor_shape(emit_tensor), requested_bits)
            manifest_transform = provenance.transform if provenance and provenance.transform != "none" else emission.transform
            emit_source_names = emission.source_names or source_names
            status = emit_policy.action
            precision = emit_policy.precision
            gptq_used = False
            hessian_missing_reason = None
            module_name = adapter.module_target_name(name, model) or _module_name(name)
            try:
                if not match.recognized and args.strict:
                    raise RuntimeError(f"unrecognized tensor in strict mode: {name}")
                if emit_policy.action == "convert" and out_path is not None:
                    _validate_cq_layout(emit_policy, _tensor_shape(emit_tensor), name, out_path.name)
                    if emit_policy.use_gptq and int(hessian_samples.get(module_name, 0)) <= 0:
                        hessian_missing_reason = "expected GPTQ target had zero samples"
                        if args.strict:
                            raise RuntimeError(f"{name}: {hessian_missing_reason} ({module_name})")
                    hessian = hessians_np.get(module_name)
                    input_scale = None
                    if emit_policy.rotation == "orthogonal" or name.endswith("embed_tokens_per_layer.weight"):
                        input_scale = _row_table_scale(emit_tensor, scale_token_ids)
                    else:
                        input_scale = _input_scale_from_diag(diag_np.get(module_name), emit_tensor)
                    if emit_policy.rotation == "orthogonal":
                        cq = quantize_orthogonal(emit_tensor, bits=int(emit_policy.bits or 4), input_scale=input_scale)
                    else:
                        cq = quantize_hadamard(
                            emit_tensor,
                            bits=int(emit_policy.bits or requested_bits),
                            hessian=hessian,
                            use_gptq=emit_policy.use_gptq,
                            input_scale=input_scale,
                        )
                    cq = _scale_cq_norms(cq, adapter.scale_factor(out_path.name))
                    if getattr(emit_policy, "layout", "row_major") == "interleaved_4row":
                        cq = replace(cq, interleaved_4row=True)
                    write_cq_tensor(out_path, cq)
                    gptq_used = cq.gptq_used
                    status = "converted" if match.recognized else "unrecognized"
                elif emit_policy.action == "fallback" and out_path is not None:
                    _save_fallback_tensor(emit_tensor, out_path, precision, family)
                    status = "fallback" if match.recognized else "unrecognized"
                elif emit_policy.action == "ignored":
                    status = "ignored"
            except Exception as exc:
                if args.strict or getattr(emit_policy, "layout", "row_major") == "interleaved_4row":
                    raise
                if out_path is not None:
                    _save_fallback_tensor(emit_tensor, out_path, "FP16", family)
                    status = "fallback" if match.recognized else "unrecognized"
                    precision = "FP16"
                    emit_policy = replace(emit_policy, action="fallback", precision="FP16", fallback_reason=str(exc))
            rows.append({
                "source_name": name,
                "hf_name": match.hf_name or name,
                "adapter_name": match.adapter_name or name,
                "output_file": str(out_path.name) if out_path else None,
                "shape": list(_tensor_shape(emit_tensor)),
                "dtype": str(getattr(emit_tensor, "dtype", "")),
                "component": emit_policy.component,
                "policy": emit_policy.action,
                "precision": precision,
                "bits": emit_policy.bits,
                "status": status,
                "required": bool(emit_policy.action != "ignored"),
                "fallback_reason": emit_policy.fallback_reason,
                "hessian_samples": int(hessian_samples.get(module_name, 0)),
                "gptq_used": bool(gptq_used),
                "bytes": _tensor_bytes(out_path),
                "scale_factor": float(adapter.scale_factor(out_path.name)) if out_path else 1.0,
                "recognized": bool(match.recognized),
                "adapter_family": family,
                "source_names": emit_source_names,
                "transform": manifest_transform,
                "qdq_restore": qdq_restore,
                "hessian_missing_reason": hessian_missing_reason,
            })

    summary = write_reports(out_dir, rows)
    print_summary(summary)
    unrecognized = [r["source_name"] for r in rows if r["status"] == "unrecognized"]
    if unrecognized and not args.strict:
        print(f"\nWarning: {len(unrecognized)} unrecognized tensors were exported with generic names or FP16 fallback.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cactus.convert")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("convert")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bits", type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument("--language-bits", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--vision-bits", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--audio-bits", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--embedding-bits", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--calibration-manifest")
    p.add_argument("--device", default="auto")
    p.add_argument("--model-family", default="auto", choices=sorted(SUPPORTED_FAMILIES))
    p.add_argument("--strict", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-model-load", action="store_true")
    p.add_argument("--max-language-examples", type=int)
    p.add_argument("--max-vision-examples", type=int)
    p.add_argument("--max-audio-examples", type=int)
    p.add_argument("--max-transcription-examples", type=int)
    p.add_argument("--max-embedding-examples", type=int)
    p.add_argument("--hessian-cache-in")
    p.add_argument("--hessian-cache-out")
    p.add_argument("--calibration-batch-size", type=int, default=1)
    p.add_argument("--hessian-gpu-flush-interval", type=int, default=16)
    p.add_argument("--preprocessed-cache-dir")
    p.add_argument("--hessian-progress", action="store_true")
    p.set_defaults(func=convert)
    q = sub.add_parser("qdq")
    q.add_argument("input", type=Path, help="Cactus `.weights` directory or tar archive")
    q.add_argument("--out", type=Path, required=True, help="Output HF-style QDQ checkpoint directory")
    q.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    q.add_argument("--model-family", default="auto", choices=sorted(SUPPORTED_FAMILIES))
    q.add_argument("--shard-size-gb", type=float, default=4.0)
    q.add_argument("--row-batch-size", type=int, default=2048)
    q.add_argument("--tmp-dir", type=Path)
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=lambda args: print(f"done: wrote {convert_qdq(args)['written_count']} tensors"))
    v = sub.add_parser("validate")
    v.add_argument("--source-model", required=True)
    v.add_argument("--qdq", required=True)
    v.add_argument("--out")
    v.add_argument("--strict", action="store_true")
    v.add_argument("--model-family", default="auto", choices=sorted(SUPPORTED_FAMILIES))
    v.set_defaults(func=validate_qdq)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
