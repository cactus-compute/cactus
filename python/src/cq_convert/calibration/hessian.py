from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class HessianStats:
    hessians: dict[str, object] = field(default_factory=dict)
    diag: dict[str, object] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    unresolved_targets: list[str] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    error_samples: list[dict[str, str]] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def _flatten_inputs(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x.reshape(-1, x.shape[-1]).detach().float()


def collect_text_hessians(model, tokenizer, texts: list[str], target_names: set[str], device: str, max_length: int = 512) -> HessianStats:
    if torch is None:
        return HessianStats()
    stats = HessianStats()
    handles = []
    modules = dict(model.named_modules())
    for name in target_names:
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            continue

        def hook(_module, inputs, _name=name):
            x = _flatten_inputs(inputs)
            if x.numel() == 0:
                return
            h = (x.T @ x).detach().to("cpu")
            if _name not in stats.hessians:
                stats.hessians[_name] = h
                stats.diag[_name] = torch.diag(h).clone()
                stats.samples[_name] = int(x.shape[0])
            else:
                stats.hessians[_name] += h
                stats.diag[_name] += torch.diag(h)
                stats.samples[_name] += int(x.shape[0])

        handles.append(module.register_forward_pre_hook(hook))
    if not handles or not texts:
        return stats
    model.eval()
    model.to(device)
    with torch.no_grad():
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                model(**batch)
            except Exception:
                continue
    for h in handles:
        h.remove()
    for name, count in list(stats.samples.items()):
        if count > 0:
            stats.hessians[name] = stats.hessians[name] / float(count)
            stats.diag[name] = stats.diag[name] / float(count)
    return stats


def _load_audio_16k(path: Path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != 16000:
        old_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        new_len = max(1, int(round(audio.shape[0] * 16000 / sr)))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(new_x, old_x, audio).astype(np.float32)
    return audio


def _read_jsonl(path: str | Path, limit: int | None):
    import json

    if limit is not None and limit <= 0:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _row_language_text(row: dict) -> str:
    if "messages" in row:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in row["messages"])
    if "prompt_text" in row or "completion_text" in row:
        return (row.get("prompt_text", "") + row.get("completion_text", "")).strip()
    return (row.get("prompt", "") + row.get("completion", "")).strip()


def _register_hooks(model, target_names: set[str]) -> tuple[HessianStats, list, dict[str, object]]:
    stats = HessianStats()
    handles = []
    gpu_accum: dict[str, object] = {}
    modules = dict(model.named_modules())

    def add_cpu_hessian(name: str, h):
        h = h.detach().to("cpu")
        if name not in stats.hessians:
            stats.hessians[name] = h
            stats.diag[name] = torch.diag(h).clone()
        else:
            stats.hessians[name] += h
            stats.diag[name] += torch.diag(h)

    for name in target_names:
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            stats.unresolved_targets.append(name)
            continue

        def hook(_module, inputs, _name=name):
            x = _flatten_inputs(inputs)
            if x.numel() == 0:
                return
            h = (x.T @ x).detach()
            if h.is_cuda:
                if _name not in gpu_accum:
                    gpu_accum[_name] = h
                else:
                    gpu_accum[_name] += h
            else:
                add_cpu_hessian(_name, h)
            stats.samples[_name] = stats.samples.get(_name, 0) + int(x.shape[0])

        handles.append(module.register_forward_pre_hook(hook))
    return stats, handles, {"gpu_accum": gpu_accum, "add_cpu_hessian": add_cpu_hessian}


def _record_error(stats: HessianStats, exc: Exception, context: str) -> None:
    key = type(exc).__name__
    stats.errors[key] = stats.errors.get(key, 0) + 1
    if len(stats.error_samples) < 5:
        stats.error_samples.append({"context": context, "type": key, "message": str(exc)[:500]})


def collect_manifest_hessians(
    model,
    processor,
    manifest: dict,
    limits: dict[str, int | None],
    target_names: set[str],
    device: str,
    max_length: int = 1024,
    adapter=None,
    batch_size: int = 1,
    gpu_flush_interval: int = 16,
    progress_path: str | Path | None = None,
    preprocessed_cache_dir: str | Path | None = None,
) -> HessianStats:
    if torch is None:
        return HessianStats()
    started = time.perf_counter()
    stats, handles, hook_state = _register_hooks(model, target_names)
    if not handles:
        return stats
    model.eval()
    model.to(device)

    tokenizer = getattr(processor, "tokenizer", processor)
    batch_size = max(1, int(batch_size or 1))
    gpu_flush_interval = max(1, int(gpu_flush_interval or 1))
    progress_file = Path(progress_path) if progress_path else None
    cache_dir = Path(preprocessed_cache_dir) if preprocessed_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    forwards = 0

    def add_time(name: str, delta: float):
        stats.timings[name] = stats.timings.get(name, 0.0) + float(delta)

    def flush_gpu():
        gpu_accum = hook_state["gpu_accum"]
        add_cpu_hessian = hook_state["add_cpu_hessian"]
        if not gpu_accum:
            return
        t0 = time.perf_counter()
        for name, h in list(gpu_accum.items()):
            add_cpu_hessian(name, h)
        gpu_accum.clear()
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        add_time("hessian_gpu_flush_seconds", time.perf_counter() - t0)

    def write_progress():
        if progress_file is None:
            return
        payload = {
            "elapsed_seconds": time.perf_counter() - started,
            "forwards": forwards,
            "rows": dict(stats.rows),
            "target_modules": len(target_names),
            "resolved_modules": len(handles),
            "modules_with_samples": len(stats.samples),
            "sample_counts_min": min(stats.samples.values()) if stats.samples else 0,
            "sample_counts_max": max(stats.samples.values()) if stats.samples else 0,
            "unresolved_targets": len(stats.unresolved_targets),
            "errors": dict(stats.errors),
            "timings": dict(stats.timings),
        }
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def maybe_flush():
        nonlocal forwards
        forwards += 1
        if forwards % gpu_flush_interval == 0:
            flush_gpu()
            write_progress()

    def cache_key(kind: str, source: str, payload: object) -> Path | None:
        if cache_dir is None:
            return None
        raw = json.dumps({"kind": kind, "source": source, "payload": payload}, sort_keys=True, default=str).encode("utf-8")
        return cache_dir / f"{hashlib.sha1(raw).hexdigest()}.pt"

    def tensor_cache_load(path: Path | None):
        if path is None or not path.exists():
            return None
        t0 = time.perf_counter()
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        add_time("preprocess_cache_read_seconds", time.perf_counter() - t0)
        return obj

    def tensor_cache_save(path: Path | None, obj) -> None:
        if path is None or path.exists():
            return
        t0 = time.perf_counter()
        torch.save(obj, path)
        add_time("preprocess_cache_write_seconds", time.perf_counter() - t0)

    def run_inputs(inputs, context: str):
        nonlocal forwards
        try:
            model_dtype = next(model.parameters()).dtype
        except Exception:
            model_dtype = None
        t0 = time.perf_counter()
        inputs = {
            k: (
                v.to(device=device, dtype=model_dtype)
                if model_dtype is not None and torch.is_tensor(v) and v.is_floating_point()
                else (v.to(device) if hasattr(v, "to") else v)
            )
            for k, v in inputs.items()
        }
        add_time("input_to_device_seconds", time.perf_counter() - t0)
        with torch.no_grad():
            try:
                t0 = time.perf_counter()
                model(**inputs)
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                add_time(f"{context}_forward_seconds", time.perf_counter() - t0)
                maybe_flush()
            except Exception as exc:
                _record_error(stats, exc, context)
                return

    def run_text_batch(texts: list[str]):
        if not texts:
            return
        stats.rows["language"] = stats.rows.get("language", 0) + len(texts)
        key = cache_key("language", "tokenizer", texts)
        inputs = tensor_cache_load(key)
        if inputs is None:
            t0 = time.perf_counter()
            try:
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            except Exception:
                if len(texts) == 1:
                    inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=max_length)
                    add_time("language_preprocess_seconds", time.perf_counter() - t0)
                    tensor_cache_save(key, inputs)
                    run_inputs(inputs, "language")
                    return
                for text in texts:
                    run_text_batch([text])
                return
            add_time("language_preprocess_seconds", time.perf_counter() - t0)
            tensor_cache_save(key, inputs)
        run_inputs(inputs, "language")

    try:
        lang = manifest.get("language")
        if lang:
            path = lang["path"] if isinstance(lang, dict) else lang
            text_batch: list[str] = []
            for row in _read_jsonl(path, limits.get("language")):
                text = _row_language_text(row)
                if text:
                    text_batch.append(text)
                    if len(text_batch) >= batch_size:
                        run_text_batch(text_batch)
                        text_batch = []
                    continue
                if text_batch:
                    run_text_batch(text_batch)
                    text_batch = []
                ids = row.get("full_token_ids") or ((row.get("prompt_token_ids") or []) + (row.get("completion_token_ids") or []))
                if ids:
                    stats.rows["language_ids"] = stats.rows.get("language_ids", 0) + 1
                    run_inputs({"input_ids": torch.tensor([ids[:max_length]], dtype=torch.long)}, "language_ids")
            if text_batch:
                run_text_batch(text_batch)

        vision = manifest.get("vision")
        if vision and hasattr(processor, "apply_chat_template"):
            from PIL import Image

            base = Path(vision.get("base_dir", "."))
            vision_batch = []
            vision_keys = []

            def run_vision_batch(batch, keys):
                if not batch:
                    return
                stats.rows["vision"] = stats.rows.get("vision", 0) + len(batch)
                key = cache_key("vision", "processor", keys)
                inputs = tensor_cache_load(key)
                if inputs is None:
                    t0 = time.perf_counter()
                    try:
                        inputs = processor.apply_chat_template(batch, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                    except Exception:
                        if len(batch) == 1:
                            raise
                        stats.rows["vision"] -= len(batch)
                        for item, item_key in zip(batch, keys):
                            run_vision_batch([item], [item_key])
                        return
                    add_time("vision_preprocess_seconds", time.perf_counter() - t0)
                    tensor_cache_save(key, inputs)
                run_inputs(inputs, "vision")

            for row in _read_jsonl(vision["path"], limits.get("vision")):
                image_rel = row.get("image_path") or (row.get("image_paths") or [None])[0]
                if not image_rel:
                    continue
                image_path = base / image_rel
                if not image_path.exists():
                    continue
                image = Image.open(image_path).convert("RGB")
                prompt = row.get("prompt_text", row.get("prompt", ""))
                vision_batch.append([{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}])
                vision_keys.append({"image_path": str(image_path), "prompt": prompt})
                if len(vision_batch) >= batch_size:
                    run_vision_batch(vision_batch, vision_keys)
                    vision_batch = []
                    vision_keys = []
            if vision_batch:
                run_vision_batch(vision_batch, vision_keys)

        audio = manifest.get("audio")
        if audio and hasattr(processor, "apply_chat_template"):
            base = Path(audio.get("base_dir", "."))
            for row in _read_jsonl(audio["path"], limits.get("audio")):
                audio_rel = row.get("audio_path")
                if not audio_rel:
                    continue
                audio_path = base / audio_rel
                if not audio_path.exists():
                    continue
                audio_arr = _load_audio_16k(audio_path)
                messages = [[{"role": "user", "content": [{"type": "audio", "audio": audio_arr}, {"type": "text", "text": row.get("prompt_text", row.get("prompt", ""))}]}]]
                stats.rows["audio"] = stats.rows.get("audio", 0) + 1
                run_inputs(processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"), "audio")

        transcription = manifest.get("transcription")
        if transcription:
            base = Path(transcription.get("base_dir", "."))
            for row in _read_jsonl(transcription["path"], limits.get("transcription")):
                inputs = None
                if adapter is not None:
                    try:
                        inputs = adapter.build_calibration_inputs(row, processor, "transcription", base)
                    except Exception as exc:
                        _record_error(stats, exc, "transcription_builder")
                        inputs = None
                if inputs is None and hasattr(processor, "apply_chat_template"):
                    audio_rel = row.get("audio_path")
                    if not audio_rel:
                        continue
                    audio_path = base / audio_rel
                    if not audio_path.exists():
                        continue
                    audio_arr = _load_audio_16k(audio_path)
                    messages = [[{"role": "user", "content": [{"type": "audio", "audio": audio_arr}, {"type": "text", "text": row.get("prompt_text", row.get("prompt", ""))}]}]]
                    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                if inputs is not None:
                    stats.rows["transcription"] = stats.rows.get("transcription", 0) + 1
                    run_inputs(inputs, "transcription")
    finally:
        flush_gpu()
        write_progress()
        for h in handles:
            h.remove()

    for name, count in list(stats.samples.items()):
        if count > 0:
            stats.hessians[name] = stats.hessians[name] / float(count)
            stats.diag[name] = stats.diag[name] / float(count)
    stats.timings["total_seconds"] = time.perf_counter() - started
    write_progress()
    return stats
