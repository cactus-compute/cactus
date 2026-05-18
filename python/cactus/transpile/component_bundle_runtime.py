from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import re
import time

import numpy as np
import torch

from cactus.transpile.audio_preprocess import generic_log_mel_features as _generic_log_mel_features
from cactus.transpile.audio_preprocess import load_audio_waveform as _load_audio_waveform
from cactus.transpile.audio_preprocess import prepare_cactus_audio_features


def _bundle_root_from_manifest(manifest: Mapping[str, object]) -> Path | None:
    raw_root = manifest.get("_bundle_root")
    if not isinstance(raw_root, str) or not raw_root:
        return None
    root = Path(raw_root).expanduser().resolve()
    return root if root.exists() else None


def _looks_like_tokenizer_source(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "sentencepiece.bpe.model",
            "tokenizer.model",
        )
    )


def _looks_like_processor_source(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            "processor_config.json",
            "preprocessor_config.json",
            "image_processor_config.json",
            "feature_extractor_config.json",
        )
    )


def _pretrained_source_candidates(
    manifest: Mapping[str, object],
    *,
    processor: bool,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str) or not value:
            return
        if value in seen:
            return
        seen.add(value)
        candidates.append(value)

    bundle_root = _bundle_root_from_manifest(manifest)
    if bundle_root is not None:
        if processor:
            if _looks_like_processor_source(bundle_root):
                add(str(bundle_root))
        elif _looks_like_tokenizer_source(bundle_root):
            add(str(bundle_root))

    model_source = str(manifest.get("model_source", "") or "")
    if model_source:
        source_path = Path(model_source).expanduser()
        if source_path.exists():
            add(str(source_path.resolve()))
        elif not source_path.is_absolute():
            add(model_source)

    add(manifest.get("model_id"))
    return candidates


def component_input_names(component: object) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(component, "_input_names", ()))


def _run_seq2seq_transcription_bundle(
    *,
    component_graphs: dict[str, object],
    manifest: dict[str, object],
    audio_file: str | Path,
    prompt: str | None,
    torch_dtype: torch.dtype,
    max_new_tokens: int | None,
    stop_sequences: tuple[str, ...],
) -> dict[str, object]:
    if "audio_encoder" not in component_graphs or "decoder" not in component_graphs:
        raise ValueError("seq2seq_transcription bundle must include audio_encoder and decoder graphs")

    inputs_meta = manifest.get("inputs")
    if not isinstance(inputs_meta, dict):
        inputs_meta = {}
    input_shapes = inputs_meta.get("input_shapes") if isinstance(inputs_meta, dict) else {}
    if not isinstance(input_shapes, dict):
        input_shapes = {}
    expected_shape = input_shapes.get("input_features")
    if not (isinstance(expected_shape, list) and len(expected_shape) == 3):
        raise ValueError("seq2seq_transcription bundle manifest is missing inputs.input_shapes.input_features")

    tokenizer = None
    try:
        tokenizer = _load_bundle_tokenizer(manifest)
    except Exception:
        tokenizer = None

    prompt_token_ids = _resolve_seq2seq_prompt_token_ids(
        manifest=manifest,
        prompt=prompt,
        tokenizer=tokenizer,
    )
    if not prompt_token_ids:
        raise ValueError("seq2seq_transcription bundle input token ids are empty")

    _attach_component_io_names(manifest, component_graphs)
    encoder = component_graphs["audio_encoder"]
    decoder = component_graphs["decoder"]
    encoder_inputs = component_input_names(encoder)
    decoder_inputs = component_input_names(decoder)
    if encoder_inputs and encoder_inputs != ("input_features",):
        raise ValueError(
            "seq2seq_transcription audio_encoder must accept logical input ('input_features',), "
            f"got {encoder_inputs!r}"
        )
    if decoder_inputs and decoder_inputs != ("decoder_input_ids", "encoder_hidden_states"):
        raise ValueError(
            "seq2seq_transcription decoder must accept logical inputs "
            "('decoder_input_ids', 'encoder_hidden_states'), "
            f"got {decoder_inputs!r}"
        )

    preprocess_start = time.perf_counter()
    input_features, active_frames = _prepare_generic_audio_encoder_features(
        audio_file=audio_file,
        manifest=manifest,
        expected_shape=expected_shape,
        torch_dtype=torch_dtype,
    )
    preprocess_end = time.perf_counter()

    encoder_start = time.perf_counter()
    encoder.set_inputs([input_features])
    encoder_outputs = encoder.execute()
    encoder_end = time.perf_counter()
    if not encoder_outputs:
        raise RuntimeError("seq2seq_transcription encoder graph produced no outputs")
    encoder_hidden_states = np.asarray(encoder_outputs[0].numpy())

    stored_target_token_count = int(inputs_meta.get("target_token_count", 0) or 0)
    target_token_count = max(stored_target_token_count, len(prompt_token_ids))
    if target_token_count <= 0:
        raise ValueError("seq2seq_transcription bundle manifest did not provide a valid target token count")
    if len(prompt_token_ids) > target_token_count:
        raise ValueError(
            f"prompt token length {len(prompt_token_ids)} exceeds transpiled bundle context {target_token_count}; "
            "re-transpile with a larger --max-new-tokens budget or use a shorter prompt"
        )

    padding_token_id = _resolve_bundle_padding_token_id(inputs_meta, tokenizer)
    input_array = np.full((1, target_token_count), padding_token_id, dtype=np.int64)
    input_array[0, : len(prompt_token_ids)] = np.asarray(prompt_token_ids, dtype=np.int64)
    if hasattr(decoder, "set_external_inputs"):
        bound_decoder_inputs = decoder.set_external_inputs([input_array, encoder_hidden_states])
        input_array = bound_decoder_inputs[0]
        encoder_hidden_states = bound_decoder_inputs[1]
    else:
        decoder.set_inputs([input_array, encoder_hidden_states])

    available_headroom = max(0, target_token_count - len(prompt_token_ids))
    if max_new_tokens is None:
        token_budget = available_headroom if available_headroom > 0 else 1
    else:
        requested = max(0, int(max_new_tokens))
        if available_headroom > 0:
            token_budget = min(requested, available_headroom)
        else:
            token_budget = 1 if requested > 0 else 0

    default_stop_sequences = ("<|endoftext|>", "<|endoftranscript|>", "</s>", "<pad>")
    resolved_stop_sequences = stop_sequences or default_stop_sequences
    encoded_stop_sequences = _encode_stop_sequences(tokenizer, resolved_stop_sequences)
    eos_token_id = inputs_meta.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))
    suppress_tokens = [int(value) for value in inputs_meta.get("suppress_tokens", []) if isinstance(value, int)]
    begin_suppress_tokens = [int(value) for value in inputs_meta.get("begin_suppress_tokens", []) if isinstance(value, int)]
    if tokenizer is not None and (
        "whisper" in str(manifest.get("family", "") or "").lower()
        or "whisper" in str(manifest.get("model_id", "") or "").lower()
    ):
        eos_int = int(eos_token_id) if isinstance(eos_token_id, int) else None
        whisper_special_ids = [
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", []) or []
            if eos_int is None or int(token_id) != eos_int
        ]
        suppress_tokens = sorted(set(suppress_tokens).union(whisper_special_ids))

    generated_ids: list[int] = []
    logits_shape: list[int] | None = None
    current_length = len(prompt_token_ids)
    first_token_ms = 0.0
    stop_reason = "max_new_tokens"
    decoder_start = time.perf_counter()

    for step_index in range(token_budget):
        outputs = decoder.execute()
        if not outputs:
            raise RuntimeError("seq2seq_transcription decoder graph produced no outputs")
        logits = outputs[0].numpy()
        logits_shape = list(logits.shape)
        if logits.ndim != 3:
            raise RuntimeError(f"expected decoder logits with shape [batch, seq, vocab], got {list(logits.shape)}")
        token_position = current_length - 1
        if logits.shape[1] == 1:
            token_position = 0
        next_token_id = _select_next_token_with_suppression(
            np.asarray(logits[0, token_position]),
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens if step_index == 0 else (),
        )
        generated_ids.append(next_token_id)
        if step_index == 0:
            first_token_ms = (time.perf_counter() - decoder_start) * 1000.0

        if eos_token_id is not None and next_token_id == int(eos_token_id):
            stop_reason = "eos_token"
            break
        if _trim_stop_suffix(generated_ids, encoded_stop_sequences):
            stop_reason = "stop_sequence"
            break
        if current_length >= target_token_count:
            stop_reason = "context_limit"
            break
        if step_index + 1 >= token_budget:
            break

        input_array[0, current_length] = next_token_id
        current_length += 1

    decoder_end = time.perf_counter()
    transcript = _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=True).strip()
    if not transcript:
        transcript = _strip_whisper_control_tokens(
            _decode_generated_text(tokenizer, generated_ids, skip_special_tokens=False)
        ).strip()
    decode_time_ms = max(0.0, (decoder_end - decoder_start) * 1000.0 - first_token_ms)
    decode_tps = (
        ((len(generated_ids) - 1) * 1000.0) / decode_time_ms
        if len(generated_ids) > 1 and decode_time_ms > 0.0
        else 0.0
    )

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "audio_file": str(Path(audio_file).expanduser().resolve()),
        "component_order": list(manifest.get("component_order", [])),
        "active_feature_frames": active_frames,
        "input_shape": list(input_features.shape),
        "encoder_hidden_shape": list(encoder_hidden_states.shape),
        "output_shape": logits_shape or [],
        "input_ids": prompt_token_ids,
        "generated_token_ids": generated_ids,
        "transcript": transcript,
        "response": transcript,
        "preprocess_ms": (preprocess_end - preprocess_start) * 1000.0,
        "encoder_ms": (encoder_end - encoder_start) * 1000.0,
        "decoder_ms": (decoder_end - decoder_start) * 1000.0,
        "total_ms": (decoder_end - preprocess_start) * 1000.0,
        "time_to_first_token_ms": first_token_ms,
        "decode_tps": decode_tps,
        "decode_tokens": len(generated_ids),
        "stop_reason": stop_reason,
    }


def _prepare_generic_audio_encoder_features(
    *,
    audio_file: str | Path,
    manifest: dict[str, object],
    expected_shape: list[object],
    torch_dtype: torch.dtype,
) -> tuple[np.ndarray, int]:
    family = str(manifest.get("family", "") or "")
    family_lower = family.strip().lower()
    inputs_meta = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    sample_rate = int(inputs_meta.get("sample_rate", 16000) if isinstance(inputs_meta, dict) else 16000)
    batch = int(expected_shape[0])
    if batch != 1:
        raise ValueError("saved audio encoder bundle runtime currently expects batch size 1")

    if "whisper" in family_lower:
        expected_mels = int(expected_shape[1])
        expected_frames = int(expected_shape[2])
        try:
            features, active_frames = prepare_cactus_audio_features(
                audio_file,
                model_type="whisper",
                expected_frames=expected_frames,
                expected_mels=expected_mels,
                torch_dtype=torch_dtype,
                layout="mels_frames",
            )
            return np.ascontiguousarray(features.detach().cpu().numpy()), active_frames
        except Exception:
            pass
    else:
        expected_frames = int(expected_shape[1])
        expected_mels = int(expected_shape[2])

    waveform = _load_audio_waveform(audio_file, target_sample_rate=sample_rate)
    features, feature_length = _generic_log_mel_features(
        waveform,
        sample_rate=sample_rate,
        num_mels=expected_mels,
        n_fft=400,
        hop_length=160,
        frame_length=400,
        preemphasis=None,
    )
    active_frames = min(feature_length, expected_frames)
    features = features[:active_frames, :]
    if expected_frames > active_frames:
        features = np.pad(features, ((0, expected_frames - active_frames), (0, 0)), mode="constant")
    if "whisper" in family_lower:
        features = np.ascontiguousarray(features.T)
    features = np.ascontiguousarray(features, dtype=np.float16 if torch_dtype == torch.float16 else np.float32)
    return np.expand_dims(features, axis=0), active_frames


def _parse_nested_manifest_input_ids(value: object) -> list[int] | None:
    if isinstance(value, list) and value:
        if all(isinstance(item, int) for item in value):
            return [int(item) for item in value]
        first = value[0]
        if isinstance(first, list) and all(isinstance(item, int) for item in first):
            return [int(item) for item in first]
    return None


def _patch_missing_lzma_backport() -> str | None:
    try:
        import importlib.util
        import sys

        if importlib.util.find_spec("_lzma") is not None:
            return None
        if importlib.util.find_spec("backports.lzma") is None:
            return None
        import backports.lzma as backports_lzma  # type: ignore

        sys.modules.setdefault("lzma", backports_lzma)
        return "using backports.lzma because this Python build is missing _lzma"
    except Exception:
        return None


def _load_bundle_tokenizer(manifest: dict[str, object]):
    _patch_missing_lzma_backport()
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"transformers is required to tokenize --prompt: {exc}") from exc

    tokenizer_sources = _pretrained_source_candidates(manifest, processor=False)
    if not tokenizer_sources:
        raise ValueError("bundle manifest is missing model_source/model_id; provide --input-ids instead")
    errors: list[str] = []
    for source in tokenizer_sources:
        try:
            return AutoTokenizer.from_pretrained(
                source,
                local_files_only=Path(source).exists(),
                trust_remote_code=True,
            )
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    raise RuntimeError(
        "failed to load tokenizer assets for prompt tokenization. "
        "The CQ weights are present, but tokenizer files are also required for text prompts. "
        f"Tried: {'; '.join(errors)}"
    )


def _tokenize_bundle_prompt(
    tokenizer: object,
    prompt: str,
    *,
    enable_thinking_if_supported: bool = False,
) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            encoded = apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                enable_thinking=bool(enable_thinking_if_supported),
            )
            ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(value) for value in ids]
        except Exception:
            pass

    encoded = tokenizer(prompt, return_tensors=None)  # type: ignore[operator]
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _resolve_bundle_padding_token_id(inputs_meta: Mapping[str, object] | None, tokenizer: object | None) -> int:
    if isinstance(inputs_meta, Mapping):
        value = inputs_meta.get("padding_token_id")
        if isinstance(value, int) and value >= 0:
            return int(value)
    for attr_name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr_name, None) if tokenizer is not None else None
        if isinstance(token_id, int) and token_id >= 0:
            return int(token_id)
    return 0


def _encode_stop_sequences(tokenizer: object | None, stop_sequences: tuple[str, ...]) -> list[list[int]]:
    if tokenizer is None or not stop_sequences:
        return []
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return []
    encoded: list[list[int]] = []
    for stop_sequence in stop_sequences:
        try:
            token_ids = list(encode(stop_sequence, add_special_tokens=False))
        except TypeError:
            token_ids = list(encode(stop_sequence))
        if token_ids:
            encoded.append([int(token_id) for token_id in token_ids])
    return encoded


def _has_token_suffix(token_ids: list[int], suffix: list[int]) -> bool:
    if not suffix or len(token_ids) < len(suffix):
        return False
    return token_ids[-len(suffix) :] == suffix


def _trim_stop_suffix(token_ids: list[int], stop_sequences: list[list[int]]) -> bool:
    for stop_sequence in stop_sequences:
        if _has_token_suffix(token_ids, stop_sequence):
            del token_ids[-len(stop_sequence) :]
            return True
    return False


def _decode_generated_text(tokenizer: object | None, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    if tokenizer is None:
        return ""
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ""
    try:
        return str(decode(token_ids, skip_special_tokens=skip_special_tokens))
    except TypeError:
        return str(decode(token_ids))


def _resolve_seq2seq_prompt_token_ids(
    *,
    manifest: dict[str, object],
    prompt: str | None,
    tokenizer: object | None,
) -> list[int]:
    if prompt:
        if tokenizer is None:
            raise ValueError("transformers tokenizer is required when providing --prompt for seq2seq bundles")
        return _tokenize_bundle_prompt(tokenizer, prompt, enable_thinking_if_supported=False)

    inputs_meta = manifest.get("inputs")
    if isinstance(inputs_meta, dict):
        stored_ids = _parse_nested_manifest_input_ids(inputs_meta.get("decoder_input_ids"))
        if stored_ids:
            return stored_ids
        decoder_start_token_id = inputs_meta.get("decoder_start_token_id")
        if isinstance(decoder_start_token_id, int):
            return [int(decoder_start_token_id)]
    return []


def _select_next_token_with_suppression(
    logits: np.ndarray,
    *,
    suppress_tokens: list[int] | tuple[int, ...],
    begin_suppress_tokens: list[int] | tuple[int, ...],
) -> int:
    masked = np.asarray(logits, dtype=np.float32).copy()
    vocab_size = masked.shape[-1]
    for token_id in (*suppress_tokens, *begin_suppress_tokens):
        token_index = int(token_id)
        if 0 <= token_index < vocab_size:
            masked[token_index] = -np.inf
    return int(np.argmax(masked))


def _strip_whisper_control_tokens(text: str) -> str:
    cleaned = re.sub(r"<\|\d+(?:\.\d+)?\|>", " ", text)
    cleaned = re.sub(r"<\|[^|>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _attach_component_io_names(
    manifest: dict[str, object],
    component_graphs: dict[str, object],
) -> None:
    for component_entry in manifest.get("components", []):
        if not isinstance(component_entry, dict):
            continue
        name = str(component_entry.get("component", "")).strip()
        if not name or name not in component_graphs:
            continue
        component = component_graphs[name]
        logical_inputs = tuple(str(value) for value in component_entry.get("logical_inputs", []))
        logical_outputs = tuple(str(value) for value in component_entry.get("logical_outputs", []))
        if not logical_inputs or not logical_outputs:
            raise ValueError(
                f"component bundle manifest is missing logical IO names for component={name!r}"
            )
        component._input_names = logical_inputs
        component._output_names = logical_outputs

