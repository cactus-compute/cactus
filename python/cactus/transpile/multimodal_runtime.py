from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cactus.transpile.audio_preprocess import audio_duration_limit_seconds
from cactus.transpile.audio_preprocess import prepare_native_gemma4_audio_features
from cactus.transpile.media_limits import resize_static_image
from cactus.transpile.runtime_support import ensure_transformers_supports_gemma4
from cactus.transpile.runtime_support import patch_torch_flex_attention_compat
from cactus.transpile.runtime_support import patch_transformers_torchvision_probe
from cactus.transpile.runtime_support import PreparedInputs


def _resolve_audio_sample_rate(processor: object) -> int:
    for attr_name in ("feature_extractor", "tokenizer"):
        child = getattr(processor, attr_name, None)
        sample_rate = getattr(child, "sampling_rate", None)
        if isinstance(sample_rate, int) and sample_rate > 0:
            return sample_rate
    sample_rate = getattr(processor, "sampling_rate", None)
    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate
    return 16000


_GEMMA4_MULTIMODAL_INPUT_ORDER = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "pixel_values",
    "pixel_position_ids",
    "input_features",
    "input_features_mask",
)


def _normalize_multimodal_prompt(
    prompt: str,
    *,
    image_token: str | None,
    num_images: int,
    audio_token: str | None,
    num_audio_segments: int,
) -> str:
    normalized = prompt.strip()
    prefixes: list[str] = []

    if image_token and num_images > 0:
        image_count = normalized.count(image_token)
        if image_count < num_images:
            prefixes.append(" ".join(image_token for _ in range(num_images - image_count)))
    if audio_token and num_audio_segments > 0:
        audio_count = normalized.count(audio_token)
        if audio_count < num_audio_segments:
            prefixes.append(" ".join(audio_token for _ in range(num_audio_segments - audio_count)))

    if prefixes:
        prefix = "\n".join(part for part in prefixes if part)
        if normalized:
            return f"{prefix}\n{normalized}"
        return prefix
    return normalized


def _build_gemma4_chat_prompt(
    *,
    prompt: str,
    image_token: str | None,
    num_images: int,
    audio_token: str | None,
    num_audio_segments: int,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
) -> str:
    result = "<bos>"
    normalized_system = system_prompt.strip()
    if enable_thinking_if_supported or normalized_system:
        result += "<|turn>system\n"
        if enable_thinking_if_supported:
            result += "<|think|>"
        result += normalized_system
        result += "<turn|>\n"

    result += "<|turn>user\n"
    if image_token and num_images > 0:
        for _ in range(num_images):
            result += f"\n\n{image_token}\n\n"
    result += prompt
    if audio_token and num_audio_segments > 0:
        result += "".join(audio_token for _ in range(num_audio_segments))
    result += "<turn|>\n"
    result += "<|turn>model\n"
    return result


def _gemma4_split_cactus_newline_token_merges(batch: object) -> None:
    input_ids = batch.get("input_ids") if hasattr(batch, "get") else None
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return

    expansions = {
        108: (107, 107),
        109: (107, 107, 107),
    }
    if not any(int(token) in expansions for token in input_ids.reshape(-1).tolist()):
        return

    lengths: list[list[int]] = []
    max_len = 0
    for row in input_ids.detach().cpu().tolist():
        row_lengths = [len(expansions.get(int(token), (int(token),))) for token in row]
        lengths.append(row_lengths)
        max_len = max(max_len, sum(row_lengths))

    for key in ("input_ids", "attention_mask", "token_type_ids"):
        value = batch.get(key) if hasattr(batch, "get") else None
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        expanded = torch.full(
            (value.shape[0], max_len),
            0,
            dtype=value.dtype,
            device=value.device,
        )
        for row_idx, row in enumerate(value.detach().cpu().tolist()):
            out: list[int] = []
            for token_idx, item in enumerate(row):
                if key == "input_ids":
                    out.extend(expansions.get(int(item), (int(item),)))
                else:
                    out.extend([int(item)] * lengths[row_idx][token_idx])
            expanded[row_idx, : len(out)] = torch.tensor(out, dtype=value.dtype, device=value.device)
        batch[key] = expanded


def _load_image_inputs(image_files: tuple[str, ...]) -> list[object]:
    if not image_files:
        return []
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pillow is required for --image-file: {exc}") from exc

    images: list[object] = []
    for image_file in image_files:
        path = Path(image_file).resolve()
        if not path.exists():
            raise RuntimeError(f"image_file does not exist: {path}")
        with Image.open(path) as image:
            images.append(resize_static_image(image.convert("RGB")).copy())
    return images


def _get_processor_image_attr(processor: object, name: str, default: object) -> object:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, name):
        return getattr(image_processor, name)
    if isinstance(image_processor, dict) and name in image_processor:
        return image_processor[name]
    return default


def _image_channel_array(value: object, default: float) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size >= 3:
            return array[:3]
        if array.size == 1:
            return np.full((3,), float(array[0]), dtype=np.float32)
    if isinstance(value, (int, float)):
        return np.full((3,), float(value), dtype=np.float32)
    return np.full((3,), float(default), dtype=np.float32)


def _prepare_gemma4_native_image_tensors(
    processor: object,
    image_files: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not image_files:
        return None
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pillow is required for Gemma4 native image preprocessing: {exc}") from exc

    patch_size = int(_get_processor_image_attr(processor, "patch_size", 16))
    pooling_kernel_size = int(_get_processor_image_attr(processor, "pooling_kernel_size", 3))
    max_soft_tokens = int(_get_processor_image_attr(processor, "max_soft_tokens", 280))
    rescale_factor = float(_get_processor_image_attr(processor, "rescale_factor", 1.0 / 255.0))
    # Native Cactus Gemma4 uses its config defaults here, even though the HF
    # processor advertises do_normalize=False. Match the C++ path exactly.
    image_mean = np.full((3,), 0.5, dtype=np.float32)
    image_std = np.full((3,), 0.5, dtype=np.float32)
    max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size
    side_multiple = pooling_kernel_size * patch_size
    patch_dim = 3 * patch_size * patch_size
    if patch_size <= 0 or pooling_kernel_size <= 0 or max_patches <= 0:
        return None

    try:
        resample_bilinear = Image.Resampling.BILINEAR
    except AttributeError:  # pragma: no cover
        resample_bilinear = Image.BILINEAR

    pixel_batches: list[np.ndarray] = []
    position_batches: list[np.ndarray] = []
    for image_file in image_files:
        path = Path(image_file).resolve()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            target_pixels = float(max_patches * patch_size * patch_size)
            factor = float(np.sqrt(target_pixels / max(1.0, float(width * height))))
            target_h = int(np.floor(factor * height / side_multiple)) * side_multiple
            target_w = int(np.floor(factor * width / side_multiple)) * side_multiple
            if target_h == 0:
                target_h = side_multiple
            if target_w == 0:
                target_w = side_multiple
            if (target_w, target_h) != rgb.size:
                rgb = rgb.resize((target_w, target_h), resample=resample_bilinear)
            array = np.asarray(rgb, dtype=np.float32) * rescale_factor
            array = (array - image_mean.reshape(1, 1, 3)) / image_std.reshape(1, 1, 3)

        patch_h = target_h // patch_size
        patch_w = target_w // patch_size
        num_patches = patch_h * patch_w
        if num_patches > max_patches:
            raise RuntimeError(
                f"Gemma4 native image preprocessing produced {num_patches} patches, "
                f"but max_patches={max_patches}"
            )
        chw = np.transpose(array, (2, 0, 1))
        patches = (
            chw.reshape(3, patch_h, patch_size, patch_w, patch_size)
            .transpose(1, 3, 2, 4, 0)
            .reshape(num_patches, patch_dim)
        )

        padded_patches = np.zeros((max_patches, patch_dim), dtype=np.float32)
        padded_patches[:num_patches] = patches
        positions = np.full((max_patches, 2), -1, dtype=np.int64)
        valid_positions = np.zeros((num_patches, 2), dtype=np.int64)
        for patch_y in range(patch_h):
            row_start = patch_y * patch_w
            valid_positions[row_start : row_start + patch_w, 0] = np.arange(patch_w, dtype=np.int64)
            valid_positions[row_start : row_start + patch_w, 1] = patch_y
        positions[:num_patches] = valid_positions

        pixel_batches.append(padded_patches)
        position_batches.append(positions)

    return (
        torch.from_numpy(np.stack(pixel_batches, axis=0)),
        torch.from_numpy(np.stack(position_batches, axis=0)),
    )


def _gemma4_image_soft_token_counts(
    pixel_position_ids: torch.Tensor,
    *,
    pooling_kernel_size: int,
) -> list[int]:
    counts: list[int] = []
    for positions in pixel_position_ids:
        valid = positions[(positions != -1).any(dim=-1)]
        if valid.numel() == 0:
            counts.append(0)
            continue
        max_x = int(valid[:, 0].max().item()) + 1
        max_y = int(valid[:, 1].max().item()) + 1
        counts.append((max_x // pooling_kernel_size) * (max_y // pooling_kernel_size))
    return counts


def _gemma4_audio_soft_token_count(num_frames: int) -> int:
    after_stage1 = (int(num_frames) + 1) // 2
    return (after_stage1 + 1) // 2


def _resolve_gemma4_audio_mels(processor: object, default: int = 128) -> int:
    for child_name in ("feature_extractor", "audio_processor", "audio_feature_extractor"):
        child = getattr(processor, child_name, None)
        if child is None:
            continue
        for attr_name in ("feature_size", "num_mel_bins", "n_mels", "input_feat_size"):
            value = getattr(child, attr_name, None)
            if isinstance(value, int) and value > 0:
                return int(value)
    for attr_name in ("feature_size", "num_mel_bins", "n_mels", "input_feat_size"):
        value = getattr(processor, attr_name, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    return default


def _build_gemma4_native_chat_prompt(
    *,
    prompt: str,
    image_soft_token_counts: tuple[int, ...],
    audio_soft_token_count: int,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
) -> str:
    result = "<bos>"
    normalized_system = system_prompt.strip()
    if enable_thinking_if_supported or normalized_system:
        result += "<|turn>system\n"
        if enable_thinking_if_supported:
            result += "<|think|>"
        result += normalized_system
        result += "<turn|>\n"

    result += "<|turn>user\n"
    for count in image_soft_token_counts:
        if count > 0:
            result += "\n\n<|image>"
            result += "<|image|>" * int(count)
            result += "<image|>\n\n"
    result += prompt
    if audio_soft_token_count > 0:
        result += "<|audio>"
        result += "<|audio|>" * int(audio_soft_token_count)
        result += "<audio|>"
    result += "<turn|>\n"
    result += "<|turn>model\n"
    return result


def _tokenize_gemma4_native_prompt(
    processor: object,
    prompt: str,
    *,
    image_token: str | None,
    audio_token: str | None,
) -> dict[str, torch.Tensor]:
    tokenizer = getattr(processor, "tokenizer", processor)
    if not callable(tokenizer):
        raise RuntimeError("Gemma4 native prompt path requires a tokenizer")
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = batch["input_ids"]
    if "attention_mask" not in batch:
        batch["attention_mask"] = torch.ones_like(input_ids)

    token_type_ids = torch.zeros_like(input_ids)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        if image_token:
            image_id = convert(image_token)
            if isinstance(image_id, int) and image_id >= 0:
                token_type_ids = torch.where(input_ids == image_id, torch.ones_like(token_type_ids), token_type_ids)
        if audio_token:
            audio_id = convert(audio_token)
            if isinstance(audio_id, int) and audio_id >= 0:
                token_type_ids = torch.where(input_ids == audio_id, torch.full_like(token_type_ids, 2), token_type_ids)
    batch["token_type_ids"] = token_type_ids
    return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}


def prepare_gemma4_multimodal_inputs(
    processor: object | None,
    *,
    prompt: str,
    image_files: tuple[str, ...],
    audio_file: str | None,
    torch_dtype: torch.dtype,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
    use_gemma4_chat_template: bool = False,
) -> PreparedInputs:
    if processor is None:
        raise RuntimeError("multimodal Gemma4 transpile requires an AutoProcessor with image and audio support")

    images = _load_image_inputs(image_files)
    audio_waveforms: list[np.ndarray] = []
    sample_rate: int | None = None
    if audio_file:
        sample_rate = _resolve_audio_sample_rate(processor)
        from cactus.transpile.audio_preprocess import load_audio_waveform

        audio_waveforms.append(
            load_audio_waveform(
                audio_file,
                target_sample_rate=sample_rate,
                max_seconds=audio_duration_limit_seconds(),
            )
        )

    image_token = getattr(processor, "image_token", None)
    audio_token = getattr(processor, "audio_token", None)
    processor_prompt = _normalize_multimodal_prompt(
        prompt,
        image_token=image_token if isinstance(image_token, str) else None,
        num_images=len(images),
        audio_token=audio_token if isinstance(audio_token, str) else None,
        num_audio_segments=len(audio_waveforms),
    )
    normalized_prompt = processor_prompt
    if use_gemma4_chat_template:
        normalized_prompt = prompt.strip()
        native_image_tensors = _prepare_gemma4_native_image_tensors(processor, image_files)
        native_audio: torch.Tensor | None = None
        native_audio_mask: torch.Tensor | None = None
        native_audio_frames: int | None = None
        if audio_file:
            expected_mels = _resolve_gemma4_audio_mels(processor)
            try:
                native_audio, native_audio_mask, native_audio_frames = prepare_native_gemma4_audio_features(
                    audio_file,
                    expected_mels=expected_mels,
                    torch_dtype=torch_dtype,
                    max_seconds=audio_duration_limit_seconds(),
                    pad_to_max_seconds=True,
                )
            except Exception as exc:
                print(f"note=falling back to processor gemma4 audio features: {exc}")
                native_audio = None
                native_audio_mask = None
                native_audio_frames = None

        if native_image_tensors is not None and (not audio_file or native_audio is not None):
            pixel_values, pixel_position_ids = native_image_tensors
            pooling_kernel_size = int(_get_processor_image_attr(processor, "pooling_kernel_size", 3))
            image_soft_counts = tuple(
                _gemma4_image_soft_token_counts(
                    pixel_position_ids,
                    pooling_kernel_size=pooling_kernel_size,
                )
            )
            audio_soft_count = _gemma4_audio_soft_token_count(native_audio_frames or 0) if audio_file else 0
            processor_prompt = _build_gemma4_native_chat_prompt(
                prompt=normalized_prompt,
                image_soft_token_counts=image_soft_counts,
                audio_soft_token_count=audio_soft_count,
                system_prompt=system_prompt,
                enable_thinking_if_supported=enable_thinking_if_supported,
            )
            batch = _tokenize_gemma4_native_prompt(
                processor,
                processor_prompt,
                image_token=image_token if isinstance(image_token, str) else None,
                audio_token=audio_token if isinstance(audio_token, str) else None,
            )
            batch["pixel_values"] = pixel_values
            batch["pixel_position_ids"] = pixel_position_ids
            if native_audio is not None and native_audio_mask is not None:
                batch["input_features"] = native_audio
                batch["input_features_mask"] = native_audio_mask
                batch["native_audio_frames"] = int(native_audio_frames or native_audio.shape[1])
        else:
            processor_prompt = _build_gemma4_chat_prompt(
                prompt=normalized_prompt,
                image_token=image_token if isinstance(image_token, str) else None,
                num_images=len(images),
                audio_token=audio_token if isinstance(audio_token, str) else None,
                num_audio_segments=len(audio_waveforms),
                system_prompt=system_prompt,
                enable_thinking_if_supported=enable_thinking_if_supported,
            )
            batch = processor(
                text=processor_prompt,
                images=images or None,
                audio=audio_waveforms or None,
                return_tensors="pt",
            )
            _gemma4_split_cactus_newline_token_merges(batch)
            if native_image_tensors is not None:
                batch["pixel_values"], batch["pixel_position_ids"] = native_image_tensors
    else:
        batch = processor(
            text=processor_prompt,
            images=images or None,
            audio=audio_waveforms or None,
            return_tensors="pt",
        )

    ordered_keys = [
        key
        for key in _GEMMA4_MULTIMODAL_INPUT_ORDER
        if isinstance(batch.get(key), torch.Tensor)
    ]
    if not ordered_keys:
        raise RuntimeError("Gemma4 multimodal processor did not return any tensor inputs")

    tensors: list[torch.Tensor] = []
    for key in ordered_keys:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
        tensors.append(value)

    metadata: dict[str, object] = {
        "prompt": normalized_prompt,
        "processor_prompt": processor_prompt,
        "image_files": [str(Path(path).resolve()) for path in image_files],
        "input_shapes": {
            name: list(tensor.shape)
            for name, tensor in zip(ordered_keys, tensors)
        },
    }
    if audio_file:
        metadata["audio_file"] = str(Path(audio_file).resolve())
    if sample_rate is not None:
        metadata["sample_rate"] = sample_rate
    native_audio_frames = batch.get("native_audio_frames")
    if isinstance(native_audio_frames, int):
        metadata["native_audio_frames"] = native_audio_frames

    return PreparedInputs(
        names=tuple(ordered_keys[: len(tensors)]),
        tensors=tuple(tensors),
        metadata=metadata,
    )


_patch_transformers_torchvision_probe = patch_transformers_torchvision_probe
_patch_torch_flex_attention_compat = patch_torch_flex_attention_compat
_prepare_gemma4_multimodal_inputs = prepare_gemma4_multimodal_inputs
