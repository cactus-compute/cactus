from __future__ import annotations

import json
import shutil
from pathlib import Path


TOKENIZER_FILES = [
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
]


def runtime_sidecar_config(out_dir: Path) -> dict:
    config_path = out_dir / "preprocessor_config.json"
    if not config_path.exists():
        config_path = out_dir / "image_processor_config.json"
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result = {}
    for key in (
        "do_image_splitting",
        "do_resize",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
        "downsample_factor",
        "encoder_patch_size",
        "max_image_tokens",
        "max_num_patches",
        "max_pixels_tolerance",
        "max_tiles",
        "min_image_tokens",
        "min_tiles",
        "rescale_factor",
        "tile_size",
        "use_thumbnail",
    ):
        value = data.get(key)
        if value is not None:
            result[key] = value

    for key in ("image_mean", "image_std"):
        value = data.get(key)
        if isinstance(value, list) and value:
            result[key] = value[0]
        elif value is not None:
            result[key] = value

    return result


def copy_runtime_files(
    model_path: str | Path,
    out_dir: Path,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> None:
    src_dir = Path(model_path)
    if not src_dir.exists() or not src_dir.is_dir():
        try:
            from huggingface_hub import snapshot_download

            src_dir = Path(
                snapshot_download(
                    repo_id=str(model_path),
                    allow_patterns=TOKENIZER_FILES,
                    token=token,
                    cache_dir=cache_dir,
                )
            )
        except Exception:
            return
    for name in TOKENIZER_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def write_config_txt(config: dict, out_dir: Path) -> None:
    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (list, tuple)):
            return ",".join(str(x) for x in v)
        return str(v)
    with (out_dir / "config.txt").open("w", encoding="utf-8") as f:
        for key in sorted(config):
            val = config[key]
            if isinstance(val, (str, int, float, bool, list, tuple)):
                f.write(f"{key}={fmt(val)}\n")
    (out_dir / "hf_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
