from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from ..cactus_adapters.config_utils import prompt_dim_from_dictionary


_NEMO_EXPORT_FILES = {
    "model_config.yaml",
    "model_weights.ckpt",
    "vocab.txt",
}


def ensure_nemo_asr_source(
    model_id_or_path: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> str | None:
    nemo_path = _find_single_nemo(model_id_or_path, token=token, cache_dir=cache_dir)
    if nemo_path is None:
        return None

    out = nemo_path.parent / f"{nemo_path.stem}-cactus-hf"
    if (
        (out / "config.json").exists()
        and (out / "pytorch_model.bin").exists()
        and (out / "vocab.txt").exists()
        and (out / "tokenizer_config.txt").exists()
    ):
        return str(out)

    tmp = out.with_name(f"{out.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    with tarfile.open(nemo_path, "r:*") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if not _keep_nemo_member(name):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted, (tmp / name).open("wb") as dst:
                shutil.copyfileobj(extracted, dst, length=1024 * 1024)

    weights = tmp / "model_weights.ckpt"
    config = _read_yaml(tmp / "model_config.yaml")
    if not weights.exists() or not config:
        raise RuntimeError(f"{nemo_path} is missing model_config.yaml or model_weights.ckpt")

    hf_config = _asr_config_from_nemo(config, model_id_or_path=model_id_or_path)
    (tmp / "config.json").write_text(json.dumps(hf_config, indent=2, sort_keys=True), encoding="utf-8")
    weights.rename(tmp / "pytorch_model.bin")
    _write_asr_tokenizer_files(tmp, hf_config)

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return str(out)


def ensure_parakeet_tdt_nemo_source(
    model_id_or_path: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> str | None:
    return ensure_nemo_asr_source(model_id_or_path, token=token, cache_dir=cache_dir)


def _find_single_nemo(model_id_or_path: str, *, token: str | None, cache_dir: str | None) -> Path | None:
    path = Path(model_id_or_path)
    if path.suffix == ".nemo" and path.is_file():
        return path
    if path.is_dir():
        files = sorted(path.glob("*.nemo"))
        return files[0] if len(files) == 1 else None

    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [name for name in list_repo_files(model_id_or_path, token=token) if name.endswith(".nemo")]
        if len(files) != 1:
            return None
        return Path(hf_hub_download(model_id_or_path, files[0], token=token, cache_dir=cache_dir))
    except Exception:
        return None


def _keep_nemo_member(name: str) -> bool:
    return (
        name in _NEMO_EXPORT_FILES
        or name.endswith("_vocab.txt")
        or name.endswith("_tokenizer.model")
        or name.endswith("_tokenizer.vocab")
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to import NeMo ASR .nemo checkpoints") from exc
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def _asr_config_from_nemo(root: dict[str, Any], *, model_id_or_path: str = "") -> dict[str, Any]:
    if _is_nemotron_asr_config(root, model_id_or_path=model_id_or_path):
        return _nemotron_asr_config_from_nemo(root)
    return _parakeet_tdt_config_from_nemo(root)


def _is_nemotron_asr_config(root: dict[str, Any], *, model_id_or_path: str = "") -> bool:
    lowered_id = model_id_or_path.lower()
    target = str(_get(root, "target", "") or "").lower()
    model_defaults = _get(root, "model_defaults", {}) or {}
    has_prompt = (
        "nemotron" in lowered_id
        or "withprompt" in target
        or "prompt" in target
        or bool(_get(model_defaults, "prompt_dictionary", None))
        or int(_get(root, "num_prompts", _get(model_defaults, "num_prompts", 0)) or 0) > 0
    )
    loss = _get(root, "loss", {}) or {}
    decoding = _get(root, "decoding", {}) or {}
    is_tdt = (
        str(_get(loss, "loss_name", "") or "").lower() == "tdt"
        or str(_get(decoding, "model_type", "") or "").lower() == "tdt"
        or bool(_get(model_defaults, "tdt_durations", None))
    )
    return has_prompt and not is_tdt


def _parakeet_tdt_config_from_nemo(root: dict[str, Any]) -> dict[str, Any]:
    decoder = _get(root, "decoder", {}) or {}
    prednet = _get(decoder, "prednet", _get(decoder, "prediction", {})) or {}
    joint = _get(root, "joint", {}) or {}
    labels = [str(token) for token in (_get(joint, "vocabulary", []) or [])]

    decoder_vocab = int(_get(decoder, "vocab_size", len(labels)))

    config = dict(root)
    config.update(architectures=["ParakeetForTDT"], model_type="parakeet_tdt", labels=labels)
    config["decoder"] = {**decoder, "prediction": dict(prednet), "vocab_size": decoder_vocab}
    config["joint"] = {**joint, "vocabulary": labels}
    return config


def _nemotron_asr_config_from_nemo(root: dict[str, Any]) -> dict[str, Any]:
    decoder = _get(root, "decoder", {}) or {}
    prednet = _get(decoder, "prednet", _get(decoder, "prediction", {})) or {}
    joint = _get(root, "joint", {}) or {}
    jointnet = _get(joint, "jointnet", {}) or {}
    model_defaults = _get(root, "model_defaults", {}) or {}
    labels = [str(token) for token in (_get(joint, "vocabulary", _get(root, "labels", [])) or [])]
    decoder_vocab = int(_get(decoder, "vocab_size", len(labels)))
    prompt_dictionary = _get(model_defaults, "prompt_dictionary", {}) or {}
    prompt_dim = prompt_dim_from_dictionary(
        prompt_dictionary,
        _get(root, "prompt_dim", 0),
        _get(root, "num_prompts", 0),
        _get(model_defaults, "num_prompts", 0),
    )
    default_prompt_id = int(prompt_dictionary.get("auto", 0)) if isinstance(prompt_dictionary, dict) else 0

    config = dict(root)
    config.update(
        architectures=["NemotronASRForRNNT"],
        model_type="nemotron_asr",
        labels=labels,
        prompt_dictionary={str(k): int(v) for k, v in prompt_dictionary.items()} if isinstance(prompt_dictionary, dict) else {},
        prompt_dim=prompt_dim,
        default_prompt_id=default_prompt_id,
        rnnt_blank_id=decoder_vocab,
    )
    config["decoder"] = {**decoder, "prediction": dict(prednet), "vocab_size": decoder_vocab}
    config["joint"] = {**joint, "jointnet": dict(jointnet), "vocabulary": labels}
    return config


def _write_asr_tokenizer_files(root: Path, config: dict[str, Any]) -> None:
    tokenizer_model = next(iter(sorted(root.glob("*_tokenizer.model"))), None)
    if tokenizer_model is not None and not (root / "tokenizer.model").exists():
        tokenizer_model.rename(root / "tokenizer.model")

    labels = config.get("labels")
    if isinstance(labels, list) and labels:
        with (root / "vocab.txt").open("w", encoding="utf-8") as f:
            for idx, token in enumerate(labels):
                f.write(f"{idx}\t{token}\n")
    else:
        nemo_vocab = next(iter(sorted(root.glob("*_vocab.txt"))), None)
        if nemo_vocab is not None and not (root / "vocab.txt").exists():
            nemo_vocab.rename(root / "vocab.txt")

    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (root / "tokenizer_config.txt").write_text(
        "\n".join((
            "tokenizer_type=bpe",
            "vocab_format=id_tab_token",
            "normalizer=metaspace",
            "decoder=replace_metaspace",
            "byte_fallback=false",
            "",
        )),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({
            "model_type": config.get("model_type", "parakeet_tdt"),
            "tokenizer_class": "SentencePieceProcessor",
            "tokenizer_type": "bpe",
            "vocab_format": "id_tab_token",
            "normalizer": "metaspace",
            "decoder": "replace_metaspace",
        }, indent=2),
        encoding="utf-8",
    )
