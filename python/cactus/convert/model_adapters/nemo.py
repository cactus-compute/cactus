from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


_NEMO_EXPORT_FILES = {
    "model_config.yaml",
    "model_weights.ckpt",
    "vocab.txt",
}


def ensure_parakeet_tdt_nemo_source(
    model_id_or_path: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> str | None:
    nemo_path = _find_single_nemo(model_id_or_path, token=token, cache_dir=cache_dir)
    if nemo_path is None:
        return None

    out = nemo_path.parent / f"{nemo_path.stem}-cactus-hf"
    if (out / "config.json").exists() and (out / "pytorch_model.bin").exists() and (out / "vocab.txt").exists():
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

    hf_config = _parakeet_tdt_config_from_nemo(config)
    (tmp / "config.json").write_text(json.dumps(hf_config, indent=2, sort_keys=True), encoding="utf-8")
    weights.rename(tmp / "pytorch_model.bin")
    _write_parakeet_tdt_tokenizer_files(tmp, hf_config)

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return str(out)


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
        raise RuntimeError("PyYAML is required to import Parakeet TDT .nemo checkpoints") from exc
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


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


def _write_parakeet_tdt_tokenizer_files(root: Path, config: dict[str, Any]) -> None:
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

    (root / "tokenizer_config.json").write_text(
        json.dumps({"model_type": "parakeet_tdt", "tokenizer_class": "SentencePieceProcessor"}, indent=2),
        encoding="utf-8",
    )
