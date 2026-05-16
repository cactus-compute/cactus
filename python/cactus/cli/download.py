"""CQ archive discovery and download helpers.

The resolver in this module is deliberately independent of argparse and the
rest of the CLI. Keep this logic small and data-shaped so other SDKs can port
the same model/repo/combo behavior without inheriting Python CLI details.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODALITIES = ("L", "V", "A")
COMBO_RE = re.compile(r"(?:[LVA][1-4])+", re.IGNORECASE)
COMBO_PART_RE = re.compile(r"([LVA])([1-4])", re.IGNORECASE)
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True)
class CqArchive:
    filename: str
    combo: dict[str, int]
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class CqResolution:
    repo_id: str
    local_name: str
    archive: CqArchive
    requested: dict[str, int]
    warnings: tuple[str, ...]
    available: tuple[CqArchive, ...]


def normalize_model_key(model_id: str) -> str:
    return str(model_id).strip().lower().replace("_", "-")


def default_local_name(model_id: str) -> str:
    name = str(model_id).strip().split("/")[-1].lower().replace("_", "-")
    return name.removesuffix("-cq")


def suggested_cq_repo(model_id: str) -> str:
    raw = str(model_id).strip().replace("_", "-")
    name = raw.split("/")[-1]
    if not name.lower().endswith("-cq"):
        name = f"{name}-cq"
    return f"Cactus-Compute/{name}"


def resolve_cq_repo(model_id: str) -> tuple[str, str]:
    raw = str(model_id).strip()
    key = normalize_model_key(raw)

    if "/" in raw and key.startswith("cactus-compute/") and key.endswith("-cq"):
        return raw, default_local_name(raw)

    raise RuntimeError(
        "download-cq expects a Cactus-Compute CQ repo ending in -cq. "
        f"If you meant to download CQ weights for {raw}, try {suggested_cq_repo(raw)}."
    )


def archive_stem(filename: str) -> str:
    for suffix in ARCHIVE_SUFFIXES:
        if filename.lower().endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def parse_combo(filename: str) -> dict[str, int]:
    stem = archive_stem(Path(filename).name)
    if not COMBO_RE.fullmatch(stem):
        return {}
    combo: dict[str, int] = {}
    seen = set()
    for modality, bits in COMBO_PART_RE.findall(stem):
        modality = modality.upper()
        if modality in seen:
            return {}
        seen.add(modality)
        combo[modality] = int(bits)
    return combo


def is_supported_archive(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def archives_from_repo_files(repo_files: Iterable[str], sizes: dict[str, int] | None = None,
                             sha256s: dict[str, str] | None = None) -> tuple[CqArchive, ...]:
    sizes = sizes or {}
    sha256s = sha256s or {}
    archives = []
    for filename in repo_files:
        if not is_supported_archive(filename):
            continue
        combo = parse_combo(filename)
        if "L" not in combo:
            continue
        archives.append(CqArchive(filename=filename, combo=combo,
                                  size=sizes.get(filename), sha256=sha256s.get(filename)))
    return tuple(sorted(archives, key=lambda a: a.filename))


def combo_label(combo: dict[str, int]) -> str:
    return "".join(f"{m}{combo[m]}" for m in MODALITIES if m in combo)


def resolve_archive(repo_id: str, local_name: str, archives: Iterable[CqArchive],
                    requested: dict[str, int]) -> CqResolution:
    available = tuple(archives)
    if not available:
        raise RuntimeError(f"No CQ archives found in {repo_id}")

    requested = {k.upper(): int(v) for k, v in requested.items() if v is not None}
    repo_modalities = {m for archive in available for m in archive.combo}
    effective = {}
    warnings = []
    for modality in MODALITIES:
        if modality not in requested:
            continue
        if modality not in repo_modalities:
            warnings.append(
                f"Requested {modality}{requested[modality]}, but {repo_id} has no {modality} modality; ignoring it."
            )
            continue
        effective[modality] = requested[modality]

    matches = [
        archive for archive in available
        if all(archive.combo.get(modality) == bits for modality, bits in effective.items())
    ]
    if not matches:
        wanted = combo_label(effective) or combo_label(requested)
        choices = ", ".join(combo_label(a.combo) for a in available)
        raise RuntimeError(f"CQ combo {wanted} not found in {repo_id}. Available combos: {choices}")

    def score(archive: CqArchive) -> tuple[int, int, str]:
        # Prefer archives that do not add unrequested modalities when possible,
        # then prefer higher precision for unspecified modalities.
        extra_modalities = len(set(archive.combo) - set(effective))
        precision_sum = sum(archive.combo.values())
        return (extra_modalities, -precision_sum, archive.filename)

    return CqResolution(
        repo_id=repo_id,
        local_name=local_name,
        archive=sorted(matches, key=score)[0],
        requested=requested,
        warnings=tuple(warnings),
        available=available,
    )


def _safe_zip_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (out_dir / info.filename).resolve()
            if target != out_dir and out_dir not in target.parents:
                raise RuntimeError(f"Unsafe path in zip archive: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode in (0o120000, 0o10000):
                raise RuntimeError(f"Refusing unsafe archive member: {info.filename}")
        zf.extractall(out_dir)


def _safe_tar_extract(tar_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if target != out_dir and out_dir not in target.parents:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing link in tar archive: {member.name}")
        tf.extractall(out_dir)


def safe_extract_archive(archive_path: Path, out_dir: Path) -> None:
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        _safe_zip_extract(archive_path, out_dir)
    elif lower.endswith((".tar", ".tar.gz", ".tgz")):
        _safe_tar_extract(archive_path, out_dir)
    else:
        raise RuntimeError(f"Unsupported CQ archive format: {archive_path.name}")


def promote_single_root(output_dir: Path) -> None:
    if (output_dir / "config.txt").exists():
        return

    children = [path for path in output_dir.iterdir() if path.name != "__MACOSX"]
    if len(children) != 1 or not children[0].is_dir():
        raise RuntimeError("CQ archive must contain config.txt at the archive root or under one top-level directory")

    nested = children[0]
    if not (nested / "config.txt").exists():
        raise RuntimeError("CQ archive has one top-level directory, but it does not contain config.txt")

    for child in nested.iterdir():
        target = output_dir / child.name
        if target.exists():
            raise RuntimeError(f"Cannot promote archive member over existing path: {target}")
        child.rename(target)
    nested.rmdir()


def validate_extracted_cq(output_dir: Path) -> None:
    required = ("config.txt", "token_embeddings.weights", "vocab.txt", "tokenizer_config.txt")
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Downloaded CQ package is missing required file(s): {', '.join(missing)}")

    tokenizer_config = {}
    for line in (output_dir / "tokenizer_config.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        tokenizer_config[key.strip()] = value.strip()

    tokenizer_type = tokenizer_config.get("tokenizer_type", "").lower()
    vocab_format = tokenizer_config.get("vocab_format", "").lower()
    optional_required = ["special_tokens.json", "tokenizer.json"]
    if tokenizer_type == "bpe":
        optional_required.append("merges.txt")

    missing_sidecars = [name for name in optional_required if not (output_dir / name).exists()]
    if missing_sidecars:
        raise RuntimeError(f"Downloaded CQ package is missing tokenizer sidecar file(s): {', '.join(missing_sidecars)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive_sha256(archive_path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    actual = sha256_file(archive_path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Downloaded archive checksum mismatch for {archive_path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )


def _download_with_urllib(resolution: CqResolution, *, token=None, cache_dir=None, revision=None) -> Path:
    quoted_repo = urllib.parse.quote(resolution.repo_id, safe="")
    quoted_file = urllib.parse.quote(resolution.archive.filename)
    revision = revision or "main"
    url = f"https://huggingface.co/{resolution.repo_id}/resolve/{urllib.parse.quote(str(revision), safe='')}/{quoted_file}"

    cache_root = Path(cache_dir).expanduser() if cache_dir else Path(tempfile.gettempdir()) / "cactus-cq-cache"
    archive_dir = cache_root / quoted_repo / str(revision)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / Path(resolution.archive.filename).name
    if archive_path.exists() and (
        resolution.archive.size is None or archive_path.stat().st_size == resolution.archive.size
    ):
        return archive_path

    headers = {"User-Agent": "cactus-cq-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp_path.rename(archive_path)
    return archive_path


def write_download_metadata(output_dir: Path, resolution: CqResolution, archive_path: Path) -> None:
    metadata = {
        "repo_id": resolution.repo_id,
        "local_name": resolution.local_name,
        "archive": resolution.archive.filename,
        "combo": resolution.archive.combo,
        "requested": resolution.requested,
        "warnings": list(resolution.warnings),
        "archive_size": resolution.archive.size if resolution.archive.size is not None else archive_path.stat().st_size,
        "archive_sha256": resolution.archive.sha256,
    }
    if metadata["archive_sha256"] is None and os.getenv("CACTUS_CQ_HASH_DOWNLOAD", "") == "1":
        metadata["archive_sha256"] = sha256_file(archive_path)
    (output_dir / ".cactus_cq_download.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def download_cq_archive(resolution: CqResolution, output_dir: Path, *, token=None,
                        cache_dir=None, revision=None, force=False, dry_run=False) -> Path:
    if dry_run:
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if output_dir.exists() and (output_dir / "config.txt").exists():
        return output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not a complete CQ package: {output_dir}")

    try:
        from huggingface_hub import hf_hub_download
        archive_path = Path(hf_hub_download(
            repo_id=resolution.repo_id,
            filename=resolution.archive.filename,
            repo_type="model",
            token=token,
            cache_dir=cache_dir,
            revision=revision,
        ))
    except ImportError:
        archive_path = _download_with_urllib(
            resolution,
            token=token,
            cache_dir=cache_dir,
            revision=revision,
        )
    verify_archive_sha256(archive_path, resolution.archive.sha256)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)) as tmp:
        tmp_dir = Path(tmp)
        safe_extract_archive(archive_path, tmp_dir)
        promote_single_root(tmp_dir)
        validate_extracted_cq(tmp_dir)
        write_download_metadata(tmp_dir, resolution, archive_path)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_dir.rename(output_dir)

    return output_dir


def list_hf_cq_archives(repo_id: str, *, token=None, revision=None) -> tuple[CqArchive, ...]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        quoted = urllib.parse.quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/models/{quoted}?blobs=true"
        if revision:
            url += f"&revision={urllib.parse.quote(str(revision), safe='')}"
        headers = {"User-Agent": "cactus-cq-downloader"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
        files = []
        sizes = {}
        sha256s = {}
        for sibling in data.get("siblings", []):
            filename = sibling.get("rfilename")
            if not filename:
                continue
            files.append(filename)
            lfs = sibling.get("lfs") or {}
            if "size" in lfs:
                sizes[filename] = int(lfs["size"])
            elif "size" in sibling:
                sizes[filename] = int(sibling["size"])
            if "sha256" in lfs:
                sha256s[filename] = lfs["sha256"]
        return archives_from_repo_files(files, sizes=sizes, sha256s=sha256s)

    info = HfApi().model_info(repo_id, revision=revision, token=token, files_metadata=True)
    files = []
    sizes = {}
    sha256s = {}
    for sibling in info.siblings:
        filename = sibling.rfilename
        files.append(filename)
        size = getattr(sibling, "size", None)
        if size is not None:
            sizes[filename] = int(size)
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            if "sha256" in lfs:
                sha256s[filename] = lfs["sha256"]
            if "size" in lfs:
                sizes[filename] = int(lfs["size"])
    return archives_from_repo_files(files, sizes=sizes, sha256s=sha256s)


# ---------------------------------------------------------------------------
# Path helpers and convenience entry points
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_model_dir_name(model_id: str) -> str:
    """Convert HuggingFace model ID to local directory name."""
    return model_id.split("/")[-1].lower()


def get_weights_dir(model_id: str) -> Path:
    """Return ``<project>/weights/<model_name>``."""
    if "silero-vad" in model_id.lower():
        return _PROJECT_ROOT / "weights" / "silero-vad"
    return _PROJECT_ROOT / "weights" / get_model_dir_name(model_id)


def download_legacy_fp16_zip(model_id: str, weights_dir: Path) -> bool:
    """Download pre-converted FP16 weights zip from Cactus-Compute (legacy fallback).

    Returns True on success, False if unavailable.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        return False

    model_name = get_model_dir_name(model_id)
    repo_id = f"Cactus-Compute/{model_id.split('/')[-1]}"

    try:
        apple_zip = f"{model_name}-fp16-apple.zip"
        standard_zip = f"{model_name}-fp16.zip"
        repo_files = list_repo_files(repo_id, repo_type="model")

        zip_file = None
        if f"weights/{apple_zip}" in repo_files:
            zip_file = apple_zip
        elif f"weights/{standard_zip}" in repo_files:
            zip_file = standard_zip
        else:
            return False

        print(f"Downloading {repo_id}/{zip_file} ...")
        zip_path = hf_hub_download(repo_id=repo_id, filename=f"weights/{zip_file}", repo_type="model")

        weights_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(weights_dir)

        if not (weights_dir / "config.txt").exists():
            if weights_dir.exists():
                shutil.rmtree(weights_dir)
            return False

        print(f"Model ready at {weights_dir}")
        return True
    except Exception:
        if weights_dir.exists():
            shutil.rmtree(weights_dir)
        return False


def ensure_model(model_id: str) -> Path:
    """Return the weights directory, downloading CQ weights if necessary.

    Tries CQ first, falls back to legacy FP16 zip.
    Raises ``RuntimeError`` if the model cannot be obtained.
    """
    weights_dir = get_weights_dir(model_id)
    if (weights_dir / "config.txt").exists():
        return weights_dir

    try:
        cq_repo = suggested_cq_repo(model_id)
        archives = list_hf_cq_archives(cq_repo)
        if archives:
            resolution = resolve_archive(cq_repo, default_local_name(model_id), archives, {"L": 4})
            download_cq_archive(resolution, weights_dir)
            return weights_dir
    except Exception:
        pass

    if not download_legacy_fp16_zip(model_id, weights_dir):
        raise RuntimeError(f"Could not download model {model_id}")
    return weights_dir


def cmd_download(args):
    """Download CQ model weights from Cactus-Compute."""
    from .common import print_color, RED, GREEN, YELLOW, get_effective_weights_dir

    model_id = args.model_id
    weights_dir = get_effective_weights_dir(model_id, args)
    reconvert = getattr(args, 'reconvert', False)
    token = getattr(args, 'token', None)
    cache_dir = getattr(args, 'cache_dir', None)

    if reconvert and weights_dir.exists():
        print_color(YELLOW, f"Removing cached weights for reconversion...")
        shutil.rmtree(weights_dir)

    if weights_dir.exists() and (weights_dir / "config.txt").exists():
        print_color(GREEN, f"Model weights found at {weights_dir}")
        return 0

    print()
    print_color(YELLOW, f"Model weights not found. Downloading {model_id}...")
    print("=" * 45)

    if not reconvert:
        cq_repo_id = model_id if (model_id.lower().endswith('-cq') and '/' in model_id) else suggested_cq_repo(model_id)
        requested = {"L": getattr(args, 'language_bits', 4) or 4}
        vbits = getattr(args, 'vision_bits', None)
        abits = getattr(args, 'audio_bits', None)
        if vbits is not None: requested["V"] = vbits
        if abits is not None: requested["A"] = abits

        try:
            archives = list_hf_cq_archives(cq_repo_id, token=token)
            if archives:
                local_name = get_model_dir_name(model_id)
                resolution = resolve_archive(cq_repo_id, local_name, archives, requested)
                for warning in resolution.warnings:
                    print_color(YELLOW, f"  {warning}")
                size_text = f" ({resolution.archive.size / (1024 * 1024):.1f} MiB)" if resolution.archive.size else ""
                print(f"  Downloading {resolution.archive.filename} [{combo_label(resolution.archive.combo)}]{size_text}")
                download_cq_archive(resolution, weights_dir, token=token, cache_dir=cache_dir)
                print_color(GREEN, f"CQ model ready at {weights_dir}")
                return 0
        except Exception as cq_exc:
            print(f"  CQ weights not available ({cq_exc})")

        print_color(RED, f"No CQ weights found for {model_id}")
        print()
        print("To convert from HuggingFace, re-run with --reconvert:")
        print(f"  cactus download {model_id} --reconvert")
        print()
        print("Or convert directly:")
        print(f"  cactus convert {model_id}")
        return 1

    print_color(YELLOW, f"Converting {model_id} using CQ pipeline...")
    try:
        from ..convert.cli import main as convert_main
        bits = getattr(args, 'language_bits', 4) or 4
        convert_args = ['convert', '--model', model_id, '--out', str(weights_dir), '--bits', str(bits), '--force']
        if token:
            convert_args.extend(['--token', token])
        if cache_dir:
            convert_args.extend(['--cache-dir', cache_dir])
        convert_main(convert_args)
        print_color(GREEN, f"Model converted and ready at {weights_dir}")
        return 0
    except SystemExit as e:
        return e.code if e.code else 0
    except Exception as e:
        print_color(RED, f"Conversion failed: {e}")
        return 1
