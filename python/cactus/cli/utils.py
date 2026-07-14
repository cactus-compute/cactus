from __future__ import annotations

import argparse
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


CQ_VARIANT_RE = re.compile(r"-cq(\d+(?:\.\d+)?)$", re.IGNORECASE)
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")

ALLOWED_BITS = (1, 2, 3, 4, 2.54, 3.26)


def normalize_bits(bits: str | int | float) -> int | float:
    value = float(bits)
    return int(value) if value == int(value) else value


def variant_suffix(bits: int | float) -> str:
    return f"cq{normalize_bits(bits)}"


def bits_arg(value: str) -> int | float:
    try:
        bits = normalize_bits(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid --bits value {value!r}")
    if bits not in ALLOWED_BITS:
        allowed = ", ".join(str(b) for b in ALLOWED_BITS)
        raise argparse.ArgumentTypeError(f"unsupported --bits {value!r}; choose from: {allowed}")
    return bits


@dataclass(frozen=True)
class CqArchive:
    filename: str
    bits: int | float
    size: int | None = None
    sha256: str | None = None

    @property
    def suffix(self) -> str:
        return variant_suffix(self.bits)


@dataclass(frozen=True)
class CqResolution:
    repo_id: str
    local_name: str
    archive: CqArchive
    available: tuple[CqArchive, ...]


def suggested_cq_repo(model_id: str) -> str:
    name = str(model_id).strip().replace("_", "-").split("/")[-1]
    name = re.sub(r"-cq\d*(?:\.\d+)?$", "", name, flags=re.IGNORECASE)
    return f"Cactus-Compute/{name}"


def archive_stem(filename: str) -> str:
    for suffix in ARCHIVE_SUFFIXES:
        if filename.lower().endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def parse_cq_variant(filename: str) -> int | float | None:
    stem = archive_stem(Path(filename).name)
    m = CQ_VARIANT_RE.search(stem)
    if not m:
        return None
    return normalize_bits(m.group(1))


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
        bits = parse_cq_variant(filename)
        if bits is None:
            continue
        archives.append(CqArchive(filename=filename, bits=bits,
                                  size=sizes.get(filename), sha256=sha256s.get(filename)))
    return tuple(sorted(archives, key=lambda a: (isinstance(a.bits, float), a.bits)))


VERSION_TAG_RE = re.compile(r"^v?(\d+(?:\.\d+){0,2})$", re.IGNORECASE)


def parse_version(text: str | None) -> tuple[int, int, int] | None:
    if not text:
        return None
    m = VERSION_TAG_RE.match(str(text).strip())
    if not m:
        return None
    parts = [int(p) for p in m.group(1).split(".")]
    parts += [0] * (3 - len(parts))
    return tuple(parts[:3])


def list_version_tags(repo_id: str, *, token=None) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    try:
        from huggingface_hub import HfApi
        refs = HfApi().list_repo_refs(repo_id, token=token)
        names = [t.name for t in refs.tags]
    except ImportError:
        quoted = urllib.parse.quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/models/{quoted}/refs"
        headers = {"User-Agent": "cactus-cq-downloader"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
        names = [t.get("name", "") for t in data.get("tags", [])]
    out = []
    for name in names:
        version = parse_version(name)
        if version is not None:
            out.append((name, version))
    return tuple(out)


def resolve_weight_revision(repo_id: str, runtime_version: str | None = None,
                            *, token=None) -> str | None:
    if runtime_version is None:
        from .. import __version__ as runtime_version
    runtime = parse_version(runtime_version)
    if runtime is None:
        return None
    try:
        tags = list_version_tags(repo_id, token=token)
    except Exception:
        return None
    eligible = [(name, version) for name, version in tags if version <= runtime]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item[1])[0]


def resolve_archive(repo_id: str, local_name: str, archives: Iterable[CqArchive],
                    bits: int | float) -> CqResolution:
    available = tuple(archives)
    if not available:
        raise RuntimeError(f"No CQ archives found in {repo_id}")

    wanted = variant_suffix(bits)
    match = next((a for a in available if a.bits == bits), None)
    if not match:
        choices = ", ".join(a.suffix for a in available)
        raise RuntimeError(f"{wanted} not found in {repo_id}. Available: {choices}")

    return CqResolution(
        repo_id=repo_id,
        local_name=local_name,
        archive=match,
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


def validate_extracted_bundle(output_dir: Path) -> None:
    required = (
        "config.txt",
        "vocab.txt",
        "components/manifest.json",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Downloaded bundle is missing required file(s): {', '.join(missing)}")

    if not any(output_dir.rglob("*.weights")):
        raise RuntimeError("Downloaded bundle contains no weight files")

    tokenizer_config_txt = output_dir / "tokenizer_config.txt"
    if not tokenizer_config_txt.exists():
        return

    tokenizer_config = {}
    for line in tokenizer_config_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        tokenizer_config[key.strip()] = value.strip()

    tokenizer_type = tokenizer_config.get("tokenizer_type", "").lower()
    optional_required = ["special_tokens.json"]
    if tokenizer_type == "sentencepiece":
        optional_required.append("merges.txt")
    else:
        optional_required.append("tokenizer.json")
        if tokenizer_type == "bpe":
            optional_required.append("merges.txt")

    missing_sidecars = [name for name in optional_required if not (output_dir / name).exists()]
    if missing_sidecars:
        raise RuntimeError(f"Downloaded bundle is missing tokenizer sidecar file(s): {', '.join(missing_sidecars)}")


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


def _download_with_urllib(resolution: CqResolution, *, token=None, revision=None) -> Path:
    quoted_repo = urllib.parse.quote(resolution.repo_id, safe="")
    quoted_file = urllib.parse.quote(resolution.archive.filename)
    revision = revision or "main"
    url = f"https://huggingface.co/{resolution.repo_id}/resolve/{urllib.parse.quote(str(revision), safe='')}/{quoted_file}"

    cache_root = Path(tempfile.gettempdir()) / "cactus-cq-cache"
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
        "bits": resolution.archive.bits,
        "archive_size": resolution.archive.size if resolution.archive.size is not None else archive_path.stat().st_size,
        "archive_sha256": resolution.archive.sha256,
    }
    if metadata["archive_sha256"] is None and os.getenv("CACTUS_CQ_HASH_DOWNLOAD", "") == "1":
        metadata["archive_sha256"] = sha256_file(archive_path)
    (output_dir / ".cactus_cq_download.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def download_cq_archive(resolution: CqResolution, output_dir: Path, *, token=None,
                        revision=None) -> Path:

    if output_dir.exists() and (output_dir / "components" / "manifest.json").exists():
        return output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not a complete bundle: {output_dir}")

    try:
        from huggingface_hub import hf_hub_download
        archive_path = Path(hf_hub_download(
            repo_id=resolution.repo_id,
            filename=resolution.archive.filename,
            repo_type="model",
            token=token,
            revision=revision,
        ))
    except ImportError:
        archive_path = _download_with_urllib(
            resolution,
            token=token,
            revision=revision,
        )
    verify_archive_sha256(archive_path, resolution.archive.sha256)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)) as tmp:
        tmp_dir = Path(tmp)
        safe_extract_archive(archive_path, tmp_dir)
        promote_single_root(tmp_dir)
        validate_extracted_bundle(tmp_dir)
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

    try:
        info = HfApi().model_info(repo_id, revision=revision, token=token, files_metadata=True)
    except Exception as exc:
        raise RuntimeError(f"could not query {repo_id} on huggingface.co: {exc}") from exc
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

