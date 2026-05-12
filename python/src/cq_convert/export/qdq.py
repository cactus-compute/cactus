from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from scipy.linalg import hadamard

from ..model_adapters.detection import detect_family
from ..model_adapters.adapters import adapter_for_family
from ..model_adapters.naming import restore_hf_key_for_family

CACTUS_MAGIC = b"CACT"
HEADER_SIZE = 84
ALIGNMENT_DEFAULT = 32
FLAG_ORTHOGONAL_ROTATION = 1 << 1
FLAG_INTERLEAVED = 1 << 3

PRECISION_INT8 = 0
PRECISION_FP16 = 1
PRECISION_FP32 = 2
PRECISION_CQ = {3: 1, 4: 2, 5: 3, 6: 4}

CONFIG_FILES = {
    "config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer_config.txt",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "tokenizer.model",
    "config.txt",
}


@dataclass(frozen=True)
class CactusHeader:
    path: Path
    flags: int
    alignment: int
    ndim: int
    dims: tuple[int, int, int, int]
    precision: int
    data_bytes: int
    scales_bytes: int
    group_size: int
    num_groups: int
    original_n: int

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.dims[: self.ndim])

    @property
    def bits(self) -> int:
        if self.precision not in PRECISION_CQ:
            raise ValueError(f"{self.path}: precision {self.precision} is not CQ")
        return PRECISION_CQ[self.precision]


class ShardedSafetensorsWriter:
    def __init__(self, out_dir: Path, shard_size_bytes: int) -> None:
        self.out_dir = out_dir
        self.shard_size_bytes = int(shard_size_bytes)
        self.current: dict[str, torch.Tensor] = {}
        self.current_bytes = 0
        self.shards: list[tuple[Path, list[str]]] = []
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().cpu().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if self.current and self.current_bytes + nbytes > self.shard_size_bytes:
            self.flush()
        self.current[key] = tensor
        self.current_bytes += nbytes
        self.total_size += nbytes
        if self.current_bytes >= self.shard_size_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.current:
            return
        tmp_path = self.out_dir / f"model-{len(self.shards) + 1:05d}.safetensors"
        save_file(self.current, tmp_path)
        self.shards.append((tmp_path, sorted(self.current)))
        self.current = {}
        self.current_bytes = 0

    def close(self) -> None:
        self.flush()
        if len(self.shards) == 1:
            old, keys = self.shards[0]
            final = self.out_dir / "model.safetensors"
            old.rename(final)
            for key in keys:
                self.weight_map[key] = final.name
            return
        total = len(self.shards)
        for idx, (old, keys) in enumerate(self.shards, start=1):
            final = self.out_dir / f"model-{idx:05d}-of-{total:05d}.safetensors"
            old.rename(final)
            for key in keys:
                self.weight_map[key] = final.name
        index = {
            "metadata": {"total_size": str(self.total_size)},
            "weight_map": dict(sorted(self.weight_map.items())),
        }
        (self.out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def align_offset(offset: int, alignment: int) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + alignment - rem


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_resolved = out_dir.resolve()
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if os.path.commonpath([str(out_resolved), str(target)]) != str(out_resolved):
                raise RuntimeError(f"refusing unsafe tar member path: {member.name}")
        tf.extractall(out_dir)


def materialize_input(input_path: Path, tmp_dir: Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_dir():
        return input_path, None
    if not tarfile.is_tarfile(input_path):
        raise ValueError(f"{input_path} is neither a directory nor a tar archive")
    tmp = tempfile.TemporaryDirectory(dir=str(tmp_dir) if tmp_dir else None)
    root = Path(tmp.name)
    safe_extract_tar(input_path, root)
    children = [p for p in root.iterdir() if not p.name.startswith(".")]
    return (children[0] if len(children) == 1 and children[0].is_dir() else root), tmp


def find_cactus_root(root: Path) -> Path:
    if (root / "conversion_manifest.json").exists():
        return root
    weights = [p for p in root.rglob("*.weights") if p.is_file()]
    if not weights:
        raise ValueError(f"no .weights files found under {root}")
    counts: dict[Path, int] = {}
    for path in weights:
        counts[path.parent] = counts.get(path.parent, 0) + 1
    return max(counts, key=counts.get)


def copy_config_files(src_root: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in src_root.rglob("*"):
        if path.is_file() and path.name in CONFIG_FILES:
            rel = path.relative_to(src_root)
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(rel))
    return sorted(set(copied))


def read_header(path: Path) -> CactusHeader:
    raw = path.read_bytes()[:HEADER_SIZE]
    if len(raw) != HEADER_SIZE:
        raise ValueError(f"{path}: too small for cactus header")
    if raw[:4] != CACTUS_MAGIC:
        raise ValueError(f"{path}: bad magic {raw[:4]!r}")
    fields = struct.unpack("<IIIQQQQIQQIIQ", raw[4:HEADER_SIZE])
    flags, alignment, ndim, d0, d1, d2, d3, precision, data_bytes, scales_bytes, group_size, num_groups, original_n = fields
    if ndim > 4:
        raise ValueError(f"{path}: invalid ndim={ndim}")
    if alignment <= 0:
        alignment = ALIGNMENT_DEFAULT
    scales_offset = align_offset(HEADER_SIZE, alignment)
    data_offset = align_offset(scales_offset + scales_bytes, alignment)
    expected = data_offset + data_bytes
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path}: size mismatch actual={actual} expected={expected}")
    return CactusHeader(path, flags, alignment, ndim, (d0, d1, d2, d3), precision, data_bytes, scales_bytes, group_size, num_groups, original_n)


def unpack_lsb_values(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    raw_bits = np.unpackbits(packed.astype(np.uint8, copy=False), bitorder="little")[: count * bits]
    raw_bits = raw_bits.reshape(count, bits)
    out = np.zeros(count, dtype=np.uint8)
    for bit in range(bits):
        out |= raw_bits[:, bit].astype(np.uint8) << bit
    return out


def dequantize_fp_file(path: Path, header: CactusHeader, out_dtype: torch.dtype) -> torch.Tensor:
    offset = align_offset(HEADER_SIZE, header.alignment)
    dtype = np.float16 if header.precision == PRECISION_FP16 else np.float32
    arr = np.fromfile(path, dtype=dtype, count=math.prod(header.shape), offset=offset)
    return torch.from_numpy(arr.reshape(header.shape).copy()).to(out_dtype)


def dequantize_int8_file(path: Path, header: CactusHeader, out_dtype: torch.dtype) -> torch.Tensor:
    scales_offset = align_offset(HEADER_SIZE, header.alignment)
    data_offset = align_offset(scales_offset + header.scales_bytes, header.alignment)
    with path.open("rb") as f:
        f.seek(scales_offset)
        scales_blob = f.read(header.scales_bytes)
        f.seek(data_offset)
        q = np.frombuffer(f.read(header.data_bytes), dtype=np.int8).copy()
    count = math.prod(header.shape)
    if q.size < count:
        raise ValueError(f"{path}: INT8 payload too small for shape {header.shape}")
    q = q[:count].reshape(header.shape).astype(np.float32)
    if header.scales_bytes > 0 and header.group_size > 0:
        scales = np.frombuffer(scales_blob, dtype=np.float16).astype(np.float32)
        if header.flags & FLAG_INTERLEAVED:
            raise ValueError(f"{path}: interleaved grouped INT8 QDQ is not implemented")
        if len(header.shape) == 1:
            group_ids = np.arange(count) // header.group_size
            arr = (q.reshape(-1) * scales[group_ids]).reshape(header.shape)
            if header.original_n and header.original_n < arr.shape[0]:
                arr = arr[: header.original_n]
        elif len(header.shape) == 2:
            n, k = header.shape
            num_groups = k // header.group_size
            if scales.size < n * num_groups:
                raise ValueError(f"{path}: grouped INT8 scale metadata too small")
            scales = scales[: n * num_groups].reshape(n, num_groups)
            arr = np.empty((n, k), dtype=np.float32)
            for col in range(k):
                arr[:, col] = q[:, col] * scales[:, col // header.group_size]
        elif len(header.shape) >= 3:
            n = header.shape[0]
            k_total = math.prod(header.shape[1:])
            if k_total % header.group_size != 0:
                raise ValueError(f"{path}: grouped INT8 inner size must be divisible by group size")
            num_groups = k_total // header.group_size
            if scales.size < n * num_groups:
                raise ValueError(f"{path}: grouped INT8 scale metadata too small")
            q2 = q.reshape(n, k_total)
            scales = scales[: n * num_groups].reshape(n, num_groups)
            arr2 = np.empty((n, k_total), dtype=np.float32)
            for col in range(k_total):
                arr2[:, col] = q2[:, col] * scales[:, col // header.group_size]
            arr = arr2.reshape(header.shape)
        else:
            raise ValueError(f"{path}: grouped INT8 QDQ supports rank >= 1 tensors only")
    else:
        arr = q
    return torch.from_numpy(arr.copy()).to(out_dtype)


def read_cq_payload(path: Path, header: CactusHeader) -> tuple[bytes, np.ndarray]:
    scales_offset = align_offset(HEADER_SIZE, header.alignment)
    data_offset = align_offset(scales_offset + header.scales_bytes, header.alignment)
    with path.open("rb") as f:
        f.seek(scales_offset)
        scales_blob = f.read(header.scales_bytes)
        f.seek(data_offset)
        packed = np.frombuffer(f.read(header.data_bytes), dtype=np.uint8).copy()
    return scales_blob, packed


def parse_normal_metadata(blob: bytes, header: CactusHeader):
    n, k = header.shape
    pos = 0
    codebook = np.frombuffer(blob, dtype=np.float16, count=1 << header.bits, offset=pos).astype(np.float32)
    pos += (1 << header.bits) * 2
    input_scale = np.frombuffer(blob, dtype=np.float16, count=k, offset=pos).astype(np.float32)
    pos += k * 4
    norms = np.frombuffer(blob, dtype=np.float16, count=n * header.num_groups, offset=pos).astype(np.float32).reshape(n, header.num_groups)
    pos += n * header.num_groups * 2
    left = np.frombuffer(blob, dtype=np.int8, count=header.group_size, offset=pos).astype(np.float32)
    pos += header.group_size
    right = np.frombuffer(blob, dtype=np.int8, count=header.group_size, offset=pos).astype(np.float32)
    pos += header.group_size
    perm = np.frombuffer(blob, dtype="<u4", count=header.group_size, offset=pos).astype(np.int64)
    return torch.from_numpy(codebook), torch.from_numpy(input_scale), torch.from_numpy(norms), torch.from_numpy(left), torch.from_numpy(right), torch.from_numpy(perm)


def parse_orthogonal_metadata(blob: bytes, header: CactusHeader):
    n, k = header.shape
    pos = 0
    codebook = np.frombuffer(blob, dtype=np.float16, count=1 << header.bits, offset=pos).astype(np.float32)
    pos += (1 << header.bits) * 2
    input_scale = np.frombuffer(blob, dtype=np.float16, count=k, offset=pos).astype(np.float32)
    pos += k * 4
    norms = np.frombuffer(blob, dtype=np.float16, count=n, offset=pos).astype(np.float32)
    pos += n * 2
    rotation = np.frombuffer(blob, dtype=np.float16, count=k * k, offset=pos).astype(np.float32).reshape(k, k)
    return torch.from_numpy(codebook), torch.from_numpy(input_scale), torch.from_numpy(norms), torch.from_numpy(rotation)


def dequantize_cq_file(path: Path, header: CactusHeader, out_dtype: torch.dtype, row_batch_size: int) -> torch.Tensor:
    if header.ndim != 2:
        raise ValueError(f"{path}: CQ tensors must be 2D, got shape={header.shape}")
    n, k = header.shape
    bits = header.bits
    scales_blob, packed = read_cq_payload(path, header)
    if header.flags & FLAG_ORTHOGONAL_ROTATION:
        packed_row_bytes = math.ceil(k * bits / 8)
        codebook, input_scale, norms, rotation = parse_orthogonal_metadata(scales_blob, header)
        out = torch.empty(n, k, dtype=out_dtype)
        packed_rows = packed.reshape(n, packed_row_bytes)
        rt = rotation.float().T.contiguous()
        scale = input_scale.float().unsqueeze(0)
        codebook = codebook.float()
        for start in range(0, n, row_batch_size):
            end = min(start + row_batch_size, n)
            idx_np = np.stack([unpack_lsb_values(row, k, bits) for row in packed_rows[start:end]])
            idx = torch.from_numpy(idx_np.astype(np.int64, copy=False))
            recon = (codebook[idx] @ rt) * norms[start:end].float().unsqueeze(1)
            out[start:end] = (recon / scale).to(out_dtype)
        return out
    packed_group_bytes = math.ceil(header.group_size * bits / 8)
    codebook, input_scale, norms, left, right, perm = parse_normal_metadata(scales_blob, header)
    signs = set(int(x) for x in left.tolist()) | set(int(x) for x in right.tolist())
    if not signs.issubset({-1, 1}):
        raise ValueError(f"{path}: signs must be +/-1, got {sorted(signs)}")
    if sorted(int(x) for x in perm.tolist()) != list(range(header.group_size)):
        raise ValueError(f"{path}: permutation is not bijective")
    base_h = torch.from_numpy((hadamard(header.group_size, dtype=float) / math.sqrt(header.group_size)).astype(np.float32))
    rotation = (left.float().unsqueeze(1) * base_h * right.float().unsqueeze(0))[:, perm.long()].contiguous()
    rt = rotation.T.contiguous()
    scale = input_scale.float().unsqueeze(0)
    codebook = codebook.float()
    out = torch.empty(n, k, dtype=out_dtype)
    packed_groups = packed.reshape(n, header.num_groups, packed_group_bytes)
    for start in range(0, n, row_batch_size):
        end = min(start + row_batch_size, n)
        idx_np = np.empty((end - start, header.num_groups, header.group_size), dtype=np.uint8)
        for row_i, row_groups in enumerate(packed_groups[start:end]):
            for group_i, group in enumerate(row_groups):
                idx_np[row_i, group_i] = unpack_lsb_values(group, header.group_size, bits)
        idx = torch.from_numpy(idx_np.astype(np.int64, copy=False))
        recon = (codebook[idx] @ rt) * norms[start:end].float().unsqueeze(-1)
        out[start:end] = (recon.reshape(end - start, k) / scale).to(out_dtype)
    return out


def load_conversion_manifest(cactus_root: Path) -> list[dict[str, Any]]:
    path = cactus_root / "conversion_manifest.json"
    if not path.exists():
        raise ValueError(f"{cactus_root} does not contain conversion_manifest.json; manifest-driven QDQ is required")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected list")
    return rows


def output_key_for_row(row: dict[str, Any], family: str) -> str:
    key = row.get("hf_name") or row.get("source_name")
    if not key:
        raise ValueError(f"manifest row has no hf/source key: {row}")
    return restore_hf_key_for_family(str(key), family)


def output_keys_for_row(row: dict[str, Any], family: str) -> list[str]:
    return adapter_for_family(family).qdq_output_keys(row)


def dequantize_path(path: Path, out_dtype: torch.dtype, row_batch_size: int) -> tuple[torch.Tensor, CactusHeader]:
    header = read_header(path)
    if header.precision in (PRECISION_FP16, PRECISION_FP32):
        tensor = dequantize_fp_file(path, header, out_dtype)
    elif header.precision == PRECISION_INT8:
        tensor = dequantize_int8_file(path, header, out_dtype)
    elif header.precision in PRECISION_CQ:
        tensor = dequantize_cq_file(path, header, out_dtype, row_batch_size)
    else:
        raise ValueError(f"{path}: unsupported precision={header.precision}")
    return tensor, header


def trim_to_manifest_shape(tensor: torch.Tensor, shape: list[int] | tuple[int, ...] | None) -> torch.Tensor:
    if not shape:
        return tensor
    expected = tuple(int(x) for x in shape)
    if tuple(tensor.shape) == expected:
        return tensor
    if len(expected) == 1 and tensor.ndim == 1 and tensor.shape[0] >= expected[0]:
        return tensor[: expected[0]]
    if len(expected) == 2 and tensor.ndim == 2 and tensor.shape[0] == expected[0] and tensor.shape[1] >= expected[1]:
        return tensor[:, : expected[1]]
    if tensor.numel() == math.prod(expected):
        return tensor.reshape(expected)
    return tensor


def infer_family(cactus_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    config_path = cactus_root / "config.json"
    if not config_path.exists():
        return "generic"
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(cactus_root), trust_remote_code=True, local_files_only=True)
        return detect_family(cfg, "auto")
    except Exception:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        model_type = str(data.get("model_type", "")).lower()
        arch = " ".join(data.get("architectures") or []).lower()
        if "gemma4" in model_type or "gemma4" in arch:
            return "gemma4"
        return "generic"


def convert_qdq(args) -> dict[str, Any]:
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to replace it")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    root, tmp = materialize_input(args.input, args.tmp_dir)
    try:
        cactus_root = find_cactus_root(root)
        family = infer_family(cactus_root, args.model_family)
        adapter = adapter_for_family(family)
        rows = load_conversion_manifest(cactus_root)
        writer = ShardedSafetensorsWriter(args.out, int(args.shard_size_gb * (1024**3)))
        report = {
            "input": str(args.input),
            "cactus_root": str(cactus_root),
            "family": family,
            "dtype": args.dtype,
            "copied_config_files": copy_config_files(cactus_root, args.out),
            "written": [],
            "skipped": [],
        }
        seen: set[str] = set()
        for row in rows:
            output_file = row.get("output_file")
            if not output_file:
                report["skipped"].append({"source_name": row.get("source_name"), "reason": "no output file"})
                continue
            path = cactus_root / str(output_file)
            if not path.exists():
                if row.get("required", row.get("status") != "ignored"):
                    raise FileNotFoundError(path)
                report["skipped"].append({"source_name": row.get("source_name"), "reason": "missing ignored output"})
                continue
            keys = output_keys_for_row(row, family)
            for key in keys:
                if key in seen:
                    raise RuntimeError(f"duplicate QDQ output key: {key}")
            tensor, header = dequantize_path(path, out_dtype, args.row_batch_size)
            scale_factor = float(row.get("scale_factor") or adapter.scale_factor(path.name))
            if scale_factor != 1.0 and torch.is_floating_point(tensor):
                tensor = tensor / scale_factor
            tensor = trim_to_manifest_shape(tensor, row.get("shape"))
            for key in keys:
                writer.add(key, tensor)
                seen.add(key)
                report["written"].append({
                    "file": path.name,
                    "key": key,
                    "shape": list(tensor.shape),
                    "precision": header.precision,
                    "scale_factor": scale_factor,
                })
            del tensor
        writer.close()
        report["written_count"] = len(report["written"])
        (args.out / "qdq_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        if tmp is not None:
            tmp.cleanup()
