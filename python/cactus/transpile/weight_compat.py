from __future__ import annotations

from dataclasses import replace
import shutil
import struct
from pathlib import Path

import numpy as np

from cactus.convert.cactus_adapters.tensor_io import CACTUS_ALIGNMENT
from cactus.convert.cactus_adapters.tensor_io import CACTUS_MAGIC
from cactus.convert.cactus_adapters.tensor_io import FLAG_INTERLEAVED
from cactus.convert.cactus_adapters.tensor_io import align_offset
from cactus.convert.model_adapters.naming import gemma4_scale_factor
from cactus.transpile.runtime_compat import Graph
from cactus.transpile.weight_binding import WeightBinding
from cactus.convert.quantization.cq import FLAG_ORTHOGONAL_ROTATION
from cactus.convert.quantization.cq import GROUP_SIZE as CQ_GROUP_SIZE
from cactus.convert.quantization.cq import PRECISION_CQ
from cactus.convert.quantization.cq import make_codebook
from cactus.convert.quantization.cq import make_hadamard_components
from cactus.convert.quantization.cq import make_hadamard_matrix
from cactus.convert.quantization.cq import make_orthogonal_rotation
from cactus.convert.quantization.cq import pack_indices_lsb
from cactus.convert.quantization.cq import quantize_hadamard
from cactus.convert.quantization.cq import quantize_orthogonal
from cactus.convert.quantization.cq import write_cq_tensor


_HEADER_SIZE = 84
_FLAG_EXTENDED_SHAPE = 1 << 4
_INT8 = int(Graph.INT8)
_INT4 = int(Graph.INT4)

_TOKEN_EMBEDDING_FILENAMES = {
    "token_embeddings.weights",
    "decoder_token_embeddings.weights",
}
_PER_LAYER_EMBEDDING_FILENAMES = {
    "embed_tokens_per_layer.weights",
}


class _OpenedTensor:
    def __init__(
        self,
        *,
        path: Path,
        precision: int,
        shape: tuple[int, ...],
        data: np.memmap,
        scales: np.memmap | None,
        group_size: int,
        num_groups: int,
        is_interleaved: bool,
        original_n: int,
        alignment: int,
    ) -> None:
        self.path = path
        self.precision = precision
        self.shape = shape
        self.data = data
        self.scales = scales
        self.group_size = group_size
        self.num_groups = num_groups
        self.is_interleaved = is_interleaved
        self.original_n = original_n
        self.alignment = alignment


def ensure_binding_compatible(binding: WeightBinding, source_tensor: object | None = None) -> WeightBinding:
    binding = ensure_embedding_binding_compatible(binding)
    return _ensure_legacy_int4_weight_binding_compatible(binding, source_tensor=source_tensor)


def ensure_embedding_binding_compatible(binding: WeightBinding) -> WeightBinding:
    if binding.kind != "embedding":
        return binding

    source_path = Path(binding.path).expanduser().resolve()
    opened = _open_cactus_tensor_file(source_path)
    if opened.precision != _INT8:
        return binding
    if len(opened.shape) != 2 or opened.scales is None or opened.group_size <= 0:
        return binding

    config = _embedding_cache_config(source_path.name, source_name=binding.source_name)
    if config is None:
        return binding

    compat_path = source_path.with_name(source_path.stem + f".cq{config['bits']}.weights")
    _materialize_cq_embedding_cache(
        opened,
        compat_path,
        bits=int(config["bits"]),
        rotation=str(config["rotation"]),
    )
    _cleanup_legacy_fp16_cache(source_path)
    return WeightBinding(path=str(compat_path), kind=binding.kind, source_name=binding.source_name)


def _embedding_cache_config(filename: str, *, source_name: str) -> dict[str, object] | None:
    normalized_source = str(source_name or "")
    if filename in _PER_LAYER_EMBEDDING_FILENAMES or normalized_source.endswith("embed_tokens_per_layer.weight"):
        return {"bits": 2, "rotation": "hadamard"}
    if filename in _TOKEN_EMBEDDING_FILENAMES:
        return {"bits": 4, "rotation": "orthogonal"}
    return {"bits": 4, "rotation": "orthogonal"}


def _ensure_legacy_int4_weight_binding_compatible(
    binding: WeightBinding,
    *,
    source_tensor: object | None,
) -> WeightBinding:
    source_path = Path(binding.path).expanduser().resolve()
    opened = _open_cactus_tensor_file(source_path)
    if not _is_legacy_packed_int4_tensor(opened):
        return binding

    compat_path = source_path.with_name(source_path.stem + ".cq4.weights")
    if compat_path.exists():
        if _can_materialize_compat_weight(source_tensor):
            _materialize_source_cq_weight(
                source_tensor,
                compat_path,
                bits=4,
                rotation="hadamard",
                scale_factor=_compat_weight_scale_factor(source_path),
                source_mtime_ns=source_path.stat().st_mtime_ns,
            )
        return WeightBinding(path=str(compat_path), kind=binding.kind, source_name=binding.source_name)

    if not _can_materialize_compat_weight(source_tensor):
        raise RuntimeError(
            "legacy packed INT4 weight is not directly executable in the v2 runtime and is "
            f"missing a CQ companion file: {source_path.name}. "
            "Rerun `cactus transpile ...` so the transpiler can materialize the CQ weights, "
            "or generate them through `cq_convert` first."
        )

    _materialize_source_cq_weight(
        source_tensor,
        compat_path,
        bits=4,
        rotation="hadamard",
        scale_factor=_compat_weight_scale_factor(source_path),
        source_mtime_ns=source_path.stat().st_mtime_ns,
    )
    return WeightBinding(path=str(compat_path), kind=binding.kind, source_name=binding.source_name)


def _is_legacy_packed_int4_tensor(opened: _OpenedTensor) -> bool:
    if opened.precision != _INT4:
        return False
    if len(opened.shape) != 2 or opened.scales is None or opened.group_size <= 0:
        return False
    if not opened.is_interleaved:
        return False
    return True


def _can_materialize_compat_weight(source_tensor: object | None) -> bool:
    if source_tensor is None:
        return False
    if isinstance(source_tensor, (str, bytes, bytearray, Path)):
        return False
    if np.isscalar(source_tensor):
        return False
    return hasattr(source_tensor, "shape")


def _compat_weight_scale_factor(source_path: Path) -> float:
    parent_name = source_path.parent.name.lower()
    if "gemma-4" in parent_name or "gemma4" in parent_name:
        return gemma4_scale_factor(source_path.name)
    return 1.0

def _cleanup_legacy_fp16_cache(source_path: Path) -> None:
    legacy = source_path.with_name(source_path.stem + ".fp16.weights")
    if not legacy.exists():
        return
    try:
        legacy.unlink()
    except OSError:
        pass


def _materialize_source_cq_weight(
    tensor: object,
    out_path: Path,
    *,
    bits: int,
    rotation: str,
    scale_factor: float = 1.0,
    source_mtime_ns: int | None = None,
) -> None:
    if source_mtime_ns is not None and out_path.exists() and out_path.stat().st_mtime_ns >= source_mtime_ns:
        return

    if rotation == "orthogonal":
        cq = quantize_orthogonal(tensor, bits=bits)
    else:
        cq = quantize_hadamard(tensor, bits=bits, use_gptq=False)
    if float(scale_factor) != 1.0:
        cq = replace(cq, norms=(cq.norms.astype(np.float32) * float(scale_factor)).astype(np.float16))
    write_cq_tensor(out_path, cq)


def _open_cactus_tensor_file(path: str | Path) -> _OpenedTensor:
    tensor_path = Path(path).expanduser().resolve()
    with tensor_path.open("rb") as handle:
        header = handle.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        raise RuntimeError(f"tensor file is too small for a Cactus header: {tensor_path}")
    if header[:4] != CACTUS_MAGIC:
        raise RuntimeError(f"tensor file is missing the CACT header: {tensor_path}")

    flags = struct.unpack_from("<I", header, 4)[0]
    alignment = max(1, int(struct.unpack_from("<I", header, 8)[0]))
    ndim = int(struct.unpack_from("<I", header, 12)[0])
    dims = list(struct.unpack_from("<QQQQ", header, 16))
    precision = int(struct.unpack_from("<I", header, 48)[0])
    byte_size = int(struct.unpack_from("<Q", header, 52)[0])
    scales_bytes = int(struct.unpack_from("<Q", header, 60)[0])
    group_size = int(struct.unpack_from("<I", header, 68)[0])
    num_groups = int(struct.unpack_from("<I", header, 72)[0])
    original_n = int(struct.unpack_from("<Q", header, 76)[0])
    header_size = _HEADER_SIZE
    if flags & _FLAG_EXTENDED_SHAPE:
        with tensor_path.open("rb") as handle:
            handle.seek(_HEADER_SIZE)
            extended = handle.read(32)
        if len(extended) < 32:
            raise RuntimeError(f"tensor file is too small for extended Cactus shape header: {tensor_path}")
        dims.extend(struct.unpack("<QQQQ", extended))
        header_size += 32
    shape = tuple(int(dim) for dim in dims[:ndim] if int(dim) > 0)

    dtype_map = {
        int(Graph.INT8): np.int8,
        int(Graph.FP16): np.float16,
        int(Graph.FP32): np.float32,
        int(Graph.INT4): np.uint8,
        3: np.uint8,
        4: np.uint8,
        5: np.uint8,
        6: np.uint8,
    }
    dtype = dtype_map.get(precision)
    if dtype is None:
        raise RuntimeError(f"unsupported tensor precision {precision} in {tensor_path}")

    aligned_header = align_offset(header_size, alignment)
    scales_offset = aligned_header if scales_bytes > 0 else 0
    data_offset = align_offset(scales_offset + scales_bytes, alignment) if scales_bytes > 0 else aligned_header

    data_element_count = byte_size // np.dtype(dtype).itemsize
    data = np.memmap(tensor_path, mode="r", dtype=dtype, offset=data_offset, shape=(data_element_count,))
    scales = None
    if scales_bytes > 0:
        scales = np.memmap(
            tensor_path,
            mode="r",
            dtype=np.float16,
            offset=scales_offset,
            shape=(scales_bytes // np.dtype(np.float16).itemsize,),
        )
    return _OpenedTensor(
        path=tensor_path,
        precision=precision,
        shape=shape,
        data=data,
        scales=scales,
        group_size=group_size,
        num_groups=num_groups,
        is_interleaved=bool(flags & FLAG_INTERLEAVED),
        original_n=original_n,
        alignment=alignment,
    )


def _materialize_cq_embedding_cache(
    opened: _OpenedTensor,
    out_path: Path,
    *,
    bits: int,
    rotation: str,
) -> None:
    src_mtime_ns = opened.path.stat().st_mtime_ns
    if out_path.exists() and out_path.stat().st_mtime_ns >= src_mtime_ns:
        return

    if len(opened.shape) != 2:
        raise RuntimeError(f"expected rank-2 embedding tensor, got shape={opened.shape}")

    rows = int(opened.original_n or opened.shape[0])
    cols = int(opened.shape[1])
    if rotation == "hadamard" and cols % CQ_GROUP_SIZE != 0:
        raise RuntimeError(
            f"Hadamard CQ embedding conversion requires hidden dim divisible by {CQ_GROUP_SIZE}, "
            f"got {cols} for {opened.path.name}"
        )

    input_scale = _estimate_embedding_input_scale(opened, rows=rows, cols=cols)
    precision = int(PRECISION_CQ[int(bits)])
    group_size = cols if rotation == "orthogonal" else CQ_GROUP_SIZE
    num_groups = cols // group_size
    codebook = make_codebook(group_size, bits).astype(np.float16)
    input_scale = np.asarray(input_scale, dtype=np.float16).reshape(cols)
    recip = np.minimum(1.0 / np.maximum(input_scale.astype(np.float32), 1e-8), 65504.0).astype(np.float16)

    chunk_rows = _recommended_chunk_rows(cols=cols, rotation=rotation)
    norms_tmp = out_path.with_suffix(out_path.suffix + ".norms.tmp")
    data_tmp = out_path.with_suffix(out_path.suffix + ".data.tmp")
    final_tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    flags = 0
    trailer_parts: list[bytes] = []
    if rotation == "orthogonal":
        flags |= FLAG_ORTHOGONAL_ROTATION
        trailer_parts.append(make_orthogonal_rotation(group_size, seed=1234).astype(np.float16).tobytes())
    else:
        left, right, perm = make_hadamard_components(group_size, seed=1234)
        trailer_parts.extend([
            left.tobytes(),
            right.tobytes(),
            perm.astype("<u4", copy=False).tobytes(),
        ])

    try:
        with norms_tmp.open("wb") as norms_handle, data_tmp.open("wb") as data_handle:
            for start in range(0, rows, chunk_rows):
                stop = min(start + chunk_rows, rows)
                chunk = _dequantize_int8_rows(opened, start, stop)
                if rotation == "orthogonal":
                    packed, norms = _quantize_orthogonal_chunk(chunk, bits=bits, input_scale=input_scale)
                else:
                    packed, norms = _quantize_hadamard_chunk(chunk, bits=bits, input_scale=input_scale)
                norms_handle.write(np.ascontiguousarray(norms, dtype=np.float16).tobytes())
                data_handle.write(np.ascontiguousarray(packed, dtype=np.uint8).tobytes())

        norms_bytes = norms_tmp.read_bytes()
        scales_blob = b"".join([
            codebook.tobytes(),
            input_scale.tobytes(),
            recip.tobytes(),
            norms_bytes,
            *trailer_parts,
        ])
        packed_bytes = rows * cols * bits // 8
        with final_tmp.open("wb") as handle:
            _write_cq_header(
                handle,
                rows=rows,
                cols=cols,
                precision=precision,
                packed_bytes=packed_bytes,
                scales_bytes=len(scales_blob),
                group_size=group_size,
                num_groups=num_groups,
                flags=flags,
            )
            scales_offset = align_offset(_HEADER_SIZE, CACTUS_ALIGNMENT)
            handle.write(b"\x00" * (scales_offset - _HEADER_SIZE))
            handle.write(scales_blob)
            data_offset = align_offset(scales_offset + len(scales_blob), CACTUS_ALIGNMENT)
            current = handle.tell()
            if data_offset > current:
                handle.write(b"\x00" * (data_offset - current))
            with data_tmp.open("rb") as packed_handle:
                shutil.copyfileobj(packed_handle, handle, length=8 * 1024 * 1024)
        final_tmp.replace(out_path)
    finally:
        for temp_path in (norms_tmp, data_tmp):
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _recommended_chunk_rows(*, cols: int, rotation: str) -> int:
    if rotation == "orthogonal":
        if cols >= 8192:
            return 8
        if cols >= 4096:
            return 16
        return 64
    if cols >= 8192:
        return 32
    if cols >= 4096:
        return 48
    return 96


def _estimate_embedding_input_scale(
    opened: _OpenedTensor,
    *,
    rows: int,
    cols: int,
    max_sample_rows: int = 2048,
    sample_chunk_rows: int = 64,
) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        return np.ones((cols,), dtype=np.float16)

    stride = max(1, rows // max(1, max_sample_rows))
    sampled = 0
    accum = np.zeros((cols,), dtype=np.float64)
    for start in range(0, rows, stride * sample_chunk_rows):
        stop = min(start + sample_chunk_rows, rows)
        chunk = _dequantize_int8_rows(opened, start, stop).astype(np.float32, copy=False)
        accum += np.abs(chunk).sum(axis=0, dtype=np.float64)
        sampled += int(stop - start)
        if sampled >= max_sample_rows:
            break

    if sampled <= 0:
        return np.ones((cols,), dtype=np.float16)

    mean_abs = np.maximum(accum / float(sampled), 1e-6)
    raw = np.power(mean_abs, -0.5, dtype=np.float64)
    raw /= np.exp(np.mean(np.log(np.clip(raw, 1e-6, None))))
    return np.clip(raw, 1.0 / 8.0, 8.0).astype(np.float16)


def _quantize_hadamard_chunk(
    chunk: np.ndarray,
    *,
    bits: int,
    input_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    work = chunk.astype(np.float32, copy=False) * input_scale.astype(np.float32, copy=False)[None, :]
    rows, cols = work.shape
    group_size = CQ_GROUP_SIZE
    groups = cols // group_size
    rotation = make_hadamard_matrix(group_size, seed=1234).astype(np.float32)
    codebook = make_codebook(group_size, bits).astype(np.float32)

    indices = np.empty((rows, cols), dtype=np.uint8)
    norms = np.empty((rows, groups), dtype=np.float16)
    for group_index in range(groups):
        lo = group_index * group_size
        hi = lo + group_size
        group = work[:, lo:hi]
        row_norms = np.linalg.norm(group, axis=1).clip(min=1e-8).astype(np.float32, copy=False)
        rotated = (group / row_norms[:, None]) @ rotation
        idx = np.abs(rotated[..., None] - codebook[None, None, :]).argmin(axis=-1).astype(np.uint8)
        indices[:, lo:hi] = idx
        norms[:, group_index] = row_norms.astype(np.float16)
    return pack_indices_lsb(indices, group_size, bits), norms


def _quantize_orthogonal_chunk(
    chunk: np.ndarray,
    *,
    bits: int,
    input_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    work = chunk.astype(np.float32, copy=False) * input_scale.astype(np.float32, copy=False)[None, :]
    rows, cols = work.shape
    rotation = make_orthogonal_rotation(cols, seed=1234).astype(np.float32)
    codebook = make_codebook(cols, bits).astype(np.float32)
    norms = np.linalg.norm(work, axis=1).clip(min=1e-8).astype(np.float32, copy=False)
    rotated = (work / norms[:, None]) @ rotation
    indices = np.abs(rotated[..., None] - codebook[None, None, :]).argmin(axis=-1).astype(np.uint8)
    return pack_indices_lsb(indices, cols, bits), norms.astype(np.float16).reshape(rows, 1)


def _write_cq_header(
    handle,
    *,
    rows: int,
    cols: int,
    precision: int,
    packed_bytes: int,
    scales_bytes: int,
    group_size: int,
    num_groups: int,
    flags: int,
) -> None:
    handle.write(CACTUS_MAGIC)
    handle.write(struct.pack("<I", int(flags)))
    handle.write(struct.pack("<I", CACTUS_ALIGNMENT))
    handle.write(struct.pack("<I", 2))
    handle.write(struct.pack("<Q", rows))
    handle.write(struct.pack("<Q", cols))
    handle.write(struct.pack("<Q", 0))
    handle.write(struct.pack("<Q", 0))
    handle.write(struct.pack("<I", int(precision)))
    handle.write(struct.pack("<Q", int(packed_bytes)))
    handle.write(struct.pack("<Q", int(scales_bytes)))
    handle.write(struct.pack("<I", int(group_size)))
    handle.write(struct.pack("<I", int(num_groups)))
    handle.write(struct.pack("<Q", rows))


def _dequantize_int8_rows(opened: _OpenedTensor, start: int, stop: int) -> np.ndarray:
    if opened.scales is None or opened.group_size <= 0:
        raise RuntimeError(f"INT8 embedding file is missing grouped scales: {opened.path}")

    rows = stop - start
    cols = int(opened.shape[1])
    num_groups = int(opened.num_groups)
    row_group = int(opened.group_size)
    padded_rows = int(opened.shape[0])

    if opened.is_interleaved:
        block = 4
        padded_cols = ((cols + block - 1) // block) * block
        q_blocks = opened.data.reshape(padded_rows // block, padded_cols // block, block, block)
        s_blocks = opened.scales.reshape(padded_rows // block, num_groups, block)
        block_start = start // block
        block_end = (stop + block - 1) // block
        q = np.asarray(q_blocks[block_start:block_end], dtype=np.int8).transpose(0, 2, 1, 3).reshape(-1, padded_cols)
        s = np.asarray(s_blocks[block_start:block_end], dtype=np.float16).astype(np.float32).transpose(0, 2, 1).reshape(-1, num_groups)
        row_offset = start - block_start * block
        q = q[row_offset : row_offset + rows, :cols]
        s = s[row_offset : row_offset + rows]
    else:
        q = np.asarray(opened.data, dtype=np.int8).reshape(padded_rows, cols)[start:stop]
        s = np.asarray(opened.scales, dtype=np.float16).astype(np.float32).reshape(padded_rows, num_groups)[start:stop]

    out = np.empty((rows, cols), dtype=np.float16)
    for group_index in range(num_groups):
        lo = group_index * row_group
        hi = min(lo + row_group, cols)
        if lo >= hi:
            break
        scales = s[:, group_index : group_index + 1]
        out[:, lo:hi] = (q[:, lo:hi].astype(np.float32) * scales).astype(np.float16)
    return out
