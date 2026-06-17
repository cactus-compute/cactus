from __future__ import annotations

import io
import hashlib
import json
import os
import struct
import zipfile
from pathlib import Path
from typing import Any


_PROBE_MAGIC = b"CHP10P6\0"
_ORDERED_KEYS = (
    "norm.weight",
    "norm.bias",
    "proj.weight",
    "proj.bias",
    "attn_query",
    "head.0.weight",
    "head.0.bias",
    "head.2.weight",
    "head.2.bias",
    "head.4.weight",
    "head.4.bias",
)


def _candidate_probe_files(output_dir: Path, names: tuple[str, ...]) -> list[Path]:
    # Same convention as the Gemma4 probe: look for the checkpoint by plain
    # filename in the bundle output dir and the dir the convert command runs in
    # (cwd), plus ~/Downloads as a convenience. Names may be relative paths.
    roots = [output_dir, Path.cwd(), Path.home() / "Downloads"]
    return [root / name for root in roots for name in names]


def _candidate_probe_zips(output_dir: Path, zip_names: tuple[str, ...]) -> list[Path]:
    cwd = Path.cwd()
    return [root / name for root in (output_dir, cwd, Path.home() / "Downloads") for name in zip_names]


def _load_checkpoint_from_zip(zip_path: Path, inner_names: tuple[str, ...]) -> Any:
    import torch

    with zipfile.ZipFile(zip_path) as zf:
        available = set(zf.namelist())
        for name in inner_names:
            if name in available:
                with zf.open(name) as f:
                    return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"no probe checkpoint found in {zip_path}")


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_checkpoint(
    output_dir: Path, file_names: tuple[str, ...], zip_names: tuple[str, ...], inner_names: tuple[str, ...]
) -> tuple[Any, str] | tuple[None, None]:
    import torch

    for path in _candidate_probe_files(output_dir, file_names):
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False), str(path)
    for path in _candidate_probe_zips(output_dir, zip_names):
        if path.exists():
            return _load_checkpoint_from_zip(path, inner_names), str(path)
    return None, None


def _tensor_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(_ORDERED_KEYS):
        tensor = state[key].detach().cpu().contiguous().float()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _probe_dims(state: dict[str, Any]) -> tuple[int, int, int, int]:
    """Read (feat_dim, t_h, h1, h2) from tensor shapes and validate the
    architecture the C++ loader (Model::load_handoff_probe) assumes."""
    missing = [key for key in _ORDERED_KEYS if key not in state]
    if missing:
        raise RuntimeError(f"handoff probe checkpoint missing tensors: {', '.join(missing)}")
    feat_dim = int(state["norm.weight"].shape[0])
    t_h = int(state["proj.weight"].shape[0])
    h1 = int(state["head.0.weight"].shape[0])
    h2 = int(state["head.2.weight"].shape[0])
    expect = {
        "norm.weight": (feat_dim,), "norm.bias": (feat_dim,),
        "proj.weight": (t_h, feat_dim), "proj.bias": (t_h,),
        "attn_query": (t_h,),
        "head.0.weight": (h1, t_h), "head.0.bias": (h1,),
        "head.2.weight": (h2, h1), "head.2.bias": (h2,),
        "head.4.weight": (1, h2), "head.4.bias": (1,),
    }
    for key, shape in expect.items():
        got = tuple(int(d) for d in state[key].shape)
        if got != shape:
            raise RuntimeError(f"handoff probe tensor {key}: expected {shape}, got {got}")
    return feat_dim, t_h, h1, h2


def _write_probe_bundle(state: dict[str, Any], out_dir: Path, source: str, *, layer: int, fmt: str) -> dict:
    feat_dim, t_h, h1, h2 = _probe_dims(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / "handoff_probe.bin"
    with probe_path.open("wb") as f:
        f.write(_PROBE_MAGIC)
        f.write(struct.pack("<IIIII", 1, feat_dim, t_h, h1, h2))
        for key in _ORDERED_KEYS:
            tensor = state[key].detach().cpu().contiguous().float().numpy()
            f.write(tensor.tobytes(order="C"))
    metadata = {
        "format": fmt,
        "source": source,
        "layer": layer,
        "feat_dim": feat_dim,
        "t_h": t_h,
        "h1": h1,
        "h2": h2,
        "output": str(probe_path.name),
        "tensor_sha256": _tensor_hash(state),
    }
    (out_dir / "handoff_probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def export_gemma4_handoff_probe(output_dir: str | Path, *, model_id: str | None = None) -> bool:
    """Package the Gemma4 v10p6 cloud-handoff probe into a C++-readable bundle file."""
    model_key = (model_id or "").lower()
    if model_key and "gemma-4" not in model_key and "gemma4" not in model_key:
        return False

    out_dir = Path(output_dir)
    checkpoint, source = _load_checkpoint(
        out_dir,
        file_names=("probe.pt", "global_attn_probe_v10p6.pt",
                    "v10p6_probe_release/global_attn_probe_v10p6.pt"),
        zip_names=("v10p6_probe_release.zip",),
        inner_names=(
            "v10p6_probe_release/global_attn_probe_v10p6.pt",
            "global_attn_probe_v10p6.pt",
            "probe.pt",
        ),
    )
    if checkpoint is None:
        return False
    state = _state_dict_from_checkpoint(checkpoint)
    _write_probe_bundle(state, out_dir, source, layer=28, fmt="cactus_handoff_probe_v10p6")
    return True


def export_parakeet_handoff_probe(output_dir: str | Path, *, model_id: str | None = None) -> bool:
    """Package the Parakeet-TDT cloud-handoff probe into a C++-readable bundle.

    Reads the probe dims from the checkpoint tensors (feat_dim=1024, t_h=32,
    h1=128, h2=64 for parakeet-tdt-0.6b-v2), so it is not tied to the Gemma4
    1536-d shape. The capture layer is informational metadata; the engine taps
    whatever layer the transpiler exposes as probe_hidden.

    Detected like the Gemma4 probe: a plain ``parakeet_handoff_probe.pt`` in the
    bundle output dir or the convert working directory (drop the file next to
    where you run ``cactus convert``)."""
    model_key = (model_id or "").lower()
    if model_key and "parakeet" not in model_key:
        return False

    out_dir = Path(output_dir)
    checkpoint, source = _load_checkpoint(
        out_dir,
        file_names=("parakeet_handoff_probe.pt",),
        zip_names=(),
        inner_names=(),
    )
    if checkpoint is None:
        return False
    state = _state_dict_from_checkpoint(checkpoint)
    layer = int(os.environ.get("CACTUS_PARAKEET_PROBE_LAYER", "17"))
    _write_probe_bundle(state, out_dir, source, layer=layer, fmt="cactus_handoff_probe_parakeet")
    return True
