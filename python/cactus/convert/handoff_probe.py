from __future__ import annotations

import io
import hashlib
import json
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


def _candidate_probe_files(output_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    return [
        output_dir / "probe.pt",
        output_dir / "global_attn_probe_v10p6.pt",
        cwd / "probe.pt",
        cwd / "v10p6_probe_release" / "global_attn_probe_v10p6.pt",
        Path.home() / "Downloads" / "probe.pt",
        Path.home() / "Downloads" / "v10p6_probe_release" / "global_attn_probe_v10p6.pt",
    ]


def _candidate_probe_zips(output_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    return [
        output_dir / "v10p6_probe_release.zip",
        cwd / "v10p6_probe_release.zip",
        Path.home() / "Downloads" / "v10p6_probe_release.zip",
    ]


def _load_checkpoint_from_zip(zip_path: Path) -> Any:
    import torch

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for name in (
            "v10p6_probe_release/global_attn_probe_v10p6.pt",
            "global_attn_probe_v10p6.pt",
            "probe.pt",
        ):
            if name in names:
                with zf.open(name) as f:
                    return torch.load(io.BytesIO(f.read()), map_location="cpu")
    raise FileNotFoundError(f"no probe checkpoint found in {zip_path}")


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_checkpoint(output_dir: Path) -> tuple[Any, str] | tuple[None, None]:
    import torch

    for path in _candidate_probe_files(output_dir):
        if path.exists():
            return torch.load(path, map_location="cpu"), str(path)
    for path in _candidate_probe_zips(output_dir):
        if path.exists():
            return _load_checkpoint_from_zip(path), str(path)
    return None, None


def _tensor_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(_ORDERED_KEYS):
        tensor = state[key].detach().cpu().contiguous().float()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _probe_state_dict(checkpoint: Any) -> dict[str, Any]:
    state = _state_dict_from_checkpoint(checkpoint)
    missing = [key for key in _ORDERED_KEYS if key not in state]
    if missing:
        raise RuntimeError(f"handoff probe checkpoint missing tensors: {', '.join(missing)}")
    return state


def _write_probe_bundle(out_dir: Path, state: dict[str, Any], *, feat_dim: int, t_h: int,
                        h1: int, h2: int, fmt: str, layer: int, source: str | None) -> bool:
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
        "output": probe_path.name,
        "tensor_sha256": _tensor_hash(state),
    }
    (out_dir / "handoff_probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def export_gemma4_handoff_probe(output_dir: str | Path, *, model_id: str | None = None) -> bool:
    """Package the Gemma4 v10p6 cloud-handoff probe into a C++-readable bundle file."""
    model_key = (model_id or "").lower()
    if model_key and "gemma-4" not in model_key and "gemma4" not in model_key:
        return False

    out_dir = Path(output_dir)
    checkpoint, source = _load_checkpoint(out_dir)
    if checkpoint is None:
        return False

    state = _probe_state_dict(checkpoint)
    return _write_probe_bundle(
        out_dir, state, feat_dim=1536, t_h=32, h1=128, h2=64,
        fmt="cactus_handoff_probe_v10p6", layer=28, source=source,
    )


_PARAKEET_PROBE_FILE = "parakeet_v3_probe.pt"
_PARAKEET_PROBE_LAYER = 23


def _candidate_parakeet_probe_files(output_dir: Path) -> list[Path]:
    return [
        output_dir / _PARAKEET_PROBE_FILE,
        Path.cwd() / _PARAKEET_PROBE_FILE,
        Path.home() / "Downloads" / _PARAKEET_PROBE_FILE,
    ]


def export_parakeet_handoff_probe(output_dir: str | Path, *, model_id: str | None = None) -> bool:
    """Package the Parakeet-TDT-v3 cloud-handoff probe into a C++-readable bundle file."""
    model_key = (model_id or "").lower()
    if model_key and "parakeet" not in model_key:
        return False

    out_dir = Path(output_dir)
    checkpoint = source = None
    for path in _candidate_parakeet_probe_files(out_dir):
        if path.exists():
            import torch

            checkpoint, source = torch.load(path, map_location="cpu"), str(path)
            break
    if checkpoint is None:
        return False

    state = _probe_state_dict(checkpoint)
    return _write_probe_bundle(
        out_dir, state,
        feat_dim=int(state["norm.weight"].shape[0]),
        t_h=int(state["proj.weight"].shape[0]),
        h1=int(state["head.0.weight"].shape[0]),
        h2=int(state["head.2.weight"].shape[0]),
        fmt="cactus_handoff_probe_parakeet", layer=_PARAKEET_PROBE_LAYER, source=source,
    )
