"""Package the gemma-4-e2b-it cloud-handoff probe into the handoff_probe.bin read by the engine."""
from __future__ import annotations

import json
import struct
from pathlib import Path

_PROBE_MODEL = "gemma-4-e2b-it"
_PROBE_MAGIC = b"CHP10P6\0"
_PROBE_FORMAT = "cactus_handoff_probe_v10p6"
_PROBE_LAYER = 28
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


def export_handoff_probe(output_dir: str | Path, model_id: str) -> bool:
    """Write handoff_probe.bin into output_dir; returns False for models without a probe."""
    from ..cli.download import get_model_dir_name

    if get_model_dir_name(model_id) != _PROBE_MODEL:
        return False

    probe_path = Path(__file__).resolve().parent / "assets" / _PROBE_MODEL / "probe.pt"
    if not probe_path.exists():
        raise RuntimeError(f"missing packaged probe asset: {probe_path}")

    import torch

    state = torch.load(probe_path, map_location="cpu")
    missing = [key for key in _ORDERED_KEYS if key not in state]
    if missing:
        raise RuntimeError(f"probe checkpoint missing tensors: {', '.join(missing)}")

    tensors = {key: state[key].detach().contiguous().float().numpy() for key in _ORDERED_KEYS}
    feat_dim = int(tensors["norm.weight"].shape[0])
    t_h = int(tensors["proj.weight"].shape[0])
    h1 = int(tensors["head.0.weight"].shape[0])
    h2 = int(tensors["head.2.weight"].shape[0])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "handoff_probe.bin").open("wb") as f:
        f.write(_PROBE_MAGIC)
        f.write(struct.pack("<IIIII", 1, feat_dim, t_h, h1, h2))
        for key in _ORDERED_KEYS:
            f.write(tensors[key].tobytes(order="C"))

    metadata = {
        "format": _PROBE_FORMAT,
        "layer": _PROBE_LAYER,
        "feat_dim": feat_dim,
        "t_h": t_h,
    }
    (out_dir / "handoff_probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return True
