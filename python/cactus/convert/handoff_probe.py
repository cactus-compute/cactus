from __future__ import annotations

import io
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any


_PROBE_MAGIC = b"CHP10P6\0"
_V1_ORDERED_KEYS = (
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
_SINGLE_KV_MAXCTX = 2560
_SINGLE_KV_MAXGEN = 512
_SINGLE_KV_N_OBS = 128
_SINGLE_KV_MAXSUM = 40
_GEMMA4_EOT_TOKEN_ID = 106
_GEN_ROUTER_PROBE_FILE = "wide_router_probe.pt"
_GEN_ROUTER_MAXCTX = 2560
_GEN_ROUTER_MAXGEN = 512
_GEN_ROUTER_N_OBS = 128
_GEN_ROUTER_MAXSUM = 40
_GEN_ROUTER_DEFAULT_BETA = 5.0
_GEN_ROUTER_ORDERED_KEYS = (
    "norm.weight",
    "norm.bias",
    "k.weight",
    "k.bias",
    "v.weight",
    "v.bias",
    "q",
    "head.0.weight",
    "head.0.bias",
    "head.2.weight",
    "head.2.bias",
)


def _packaged_asset_dir(model_id: str | None) -> Path | None:
    if not model_id:
        return None
    from ..cli.download import get_model_dir_name

    return Path(__file__).resolve().parent / "assets" / get_model_dir_name(model_id)


def _candidate_probe_files(output_dir: Path, model_id: str | None = None) -> list[Path]:
    cwd = Path.cwd()
    candidates: list[Path] = []
    asset_dir = _packaged_asset_dir(model_id)
    if asset_dir is not None:
        candidates.append(asset_dir / _GEN_ROUTER_PROBE_FILE)
        candidates.append(asset_dir / "prod_probe.pt")
        candidates.append(asset_dir / "probe.pt")
    candidates += [
        output_dir / _GEN_ROUTER_PROBE_FILE,
        output_dir / "prod_probe.pt",
        output_dir / "probe.pt",
        output_dir / "global_attn_probe_v10p6.pt",
        cwd / _GEN_ROUTER_PROBE_FILE,
        cwd / "prod_probe.pt",
        cwd / "probe.pt",
        cwd / "handoff_probe_pkg" / "prod_probe.pt",
        cwd / "v10p6_probe_release" / "global_attn_probe_v10p6.pt",
        Path.home() / "Downloads" / _GEN_ROUTER_PROBE_FILE,
        Path.home() / "Downloads" / "prod_probe.pt",
        Path.home() / "Downloads" / "probe.pt",
        Path.home() / "Downloads" / "handoff_probe_pkg" / "prod_probe.pt",
        Path.home() / "Downloads" / "v10p6_probe_release" / "global_attn_probe_v10p6.pt",
    ]
    return candidates


def _candidate_probe_zips(output_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    return [
        output_dir / "handoff_router_probe.zip",
        output_dir / "handoff-router-probe.zip",
        output_dir / "handoff_probe_pkg.zip",
        output_dir / "v10p6_probe_release.zip",
        cwd / "handoff_router_probe.zip",
        cwd / "handoff-router-probe.zip",
        cwd / "handoff_probe_pkg.zip",
        cwd / "v10p6_probe_release.zip",
        Path.home() / "Downloads" / "handoff_router_probe.zip",
        Path.home() / "Downloads" / "handoff-router-probe.zip",
        Path.home() / "Downloads" / "handoff_probe_pkg.zip",
        Path.home() / "Downloads" / "v10p6_probe_release.zip",
    ]


def _load_checkpoint_from_zip(zip_path: Path) -> Any:
    import torch

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for name in (
            f"handoff_router_probe/{_GEN_ROUTER_PROBE_FILE}",
            _GEN_ROUTER_PROBE_FILE,
            "handoff_probe_pkg/prod_probe.pt",
            "prod_probe.pt",
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
        value = checkpoint.get("state")
        if isinstance(value, dict):
            return value
        for key in ("state_dict", "model_state"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_checkpoint(output_dir: Path, model_id: str | None = None) -> tuple[Any, str] | tuple[None, None]:
    import torch

    for path in _candidate_probe_files(output_dir, model_id):
        if path.exists():
            return torch.load(path, map_location="cpu"), str(path)
    for path in _candidate_probe_zips(output_dir):
        if path.exists():
            return _load_checkpoint_from_zip(path), str(path)
    return None, None


def _tensor_hash(state: dict[str, Any], keys: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        tensor = state[key].detach().cpu().contiguous().float()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _probe_state_dict(checkpoint: Any, required_keys: tuple[str, ...]) -> dict[str, Any]:
    state = _state_dict_from_checkpoint(checkpoint)
    missing = [key for key in required_keys if key not in state]
    if missing:
        raise RuntimeError(f"handoff probe checkpoint missing tensors: {', '.join(missing)}")
    return state


def _write_probe_bundle_v1(out_dir: Path, state: dict[str, Any], *, feat_dim: int, t_h: int,
                           h1: int, h2: int, fmt: str, layer: int, source: str | None) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / "handoff_probe.bin"
    with probe_path.open("wb") as f:
        f.write(_PROBE_MAGIC)
        f.write(struct.pack("<IIIII", 1, feat_dim, t_h, h1, h2))
        for key in _V1_ORDERED_KEYS:
            tensor = state[key].detach().cpu().contiguous().float().numpy()
            f.write(tensor.tobytes(order="C"))

    metadata = {
        "format": fmt,
        "source": source,
        "layer": layer,
        "feat_dim": feat_dim,
        "t_h": t_h,
        "output": probe_path.name,
        "tensor_sha256": _tensor_hash(state, _V1_ORDERED_KEYS),
    }
    (out_dir / "handoff_probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def _single_kv_keys(n_streams: int) -> tuple[str, ...]:
    keys: list[str] = ["norm.weight", "norm.bias"]
    for prefix in ("pk", "pv"):
        for idx in range(n_streams):
            keys.append(f"{prefix}.{idx}.weight")
            keys.append(f"{prefix}.{idx}.bias")
    keys.extend((
        "aemb.weight",
        "q",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
    ))
    return tuple(keys)


def _checkpoint_meta(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    return {k: v for k, v in checkpoint.items() if k not in ("state", "state_dict", "model_state")}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _jsonable_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _jsonable(v) for k, v in meta.items()}


def _write_single_kv_probe_bundle(out_dir: Path, checkpoint: Any, *, layer: int, source: str | None) -> bool:
    meta = _checkpoint_meta(checkpoint)
    n_streams = int(meta.get("n_streams", 3))
    keys = _single_kv_keys(n_streams)
    state = _probe_state_dict(checkpoint, keys)

    feat_dim = int(state["norm.weight"].shape[0])
    t_h = int(meta.get("t_h", state["q"].shape[0]))
    mlp = int(state["head.0.weight"].shape[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / "handoff_probe.bin"
    with probe_path.open("wb") as f:
        f.write(_PROBE_MAGIC)
        f.write(struct.pack(
            "<IIIIIIIIII",
            2,
            feat_dim,
            t_h,
            mlp,
            n_streams,
            _SINGLE_KV_MAXCTX,
            _SINGLE_KV_MAXGEN,
            _SINGLE_KV_N_OBS,
            _SINGLE_KV_MAXSUM,
            _GEMMA4_EOT_TOKEN_ID,
        ))
        for key in keys:
            tensor = state[key].detach().cpu().contiguous().float().numpy()
            f.write(tensor.tobytes(order="C"))

    metadata = {
        "format": "cactus_handoff_probe_singlekv_v1",
        "source": source,
        "layer": layer,
        "feat_dim": feat_dim,
        "t_h": t_h,
        "mlp": mlp,
        "n_streams": n_streams,
        "max_context_tokens": _SINGLE_KV_MAXCTX,
        "max_generation_tokens": _SINGLE_KV_MAXGEN,
        "observation_tokens": _SINGLE_KV_N_OBS,
        "summary_tokens": _SINGLE_KV_MAXSUM,
        "eot_token_id": _GEMMA4_EOT_TOKEN_ID,
        "output": probe_path.name,
        "tensor_sha256": _tensor_hash(state, keys),
        "checkpoint_meta": _jsonable_meta(meta),
    }
    (out_dir / "handoff_probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def _write_gen_router_probe_bundle(out_dir: Path, checkpoint: Any, *, layer: int, source: str | None) -> bool:
    meta = _checkpoint_meta(checkpoint)
    state = _probe_state_dict(checkpoint, _GEN_ROUTER_ORDERED_KEYS)

    feat_dim = int(state["norm.weight"].shape[0])
    t_h = int(state["q"].shape[0])
    mlp = int(state["head.0.weight"].shape[0])
    beta = float(meta.get("beta", _GEN_ROUTER_DEFAULT_BETA))
    if not math.isfinite(beta) or beta <= 0.0:
        raise RuntimeError(f"unsupported Gemma4 router probe beta: {beta!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / "handoff_probe.bin"
    with probe_path.open("wb") as f:
        f.write(_PROBE_MAGIC)
        f.write(struct.pack(
            "<IIIIIIIII",
            3,
            feat_dim,
            t_h,
            mlp,
            _GEN_ROUTER_MAXCTX,
            _GEN_ROUTER_MAXGEN,
            _GEN_ROUTER_N_OBS,
            _GEN_ROUTER_MAXSUM,
            _GEMMA4_EOT_TOKEN_ID,
        ))
        f.write(struct.pack("<f", beta))
        for key in _GEN_ROUTER_ORDERED_KEYS:
            tensor = state[key].detach().cpu().contiguous().float().numpy()
            f.write(tensor.tobytes(order="C"))

    metadata = {
        "format": "cactus_handoff_probe_gen512_router_v1",
        "source": source,
        "layer": layer,
        "feat_dim": feat_dim,
        "t_h": t_h,
        "mlp": mlp,
        "max_context_tokens": _GEN_ROUTER_MAXCTX,
        "max_generation_tokens": _GEN_ROUTER_MAXGEN,
        "observation_tokens": _GEN_ROUTER_N_OBS,
        "summary_tokens": _GEN_ROUTER_MAXSUM,
        "eot_token_id": _GEMMA4_EOT_TOKEN_ID,
        "beta": beta,
        "output": probe_path.name,
        "tensor_sha256": _tensor_hash(state, _GEN_ROUTER_ORDERED_KEYS),
        "checkpoint_meta": _jsonable_meta(meta),
    }
    (out_dir / "handoff_probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def export_gemma4_handoff_probe(output_dir: str | Path, *, model_id: str | None = None) -> bool:
    """Package the Gemma4 cloud-handoff probe into a C++-readable bundle file."""
    model_key = (model_id or "").lower()
    if model_key and "gemma-4" not in model_key and "gemma4" not in model_key:
        return False

    out_dir = Path(output_dir)
    checkpoint, source = _load_checkpoint(out_dir, model_id)
    if checkpoint is None:
        return False

    state = _state_dict_from_checkpoint(checkpoint)
    meta = _checkpoint_meta(checkpoint)
    probe_name = meta.get("probe")
    if probe_name == "gen512":
        return _write_gen_router_probe_bundle(out_dir, checkpoint, layer=28, source=source)
    if {"k.weight", "v.weight", "q"}.issubset(state):
        raise RuntimeError(f"unsupported Gemma4 handoff probe type: {probe_name!r}")
    if "pk.0.weight" in state and "pv.0.weight" in state and "q" in state:
        return _write_single_kv_probe_bundle(out_dir, checkpoint, layer=28, source=source)

    state = _probe_state_dict(checkpoint, _V1_ORDERED_KEYS)
    return _write_probe_bundle_v1(
        out_dir, state, feat_dim=1536, t_h=32, h1=128, h2=64,
        fmt="cactus_handoff_probe_v10p6", layer=28, source=source,
    )


_PARAKEET_PROBE_FILE = "parakeet_v3_probe.pt"
_PARAKEET_PROBE_LAYER = 23


def _candidate_parakeet_probe_files(output_dir: Path, model_id: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    asset_dir = _packaged_asset_dir(model_id)
    if asset_dir is not None:
        candidates.append(asset_dir / _PARAKEET_PROBE_FILE)
    candidates += [
        output_dir / _PARAKEET_PROBE_FILE,
        Path.cwd() / _PARAKEET_PROBE_FILE,
        Path.home() / "Downloads" / _PARAKEET_PROBE_FILE,
    ]
    return candidates


def export_parakeet_handoff_probe(output_dir: str | Path, *, model_id: str | None = None) -> bool:
    """Package the Parakeet-TDT-v3 cloud-handoff probe into a C++-readable bundle file."""
    model_key = (model_id or "").lower()
    if model_key and "parakeet" not in model_key:
        return False

    out_dir = Path(output_dir)
    checkpoint = source = None
    for path in _candidate_parakeet_probe_files(out_dir, model_id):
        if path.exists():
            import torch

            checkpoint, source = torch.load(path, map_location="cpu"), str(path)
            break
    if checkpoint is None:
        return False

    state = _probe_state_dict(checkpoint, _V1_ORDERED_KEYS)
    return _write_probe_bundle_v1(
        out_dir, state,
        feat_dim=int(state["norm.weight"].shape[0]),
        t_h=int(state["proj.weight"].shape[0]),
        h1=int(state["head.0.weight"].shape[0]),
        h2=int(state["head.2.weight"].shape[0]),
        fmt="cactus_handoff_probe_parakeet", layer=_PARAKEET_PROBE_LAYER, source=source,
    )
