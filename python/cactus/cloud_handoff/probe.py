"""Wrongness probe used by cloud handoff experiments.

The bundled v10p6 probe reads Gemma 4 E2B-it layer-28 hidden states with shape
``(generated_tokens, 1536)`` and predicts ``p(local_answer_is_wrong)``.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

RELEASE_NAME = "v10p6_probe_release"
CHECKPOINT_NAME = "global_attn_probe_v10p6.pt"
CONFIG_NAME = "config.json"
BINARY_NAME = "global_attn_probe_v10p6.bin"
EXPECTED_PARAMETER_COUNT = 64833


def _release_resource(name: str):
    return files(__package__).joinpath("models", RELEASE_NAME, name)


def _read_json(path_or_resource: str | Path | Any) -> dict[str, Any]:
    if isinstance(path_or_resource, (str, Path)):
        with Path(path_or_resource).open("r", encoding="utf-8") as fh:
            return json.load(fh)
    with path_or_resource.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_release_config() -> dict[str, Any]:
    """Load the bundled v10p6 architecture config."""
    return _read_json(_release_resource(CONFIG_NAME))


class _SelfAttnBlock(nn.Module):
    """Small transformer-style self-attention block for non-p6 sweep variants."""

    def __init__(self, t_h: int, n_heads: int = 4):
        super().__init__()
        if t_h % n_heads != 0:
            raise ValueError(f"t_h={t_h} must be divisible by n_heads={n_heads}")
        self.attn = nn.MultiheadAttention(
            embed_dim=t_h,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(t_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attn_out)


class GlobalAttnPoolProbeV10(nn.Module):
    """Configurable v10 attention-pool wrongness probe.

    Forward expects one rollout of hidden states with shape ``(T, feat_dim)``.
    Apply ``torch.sigmoid(logit)`` to get ``p(local_answer_is_wrong)``.
    """

    def __init__(
        self,
        feat_dim: int | None = None,
        t_h: int = 32,
        dropout: float = 0.20,
        n_attn_layers: int = 1,
        attn_kind: str = "single_query",
        n_query_heads: int = 4,
        mlp_hidden_dims: Sequence[int] | None = None,
        layers_to_use: Sequence[int] | None = None,
        base_feat_dim: int = 1536,
        max_seq_len: int = 1024,
        self_attn_heads: int = 4,
    ):
        super().__init__()
        layers = [28] if layers_to_use is None else [int(x) for x in layers_to_use]
        if not layers:
            raise ValueError("layers_to_use must contain at least one layer")
        if not 1 <= n_attn_layers <= 4:
            raise ValueError("n_attn_layers must be in [1, 4]")
        if attn_kind not in {"single_query", "multihead_query"}:
            raise ValueError("attn_kind must be 'single_query' or 'multihead_query'")
        if n_query_heads < 1:
            raise ValueError("n_query_heads must be >= 1")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")

        expected_feat_dim = len(layers) * int(base_feat_dim)
        if feat_dim is None:
            feat_dim = expected_feat_dim
        elif int(feat_dim) != expected_feat_dim:
            raise ValueError(
                f"feat_dim={feat_dim} does not match layers_to_use={layers}; "
                f"expected {expected_feat_dim}"
            )

        hidden_dims = [128, 64] if mlp_hidden_dims is None else [int(x) for x in mlp_hidden_dims]
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("mlp_hidden_dims must contain positive integers")

        self.feat_dim = int(feat_dim)
        self.base_feat_dim = int(base_feat_dim)
        self.layers_to_use = layers
        self.t_h = int(t_h)
        self.dropout = float(dropout)
        self.n_attn_layers = int(n_attn_layers)
        self.attn_kind = attn_kind
        self.n_query_heads = int(n_query_heads)
        self.mlp_hidden_dims = hidden_dims
        self.max_seq_len = int(max_seq_len)
        self.self_attn_heads = int(self_attn_heads)

        self.norm = nn.LayerNorm(self.feat_dim)
        self.input_dropout = nn.Dropout(self.dropout)
        self.proj = nn.Linear(self.feat_dim, self.t_h)

        if self.attn_kind == "single_query":
            self.attn_query = nn.Parameter(torch.randn(self.t_h) / math.sqrt(self.t_h))
        else:
            self.attn_query = nn.Parameter(
                torch.randn(self.n_query_heads, self.t_h) / math.sqrt(self.t_h)
            )

        if self.n_attn_layers >= 2:
            self.pos_embed = nn.Parameter(torch.zeros(self.max_seq_len, self.t_h))
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
            self.self_attn_blocks = nn.ModuleList(
                [_SelfAttnBlock(self.t_h, n_heads=self.self_attn_heads) for _ in range(self.n_attn_layers)]
            )
        else:
            self.register_parameter("pos_embed", None)
            self.self_attn_blocks = nn.ModuleList()

        head_layers: list[nn.Module] = []
        in_dim = self.t_h
        for hidden_dim in self.mlp_hidden_dims:
            head_layers.append(nn.Linear(in_dim, hidden_dim))
            head_layers.append(nn.ReLU())
            in_dim = hidden_dim
        head_layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def _pool(self, u: torch.Tensor) -> torch.Tensor:
        if self.attn_kind == "single_query":
            scores = (u @ self.attn_query) / math.sqrt(self.t_h)
            alpha = torch.softmax(scores, dim=0)
            return (alpha[:, None] * u).sum(dim=0)
        scores = (self.attn_query @ u.transpose(0, 1)) / math.sqrt(self.t_h)
        alpha = torch.softmax(scores, dim=1)
        pooled = alpha @ u
        return pooled.mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected (T, {self.feat_dim}), got {tuple(x.shape)}")
        if x.shape[1] != self.feat_dim:
            raise ValueError(f"expected feature dim {self.feat_dim}, got {x.shape[1]}")
        if x.shape[0] == 0:
            raise ValueError("expected at least one token, got T=0")

        u = torch.relu(self.proj(self.input_dropout(self.norm(x))))
        if self.self_attn_blocks:
            if u.shape[0] > self.max_seq_len:
                warnings.warn(
                    f"truncating sequence from {u.shape[0]} to {self.max_seq_len}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                u = u[: self.max_seq_len]
            u = u + self.pos_embed[: u.shape[0]]
            u_b = u.unsqueeze(0)
            for block in self.self_attn_blocks:
                u_b = block(u_b)
            u = u_b.squeeze(0)
        return self.head(self._pool(u)).squeeze(-1)


@dataclass(frozen=True)
class WrongnessProbeResult:
    """A scalar cloud-handoff decision signal from hidden states."""

    logit: float
    probability_wrong: float
    confidence: float
    token_count: int
    feature_dim: int
    layers_to_use: tuple[int, ...]


def parameter_count(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def export_probe_binary(output_path: str | Path) -> Path:
    """Export the bundled checkpoint as a flat float32 file for the C++ runtime."""
    probe = load_default_probe("cpu")
    state = probe.state_dict()
    tensor_order = (
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        fh.write(b"CCHP10P6")
        fh.write(struct.pack("<I", 1))
        fh.write(struct.pack("<I", len(tensor_order)))
        for name in tensor_order:
            tensor = state[name].detach().cpu().contiguous().view(-1).float()
            encoded = name.encode("utf-8")
            fh.write(struct.pack("<H", len(encoded)))
            fh.write(encoded)
            fh.write(struct.pack("<I", tensor.numel()))
            fh.write(tensor.numpy().astype("<f4", copy=False).tobytes())
    return output


def _constructor_config(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "feat_dim",
        "base_feat_dim",
        "layers_to_use",
        "t_h",
        "dropout",
        "n_attn_layers",
        "attn_kind",
        "n_query_heads",
        "mlp_hidden_dims",
        "max_seq_len",
        "self_attn_heads",
    }
    return {key: value for key, value in config.items() if key in allowed}


def _load_torch_checkpoint(path_or_resource: str | Path | Any, device: str):
    if isinstance(path_or_resource, (str, Path)):
        return torch.load(Path(path_or_resource), map_location=device, weights_only=False)
    with as_file(path_or_resource) as resolved:
        return torch.load(resolved, map_location=device, weights_only=False)


def load_probe(
    checkpoint_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    device: str = "cpu",
    **config_overrides: Any,
) -> GlobalAttnPoolProbeV10:
    """Load a v10 wrongness probe checkpoint.

    By default this loads the bundled v10p6 release from package data.
    """
    config = load_release_config() if config_path is None else _read_json(config_path)
    config.update(config_overrides)
    probe = GlobalAttnPoolProbeV10(**_constructor_config(config)).to(device).eval()

    source = _release_resource(CHECKPOINT_NAME) if checkpoint_path is None else checkpoint_path
    checkpoint = _load_torch_checkpoint(source, device)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model", "params"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    probe.load_state_dict(state)
    return probe


def load_default_probe(device: str = "cpu") -> GlobalAttnPoolProbeV10:
    """Load a fresh bundled v10p6 probe."""
    return load_probe(device=device)


@lru_cache(maxsize=4)
def get_default_probe(device: str = "cpu") -> GlobalAttnPoolProbeV10:
    """Load and cache the bundled v10p6 probe for repeated scoring."""
    return load_default_probe(device=device)


def _coerce_hidden_states(hidden_states: Any, *, device: str) -> torch.Tensor:
    if torch.is_tensor(hidden_states):
        x = hidden_states.to(device=device, dtype=torch.float32)
    else:
        x = torch.as_tensor(hidden_states, dtype=torch.float32, device=device)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x


def score_hidden_states(
    hidden_states: Any,
    *,
    probe: GlobalAttnPoolProbeV10 | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> WrongnessProbeResult:
    """Score a generated rollout.

    ``confidence`` is ``1 - p_wrong`` so it can be compared with existing
    cloud-handoff confidence thresholds.
    """
    model = probe or (load_probe(checkpoint_path, device=device) if checkpoint_path else get_default_probe(device))
    x = _coerce_hidden_states(hidden_states, device=device)
    with torch.no_grad():
        logit_tensor = model(x)
        probability_wrong = torch.sigmoid(logit_tensor)
    logit = float(logit_tensor.detach().cpu().item())
    p_wrong = float(probability_wrong.detach().cpu().item())
    return WrongnessProbeResult(
        logit=logit,
        probability_wrong=p_wrong,
        confidence=1.0 - p_wrong,
        token_count=int(x.shape[0]),
        feature_dim=int(x.shape[1]) if x.ndim >= 2 else 0,
        layers_to_use=tuple(model.layers_to_use),
    )


def _demo(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bundled v10p6 wrongness probe smoke test.")
    parser.add_argument("--tokens", type=int, nargs="*", default=[12, 180, 1024])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    probe = get_default_probe("cpu")
    print(json.dumps({
        "release": RELEASE_NAME,
        "parameter_count": parameter_count(probe),
        "config": load_release_config(),
    }, indent=2))
    for token_count in args.tokens:
        x = torch.randn(token_count, probe.feat_dim)
        result = score_hidden_states(x, probe=probe)
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
