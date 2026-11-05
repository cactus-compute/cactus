from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


@dataclass
class TransformerConfig:
    vocab_size: int = 16384
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 2
    num_layers: int = 20
    qk_head_dim: int = 48
    v_head_dim: int = 64
    max_seq_len: int = 4096
    pad_token_id: int = 0
    embedding_dim: int = 128
    embedding_probes: int = 4
    embedding_queries: int = 4
    confidence_probes: int = 4
    confidence_queries: int = 4
    router_probes: int = 4
    router_queries: int = 4
    rope_theta: float = 100000.0
    dtype: str = "bfloat16"
    flash: bool = True
    engram_orders: tuple = (2, 3)
    engram_heads: int = 0
    engram_slots: int = 18432
    engram_seed_heads: int = 0
    engram_layers: tuple = (3, 7, 11, 15, 19)
    global_layers: tuple = (4, 9, 14, 19)
    sliding_window: int = 1024
    ladder_depths: tuple = ()
    ladder_sample: bool = False
    ladder_widths: tuple = ()
    ladder_order: tuple = ()
    mhc_lanes: int = 4
    qkv_conv_taps: int = 3
    out_vocab: int = 0
    kv_window: int = 0
    kv_bits: int = 8
    act_bits: int = 8
    weight_bits: str = ""
    remat: bool = True
    scan_unroll: int = 1

    def __init__(self, **kwargs):
        valid = {f.name for f in self.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if k in valid:
                setattr(self, k, v)
        legacy_attn = kwargs.get("attn_dim", 0)
        if legacy_attn:
            legacy_head = legacy_attn // self.num_heads
            if "qk_head_dim" not in kwargs:
                self.qk_head_dim = legacy_head
            if "v_head_dim" not in kwargs:
                self.v_head_dim = legacy_head
            if "sliding_window" not in kwargs:
                self.sliding_window = 0
            if "global_layers" not in kwargs:
                self.global_layers = ()
        self.engram_layers = tuple(self.engram_layers)
        self.global_layers = tuple(self.global_layers)
        self.ladder_depths = tuple(self.ladder_depths)
        self.ladder_widths = tuple(self.ladder_widths)
        self.ladder_order = tuple(self.ladder_order)

    @classmethod
    def from_saved(cls, saved):
        if not isinstance(saved, dict):
            saved = dict(saved.__dict__)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        saved = {k: v for k, v in saved.items() if k in allowed}
        for key, off in (("qk_head_dim", 0), ("v_head_dim", 0),
                         ("sliding_window", 0), ("global_layers", ()),
                         ("ladder_depths", ()), ("ladder_sample", False),
                         ("ladder_widths", ()), ("ladder_order", ()),
                         ("engram_seed_heads", 0),
                         ("qkv_conv_taps", 0), ("out_vocab", 0)):
            saved.setdefault(key, off)
        return cls(**saved)

    @property
    def jax_dtype(self):
        return DTYPE_MAP[self.dtype]


def head_dims(config):
    legacy = (getattr(config, "attn_dim", 0) or config.d_model) // config.num_heads
    qk = getattr(config, "qk_head_dim", 0) or legacy
    v = getattr(config, "v_head_dim", 0) or legacy
    return qk, v


def _ladder_layer_order(num_layers):
    if num_layers < 1:
        raise ValueError("models require at least one layer")
    if num_layers == 1:
        return (0,)
    selected = [0, num_layers - 1]
    order = list(selected)
    while len(order) < num_layers:
        selected.sort()
        _gap, left, right = max(
            ((right - left, left, right)
             for left, right in zip(selected, selected[1:])
             if right - left > 1),
            key=lambda item: (item[0], -item[1]),
        )
        candidate = (left + right) // 2
        selected.append(candidate)
        order.append(candidate)
    return tuple(order)


def ladder_order(spec):
    """Block selection order for a model: the bisection order of its depth, or
    the parent's order carried by a sliced rung so nested slices stay trained."""
    if isinstance(spec, int):
        return _ladder_layer_order(spec)
    saved = tuple(getattr(spec, "ladder_order", ()) or ())
    if saved:
        if sorted(saved) != list(range(spec.num_layers)):
            raise ValueError(f"ladder_order {saved} is not a permutation of "
                             f"{spec.num_layers} blocks")
        return saved
    return _ladder_layer_order(spec.num_layers)


def ladder_layer_indices(spec, depth):
    """Return the stable, nested original-layer indices for a depth rung.

    Every deployable rung keeps both endpoint blocks. Interior blocks are
    added one at a time by bisecting the largest remaining gap, producing
    deterministic nested subnetworks with spatially balanced coverage.
    `spec` is a layer count or a config (whose ladder_order wins).
    """
    num_layers = spec if isinstance(spec, int) else spec.num_layers
    if not 2 <= depth <= num_layers:
        raise ValueError(
            f"ladder depth must be in [2, {num_layers}], got {depth}")
    return tuple(sorted(ladder_order(spec)[:depth]))


def ladder_config(config, depth):
    assert depth < config.num_layers
    assert tuple(config.engram_layers) == tuple(sorted(config.engram_layers))
    selected = ladder_layer_indices(config, depth)
    remap = {layer: i for i, layer in enumerate(selected)}
    fields = dict(config.__dict__)
    fields["num_layers"] = depth
    fields["ladder_order"] = tuple(remap[l] for l in ladder_order(config) if l in remap)
    fields["global_layers"] = tuple(
        remap[l] for l in config.global_layers if l in remap)
    fields["engram_layers"] = tuple(
        remap[l] for l in config.engram_layers if l in remap)
    fields["ladder_depths"] = ()
    fields["ladder_sample"] = False
    return TransformerConfig(**fields)


def ladder_slice(params, config, depth):
    selected = ladder_layer_indices(config, depth)
    selected_set = set(selected)
    selected_array = jnp.asarray(selected, dtype=jnp.int32)
    rows = jnp.asarray((0, *(layer + 1 for layer in selected)), jnp.int32)
    out = {}
    for k, v in params.items():
        if k == "stack":
            out[k] = {sk: (sv if sk == "final_norm"
                           else jax.tree.map(
                               lambda a: jnp.take(a, selected_array, axis=0), sv))
                      for sk, sv in v.items()}
        elif k in HEAD_KEYS:
            def take_rows(name, axis):
                return jax.tree.map(lambda a: jnp.take(a, rows, axis=axis), v[name])
            out[k] = {**v, "probes": take_rows("probes", 0),
                      "gain": take_rows("gain", 0), "row_bias": take_rows("row_bias", 1)}
        elif k.startswith("engrams_"):
            site = int(k.rsplit("_", 1)[1])
            if config.engram_layers[site] in selected_set:
                new_site = sum(
                    layer in selected_set for layer in config.engram_layers[:site])
                out[f"engrams_{new_site}"] = v
        else:
            out[k] = v
    return out


def ladder_layer_ranks(spec):
    """Map each original block to its selection rank in the nested ladder."""
    order = ladder_order(spec)
    ranks = [0] * len(order)
    for rank, layer in enumerate(order):
        ranks[layer] = rank
    return tuple(ranks)


def ladder_row_keep(config, exit_depth=None):
    if exit_depth is None:
        return None
    ranks = jnp.asarray(ladder_layer_ranks(config), jnp.int32)
    return jnp.concatenate([jnp.ones((1,), jnp.float32),
                            (ranks < exit_depth).astype(jnp.float32)])


def width_config(config, w):
    assert 2 * w == config.d_model
    assert config.d_model & (config.d_model - 1) == 0
    assert config.num_heads % 2 == 0
    assert _hada_blocks(w)[1] == _hada_blocks(config.d_model)[1]
    orders, heads, _ = engram_geometry(config)
    if config.engram_layers:
        assert heads % 2 == 0
    fields = dict(config.__dict__)
    fields.update(
        d_model=w,
        num_heads=config.num_heads // 2,
        engram_seed_heads=getattr(config, "engram_seed_heads", 0) or heads,
        ladder_depths=(),
        ladder_sample=False,
        ladder_widths=(),
    )
    return TransformerConfig(**fields)


def _parameter_path(path):
    """Return a stable parameter path through optional Flax metadata nodes."""
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif getattr(entry, "name", None) == "value":
            continue
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return "/".join(parts)


def width_slice(params, config, w):
    D = config.d_model
    orders, heads, sub_dim = engram_geometry(config)
    nh = config.num_heads // 2
    qk_hd, v_hd = head_dims(config)
    q_cols, o_rows = nh * qk_hd, nh * v_hd
    ba_c = _hada_blocks(1 << (w - 1).bit_length())[0]
    kept = (np.concatenate([oi * heads + np.arange(heads // 2)
                            for oi in range(len(orders))])
            if config.engram_layers else None)
    kv_rows = ((kept[:, None] * sub_dim + np.arange(sub_dim)).ravel()
               if config.engram_layers else None)
    lane_rows = (np.arange(config.mhc_lanes)[:, None] * D + np.arange(w)).ravel()

    def cut(path, a):
        p = _parameter_path(path)
        if p.startswith("ab_scales"):
            return a
        if "engrams_" in p:
            if p.endswith("/embedding"):
                return a[kept]
            if p.endswith("key_proj/kernel") or p.endswith("value_proj/kernel"):
                return a[kv_rows][:, :w]
            if p.endswith("/taps"):
                return a[..., :w]
            return a
        if ("embedding_head" in p or "confidence_head" in p
                or "router_head" in p):
            if p.endswith("/probes") or p.endswith("/query"):
                return a[..., :w]
            if p.endswith("proj/kernel"):
                return a.reshape(-1, D, a.shape[-1])[:, :w].reshape(-1, a.shape[-1])
            return a
        if "mhc_phi" in p:
            return a[:, lane_rows]
        if "hadamard_mlp" in p:
            last = p.rsplit("/", 1)[1]
            if last in ("w1a", "w2a", "w3a"):
                return a[..., :ba_c, :ba_c]
            if last in ("d1", "d2", "b2", "d3", "d4"):
                return a[..., :w]
            if last == "cond_v":
                return a[..., :w, :]
            if last == "cond_u":
                return a[..., :w]
            return a
        if p.endswith("q_proj/kernel"):
            return a[..., :w, :q_cols]
        if p.endswith("gate_proj/kernel"):
            return a[..., :w, :o_rows]
        if p.endswith("k_proj/kernel") or p.endswith("v_proj/kernel"):
            return a[..., :w, :]
        if p.endswith("out_proj/kernel"):
            return a[..., :o_rows, :w]
        if p.endswith("/q_taps"):
            return a[..., :q_cols]
        if p.endswith("/k_taps") or p.endswith("/v_taps"):
            return a
        if p.endswith("q_norm/scale") or p.endswith("k_norm/scale"):
            return a
        if p.endswith("/scale"):
            return a[..., :w]
        if p == "embedding/embedding":
            return a[..., :w]
        return a

    return jax.tree_util.tree_map_with_path(cut, params)


ENGRAM_SUB_DIM = 128
HEAD_KEYS = ("embedding_head", "confidence_head", "router_head")


def engram_geometry(config):
    orders = tuple(config.engram_orders)
    heads = config.engram_heads or max(1, config.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = config.d_model // (len(orders) * heads)
    return orders, heads, sub_dim


def _hada_blocks(n):
    b = 1 << ((n - 1).bit_length() // 2)
    return b, n // b

