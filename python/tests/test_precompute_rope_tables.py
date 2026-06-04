import json
from pathlib import Path

import pytest
import torch

from cactus.transpile.canonicalize.utils import rebuild_graph
from cactus.transpile.graph_ir import IRGraph
from cactus.transpile.graph_ir import IRNode
from cactus.transpile.graph_ir import IRValue
from cactus.transpile.optimize_graph import precompute_rope_tables


WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
GEMMA_BUNDLE = WEIGHTS / "gemma-4-e2b-it"
GEMMA_16K_BUNDLE = WEIGHTS / "gemma-4-e2b-16k"
QWEN_BUNDLE = WEIGHTS / "qwen3-0.6b"

TEST_MAX_SEQ = 16384

# inv_freq = 1 / theta ** (arange(0, dim, 2) / dim), exactly the constant the
# real graph carries. The serialized optimized_ir drops multi-element tensor
# payloads, so we re-materialize the genuine per-theta inv_freq. The (theta,
# head_dim) spec is read from the bundle's config.json at test time (not
# hardcoded) so a model shipped with the wrong rope_theta / head_dim fails here
# instead of being matched against a self-referential reconstruction.
def _gemma_inv_freq_spec(bundle: Path) -> dict[str, tuple[float, int]]:
    text = json.loads((bundle / "config.json").read_text())["text_config"]
    rope = text["rope_parameters"]
    return {
        "v_module_backbone_rotary_emb_sliding_attention_inv_freq": (
            float(rope["sliding_attention"]["rope_theta"]),
            int(text["head_dim"]),
        ),
        "v_module_backbone_rotary_emb_full_attention_inv_freq": (
            float(rope["full_attention"]["rope_theta"]),
            int(text["global_head_dim"]),
        ),
    }


def _qwen_inv_freq_spec(bundle: Path) -> dict[str, tuple[float, int]]:
    cfg = json.loads((bundle / "config.json").read_text())
    return {
        "v_module_backbone_rotary_emb_inv_freq": (
            float(cfg["rope_theta"]),
            int(cfg["head_dim"]),
        ),
    }


# The IR these tests load lives in transpiled weight bundles under weights/
# (gitignored, ~9MB), so skip when they are absent (e.g. CI) rather than failing
# collection; run locally after transpiling the bundles.
_BUNDLES_PRESENT = (GEMMA_BUNDLE / "config.json").exists() and (QWEN_BUNDLE / "config.json").exists()
pytestmark = pytest.mark.skipif(
    not _BUNDLES_PRESENT, reason="requires transpiled weight bundles under weights/ (gitignored)"
)

GEMMA_INV_FREQ = _gemma_inv_freq_spec(GEMMA_BUNDLE) if _BUNDLES_PRESENT else {}
GEMMA_16K_INV_FREQ = _gemma_inv_freq_spec(GEMMA_16K_BUNDLE) if (GEMMA_16K_BUNDLE / "config.json").exists() else {}
QWEN_INV_FREQ = _qwen_inv_freq_spec(QWEN_BUNDLE) if _BUNDLES_PRESENT else {}


def _inv_freq(theta: float, head_dim: int) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def test_gemma_inv_freq_spec_matches_config() -> None:
    """Guard the reconstruction source: the (theta, head_dim) the parity tests
    feed the pass must equal the bundle's real config.json rope_parameters, so a
    wrong-theta or wrong-head_dim bundle fails instead of self-referentially
    passing."""
    text = json.loads((GEMMA_BUNDLE / "config.json").read_text())["text_config"]
    rope = text["rope_parameters"]
    assert GEMMA_INV_FREQ == {
        "v_module_backbone_rotary_emb_sliding_attention_inv_freq": (
            float(rope["sliding_attention"]["rope_theta"]),
            int(text["head_dim"]),
        ),
        "v_module_backbone_rotary_emb_full_attention_inv_freq": (
            float(rope["full_attention"]["rope_theta"]),
            int(text["global_head_dim"]),
        ),
    }
    sliding = GEMMA_INV_FREQ["v_module_backbone_rotary_emb_sliding_attention_inv_freq"]
    full = GEMMA_INV_FREQ["v_module_backbone_rotary_emb_full_attention_inv_freq"]
    assert sliding[0] != full[0], "sliding and full thetas must be distinct (dual-theta)"
    assert sliding[1] != full[1], "sliding and full head_dims must be distinct"


def _load_ir(path: Path, inv_freq_spec: dict[str, tuple[float, int]], *, max_seq: int = TEST_MAX_SEQ) -> IRGraph:
    payload = json.loads(path.read_text())
    graph = payload["graph"]
    values = {
        vid: IRValue(
            id=vid,
            shape=tuple(v["shape"]) if v.get("shape") is not None else None,
            dtype=v.get("dtype"),
        )
        for vid, v in graph["values"].items()
    }
    nodes: dict[str, IRNode] = {}
    order: list[str] = []
    for n in graph["nodes"]:
        nodes[n["id"]] = IRNode(
            id=n["id"],
            op=n["op"],
            inputs=list(n["inputs"]),
            outputs=list(n["outputs"]),
            attrs=dict(n.get("attrs", {})),
            meta=dict(n.get("meta", {})),
            kind=n.get("kind", "generic"),
        )
        order.append(n["id"])
    constants = {
        cid: _inv_freq(*inv_freq_spec[cid])
        for cid in graph["constants"]
        if cid in inv_freq_spec
    }
    meta = dict(graph["meta"])
    meta["max_cache_seq_len"] = max_seq
    ir = IRGraph(
        values=values,
        nodes=nodes,
        order=order,
        inputs=list(graph["inputs"]),
        outputs=list(graph["outputs"]),
        constants=constants,
        meta=meta,
    )
    rebuild_graph(ir)
    return ir


def _cos_sin_inputs(payload_path: Path) -> dict[str, str]:
    """Map each scalar_cos/scalar_sin output id -> the int position graph input the
    original angle matmul consumed, mirroring the pass's index_shape constraint
    (the int value whose shape == cos/sin output shape minus its last dim)."""
    graph = json.loads(payload_path.read_text())["graph"]
    producer = {o: n for n in graph["nodes"] for o in n["outputs"]}
    values = graph["values"]

    def find_position(value_id: str, index_shape: tuple[int, ...]) -> str | None:
        seen: set[str] = set()
        stack = [value_id]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            val = values.get(cur, {})
            dtype = str(val.get("dtype"))
            shape = val.get("shape")
            if (
                dtype in {"int32", "int64", "i32", "i64"}
                and shape is not None
                and tuple(int(d) for d in shape) == index_shape
            ):
                return cur
            node = producer.get(cur)
            if node is None:
                continue
            stack.extend(node["inputs"])
        return None

    result: dict[str, str] = {}
    for n in graph["nodes"]:
        if n["op"] in {"scalar_cos", "scalar_sin"}:
            out_shape = values[n["outputs"][0]]["shape"]
            index_shape = tuple(int(d) for d in out_shape[:-1])
            result[n["outputs"][0]] = find_position(n["inputs"][0], index_shape)
    return result


def _rope_nodes(ir: IRGraph) -> tuple[list[IRNode], list[str]]:
    embeddings = [
        n
        for n in ir.nodes.values()
        if n.op == "embedding" and n.inputs and n.inputs[0].startswith("c_rope_table_")
    ]
    tables = [c for c in ir.constants if c.startswith("c_rope_table_")]
    return embeddings, tables


# --------------------------------------------------------------------------- #
# fires
# --------------------------------------------------------------------------- #


def test_gemma_decoder_step_rewrites_four_cos_sin() -> None:
    ir = _load_ir(GEMMA_BUNDLE / "optimized_ir_decoder_step.json", GEMMA_INV_FREQ)
    assert precompute_rope_tables(ir) is True
    embeddings, tables = _rope_nodes(ir)
    assert len(embeddings) == 4
    assert len(tables) == 4
    assert all(n.op != "scalar_cos" and n.op != "scalar_sin" for n in ir.nodes.values())


def test_gemma_prefill_chunk_rewrites_eight_cos_sin() -> None:
    ir = _load_ir(GEMMA_BUNDLE / "optimized_ir_decoder_prefill_chunk.json", GEMMA_INV_FREQ)
    assert precompute_rope_tables(ir) is True
    embeddings, tables = _rope_nodes(ir)
    assert len(embeddings) == 8
    assert len(tables) == 8


def test_gemma_media_step_rewrites_its_rope_tables() -> None:
    path = GEMMA_16K_BUNDLE / "components" / "decoder_media_step" / "optimized_ir.json"
    if not path.exists():
        path = GEMMA_16K_BUNDLE / "optimized_ir_decoder_media_step.json"
    if not path.exists():
        pytest.skip(
            "gemma4 component pipeline emits no decoder_media_step component "
            "(only lm_encoder_media_step, which carries no rope angle path); "
            "this guards future models that do emit a decoder media step"
        )
    ir = _load_ir(path, GEMMA_16K_INV_FREQ)
    assert precompute_rope_tables(ir) is True
    embeddings, tables = _rope_nodes(ir)
    assert len(embeddings) >= 4
    assert len(tables) >= 4
    widths = {int(ir.constants[t].shape[-1]) for t in tables}
    assert 256 in widths and 512 in widths


# --------------------------------------------------------------------------- #
# two thetas: both widths exist and each table is the correct fp64 cos/sin
# --------------------------------------------------------------------------- #


def _table_by_width(ir: IRGraph) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    for node in ir.nodes.values():
        if node.op != "embedding" or not node.inputs[0].startswith("c_rope_table_"):
            continue
        table = ir.constants[node.inputs[0]]
        width = int(table.shape[-1])
        kind = "cos" if "cos" in node.inputs[0] else "sin"
        out.setdefault(width, {})[kind] = table
    return out


def test_gemma_two_thetas_tables_match_fresh_fp64_cos_sin() -> None:
    ir = _load_ir(GEMMA_BUNDLE / "optimized_ir_decoder_step.json", GEMMA_INV_FREQ)
    precompute_rope_tables(ir)
    by_width = _table_by_width(ir)
    assert set(by_width.keys()) == {256, 512}

    expected = {
        head_dim: _inv_freq(theta, head_dim).to(torch.float64)
        for theta, head_dim in GEMMA_INV_FREQ.values()
    }
    assert set(expected.keys()) == {256, 512}
    probe_positions = [0, 1, 2000, 2049, TEST_MAX_SEQ - 1]
    for width, inv in expected.items():
        positions = torch.tensor(probe_positions, dtype=torch.float64).reshape(-1, 1)
        freqs = positions * inv.reshape(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        exp_cos = torch.cos(emb).to(torch.float16)
        exp_sin = torch.sin(emb).to(torch.float16)
        got_cos = by_width[width]["cos"][probe_positions]
        got_sin = by_width[width]["sin"][probe_positions]
        assert torch.equal(got_cos, exp_cos), f"cos width {width}"
        assert torch.equal(got_sin, exp_sin), f"sin width {width}"


def test_gemma_width256_uses_sliding_not_global_inv_freq() -> None:
    """Cross-check the per-theta binding: the width-256 table must NOT match the
    global theta and vice versa."""
    ir = _load_ir(GEMMA_BUNDLE / "optimized_ir_decoder_step.json", GEMMA_INV_FREQ)
    precompute_rope_tables(ir)
    by_width = _table_by_width(ir)

    wrong = _inv_freq(1000000.0, 256).to(torch.float64)  # global theta on sliding width
    positions = torch.tensor([2049.0], dtype=torch.float64).reshape(-1, 1)
    freqs = positions * wrong.reshape(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    wrong_cos = torch.cos(emb).to(torch.float16)
    assert not torch.equal(by_width[256]["cos"][[2049]], wrong_cos)


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #


def test_gemma_table_width_matches_output_and_halves_are_byte_equal() -> None:
    payload = json.loads((GEMMA_BUNDLE / "optimized_ir_decoder_step.json").read_text())
    out_width = {
        n["outputs"][0]: int(payload["graph"]["values"][n["outputs"][0]]["shape"][-1])
        for n in payload["graph"]["nodes"]
        if n["op"] in {"scalar_cos", "scalar_sin"}
    }
    ir = _load_ir(GEMMA_BUNDLE / "optimized_ir_decoder_step.json", GEMMA_INV_FREQ)
    precompute_rope_tables(ir)
    for node in ir.nodes.values():
        if node.op != "embedding" or not node.inputs[0].startswith("c_rope_table_"):
            continue
        out_id = node.outputs[0]
        table = ir.constants[node.inputs[0]]
        width = int(table.shape[-1])
        assert width == out_width[out_id]
        half = width // 2
        assert torch.equal(table[:, :half], table[:, half:])


# --------------------------------------------------------------------------- #
# position binding
# --------------------------------------------------------------------------- #


def test_gemma_embedding_index_is_original_position_input() -> None:
    path = GEMMA_BUNDLE / "optimized_ir_decoder_step.json"
    pos_for_output = _cos_sin_inputs(path)
    ir = _load_ir(path, GEMMA_INV_FREQ)
    precompute_rope_tables(ir)
    for node in ir.nodes.values():
        if node.op != "embedding" or not node.inputs[0].startswith("c_rope_table_"):
            continue
        out_id = node.outputs[0]
        assert node.inputs[1] == pos_for_output[out_id]
    assert set(pos_for_output.values()) == {"v_args_2"}


# --------------------------------------------------------------------------- #
# Qwen regression: single width table
# --------------------------------------------------------------------------- #


def test_qwen_single_width_table_still_correct() -> None:
    path = QWEN_BUNDLE / "optimized_ir_decoder_step.json"
    pos_for_output = _cos_sin_inputs(path)
    ir = _load_ir(path, QWEN_INV_FREQ)
    assert precompute_rope_tables(ir) is True
    embeddings, tables = _rope_nodes(ir)
    assert len(embeddings) == 2
    assert len(tables) == 2

    widths = {int(ir.constants[t].shape[-1]) for t in tables}
    qwen_theta, qwen_head_dim = QWEN_INV_FREQ["v_module_backbone_rotary_emb_inv_freq"]
    assert widths == {qwen_head_dim}

    inv = _inv_freq(qwen_theta, qwen_head_dim).to(torch.float64)
    positions = torch.tensor([0.0, 1.0, 2049.0, TEST_MAX_SEQ - 1], dtype=torch.float64).reshape(-1, 1)
    freqs = positions * inv.reshape(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    for node in ir.nodes.values():
        if node.op != "embedding" or not node.inputs[0].startswith("c_rope_table_"):
            continue
        table = ir.constants[node.inputs[0]]
        ref = (torch.cos(emb) if "cos" in node.inputs[0] else torch.sin(emb)).to(torch.float16)
        idx = [0, 1, 2049, TEST_MAX_SEQ - 1]
        assert torch.equal(table[idx], ref)
        assert node.inputs[1] == pos_for_output[node.outputs[0]]


# --------------------------------------------------------------------------- #
# inert: lm_encoder has no rope angle path
# --------------------------------------------------------------------------- #


def test_lm_encoder_is_inert() -> None:
    ir = _load_ir(GEMMA_BUNDLE / "optimized_ir_lm_encoder_step.json", GEMMA_INV_FREQ)
    assert precompute_rope_tables(ir) is False
    embeddings, tables = _rope_nodes(ir)
    assert tables == []


def test_vision_encoder_is_inert() -> None:
    """The vision tower's axial rope has the same cat(halves) <- matmul(inv_freq,
    position) shape as decoder rope, but its position is a mid-graph ``select``
    (not a graph input). The pass must NOT fire, or it corrupts vision RoPE. The
    inv_freq constant is materialized so the pass is rejected by the position
    graph-input check, not by a missing constant."""
    path = GEMMA_BUNDLE / "optimized_ir_vision_encoder.json"
    if not path.exists():
        pytest.skip("no vision encoder component for this bundle")
    graph = json.loads(path.read_text())["graph"]
    spec = {
        cid: (10000.0, int(graph["values"][cid]["shape"][0]) * 2)
        for cid in graph["constants"]
        if "inv_freq" in cid
    }
    assert spec, "expected a vision rope inv_freq constant"
    assert any(n["op"] in {"scalar_cos", "scalar_sin"} for n in graph["nodes"]), "vision encoder should have rope trig"
    ir = _load_ir(path, spec)
    assert precompute_rope_tables(ir) is False
    _, tables = _rope_nodes(ir)
    assert tables == []
