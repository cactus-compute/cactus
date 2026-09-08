from __future__ import annotations

import time

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

try:
    from cactus.transpile.capture_pytorch import capture_model
    from cactus.transpile.lower import transpile_ir
except Exception as exc:  # pragma: no cover
    pytest.skip(f"cactus runtime library unavailable: {exc}", allow_module_level=True)

MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
BOS, EOS, MAX_LEN = 49406, 49407, 77


class _LastHiddenState(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids):
        return self.text_model(input_ids).last_hidden_state


def _text_encoder(dtype):
    from transformers import CLIPTextModel

    try:
        encoder = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
        )
    except Exception as exc:
        pytest.skip(f"{MODEL_ID} not cached: {exc}")
    return _LastHiddenState(encoder.eval())


def _token_ids():
    torch.manual_seed(0)
    ids = torch.randint(0, BOS, (1, MAX_LEN))
    ids[0, 0] = BOS
    ids[0, -1] = EOS
    return ids


def test_clip_text_encoder_transpiles_to_the_expected_ops():
    captured = capture_model(_text_encoder(torch.float32), (_token_ids(),))
    nodes = captured.ir_graph.nodes
    nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.op] = counts.get(node.op, 0) + 1

    assert counts["linear"] == 72
    assert counts["layer_norm"] == 25
    assert counts["attention"] == 12
    assert counts["embedding"] == 2
    assert counts["sigmoid"] == 12


def test_clip_text_encode_matches_torch_and_reports_latency():
    ids = _token_ids()
    transpiled = transpile_ir(capture_model(_text_encoder(torch.float16), (ids,)).ir_graph)
    transpiled.set_input(0, ids.numpy())
    encoded = np.asarray(transpiled.execute()[0].numpy(), dtype=np.float32)

    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        transpiled.execute()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs

    with torch.no_grad():
        reference = _text_encoder(torch.float32)(ids).float().numpy()

    assert encoded.shape == reference.shape == (1, MAX_LEN, 768)
    diff = np.abs(reference - encoded)
    relative = diff.mean() / np.abs(reference).mean()
    print(
        f"\n  clip text encode 77 tokens: {elapsed_ms:.1f} ms"
        f"  max_abs_diff={diff.max():.4f}  relative_mean_err={relative * 100:.3f}%"
    )

    assert relative < 0.01, f"transpiled encode drifted from torch: {relative * 100:.3f}%"
