from __future__ import annotations

import time

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

try:
    from cactus.transpile.capture_pytorch import capture_model
    from cactus.transpile.lower import transpile_ir
except Exception as exc:  # pragma: no cover
    pytest.skip(f"cactus runtime library unavailable: {exc}", allow_module_level=True)

MODEL_ID = "madebyollin/taesd"
LATENT_SHAPE = (1, 4, 64, 64)


def _decoder():
    from diffusers import AutoencoderTiny

    try:
        vae = AutoencoderTiny.from_pretrained(MODEL_ID, torch_dtype=torch.float32, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"{MODEL_ID} not cached: {exc}")
    return vae.eval().decoder


def _ir_nodes(captured):
    nodes = captured.ir_graph.nodes
    return list(nodes.values()) if isinstance(nodes, dict) else list(nodes)


def test_taesd_decoder_transpiles_to_the_expected_ops():
    torch.manual_seed(0)
    captured = capture_model(_decoder(), (torch.randn(*LATENT_SHAPE),))
    counts: dict[str, int] = {}
    for node in _ir_nodes(captured):
        counts[node.op] = counts.get(node.op, 0) + 1

    assert counts["conv2d"] == 35
    assert counts["upsample_nearest2d"] == 3
    assert counts["relu"] == 31
    assert set(counts) == {
        "conv2d", "relu", "add", "upsample_nearest2d",
        "scalar_multiply", "scalar_divide", "tanh", "scalar_subtract",
    }


def test_taesd_decode_matches_torch_and_reports_latency():
    torch.manual_seed(0)
    decoder = _decoder()
    latent = torch.randn(*LATENT_SHAPE)

    transpiled = transpile_ir(capture_model(decoder, (latent,)).ir_graph)
    transpiled.set_input(0, latent.numpy().astype(np.float16))
    decoded = np.asarray(transpiled.execute()[0].numpy(), dtype=np.float32)

    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        transpiled.execute()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs

    with torch.no_grad():
        reference = decoder(latent).float().numpy()

    assert decoded.shape == reference.shape == (1, 3, 512, 512)
    diff = np.abs(reference - decoded)
    relative = diff.mean() / np.abs(reference).mean()
    print(
        f"\n  taesd decode 512x512: {elapsed_ms:.1f} ms"
        f"  max_abs_diff={diff.max():.4f}  relative_mean_err={relative * 100:.3f}%"
    )

    assert relative < 0.01, f"transpiled decode drifted from torch: {relative * 100:.3f}%"
    assert diff.max() < 0.2, f"transpiled decode has an outlier: {diff.max():.4f}"
