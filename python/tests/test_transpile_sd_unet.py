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

MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
LATENT_SHAPE = (1, 4, 64, 64)
TEXT_EMB_SHAPE = (1, 77, 768)
GUIDANCE_EMB_SHAPE = (1, 256)


class _Denoiser(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
        ).sample


def _denoiser(dtype):
    from diffusers import UNet2DConditionModel

    try:
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet", torch_dtype=dtype, local_files_only=True
        )
    except Exception as exc:
        pytest.skip(f"{MODEL_ID} not cached: {exc}")
    return _Denoiser(unet.eval())


def _example_inputs():
    torch.manual_seed(0)
    return (
        torch.randn(*LATENT_SHAPE),
        torch.tensor([999.0]),
        torch.randn(*TEXT_EMB_SHAPE),
        torch.randn(*GUIDANCE_EMB_SHAPE),
    )


def test_sd_unet_denoise_matches_torch_and_reports_latency():
    inputs = _example_inputs()
    half = tuple(a.to(torch.float16) for a in inputs)

    captured = capture_model(_denoiser(torch.float16), half)
    nodes = captured.ir_graph.nodes
    nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.op] = counts.get(node.op, 0) + 1
    assert counts["conv2d"] == 98
    assert counts["attention"] == 32
    assert counts["group_norm"] == 61
    assert counts["upsample_nearest2d"] == 3
    assert counts["silu"] == 68

    transpiled = transpile_ir(captured.ir_graph)
    for i, tensor in enumerate(half):
        transpiled.set_input(i, tensor.numpy())
    eps = np.asarray(transpiled.execute()[0].numpy(), dtype=np.float32)

    runs = 3
    start = time.perf_counter()
    for _ in range(runs):
        transpiled.execute()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs

    with torch.no_grad():
        reference = _denoiser(torch.float32)(*[a.float() for a in half]).float().numpy()

    assert eps.shape == reference.shape == LATENT_SHAPE
    diff = np.abs(reference - eps)
    relative = diff.mean() / np.abs(reference).mean()
    print(
        f"\n  sd unet denoise 64x64: {elapsed_ms:.0f} ms"
        f"  max_abs_diff={diff.max():.4f}  relative_mean_err={relative * 100:.3f}%"
    )

    assert relative < 0.01, f"transpiled denoise drifted from torch: {relative * 100:.3f}%"
