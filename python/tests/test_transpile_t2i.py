"""End-to-end text-to-image through transpiled graphs only.

CLIP text encoder, LCM UNet, and TAESD decoder each go through
capture_model -> transpile_ir; the LCM scheduler step stays host-side numpy.
The reference is the identical loop run in torch fp32.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

try:
    from cactus.transpile.capture_pytorch import capture_model
    from cactus.transpile.lower import transpile_ir
except Exception as exc:  # pragma: no cover
    pytest.skip(f"cactus runtime library unavailable: {exc}", allow_module_level=True)

SD_MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
TAESD_MODEL_ID = "madebyollin/taesd"
PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 4
GUIDANCE = 8.5
SEED = 1234
TIMESTEP_SCALING = 10.0
SIGMA_DATA = 0.5


def _load(cls_name, model_id, subfolder, dtype):
    if cls_name == "CLIPTextModel":
        from transformers import CLIPTextModel as cls
    elif cls_name == "UNet2DConditionModel":
        from diffusers import UNet2DConditionModel as cls
    else:
        from diffusers import AutoencoderTiny as cls
    try:
        kwargs = {"subfolder": subfolder} if subfolder else {}
        return cls.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True, **kwargs).eval()
    except Exception as exc:
        pytest.skip(f"{model_id} not cached: {exc}")


class _LastHiddenState(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids):
        return self.text_model(input_ids).last_hidden_state


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


def _guidance_embedding(w: float, embedding_dim: int = 256) -> torch.Tensor:
    scaled = torch.tensor([w * 1000.0])
    half_dim = embedding_dim // 2
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-np.log(10000.0) / (half_dim - 1)))
    angles = scaled[:, None] * freq[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def _scheduler_scalars():
    from diffusers import LCMScheduler

    sched = LCMScheduler.from_pretrained(SD_MODEL_ID, subfolder="scheduler", local_files_only=True)
    assert sched.config.prediction_type == "epsilon"
    assert float(sched.config.timestep_scaling) == TIMESTEP_SCALING
    sched.set_timesteps(STEPS)
    timesteps = sched.timesteps
    rows = []
    for i, t in enumerate(timesteps):
        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else t
        a_t, a_prev = sched.alphas_cumprod[t], sched.alphas_cumprod[prev_t]
        st = float(t) * TIMESTEP_SCALING
        c_skip = SIGMA_DATA**2 / (st * st + SIGMA_DATA**2)
        c_out = st / (st * st + SIGMA_DATA**2) ** 0.5
        rows.append((float(t), a_t.sqrt().item(), (1 - a_t).sqrt().item(), c_skip, c_out,
                     a_prev.sqrt().item(), (1 - a_prev).sqrt().item()))
    return rows


def _lcm_loop(scalars, latent, noises, denoise):
    x = latent
    for i, (t, sa, sb, c_skip, c_out, sap, sbp) in enumerate(scalars):
        eps = denoise(x, t)
        x0 = (x - sb * eps) / sa
        denoised = c_out * x0 + c_skip * x
        x = sap * denoised + sbp * noises[i] if i < len(scalars) - 1 else denoised
    return x


def test_t2i_pipeline_matches_torch():
    from transformers import CLIPTokenizer

    try:
        tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL_ID, subfolder="tokenizer", local_files_only=True)
    except Exception as exc:
        pytest.skip(f"{SD_MODEL_ID} not cached: {exc}")

    ids = tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).input_ids

    scalars = _scheduler_scalars()
    generator = torch.Generator("cpu").manual_seed(SEED)
    latent = torch.randn(1, 4, 64, 64, generator=generator)
    noises = [torch.randn(1, 4, 64, 64, generator=generator) for _ in range(STEPS - 1)]
    w_emb = _guidance_embedding(GUIDANCE - 1.0)

    # --- transpiled pipeline ---
    text_graph = transpile_ir(
        capture_model(_LastHiddenState(_load("CLIPTextModel", SD_MODEL_ID, "text_encoder", torch.float16)), (ids,)).ir_graph
    )
    start = time.perf_counter()
    text_graph.set_input(0, ids.numpy())
    text_emb = np.asarray(text_graph.execute()[0].numpy(), dtype=np.float16)
    clip_ms = (time.perf_counter() - start) * 1000.0

    unet_example = (
        latent.to(torch.float16),
        torch.tensor([scalars[0][0]], dtype=torch.float16),
        torch.from_numpy(text_emb),
        w_emb.to(torch.float16),
    )
    unet_graph = transpile_ir(
        capture_model(_Denoiser(_load("UNet2DConditionModel", SD_MODEL_ID, "unet", torch.float16)), unet_example).ir_graph
    )

    def denoise_cactus(x, t):
        unet_graph.set_input(0, x.astype(np.float16))
        unet_graph.set_input(1, np.array([t], dtype=np.float16))
        unet_graph.set_input(2, text_emb)
        unet_graph.set_input(3, w_emb.numpy().astype(np.float16))
        return np.asarray(unet_graph.execute()[0].numpy(), dtype=np.float32)

    start = time.perf_counter()
    final_latent = _lcm_loop(scalars, latent.numpy().astype(np.float32), [n.numpy() for n in noises], denoise_cactus)
    unet_ms = (time.perf_counter() - start) * 1000.0 / STEPS

    decoder = _load("AutoencoderTiny", TAESD_MODEL_ID, None, torch.float32).decoder
    decoder_graph = transpile_ir(capture_model(decoder, (torch.randn(1, 4, 64, 64),)).ir_graph)
    start = time.perf_counter()
    decoder_graph.set_input(0, final_latent.astype(np.float16))
    image = np.asarray(decoder_graph.execute()[0].numpy(), dtype=np.float32).clip(0.0, 1.0)
    decode_ms = (time.perf_counter() - start) * 1000.0

    # --- torch fp32 reference, same math ---
    ref_text = _LastHiddenState(_load("CLIPTextModel", SD_MODEL_ID, "text_encoder", torch.float32))
    ref_unet = _Denoiser(_load("UNet2DConditionModel", SD_MODEL_ID, "unet", torch.float32))
    with torch.no_grad():
        ref_emb = ref_text(ids)

        def denoise_torch(x, t):
            with torch.no_grad():
                return ref_unet(torch.from_numpy(x), torch.tensor([t]), ref_emb, w_emb).numpy()

        ref_latent = _lcm_loop(scalars, latent.numpy().astype(np.float32), [n.numpy() for n in noises], denoise_torch)
        ref_image = decoder(torch.from_numpy(ref_latent)).clamp(0.0, 1.0).numpy()

    assert image.shape == ref_image.shape == (1, 3, 512, 512)
    latent_rel = np.abs(final_latent - ref_latent).mean() / np.abs(ref_latent).mean()
    mse = np.mean((image - ref_image) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else np.inf
    print(
        f"\n  t2i 512x512 {STEPS}-step: clip {clip_ms:.0f} ms  unet {unet_ms:.0f} ms/step  decode {decode_ms:.0f} ms"
        f"  latent_rel_err={latent_rel * 100:.2f}%  psnr={psnr:.1f} dB"
    )

    assert latent_rel < 0.05, f"final latent drifted from torch: {latent_rel * 100:.2f}%"
    assert psnr > 35.0, f"decoded image diverged from torch: {psnr:.1f} dB"
