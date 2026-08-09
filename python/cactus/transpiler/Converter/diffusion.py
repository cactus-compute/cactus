from __future__ import annotations

from typing import Any

import torch

from . import constants
from ..ModelProfiles import models as MP_Models


class TextEncoderLastHiddenState(torch.nn.Module):
    #Wraps the inner transformer so exported parameter targets match the checkpoint's text_model.* keys
    def __init__(self, clip_text_model: torch.nn.Module):
        super().__init__()
        self.text_model = clip_text_model.text_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model(input_ids).last_hidden_state


class TaesdDecode(torch.nn.Module):
    #The attribute name keeps exported parameter targets on the checkpoint's decoder.* keys
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class UnetEpsilon(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
        ).sample


def create_diffusion_model(
    mp: MP_Models.ModelProfile,
    model_id: str,
    inference_mode: str,
    input_cls: Any,
    model_cls: Any,
) -> Any:
    from .models import EXPORT_PATCHES, load_configs

    kind, source = component_source(mp, inference_mode)
    configs = load_configs(mp, model_id)
    for patch in mp.export_patches:
        EXPORT_PATCHES[patch]()
    module = load_component(kind, source, model_id)
    input_ = input_cls(
        args=(),
        kwargs=component_export_inputs(kind, configs),
        modalities=("text",),
        inference_mode=inference_mode,
    )
    return model_cls(name=model_id, model_profile=mp, input=input_, model=module)


def component_source(mp: MP_Models.ModelProfile, inference_mode: str) -> tuple[str, str]:
    for mode, spec in getattr(mp, "component_sources", ()):
        if mode == inference_mode:
            kind, _, source = spec.partition(":")
            return kind, source
    raise ValueError(
        f"profile {mp.model_profiles!r} declares no component source for mode {inference_mode!r}"
    )


def load_component(kind: str, source: str, model_id: str) -> torch.nn.Module:
    if kind == "clip_text":
        from transformers import CLIPTextModel

        return TextEncoderLastHiddenState(from_pretrained_local_first(CLIPTextModel, model_id, subfolder=source))
    if kind == "sd_unet":
        from diffusers import UNet2DConditionModel

        return UnetEpsilon(from_pretrained_local_first(UNet2DConditionModel, model_id, subfolder=source))
    if kind == "taesd_decoder":
        from diffusers import AutoencoderTiny

        return TaesdDecode(from_pretrained_local_first(AutoencoderTiny, source).decoder)
    raise ValueError(f"unknown diffusion component kind {kind!r}")


def from_pretrained_local_first(model_class: Any, repo_id: str, **kwargs: Any) -> torch.nn.Module:
    if constants.token is not None:
        kwargs.setdefault("token", constants.token)
    try:
        model = model_class.from_pretrained(repo_id, torch_dtype=torch.float16, local_files_only=True, **kwargs)
    except Exception:
        model = model_class.from_pretrained(repo_id, torch_dtype=torch.float16, **kwargs)
    return model.eval()


def component_export_inputs(kind: str, configs: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    unet_config = configs.get("unet/config.json", {})
    text_config = configs.get("text_encoder/config.json", {})
    tokens = int(text_config.get("max_position_embeddings", 77))
    latent_channels = int(unet_config.get("in_channels", 4))
    #Graphs are shape-specialized; export 64x64 latents (512x512 images) rather than the config's training sample_size
    latent_size = 64
    if kind == "clip_text":
        return {"input_ids": torch.zeros((1, tokens), dtype=torch.int64)}
    if kind == "sd_unet":
        return {
            "sample": torch.zeros((1, latent_channels, latent_size, latent_size), dtype=torch.float16),
            "timestep": torch.full((1,), 999.0, dtype=torch.float16),
            "encoder_hidden_states": torch.zeros(
                (1, tokens, int(unet_config.get("cross_attention_dim", 768))), dtype=torch.float16
            ),
            "timestep_cond": torch.zeros(
                (1, int(unet_config.get("time_cond_proj_dim", 256))), dtype=torch.float16
            ),
        }
    if kind == "taesd_decoder":
        return {"x": torch.zeros((1, latent_channels, latent_size, latent_size), dtype=torch.float16)}
    raise ValueError(f"unknown diffusion component kind {kind!r}")
