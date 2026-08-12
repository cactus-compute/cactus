from __future__ import annotations

from typing import Any

import torch


class TextEncoderLastHiddenState(torch.nn.Module):
    #Wraps the inner transformer so exported parameter targets match the checkpoint's text_model.* keys
    def __init__(self, clip_text_model: torch.nn.Module):
        super().__init__()
        self.text_model = clip_text_model.text_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model(input_ids).last_hidden_state


class TaesdDecode(torch.nn.Module):
    #The attribute name keeps exported parameter targets on the checkpoint's decoder.* keys
    def __init__(self, autoencoder: torch.nn.Module):
        super().__init__()
        self.decoder = autoencoder.decoder

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


def clip_text_class() -> Any:
    from transformers import CLIPTextModel

    return CLIPTextModel


def sd_unet_class() -> Any:
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel


def taesd_class() -> Any:
    from diffusers import AutoencoderTiny

    return AutoencoderTiny


#diffusers is an optional dependency, so the classes resolve on use rather than joining LOAD_STRATEGIES
COMPONENT_LOADERS = {
    "clip_text": (clip_text_class, TextEncoderLastHiddenState),
    "sd_unet": (sd_unet_class, UnetEpsilon),
    "taesd_decoder": (taesd_class, TaesdDecode),
}
