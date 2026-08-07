from __future__ import annotations

import torch

from cactus.convert.model_adapters.adapters import adapter_for_family
from cactus.convert.model_adapters.detection import detect_family


def test_diffusers_autoencoder_tiny_config_selects_taesd():
    assert detect_family({"_class_name": "AutoencoderTiny", "latent_channels": 4}) == "taesd"


def test_taesd_converts_from_checkpoint_without_a_transformers_model_class():
    adapter = adapter_for_family("taesd")
    assert adapter.model_class({"_class_name": "AutoencoderTiny"}) is None
    assert adapter.load_processor("madebyollin/taesd") is None


def test_taesd_keeps_encoder_and_decoder_names_distinct():
    adapter = adapter_for_family("taesd")
    tensor = torch.zeros(64, 64, 3, 3)
    encoder = adapter.name_tensor("encoder.layers.12.conv.0.weight", tensor, None)
    decoder = adapter.name_tensor("decoder.layers.12.conv.0.weight", tensor, None)
    assert encoder.output_name == "encoder_layers_12_conv_0.weights"
    assert decoder.output_name == "decoder_layers_12_conv_0.weights"


def test_taesd_conv_tensors_stay_fp16():
    adapter = adapter_for_family("taesd")
    match = adapter.name_tensor("decoder.layers.0.weight", torch.zeros(64, 4, 3, 3), None)
    policy = adapter.policy(match, (64, 4, 3, 3), 4)
    assert policy.precision == "FP16"
