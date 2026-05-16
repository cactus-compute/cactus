from cactus.convert.model_adapters.naming import cactus_name_for_tensor
from cactus.convert.model_adapters.policy import policy_for_tensor
from cactus.convert.cli import _bits_for_component
from cactus.convert.model_adapters.detection import detect_family
from cactus.convert.cactus_adapters.config_utils import extract_parakeet_tdt_config, extract_whisper_config
from cactus.convert.cli import _augment_state_dict_for_family
from cactus.convert.model_adapters.adapters import adapter_for_family
import torch


def test_policy_embedding_cq4():
    match = cactus_name_for_tensor("model.embed_tokens.weight", "generic", 1)
    p = policy_for_tensor(match, (100, 64), 2, "generic")
    assert p.precision == "CQ4"
    assert p.rotation == "orthogonal"


def test_adapter_registry_preserves_family_names():
    assert adapter_for_family("qwen").family == "qwen"
    assert adapter_for_family("lfm2").family == "lfm2"


def test_policy_gemma4_pli_cq2():
    name = "model.language_model.embed_tokens_per_layer.weight"
    match = cactus_name_for_tensor(name, "gemma4", 1)
    p = policy_for_tensor(match, (10, 128), 4, "gemma4")
    assert p.precision == "CQ2"


def test_gemma4_adapter_disables_gptq_for_unhookable_tensors():
    adapter = adapter_for_family("gemma4")
    shared_kv = cactus_name_for_tensor("model.language_model.layers.15.self_attn.k_proj.weight", "gemma4", 35)
    shared_policy = adapter.policy(shared_kv, (256, 1536), 4)
    assert shared_policy.precision == "CQ4"
    assert not shared_policy.use_gptq

    vision_proj = cactus_name_for_tensor("model.embed_vision.embedding_projection.weight", "gemma4", 35)
    vision_policy = adapter.policy(vision_proj, (1536, 1152), 4)
    assert vision_policy.precision == "CQ4"
    assert not vision_policy.use_gptq

    vision_tower = cactus_name_for_tensor("model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight", "gemma4", 35)
    vision_tower_policy = adapter.policy(vision_tower, (1152, 4304), 4)
    assert vision_tower_policy.precision == "CQ4"
    assert vision_tower_policy.use_gptq
    assert adapter.module_target_name("model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight", None) == "model.vision_tower.encoder.layers.0.mlp.down_proj.linear"


def test_policy_audio_no_gptq():
    match = cactus_name_for_tensor("model.audio_tower.output_proj.weight", "gemma4", 1)
    p = policy_for_tensor(match, (128, 128), 4, "gemma4")
    assert p.component == "audio"
    assert not p.use_gptq


def test_policy_bias_int8():
    match = cactus_name_for_tensor("model.audio_tower.output_proj.bias", "gemma4", 1)
    p = policy_for_tensor(match, (1536,), 4, "gemma4")
    assert p.precision == "INT8"
    assert p.bits == 8


def test_policy_position_embedding_fp16():
    match = cactus_name_for_tensor("model.vision_tower.vision_model.embeddings.position_embedding.weight", "lfm2", 1)
    p = policy_for_tensor(match, (256, 1152), 4, "lfm2")
    assert p.precision == "FP16"
    assert p.fallback_reason == "position embedding tensor"


def test_policy_lfm_depthwise_conv_int8():
    match = cactus_name_for_tensor("model.layers.0.conv.conv.weight", "lfm2", 16)
    p = policy_for_tensor(match, (1024, 1, 3), 4, "lfm2")
    assert p.precision == "INT8"
    assert p.fallback_reason == "depthwise conv tensor"


def test_cli_component_bit_overrides():
    class Args:
        bits = 4
        language_bits = 2
        vision_bits = 4
        audio_bits = 3
        embedding_bits = None

    assert _bits_for_component("language", Args) == 2
    assert _bits_for_component("vision", Args) == 4
    assert _bits_for_component("audio", Args) == 3
    assert _bits_for_component("transcription", Args) == 3
    assert _bits_for_component("embedding", Args) == 4


def test_policy_parakeet_transcription_no_gptq():
    match = cactus_name_for_tensor("encoder.layers.0.self_attn.q_proj.weight", "parakeet", 24)
    p = policy_for_tensor(match, (1024, 1024), 4, "parakeet")
    assert p.component == "transcription"
    assert p.precision == "CQ4"
    assert not p.use_gptq


def test_parakeet_tdt_hf_config_detection_and_extraction():
    cfg = {
        "model_type": "parakeet_tdt",
        "vocab_size": 8193,
        "pad_token_id": 2,
        "blank_token_id": 8192,
        "decoder_hidden_size": 640,
        "num_decoder_layers": 2,
        "durations": [0, 1, 2, 3, 4],
        "encoder_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 4096,
            "max_position_embeddings": 5000,
            "conv_kernel_size": 9,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
            "num_mel_bins": 128,
            "hidden_act": "silu",
        },
    }
    assert detect_family(cfg) == "parakeet_tdt"
    extracted = extract_parakeet_tdt_config(cfg)
    assert extracted["num_layers"] == 24
    assert extracted["num_mel_bins"] == 128
    assert extracted["predictor_hidden_dim"] == 640
    assert extracted["predictor_num_layers"] == 2
    assert extracted["tdt_durations"] == [0, 1, 2, 3, 4]
    assert extracted["tdt_blank_id"] == 8192


def test_whisper_hf_config_extraction():
    cfg = {
        "model_type": "whisper",
        "vocab_size": 51865,
        "d_model": 384,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "encoder_attention_heads": 6,
        "decoder_attention_heads": 6,
        "encoder_ffn_dim": 1536,
        "decoder_ffn_dim": 1536,
        "max_target_positions": 448,
        "num_mel_bins": 80,
        "pad_token_id": 50257,
        "bos_token_id": 50257,
        "eos_token_id": 50257,
        "tie_word_embeddings": False,
    }
    extracted = extract_whisper_config(cfg)
    assert detect_family(cfg) == "whisper"
    assert extracted["hidden_dim"] == 384
    assert extracted["num_encoder_layers"] == 4
    assert extracted["num_decoder_layers"] == 4
    assert extracted["attention_head_dim"] == 64
    assert extracted["context_length"] == 448
    assert not extracted["tie_word_embeddings"]


def test_parakeet_tdt_lstm_biases_are_combined_for_runtime():
    state = {
        "decoder.lstm.bias_ih_l0": torch.tensor([1.0, 2.0]),
        "decoder.lstm.bias_hh_l0": torch.tensor([0.5, -1.0]),
    }
    out = _augment_state_dict_for_family(state, "parakeet_tdt")
    assert "decoder.lstm.bias_ih_l0" not in out
    assert "decoder.lstm.bias_hh_l0" not in out
    assert torch.equal(out["decoder.lstm.bias_l0"], torch.tensor([1.5, 1.0]))


def test_parakeet_tdt_lstm_bias_normalization_records_provenance():
    state = {
        "decoder.lstm.bias_ih_l0": torch.tensor([1.0, 2.0]),
        "decoder.lstm.bias_hh_l0": torch.tensor([0.5, -1.0]),
    }
    normalized = adapter_for_family("parakeet_tdt").normalize_state_dict(state)
    provenance = normalized.provenance["decoder.lstm.bias_l0"]
    assert provenance.source_names == ["decoder.lstm.bias_ih_l0", "decoder.lstm.bias_hh_l0"]
    assert provenance.transform == "parakeet_tdt_lstm_bias_sum"


def test_policy_parakeet_pointwise_conv_int8_and_conv_bias_fp16():
    weight = cactus_name_for_tensor("encoder.layers.0.conv.pointwise_conv1.weight", "parakeet_tdt", 24)
    weight_policy = policy_for_tensor(weight, (2048, 1024, 1), 4, "parakeet_tdt")
    assert weight_policy.precision == "INT8"
    assert weight_policy.fallback_reason == "pointwise conv tensor"

    bias = cactus_name_for_tensor("encoder.layers.0.conv.pointwise_conv1.bias", "parakeet_tdt", 24)
    bias_policy = policy_for_tensor(bias, (2048,), 4, "parakeet_tdt")
    assert bias_policy.precision == "FP16"
    assert bias_policy.fallback_reason == "conv bias tensor"


def test_parakeet_adapter_conv_transform_is_scoped():
    adapter = adapter_for_family("parakeet_tdt")
    match = cactus_name_for_tensor("encoder.pre_encode.conv.0.weight", "parakeet_tdt", 24)
    tensor = torch.randn(256, 3, 3, 1)
    transformed, transform = adapter.transform_tensor(match, tensor)
    assert transform == "parakeet_conv_hwio_to_oihw"
    assert transformed.shape == (256, 1, 3, 3)

    other = cactus_name_for_tensor("encoder.layers.0.feed_forward1.linear1.weight", "parakeet_tdt", 24)
    untouched, transform = adapter.transform_tensor(other, torch.randn(4, 4))
    assert transform == "none"
    assert untouched.shape == (4, 4)
