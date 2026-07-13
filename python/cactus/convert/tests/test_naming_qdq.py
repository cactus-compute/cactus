from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.convert.export.qdq import convert_qdq
from cactus.convert.model_adapters.adapters import adapter_for_family
from cactus.convert.model_adapters.naming import cactus_name_for_tensor, restore_hf_key_for_family


def test_gemma4_audio_aux_roundtrip_name():
    hf = "model.audio_tower.layers.0.self_attn.q_proj.input_min"
    match = cactus_name_for_tensor(hf, "gemma4", 35)
    assert match.output_name == "audio_conformer_0_attention_attn_q_proj_input_min.weights"
    assert match.hf_name == hf
    assert match.adapter_name == "model.audio_tower.conformer.0.attention.attn.q_proj.input_min"
    assert restore_hf_key_for_family(match.adapter_name, "gemma4") == hf
    assert "q_proj_proj" not in restore_hf_key_for_family(match.adapter_name, "gemma4")


def test_needle_encoder_layers_beyond_decoder_depth_are_recognized():
    match = cactus_name_for_tensor("model.encoder.layers.11.self_attn.k_proj.weight", "needle", 12)
    assert match.recognized
    assert match.output_name == "encoder_layer_11_attn_k.weights"


def test_needle_global_norms_map_to_bundle_names():
    encoder_norm = cactus_name_for_tensor("model.encoder.final_norm.weight", "needle")
    assert encoder_norm.recognized
    assert encoder_norm.output_name == "encoder_layer_norm_weight.weights"
    decoder_norm = cactus_name_for_tensor("model.decoder.norm.weight", "needle")
    assert decoder_norm.recognized
    assert decoder_norm.output_name == "output_norm.weights"


def test_manifest_qdq_restores_aux_stats_without_suffix(tmp_path):
    cactus = tmp_path / "cactus"
    out = tmp_path / "qdq"
    cactus.mkdir()
    tensor = torch.tensor([0.25], dtype=torch.float16)
    output_file = "audio_conformer_0_attention_attn_q_proj_input_min.weights"
    save_tensor_with_header(tensor, cactus / output_file, precision="FP16")
    (cactus / "conversion_manifest.json").write_text(
        """[
  {
    "source_name": "model.audio_tower.conformer.0.attention.attn.q_proj.input_min",
    "hf_name": "model.audio_tower.layers.0.self_attn.q_proj.input_min",
    "adapter_name": "model.audio_tower.conformer.0.attention.attn.q_proj.input_min",
    "output_file": "audio_conformer_0_attention_attn_q_proj_input_min.weights",
    "shape": [1],
    "dtype": "torch.float16",
    "component": "audio",
    "policy": "fallback",
    "precision": "FP16",
    "status": "fallback",
    "required": true,
    "scale_factor": 1.0
  }
]""",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input=cactus,
        out=out,
        dtype="float16",
        model_family="gemma4",
        shard_size_gb=1.0,
        row_batch_size=64,
        tmp_dir=None,
        force=True,
    )
    report = convert_qdq(args)
    assert report["written_count"] == 1
    tensors = load_file(out / "model.safetensors")
    assert set(tensors) == {"model.audio_tower.layers.0.self_attn.q_proj.input_min"}
    assert tensors["model.audio_tower.layers.0.self_attn.q_proj.input_min"].shape == (1,)


def test_manifest_qdq_trims_int8_bias_padding(tmp_path):
    cactus = tmp_path / "cactus"
    out = tmp_path / "qdq"
    cactus.mkdir()
    output_file = "projector_linear1.bias.weights"
    save_tensor_with_header(torch.tensor([1.0, -2.0, 3.0]), cactus / output_file, precision="INT8", allow_int8_bias=True)
    (cactus / "conversion_manifest.json").write_text(
        """[
  {
    "source_name": "model.multi_modal_projector.linear_1.bias",
    "hf_name": "model.multi_modal_projector.linear_1.bias",
    "adapter_name": "model.multi_modal_projector.linear_1.bias",
    "output_file": "projector_linear1.bias.weights",
    "shape": [3],
    "dtype": "torch.float32",
    "component": "vision",
    "policy": "fallback",
    "precision": "INT8",
    "status": "fallback",
    "required": true,
    "scale_factor": 1.0
  }
]""",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input=cactus,
        out=out,
        dtype="float16",
        model_family="generic",
        shard_size_gb=1.0,
        row_batch_size=64,
        tmp_dir=None,
        force=True,
    )
    convert_qdq(args)
    tensors = load_file(out / "model.safetensors")
    assert tensors["model.multi_modal_projector.linear_1.bias"].shape == (3,)


def test_manifest_qdq_uses_adapter_output_keys(tmp_path):
    cactus = tmp_path / "cactus"
    out = tmp_path / "qdq"
    cactus.mkdir()
    output_file = "tdt_predictor_lstm_0_bias.weights"
    save_tensor_with_header(torch.tensor([1.0, 2.0]), cactus / output_file, precision="FP16")
    (cactus / "conversion_manifest.json").write_text(
        """[
  {
    "source_name": "decoder.lstm.bias_l0",
    "hf_name": "decoder.lstm.bias_l0",
    "adapter_name": "decoder.lstm.bias_l0",
    "source_names": ["decoder.lstm.bias_ih_l0", "decoder.lstm.bias_hh_l0"],
    "transform": "parakeet_tdt_lstm_bias_sum",
    "qdq_restore": "hf_key",
    "output_file": "tdt_predictor_lstm_0_bias.weights",
    "shape": [2],
    "dtype": "torch.float32",
    "component": "transcription",
    "policy": "fallback",
    "precision": "FP16",
    "status": "fallback",
    "required": true,
    "scale_factor": 1.0
  }
]""",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input=cactus,
        out=out,
        dtype="float16",
        model_family="parakeet_tdt",
        shard_size_gb=1.0,
        row_batch_size=64,
        tmp_dir=None,
        force=True,
    )
    convert_qdq(args)
    tensors = load_file(out / "model.safetensors")
    assert set(tensors) == {"decoder.lstm.bias_l0"}


def test_manifest_qdq_restores_int8_depthwise_conv_shape(tmp_path):
    from cactus.convert.cactus_adapters.tensor_io import save_depthwise_conv_int8_with_header

    cactus = tmp_path / "cactus"
    out = tmp_path / "qdq"
    cactus.mkdir()
    output_file = "layer_0_conv_depthwise.weights"
    save_depthwise_conv_int8_with_header(torch.randn(4, 1, 3), cactus / output_file)
    (cactus / "conversion_manifest.json").write_text(
        """[
  {
    "source_name": "model.layers.0.conv.conv.weight",
    "hf_name": "model.layers.0.conv.conv.weight",
    "adapter_name": "model.layers.0.conv.conv.weight",
    "output_file": "layer_0_conv_depthwise.weights",
    "shape": [4, 1, 3],
    "dtype": "torch.float32",
    "component": "language",
    "policy": "fallback",
    "precision": "INT8",
    "status": "fallback",
    "required": true,
    "scale_factor": 1.0
  }
]""",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        input=cactus,
        out=out,
        dtype="float16",
        model_family="lfm2",
        shard_size_gb=1.0,
        row_batch_size=64,
        tmp_dir=None,
        force=True,
    )
    convert_qdq(args)
    tensors = load_file(out / "model.safetensors")
    assert tensors["model.layers.0.conv.conv.weight"].shape == (4, 1, 3)


def test_lfm_vl_siglip2_layer_names_are_cactus_runtime_names():
    cases = {
        "model.vision_tower.vision_model.encoder.layers.3.layer_norm1.bias": "vision_layer_3_layer_norm1.bias.weights",
        "model.vision_tower.vision_model.encoder.layers.3.layer_norm2.weight": "vision_layer_3_layer_norm2.weights",
        "model.vision_tower.vision_model.encoder.layers.3.mlp.fc2.weight": "vision_layer_3_ffn_fc2.weights",
        "model.vision_tower.vision_model.encoder.layers.3.self_attn.out_proj.bias": "vision_layer_3_self_attn_out.bias.weights",
    }
    for source, expected in cases.items():
        match = cactus_name_for_tensor(source, "lfm2", 16)
        assert match.recognized
        assert match.component == "vision"
        assert match.output_name == expected


def test_qwen_vl_visual_names_are_cactus_style():
    cases = {
        "model.visual.patch_embed.proj.weight": "vision_patch_embedding.weights",
        "model.visual.blocks.12.attn.qkv.bias": "vision_layer_12_self_attn_qkv.bias.weights",
        "model.visual.blocks.12.mlp.linear_fc2.weight": "vision_layer_12_ffn_fc2.weights",
        "model.visual.blocks.12.norm1.weight": "vision_layer_12_layer_norm1.weights",
        "model.visual.merger.norm.bias": "vision_merger_norm.bias.weights",
        "model.visual.deepstack_merger_list.2.linear_fc1.bias": "vision_deepstack_merger_2_linear_fc1.bias.weights",
    }
    for source, expected in cases.items():
        match = cactus_name_for_tensor(source, "qwen", 28)
        assert match.recognized
        assert match.component == "vision"
        assert match.output_name == expected


def test_whisper_layer_names_include_runtime_block_prefix():
    cases = {
        "encoder.layers.2.self_attn.q_proj.weight": "encoder.layer_2_self_attn_q.weights",
        "model.encoder.layers.2.self_attn.q_proj.weight": "encoder.layer_2_self_attn_q.weights",
        "model.encoder.conv1.weight": "encoder_conv1_weight.weights",
        "encoder.layers.2.fc1.bias": "encoder.layer_2_mlp_fc1.bias",
        "decoder.layers.4.encoder_attn.out_proj.bias": "decoder.layer_4_encoder_attn_output.bias",
        "model.decoder.layers.4.encoder_attn.out_proj.bias": "decoder.layer_4_encoder_attn_output.bias",
        "decoder.layers.4.self_attn_layer_norm.weight": "decoder.layer_4_self_attn_norm.weights",
        "model.decoder.embed_tokens.weight": "decoder_token_embeddings.weights",
        "proj_out.weight": "output_weight.weights",
    }
    for source, expected in cases.items():
        match = cactus_name_for_tensor(source, "whisper", 8)
        assert match.recognized
        assert match.output_name == expected
        if source.startswith(("encoder.layers.", "decoder.layers.")):
            assert match.component == "transcription"


def test_whisper_unknown_layer_name_is_not_recognized():
    match = cactus_name_for_tensor("encoder.layers.2.some_new_projection.weight", "whisper", 8)
    assert not match.recognized
    assert match.output_name == "encoder_layers_2_some_new_projection.weights"


def test_parakeet_ctc_names_are_cactus_runtime_names():
    cases = {
        "encoder.subsampling.layers.0.weight": "subsampling_conv0_weight.weights",
        "encoder.subsampling.linear.bias": "subsampling_linear_bias.bias",
        "ctc_head.weight": "ctc_head_weight.weights",
        "encoder.layers.3.feed_forward1.linear1.weight": "layer_3_ff1_linear1.weights",
        "encoder.layers.3.self_attn.relative_k_proj.weight": "layer_3_self_attn_relative_k.weights",
        "encoder.layers.3.conv.norm.running_var": "layer_3_conv_batchnorm_running_var.weights",
        "encoder.layers.3.norm_self_att.bias": "layer_3_norm_self_attn.bias",
    }
    for source, expected in cases.items():
        match = cactus_name_for_tensor(source, "parakeet", 24)
        assert match.recognized
        assert match.component == "transcription"
        assert match.output_name == expected


def test_parakeet_tdt_names_are_cactus_runtime_names():
    cases = {
        "encoder.pre_encode.conv.0.weight": "subsampling_conv0_weight.weights",
        "decoder.embedding.weight": "tdt_predictor_embed.weights",
        "decoder.prediction.embed.weight": "tdt_predictor_embed.weights",
        "decoder.prediction.dec_rnn.lstm.1.Wx": "tdt_predictor_lstm_1_weight_ih.weights",
        "decoder.lstm.weight_ih_l1": "tdt_predictor_lstm_1_weight_ih.weights",
        "decoder.prediction.dec_rnn.lstm.1.Wh": "tdt_predictor_lstm_1_weight_hh.weights",
        "decoder.lstm.weight_hh_l1": "tdt_predictor_lstm_1_weight_hh.weights",
        "decoder.prediction.dec_rnn.lstm.1.bias": "tdt_predictor_lstm_1_bias.weights",
        "decoder.lstm.bias_l1": "tdt_predictor_lstm_1_bias.weights",
        "encoder_projector.weight": "tdt_joint_enc.weights",
        "decoder.decoder_projector.bias": "tdt_joint_pred.bias",
        "joint.joint_net.0.weight": "tdt_joint_out.weights",
        "joint.head.bias": "tdt_joint_out.bias",
        "encoder.layers.5.self_attention.linear_out.bias": "layer_5_self_attn_output.bias",
    }
    for source, expected in cases.items():
        match = cactus_name_for_tensor(source, "parakeet_tdt", 24)
        assert match.recognized
        assert match.component == "transcription"
        assert match.output_name == expected


def test_parakeet_batchnorm_tracking_tensors_are_ignored():
    match = cactus_name_for_tensor("encoder.layers.0.conv.norm.num_batches_tracked", "parakeet", 24)
    assert match.recognized
    assert match.output_name is None


def test_nomic_normalizes_global_tensors():
    adapter = adapter_for_family("nomic")
    state = {
        "embeddings.word_embeddings.weight": torch.ones(4, 3),
        "embeddings.token_type_embeddings.weight": torch.full((1, 3), 2.0),
        "emb_ln.weight": torch.arange(3.0),
        "emb_ln.bias": torch.arange(3.0) + 10,
    }
    normalized = adapter.normalize_state_dict(state)
    assert set(normalized.state_dict) == {"token_embeddings", "embedding_layernorm.weight", "embedding_layernorm.bias"}
    assert torch.equal(normalized.state_dict["token_embeddings"], torch.full((4, 3), 3.0))
    assert normalized.provenance["token_embeddings"].source_names == [
        "embeddings.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
    ]
    assert normalized.provenance["token_embeddings"].qdq_restore == "adapter_key"
    assert adapter.name_tensor("token_embeddings", normalized.state_dict["token_embeddings"], 12).output_name == "token_embeddings.weights"
    assert adapter.name_tensor("embedding_layernorm.weight", normalized.state_dict["embedding_layernorm.weight"], 12).output_name == "embedding_layernorm.weight"


def test_nomic_norm2_weight_uses_runtime_name():
    adapter = adapter_for_family("nomic")
    match = adapter.name_tensor("encoder.layers.3.norm2.weight", torch.ones(768), 12)
    assert match.recognized
    assert match.output_name == "layer_3_norm2.weights"


def test_nomic_keeps_qkv_and_moe_experts_fused():
    # The v2 transpile path binds graph weights by their HF parameter name, so the
    # converter emits one fused tensor per HF parameter (no q/k/v or per-expert split).
    # w2 is stored transposed so the second expert matmul can consume it as a direct
    # linear weight in the transpiled graph.
    adapter = adapter_for_family("nomic")
    adapter.num_experts = 8

    qkv = torch.arange(2304 * 2, dtype=torch.float32).reshape(2304, 2)
    match = adapter.name_tensor("encoder.layers.0.attn.Wqkv.weight", qkv, 12)
    emissions = adapter.expand_tensor(match, qkv)
    assert [e.output_name for e in emissions] == ["layer_0_attn_qkv.weights"]
    assert tuple(emissions[0].tensor.shape) == (2304, 2)

    w1 = torch.empty(24576, 2)
    match = adapter.name_tensor("encoder.layers.1.mlp.experts.mlp.w1", w1, 12)
    emissions = adapter.expand_tensor(match, w1)
    assert [e.output_name for e in emissions] == ["layer_1_mlp_experts_w1.weights"]
    assert tuple(emissions[0].tensor.shape) == (24576, 2)

    w2 = torch.empty(24576, 2)
    match = adapter.name_tensor("encoder.layers.1.mlp.experts.mlp.w2", w2, 12)
    emissions = adapter.expand_tensor(match, w2)
    assert [e.output_name for e in emissions] == ["layer_1_mlp_experts_w2.weights"]
    assert tuple(emissions[0].tensor.shape) == (2, 24576)


def test_nomic_qdq_runtime_keys_are_unique(tmp_path):
    cactus = tmp_path / "cactus"
    out = tmp_path / "qdq"
    cactus.mkdir()
    save_tensor_with_header(torch.ones(2, 3), cactus / "layer_0_attn_q.weights", precision="FP16")
    save_tensor_with_header(torch.ones(2, 3) * 2, cactus / "layer_0_attn_k.weights", precision="FP16")
    (cactus / "conversion_manifest.json").write_text(
        """[
  {
    "source_name": "encoder.layers.0.attn.Wqkv.weight",
    "hf_name": "encoder.layers.0.attn.Wqkv.weight",
    "adapter_name": "encoder.layers.0.attn.Wqkv.weight",
    "output_file": "layer_0_attn_q.weights",
    "shape": [2, 3],
    "dtype": "torch.float32",
    "component": "language",
    "policy": "fallback",
    "precision": "FP16",
    "status": "fallback",
    "required": true,
    "qdq_restore": "runtime_key",
    "scale_factor": 1.0
  },
  {
    "source_name": "encoder.layers.0.attn.Wqkv.weight",
    "hf_name": "encoder.layers.0.attn.Wqkv.weight",
    "adapter_name": "encoder.layers.0.attn.Wqkv.weight",
    "output_file": "layer_0_attn_k.weights",
    "shape": [2, 3],
    "dtype": "torch.float32",
    "component": "language",
    "policy": "fallback",
    "precision": "FP16",
    "status": "fallback",
    "required": true,
    "qdq_restore": "runtime_key",
    "scale_factor": 1.0
  }
]""",
        encoding="utf-8",
    )
    report = convert_qdq(
        SimpleNamespace(
            input=cactus,
            out=out,
            dtype="float16",
            model_family="nomic",
            shard_size_gb=1.0,
            row_batch_size=64,
            tmp_dir=None,
            force=True,
        )
    )
    tensors = load_file(out / "model.safetensors")
    assert report["written_count"] == 2
    assert set(tensors) == {"layer_0_attn_q", "layer_0_attn_k"}
