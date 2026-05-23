from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from cactus.convert.cactus_adapters.tensor_io import save_tensor_with_header
from cactus.convert.export.qdq import convert_qdq
from cactus.convert.model_adapters.adapters import adapter_for_family
from cactus.convert.model_adapters.detection import detect_family


def _cfg() -> dict[str, object]:
    return {
        "model_type": "nomic_bert",
        "vocab_size": 250048,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_inner": 3072,
        "n_positions": 2048,
        "num_experts": 8,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
    }


def test_nomic_detection_and_runtime_config():
    adapter = adapter_for_family("nomic")
    assert detect_family(_cfg(), "auto") == "nomic"
    assert adapter.runtime_model_type() == "bert"
    runtime = adapter.runtime_config(_cfg())
    assert runtime["num_layers"] == 12
    assert runtime["hidden_dim"] == 768
    assert runtime["attention_heads"] == 12
    assert runtime["num_experts"] == 8
    assert runtime["num_experts_per_tok"] == 2
    assert runtime["moe_every_n_layers"] == 2


def test_lfm2_vl_adapter_selects_runtime_safe_model_class():
    from transformers import Lfm2VlForConditionalGeneration

    adapter = adapter_for_family("lfm2")
    cfg = {"model_type": "lfm2", "architectures": ["Lfm2VlForConditionalGeneration"]}
    assert adapter.model_class(cfg) is Lfm2VlForConditionalGeneration


def test_lfm2_processor_fallback_handles_tokenizers_backend(tmp_path):
    import json

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Lfm2VlProcessor

    tokenizer = Tokenizer(WordLevel({"<|pad|>": 0, "<|startoftext|>": 1, "<|im_end|>": 2, "<image>": 3, "hello": 4}, unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "bos_token": "<|startoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|pad|>",
                "image_token": "<image>",
                "image_start_token": "<image>",
                "image_end_token": "<image>",
                "image_thumbnail": "<image>",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "Lfm2VlImageProcessorFast",
                "do_resize": True,
                "size": {"height": 512, "width": 512},
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "do_pad": True,
                "data_format": "channels_first",
            }
        ),
        encoding="utf-8",
    )

    processor = adapter_for_family("lfm2").load_processor(str(tmp_path))
    assert isinstance(processor, Lfm2VlProcessor)
    assert processor.image_token == "<image>"
    assert processor.image_token_id == 3


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


def test_nomic_expands_qkv_and_moe_runtime_channels():
    adapter = adapter_for_family("nomic")
    adapter.runtime_config(_cfg())

    qkv = torch.arange(2304 * 2, dtype=torch.float32).reshape(2304, 2)
    match = adapter.name_tensor("encoder.layers.0.attn.Wqkv.weight", qkv, 12)
    emissions = adapter.expand_tensor(match, qkv)
    assert [e.output_name for e in emissions] == [
        "layer_0_attn_q.weights",
        "layer_0_attn_k.weights",
        "layer_0_attn_v.weights",
    ]
    assert [tuple(e.tensor.shape) for e in emissions] == [(768, 2), (768, 2), (768, 2)]
    assert {e.qdq_restore for e in emissions} == {"runtime_key"}

    w1 = torch.empty(24576, 2)
    match = adapter.name_tensor("encoder.layers.1.mlp.experts.mlp.w1", w1, 12)
    emissions = adapter.expand_tensor(match, w1)
    assert len(emissions) == 8
    assert emissions[0].output_name == "layer_1_mlp_expert_0.mlp1.weights"
    assert emissions[-1].output_name == "layer_1_mlp_expert_7.mlp1.weights"
    assert tuple(emissions[0].tensor.shape) == (3072, 2)

    w2 = torch.empty(24576, 2)
    match = adapter.name_tensor("encoder.layers.1.mlp.experts.mlp.w2", w2, 12)
    emissions = adapter.expand_tensor(match, w2)
    assert len(emissions) == 8
    assert emissions[0].output_name == "layer_1_mlp_expert_0.mlp2.weights"
    assert tuple(emissions[0].tensor.shape) == (2, 3072)


def test_nomic_uses_gptq_only_for_hookable_linear_modules():
    adapter = adapter_for_family("nomic")
    qkv_match = adapter.name_tensor("encoder.layers.0.attn.Wqkv.weight", torch.empty(2304, 768), 12)
    qkv_policy = adapter.policy(qkv_match, (768, 768), 4)
    assert qkv_policy.use_gptq
    assert adapter.module_target_name("encoder.layers.0.attn.Wqkv.weight", None) == "encoder.layers.0.attn.Wqkv"

    expert_match = adapter.name_tensor("encoder.layers.1.mlp.experts.mlp.w1", torch.empty(24576, 768), 12)
    expert_policy = adapter.policy(expert_match, (3072, 768), 4)
    assert not expert_policy.use_gptq
    assert adapter.module_target_name("encoder.layers.1.mlp.experts.mlp.w1", None) is None


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
