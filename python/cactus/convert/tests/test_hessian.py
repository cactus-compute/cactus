from __future__ import annotations

import numpy as np
import torch

from cactus.convert.calibration.hessian import HessianStats, collect_manifest_hessians
from cactus.convert.cli import _load_hessian_artifacts, _save_hessian_artifacts
from cactus.convert.quantization.cq import quantize_hadamard
from cactus.models.needle.configuration_needle import NeedleConfig
from cactus.models.needle.modeling_needle import NeedleForCausalLM


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, input_ids):
        x = torch.nn.functional.one_hot(input_ids.clamp(0, 1), num_classes=2).float()
        return self.proj(x)


class FailingModel(TinyModel):
    def forward(self, input_ids):
        raise RuntimeError("intentional failure")


class TinyTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.tensor([[0, 1]], dtype=torch.long)}


def test_hessian_records_unresolved_targets():
    stats = collect_manifest_hessians(TinyModel(), TinyTokenizer(), {}, {}, {"missing"}, "cpu")
    assert stats.unresolved_targets == ["missing"]


def test_hessian_records_forward_errors(tmp_path):
    rows = tmp_path / "language.jsonl"
    rows.write_text('{"prompt_text":"hello","completion_text":"world"}\n', encoding="utf-8")
    manifest = {"language": {"path": str(rows)}}
    stats = collect_manifest_hessians(FailingModel(), TinyTokenizer(), manifest, {"language": 1}, {"proj"}, "cpu")
    assert stats.errors == {"RuntimeError": 1}
    assert stats.error_samples[0]["context"] == "language"


def test_hessian_artifact_cache_roundtrip(tmp_path):
    stats = HessianStats()
    stats.hessians["layers.0"] = torch.eye(2)
    stats.diag["layers.0"] = torch.ones(2)
    stats.samples["layers.0"] = 7
    cache = tmp_path / "cache"

    samples = _save_hessian_artifacts(cache, stats)
    loaded = _load_hessian_artifacts(cache)

    assert samples == {"layers.0": 7}
    assert loaded.samples == {"layers.0": 7}
    assert torch.equal(loaded.hessians["layers.0"], torch.eye(2))
    assert torch.equal(loaded.diag["layers.0"], torch.ones(2))


def test_needle_manifest_hessian_is_used_by_gptq(tmp_path):
    rows = tmp_path / "language.jsonl"
    rows.write_text('{"prompt_text":"hello","completion_text":"world"}\n', encoding="utf-8")
    manifest = {"language": {"path": str(rows)}}
    config = NeedleConfig(
        vocab_size=16,
        hidden_size=64,
        d_model=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        torch_dtype="float32",
    )
    model = NeedleForCausalLM(config).float()
    target = "model.encoder.layers.0.self_attn.q_proj"

    stats = collect_manifest_hessians(
        model,
        TinyTokenizer(),
        manifest,
        {"language": 1},
        {target},
        "cpu",
    )

    assert stats.errors == {}
    assert stats.unresolved_targets == []
    assert stats.samples[target] > 0
    assert stats.hessians[target].shape == (64, 64)

    result = quantize_hadamard(
        model.state_dict()[f"{target}.weight"],
        bits=4,
        hessian=stats.hessians[target].numpy(),
        use_gptq=True,
    )

    assert result.gptq_used is True


def test_hadamard_rejects_mismatched_hessian_for_gptq():
    result = quantize_hadamard(
        torch.randn(4, 64),
        bits=4,
        hessian=np.eye(32, dtype=np.float32),
        use_gptq=True,
    )

    assert result.gptq_used is False
