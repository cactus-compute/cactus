from __future__ import annotations

import torch

from cactus.convert.calibration.hessian import HessianStats, collect_manifest_hessians
from cactus.convert.cli import _load_hessian_artifacts, _save_hessian_artifacts


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
