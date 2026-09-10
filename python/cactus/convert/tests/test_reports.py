from __future__ import annotations

import json
from pathlib import Path

from cactus.convert.export.reports import print_summary, write_reports


def test_weights_manifest_includes_bias_tensors(tmp_path: Path) -> None:
    write_reports(
        tmp_path,
        [
            {
                "source_name": "decoder.layers.0.fc1.bias",
                "output_file": "decoder.layer_0_mlp_fc1.bias",
                "shape": [4],
                "precision": "FP16",
                "status": "fallback",
                "component": "transcription",
            },
            {
                "source_name": "decoder.layers.0.fc1.weight",
                "output_file": "decoder.layer_0_mlp_fc1.weights",
                "shape": [4, 4],
                "precision": "CQ4",
                "status": "converted",
                "component": "transcription",
            },
        ],
    )

    manifest = json.loads((tmp_path / "weights_manifest.json").read_text())
    output_names = {row["output_name"] for row in manifest["weights"]}

    assert "decoder.layer_0_mlp_fc1.bias" in output_names
    assert "decoder.layer_0_mlp_fc1.weights" in output_names


def test_reports_make_all_gptq_fallbacks_visible(tmp_path: Path, capsys) -> None:
    summary = write_reports(
        tmp_path,
        [
            {
                "source_name": "layers.0.q_proj.weight",
                "output_file": "model.layer_0_attention_q_proj.weights",
                "shape": [4, 4],
                "precision": "CQ4",
                "status": "converted",
                "component": "language",
                "gptq_expected": True,
                "gptq_used": False,
                "hessian_samples": 0,
            },
            {
                "source_name": "layers.0.k_proj.weight",
                "output_file": "model.layer_0_attention_k_proj.weights",
                "shape": [4, 4],
                "precision": "CQ4",
                "status": "converted",
                "component": "language",
                "gptq_expected": True,
                "gptq_used": True,
                "hessian_samples": 8,
            },
            {
                "source_name": "layers.0.v_proj.weight",
                "output_file": "model.layer_0_attention_v_proj.weights",
                "shape": [4, 4],
                "precision": "CQ4",
                "status": "converted",
                "component": "language",
                "gptq_expected": True,
                "gptq_used": False,
                "hessian_samples": 8,
            },
        ],
    )

    assert summary["gptq"] == {
        "expected_tensors": 3,
        "calibrated_tensors": 1,
        "uncalibrated_tensors": 2,
        "zero_sample_tensors": 1,
        "unusable_hessian_tensors": 1,
    }
    saved = json.loads((tmp_path / "conversion_summary.json").read_text())
    assert saved["gptq"] == summary["gptq"]

    print_summary(summary)
    output = capsys.readouterr().out
    assert "GPTQ calibrated 1/3" in output
    assert "WARNING: 2 GPTQ-eligible CQ tensors used RTN fallback" in output
    assert "1 zero-sample, 1 unusable Hessian" in output
