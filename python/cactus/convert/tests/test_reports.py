from __future__ import annotations

import json
from pathlib import Path

from cactus.convert.export.reports import write_reports


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
