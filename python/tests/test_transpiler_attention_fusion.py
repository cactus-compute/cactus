import json
import tempfile
import unittest
from pathlib import Path

import torch

from cactus.transpiler.Converter import models as CModels
from cactus.transpiler.IR import simplify_ir

FUSION_FIELDS = ("generic", "attention")


class _Sdpa(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, **self.kwargs)


def _fused_attention_nodes(module):
    args = (torch.randn(1, 8, 32, 64), torch.randn(1, 8, 32, 64), torch.randn(1, 8, 32, 64))
    table = torch.export.default_decompositions()
    table.pop(torch.ops.aten.scaled_dot_product_attention.default, None)
    exported = torch.export.export(module.eval(), args, strict=False).run_decompositions(table)
    records = [CModels.LayerRecord.from_node(i, node) for i, node in enumerate(exported.graph.nodes)]
    layer_map = CModels.LayerMap.from_data(
        x=exported, name="sdpa-fusion-test", model_task="prefill_no_cache", nodes_list=records
    )
    with tempfile.TemporaryDirectory() as tmp:
        simplified = Path(tmp) / "simplified.json"
        simplify_ir.write_simplified_json(
            layer_map, simplified, input_modalities=("text",), fusion_fields=FUSION_FIELDS,
            disabled_fusion_fields=(), disabled_fusions=(),
        )
        nodes = json.loads(simplified.read_text(encoding="utf-8"))["nodes"]
    return [node for node in nodes if node["target"] == "cactus.attention"]


class TestAttentionDirectFusion(unittest.TestCase):
    def test_preserved_attention_fuses_with_the_aten_layout(self):
        fused = _fused_attention_nodes(_Sdpa())
        self.assertEqual(len(fused), 1)
        kwargs = fused[0]["kwargs"]
        self.assertEqual(kwargs["input_layout"], "bhqd_bhsd_bhsd")
        self.assertEqual(kwargs["output_layout"], "bhqd")

    def test_omitted_is_causal_follows_the_aten_default(self):
        fused = _fused_attention_nodes(_Sdpa())
        self.assertIs(fused[0]["kwargs"]["is_causal"], False)

    def test_explicit_is_causal_is_carried_through(self):
        fused = _fused_attention_nodes(_Sdpa(is_causal=True))
        self.assertIs(fused[0]["kwargs"]["is_causal"], True)


if __name__ == "__main__":
    unittest.main()
