import tempfile
import unittest
from pathlib import Path

import numpy as np

from cactus.transpiler.Converter import models as CModels
from cactus.transpiler.Fusions import models as FModels
from cactus.transpiler.Generator import models as GModels
from cactus.transpiler.Generator.lowerings import lowering_core
from cactus.transpiler.Generator.lowerings.lowering_nn_ops import upsample_scale
from cactus.transpiler.IR import models as IRModels


def upsample_graph(target: str, args: list, attrs: dict | None = None) -> IRModels.Graph:
    x = IRModels.Node(
        index=0,
        name="x",
        node_type="placeholder",
        target="x",
        args=[],
        kwargs={},
        users=(),
        tensor_output_meta={"shape": [1, 2, 2, 3], "dtype": "torch.float16"},
        module_stack=None,
        value_kind=FModels.ValueKind.USER_INPUT,
        ir_metadata={"logical_input": "x"},
    )
    up = IRModels.Node(
        index=1,
        name="up",
        node_type="call_function",
        target=target,
        args=args,
        kwargs={},
        users=(),
        tensor_output_meta={"shape": [1, 2, 4, 6], "dtype": "torch.float16"},
        module_stack=None,
        value_kind=FModels.ValueKind.ACTIVATION,
        attrs=attrs or {},
    )
    output = IRModels.Node(
        index=2,
        name="output",
        node_type="output",
        target="output",
        args=[[{"node": "up"}]],
        kwargs={},
        users=(),
        tensor_output_meta=None,
        module_stack=None,
        value_kind=FModels.ValueKind.OUTPUT,
    )
    return IRModels.rebuild_graph(
        (x, up, output),
        IRModels.Graph(source=None, sources=(), outputs=(), nodes=(), nodes_map={}),
    )


class TestTranspilerUpsample(unittest.TestCase):
    def test_ir_extracts_attrs_for_both_aten_overloads(self):
        records = [
            CModels.LayerRecord(
                index=0, name="x", node_type="placeholder", target="x", args=[], kwargs={},
                users=[], tensor_output_meta={"shape": [1, 2, 2, 3], "dtype": "torch.float16"},
                module_stack=None,
            ),
            CModels.LayerRecord(
                index=1, name="up_vec", node_type="call_function",
                target="aten.upsample_nearest2d.vec",
                args=[{"node": "x"}, None, [2.0, 2.0]], kwargs={}, users=[],
                tensor_output_meta={"shape": [1, 2, 4, 6], "dtype": "torch.float16"},
                module_stack=None,
            ),
            CModels.LayerRecord(
                index=2, name="up_size", node_type="call_function",
                target="aten.upsample_nearest2d.default",
                args=[{"node": "up_vec"}, [8, 12]], kwargs={}, users=[],
                tensor_output_meta={"shape": [1, 2, 8, 12], "dtype": "torch.float16"},
                module_stack=None,
            ),
        ]
        layer_map = CModels.LayerMap(
            model_name="upsample-attrs-test", task="prefill_no_cache",
            graph_signature="", range_constants="", nodes=records,
        )
        graph = IRModels.Graph.from_map(layer_map)

        self.assertEqual(graph.nodes_map["up_vec"].attrs.get("scale_factors"), [2.0, 2.0])
        self.assertEqual(graph.nodes_map["up_size"].attrs.get("output_size"), [8, 12])

    def test_lowering_rules_cover_every_upsample_target(self):
        rules = lowering_core.build_lowering_rules()
        for target in (
            "cactus.upsample_nearest2d",
            "aten.upsample_nearest2d.default",
            "aten.upsample_nearest2d.vec",
        ):
            self.assertIn(target, rules)

    def test_scale_derives_from_output_size_and_rejects_anisotropic(self):
        graph = upsample_graph("aten.upsample_nearest2d.vec", [{"node": "x"}], {"output_size": [4, 6]})
        self.assertEqual(upsample_scale(graph.nodes_map["up"]), 2)

        from cactus.transpiler.Generator.errors import UnsupportedLoweringError

        bad = upsample_graph("aten.upsample_nearest2d.vec", [{"node": "x"}], {"scale_factors": [2.0, 3.0]})
        with self.assertRaises(UnsupportedLoweringError):
            upsample_scale(bad.nodes_map["up"])

    def test_lowered_graph_executes_exact_nearest_neighbour(self):
        try:
            from cactus.bindings.cactus import Tensor
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"cactus runtime library unavailable: {exc}")

        graph = upsample_graph(
            "aten.upsample_nearest2d.vec",
            [{"node": "x"}, None, [2.0, 2.0]],
            {"output_size": None, "scale_factors": [2.0, 2.0]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = GModels.GeneratorConfig(output_dir=Path(tmp))
            component = GModels.ComponentGraph.from_ir("upsample", graph, config)
            lowering_core.lower_component(component, config, lowering_core.build_lowering_rules())

        self.assertEqual(component.unsupported_nodes, [])
        self.assertEqual(len(component.runtime_input_ids), 1)
        self.assertEqual(len(component.output_node_ids), 1)

        g = component.graph
        g.retain_outputs(component.output_node_ids)
        data = np.arange(12, dtype=np.float16).reshape(1, 2, 2, 3)
        g.set_input(Tensor(g, component.runtime_input_ids[0], (1, 2, 2, 3), g.FP16), data)
        g.execute()
        result = Tensor(g, component.output_node_ids[0], (1, 2, 4, 6), g.FP16).numpy()

        expected = data.repeat(2, axis=2).repeat(2, axis=3)
        np.testing.assert_array_equal(np.asarray(result, dtype=np.float16), expected)


if __name__ == "__main__":
    unittest.main()
