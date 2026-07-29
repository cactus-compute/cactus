import unittest

from cactus.transpiler.Fusions import models as FModels
from cactus.transpiler.Generator import component_split
from cactus.transpiler.IR import models as IRModels


class TestGeneratorComponentSplit(unittest.TestCase):
    def test_component_extraction_rewrites_aliases_and_logical_outputs(self) -> None:
        input_ids = IRModels.Node(
            index=0,
            name="input_ids",
            node_type="placeholder",
            target="input_ids",
            args=[],
            kwargs={},
            users=(),
            tensor_output_meta={"shape": [1, 1], "dtype": "torch.int64"},
            module_stack=None,
            value_kind=FModels.ValueKind.USER_INPUT,
        )
        weight = IRModels.Node(
            index=1,
            name="p_embed_weight",
            node_type="placeholder",
            target="p_embed_weight",
            args=[],
            kwargs={},
            users=(),
            tensor_output_meta={"shape": [10, 4], "dtype": "torch.float16"},
            module_stack=None,
            value_kind=FModels.ValueKind.PARAMETER,
        )
        cleanup = IRModels.Node(
            index=2,
            name="index_put",
            node_type="call_function",
            target="aten.index_put.default",
            args=[{"node": "input_ids"}],
            kwargs={},
            users=(),
            tensor_output_meta={"shape": [1, 1], "dtype": "torch.int64"},
            module_stack=None,
            value_kind=FModels.ValueKind.ACTIVATION,
        )
        embedding = IRModels.Node(
            index=3,
            name="embedding",
            node_type="call_function",
            target="aten.embedding.default",
            args=[{"node": "p_embed_weight"}, {"node": "index_put"}],
            kwargs={},
            users=(),
            tensor_output_meta={"shape": [1, 1, 4], "dtype": "torch.float16"},
            module_stack=None,
            value_kind=FModels.ValueKind.ACTIVATION,
        )
        scaled = IRModels.Node(
            index=4,
            name="scaled",
            node_type="call_function",
            target="aten.mul.Tensor",
            args=[{"node": "embedding"}, 2.0],
            kwargs={},
            users=(),
            tensor_output_meta={"shape": [1, 1, 4], "dtype": "torch.float16"},
            module_stack=None,
            value_kind=FModels.ValueKind.ACTIVATION,
            attrs={"other": 2.0},
        )
        output = IRModels.Node(
            index=5,
            name="output",
            node_type="output",
            target="output",
            args=[[{"node": "scaled"}]],
            kwargs={},
            users=(),
            tensor_output_meta=None,
            module_stack=None,
            value_kind=FModels.ValueKind.OUTPUT,
        )
        graph = IRModels.rebuild_graph((input_ids, weight, cleanup, embedding, scaled, output), empty_graph())
        spec = component_split.ComponentSplitSpec(
            name="lm_encoder_step",
            graph=graph,
            outputs=(
                component_split.OutputSpec("scaled", "inputs_embeds"),
                component_split.OutputSpec("position_ids", "position_ids"),
            ),
            placeholders=(component_split.PlaceholderSpec("position_ids", "position_ids", tensor_node="input_ids"),),
            ref_aliases={"index_put": "input_ids"},
        )

        component = component_split.extract_component_graph(spec)

        self.assertNotIn("index_put", component.nodes_map)
        self.assertEqual(component.nodes_map["embedding"].args, [{"node": "p_embed_weight"}, {"node": "input_ids"}])
        self.assertEqual(component.nodes_map["scaled"].ir_metadata["logical_output"], "inputs_embeds")
        self.assertEqual(component.nodes_map["position_ids"].ir_metadata["logical_input"], "position_ids")
        self.assertEqual(component.nodes_map["position_ids"].ir_metadata["logical_output"], "position_ids")


def empty_graph() -> IRModels.Graph:
    return IRModels.Graph(
        source=None,
        sources=(),
        outputs=(),
        nodes=(),
        nodes_map={},
    )


if __name__ == "__main__":
    unittest.main()
