import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cactus.transpiler.ModelProfiles.profiles import GEMMA4_E2B_PROFILE
from cactus.transpiler.Generator import models as GModels
from cactus.transpiler.RuntimePlan import models as RPModels


class FakeTensor:

    def __init__(self, node_id):
        self.id = node_id


class TestRuntimePlan(unittest.TestCase):

    def test_generator_component_manifest_keeps_logical_names(self):
        component = GModels.ComponentGraph(
            name="decoder_step",
            ir_graph=None,
            output_path=Path("decoder_step.cactus"),
            manifest_path=Path("decoder_step.graph_manifest.json"),
        )

        component.add_runtime_input(FakeTensor(1), "input_ids")
        component.add_runtime_input(FakeTensor(2), "position_ids")
        component.add_output(FakeTensor(3), "logits")

        manifest = GModels.ComponentGraphManifest.from_component(component).to_dict()

        self.assertEqual(manifest["logical_inputs"], ["input_ids", "position_ids"])
        self.assertEqual(manifest["logical_outputs"], ["logits"])
        self.assertEqual(manifest["graph"], "decoder_step.cactus")

    def test_writes_engine_manifest_from_generator_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            generator_manifest = bundle_dir / "decoder_step.graph_manifest.json"
            generator_manifest.write_text(
                json.dumps(
                    {
                        "component": "decoder_step",
                        "graph_path": "decoder_step.cactus",
                        "runtime_input_node_ids": [1, 2],
                        "logical_inputs": ["input_ids", "position_ids"],
                        "output_node_ids": [3],
                        "logical_outputs": ["logits"],
                        "weight_bindings": [
                            {
                                "node_id": 4,
                                "path": "weights/model.layers.0.self_attn.q_proj.weight.bin",
                            }
                        ],
                        "cache_state_node_ids": [
                            {
                                "layer_key": "kv:0",
                                "key": 5,
                                "value": 6,
                                "cache_kind": "kv",
                                "tensor_indices": [0, 1],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            plan = RPModels.runtime_plan_from_generator_manifests(
                {"decoder_step": generator_manifest},
                bundle_dir=bundle_dir,
                model_profile=GEMMA4_E2B_PROFILE,
            )
            engine_manifest_path, runtime_plan_path = plan.write(bundle_dir)

            engine_manifest = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
            runtime_plan = json.loads(runtime_plan_path.read_text(encoding="utf-8"))
            component = engine_manifest["components"][0]

            self.assertEqual(engine_manifest["family"], "gemma4_e2b")
            self.assertEqual(component["component"], "decoder_step")
            self.assertEqual(component["graph"], "decoder_step.cactus")
            self.assertEqual(component["logical_inputs"], ["input_ids", "position_ids"])
            self.assertEqual(component["logical_outputs"], ["logits"])
            self.assertEqual(component["bound_constant_bindings"][0]["node_id"], 4)
            self.assertEqual(component["cache_state_node_ids"][0]["layer_key"], "kv:0")
            self.assertTrue(runtime_plan["routes"])
            self.assertEqual(runtime_plan["routes"][0]["edges"][0]["inputs"], ["text_embed"])


if __name__ == "__main__":
    unittest.main()
