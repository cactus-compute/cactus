import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cactus.transpile.component_plan import infer_component_plan_from_config


def test_nomic_config_routes_to_text_embedding():
    config = {"model_type": "nomic_bert", "architectures": ["NomicBertModel"]}
    plan = infer_component_plan_from_config(config, model_id="nomic-ai/nomic-embed-text-v2-moe")
    assert plan is not None
    assert plan.task == "text_embedding"
    assert plan.components == ("text_embedding",)
    assert plan.force_component_pipeline


def test_nomic_model_id_routes_to_text_embedding():
    plan = infer_component_plan_from_config({}, model_id="nomic-ai/nomic-embed-text-v2-moe")
    assert plan is not None
    assert plan.task == "text_embedding"


def test_family_key_detects_nomic():
    import torch

    from cactus.transpile.model_adapters import _family_key

    class _Cfg:
        model_type = "nomic_bert"

    class _StubNomic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()

    assert _family_key(_StubNomic()) == "nomic"
