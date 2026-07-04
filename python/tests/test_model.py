import unittest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent

from cactus import (
    cactus_init,
    cactus_destroy,
    cactus_complete,
    cactus_embed,
    cactus_image_embed,
    cactus_audio_embed,
    cactus_transcribe,
)
from cactus.cli.model import ensure_bundle
from cactus.cli.download import resolve_weights_variant


_PLATFORM = resolve_weights_variant("general")
_ASSETS_DIR = PROJECT_ROOT / "cactus-engine" / "tests" / "assets"
_TEST_IMAGE = _ASSETS_DIR / "test_monkey.png"
_TEST_AUDIO = _ASSETS_DIR / "test.wav"

class TestVLMModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_dir = ensure_bundle("LiquidAI/LFM2-VL-450M", platform=_PLATFORM)
        cls.model = cactus_init(str(cls.weights_dir), None, False)

    @classmethod
    def tearDownClass(cls):
        cactus_destroy(cls.model)

    def test_text_completion(self):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = cactus_complete(self.model, messages, None, None, None)
        print(f"\n  completion: {json.dumps(result, indent=2)}")
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False))
        self.assertGreater(len(result.get("response", "")), 0)

    def test_image_embedding(self):
        embedding = cactus_image_embed(self.model, str(_TEST_IMAGE))
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    def test_vlm_image_completion(self):
        messages = [{
            "role": "user",
            "content": "Describe this image",
            "images": [str(_TEST_IMAGE)],
        }]
        result = cactus_complete(self.model, messages, None, None, None)
        print(f"\n  vlm completion: {json.dumps(result, indent=2)}")
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False))
        self.assertGreater(len(result.get("response", "")), 0)


class TestWhisperModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_dir = ensure_bundle("openai/whisper-small", platform=_PLATFORM)
        cls.model = cactus_init(str(cls.weights_dir), None, False)

    @classmethod
    def tearDownClass(cls):
        cactus_destroy(cls.model)

    def test_transcription(self):
        prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
        result = cactus_transcribe(
            self.model,
            str(_TEST_AUDIO),
            prompt,
            None,
            None,
            None,
        )
        print(f"\n  transcription: {json.dumps(result, indent=2)}")
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False))
        self.assertGreater(len(result.get("response", "")), 0)

    def test_audio_embedding(self):
        embedding = cactus_audio_embed(self.model, str(_TEST_AUDIO))
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)


def _cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


class TestNomicEmbedding(unittest.TestCase):
    MODEL_ID = "nomic-ai/nomic-embed-text-v2-moe"

    @classmethod
    def setUpClass(cls):
        cls.weights_dir = ensure_bundle(cls.MODEL_ID, platform=_PLATFORM)
        cls.model = cactus_init(str(cls.weights_dir), None, False)

    @classmethod
    def tearDownClass(cls):
        cactus_destroy(cls.model)

    def test_text_embedding_shape_and_determinism(self):
        v1 = cactus_embed(self.model, "Paris is the capital of France.", True)
        v2 = cactus_embed(self.model, "Paris is the capital of France.", True)
        self.assertIsInstance(v1, list)
        self.assertGreater(len(v1), 0)
        self.assertEqual(len(v1), len(v2))
        self.assertLess(max(abs(a - b) for a, b in zip(v1, v2)), 1e-5)

    def test_retrieval_discrimination(self):
        query = cactus_embed(self.model, "search_query: What is the capital of France?", True)
        relevant = cactus_embed(self.model, "search_document: Paris is the capital of France.", True)
        unrelated = cactus_embed(self.model, "search_document: The Great Barrier Reef is off Australia.", True)
        self.assertGreater(_cosine(query, relevant), _cosine(query, unrelated))

    def test_hf_parity(self):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        texts = [
            "search_query: What is the capital of France?",
            "search_document: Paris is the capital and largest city of France.",
            "search_document: The Great Barrier Reef is located off Australia.",
        ]
        cactus_vecs = [cactus_embed(self.model, t, True) for t in texts]

        tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        hf = AutoModel.from_pretrained(self.MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32).eval()

        def hf_embed(text):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                hs = hf(**enc).last_hidden_state
            mask = enc["attention_mask"][..., None].float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return F.normalize(pooled, p=2, dim=1)[0].tolist()

        for text, cactus_vec in zip(texts, cactus_vecs):
            cos = _cosine(cactus_vec, hf_embed(text))
            print(f"\n  HF parity cos ({text[:30]}...): {cos:.4f}")
            # CQ4-quantized weights vs FP reference; FP parity is ~1.0 (see plan notes).
            self.assertGreater(cos, 0.85)


if __name__ == "__main__":
    unittest.main()
