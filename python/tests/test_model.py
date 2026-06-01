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


def _find_asset(name):
    for candidate in (
        PROJECT_ROOT / "python" / "cactus" / "assets" / name,
        PROJECT_ROOT / "cactus-engine" / "tests" / "assets" / name,
        PROJECT_ROOT / "tests" / "assets" / name,
    ):
        if candidate.exists():
            return candidate
    return None


_TEST_IMAGE = _find_asset("test_monkey.png")
_TEST_AUDIO = _find_asset("test.wav")

_EMBED_NOT_IMPLEMENTED = (
    "Embeddings are not wired up for transpiled bundles in the v2 engine "
    "(Model::get_embeddings stub + image/audio embed paths gated)."
)


class TestVLMModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_dir = ensure_bundle("LiquidAI/LFM2-VL-450M")
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

    @unittest.skip(_EMBED_NOT_IMPLEMENTED)
    def test_text_embedding(self):
        embedding = cactus_embed(self.model, "Hello world", True)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    @unittest.skip(_EMBED_NOT_IMPLEMENTED)
    def test_image_embedding(self):
        embedding = cactus_image_embed(self.model, str(_TEST_IMAGE))
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    @unittest.skipUnless(_TEST_IMAGE is not None, "test_monkey.png not found")
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
        cls.weights_dir = ensure_bundle("openai/whisper-small")
        cls.model = cactus_init(str(cls.weights_dir), None, False)

    @classmethod
    def tearDownClass(cls):
        cactus_destroy(cls.model)

    @unittest.skipUnless(_TEST_AUDIO is not None, "test.wav not found")
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

    @unittest.skip(_EMBED_NOT_IMPLEMENTED)
    def test_audio_embedding(self):
        embedding = cactus_audio_embed(self.model, str(_TEST_AUDIO))
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)


if __name__ == "__main__":
    unittest.main()
