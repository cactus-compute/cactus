from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tests" / "android" / "transpiled_tdt.cpp"


class TranspiledTDTRunnerTests(unittest.TestCase):
    def test_initial_decoder_token_uses_configured_blank(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("int last_token = cfg.blank_id;", source)
        self.assertNotIn("last_token = cfg.blank_id == static_cast<int>(vocab.size() - 1)", source)
        self.assertNotIn("token_class_count == static_cast<int>(vocab_size + 1)", source)

    def test_mapped_features_are_converted_to_encoder_input_precision(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("const auto& input_features_buf = encoder.get_output_buffer(1);", source)
        self.assertIn("copy_tensor_converting(", source)
        self.assertIn("input_features_buf.precision", source)


if __name__ == "__main__":
    unittest.main()
