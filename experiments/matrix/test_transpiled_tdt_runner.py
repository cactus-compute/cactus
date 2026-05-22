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


if __name__ == "__main__":
    unittest.main()
