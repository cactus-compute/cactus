import unittest

import torch

from cactus.cloud_handoff import (
    EXPECTED_PARAMETER_COUNT,
    export_probe_binary,
    get_default_probe,
    score_hidden_states,
)
from cactus.cloud_handoff.probe import load_release_config, parameter_count


class CloudHandoffProbeTest(unittest.TestCase):
    def test_v10p6_probe_loads_and_scores_hidden_states(self):
        probe = get_default_probe("cpu")

        self.assertEqual(probe.feat_dim, 1536)
        self.assertEqual(probe.layers_to_use, [28])
        self.assertEqual(parameter_count(probe), EXPECTED_PARAMETER_COUNT)

        torch.manual_seed(123)
        hidden = torch.randn(12, probe.feat_dim)
        first = score_hidden_states(hidden, probe=probe)
        second = score_hidden_states(hidden, probe=probe)

        self.assertEqual(first, second)
        self.assertEqual(first.token_count, 12)
        self.assertEqual(first.feature_dim, 1536)
        self.assertEqual(first.layers_to_use, (28,))
        self.assertGreaterEqual(first.probability_wrong, 0.0)
        self.assertLessEqual(first.probability_wrong, 1.0)
        self.assertAlmostEqual(first.confidence, 1.0 - first.probability_wrong)

    def test_v10p6_probe_accepts_single_batch_dimension(self):
        probe = get_default_probe("cpu")

        torch.manual_seed(456)
        hidden = torch.randn(1, 4, probe.feat_dim)
        result = score_hidden_states(hidden, probe=probe)

        self.assertEqual(result.token_count, 4)
        self.assertEqual(result.feature_dim, 1536)

    def test_v10p6_release_config_matches_probe_shape(self):
        config = load_release_config()

        self.assertEqual(config["feat_dim"], 1536)
        self.assertEqual(config["layers_to_use"], [28])
        self.assertEqual(config["max_seq_len"], 1024)

    def test_v10p6_probe_exports_cpp_binary(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = export_probe_binary(Path(tmp) / "probe.bin")
            data = path.read_bytes()
        self.assertEqual(data[:8], b"CCHP10P6")
        self.assertGreater(len(data), EXPECTED_PARAMETER_COUNT * 4)


if __name__ == "__main__":
    unittest.main()
