"""
Unit tests for metrics.parse_cactus_response.

All tests run without a connected Android device.  The fixtures cover the
happy path, common failure modes (run failure, truncated JSON, extra debug
lines), and edge cases in numeric parsing.
"""

import sys
import os
import unittest

# Allow imports from the parent package without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import (
    CactusResult,
    BatteryState,
    ThermalSnapshot,
    compute_battery_drain,
    detect_thermal_throttle,
    parse_cactus_response,
    peak_thermal,
)


# ---------------------------------------------------------------------------
# parse_cactus_response
# ---------------------------------------------------------------------------


class TestParseCactusResponse(unittest.TestCase):
    """Tests for the JSON parser that handles cactus binary output."""

    def _make_json(self, **overrides) -> str:
        """Return a minimal valid cactus response JSON string."""
        base = {
            "success": True,
            "error": None,
            "cloud_handoff": False,
            "response": "Hello!",
            "confidence": 0.91,
            "time_to_first_token_ms": 45.23,
            "total_time_ms": 163.67,
            "prefill_tps": 1621.89,
            "decode_tps": 168.42,
            "ram_usage_mb": 245.67,
            "prefill_tokens": 28,
            "decode_tokens": 50,
            "total_tokens": 78,
        }
        base.update(overrides)
        import json
        return json.dumps(base)

    def test_happy_path(self):
        """Standard output parses to expected field values."""
        result = parse_cactus_response(self._make_json())
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.prefill_tps, 1621.89)
        self.assertAlmostEqual(result.decode_tps, 168.42)
        self.assertAlmostEqual(result.time_to_first_token_ms, 45.23)
        self.assertAlmostEqual(result.total_time_ms, 163.67)
        self.assertAlmostEqual(result.ram_usage_mb, 245.67)
        self.assertEqual(result.prefill_tokens, 28)
        self.assertEqual(result.decode_tokens, 50)

    def test_debug_lines_before_json(self):
        """Parser tolerates logcat/debug lines printed before the JSON block."""
        prefix = (
            "I/cactus ( 1234): loading model...\n"
            "D/cactus ( 1234): kv-cache allocated\n"
        )
        output = prefix + self._make_json()
        result = parse_cactus_response(output)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.decode_tps, 168.42)

    def test_failure_response_raises(self):
        """A success=false response raises ValueError with the error message."""
        raw = self._make_json(success=False, error="model file not found")
        with self.assertRaises(ValueError) as ctx:
            parse_cactus_response(raw)
        self.assertIn("model file not found", str(ctx.exception))

    def test_no_json_raises(self):
        """Output with no JSON block raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_cactus_response("loading...\ncomplete\n")
        self.assertIn("No JSON response found", str(ctx.exception))

    def test_malformed_json_raises(self):
        """Truncated JSON raises ValueError, not a raw json.JSONDecodeError."""
        with self.assertRaises(ValueError):
            parse_cactus_response('{"success": true, "decode_tps": 100')

    def test_missing_optional_fields_default_to_zero(self):
        """Fields that are absent in older binary versions default to 0.0."""
        import json
        minimal = json.dumps({"success": True})
        result = parse_cactus_response(minimal)
        self.assertTrue(result.success)
        self.assertEqual(result.decode_tps, 0.0)
        self.assertEqual(result.ram_usage_mb, 0.0)
        self.assertEqual(result.prefill_tokens, 0)

    def test_float_coercion(self):
        """Integer JSON values are coerced to float without error."""
        raw = self._make_json(decode_tps=200, prefill_tps=1500, ram_usage_mb=300)
        result = parse_cactus_response(raw)
        self.assertIsInstance(result.decode_tps, float)
        self.assertIsInstance(result.prefill_tps, float)
        self.assertIsInstance(result.ram_usage_mb, float)

    def test_zero_decode_tps_allowed(self):
        """decode_tps=0 is valid (e.g. very short outputs) and should not raise."""
        raw = self._make_json(decode_tps=0, decode_tokens=0)
        result = parse_cactus_response(raw)
        self.assertEqual(result.decode_tps, 0.0)

    def test_high_throughput_values(self):
        """Values at the top end of Apple NPU throughput parse correctly."""
        raw = self._make_json(prefill_tps=900000.0, decode_tps=900.0)
        result = parse_cactus_response(raw)
        self.assertAlmostEqual(result.prefill_tps, 900000.0)

    def test_json_embedded_in_longer_output(self):
        """JSON block appearing mid-stream (e.g. streaming mode) is extracted."""
        output = (
            "chunk: Hello\n"
            "chunk: world\n"
        ) + self._make_json() + "\ndone\n"
        result = parse_cactus_response(output)
        self.assertTrue(result.success)


# ---------------------------------------------------------------------------
# Battery drain helpers
# ---------------------------------------------------------------------------


class TestBatteryDrain(unittest.TestCase):
    def test_positive_drain(self):
        before = BatteryState(level=82, temperature_c=29.5)
        after = BatteryState(level=80, temperature_c=31.2)
        self.assertAlmostEqual(compute_battery_drain(before, after), 2.0)

    def test_charging_clamped_to_zero(self):
        """If the device charged during the run, drain is 0, not negative."""
        before = BatteryState(level=60, temperature_c=28.0)
        after = BatteryState(level=62, temperature_c=27.5)
        self.assertEqual(compute_battery_drain(before, after), 0.0)

    def test_no_change(self):
        state = BatteryState(level=100, temperature_c=25.0)
        self.assertEqual(compute_battery_drain(state, state), 0.0)


# ---------------------------------------------------------------------------
# Thermal throttle detection
# ---------------------------------------------------------------------------


class TestThermalThrottle(unittest.TestCase):
    def _snap(self, cur_khz: int, max_khz: int) -> ThermalSnapshot:
        return ThermalSnapshot(
            temperatures_c=[40.0],
            cpu0_freq_khz=cur_khz,
            cpu0_max_freq_khz=max_khz,
        )

    def test_no_throttle(self):
        """Full-speed operation should not trigger throttle detection."""
        snaps = [self._snap(3000000, 3000000) for _ in range(5)]
        self.assertFalse(detect_thermal_throttle(snaps))

    def test_throttle_detected(self):
        """A >15% frequency drop in any single snapshot triggers detection."""
        snaps = [
            self._snap(3000000, 3000000),
            self._snap(3000000, 3000000),
            self._snap(2400000, 3000000),  # 20% drop — throttled
            self._snap(3000000, 3000000),
        ]
        self.assertTrue(detect_thermal_throttle(snaps))

    def test_borderline_not_throttled(self):
        """Exactly 14.9% drop should not trigger the 15% threshold."""
        cur = int(3000000 * 0.851)
        snaps = [self._snap(cur, 3000000)]
        self.assertFalse(detect_thermal_throttle(snaps))

    def test_empty_snapshots(self):
        """No snapshots → no throttle detected."""
        self.assertFalse(detect_thermal_throttle([]))

    def test_zero_max_freq_ignored(self):
        """Snapshots where max_freq=0 (sysfs unavailable) are skipped safely."""
        snaps = [ThermalSnapshot(temperatures_c=[], cpu0_freq_khz=0, cpu0_max_freq_khz=0)]
        self.assertFalse(detect_thermal_throttle(snaps))

    def test_custom_threshold(self):
        """A 10% threshold catches moderate throttling that 15% misses."""
        snaps = [self._snap(2700000, 3000000)]  # 10% drop exactly
        self.assertFalse(detect_thermal_throttle(snaps, threshold=0.10))
        self.assertFalse(detect_thermal_throttle(snaps, threshold=0.11))
        # 9% drop with 10% threshold → no throttle
        snaps2 = [self._snap(2730000, 3000000)]
        self.assertFalse(detect_thermal_throttle(snaps2, threshold=0.10))


# ---------------------------------------------------------------------------
# Peak thermal
# ---------------------------------------------------------------------------


class TestPeakThermal(unittest.TestCase):
    def _snap(self, temps: list[float]) -> ThermalSnapshot:
        return ThermalSnapshot(
            temperatures_c=temps,
            cpu0_freq_khz=3000000,
            cpu0_max_freq_khz=3000000,
        )

    def test_single_snapshot(self):
        self.assertAlmostEqual(peak_thermal([self._snap([38.0, 41.5, 35.0])]), 41.5)

    def test_across_snapshots(self):
        snaps = [self._snap([38.0]), self._snap([52.3]), self._snap([45.0])]
        self.assertAlmostEqual(peak_thermal(snaps), 52.3)

    def test_empty_snapshots(self):
        self.assertEqual(peak_thermal([]), 0.0)

    def test_empty_zones(self):
        self.assertEqual(peak_thermal([self._snap([])]), 0.0)


if __name__ == "__main__":
    unittest.main()
