"""
Unit tests for renderer.py.

Validates that the Markdown output matches the exact format used by the
Cactus README table so that copy-pasting results into the README doesn't
require manual reformatting.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from renderer import (
    _format_latency_cell,
    _format_ram,
    _format_tps_cell,
    render_device_header,
    render_extended_table,
    render_readme_table,
)


# ---------------------------------------------------------------------------
# Cell formatting helpers
# ---------------------------------------------------------------------------


class TestFormatTpsCell(unittest.TestCase):
    """Tests for the prefill_tps/decode_tps cell formatter."""

    def test_standard(self):
        data = {"prefill_tps": 255.0, "decode_tps": 37.0}
        self.assertEqual(_format_tps_cell(data), "255/37")

    def test_rounding(self):
        data = {"prefill_tps": 1621.89, "decode_tps": 168.42}
        self.assertEqual(_format_tps_cell(data), "1622/168")

    def test_with_stddev(self):
        data = {"prefill_tps": 255.0, "decode_tps": 37.0, "decode_tps_stddev": 2.1}
        cell = _format_tps_cell(data)
        self.assertIn("255/37", cell)
        self.assertIn("±2.1", cell)

    def test_tiny_stddev_omitted(self):
        """Stddev < 0.05 is considered noise and should not appear in the cell."""
        data = {"prefill_tps": 100.0, "decode_tps": 20.0, "decode_tps_stddev": 0.04}
        self.assertEqual(_format_tps_cell(data), "100/20")

    def test_missing_prefill(self):
        self.assertEqual(_format_tps_cell({"decode_tps": 37.0}), "-")

    def test_missing_decode(self):
        self.assertEqual(_format_tps_cell({"prefill_tps": 255.0}), "-")

    def test_empty_dict(self):
        self.assertEqual(_format_tps_cell({}), "-")

    def test_none(self):
        self.assertEqual(_format_tps_cell(None), "-")

    def test_zero_decode_renders(self):
        """Zero is a valid value (e.g. decode skipped) and should render, not show '-'."""
        data = {"prefill_tps": 500.0, "decode_tps": 0.0}
        self.assertEqual(_format_tps_cell(data), "500/0")


class TestFormatLatencyCell(unittest.TestCase):
    """Tests for the latency_s/decode_tps cell formatter (VLM + STT columns)."""

    def test_with_ttft(self):
        data = {"ttft_ms": 400.0, "decode_tps": 34.0}
        self.assertEqual(_format_latency_cell(data), "0.4s/34")

    def test_no_npu_missing_ttft(self):
        """Android devices without NPU support have no TTFT — renders as -/tps."""
        data = {"ttft_ms": 0.0, "decode_tps": 34.0}
        self.assertEqual(_format_latency_cell(data), "-/34")

    def test_none_ttft(self):
        data = {"ttft_ms": None, "decode_tps": 34.0}
        self.assertEqual(_format_latency_cell(data), "-/34")

    def test_missing_decode(self):
        self.assertEqual(_format_latency_cell({"ttft_ms": 400.0}), "-")

    def test_empty(self):
        self.assertEqual(_format_latency_cell({}), "-")

    def test_none(self):
        self.assertEqual(_format_latency_cell(None), "-")

    def test_latency_precision(self):
        """Latency should show one decimal place (e.g. 0.3s, 13.3s)."""
        data = {"ttft_ms": 13300.0, "decode_tps": 11.0}
        self.assertEqual(_format_latency_cell(data), "13.3s/11")

    def test_sub_second_latency(self):
        data = {"ttft_ms": 200.0, "decode_tps": 98.0}
        self.assertEqual(_format_latency_cell(data), "0.2s/98")


class TestFormatRam(unittest.TestCase):
    def test_mb_under_1024(self):
        self.assertEqual(_format_ram(869.0), "869MB")

    def test_gb_over_1024(self):
        self.assertEqual(_format_ram(1536.0), "1.5GB")

    def test_exactly_1024(self):
        self.assertEqual(_format_ram(1024.0), "1.0GB")

    def test_small_mb(self):
        self.assertEqual(_format_ram(76.0), "76MB")

    def test_none(self):
        self.assertEqual(_format_ram(None), "-")

    def test_string_number(self):
        """Numeric strings should coerce cleanly."""
        self.assertEqual(_format_ram("512"), "512MB")

    def test_non_numeric_string(self):
        self.assertEqual(_format_ram("unknown"), "-")


# ---------------------------------------------------------------------------
# render_readme_table
# ---------------------------------------------------------------------------


class TestRenderReadmeTable(unittest.TestCase):
    def _device(self, **kwargs) -> dict:
        base = {
            "device_name": "Test Device",
            "lfm_1_2b": {"prefill_tps": 255.0, "decode_tps": 37.0},
            "lfmvl_1_6b": {"ttft_ms": 0.0, "decode_tps": 34.0},
            "parakeet_1_1b": {"ttft_ms": 0.0, "decode_tps": 250.0},
            "ram_mb": 1536.0,
            "thermal_throttle_detected": False,
        }
        base.update(kwargs)
        return base

    def test_header_row(self):
        table = render_readme_table([self._device()])
        first_line = table.splitlines()[0]
        self.assertIn("Device", first_line)
        self.assertIn("LFM 1.2B", first_line)
        self.assertIn("LFMVL 1.6B", first_line)
        self.assertIn("Parakeet 1.1B", first_line)
        self.assertIn("RAM", first_line)

    def test_separator_row(self):
        table = render_readme_table([self._device()])
        lines = table.splitlines()
        self.assertTrue(lines[1].startswith("|---"))

    def test_device_row_values(self):
        table = render_readme_table([self._device()])
        data_row = table.splitlines()[2]
        self.assertIn("Test Device", data_row)
        self.assertIn("255/37", data_row)
        self.assertIn("1.5GB", data_row)

    def test_throttle_warning_appended(self):
        device = self._device(thermal_throttle_detected=True)
        table = render_readme_table([device])
        data_row = table.splitlines()[2]
        self.assertIn("⚠️", data_row)

    def test_multiple_devices(self):
        devices = [
            self._device(device_name="Device A"),
            self._device(device_name="Device B"),
        ]
        lines = render_readme_table(devices).splitlines()
        self.assertEqual(len(lines), 4)  # header + sep + 2 devices

    def test_missing_lfmvl_renders_dash(self):
        device = self._device()
        del device["lfmvl_1_6b"]
        table = render_readme_table([device])
        data_row = table.splitlines()[2]
        # The VLM column should show "-"
        cols = [c.strip() for c in data_row.split("|")]
        self.assertEqual(cols[3], "-")  # lfmvl column


# ---------------------------------------------------------------------------
# render_extended_table
# ---------------------------------------------------------------------------


class TestRenderExtendedTable(unittest.TestCase):
    def _device(self) -> dict:
        return {
            "device_name": "Test Device",
            "lfm_1_2b": {"prefill_tps": 118.0, "decode_tps": 23.0},
            "ram_mb": 1228.0,
            "battery_drain_pct": 2.1,
            "thermal_peak_celsius": 41.3,
            "thermal_throttle_detected": False,
            "memory_pressure_events": 0,
        }

    def test_extra_columns_present(self):
        table = render_extended_table([self._device()])
        header = table.splitlines()[0]
        self.assertIn("Battery Drain", header)
        self.assertIn("Peak Temp", header)
        self.assertIn("Throttled", header)
        self.assertIn("OOM Events", header)

    def test_battery_drain_formatted(self):
        table = render_extended_table([self._device()])
        data_row = table.splitlines()[2]
        self.assertIn("2.1%", data_row)

    def test_temperature_formatted(self):
        table = render_extended_table([self._device()])
        data_row = table.splitlines()[2]
        self.assertIn("41°C", data_row)

    def test_throttle_yes_shows_warning(self):
        device = self._device()
        device["thermal_throttle_detected"] = True
        table = render_extended_table([device])
        data_row = table.splitlines()[2]
        self.assertIn("⚠️ Yes", data_row)

    def test_throttle_no_plain_text(self):
        table = render_extended_table([self._device()])
        data_row = table.splitlines()[2]
        self.assertIn("No", data_row)
        self.assertNotIn("⚠️", data_row)


# ---------------------------------------------------------------------------
# render_device_header
# ---------------------------------------------------------------------------


class TestRenderDeviceHeader(unittest.TestCase):
    def test_fields_present(self):
        header = render_device_header(
            {
                "device_name": "Motorola Edge 50 Neo",
                "chipset": "Snapdragon 7s Gen 3",
                "total_ram_mb": 12288,
                "android_version": "14",
            },
            {"precision": "INT4"},
        )
        self.assertIn("Motorola Edge 50 Neo", header)
        self.assertIn("Snapdragon 7s Gen 3", header)
        self.assertIn("INT4", header)
        self.assertIn("Android", header)

    def test_markdown_heading_level(self):
        header = render_device_header(
            {"device_name": "Test", "android_version": "14"},
            {"precision": "INT4"},
        )
        self.assertTrue(header.startswith("### Test"))


if __name__ == "__main__":
    unittest.main()
