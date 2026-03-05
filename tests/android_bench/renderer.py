"""
Output rendering for the Cactus Android benchmark harness.

Three output formats are supported:

1. README table  — GitHub Markdown matching the exact column layout in
                   the Cactus README Benchmarks section.
2. Extended table — same devices but with extra battery / thermal columns.
3. JSON artifact  — machine-readable output for CI post-processing and
                    programmatic comparison across releases.
"""

import datetime
import json
import math
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Public rendering functions
# ---------------------------------------------------------------------------


def render_readme_table(results: list[dict]) -> str:
    """
    Render results as a GitHub Markdown table matching the Cactus README format.

    Column layout (mirroring README.md exactly):
        | Device | LFM 1.2B | LFMVL 1.6B | Parakeet 1.1B | RAM |

    Cell formats:
        LFM 1.2B      → ``prefill_tps/decode_tps``  (e.g. ``255/37``)
        LFMVL 1.6B    → ``latency_s/decode_tps``   (e.g. ``0.4s/34`` or ``-/34``)
        Parakeet 1.1B → ``latency_s/decode_tps``   (e.g. ``-/250k+``)
        RAM           → MB or GB depending on magnitude

    Devices with detected thermal throttling have ``⚠️`` appended to the
    device name so reviewers notice the caveat without reading the extended table.

    Args:
        results: List of per-device result dicts as produced by
                 ``bench.results_to_render_payload()``.

    Returns:
        Multi-line GitHub Markdown string.
    """
    header = "| Device | LFM 1.2B | LFMVL 1.6B | Parakeet 1.1B | RAM |"
    separator = "|--------|----------|------------|---------------|-----|"
    rows = [header, separator]

    for r in results:
        device = r.get("device_name", "Unknown")
        if r.get("thermal_throttle_detected"):
            device = f"{device} ⚠️"

        lfm_cell = _format_tps_cell(r.get("lfm_1_2b", {}))
        lfmvl_cell = _format_latency_cell(r.get("lfmvl_1_6b", {}))
        parakeet_cell = _format_latency_cell(r.get("parakeet_1_1b", {}))
        ram_cell = _format_ram(r.get("ram_mb"))

        rows.append(
            f"| {device} | {lfm_cell} | {lfmvl_cell} | {parakeet_cell} | {ram_cell} |"
        )

    return "\n".join(rows)


def render_extended_table(results: list[dict]) -> str:
    """
    Render an extended Markdown table that includes battery drain and thermal columns.

    This table is not intended to replace the README table — it sits below it
    in the generated markdown document and provides the extra signal that
    matters for sustained-load production deployments.

    Columns:
        Device | LFM 1.2B | RAM | Battery Drain | Peak Temp | Throttled | OOM Events

    Args:
        results: List of per-device result dicts.

    Returns:
        Multi-line GitHub Markdown string.
    """
    header = (
        "| Device | LFM 1.2B | RAM | Battery Drain | Peak Temp | Throttled | OOM Events |"
    )
    separator = (
        "|--------|----------|-----|---------------|-----------|-----------|------------|"
    )
    rows = [header, separator]

    for r in results:
        device = r.get("device_name", "Unknown")
        lfm_cell = _format_tps_cell(r.get("lfm_1_2b", {}))
        ram_cell = _format_ram(r.get("ram_mb"))

        battery = r.get("battery_drain_pct")
        battery_cell = f"{battery:.1f}%" if battery is not None else "-"

        peak_temp = r.get("thermal_peak_celsius")
        temp_cell = f"{peak_temp:.0f}°C" if peak_temp is not None else "-"

        throttled = r.get("thermal_throttle_detected", False)
        throttle_cell = "⚠️ Yes" if throttled else "No"

        oom = r.get("memory_pressure_events", 0)
        oom_cell = str(oom) if oom is not None else "-"

        rows.append(
            f"| {device} | {lfm_cell} | {ram_cell} | {battery_cell} "
            f"| {temp_cell} | {throttle_cell} | {oom_cell} |"
        )

    return "\n".join(rows)


def render_device_header(device_info: dict, scenario_info: dict) -> str:
    """
    Render a Markdown section header with device metadata.

    Placed above both tables so readers immediately know which device the
    numbers belong to, without needing to cross-reference an external registry.

    Args:
        device_info: Dict returned by ``ADBClient.device_info()``, optionally
                     extended with ``device_name``, ``chipset``, ``ram_gb``.
        scenario_info: Dict with at least a ``precision`` key.

    Returns:
        Multi-line Markdown string.
    """
    name = device_info.get("device_name") or device_info.get("model", "Unknown Device")
    chipset = device_info.get("chipset", "unknown")
    ram_gb = device_info.get("ram_gb") or _mb_to_gb_str(device_info.get("total_ram_mb"))
    android = device_info.get("android_version", "unknown")
    precision = scenario_info.get("precision", "INT4")
    run_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    lines = [
        f"### {name}",
        "",
        f"- **Chipset**: {chipset}",
        f"- **RAM**: {ram_gb}",
        f"- **Android**: {android}",
        f"- **Precision**: {precision}",
        f"- **Run date**: {run_date}",
        "",
    ]
    return "\n".join(lines)


def render_json_artifact(results: list[dict], metadata: dict) -> str:
    """
    Serialize benchmark results to a structured JSON string.

    The schema version field lets downstream consumers detect breaking changes
    to the result format without parsing the full structure.

    Args:
        results: List of scenario result dicts (from ``_scenario_result_to_dict``).
        metadata: Arbitrary key/value pairs to include in the artifact header
                  (device info, run timestamp, scenarios file path, etc.).

    Returns:
        Pretty-printed JSON string (2-space indent).
    """
    artifact = {
        "schema_version": "1.0",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "metadata": metadata,
        "results": results,
    }
    return json.dumps(artifact, indent=2)


# ---------------------------------------------------------------------------
# Cell formatting helpers
# ---------------------------------------------------------------------------


def _format_tps_cell(data: Optional[dict]) -> str:
    """
    Format a ``prefill_tps/decode_tps`` cell for LFM completion scenarios.

    When a stddev is available the cell shows ``P/D±σ`` (e.g. ``255/37±2.1``).
    Returns ``-`` for missing or empty data.
    """
    if not data:
        return "-"

    prefill = data.get("prefill_tps")
    decode = data.get("decode_tps")
    if prefill is None or decode is None:
        return "-"

    stddev = data.get("decode_tps_stddev")
    if stddev and stddev > 0.05:
        return f"{prefill:.0f}/{decode:.0f}±{stddev:.1f}"
    return f"{prefill:.0f}/{decode:.0f}"


def _format_latency_cell(data: Optional[dict]) -> str:
    """
    Format a ``latency_s/decode_tps`` cell for VLM and STT scenarios.

    A missing TTFT (no NPU acceleration) is rendered as ``-`` in the latency
    position, matching the existing README convention for Android devices.
    """
    if not data:
        return "-"

    ttft_ms = data.get("ttft_ms")
    decode = data.get("decode_tps")
    if decode is None:
        return "-"

    if ttft_ms and ttft_ms > 0:
        return f"{ttft_ms / 1000:.1f}s/{decode:.0f}"
    return f"-/{decode:.0f}"


def _format_ram(ram_mb: Any) -> str:
    """
    Format a RAM value, converting to GB when >= 1024 MB.

    Mirrors the README convention: ``76MB``, ``1.5GB``, etc.
    """
    if ram_mb is None:
        return "-"
    try:
        mb = float(ram_mb)
    except (TypeError, ValueError):
        return "-"

    if mb >= 1024:
        return f"{mb / 1024:.1f}GB"
    return f"{mb:.0f}MB"


def _mb_to_gb_str(total_ram_mb: Any) -> str:
    """Convert a total_ram_mb value to a human-readable GB string."""
    if not total_ram_mb:
        return "?"
    try:
        return f"{int(total_ram_mb) / 1024:.0f} GB"
    except (TypeError, ValueError):
        return "?"
