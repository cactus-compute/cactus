"""
Metric collection and result parsing for the Cactus Android benchmark harness.

Two categories of metrics are collected:

Standard (matching the existing Cactus README table):
    - decode_tps / prefill_tps
    - time_to_first_token_ms
    - ram_mb at 4k context

New (not tracked anywhere in mobile inference today):
    - battery_drain_pct   — battery % consumed during the benchmark run
    - thermal_throttle_detected — whether the SoC clock dropped >15% below max
    - thermal_peak_celsius — highest thermal zone reading during the run
    - memory_pressure_events — OOM killer / low-memory events in dmesg
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from adb_client import ADBClient, ADBError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CactusResult:
    """Parsed output from a single cactus inference run."""

    success: bool
    prefill_tps: float
    decode_tps: float
    time_to_first_token_ms: float
    total_time_ms: float
    ram_usage_mb: float
    prefill_tokens: int
    decode_tokens: int


@dataclass
class BatteryState:
    """Battery level and temperature at a single point in time."""

    level: int           # 0-100 %
    temperature_c: float # degrees Celsius


@dataclass
class ThermalSnapshot:
    """All thermal zone temperatures plus CPU0 frequency at one instant."""

    temperatures_c: list[float]
    cpu0_freq_khz: int
    cpu0_max_freq_khz: int
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class MetricsBundle:
    """
    Aggregated metrics for one benchmark scenario after all iterations complete.

    The stddev fields are populated from the per-iteration samples so the
    rendered table can show mean ± stddev instead of a bare mean.
    """

    # Standard inference metrics (matching README table columns)
    decode_tps: float
    prefill_tps: float
    ttft_ms: float
    ram_mb: float

    # New metrics not previously tracked for Android
    battery_drain_pct: float
    thermal_throttle_detected: bool
    thermal_peak_celsius: float
    memory_pressure_events: int

    # Variability across measurement iterations
    decode_tps_stddev: float = 0.0
    prefill_tps_stddev: float = 0.0


# ---------------------------------------------------------------------------
# Cactus output parser
# ---------------------------------------------------------------------------


def parse_cactus_response(output: str) -> CactusResult:
    """
    Parse the JSON object that cactus writes to stdout after an inference run.

    The engine may emit debug lines before the JSON block. We search for the
    first ``{...}`` containing the ``"success"`` key rather than assuming the
    entire stdout is JSON, so the parser is robust to verbose build outputs
    and logcat noise that ADB sometimes injects.

    Expected fields (from the Engine API docs in README.md):
        success, time_to_first_token_ms, total_time_ms,
        prefill_tps, decode_tps, ram_usage_mb,
        prefill_tokens, decode_tokens

    Args:
        output: Raw stdout captured from `adb shell ./test_performance ...`.

    Returns:
        Populated CactusResult.

    Raises:
        ValueError: If no parseable JSON is found or ``success`` is false.
    """
    # Extract the outermost JSON object that contains "success"
    match = re.search(r"\{[^{}]*\"success\"[^{}]*\}", output, re.DOTALL)
    if not match:
        preview = output[:300].replace("\n", "\\n")
        raise ValueError(f"No JSON response found in cactus output: {preview!r}")

    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in cactus output: {exc}") from exc

    if not raw.get("success"):
        error_detail = raw.get("error") or "unknown error"
        raise ValueError(f"Cactus run reported failure: {error_detail}")

    return CactusResult(
        success=True,
        prefill_tps=float(raw.get("prefill_tps", 0.0)),
        decode_tps=float(raw.get("decode_tps", 0.0)),
        time_to_first_token_ms=float(raw.get("time_to_first_token_ms", 0.0)),
        total_time_ms=float(raw.get("total_time_ms", 0.0)),
        ram_usage_mb=float(raw.get("ram_usage_mb", 0.0)),
        prefill_tokens=int(raw.get("prefill_tokens", 0)),
        decode_tokens=int(raw.get("decode_tokens", 0)),
    )


# ---------------------------------------------------------------------------
# Battery metrics
# ---------------------------------------------------------------------------


def read_battery(adb: ADBClient) -> BatteryState:
    """
    Read battery level and temperature from ``dumpsys battery``.

    Android reports battery temperature in tenths of a degree Celsius, so
    we divide by 10. Level is already a 0-100 integer.

    Args:
        adb: Connected ADBClient.

    Returns:
        Current BatteryState.
    """
    output = adb.shell("dumpsys battery")
    level = 0
    temp_c = 0.0

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("level:"):
            try:
                level = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("temperature:"):
            try:
                raw_temp = int(stripped.split(":", 1)[1].strip())
                temp_c = raw_temp / 10.0
            except ValueError:
                pass

    return BatteryState(level=level, temperature_c=temp_c)


def compute_battery_drain(before: BatteryState, after: BatteryState) -> float:
    """
    Return battery percentage consumed between two BatteryState readings.

    A positive result means the battery discharged. Negative values (charging
    during the benchmark) are clamped to 0.0 so they don't mislead callers.

    Args:
        before: Reading taken before the benchmark run.
        after:  Reading taken after the benchmark run.

    Returns:
        Drain in percentage points (0.0 if charging or unchanged).
    """
    return max(0.0, float(before.level - after.level))


# ---------------------------------------------------------------------------
# Thermal metrics
# ---------------------------------------------------------------------------


def read_thermal_snapshot(adb: ADBClient) -> ThermalSnapshot:
    """
    Read all thermal zone temperatures and CPU0 current/max clock frequencies.

    Thermal zone ``temp`` files report values in millidegrees Celsius on most
    Android devices (e.g. ``42000`` = 42°C). The heuristic below treats any
    value >1000 as millidegrees and divides by 1000; values ≤1000 are assumed
    to already be in Celsius (seen on some older Exynos devices).

    CPU frequency values from ``cpufreq`` are in kHz.

    Args:
        adb: Connected ADBClient.

    Returns:
        Snapshot with all zone temperatures and CPU0 frequency pair.
    """
    raw_temps = adb.shell("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null")
    temperatures_c: list[float] = []
    for line in raw_temps.splitlines():
        line = line.strip()
        if line.isdigit():
            val = int(line)
            temperatures_c.append(val / 1000.0 if val > 1000 else float(val))

    cur_raw = adb.shell(
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null"
    )
    max_raw = adb.shell(
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null"
    )

    cur_freq = int(cur_raw) if cur_raw.isdigit() else 0
    max_freq = int(max_raw) if max_raw.isdigit() else 1  # avoid div-by-zero

    return ThermalSnapshot(
        temperatures_c=temperatures_c,
        cpu0_freq_khz=cur_freq,
        cpu0_max_freq_khz=max_freq,
        timestamp=time.monotonic(),
    )


def detect_thermal_throttle(
    snapshots: list[ThermalSnapshot], threshold: float = 0.15
) -> bool:
    """
    Return True if CPU0 dropped more than ``threshold`` below its rated max.

    A >15% sustained frequency reduction is the standard indicator of thermal
    throttling on Qualcomm Snapdragon SoCs. Mediatek and Exynos use the same
    cpufreq sysfs interface, so this heuristic is chip-family agnostic.

    Args:
        snapshots: List of ThermalSnapshot collected during the run.
        threshold: Fractional drop that counts as throttling (default 0.15 = 15%).

    Returns:
        True if throttling was detected in any snapshot.
    """
    for snap in snapshots:
        if snap.cpu0_max_freq_khz > 0:
            ratio = snap.cpu0_freq_khz / snap.cpu0_max_freq_khz
            if ratio < (1.0 - threshold):
                return True
    return False


def peak_thermal(snapshots: list[ThermalSnapshot]) -> float:
    """
    Return the highest temperature (°C) seen across all zones and snapshots.

    Args:
        snapshots: List of ThermalSnapshot collected during the run.

    Returns:
        Peak temperature in degrees Celsius, or 0.0 if no data.
    """
    all_temps = [t for snap in snapshots for t in snap.temperatures_c]
    return max(all_temps) if all_temps else 0.0


# ---------------------------------------------------------------------------
# Memory pressure
# ---------------------------------------------------------------------------


def read_memory_pressure_events(adb: ADBClient) -> int:
    """
    Count low-memory and OOM killer events in the kernel ring buffer.

    We grep dmesg for three patterns that indicate memory pressure on Android:
      - "low memory" — generic low-memory warnings from the kernel
      - "oom" — Out-of-Memory killer activations
      - "killed process" — process killed by lmkd (Android low-memory killer)

    The count is used as a delta (after - before) to isolate events that
    occurred specifically during the benchmark run.

    Args:
        adb: Connected ADBClient.

    Returns:
        Number of matching lines in dmesg, or 0 on failure.
    """
    output = adb.shell(
        "dmesg 2>/dev/null | grep -icE 'low memory|\\boom\\b|killed process'"
    )
    try:
        return int(output.strip())
    except ValueError:
        return 0
