"""
Android benchmarking harness for the Cactus inference engine.

Pushes the compiled test_performance binary and model weights to a connected
Android device via ADB, runs each scenario with warmup + measurement iterations,
collects battery drain and thermal data throughout, and writes the results as
both a GitHub Markdown table (matching the README format) and a JSON artifact.

Usage:
    cd tests/android_bench
    python bench.py --serial <adb_serial>

    # With explicit paths:
    python bench.py \\
        --serial R5CN70KLXGE \\
        --binary ../../build/android/test_performance \\
        --weights ../../weights \\
        --precision INT4 \\
        --output results/

Requirements:
    pip install -r requirements.txt
    Android NDK + platform tools installed, `adb` on PATH.
    Device connected with USB debugging enabled.
"""

import argparse
import json
import logging
import math
import os
import pathlib
import signal
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml

from adb_client import ADBClient, ADBError
from metrics import (
    CactusResult,
    MetricsBundle,
    ThermalSnapshot,
    compute_battery_drain,
    detect_thermal_throttle,
    parse_cactus_response,
    peak_thermal,
    read_battery,
    read_memory_pressure_events,
    read_thermal_snapshot,
)
from renderer import (
    render_device_header,
    render_extended_table,
    render_json_artifact,
    render_readme_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE_BENCH_DIR = "/data/local/tmp/cactus_bench"
WARMUP_ITERS = 3
MEASURE_ITERS = 5
RETRY_LIMIT = 3
THERMAL_POLL_INTERVAL_S = 2.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """One row in scenarios.yaml — a fully-specified inference workload."""

    name: str
    model: str
    precision: str          # INT4 | INT8 | FP16
    task: str               # completion | vlm | stt
    context_length: int
    prompt: str
    max_tokens: int
    audio_file: Optional[str] = None
    image_file: Optional[str] = None


@dataclass
class IterationResult:
    """Raw output from a single inference iteration."""

    prefill_tps: float
    decode_tps: float
    ttft_ms: float
    ram_mb: float
    success: bool
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Aggregated outcome for one scenario after all warmup + measurement iters."""

    scenario_name: str
    model: str
    precision: str
    iterations: list[IterationResult] = field(default_factory=list)
    metrics: Optional[MetricsBundle] = None
    failed: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Return arithmetic mean or 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    """Return sample standard deviation or 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# Background thermal poller
# ---------------------------------------------------------------------------


class ThermalPoller(threading.Thread):
    """
    Daemon thread that samples thermal zones and CPU frequency every few seconds.

    Runs in the background during warmup and measurement iterations so we get
    a continuous temperature trace without blocking the benchmark loop.
    """

    def __init__(self, adb: ADBClient, interval_s: float = THERMAL_POLL_INTERVAL_S) -> None:
        super().__init__(daemon=True, name=f"thermal-{adb.serial}")
        self.adb = adb
        self.interval_s = interval_s
        self.snapshots: list[ThermalSnapshot] = []
        self._stop = threading.Event()

    def run(self) -> None:
        """Poll until stop() is called."""
        while not self._stop.wait(self.interval_s):
            try:
                self.snapshots.append(read_thermal_snapshot(self.adb))
            except ADBError as exc:
                logger.debug("Thermal poll skipped: %s", exc)

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to exit."""
        self._stop.set()
        self.join(timeout=self.interval_s + 2)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """
    Orchestrates the full benchmark lifecycle on a connected Android device.

    Responsibilities:
      - Push the binary and weights once at setup time.
      - For each scenario: run warmup iters (discarded), then MEASURE_ITERS
        measurement iters with retry logic and partial-result saves on ADB drop.
      - Collect battery state before/after each scenario and continuously poll
        thermal sensors in the background.
      - Aggregate iteration outputs into MetricsBundle (mean ± stddev).
    """

    def __init__(
        self,
        adb: ADBClient,
        scenarios: list[ScenarioConfig],
        binary_path: str,
        weights_dir: str,
        output_dir: str,
        assets_dir: Optional[str] = None,
    ) -> None:
        self.adb = adb
        self.scenarios = scenarios
        self.binary_path = binary_path
        self.weights_dir = weights_dir
        self.output_dir = pathlib.Path(output_dir)
        self.assets_dir = assets_dir
        self.results: list[ScenarioResult] = []
        self._binary_name = os.path.basename(binary_path)

    def setup(self) -> None:
        """
        Push the benchmark binary and model weights to the device.

        Creates ``DEVICE_BENCH_DIR/{weights,assets}`` on the device if they
        don't already exist, then pushes the binary (with +x) and the local
        weights directory.

        Raises:
            ADBError: If any push fails.
        """
        logger.info("[%s] Setting up benchmark environment", self.adb.serial)
        self.adb.shell(f"mkdir -p {DEVICE_BENCH_DIR}/weights {DEVICE_BENCH_DIR}/assets")

        remote_bin = f"{DEVICE_BENCH_DIR}/{self._binary_name}"
        logger.info("Pushing benchmark binary → %s", remote_bin)
        self.adb.push(self.binary_path, remote_bin)
        self.adb.shell(f"chmod +x {remote_bin}")

        logger.info("Pushing model weights → %s/weights/", DEVICE_BENCH_DIR)
        self.adb.push(self.weights_dir, f"{DEVICE_BENCH_DIR}/weights/")

        if self.assets_dir and os.path.exists(self.assets_dir):
            logger.info("Pushing test assets → %s/assets/", DEVICE_BENCH_DIR)
            self.adb.push(self.assets_dir, f"{DEVICE_BENCH_DIR}/assets/")

    def run_all(self) -> list[ScenarioResult]:
        """
        Execute every scenario and return the collected ScenarioResult list.

        Partial results are written to disk after each scenario so that a
        mid-run ADB disconnection doesn't lose all collected data.

        Returns:
            List of ScenarioResult, one per scenario (failed scenarios included).
        """
        total = len(self.scenarios)
        for i, scenario in enumerate(self.scenarios, start=1):
            logger.info("Scenario %d/%d: %s (%s %s)", i, total,
                        scenario.name, scenario.precision, scenario.task)
            result = self._run_scenario(scenario)
            self.results.append(result)
            _atomic_write_json(
                self.output_dir / "results_partial.json",
                [_scenario_to_dict(r) for r in self.results],
            )
        return self.results

    # ------------------------------------------------------------------
    # Private scenario execution
    # ------------------------------------------------------------------

    def _run_scenario(self, scenario: ScenarioConfig) -> ScenarioResult:
        """Run one scenario: collect pre-run metrics, warmup, measure, post-run."""
        result = ScenarioResult(
            scenario_name=scenario.name,
            model=scenario.model,
            precision=scenario.precision,
        )

        battery_before = self._safe_read_battery()
        oom_before = self._safe_read_oom()

        poller = ThermalPoller(self.adb)
        poller.start()

        try:
            logger.info("  Warmup (%d iters)…", WARMUP_ITERS)
            for wi in range(WARMUP_ITERS):
                try:
                    self._run_once(scenario, label=f"warmup-{wi + 1}")
                except (ADBError, ValueError) as exc:
                    logger.debug("  Warmup iter %d failed (ignored): %s", wi + 1, exc)

            logger.info("  Measuring (%d iters)…", MEASURE_ITERS)
            for mi in range(MEASURE_ITERS):
                iter_result = self._run_with_retry(scenario, label=f"iter-{mi + 1}")
                result.iterations.append(iter_result)

        except ADBError as exc:
            logger.error("ADB disconnected during '%s': %s", scenario.name, exc)
            result.failed = True
            result.error = str(exc)
            _atomic_write_json(
                self.output_dir / "results_partial.json",
                [_scenario_to_dict(r) for r in self.results + [result]],
            )
        finally:
            poller.stop()

        battery_after = self._safe_read_battery()
        oom_after = self._safe_read_oom()

        successful = [it for it in result.iterations if it.success]
        if not successful:
            result.failed = True
            result.error = result.error or "All measurement iterations failed"
            return result

        decode_vals = [it.decode_tps for it in successful]
        prefill_vals = [it.prefill_tps for it in successful]
        ttft_vals = [it.ttft_ms for it in successful]
        ram_vals = [it.ram_mb for it in successful]

        battery_drain = (
            compute_battery_drain(battery_before, battery_after)
            if battery_before and battery_after
            else 0.0
        )

        result.metrics = MetricsBundle(
            decode_tps=_mean(decode_vals),
            prefill_tps=_mean(prefill_vals),
            ttft_ms=_mean(ttft_vals),
            ram_mb=_mean(ram_vals),
            battery_drain_pct=battery_drain,
            thermal_throttle_detected=detect_thermal_throttle(poller.snapshots),
            thermal_peak_celsius=peak_thermal(poller.snapshots),
            memory_pressure_events=max(0, oom_after - oom_before),
            decode_tps_stddev=_stddev(decode_vals),
            prefill_tps_stddev=_stddev(prefill_vals),
        )

        m = result.metrics
        logger.info(
            "  → decode=%.1f±%.1f tps  prefill=%.1f tps  TTFT=%.1fms  RAM=%s",
            m.decode_tps, m.decode_tps_stddev,
            m.prefill_tps, m.ttft_ms,
            f"{m.ram_mb:.0f}MB",
        )
        if m.thermal_throttle_detected:
            logger.warning("  ⚠ Thermal throttling detected — results may be understated")

        return result

    def _run_with_retry(
        self, scenario: ScenarioConfig, label: str
    ) -> IterationResult:
        """
        Run a single measurement iteration, retrying up to RETRY_LIMIT times.

        Retries use exponential backoff (2, 4, 8 seconds) to give the device
        time to settle before the next attempt.

        Returns:
            IterationResult with ``success=False`` if all retries are exhausted.
        """
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                return self._run_once(scenario, label=label)
            except (ADBError, ValueError) as exc:
                logger.warning("  %s attempt %d/%d failed: %s", label, attempt, RETRY_LIMIT, exc)
                if attempt < RETRY_LIMIT:
                    time.sleep(2 ** attempt)

        return IterationResult(
            prefill_tps=0.0, decode_tps=0.0, ttft_ms=0.0, ram_mb=0.0,
            success=False, error=f"Failed after {RETRY_LIMIT} attempts",
        )

    def _run_once(self, scenario: ScenarioConfig, label: str) -> IterationResult:
        """
        Invoke the benchmark binary for one iteration and parse its output.

        The binary is the ``test_performance`` executable from the existing
        tests/ build, configured via environment variables that mirror the
        pattern established in tests/android/run.sh.

        Args:
            scenario: The scenario to run.
            label: Human-readable label for log messages.

        Returns:
            Parsed IterationResult.

        Raises:
            ADBError: If the shell command fails or times out.
            ValueError: If the binary output cannot be parsed.
        """
        model_slug = scenario.model.split("/")[-1].lower()
        model_path = f"{DEVICE_BENCH_DIR}/weights/{model_slug}"

        env = (
            f"CACTUS_TEST_MODEL={model_path} "
            f"CACTUS_NO_CLOUD_TELE=1 "
        )
        if scenario.task == "stt" and scenario.audio_file:
            env += f"CACTUS_TEST_ASSETS={DEVICE_BENCH_DIR}/assets "
        if scenario.task == "vlm" and scenario.image_file:
            env += f"CACTUS_TEST_ASSETS={DEVICE_BENCH_DIR}/assets "

        cmd = (
            f"cd {DEVICE_BENCH_DIR} && "
            f"{env}"
            f"./{self._binary_name} "
            f"--context-length {scenario.context_length} "
            f"--max-tokens {scenario.max_tokens} "
            f"--task {scenario.task} "
            f"--precision {scenario.precision}"
        )
        logger.debug("  [%s] %s", label, cmd)

        output = self.adb.shell(cmd, timeout=300)
        parsed = parse_cactus_response(output)

        # Prefer the RAM value reported in the JSON response (already peak RSS).
        # Fall back to /proc/meminfo only if the binary didn't populate it.
        ram_mb = parsed.ram_usage_mb if parsed.ram_usage_mb > 0 else self._fallback_ram_mb()

        return IterationResult(
            prefill_tps=parsed.prefill_tps,
            decode_tps=parsed.decode_tps,
            ttft_ms=parsed.time_to_first_token_ms,
            ram_mb=ram_mb,
            success=True,
        )

    def _fallback_ram_mb(self) -> float:
        """Read MemAvailable from /proc/meminfo as a RAM proxy."""
        try:
            line = self.adb.shell("cat /proc/meminfo | grep MemAvailable")
            return int(line.split()[1]) / 1024.0
        except Exception:
            return 0.0

    def _safe_read_battery(self):
        """Read battery state, returning None on failure."""
        try:
            return read_battery(self.adb)
        except ADBError as exc:
            logger.debug("Battery read failed: %s", exc)
            return None

    def _safe_read_oom(self) -> int:
        """Read OOM event count, returning 0 on failure."""
        try:
            return read_memory_pressure_events(self.adb)
        except ADBError as exc:
            logger.debug("OOM read failed: %s", exc)
            return 0


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_scenarios(path: str, precision_override: Optional[str] = None) -> list[ScenarioConfig]:
    """
    Load and validate scenario definitions from a YAML file.

    If ``precision_override`` is provided, it replaces the precision value for
    every scenario — useful for running a full INT8 sweep without editing the
    YAML.

    Args:
        path: Filesystem path to scenarios.yaml.
        precision_override: Optional precision string to apply to all scenarios.

    Returns:
        List of ScenarioConfig objects.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        KeyError: If a required field is missing from a scenario entry.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    configs: list[ScenarioConfig] = []
    for item in raw.get("scenarios", []):
        configs.append(
            ScenarioConfig(
                name=item["name"],
                model=item["model"],
                precision=precision_override or item.get("precision", "INT4"),
                task=item.get("task", "completion"),
                context_length=item.get("context_length", 1024),
                prompt=item.get("prompt", ""),
                max_tokens=item.get("max_tokens", 100),
                audio_file=item.get("audio_file"),
                image_file=item.get("image_file"),
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def _scenario_to_dict(r: ScenarioResult) -> dict:
    """Serialise a ScenarioResult to a JSON-compatible dict."""
    d: dict = {
        "scenario_name": r.scenario_name,
        "model": r.model,
        "precision": r.precision,
        "failed": r.failed,
        "error": r.error,
        "iterations": [
            {
                "prefill_tps": it.prefill_tps,
                "decode_tps": it.decode_tps,
                "ttft_ms": it.ttft_ms,
                "ram_mb": it.ram_mb,
                "success": it.success,
                "error": it.error,
            }
            for it in r.iterations
        ],
    }
    if r.metrics:
        m = r.metrics
        d["metrics"] = {
            "decode_tps": m.decode_tps,
            "decode_tps_stddev": m.decode_tps_stddev,
            "prefill_tps": m.prefill_tps,
            "prefill_tps_stddev": m.prefill_tps_stddev,
            "ttft_ms": m.ttft_ms,
            "ram_mb": m.ram_mb,
            "battery_drain_pct": m.battery_drain_pct,
            "thermal_throttle_detected": m.thermal_throttle_detected,
            "thermal_peak_celsius": m.thermal_peak_celsius,
            "memory_pressure_events": m.memory_pressure_events,
        }
    return d


def results_to_render_payload(
    scenario_results: list[ScenarioResult],
    device_info: dict,
) -> list[dict]:
    """
    Map scenario results onto the column layout expected by the renderer.

    The README table has three columns keyed on model type (lfm_1_2b,
    lfmvl_1_6b, parakeet_1_1b). Scenario names are matched by substring so
    the mapping survives minor naming changes in scenarios.yaml.

    Args:
        scenario_results: Completed scenario results from BenchmarkRunner.run_all().
        device_info: Device metadata dict from ADBClient.device_info().

    Returns:
        Single-element list containing the device payload dict for the renderer.
    """
    payload: dict = {
        "device_name": device_info.get("model", "Unknown"),
        "thermal_throttle_detected": False,
        "thermal_peak_celsius": 0.0,
        "battery_drain_pct": 0.0,
        "memory_pressure_events": 0,
    }

    for r in scenario_results:
        if r.failed or r.metrics is None:
            continue

        m = r.metrics
        metrics_dict = {
            "prefill_tps": m.prefill_tps,
            "prefill_tps_stddev": m.prefill_tps_stddev,
            "decode_tps": m.decode_tps,
            "decode_tps_stddev": m.decode_tps_stddev,
            "ttft_ms": m.ttft_ms,
            "ram_mb": m.ram_mb,
        }

        name = r.scenario_name.lower()
        if "lfm_1_2b" in name or ("lfm" in name and "vl" not in name and "1_2" in name):
            payload["lfm_1_2b"] = metrics_dict
        elif "lfmvl" in name or "vlm" in name or ("lfm" in name and "vl" in name):
            payload["lfmvl_1_6b"] = metrics_dict
        elif "parakeet" in name or "stt" in name:
            payload["parakeet_1_1b"] = metrics_dict

        # Accumulate worst-case battery/thermal across all scenarios
        if m.thermal_throttle_detected:
            payload["thermal_throttle_detected"] = True
        if m.thermal_peak_celsius > payload["thermal_peak_celsius"]:
            payload["thermal_peak_celsius"] = m.thermal_peak_celsius
        if m.battery_drain_pct > payload["battery_drain_pct"]:
            payload["battery_drain_pct"] = m.battery_drain_pct
        payload["memory_pressure_events"] += m.memory_pressure_events

        # RAM from 4k context scenario, fall back to any scenario
        if "4k" in name:
            payload["ram_mb"] = m.ram_mb
        else:
            payload.setdefault("ram_mb", m.ram_mb)

    return [payload]


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: pathlib.Path, data: object) -> None:
    """
    Write ``data`` as JSON to ``path`` using an atomic rename.

    Writes to a sibling temp file first, then calls ``os.replace()`` so
    the destination is never left in a partially-written state. Safe to
    call concurrently from the SIGINT handler.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=str(path.parent), delete=False, suffix=".tmp"
    ) as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for ``python bench.py`` and ``python -m bench``."""
    parser = argparse.ArgumentParser(
        description="Android benchmarking harness for the Cactus inference engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--serial", required=True,
        help="ADB device serial from `adb devices`",
    )
    parser.add_argument(
        "--binary",
        default="../../build/android/test_performance",
        help="Path to the arm64-v8a test_performance binary",
    )
    parser.add_argument(
        "--weights",
        default="../../weights",
        help="Local weights/ directory containing model GGUF files",
    )
    parser.add_argument(
        "--scenarios",
        default="scenarios.yaml",
        help="Path to scenarios YAML file",
    )
    parser.add_argument(
        "--assets",
        default="../../tests/assets",
        help="Path to test assets directory (audio/image files for VLM/STT)",
    )
    parser.add_argument(
        "--precision",
        choices=["INT4", "INT8", "FP16"],
        default=None,
        help="Override precision for all scenarios (default: use scenarios.yaml value)",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results JSON and Markdown",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    script_dir = pathlib.Path(__file__).parent
    binary_path = (script_dir / args.binary).resolve()
    weights_dir = (script_dir / args.weights).resolve()
    scenarios_path = (script_dir / args.scenarios).resolve()
    assets_dir = (script_dir / args.assets).resolve()
    output_dir = (script_dir / args.output).resolve()

    if not binary_path.exists():
        logger.error(
            "Binary not found: %s\n"
            "Build for Android first:\n"
            "  cactus build --android\n"
            "  # or: cmake --build build/android -j$(nproc)",
            binary_path,
        )
        return 1

    if not weights_dir.exists():
        logger.error(
            "Weights directory not found: %s\n"
            "Download models first:\n"
            "  cactus download LiquidAI/LFM2.5-1.2B-Instruct\n"
            "  cactus download LiquidAI/LFM2.5-VL-1.6B\n"
            "  cactus download nvidia/parakeet-ctc-1.1b",
            weights_dir,
        )
        return 1

    try:
        adb = ADBClient(serial=args.serial)
    except ADBError as exc:
        logger.error("%s", exc)
        return 1

    device_info = adb.device_info()
    logger.info(
        "Connected: %s (Android %s, %d MB RAM)",
        device_info["model"],
        device_info["android_version"],
        device_info.get("total_ram_mb", 0),
    )

    scenarios = load_scenarios(str(scenarios_path), precision_override=args.precision)
    logger.info("Loaded %d scenario(s) from %s", len(scenarios), scenarios_path.name)

    runner = BenchmarkRunner(
        adb=adb,
        scenarios=scenarios,
        binary_path=str(binary_path),
        weights_dir=str(weights_dir),
        output_dir=str(output_dir),
        assets_dir=str(assets_dir) if assets_dir.exists() else None,
    )

    def _on_interrupt(sig, frame):
        logger.warning("Interrupted — partial results saved to %s", output_dir)
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_interrupt)

    logger.info("Setting up device…")
    runner.setup()

    logger.info("Running benchmarks…")
    scenario_results = runner.run_all()

    # Build render payload and write output files
    precision_label = args.precision or (scenarios[0].precision if scenarios else "INT4")
    render_payload = results_to_render_payload(scenario_results, device_info)
    scenario_info = {"precision": precision_label}

    header_md = render_device_header(
        {**device_info, "device_name": device_info["model"]},
        scenario_info,
    )
    readme_table = render_readme_table(render_payload)
    extended_table = render_extended_table(render_payload)
    json_str = render_json_artifact(
        [_scenario_to_dict(r) for r in scenario_results],
        metadata={
            "device": device_info,
            "serial": args.serial,
            "precision": precision_label,
            "scenarios_file": scenarios_path.name,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "results.json", json.loads(json_str))

    md_content = (
        f"{header_md}\n\n"
        f"#### Standard Benchmark Table\n\n{readme_table}\n\n"
        f"#### Extended Metrics (battery + thermal)\n\n{extended_table}\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", dir=str(output_dir), delete=False, suffix=".tmp"
    ) as f:
        f.write(md_content)
        tmp_md = f.name
    os.replace(tmp_md, str(output_dir / "results.md"))

    # Print to stdout for CI log capture
    print("\n" + "=" * 64)
    print(header_md)
    print(readme_table)
    print()
    print(extended_table)
    print("=" * 64)
    logger.info("Results written to %s/", output_dir)

    failed = sum(1 for r in scenario_results if r.failed)
    if failed:
        logger.warning("%d/%d scenario(s) failed", failed, len(scenario_results))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
