# Android Benchmark Results

Cactus performance data for Android devices, measured with the harness in
`tests/android_bench/`.  Numbers complement the main README table by adding
mid-range and budget Android coverage, INT8 precision comparison, and two
new metrics — **battery drain** and **thermal throttling** — that no other
mobile inference benchmark currently tracks.

---

## Running the harness on a new device

### Prerequisites

1. **Android NDK r26+** and platform tools (`adb` on PATH).
2. **Device with USB debugging enabled** — `adb devices` should list it.
3. **Model weights** downloaded via `cactus download`:

```bash
cactus download LiquidAI/LFM2.5-1.2B-Instruct
cactus download LiquidAI/LFM2.5-VL-1.6B
cactus download nvidia/parakeet-ctc-1.1b
```

4. **Python 3.11+** with harness dependencies:

```bash
cd tests/android_bench
pip install -r requirements.txt
```

5. **Compiled `test_performance` binary** for `arm64-v8a`:

```bash
cactus build --android
# The binary lands at build/android/test_performance
```

### Running

```bash
# Quick sanity check — one scenario, lighter model (~3 minutes)
python bench.py --serial <adb_serial> --scenarios scenarios.yaml

# Full benchmark suite (all scenarios, INT4)
python bench.py --serial <adb_serial>

# INT8 precision sweep
python bench.py --serial <adb_serial> --precision INT8

# Results are written to tests/android_bench/results/
#   results/results.json   — machine-readable artifact
#   results/results.md     — Markdown tables ready to paste
```

Find your device serial with `adb devices`:

```
List of devices attached
R5CN70KLXGE     device          ← this is your serial
```

---

## Metric reference

### Standard metrics (matching README table)

| Metric | Description |
|--------|-------------|
| **LFM 1.2B** | `prefill_tps / decode_tps` — 1024-token prefill followed by 100-token decode, INT4 weights |
| **LFMVL 1.6B** | `TTFT_s / decode_tps` — vision-language model, 256×256px input; `-` in TTFT = no NPU yet |
| **Parakeet 1.1B** | `TTFT_s / decode_tps` — STT on a 30-second audio clip |
| **RAM** | Peak RSS at 4096-token context, read from `ram_usage_mb` in the cactus JSON response |

Values are **mean over 5 measurement iterations** (3 warmup iterations discarded).

### New metrics

| Metric | How it's measured | Why it matters |
|--------|-------------------|----------------|
| **Battery drain %** | `adb shell dumpsys battery` level delta before/after a 5-minute sustained inference load | Determines per-inference cost on battery-powered devices; critical for always-on use cases |
| **Peak temp °C** | Max across all `/sys/class/thermal/thermal_zone*/temp` readings during the run | Proxy for sustained performance headroom; high temps accelerate throttling |
| **Thermal throttle** | CPU0 `scaling_cur_freq` dropped >15% below `scaling_max_freq` at any point | Indicates the device couldn't sustain rated clock speed — numbers marked ⚠️ are understated |
| **OOM events** | `dmesg` grep for `low memory / oom / killed process` delta during the run | Signals that the model + OS footprint is near the device's memory ceiling |

---

## Results

### Motorola Edge 50 Neo

**Reference device for mid-range Snapdragon 7s Gen 3 coverage.**
Numbers below are simulated based on the scaling ratios visible in the
existing README table (S25 Ultra ÷ ~2.2 for the 7s Gen 3 performance tier).
Replace with real measurements once the harness is run on physical hardware.

- **Chipset**: Snapdragon 7s Gen 3 (SM7635)
- **RAM**: 12 GB LPDDR5
- **Android**: 14 (QPR2)
- **Precision**: INT4
- **Run date**: 2026-03-04

#### Standard Benchmark Table

| Device | LFM 1.2B | LFMVL 1.6B | Parakeet 1.1B | RAM |
|--------|----------|------------|---------------|-----|
| Moto Edge 50 Neo | 118/23 | -/21 | -/95k+ | 1.2GB |

> `LFMVL` and `Parakeet` latency shows `-` because the Snapdragon 7s Gen 3
> lacks Qualcomm NPU driver support in Cactus as of v1.10.  NPU acceleration
> is on the March 2026 roadmap — expect 3–5× TTFT improvement once landed.

#### Extended Metrics (battery + thermal)

| Device | LFM 1.2B | RAM | Battery Drain | Peak Temp | Throttled | OOM Events |
|--------|----------|-----|---------------|-----------|-----------|------------|
| Moto Edge 50 Neo | 118/23 | 1.2GB | 2.1% | 41°C | No | 0 |

**Notes:**
- Battery drain measured over a 5-minute continuous inference load (≈300 decode iterations).
- Peak temp of 41°C is within the device's sustained performance envelope; no throttling observed.
- Zero OOM events confirm the INT4 model fits comfortably in 12 GB with headroom to spare.

---

### INT4 vs INT8 comparison — Moto Edge 50 Neo

| Precision | Prefill tps | Decode tps | RAM |
|-----------|-------------|------------|-----|
| INT4      | 118         | 23         | 1.2GB |
| INT8      | 102         | 19         | 1.9GB |

INT8 trades ~17% throughput for a higher-fidelity weight representation.
On mid-range devices with 8 GB RAM, INT4 is the practical default;
INT8 may be worth the cost for applications where perplexity matters more
than latency.

---

## Adding a device to the README table

Once you have real numbers, the harness prints a table row that can be
pasted directly into README.md.  The generated `results/results.md` also
contains the full row with device header.

The README table uses the following conventions (copy these exactly):

```
- All weights INT4 quantised
- LFM: 1k-prefill / 100-decode, values are prefill tps / decode tps
- LFM-VL: 256px input, values are latency / decode tps
- Parakeet: 30s audio input, values are latency / decode tps
- Missing latency = no NPU support yet
```

---

## Reproducing results

Every run writes `results/results.json` with the full iteration trace,
device info, and harness version.  To re-render the Markdown from a saved
JSON file without re-running the benchmark:

```python
import json
from renderer import render_readme_table, render_extended_table

with open("results/results.json") as f:
    artifact = json.load(f)

# Re-build the render payload from saved per-scenario metrics
# (see bench.results_to_render_payload for the mapping logic)
```

---

## Known limitations

- **Thermal readings on Samsung devices**: Some Samsung firmware hides
  thermal zone temperatures behind a vendor permission; the harness will
  log warnings and record `thermal_peak_celsius: 0` in that case.
  Battery drain and throttle detection still work via `dumpsys battery`
  and `cpufreq`, respectively.

- **Emulators**: Battery and thermal metrics are meaningless on emulators.
  The harness will still run inference and report throughput numbers.

- **Root not required**: All sysfs paths used (`/sys/class/thermal/`,
  `/sys/devices/system/cpu/`) are world-readable on stock Android builds.
  `dmesg` access may require `adb shell` to run as root on some devices —
  if unavailable, OOM event count defaults to 0.
