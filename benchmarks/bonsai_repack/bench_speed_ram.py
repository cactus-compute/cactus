"""Speed + RAM benchmark for a full-27B bundle, greedy local-only.

Usage: bench_speed_ram.py <bundle_dir> <out.json>
Run under `/usr/bin/time -l` to capture the process-wide peak footprint too.

Reports per workload: prefill_tokens, prefill_tps, time_to_first_token_ms,
decode_tokens, decode_tps, and RSS checkpoints (current + peak) around each
phase. Prompts are sized in tokens by repeating a paragraph; greedy decoding
with auto_handoff off, matching the gate E harness conditions.
"""

import json
import resource
import subprocess
import sys
import time

from cactus.bindings.cactus import cactus_complete, cactus_destroy, cactus_init

SYSTEM = "You are a helpful assistant"
PARA = (
    "The quick brown fox jumps over the lazy dog while the river runs east "
    "past the old mill, and the miller counts sacks of grain under a pale "
    "autumn sky, noting each delivery in a leather-bound ledger. "
)
WORKLOADS = [
    {"name": "short", "paras": 0, "max_tokens": 128, "runs": 3},
    {"name": "medium", "paras": 6, "max_tokens": 128, "runs": 1},
    {"name": "long", "paras": 14, "max_tokens": 128, "runs": 1},
]


def rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(subprocess.os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024.0


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def footprint_mb() -> float | None:
    """macOS phys_footprint of this process (the number Activity Monitor shows)."""
    try:
        out = subprocess.run(["footprint", str(subprocess.os.getpid())],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if "phys_footprint" in line.lower() or "footprint:" in line.lower():
                return line.strip()
    except Exception:
        pass
    return None


def build_prompt(paras: int) -> str:
    if paras == 0:
        return "Explain why the sky is blue in about one hundred words."
    return (
        PARA * paras
        + "\nSummarize the passage above in one sentence, then say how many "
        "times the mill is mentioned."
    )


def main(bundle: str, out_path: str) -> None:
    results = {"bundle": bundle, "workloads": [], "rss_mb": {}}
    results["rss_mb"]["before_init"] = rss_mb()

    t0 = time.perf_counter()
    model = cactus_init(bundle)
    init_s = time.perf_counter() - t0
    results["init_seconds"] = round(init_s, 2)
    results["rss_mb"]["after_init"] = rss_mb()
    print(f"init {init_s:.2f}s, rss {results['rss_mb']['after_init']:.0f} MB")

    try:
        for w in WORKLOADS:
            prompt = build_prompt(w["paras"])
            for run in range(w["runs"]):
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ]
                t0 = time.perf_counter()
                resp = cactus_complete(
                    model, messages,
                    options={"temperature": 0.0, "max_tokens": w["max_tokens"],
                             "auto_handoff": False},
                )
                wall = time.perf_counter() - t0
                row = {
                    "workload": w["name"],
                    "run": run,
                    "wall_s": round(wall, 2),
                    "prefill_tokens": resp.get("prefill_tokens"),
                    "prefill_tps": resp.get("prefill_tps"),
                    "ttft_ms": resp.get("time_to_first_token_ms"),
                    "decode_tokens": resp.get("decode_tokens"),
                    "decode_tps": resp.get("decode_tps"),
                    "rss_after_mb": rss_mb(),
                    "peak_after_mb": peak_mb(),
                }
                results["workloads"].append(row)
                print(
                    f"{w['name']}[{run}] wall {wall:5.1f}s | "
                    f"prefill {row['prefill_tokens']} tok @ {row['prefill_tps']} tps | "
                    f"ttft {row['ttft_ms']} ms | "
                    f"decode {row['decode_tokens']} tok @ {row['decode_tps']} tps | "
                    f"rss {row['rss_after_mb']:.0f} MB"
                )
    finally:
        cactus_destroy(model)

    results["rss_mb"]["final"] = rss_mb()
    results["peak_rss_gb"] = round(peak_mb() / 1024.0, 2)
    fp = footprint_mb()
    if fp:
        results["phys_footprint"] = fp
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"peak rss {results['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
