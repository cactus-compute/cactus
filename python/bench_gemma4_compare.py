#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "weights/gemma-4-e2b-it"
DEFAULT_PROMPT = "Write one friendly sentence about local AI."
RESULT_PREFIX = "RESULT_JSON="
CASE_ENVS = [
    ("mps_ram", {"CACTUS_MPS": "1", "CACTUS_WEIGHT_STORAGE": "ram"}),
    ("mps_mmap", {"CACTUS_MPS": "1", "CACTUS_WEIGHT_STORAGE": "mmap"}),
    ("cpu_ram", {"CACTUS_MPS": "0", "CACTUS_WEIGHT_STORAGE": "ram"}),
    ("cpu_mmap", {"CACTUS_MPS": "0", "CACTUS_WEIGHT_STORAGE": "mmap"}),
]
LONG_CONTEXT_BLOCK = (
    "Edge inference keeps AI workloads close to the user, which reduces latency, "
    "improves privacy, cuts bandwidth use, keeps applications responsive offline, "
    "and avoids repeated round trips to remote servers. "
    "It also gives product teams more predictable performance, because the model can "
    "respond immediately from local state instead of waiting on network conditions. "
    "When the workload is interactive, this can make the difference between a tool "
    "feeling instant and a tool feeling sluggish."
)


def build_prompt(preset):
    if preset == "short":
        return DEFAULT_PROMPT
    if preset == "long_context":
        sections = [
            "Summarize the repeated ideas below in one concise paragraph.",
            *(f"Passage {i + 1}: {LONG_CONTEXT_BLOCK}" for i in range(16)),
        ]
        return "\n".join(sections)
    raise ValueError(f"Unknown prompt preset: {preset}")


def _average_metrics(rows):
    keys = [
        "time_to_first_token_ms",
        "total_time_ms",
        "prefill_tps",
        "decode_tps",
        "ram_usage_mb",
        "prefill_tokens",
        "decode_tokens",
        "total_tokens",
    ]
    avg = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        avg[key] = statistics.fmean(values) if values else 0.0
    return avg


def _child_main(args):
    from python.src.cactus import cactus_complete, cactus_destroy, cactus_init, cactus_reset

    prompt = args.prompt if args.prompt is not None else build_prompt(args.prompt_preset)
    messages = json.dumps([{"role": "user", "content": prompt}])
    options = json.dumps({
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": args.max_tokens,
    })

    init_start = time.perf_counter()
    model = cactus_init(args.model, None, False)
    init_ms = (time.perf_counter() - init_start) * 1000.0

    def run_once():
        raw = cactus_complete(model, messages, options, None, None)
        parsed = json.loads(raw)
        if not parsed.get("success", False):
            raise RuntimeError(parsed.get("error") or "cactus_complete failed")
        return parsed

    for _ in range(args.warmup_runs):
        cactus_reset(model)
        run_once()

    runs = []
    for _ in range(args.runs):
        cactus_reset(model)
        runs.append(run_once())

    cactus_destroy(model)

    payload = {
        "label": args.label,
        "model": args.model,
        "prompt": prompt,
        "prompt_preset": args.prompt_preset,
        "env": {
            "CACTUS_MPS": os.getenv("CACTUS_MPS", ""),
            "CACTUS_DISABLE_MPS": os.getenv("CACTUS_DISABLE_MPS", ""),
            "CACTUS_WEIGHT_STORAGE": os.getenv("CACTUS_WEIGHT_STORAGE", ""),
            "CACTUS_MPS_TRACE": os.getenv("CACTUS_MPS_TRACE", ""),
            "CACTUS_MPS_TRACE_SUMMARY": os.getenv("CACTUS_MPS_TRACE_SUMMARY", ""),
        },
        "init_ms": init_ms,
        "avg": _average_metrics(runs),
        "last_response": runs[-1].get("response", "") if runs else "",
        "runs": runs,
    }
    print(RESULT_PREFIX + json.dumps(payload))


def _run_case(script_path, base_args, label, extra_env):
    child_args = [
        sys.executable,
        str(script_path),
        "--child",
        "--label",
        label,
        "--model",
        base_args.model,
        "--prompt-preset",
        base_args.prompt_preset,
        "--runs",
        str(base_args.runs),
        "--warmup-runs",
        str(base_args.warmup_runs),
        "--max-tokens",
        str(base_args.max_tokens),
    ]
    if base_args.prompt is not None:
        child_args.extend(["--prompt", base_args.prompt])

    env = os.environ.copy()
    env.update(extra_env)
    if base_args.trace_mps and label.startswith("mps_"):
        env["CACTUS_MPS_TRACE"] = "1"
        env["CACTUS_MPS_TRACE_SUMMARY"] = "1"
    proc = subprocess.run(
        child_args,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
    if payload is None:
        raise RuntimeError(f"No benchmark payload found for {label}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    return {
        "payload": payload,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _format_row(label, payload):
    avg = payload["avg"]
    return (
        f"{label:12} "
        f"prefill_tokens={avg['prefill_tokens']:7.1f}  "
        f"init={payload['init_ms']:7.1f} ms  "
        f"ttft={avg['time_to_first_token_ms']:7.1f} ms  "
        f"total={avg['total_time_ms']:7.1f} ms  "
        f"prefill={avg['prefill_tps']:7.2f} tok/s  "
        f"decode={avg['decode_tps']:7.2f} tok/s  "
        f"ram={avg['ram_usage_mb']:8.1f} MB"
    )


def _parent_main(args):
    script_path = Path(__file__).resolve()
    cases = CASE_ENVS

    if args.case != "all":
        cases = [case for case in cases if case[0] == args.case]

    results = []
    for label, env in cases:
        results.append((label, _run_case(script_path, args, label, env)))

    prompt_preview = results[0][1]["payload"]["prompt"] if results else (args.prompt or build_prompt(args.prompt_preset))
    print(f"Model:  {args.model}")
    print(f"Prompt preset: {args.prompt_preset}")
    if args.prompt is not None:
        print("Prompt source: explicit --prompt")
    print(f"Prompt preview: {prompt_preview[:160].replace(chr(10), ' ')}{'...' if len(prompt_preview) > 160 else ''}")
    print(f"Runs:   {args.runs} measured, {args.warmup_runs} warmup")
    print("")
    for label, result in results:
        print(_format_row(label, result["payload"]))
    print("")
    for label, result in results:
        print(f"[{label}] env={result['payload']['env']}")
        response = result["payload"].get("last_response", "").strip()
        if response:
            print(f"[{label}] response={response}")
        stderr = result["stderr"].strip()
        if stderr:
            print(f"[{label}] stderr:")
            print(stderr)


def main():
    parser = argparse.ArgumentParser(description="Compare Gemma 4 E2B benchmark settings.")
    case_choices = ["all", *(label for label, _ in CASE_ENVS)]
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", default="run")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-preset", choices=["short", "long_context"], default="short")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--case", choices=case_choices, default="all")
    parser.add_argument("--trace-mps", action="store_true")
    args = parser.parse_args()

    if args.child:
        _child_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
