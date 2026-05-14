#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
    "prompt_name",
    "shape",
    "mean_decode_tps",
    "min_decode_tps",
    "baseline_mean_decode_tps",
    "mean_ratio_vs_baseline",
    "min_ratio_vs_baseline",
    "avg_target_forward_ms_per_forward",
    "avg_assistant_ms_per_pass",
    "avg_assistant_passes_per_forward",
    "avg_assistant_total_ms_per_forward",
    "avg_accepted_drafts_per_forward",
    "avg_emitted_tokens_per_forward",
    "avg_other_ms_per_forward",
    "avg_total_ms_per_forward",
    "avg_allocated_ms_per_token_after_first",
    "cost_amortization_ratio",
]


def f(value):
    return float(value or 0.0)


def read_csv(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def mean(values):
    return statistics.mean(values) if values else 0.0


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_bench_rows(paths):
    rows = []
    for path in paths:
        rows.extend(read_csv(path))
    return rows


def summarize(args):
    bench_rows = load_bench_rows(args.bench_csv)
    round_rows = read_csv(Path(args.trace_dir) / "round_trace.csv")
    token_rows = read_csv(Path(args.trace_dir) / "token_trace.csv")

    bench_by_case = defaultdict(list)
    for row in bench_rows:
        prompt = row.get("prompt") or row.get("prompt_name") or "custom"
        shape = row.get("shape") or ("baseline" if row.get("mtp_max_draft") == "0" else f"base_main{row.get('mtp_max_draft')}")
        bench_by_case[(prompt, shape)].append(row)

    baseline_tps = {}
    for (prompt, shape), rows in bench_by_case.items():
        if shape == "baseline":
            baseline_tps[prompt] = mean([f(r["decode_tps"]) for r in rows])

    rounds_by_case = defaultdict(list)
    for row in round_rows:
        rounds_by_case[(row["prompt_name"], row["shape"])].append(row)

    tokens_by_case = defaultdict(list)
    for row in token_rows:
        tokens_by_case[(row["prompt_name"], row["shape"])].append(row)

    baseline_ms_per_token = {}
    for (prompt, shape), rows in tokens_by_case.items():
        if shape == "baseline":
            baseline_ms_per_token[prompt] = mean([f(r["allocated_total_ms"]) for r in rows])

    summary = []
    for key in sorted(set(bench_by_case) | set(rounds_by_case) | set(tokens_by_case)):
        prompt, shape = key
        benches = bench_by_case.get(key, [])
        rounds = rounds_by_case.get(key, [])
        tokens = tokens_by_case.get(key, [])
        decode_tps = [f(r["decode_tps"]) for r in benches]
        base_tps = baseline_tps.get(prompt, 0.0)
        target_ms = [f(r["target_forward_ms"]) for r in rounds]
        assistant_total_ms = [f(r["assistant_total_ms"]) for r in rounds]
        assistant_passes = [f(r["assistant_pass_count"]) for r in rounds]
        other_ms = [
            f(r["sampling_or_argmax_ms"]) + f(r["kv_transaction_ms"]) + f(r["callback_stream_ms"]) + f(r["loop_overhead_ms"])
            for r in rounds
        ]
        total_ms = [f(r["round_total_ms"]) for r in rounds]
        emitted = [f(r["generated_tokens_emitted"]) for r in rounds]
        accepted = [f(r["accepted_drafts"]) for r in rounds]
        allocated = [f(r["allocated_total_ms"]) for r in tokens]
        assistant_pass_sum = sum(assistant_passes)
        avg_total = mean(total_ms)
        base_ms = baseline_ms_per_token.get(prompt, 0.0)
        cost_amortization = (base_ms * mean(emitted) / avg_total) if avg_total > 0.0 else 0.0
        summary.append({
            "prompt_name": prompt,
            "shape": shape,
            "mean_decode_tps": f"{mean(decode_tps):.3f}",
            "min_decode_tps": f"{min(decode_tps) if decode_tps else 0.0:.3f}",
            "baseline_mean_decode_tps": f"{base_tps:.3f}",
            "mean_ratio_vs_baseline": f"{(mean(decode_tps) / base_tps) if base_tps else 0.0:.3f}",
            "min_ratio_vs_baseline": f"{(min(decode_tps) / base_tps) if base_tps and decode_tps else 0.0:.3f}",
            "avg_target_forward_ms_per_forward": f"{mean(target_ms):.3f}",
            "avg_assistant_ms_per_pass": f"{(sum(assistant_total_ms) / assistant_pass_sum) if assistant_pass_sum else 0.0:.3f}",
            "avg_assistant_passes_per_forward": f"{mean(assistant_passes):.3f}",
            "avg_assistant_total_ms_per_forward": f"{mean(assistant_total_ms):.3f}",
            "avg_accepted_drafts_per_forward": f"{mean(accepted):.3f}",
            "avg_emitted_tokens_per_forward": f"{mean(emitted):.3f}",
            "avg_other_ms_per_forward": f"{mean(other_ms):.3f}",
            "avg_total_ms_per_forward": f"{avg_total:.3f}",
            "avg_allocated_ms_per_token_after_first": f"{mean(allocated):.3f}",
            "cost_amortization_ratio": f"{cost_amortization:.3f}",
        })

    write_csv(Path(args.output_dir) / "summary_by_prompt_shape.csv", summary, SUMMARY_FIELDS)

    best_rows = []
    by_prompt = defaultdict(list)
    for row in summary:
        by_prompt[row["prompt_name"]].append(row)
    for prompt, rows in sorted(by_prompt.items()):
        best = max(rows, key=lambda r: float(r["cost_amortization_ratio"]))
        best_rows.append(best)
    write_csv(Path(args.output_dir) / "best_shape_by_prompt.csv", best_rows, SUMMARY_FIELDS)

    lines = [
        "# Gemma 4 MTP Fixed-Shape Cost Diagnostic Summary",
        "",
        "The diagnostic unit is post-first-token decode work. Cost amortization above 1.0 means the measured emitted-token benefit offsets the target, assistant, and other per-forward costs.",
        "",
        "## Per-Prompt Findings",
        "",
    ]
    for prompt, rows in sorted(by_prompt.items()):
        baseline = next((r for r in rows if r["shape"] == "baseline"), None)
        mtp_rows = [r for r in rows if r["shape"] != "baseline"]
        if not mtp_rows:
            continue
        best = max(mtp_rows, key=lambda r: float(r["cost_amortization_ratio"]))
        worst = min(mtp_rows, key=lambda r: float(r["cost_amortization_ratio"]))
        base_ms = float(baseline["avg_allocated_ms_per_token_after_first"]) if baseline else 0.0
        lines.append(
            f"- `{prompt}`: best `{best['shape']}` emits {best['avg_emitted_tokens_per_forward']} tokens/forward "
            f"while paying {best['avg_target_forward_ms_per_forward']} ms target + "
            f"{best['avg_assistant_total_ms_per_forward']} ms assistant + {best['avg_other_ms_per_forward']} ms other "
            f"(amortization {best['cost_amortization_ratio']}, mean TPS ratio {best['mean_ratio_vs_baseline']}); "
            f"worst `{worst['shape']}` emits {worst['avg_emitted_tokens_per_forward']} with amortization "
            f"{worst['cost_amortization_ratio']} against a {base_ms:.3f} ms baseline token cost."
        )

    lines.extend(["", "## Shape Table", ""])
    lines.append("| prompt | shape | emitted/fwd | target ms | assistant ms | other ms | total ms | amortization | mean TPS ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            f"| {row['prompt_name']} | {row['shape']} | {row['avg_emitted_tokens_per_forward']} | "
            f"{row['avg_target_forward_ms_per_forward']} | {row['avg_assistant_total_ms_per_forward']} | "
            f"{row['avg_other_ms_per_forward']} | {row['avg_total_ms_per_forward']} | "
            f"{row['cost_amortization_ratio']} | {row['mean_ratio_vs_baseline']} |"
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "diagnostic_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-csv", nargs="+", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
