#!/usr/bin/env python3
"""Generate the 5-graph benchmark deck from matmul_bench / attn_bench CSV.

Reads:
  matmul_bench.csv  (graphs 1, 2, 3)
  attn_bench.csv    (graphs 4, 5)

Writes one PNG per graph + a combined "all_graphs.png" with all five panels.

Plots use the monotonic latency envelope for each backend so the rendered
curves show scaling shape while the CSV keeps mean and p50 latencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as e:
    sys.exit(f"missing dependency: {e}. install with: pip install pandas matplotlib")


# Graph spec: (csv stem, graph filter, title, y-axis label, sweep meaning).
GRAPHS = [
    ("matmul", "gemv_d",            "Graph 1 — Q4 GEMV (1 × d) @ (d × N)",                            "Hidden dimension (d)"),
    ("matmul", "gemm_d",            "Graph 2 — Q4 GEMM (M × d) @ (d × N), M=N=512",                   "Hidden dimension (d)"),
    ("matmul", "gemm_mn",           "Graph 3 — Q4 GEMM (M × d) @ (d × N), M=N swept, d=512",          "Matrix rows/cols (M=N)"),
    ("attn",   "attn_prefill_s",    "Graph 4 — Attention FP16 (S × d) @ (d × S), d=1024, h=8",         "Sequence length (S)"),
    ("attn",   "attn_decode_cache", "Graph 5 — Q8 Decode Attention (1 × d) @ (d × 1), d=1024, h=8",    "KV cache length (cache_len)"),
]

# Stable backend → color mapping. Cactus stands out; the others share a muted
# palette. Unknown backends fall back to matplotlib's default cycle.
BACKEND_COLORS = {
    "cactus_cq4":                        "#1b9e77",
    "cactus_prefill":                    "#1b9e77",
    "cactus_decode":                     "#1b9e77",
    "ggml_q4_0":                         "#377eb8",
    "ggml_q8_0":                         "#377eb8",
    "ggml_fa_q8_prefill":                "#377eb8",
    "ggml_fa_q8_decode":                 "#377eb8",
    "ggml_mm_q8_prefill":                "#377eb8",
    "ggml_mm_q8_decode":                 "#377eb8",
    "litert_q4_fc":                      "#ff7f00",
    "litert_ruy_int8":                   "#fdbf6f",
    "litert_neon":                       "#ff7f00",
    "litert_ruy":                        "#fdbf6f",
    "litert_ruy_prefill":                "#fdbf6f",
    "litert_ruy_decode":                 "#fdbf6f",
    "litert_neon_decode":                "#ff7f00",
    "onnxrt_q4":                         "#e41a1c",
    "onnxrt_gqa_prefill":                "#e41a1c",
    "onnxrt_gqa_decode_fp16kv":          "#e41a1c",
    "executorch_q4":                     "#666666",
    "executorch_sdpa_prefill_fp32":      "#666666",
    "executorch_qsdpa_decode_int8pc":    "#666666",
}

# Stable order so the legend doesn't shuffle between graphs.
BACKEND_ORDER = list(BACKEND_COLORS.keys())

BACKEND_LABELS = {
    "cactus_cq4":                        "Cactus CQ4",
    "cactus_prefill":                    "Cactus FP16 prefill",
    "cactus_decode":                     "Cactus Q8 decode",
    "ggml_q4_0":                         "ggml Q4_0",
    "ggml_fa_q8_prefill":                "ggml FA Q8_0 prefill",
    "ggml_fa_q8_decode":                 "ggml FA Q8_0 decode",
    "ggml_mm_q8_prefill":                "ggml matmul Q8_0 prefill",
    "ggml_mm_q8_decode":                 "ggml matmul Q8_0 decode",
    "litert_q4_fc":                      "LiteRT int4 FC",
    "litert_ruy_int8":                   "LiteRT Ruy int8",
    "litert_neon":                       "LiteRT NEON int8 baseline",
    "litert_ruy":                        "LiteRT Ruy int8 baseline",
    "litert_ruy_prefill":                "LiteRT Ruy int8",
    "litert_ruy_decode":                 "LiteRT Ruy int8",
    "litert_neon_decode":                "LiteRT NEON int8",
    "onnxrt_q4":                         "ONNX RT Q4 group",
    "onnxrt_gqa_prefill":                "ONNX RT GQA prefill",
    "onnxrt_gqa_decode_fp16kv":          "ONNX RT GQA FP16 KV",
    "executorch_q4":                     "ExecuTorch Q4 channel",
    "executorch_sdpa_prefill_fp32":      "ExecuTorch SDPA FP32",
    "executorch_qsdpa_decode_int8pc":    "ExecuTorch QSDPA int8",
}

# Asterisks for backends whose precision/scheme differs from cactus's. Listed
# in plot footnotes so the y-axis comparison stays honest.
BACKEND_ASTERISKS = {
    "cactus_cq4":                        "*",
    "litert_q4_fc":                      "†",
    "litert_ruy_int8":                   "‡",
}
ASTERISK_LEGEND = {
    "*": "Cactus CQ4 uses rotor+LUT quantization to achieve the lowest observed error",
    "†": "LiteRT int4 FC uses per-output-channel q4 weights; fewer scales than group/CQ4, faster but lower-fidelity",
    "‡": "LiteRT Ruy uses per-output-channel int8 weights; it does not support q4 in this benchmark path",
}


def color_for(backend: str) -> str:
    return BACKEND_COLORS.get(backend, "#444444")


def order_backends(backends: list[str]) -> list[str]:
    known = [b for b in BACKEND_ORDER if b in backends]
    unknown = sorted(set(backends) - set(known))
    return known + unknown


def plot_graph(ax, df: pd.DataFrame, graph_filter: str, title: str, x_label: str) -> bool:
    """Line plot: X = swept dim (log), Y = monotonic latency envelope."""
    sub = df[df["graph"] == graph_filter].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"(no rows for graph={graph_filter})",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_title(title)
        ax.set_axis_off()
        return False

    latency_col = "p50_us" if "p50_us" in sub.columns else "time_us"
    latency_name = "p50" if latency_col == "p50_us" else "mean"
    sub["time_ms"] = sub[latency_col] / 1000.0
    dims = sorted(sub["sweep_dim"].unique())
    backends = order_backends(list(sub["backend"].unique()))

    asterisks_used = set()
    for backend in backends:
        b_data = (sub[sub["backend"] == backend]
                  .sort_values("sweep_dim"))
        if b_data.empty:
            continue
        # Drop non-positive times (log scale can't show zero/negative).
        b_data = b_data[b_data["time_ms"] > 0]
        mark = BACKEND_ASTERISKS.get(backend, "")
        label = BACKEND_LABELS.get(backend, backend) + (" " + mark if mark else "")
        if mark:
            asterisks_used.update(mark.split())
        b_data = b_data.assign(plot_time_ms=np.maximum.accumulate(b_data["time_ms"].to_numpy()))
        ax.plot(b_data["sweep_dim"], b_data["plot_time_ms"],
                marker="o", markersize=5, linewidth=1.8,
                label=label, color=color_for(backend))

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{latency_name} monotonic latency envelope (ms)")
    ax.set_title(title, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)

    # Asterisk footnote — only show keys actually used in this panel.
    if asterisks_used:
        notes = [f"{k} {ASTERISK_LEGEND[k]}" for k in sorted(asterisks_used,
                                                              key=lambda x: list(ASTERISK_LEGEND).index(x))]
        if latency_col == "p50_us":
            notes.append("CSV time_us=mean and p50_us=median; plotted lines use p50 cumulative max by backend")
        else:
            notes.append("CSV lacks p50_us; plotted lines use mean cumulative max by backend")
        ax.text(0.0, -0.16, "\n".join(notes),
                transform=ax.transAxes, fontsize=7, color="#555",
                ha="left", va="top")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matmul-csv", default="matmul_bench.csv",
                    help="path to matmul_bench CSV (default: matmul_bench.csv)")
    ap.add_argument("--attn-csv",   default="attn_bench.csv",
                    help="path to attn_bench CSV (default: attn_bench.csv)")
    ap.add_argument("--out-dir",    default=".",
                    help="output directory for PNGs (default: .)")
    ap.add_argument("--no-combined", action="store_true",
                    help="skip the combined all_graphs.png")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = {"matmul": Path(args.matmul_csv), "attn": Path(args.attn_csv)}
    dfs: dict[str, pd.DataFrame] = {}
    for key, path in csv_paths.items():
        if not path.exists():
            print(f"warn: {path} not found — graphs sourced from '{key}' will be empty", file=sys.stderr)
            dfs[key] = pd.DataFrame(columns=["graph", "sweep_dim", "backend", "time_us"])
            continue
        dfs[key] = pd.read_csv(path)

    # One PNG per graph.
    for source, gfilter, title, ylabel in GRAPHS:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        plot_graph(ax, dfs[source], gfilter, title, ylabel)
        fig.tight_layout()
        out_path = out_dir / f"{gfilter}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    # Combined panel.
    if not args.no_combined:
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes_flat = axes.flatten()
        for ax, (source, gfilter, title, ylabel) in zip(axes_flat, GRAPHS):
            plot_graph(ax, dfs[source], gfilter, title, ylabel)
        # Hide the unused 6th subplot.
        axes_flat[-1].set_axis_off()
        fig.suptitle("cactus vs llama.cpp vs litert vs executorch vs onnx — quantized kernels",
                     fontsize=13, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out_path = out_dir / "all_graphs.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
