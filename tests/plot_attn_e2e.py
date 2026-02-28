#!/usr/bin/env python3
import sys
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} <input.csv> [output.png]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) == 3 else "attn_e2e_comparison.png"

    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["variant"]
            token_idx = int(row["token_index"])
            time_ms = float(row["time_ms"])
            data.setdefault(variant, {}).setdefault(token_idx, []).append(time_ms)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"baseline": "#2196F3", "interleaved": "#FF5722", "deferscale": "#4CAF50"}

    for variant, tokens in sorted(data.items()):
        indices = sorted(tokens.keys())
        color = colors.get(variant, "#888888")

        decode_indices = [i for i in indices if i > 0]
        decode_means = np.array([np.mean(tokens[i]) for i in decode_indices])
        decode_stds = np.array([np.std(tokens[i]) for i in decode_indices])

        ax.plot(decode_indices, decode_means, label=variant, color=color, linewidth=1.0, alpha=0.9)
        ax.fill_between(decode_indices, decode_means - decode_stds, decode_means + decode_stds,
                        color=color, alpha=0.15)

        all_decode = []
        for i in decode_indices:
            all_decode.extend(tokens[i])
        all_decode = np.array(all_decode)
        print(f"{variant:>12s}  mean={np.mean(all_decode):.3f}ms  "
              f"p50={np.median(all_decode):.3f}ms  "
              f"p99={np.percentile(all_decode, 99):.3f}ms  "
              f"std={np.std(all_decode):.3f}ms")

    ax.set_title("Attention Kernel E2E Comparison")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Decode Latency (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    main()
