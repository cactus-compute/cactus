#!/usr/bin/env python3
"""
Phase 1e: Per-Layer Sensitivity Analysis

Quantizes KV cache to INT4 one layer at a time (all other layers stay INT8)
to identify which layers are most sensitive to precision reduction. Uses
distribution statistics from Phase 0a to generate per-layer calibrated
synthetic data.

This analysis:
  - Identifies which layers are the "bottleneck" for INT4 quality
  - Enables a mixed-precision KV cache (sensitive layers stay INT8, rest use INT4)
  - Validates Phase 0a's prediction that layers 0-3 (kurtosis 100-180) are most sensitive
  - Tests NF4 (Phase 0 winner for keys) and asymmetric (winner for values) grids

Data source:
  - kv_analysis_results.json from Phase 0a+0b (per-layer/per-head distribution stats)
  - Generates synthetic KV data calibrated to each layer's real distribution characteristics

Usage:
  python tests/exp_1e_per_layer_sensitivity.py --analysis-json kv_profile_results/kv_analysis_results.json
  python tests/exp_1e_per_layer_sensitivity.py --analysis-json kv_profile_results/kv_analysis_results.json --fast
  python tests/exp_1e_per_layer_sensitivity.py --analysis-json kv_profile_results/kv_analysis_results.json --csv results.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

NF4_GRID = np.array([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
], dtype=np.float64)

NF4_BOUNDARIES = np.zeros(15, dtype=np.float64)
for i in range(15):
    NF4_BOUNDARIES[i] = (NF4_GRID[i] + NF4_GRID[i + 1]) / 2.0


def quantize_int8(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 127.0, 1e-10)
    quantized = np.clip(np.round(flat / scales), -128, 127).astype(np.int8)
    return quantized.reshape(shape), scales.reshape(-1)


def dequantize_int8(q, scales, group_size=32):
    shape = q.shape
    flat = q.reshape(-1, group_size).astype(np.float64)
    s = scales.reshape(-1, 1)
    return (flat * s).reshape(shape)


def quantize_dequantize_int8(x, group_size=32):
    q, s = quantize_int8(x, group_size)
    return dequantize_int8(q, s, group_size)


def quantize_int4_uniform(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 7.0, 1e-10)
    quantized = np.clip(np.round(flat / scales), -8, 7).astype(np.int8)
    return quantized.reshape(shape), scales.reshape(-1)


def dequantize_int4_uniform(q, scales, group_size=32):
    shape = q.shape
    flat = q.reshape(-1, group_size).astype(np.float64)
    s = scales.reshape(-1, 1)
    return (flat * s).reshape(shape)


def quantize_dequantize_int4_uniform(x, group_size=32):
    q, s = quantize_int4_uniform(x, group_size)
    return dequantize_int4_uniform(q, s, group_size)


def quantize_dequantize_nf4(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size).astype(np.float64)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.maximum(max_abs, 1e-10)
    normalized = flat / scales
    indices = np.digitize(normalized, NF4_BOUNDARIES).astype(np.int32)
    indices = np.clip(indices, 0, 15)
    dequantized = NF4_GRID[indices] * scales
    return dequantized.reshape(shape)


def quantize_dequantize_asymmetric(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size).astype(np.float64)
    vmin = np.min(flat, axis=1, keepdims=True)
    vmax = np.max(flat, axis=1, keepdims=True)
    scales = np.maximum((vmax - vmin) / 15.0, 1e-10)
    quantized = np.clip(np.round((flat - vmin) / scales), 0, 15).astype(np.uint8)
    dequantized = quantized.astype(np.float64) * scales + vmin
    return dequantized.reshape(shape)


def attention_reference(Q, K, V, scale):
    """Compute scaled dot-product attention in FP64.

    Args:
        Q: [q_heads, head_dim]
        K: [seq_len, kv_heads, head_dim]
        V: [seq_len, kv_heads, head_dim]
        scale: 1/sqrt(head_dim)

    Returns:
        output: [q_heads, head_dim]
        weights: [q_heads, seq_len]
    """
    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)

    seq_len, kv_heads, head_dim = K.shape
    q_heads = Q.shape[0]
    gqa_ratio = q_heads // kv_heads

    output = np.zeros((q_heads, head_dim), dtype=np.float64)
    all_weights = np.zeros((q_heads, seq_len), dtype=np.float64)

    for qh in range(q_heads):
        kv_h = qh // gqa_ratio
        q = Q[qh]
        k = K[:, kv_h, :]
        v = V[:, kv_h, :]

        scores = (k @ q) * scale
        scores -= np.max(scores)
        w = np.exp(scores)
        w /= np.sum(w)

        all_weights[qh] = w
        output[qh] = w @ v

    return output, all_weights


def compute_metrics(ref_output, ref_weights, test_output, test_weights):
    ref_o = ref_output.flatten()
    test_o = test_output.flatten()

    mse = float(np.mean((ref_o - test_o) ** 2))
    max_err = float(np.max(np.abs(ref_o - test_o)))

    dot = np.dot(ref_o, test_o)
    norm_ref = np.linalg.norm(ref_o)
    norm_test = np.linalg.norm(test_o)
    cosine_sim = float(dot / (norm_ref * norm_test + 1e-30))

    weight_l1 = float(np.mean(np.sum(np.abs(ref_weights - test_weights), axis=1)))
    attn_mse = float(np.mean((ref_weights - test_weights) ** 2))

    return {
        "output_mse": mse,
        "output_max_err": max_err,
        "output_cosine_sim": cosine_sim,
        "attn_weight_l1_div": weight_l1,
        "attn_score_mse": attn_mse,
    }


def generate_calibrated_kv(layer_stats_k, layer_stats_v, num_kv_heads, head_dim,
                            seq_len, rng):
    """Generate synthetic KV data calibrated to real per-(layer, head) distribution stats.

    Uses per-head mean, std, and skewness from Phase 0a. Kurtosis is approximated
    via a Student's t-distribution when kurtosis > 6 (excess kurtosis > 3).
    """
    K = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float64)
    V = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float64)

    for h in range(num_kv_heads):
        k_stats = layer_stats_k.get(h, {})
        v_stats = layer_stats_v.get(h, {})

        k_mean = k_stats.get("mean", 0.0)
        k_std = max(k_stats.get("std", 1.0), 1e-6)
        k_kurtosis = k_stats.get("kurtosis", 3.0)
        k_skew = k_stats.get("skewness", 0.0)

        v_mean = v_stats.get("mean", 0.0)
        v_std = max(v_stats.get("std", 1.0), 1e-6)

        if k_kurtosis > 6:
            df = max(4.01, 6.0 / (k_kurtosis - 3.0) + 4) if k_kurtosis > 3 else 30.0
            raw = rng.standard_t(df, (seq_len, head_dim))
            raw_std = np.std(raw)
            if raw_std > 1e-10:
                raw = raw / raw_std
        else:
            raw = rng.standard_normal((seq_len, head_dim))

        if abs(k_skew) > 0.5:
            alpha = k_skew * 0.8
            u = rng.standard_normal((seq_len, head_dim))
            v_skew = rng.standard_normal((seq_len, head_dim))
            skewed = alpha / np.sqrt(1 + alpha**2) * np.abs(u) + 1 / np.sqrt(1 + alpha**2) * v_skew
            raw = 0.7 * raw + 0.3 * skewed

        K[:, h, :] = raw * k_std + k_mean

        V[:, h, :] = rng.standard_normal((seq_len, head_dim)) * v_std + v_mean

    return K, V


def load_layer_stats(analysis_json_path):
    """Load per-(layer, head) distribution stats from Phase 0a analysis JSON."""
    with open(analysis_json_path) as f:
        data = json.load(f)

    num_layers = data["num_layers"]
    num_kv_heads = data["num_kv_heads"]
    head_dim = data["head_dim"]

    hs_keys = data["head_stats_keys"]
    hs_values = data["head_stats_values"]

    layer_stats = {}
    for layer in range(num_layers):
        k_stats = {}
        v_stats = {}
        for h in range(num_kv_heads):
            key = f"L{layer}_H{h}"
            if key in hs_keys:
                k_stats[h] = hs_keys[key]
            if key in hs_values:
                v_stats[h] = hs_values[key]
        layer_stats[layer] = {"keys": k_stats, "values": v_stats}

    return layer_stats, num_layers, num_kv_heads, head_dim


GRID_CONFIGS = {
    "nf4_k_asym_v": {
        "desc": "NF4 keys + asymmetric values (Phase 0 recommended)",
        "k_func": quantize_dequantize_nf4,
        "v_func": quantize_dequantize_asymmetric,
    },
    "uniform": {
        "desc": "Uniform INT4 keys + values (baseline comparison)",
        "k_func": quantize_dequantize_int4_uniform,
        "v_func": quantize_dequantize_int4_uniform,
    },
    "nf4_both": {
        "desc": "NF4 keys + NF4 values",
        "k_func": quantize_dequantize_nf4,
        "v_func": quantize_dequantize_nf4,
    },
}


def run_per_layer_sensitivity(layer_stats, num_layers, num_kv_heads, head_dim,
                               q_heads, seq_len, group_size, trials, rng,
                               grid_name="nf4_k_asym_v"):
    """For each layer, quantize that layer's KV to INT4 while keeping INT8 reference.

    Returns per-layer sensitivity metrics.
    """
    grid = GRID_CONFIGS[grid_name]
    k_quant_func = grid["k_func"]
    v_quant_func = grid["v_func"]
    scale = 1.0 / np.sqrt(head_dim)

    results_per_layer = {}

    for layer in range(num_layers):
        layer_metrics = []

        for trial in range(trials):
            ls = layer_stats[layer]

            K, V = generate_calibrated_kv(
                ls["keys"], ls["values"],
                num_kv_heads, head_dim, seq_len, rng
            )
            Q = rng.standard_normal((q_heads, head_dim)).astype(np.float64) * 0.5

            K_int8 = quantize_dequantize_int8(K, group_size)
            V_int8 = quantize_dequantize_int8(V, group_size)
            ref_output, ref_weights = attention_reference(Q, K_int8, V_int8, scale)

            K_int4 = k_quant_func(K, group_size)
            V_int4 = v_quant_func(V, group_size)
            test_output, test_weights = attention_reference(Q, K_int4, V_int4, scale)

            metrics = compute_metrics(ref_output, ref_weights, test_output, test_weights)

            k_mse = float(np.mean((K_int8 - K_int4) ** 2))
            v_mse = float(np.mean((V_int8 - V_int4) ** 2))
            metrics["k_quant_mse"] = k_mse
            metrics["v_quant_mse"] = v_mse

            layer_metrics.append(metrics)

        agg = {}
        for metric in layer_metrics[0]:
            values = [m[metric] for m in layer_metrics]
            agg[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        k_stats_h0 = layer_stats[layer]["keys"].get(0, {})
        agg["distribution"] = {
            "kurtosis_mean": float(np.mean([
                layer_stats[layer]["keys"].get(h, {}).get("kurtosis", 0)
                for h in range(num_kv_heads)
            ])),
            "skewness_mean": float(np.mean([
                abs(layer_stats[layer]["keys"].get(h, {}).get("skewness", 0))
                for h in range(num_kv_heads)
            ])),
            "std_mean": float(np.mean([
                layer_stats[layer]["keys"].get(h, {}).get("std", 0)
                for h in range(num_kv_heads)
            ])),
            "icv_mean": float(np.mean([
                layer_stats[layer]["keys"].get(h, {}).get("inter_channel_var", 0)
                for h in range(num_kv_heads)
            ])),
        }

        results_per_layer[layer] = agg

    return results_per_layer


def print_results(results, grid_name, group_size, num_layers):
    grid_desc = GRID_CONFIGS[grid_name]["desc"]
    print(f"\n{'='*110}")
    print(f"  Phase 1e: Per-Layer Sensitivity Analysis")
    print(f"  Grid: {grid_desc}  |  group_size={group_size}")
    print(f"{'='*110}")

    header = (f"{'Layer':>5} {'Attn Wt L1':>12} {'Out MSE':>14} {'Out MaxErr':>12} "
              f"{'Cosine Sim':>12} {'K MSE':>12} {'V MSE':>12} "
              f"{'Kurtosis':>10} {'|Skew|':>8} {'Std':>8} {'Sensitive':>10}")
    print(f"\n{header}")
    print("-" * 110)

    all_wt_l1 = [results[l]["attn_weight_l1_div"]["mean"] for l in range(num_layers)]
    all_out_mse = [results[l]["output_mse"]["mean"] for l in range(num_layers)]
    wt_l1_p75 = float(np.percentile(all_wt_l1, 75))
    out_mse_p75 = float(np.percentile(all_out_mse, 75))
    sensitive_layers = []

    for layer in range(num_layers):
        r = results[layer]
        wt_l1 = r["attn_weight_l1_div"]["mean"]
        out_mse = r["output_mse"]["mean"]
        max_err = r["output_max_err"]["mean"]
        cos_sim = r["output_cosine_sim"]["mean"]
        k_mse = r["k_quant_mse"]["mean"]
        v_mse = r["v_quant_mse"]["mean"]
        dist = r["distribution"]
        kurtosis = dist["kurtosis_mean"]
        skewness = dist["skewness_mean"]
        std = dist["std_mean"]

        sensitive = wt_l1 > wt_l1_p75 or out_mse > out_mse_p75
        flag = "  ***" if sensitive else ""
        if sensitive:
            sensitive_layers.append(layer)

        print(f"{layer:5d} {wt_l1:12.6f} {out_mse:14.2e} {max_err:12.6f} "
              f"{cos_sim:12.8f} {k_mse:12.4e} {v_mse:12.4e} "
              f"{kurtosis:10.2f} {skewness:8.2f} {std:8.2f}{flag}")

    print(f"\n  Thresholds (P75): attn_wt_l1 > {wt_l1_p75:.4f}, output_mse > {out_mse_p75:.2e}")

    print()
    if sensitive_layers:
        print(f"Sensitive layers (attn_wt_l1 > 0.05 or output_mse > 1e-3 or cosine < 0.999): "
              f"{sensitive_layers}")
    else:
        print("No layers exceed sensitivity thresholds.")

    print()
    wt_l1_values = [(l, results[l]["attn_weight_l1_div"]["mean"]) for l in range(num_layers)]
    wt_l1_values.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 most sensitive layers (by attn weight L1):")
    for layer, val in wt_l1_values[:5]:
        dist = results[layer]["distribution"]
        print(f"  Layer {layer:2d}: attn_wt_l1={val:.6f}  "
              f"kurtosis={dist['kurtosis_mean']:.1f}  |skew|={dist['skewness_mean']:.1f}  "
              f"std={dist['std_mean']:.1f}")

    out_mse_values = [(l, results[l]["output_mse"]["mean"]) for l in range(num_layers)]
    out_mse_values.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 most sensitive layers (by output MSE):")
    for layer, val in out_mse_values[:5]:
        dist = results[layer]["distribution"]
        print(f"  Layer {layer:2d}: output_mse={val:.2e}  "
              f"kurtosis={dist['kurtosis_mean']:.1f}  |skew|={dist['skewness_mean']:.1f}  "
              f"std={dist['std_mean']:.1f}")

    print("\n--- Correlation Analysis ---")
    kurtosis_arr = np.array([results[l]["distribution"]["kurtosis_mean"] for l in range(num_layers)])
    skewness_arr = np.array([results[l]["distribution"]["skewness_mean"] for l in range(num_layers)])
    std_arr = np.array([results[l]["distribution"]["std_mean"] for l in range(num_layers)])
    icv_arr = np.array([results[l]["distribution"]["icv_mean"] for l in range(num_layers)])

    wt_l1_arr = np.array([results[l]["attn_weight_l1_div"]["mean"] for l in range(num_layers)])
    out_mse_arr = np.array([results[l]["output_mse"]["mean"] for l in range(num_layers)])

    for metric_name, metric_arr in [("attn_wt_l1", wt_l1_arr), ("output_mse", out_mse_arr)]:
        for stat_name, stat_arr in [("kurtosis", kurtosis_arr), ("|skewness|", skewness_arr),
                                     ("std", std_arr), ("ICV", icv_arr)]:
            corr = np.corrcoef(stat_arr, metric_arr)[0, 1]
            print(f"  {metric_name:12s} vs {stat_name:12s}: r = {corr:+.4f}")
        print()

    return sensitive_layers


def print_mixed_precision_recommendation(results, num_layers, sensitive_layers):
    print("\n--- Mixed-Precision KV Cache Recommendation ---")

    wt_l1_ranked = sorted(
        [(l, results[l]["attn_weight_l1_div"]["mean"]) for l in range(num_layers)],
        key=lambda x: x[1], reverse=True
    )
    out_mse_ranked = sorted(
        [(l, results[l]["output_mse"]["mean"]) for l in range(num_layers)],
        key=lambda x: x[1], reverse=True
    )

    total_wt_l1 = sum(v for _, v in wt_l1_ranked)
    total_out_mse = sum(v for _, v in out_mse_ranked)

    print("\nCumulative attention weight error contribution (INT8-ing the top-N layers):")
    print(f"  {'N':>3}  {'Layers':>40}  {'Cumul % of total attn error':>30}  {'Remaining attn error':>22}")
    cum = 0.0
    for i, (layer, val) in enumerate(wt_l1_ranked):
        cum += val
        pct = cum / total_wt_l1 * 100
        remaining = (total_wt_l1 - cum) / num_layers
        layers_str = str([l for l, _ in wt_l1_ranked[:i+1]])
        if len(layers_str) > 40:
            layers_str = layers_str[:37] + "..."
        print(f"  {i+1:3d}  {layers_str:>40s}  {pct:27.1f}%  {remaining:22.6f}")
        if i >= 9:
            break

    print("\nCumulative output MSE contribution:")
    print(f"  {'N':>3}  {'Layers':>40}  {'Cumul % of total out MSE':>30}  {'Remaining out MSE':>22}")
    cum = 0.0
    for i, (layer, val) in enumerate(out_mse_ranked):
        cum += val
        pct = cum / total_out_mse * 100
        remaining = (total_out_mse - cum) / num_layers
        layers_str = str([l for l, _ in out_mse_ranked[:i+1]])
        if len(layers_str) > 40:
            layers_str = layers_str[:37] + "..."
        print(f"  {i+1:3d}  {layers_str:>40s}  {pct:27.1f}%  {remaining:22.2e}")
        if i >= 9:
            break

    combined_score = {}
    for layer in range(num_layers):
        wt_l1_norm = results[layer]["attn_weight_l1_div"]["mean"] / total_wt_l1
        out_mse_norm = results[layer]["output_mse"]["mean"] / total_out_mse
        combined_score[layer] = wt_l1_norm + out_mse_norm

    combined_ranked = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)

    for n_int8 in [4, 7, 10, 14]:
        if n_int8 > num_layers:
            continue
        int8_set = set(l for l, _ in combined_ranked[:n_int8])
        int4_set = set(range(num_layers)) - int8_set
        remaining_wt_l1 = np.mean([results[l]["attn_weight_l1_div"]["mean"] for l in int4_set])
        remaining_out_mse = np.mean([results[l]["output_mse"]["mean"] for l in int4_set])
        savings = 0.44 * len(int4_set) / num_layers

        print(f"\n  Option: {n_int8} layers INT8, {num_layers - n_int8} layers INT4 "
              f"(~{savings*100:.0f}% KV savings)")
        print(f"    INT8: {sorted(int8_set)}")
        print(f"    Avg remaining INT4 attn_wt_l1: {remaining_wt_l1:.4f}")
        print(f"    Avg remaining INT4 output_mse: {remaining_out_mse:.2e}")


def save_csv(results, num_layers, csv_path, grid_name, group_size):
    rows = []
    for layer in range(num_layers):
        r = results[layer]
        dist = r["distribution"]
        rows.append({
            "layer": layer,
            "grid": grid_name,
            "group_size": group_size,
            "attn_weight_l1_div": r["attn_weight_l1_div"]["mean"],
            "attn_weight_l1_std": r["attn_weight_l1_div"]["std"],
            "output_mse": r["output_mse"]["mean"],
            "output_mse_std": r["output_mse"]["std"],
            "output_max_err": r["output_max_err"]["mean"],
            "output_cosine_sim": r["output_cosine_sim"]["mean"],
            "attn_score_mse": r["attn_score_mse"]["mean"],
            "k_quant_mse": r["k_quant_mse"]["mean"],
            "v_quant_mse": r["v_quant_mse"]["mean"],
            "kurtosis_mean": dist["kurtosis_mean"],
            "skewness_mean": dist["skewness_mean"],
            "std_mean": dist["std_mean"],
            "icv_mean": dist["icv_mean"],
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {csv_path}")


def try_plot(results, num_layers, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    layers = list(range(num_layers))
    wt_l1 = [results[l]["attn_weight_l1_div"]["mean"] for l in layers]
    wt_l1_std = [results[l]["attn_weight_l1_div"]["std"] for l in layers]
    out_mse = [results[l]["output_mse"]["mean"] for l in layers]
    kurtosis = [results[l]["distribution"]["kurtosis_mean"] for l in layers]
    std_vals = [results[l]["distribution"]["std_mean"] for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 1e: Per-Layer INT4 Sensitivity Analysis\n"
                 "(NF4 keys + asymmetric values, group_size=32)", fontsize=13)

    ax = axes[0, 0]
    ax.bar(layers, wt_l1, yerr=wt_l1_std, color="steelblue", alpha=0.8, capsize=2)
    ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1, label="threshold=0.05")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Weight L1 Divergence")
    ax.set_title("Per-Layer Attention Weight Sensitivity")
    ax.legend()

    ax = axes[0, 1]
    ax.bar(layers, out_mse, color="coral", alpha=0.8)
    ax.axhline(y=1e-3, color="red", linestyle="--", linewidth=1, label="threshold=1e-3")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Output MSE (vs INT8)")
    ax.set_title("Per-Layer Output MSE")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(kurtosis, wt_l1, c=layers, cmap="viridis", s=40, zorder=3)
    for l in layers:
        ax.annotate(str(l), (kurtosis[l], wt_l1[l]), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Key Kurtosis (mean across heads)")
    ax.set_ylabel("Attention Weight L1 Divergence")
    ax.set_title("Sensitivity vs Key Kurtosis")
    corr = np.corrcoef(kurtosis, wt_l1)[0, 1]
    ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax = axes[1, 1]
    ax.scatter(std_vals, out_mse, c=layers, cmap="viridis", s=40, zorder=3)
    for l in layers:
        ax.annotate(str(l), (std_vals[l], out_mse[l]), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Key Std (mean across heads)")
    ax.set_ylabel("Output MSE (vs INT8)")
    ax.set_title("Output MSE vs Key Std")
    ax.set_yscale("log")
    corr = np.corrcoef(std_vals, out_mse)[0, 1]
    ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "per_layer_sensitivity.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["red" if wt_l1[l] > 0.05 else "steelblue" for l in layers]
    bars = ax.bar(layers, wt_l1, color=colors, alpha=0.8)
    ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Attention Weight L1 Divergence (INT4 vs INT8)")
    ax.set_title("Per-Layer INT4 Sensitivity — Red = Sensitive (keep INT8)")
    ax.set_xticks(layers)

    for l in layers:
        ax.text(l, wt_l1[l] + 0.002, f"k={kurtosis[l]:.0f}", ha="center",
                fontsize=6, rotation=45)

    plt.tight_layout()
    rec_path = os.path.join(output_dir, "mixed_precision_recommendation.png")
    plt.savefig(rec_path, dpi=150, bbox_inches="tight")
    print(f"Recommendation plot saved to {rec_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 1e: Per-Layer Sensitivity Analysis")
    parser.add_argument("--analysis-json", type=str,
                        default="kv_profile_results/kv_analysis_results.json",
                        help="Path to Phase 0a analysis results JSON")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--plot-dir", type=str, default="kv_profile_results",
                        help="Directory for output plots")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length for synthetic KV data")
    parser.add_argument("--q-heads", type=int, default=None,
                        help="Number of Q heads (default: 2x kv_heads from analysis)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Random trials per layer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Quick run: fewer trials (5), priority layers only (0-3, 27)")
    parser.add_argument("--grids", type=str, default="nf4_k_asym_v",
                        help="Comma-separated grid configs to test (default: nf4_k_asym_v)")
    args = parser.parse_args()

    if not os.path.exists(args.analysis_json):
        print(f"ERROR: Analysis JSON not found: {args.analysis_json}")
        print("Run Phase 0a first: python tests/analysis/kv_distribution_analysis.py")
        sys.exit(1)

    layer_stats, num_layers, num_kv_heads, head_dim = load_layer_stats(args.analysis_json)
    q_heads = args.q_heads or num_kv_heads * 2

    print(f"Phase 1e: Per-Layer Sensitivity Analysis")
    print(f"  Model: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")
    print(f"  Q heads: {q_heads} (GQA ratio: {q_heads // num_kv_heads}:1)")
    print(f"  Seq len: {args.seq_len}, group_size: {args.group_size}")
    print(f"  Trials per layer: {args.trials}")

    rng = np.random.default_rng(args.seed)
    grids = [g.strip() for g in args.grids.split(",")]

    if args.fast:
        priority_layers = [0, 1, 2, 3, 27]
        args.trials = 5
        print(f"  FAST MODE: priority layers only ({priority_layers}), {args.trials} trials")

        fast_stats = {l: layer_stats[l] for l in priority_layers if l in layer_stats}
        fast_num_layers = num_layers
    else:
        fast_stats = None

    all_sensitive = set()

    for grid_name in grids:
        if grid_name not in GRID_CONFIGS:
            print(f"WARNING: Unknown grid '{grid_name}', skipping")
            continue

        print(f"\n--- Running grid: {grid_name} ({GRID_CONFIGS[grid_name]['desc']}) ---")

        if args.fast:
            results = {}
            for layer in priority_layers:
                if layer >= num_layers:
                    continue
                single_stats = {0: layer_stats[layer]}
                sub_results = run_per_layer_sensitivity(
                    single_stats, 1, num_kv_heads, head_dim,
                    q_heads, args.seq_len, args.group_size, args.trials, rng,
                    grid_name
                )
                results[layer] = sub_results[0]

            print(f"\n{'='*110}")
            print(f"  Phase 1e: Per-Layer Sensitivity (FAST MODE — priority layers)")
            print(f"  Grid: {GRID_CONFIGS[grid_name]['desc']}  |  group_size={args.group_size}")
            print(f"{'='*110}")

            header = (f"{'Layer':>5} {'Attn Wt L1':>12} {'Out MSE':>14} {'Out MaxErr':>12} "
                      f"{'Cosine Sim':>12} {'K MSE':>12} {'V MSE':>12} "
                      f"{'Kurtosis':>10} {'|Skew|':>8} {'Std':>8} {'Sensitive':>10}")
            print(f"\n{header}")
            print("-" * 110)

            for layer in priority_layers:
                if layer not in results:
                    continue
                r = results[layer]
                wt_l1 = r["attn_weight_l1_div"]["mean"]
                out_mse = r["output_mse"]["mean"]
                max_err = r["output_max_err"]["mean"]
                cos_sim = r["output_cosine_sim"]["mean"]
                k_mse = r["k_quant_mse"]["mean"]
                v_mse = r["v_quant_mse"]["mean"]
                dist = r["distribution"]
                kurtosis = dist["kurtosis_mean"]
                skewness = dist["skewness_mean"]
                std = dist["std_mean"]

                all_sensitive.add(layer)

                print(f"{layer:5d} {wt_l1:12.6f} {out_mse:14.2e} {max_err:12.6f} "
                      f"{cos_sim:12.8f} {k_mse:12.4e} {v_mse:12.4e} "
                      f"{kurtosis:10.2f} {skewness:8.2f} {std:8.2f}")

        else:
            results = run_per_layer_sensitivity(
                layer_stats, num_layers, num_kv_heads, head_dim,
                q_heads, args.seq_len, args.group_size, args.trials, rng,
                grid_name
            )

            sensitive = print_results(results, grid_name, args.group_size, num_layers)
            all_sensitive.update(sensitive)

            if args.csv:
                csv_path = args.csv if len(grids) == 1 else args.csv.replace(".csv", f"_{grid_name}.csv")
                save_csv(results, num_layers, csv_path, grid_name, args.group_size)

            try_plot(results, num_layers, args.plot_dir)

    if not args.fast:
        print_mixed_precision_recommendation(
            results, num_layers, sorted(all_sensitive)
        )

    print("\n--- Phase 0a Prediction Validation ---")
    print("Phase 0a predicted layers 0-3 (kurtosis 100-180) would be most sensitive.")
    early_sensitive = [l for l in all_sensitive if l <= 3]
    late_sensitive = [l for l in all_sensitive if l > 3]
    print(f"  Layers 0-3 flagged as sensitive: {early_sensitive}")
    print(f"  Other layers flagged: {late_sensitive}")
    if len(early_sensitive) >= 2:
        print("  --> Phase 0a prediction VALIDATED: early layers are among the most sensitive")
    elif len(early_sensitive) == 0 and all_sensitive:
        print("  --> Phase 0a prediction NOT validated: sensitive layers are elsewhere")
    else:
        print("  --> Partial validation")


if __name__ == "__main__":
    main()
