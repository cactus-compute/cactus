#!/usr/bin/env python3
"""
Phase 1d: Long-Context Error Accumulation Test

Measures whether INT4 quantization errors compound over many positions (O(N)
systematic bias) or remain bounded (O(sqrt(N)) random independent errors).

At each decode step, the model attends over all cached positions. At position N,
there are N cached KV pairs, each independently quantized. The output is a
weighted sum of N quantized values, each with independent quantization error e.
If errors are random and independent, the expected error grows as O(sqrt(N)).
If there are systematic biases (e.g., the quantization grid consistently rounds
certain values in the same direction), errors can grow as O(N).

Quantization grids tested (informed by Phase 0 results):
  - Uniform INT4: symmetric absmax, 16 levels {-7..7}
  - NF4: QLoRA grid, Phase 0 winner for keys (lowest MSE + attn weight disruption)
  - Asymmetric uniform: zero-point offset, Phase 0 winner for values
  - INT8: baseline (symmetric absmax, 256 levels)

Configurations tested:
  - K8V8: INT8 baseline
  - K4V4_uniform: uniform INT4 keys + values
  - K4V4_nf4_asym: NF4 keys + asymmetric values (recommended production config)
  - K4V8_nf4: NF4 INT4 keys, INT8 values
  - K8V4_asym: INT8 keys, asymmetric INT4 values (Pareto candidate from Phase 0c)

Metrics at each sequence length checkpoint:
  - KV cache reconstruction MSE (dequantized vs FP32 original)
  - Attention output cosine similarity (quantized vs FP32 reference)
  - Attention weight L1 divergence
  - Output MSE and max absolute error
  - Per-position output error (to detect position-dependent degradation)
  - Sink token contribution error (first 4 positions, never evicted in production)

Error growth analysis:
  - Fits output_mse ~ a * N^b to determine exponent b
  - b ~ 0.5 => O(sqrt(N)), random independent errors (safe)
  - b ~ 1.0 => O(N), systematic bias (problematic)
  - b > 1.0 => superlinear accumulation (catastrophic)

Usage:
  python tests/exp_1d_error_accumulation.py [--csv output.csv]
  python tests/exp_1d_error_accumulation.py --profile-dir kv_profile_results/
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

NF4_BOUNDARIES = (NF4_GRID[:-1] + NF4_GRID[1:]) / 2.0


def quantize_int8(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size).astype(np.float64)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 127.0, 1e-10)
    quantized = np.clip(np.round(flat / scales), -128, 127).astype(np.int8)
    return quantized.reshape(shape), scales.reshape(-1)


def dequantize_int8(q, scales, group_size=32):
    shape = q.shape
    flat = q.reshape(-1, group_size).astype(np.float64)
    return (flat * scales.reshape(-1, 1)).reshape(shape)


def quantize_int4_uniform(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size).astype(np.float64)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 7.0, 1e-10)
    quantized = np.clip(np.round(flat / scales), -8, 7).astype(np.int8)
    return quantized.reshape(shape), scales.reshape(-1)


def dequantize_int4_uniform(q, scales, group_size=32):
    shape = q.shape
    flat = q.reshape(-1, group_size).astype(np.float64)
    return (flat * scales.reshape(-1, 1)).reshape(shape)


def quantize_nf4(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size).astype(np.float64)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.maximum(max_abs, 1e-10)
    normalized = flat / scales
    indices = np.searchsorted(NF4_BOUNDARIES, normalized.ravel()).astype(np.uint8)
    indices = np.clip(indices, 0, 15)
    return indices.reshape(shape), scales.reshape(-1)


def dequantize_nf4(indices, scales, group_size=32):
    shape = indices.shape
    flat_idx = indices.reshape(-1, group_size).astype(np.int64)
    values = NF4_GRID[flat_idx]
    return (values * scales.reshape(-1, 1)).reshape(shape)


def quantize_int4_asymmetric(x, group_size=32):
    shape = x.shape
    flat = x.reshape(-1, group_size).astype(np.float64)
    vmin = np.min(flat, axis=1, keepdims=True)
    vmax = np.max(flat, axis=1, keepdims=True)
    scales = np.maximum((vmax - vmin) / 15.0, 1e-10)
    quantized = np.clip(np.round((flat - vmin) / scales), 0, 15).astype(np.uint8)
    return quantized.reshape(shape), scales.reshape(-1), vmin.reshape(-1)


def dequantize_int4_asymmetric(q, scales, zero_points, group_size=32):
    shape = q.shape
    flat = q.reshape(-1, group_size).astype(np.float64)
    return (flat * scales.reshape(-1, 1) + zero_points.reshape(-1, 1)).reshape(shape)


def quantize_dequantize(x, method, group_size=32):
    if method == "int8":
        q, s = quantize_int8(x, group_size)
        return dequantize_int8(q, s, group_size)
    elif method == "int4_uniform":
        q, s = quantize_int4_uniform(x, group_size)
        return dequantize_int4_uniform(q, s, group_size)
    elif method == "nf4":
        idx, s = quantize_nf4(x, group_size)
        return dequantize_nf4(idx, s, group_size)
    elif method == "int4_asymmetric":
        q, s, zp = quantize_int4_asymmetric(x, group_size)
        return dequantize_int4_asymmetric(q, s, zp, group_size)
    else:
        raise ValueError(f"Unknown method: {method}")


def reconstruction_mse(original, method, group_size=32):
    recon = quantize_dequantize(original, method, group_size)
    return float(np.mean((original.astype(np.float64) - recon) ** 2))


def attention_reference(Q, K, V, scale):
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
        "cosine_sim": cosine_sim,
        "attn_weight_l1": weight_l1,
        "attn_score_mse": attn_mse,
    }


def sink_token_metrics(Q, K_orig, V_orig, K_quant, V_quant, scale, n_sink=4):
    """Measure error contribution from sink tokens (first n_sink positions).

    Sink tokens are never evicted in production (sliding window keeps them).
    If INT4 quantization of sink tokens introduces systematic bias, that bias
    is amplified at every decode step.
    """
    seq_len = K_orig.shape[0]
    if seq_len <= n_sink:
        return None

    Q = Q.astype(np.float64)
    q_heads = Q.shape[0]
    kv_heads = K_orig.shape[1]
    head_dim = K_orig.shape[2]
    gqa_ratio = q_heads // kv_heads

    sink_errors = []
    nonsink_errors = []

    for qh in range(q_heads):
        kv_h = qh // gqa_ratio

        k_orig = K_orig[:, kv_h, :].astype(np.float64)
        v_orig = V_orig[:, kv_h, :].astype(np.float64)
        k_q = K_quant[:, kv_h, :].astype(np.float64)
        v_q = V_quant[:, kv_h, :].astype(np.float64)

        scores_orig = (k_orig @ Q[qh]) * scale
        scores_orig -= np.max(scores_orig)
        w_orig = np.exp(scores_orig)
        w_orig /= np.sum(w_orig)

        scores_q = (k_q @ Q[qh]) * scale
        scores_q -= np.max(scores_q)
        w_q = np.exp(scores_q)
        w_q /= np.sum(w_q)

        out_orig = w_orig @ v_orig
        out_q = w_q @ v_q

        # Per-position weighted error contribution
        # output = sum_i(w_i * v_i), so position i's contribution error is:
        #   w_orig_i * v_orig_i - w_q_i * v_q_i
        for pos in range(seq_len):
            contrib_orig = w_orig[pos] * v_orig[pos]
            contrib_q = w_q[pos] * v_q[pos]
            pos_err = float(np.mean((contrib_orig - contrib_q) ** 2))

            if pos < n_sink:
                sink_errors.append(pos_err)
            else:
                nonsink_errors.append(pos_err)

    return {
        "sink_mean_err": float(np.mean(sink_errors)) if sink_errors else 0.0,
        "nonsink_mean_err": float(np.mean(nonsink_errors)) if nonsink_errors else 0.0,
        "sink_total_err": float(np.sum(sink_errors)) if sink_errors else 0.0,
        "nonsink_total_err": float(np.sum(nonsink_errors)) if nonsink_errors else 0.0,
    }


def generate_realistic_kv(seq_len, kv_heads, head_dim, rng,
                           kurtosis_range=(10, 80), skew_range=(-5, 5)):
    """Generate KV data matching Phase 0a real-model distribution characteristics.

    Phase 0a found for Qwen3-0.6B:
      Keys: kurtosis 50-180, |skewness| 5-13, high inter-channel variance
      Values: kurtosis 0.5-4, |skewness| < 1.5, near-normal with non-zero mean

    Uses a mixture of Gaussian + heavy-tailed (t-distribution) components to
    achieve target kurtosis, rather than purely Gaussian data which
    underestimates quantization error by ~7x (Phase 0 finding).
    """
    K = np.zeros((seq_len, kv_heads, head_dim), dtype=np.float64)
    V = np.zeros((seq_len, kv_heads, head_dim), dtype=np.float64)

    for h in range(kv_heads):
        # Keys: heavy-tailed, high kurtosis, skewed
        k_ch_means = rng.normal(0, 1.0, head_dim)
        k_ch_stds = np.abs(rng.normal(1.0, 0.4, head_dim))
        # Use t-distribution for heavy tails (df=3 gives kurtosis~inf, df=5 gives ~6)
        df = rng.uniform(3, 6)
        k_base = rng.standard_t(df, size=(seq_len, head_dim))
        K[:, h, :] = k_base * k_ch_stds + k_ch_means

        # Add outlier channels (3% of channels with 3x dynamic range)
        n_outlier = max(1, int(0.03 * head_dim))
        outlier_ch = rng.choice(head_dim, n_outlier, replace=False)
        K[:, h, outlier_ch] *= 3.0

        # Values: near-normal, moderate tails, non-zero mean
        v_ch_means = rng.normal(0, 0.3, head_dim)
        v_ch_stds = np.abs(rng.normal(0.5, 0.12, head_dim))
        V[:, h, :] = rng.normal(v_ch_means, v_ch_stds, (seq_len, head_dim))

    return K.astype(np.float64), V.astype(np.float64)


CONFIGS = [
    {
        "name": "K8V8",
        "k_method": "int8",
        "v_method": "int8",
        "desc": "INT8 baseline",
    },
    {
        "name": "K4V4_uniform",
        "k_method": "int4_uniform",
        "v_method": "int4_uniform",
        "desc": "Uniform INT4 keys + values",
    },
    {
        "name": "K4V4_nf4_asym",
        "k_method": "nf4",
        "v_method": "int4_asymmetric",
        "desc": "NF4 keys + asymmetric values (recommended production)",
    },
    {
        "name": "K4V8_nf4",
        "k_method": "nf4",
        "v_method": "int8",
        "desc": "NF4 INT4 keys, INT8 values",
    },
    {
        "name": "K8V4_asym",
        "k_method": "int8",
        "v_method": "int4_asymmetric",
        "desc": "INT8 keys, asymmetric INT4 values (Pareto candidate)",
    },
]


def run_single_length(K_fp, V_fp, Q_fp, kv_heads, head_dim, group_size,
                      n_sink=4):
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = K_fp.shape[0]
    ref_output, ref_weights = attention_reference(Q_fp, K_fp, V_fp, scale)

    results = {}
    for cfg in CONFIGS:
        K_q = quantize_dequantize(K_fp, cfg["k_method"], group_size)
        V_q = quantize_dequantize(V_fp, cfg["v_method"], group_size)

        test_output, test_weights = attention_reference(Q_fp, K_q, V_q, scale)
        metrics = compute_metrics(ref_output, ref_weights, test_output, test_weights)

        # KV reconstruction error
        k_recon_mse = reconstruction_mse(K_fp, cfg["k_method"], group_size)
        v_recon_mse = reconstruction_mse(V_fp, cfg["v_method"], group_size)
        metrics["k_recon_mse"] = k_recon_mse
        metrics["v_recon_mse"] = v_recon_mse

        # Sink token analysis
        sink = sink_token_metrics(Q_fp, K_fp, V_fp, K_q, V_q, scale, n_sink)
        if sink:
            metrics.update(sink)

        results[cfg["name"]] = metrics

    return results


def fit_error_growth(seq_lens, mse_values):
    """Fit MSE ~ a * N^b to determine error growth exponent.

    Returns (exponent_b, coefficient_a, r_squared).
    b ~ 0.0 => constant error (best case: errors cancel perfectly)
    b ~ 0.5 => O(sqrt(N)) growth (random independent errors)
    b ~ 1.0 => O(N) growth (systematic bias)
    b > 1.0 => superlinear (catastrophic)
    """
    valid = [(n, m) for n, m in zip(seq_lens, mse_values) if m > 0 and n > 0]
    if len(valid) < 3:
        return None, None, None

    ns, ms = zip(*valid)
    log_n = np.log(np.array(ns, dtype=np.float64))
    log_m = np.log(np.array(ms, dtype=np.float64))

    # Linear regression: log(MSE) = log(a) + b * log(N)
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    result = np.linalg.lstsq(A, log_m, rcond=None)
    b, log_a = result[0]
    a = np.exp(log_a)

    # R-squared
    ss_res = np.sum((log_m - (b * log_n + log_a)) ** 2)
    ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
    r_sq = 1 - ss_res / (ss_tot + 1e-30)

    return float(b), float(a), float(r_sq)


def load_kv_stats(profile_dir):
    """Load distribution statistics from Phase 0a analysis results."""
    results_path = Path(profile_dir) / "kv_analysis_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def print_growth_table(seq_lens, all_results, configs):
    header_metrics = ["output_mse", "cosine_sim", "attn_weight_l1"]
    for metric in header_metrics:
        print(f"\n{'='*100}")
        print(f"  {metric} vs sequence length")
        print(f"{'='*100}")

        hdr = f"{'seq_len':>8}"
        for cfg in configs:
            hdr += f"  {cfg['name']:>18}"
        print(hdr)
        print("-" * 100)

        for i, sl in enumerate(seq_lens):
            row = f"{sl:>8}"
            for cfg in configs:
                val = all_results[i].get(cfg["name"], {}).get(metric, float("nan"))
                if metric == "cosine_sim":
                    row += f"  {val:>18.10f}"
                elif metric == "attn_weight_l1":
                    row += f"  {val:>18.6f}"
                else:
                    row += f"  {val:>18.2e}"
            print(row)


def print_growth_analysis(seq_lens, all_results, configs):
    print(f"\n{'='*100}")
    print(f"  Error Growth Exponent Analysis: MSE ~ a * N^b")
    print(f"{'='*100}")
    print(f"  b=0.0: constant (errors cancel)   b=0.5: O(sqrt(N)) random")
    print(f"  b=1.0: O(N) systematic bias        b>1.0: superlinear (catastrophic)")
    print(f"{'='*100}")

    metrics_to_fit = ["output_mse", "attn_weight_l1", "k_recon_mse", "v_recon_mse"]
    for metric in metrics_to_fit:
        print(f"\n  {metric}:")
        print(f"  {'Config':<20} {'exponent b':>12} {'coeff a':>14} {'R²':>8}  {'Assessment'}")
        print(f"  {'-'*80}")

        for cfg in configs:
            values = [all_results[i].get(cfg["name"], {}).get(metric, 0)
                      for i in range(len(seq_lens))]
            b, a, r2 = fit_error_growth(seq_lens, values)
            if b is not None:
                if b < 0.1:
                    assessment = "CONSTANT (excellent)"
                elif b < 0.35:
                    assessment = "SUB-SQRT (good)"
                elif b < 0.65:
                    assessment = "~SQRT(N) (expected for random errors)"
                elif b < 0.85:
                    assessment = "SUB-LINEAR (mild concern)"
                elif b < 1.15:
                    assessment = "~LINEAR O(N) (SYSTEMATIC BIAS)"
                else:
                    assessment = "SUPERLINEAR (CATASTROPHIC)"
                print(f"  {cfg['name']:<20} {b:>12.4f} {a:>14.2e} {r2:>8.4f}  {assessment}")
            else:
                print(f"  {cfg['name']:<20} {'N/A':>12} {'N/A':>14} {'N/A':>8}  insufficient data")


def print_sink_analysis(seq_lens, all_results, configs):
    print(f"\n{'='*100}")
    print(f"  Sink Token Analysis (first 4 positions)")
    print(f"{'='*100}")
    print(f"  Sink tokens are never evicted in sliding-window mode.")
    print(f"  If INT4 quantization introduces systematic bias in sink tokens,")
    print(f"  that bias is amplified at every decode step.")
    print(f"{'='*100}")

    # Use the longest sequence length for sink analysis
    longest_idx = len(seq_lens) - 1

    print(f"\n  At seq_len={seq_lens[longest_idx]}:")
    print(f"  {'Config':<20} {'Sink MSE':>14} {'Non-sink MSE':>14} {'Ratio':>8}")
    print(f"  {'-'*60}")

    for cfg in configs:
        r = all_results[longest_idx].get(cfg["name"], {})
        sink_err = r.get("sink_mean_err", float("nan"))
        nonsink_err = r.get("nonsink_mean_err", float("nan"))
        ratio = sink_err / nonsink_err if nonsink_err > 1e-30 else float("nan")
        print(f"  {cfg['name']:<20} {sink_err:>14.2e} {nonsink_err:>14.2e} {ratio:>8.2f}x")

    print()
    print("  Ratio > 2.0 suggests sink tokens accumulate disproportionate error.")
    print("  Consider keeping sink tokens at INT8 even in an INT4 cache.")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1d: Long-Context Error Accumulation Test")
    parser.add_argument("--profile-dir", type=str,
                        default=os.environ.get("CACTUS_KV_PROFILE_DIR"),
                        help="Directory with Phase 0a KV analysis results")
    parser.add_argument("--csv", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-heads", type=int, default=16,
                        help="Number of query heads (default: 16, Qwen3-0.6B)")
    parser.add_argument("--kv-heads", type=int, default=8,
                        help="Number of KV heads (default: 8, Qwen3-0.6B)")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per sequence length for averaging")
    parser.add_argument("--max-seq", type=int, default=1024,
                        help="Maximum sequence length to test")
    parser.add_argument("--step", type=int, default=64,
                        help="Sequence length step size")
    parser.add_argument("--min-seq", type=int, default=32,
                        help="Minimum sequence length")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    seq_lens = list(range(args.min_seq, args.max_seq + 1, args.step))
    if seq_lens[-1] != args.max_seq:
        seq_lens.append(args.max_seq)

    print(f"Phase 1d: Long-Context Error Accumulation Test")
    print(f"  q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
          f"head_dim={args.head_dim}, group_size={args.group_size}")
    print(f"  Sequence lengths: {seq_lens}")
    print(f"  Trials per length: {args.trials}")
    print(f"  Configs: {[c['name'] for c in CONFIGS]}")

    # Load real distribution stats if available (for reporting context)
    kv_stats = None
    if args.profile_dir:
        kv_stats = load_kv_stats(args.profile_dir)
        if kv_stats:
            print(f"\n  Loaded Phase 0a stats: {kv_stats['num_layers']} layers, "
                  f"{kv_stats['num_kv_heads']} kv_heads, "
                  f"{kv_stats['head_dim']} head_dim")

    all_aggregated = []
    csv_rows = []

    for sl_idx, seq_len in enumerate(seq_lens):
        trial_results = []
        for trial in range(args.trials):
            K, V = generate_realistic_kv(
                seq_len, args.kv_heads, args.head_dim, rng)
            Q = rng.normal(0, 0.5, (args.q_heads, args.head_dim)).astype(np.float64)

            results = run_single_length(
                K, V, Q, args.kv_heads, args.head_dim, args.group_size)
            trial_results.append(results)

            for cfg_name, metrics in results.items():
                csv_rows.append({
                    "seq_len": seq_len,
                    "trial": trial,
                    "config": cfg_name,
                    **metrics,
                })

        # Aggregate across trials
        agg = {}
        for cfg in CONFIGS:
            name = cfg["name"]
            cfg_trials = [t[name] for t in trial_results if name in t]
            if not cfg_trials:
                continue
            agg[name] = {}
            for metric in cfg_trials[0]:
                vals = [t[metric] for t in cfg_trials]
                agg[name][metric] = float(np.mean(vals))

        all_aggregated.append(agg)

        # Progress
        pct = (sl_idx + 1) / len(seq_lens) * 100
        k4v4_mse = agg.get("K4V4_nf4_asym", {}).get("output_mse", 0)
        k8v8_mse = agg.get("K8V8", {}).get("output_mse", 0)
        print(f"  [{pct:5.1f}%] seq_len={seq_len:>5}: "
              f"K4V4_nf4_asym MSE={k4v4_mse:.2e}, "
              f"K8V8 MSE={k8v8_mse:.2e}")

    # Print results
    print_growth_table(seq_lens, all_aggregated, CONFIGS)
    print_growth_analysis(seq_lens, all_aggregated, CONFIGS)
    print_sink_analysis(seq_lens, all_aggregated, CONFIGS)

    # Top-level summary
    print(f"\n{'='*100}")
    print(f"  SUMMARY")
    print(f"{'='*100}")

    for cfg in CONFIGS:
        name = cfg["name"]
        mse_vals = [all_aggregated[i].get(name, {}).get("output_mse", 0)
                    for i in range(len(seq_lens))]
        b, a, r2 = fit_error_growth(seq_lens, mse_vals)

        cos_at_max = all_aggregated[-1].get(name, {}).get("cosine_sim", 0)
        mse_at_max = all_aggregated[-1].get(name, {}).get("output_mse", 0)
        l1_at_max = all_aggregated[-1].get(name, {}).get("attn_weight_l1", 0)

        print(f"\n  {name} ({cfg['desc']}):")
        print(f"    At seq_len={seq_lens[-1]}: MSE={mse_at_max:.2e}, "
              f"cosine={cos_at_max:.8f}, attn_l1={l1_at_max:.6f}")
        if b is not None:
            growth_type = ("CONSTANT" if b < 0.1 else
                          "SUB-SQRT" if b < 0.35 else
                          "~SQRT(N)" if b < 0.65 else
                          "SUB-LINEAR" if b < 0.85 else
                          "~LINEAR" if b < 1.15 else
                          "SUPERLINEAR")
            safe = b < 0.85
            print(f"    Error growth: MSE ~ {a:.2e} * N^{b:.3f} "
                  f"(R²={r2:.4f}) => {growth_type} "
                  f"{'[SAFE]' if safe else '[CONCERN]'}")

    # Write CSV
    if args.csv and csv_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nRaw results written to {args.csv}")

    print()


if __name__ == "__main__":
    main()
