#!/usr/bin/env python3
"""
Phase 0c: Keys vs Values Sensitivity Analysis

Quantizes K to INT4 while keeping V at INT8 (and vice versa) to isolate which
is more sensitive to precision reduction.

Four configurations:
  1. K=INT8, V=INT8   (baseline — current production)
  2. K=INT4, V=INT8   (isolate K sensitivity)
  3. K=INT8, V=INT4   (isolate V sensitivity)
  4. K=INT4, V=INT4   (full INT4)

Data sources:
  - If CACTUS_KV_PROFILE_DIR is set and contains .bin files, loads captured
    FP16 KV data from Phase 0a profiling.
  - Otherwise, generates synthetic data with realistic transformer-like
    distributions (per-channel means/stds, occasional outlier channels).

Metrics reported per configuration:
  - Attention weight L1 divergence (how much "what the model attends to" shifts)
  - Output MSE (mean squared error vs FP32 reference)
  - Output cosine similarity
  - Max absolute output error
  - Attention score MSE (softmax(Q@K^T/sqrt(d)) error)

Usage:
  python tests/exp_0c_kv_sensitivity.py [--profile-dir DIR] [--csv output.csv]
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


def quantize_int8(x: np.ndarray, group_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    shape = x.shape
    flat = x.reshape(-1, group_size)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = max_abs / 127.0
    scales = np.maximum(scales, 1e-10)
    quantized = np.clip(np.round(flat / scales), -128, 127).astype(np.int8)
    return quantized.reshape(shape), scales.reshape(-1)


def dequantize_int8(q: np.ndarray, scales: np.ndarray, group_size: int = 32) -> np.ndarray:
    shape = q.shape
    flat = q.reshape(-1, group_size).astype(np.float32)
    s = scales.reshape(-1, 1)
    return (flat * s).reshape(shape)


def quantize_int4(x: np.ndarray, group_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    shape = x.shape
    flat = x.reshape(-1, group_size)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = max_abs / 7.0
    scales = np.maximum(scales, 1e-10)
    quantized = np.clip(np.round(flat / scales), -8, 7).astype(np.int8)
    return quantized.reshape(shape), scales.reshape(-1)


def dequantize_int4(q: np.ndarray, scales: np.ndarray, group_size: int = 32) -> np.ndarray:
    shape = q.shape
    flat = q.reshape(-1, group_size).astype(np.float32)
    s = scales.reshape(-1, 1)
    return (flat * s).reshape(shape)


def quantize_dequantize(x: np.ndarray, bits: int, group_size: int = 32) -> np.ndarray:
    if bits == 8:
        q, s = quantize_int8(x, group_size)
        return dequantize_int8(q, s, group_size)
    elif bits == 4:
        q, s = quantize_int4(x, group_size)
        return dequantize_int4(q, s, group_size)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def attention_reference(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float):
    """Compute scaled dot-product attention in FP64 for maximum reference accuracy.

    Args:
        Q: [q_heads, head_dim]
        K: [seq_len, kv_heads, head_dim]
        V: [seq_len, kv_heads, head_dim]
        scale: 1/sqrt(head_dim)

    Returns:
        output: [q_heads, head_dim]
        weights: [q_heads, seq_len] (attention weights after softmax)
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

    mse = np.mean((ref_o - test_o) ** 2)
    max_err = np.max(np.abs(ref_o - test_o))

    dot = np.dot(ref_o, test_o)
    norm_ref = np.linalg.norm(ref_o)
    norm_test = np.linalg.norm(test_o)
    cosine_sim = dot / (norm_ref * norm_test + 1e-30)

    weight_l1 = np.mean(np.sum(np.abs(ref_weights - test_weights), axis=1))

    attn_mse = np.mean((ref_weights - test_weights) ** 2)

    return {
        "output_mse": float(mse),
        "output_max_err": float(max_err),
        "output_cosine_sim": float(cosine_sim),
        "attn_weight_l1_div": float(weight_l1),
        "attn_score_mse": float(attn_mse),
    }


def generate_synthetic_kv(seq_len: int, kv_heads: int, head_dim: int,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic KV data with realistic transformer-like distributions.

    Characteristics modeled:
    - Per-channel mean offsets (channels have different biases)
    - Per-channel variance differences (some channels are "outlier channels")
    - Moderate kurtosis (heavier tails than Gaussian)
    - ~2-5% of channels have 3x larger dynamic range (outlier channels)
    """
    K = np.zeros((seq_len, kv_heads, head_dim), dtype=np.float32)
    V = np.zeros((seq_len, kv_heads, head_dim), dtype=np.float32)

    for h in range(kv_heads):
        k_ch_means = rng.normal(0, 0.3, head_dim)
        k_ch_stds = np.abs(rng.normal(0.5, 0.15, head_dim))
        v_ch_means = rng.normal(0, 0.2, head_dim)
        v_ch_stds = np.abs(rng.normal(0.4, 0.1, head_dim))

        n_outlier_k = max(1, int(0.03 * head_dim))
        n_outlier_v = max(1, int(0.03 * head_dim))
        outlier_k = rng.choice(head_dim, n_outlier_k, replace=False)
        outlier_v = rng.choice(head_dim, n_outlier_v, replace=False)
        k_ch_stds[outlier_k] *= 3.0
        v_ch_stds[outlier_v] *= 2.5

        K[:, h, :] = rng.normal(k_ch_means, k_ch_stds, (seq_len, head_dim))
        V[:, h, :] = rng.normal(v_ch_means, v_ch_stds, (seq_len, head_dim))

    return K.astype(np.float32), V.astype(np.float32)


def load_profile_data(profile_dir: str):
    """Load KV profile data dumped by CACTUS_KV_PROFILE_DIR.

    Expected file format:
      layer{L}_pos{P}_keys.bin   - raw FP16 data, shape [kv_heads * head_dim]
      layer{L}_pos{P}_values.bin
      layer{L}_pos{P}_meta.json  - {"layer", "position", "seq_len", "num_kv_heads", "head_dim"}
    """
    profile_path = Path(profile_dir)
    if not profile_path.exists():
        return None

    meta_files = sorted(profile_path.glob("layer*_meta.json"))
    if not meta_files:
        return None

    with open(meta_files[0]) as f:
        meta = json.load(f)

    kv_heads = meta["num_kv_heads"]
    head_dim = meta["head_dim"]

    layers = {}
    for mf in meta_files:
        with open(mf) as f:
            m = json.load(f)
        layer = m["layer"]
        if layer not in layers:
            layers[layer] = {"positions": []}
        layers[layer]["positions"].append(m["position"])

    result = {}
    for layer_idx, info in sorted(layers.items()):
        positions = sorted(set(info["positions"]))
        seq_len = len(positions)

        K = np.zeros((seq_len, kv_heads, head_dim), dtype=np.float32)
        V = np.zeros((seq_len, kv_heads, head_dim), dtype=np.float32)

        for i, pos in enumerate(positions):
            k_path = profile_path / f"layer{layer_idx}_pos{pos}_keys.bin"
            v_path = profile_path / f"layer{layer_idx}_pos{pos}_values.bin"

            if k_path.exists() and v_path.exists():
                k_raw = np.fromfile(str(k_path), dtype=np.float16)
                v_raw = np.fromfile(str(v_path), dtype=np.float16)
                K[i] = k_raw.reshape(kv_heads, head_dim).astype(np.float32)
                V[i] = v_raw.reshape(kv_heads, head_dim).astype(np.float32)

        result[layer_idx] = {"K": K, "V": V, "kv_heads": kv_heads, "head_dim": head_dim}

    return result


CONFIGS = [
    {"name": "K8V8", "k_bits": 8, "v_bits": 8, "desc": "INT8 keys, INT8 values (baseline)"},
    {"name": "K4V8", "k_bits": 4, "v_bits": 8, "desc": "INT4 keys, INT8 values"},
    {"name": "K8V4", "k_bits": 8, "v_bits": 4, "desc": "INT8 keys, INT4 values"},
    {"name": "K4V4", "k_bits": 4, "v_bits": 4, "desc": "INT4 keys, INT4 values"},
]


def run_experiment(K_fp32, V_fp32, Q_fp32, kv_heads, head_dim, group_size=32):
    """Run all 4 configurations on a single K/V/Q set.

    Args:
        K_fp32: [seq_len, kv_heads, head_dim]
        V_fp32: [seq_len, kv_heads, head_dim]
        Q_fp32: [q_heads, head_dim]
    """
    scale = 1.0 / np.sqrt(head_dim)

    ref_output, ref_weights = attention_reference(Q_fp32, K_fp32, V_fp32, scale)

    results = {}
    for cfg in CONFIGS:
        K_q = quantize_dequantize(K_fp32, cfg["k_bits"], group_size)
        V_q = quantize_dequantize(V_fp32, cfg["v_bits"], group_size)

        test_output, test_weights = attention_reference(Q_fp32, K_q, V_q, scale)
        metrics = compute_metrics(ref_output, ref_weights, test_output, test_weights)
        results[cfg["name"]] = metrics

    return results


def aggregate_results(all_results):
    """Aggregate metrics across multiple trials (layers, prompts, etc.)."""
    agg = {}
    for config_name in CONFIGS:
        name = config_name["name"]
        config_metrics = [r[name] for r in all_results if name in r]
        if not config_metrics:
            continue

        agg[name] = {}
        for metric in config_metrics[0]:
            values = [m[metric] for m in config_metrics]
            agg[name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

    return agg


def print_results(agg, group_size):
    print(f"\n{'='*90}")
    print(f"  Phase 0c: Keys vs Values Sensitivity Analysis  (group_size={group_size})")
    print(f"{'='*90}")

    header = f"{'Config':<8} {'Attn Wt L1':>12} {'Attn MSE':>12} {'Out MSE':>14} {'Out MaxErr':>12} {'Cosine Sim':>12}"
    print(f"\n{header}")
    print("-" * 90)

    for cfg in CONFIGS:
        name = cfg["name"]
        if name not in agg:
            continue
        m = agg[name]
        print(f"{name:<8} "
              f"{m['attn_weight_l1_div']['mean']:>12.6f} "
              f"{m['attn_score_mse']['mean']:>12.2e} "
              f"{m['output_mse']['mean']:>14.2e} "
              f"{m['output_max_err']['mean']:>12.6f} "
              f"{m['output_cosine_sim']['mean']:>12.8f}")

    print()

    k8v8 = agg.get("K8V8", {})
    k4v8 = agg.get("K4V8", {})
    k8v4 = agg.get("K8V4", {})
    k4v4 = agg.get("K4V4", {})

    if k4v8 and k8v4:
        print("Sensitivity comparison (mean values):")
        print(f"  K sensitivity (K4V8 attn_wt_l1): {k4v8['attn_weight_l1_div']['mean']:.6f}")
        print(f"  V sensitivity (K8V4 attn_wt_l1): {k8v4['attn_weight_l1_div']['mean']:.6f}")

        k_sens = k4v8["attn_weight_l1_div"]["mean"]
        v_sens = k8v4["attn_weight_l1_div"]["mean"]

        if k_sens > 1e-10 and v_sens > 1e-10:
            ratio = k_sens / v_sens
            print(f"  K/V sensitivity ratio:           {ratio:.2f}x")
            if ratio > 1.5:
                print(f"  --> Keys are {ratio:.1f}x MORE sensitive than values")
                print(f"      Recommendation: Consider K8V4 mixed-precision cache")
            elif ratio < 0.67:
                print(f"  --> Values are {1/ratio:.1f}x MORE sensitive than keys")
                print(f"      Recommendation: Consider K4V8 mixed-precision cache")
            else:
                print(f"  --> Keys and values have similar sensitivity")
                print(f"      Recommendation: Uniform K4V4 is appropriate")

        print()
        print(f"  K4V8 output MSE: {k4v8['output_mse']['mean']:.2e}")
        print(f"  K8V4 output MSE: {k8v4['output_mse']['mean']:.2e}")
        print(f"  K4V4 output MSE: {k4v4['output_mse']['mean']:.2e}")

    print()

    print("Per-config statistics (mean +/- std):")
    print(f"{'Config':<8} {'Attn Wt L1':>24} {'Output MSE':>24} {'Cosine Sim':>24}")
    print("-" * 90)
    for cfg in CONFIGS:
        name = cfg["name"]
        if name not in agg:
            continue
        m = agg[name]
        print(f"{name:<8} "
              f"{m['attn_weight_l1_div']['mean']:.6f} +/- {m['attn_weight_l1_div']['std']:.6f}   "
              f"{m['output_mse']['mean']:.2e} +/- {m['output_mse']['std']:.2e}   "
              f"{m['output_cosine_sim']['mean']:.8f} +/- {m['output_cosine_sim']['std']:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0c: Keys vs Values Sensitivity")
    parser.add_argument("--profile-dir", type=str, default=os.environ.get("CACTUS_KV_PROFILE_DIR"),
                        help="Directory with Phase 0a KV profile dumps")
    parser.add_argument("--csv", type=str, default=None,
                        help="Output CSV path for raw results")
    parser.add_argument("--group-size", type=int, default=32,
                        help="Quantization group size (default: 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-lens", type=str, default="32,128,512",
                        help="Comma-separated sequence lengths for synthetic experiments")
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of random trials per sequence length")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_results = []
    csv_rows = []

    profile_data = None
    if args.profile_dir:
        print(f"Loading KV profiles from {args.profile_dir}...")
        profile_data = load_profile_data(args.profile_dir)

    if profile_data:
        print(f"Loaded {len(profile_data)} layers of profile data")
        for layer_idx, data in profile_data.items():
            K, V = data["K"], data["V"]
            kv_heads = data["kv_heads"]
            head_dim = data["head_dim"]
            q_heads = args.q_heads

            Q = rng.normal(0, 0.5, (q_heads, head_dim)).astype(np.float32)

            results = run_experiment(K, V, Q, kv_heads, head_dim, args.group_size)
            all_results.append(results)

            for cfg_name, metrics in results.items():
                csv_rows.append({
                    "source": "profile",
                    "layer": layer_idx,
                    "seq_len": K.shape[0],
                    "config": cfg_name,
                    **metrics,
                })

            print(f"  Layer {layer_idx}: seq_len={K.shape[0]}, "
                  f"kv_heads={kv_heads}, head_dim={head_dim}")
    else:
        print("No profile data found, using synthetic KV distributions")
        print(f"  q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
              f"head_dim={args.head_dim}, group_size={args.group_size}")

        seq_lens = [int(s) for s in args.seq_lens.split(",")]
        for seq_len in seq_lens:
            print(f"\n  seq_len={seq_len}:")
            for trial in range(args.trials):
                K, V = generate_synthetic_kv(seq_len, args.kv_heads, args.head_dim, rng)
                Q = rng.normal(0, 0.5, (args.q_heads, args.head_dim)).astype(np.float32)

                results = run_experiment(K, V, Q, args.kv_heads, args.head_dim, args.group_size)
                all_results.append(results)

                for cfg_name, metrics in results.items():
                    csv_rows.append({
                        "source": "synthetic",
                        "layer": -1,
                        "seq_len": seq_len,
                        "trial": trial,
                        "config": cfg_name,
                        **metrics,
                    })

            seq_results = all_results[-args.trials:]
            for cfg in CONFIGS:
                name = cfg["name"]
                l1s = [r[name]["attn_weight_l1_div"] for r in seq_results]
                print(f"    {name}: attn_wt_l1={np.mean(l1s):.6f} +/- {np.std(l1s):.6f}")

    agg = aggregate_results(all_results)
    print_results(agg, args.group_size)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            if csv_rows:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
        print(f"\nRaw results written to {args.csv}")

    for group_size in [8, 16, 64, 128]:
        if group_size == args.group_size:
            continue
        print(f"\n--- Re-running with group_size={group_size} ---")
        gs_results = []

        if profile_data:
            for layer_idx, data in profile_data.items():
                K, V = data["K"], data["V"]
                Q = rng.normal(0, 0.5, (args.q_heads, data["head_dim"])).astype(np.float32)
                gs_results.append(run_experiment(K, V, Q, data["kv_heads"], data["head_dim"], group_size))
        else:
            seq_lens = [int(s) for s in args.seq_lens.split(",")]
            for seq_len in seq_lens:
                for trial in range(min(args.trials, 5)):
                    K, V = generate_synthetic_kv(seq_len, args.kv_heads, args.head_dim, rng)
                    Q = rng.normal(0, 0.5, (args.q_heads, args.head_dim)).astype(np.float32)
                    gs_results.append(run_experiment(K, V, Q, args.kv_heads, args.head_dim, group_size))

        gs_agg = aggregate_results(gs_results)
        print_results(gs_agg, group_size)


if __name__ == "__main__":
    main()
