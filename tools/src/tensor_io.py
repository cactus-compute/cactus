import numpy as np
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import torch
except ImportError:
    torch = None


def save_tensor_with_header(tensor, output_path, precision='FP32', transpose=False, stats_tracker=None, args=None, model_type=None):
    """Save a tensor to binary format with header metadata and optional quantization."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)

    original_data = data.copy()

    if model_type == 'gemma' and 'norm' in str(output_path):
        data = data + 1.0
        original_data = data.copy()

    mean_val = np.mean(original_data)
    std_val = np.std(original_data)
    min_val = np.min(original_data)
    max_val = np.max(original_data)


    if precision == 'INT8':
        filename = output_path.name
        if any(x in filename for x in ['norm', 'bias', 'vision']) or (model_type == 'bert' and 'embedding' in filename):
            precision = 'FP16'

    if precision == 'INT8':
        qmin, qmax = -128, 127
        standard_scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0

        standard_zero_point = qmax - max_val / standard_scale
        standard_zero_point_clipped = np.clip(np.round(standard_zero_point), qmin, qmax)
        test_quantized = np.clip(np.round(original_data / standard_scale + standard_zero_point_clipped), qmin, qmax)
        test_saturation = np.sum(np.abs(test_quantized) >= 127) / original_data.size

        saturation_threshold = getattr(args, 'saturation_threshold', 0.01) if args else 0.01
        if test_saturation > saturation_threshold:
            outlier_percentile = getattr(args, 'outlier_percentile', 0.01) if args else 0.01
            lower_percentile = np.percentile(original_data, outlier_percentile)
            upper_percentile = np.percentile(original_data, 100 - outlier_percentile)

            mean_val = np.mean(original_data)
            std_val = np.std(original_data)
            sigma_multiplier = getattr(args, 'sigma_multiplier', 3.5) if args else 3.5
            three_sigma_min = mean_val - sigma_multiplier * std_val
            three_sigma_max = mean_val + sigma_multiplier * std_val

            clipped_min = max(min_val, min(lower_percentile, three_sigma_min))
            clipped_max = min(max_val, max(upper_percentile, three_sigma_max))

            range_threshold = getattr(args, 'range_threshold', 0.5) if args else 0.5
            if (clipped_max - clipped_min) < range_threshold * (max_val - min_val):
                clipped_min = min_val
                clipped_max = max_val
        else:
            clipped_min = min_val
            clipped_max = max_val

        abs_max = max(abs(clipped_min), abs(clipped_max))
        scale = abs_max / 127.0 if abs_max != 0 else 1.0

        quantized_data = np.clip(np.round(original_data / scale), qmin, qmax).astype(np.int8)

        dequantized_data = quantized_data.astype(np.float32) * scale
        mse_error = np.mean((original_data - dequantized_data) ** 2)
        snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')

        original_flat = original_data.flatten()
        dequantized_flat = dequantized_data.flatten()
        cos_sim = np.dot(original_flat, dequantized_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(dequantized_flat))
        saturated_values = np.sum(np.abs(quantized_data) == 127)
        saturation_percent = (saturated_values / quantized_data.size) * 100
        data = quantized_data

        if stats_tracker:
            stats_tracker['quantized_tensors'] += 1
            stats_tracker['quantized_parameters'] += original_data.size
            stats_tracker['mse_values'].append(mse_error)
            stats_tracker['snr_values'].append(snr_db)
            stats_tracker['cos_sim_values'].append(cos_sim)
            saturation_warning_threshold = getattr(args, 'saturation_warning_threshold', 0.1) if args else 0.1
            if saturation_percent > saturation_warning_threshold:
                stats_tracker['saturation_warnings'] += 1
    elif precision == 'FP16':
        data = data.astype(np.float16)
        scale = 1.0
    else:
        data = data.astype(np.float32)
        scale = 1.0

    if stats_tracker:
        stats_tracker['total_tensors'] += 1
        stats_tracker['total_parameters'] += original_data.size

    shape = list(data.shape)
    if transpose and len(shape) == 2:
        data = data.T
        shape = [shape[1], shape[0]]

    data = data.flatten()

    with open(output_path, 'wb') as f:
        ndim = len(shape)
        f.write(struct.pack('<I', ndim))

        for dim in shape:
            f.write(struct.pack('<Q', dim))

        if precision == 'INT8':
            prec_val = 0
        elif precision == 'FP16':
            prec_val = 1
        else:
            prec_val = 2
        f.write(struct.pack('<I', prec_val))

        if precision == 'INT8':
            element_size = 1
        elif precision == 'FP16':
            element_size = 2
        else:
            element_size = 4
        byte_size = data.size * element_size
        f.write(struct.pack('<Q', byte_size))

        if precision == 'INT8':
            f.write(struct.pack('<f', scale))

        f.write(data.tobytes())

    if precision == 'INT8':
        scale_path = output_path.with_suffix('.scale')
        with open(scale_path, 'w') as f:
            f.write(f"{scale:.10f}\n")


def format_config_value(value):
    """Format a config value for writing to config.txt."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)


def create_quantization_stats():
    """Create an empty stats tracker dictionary for quantization metrics."""
    return {
        'total_tensors': 0,
        'quantized_tensors': 0,
        'total_parameters': 0,
        'quantized_parameters': 0,
        'mse_values': [],
        'snr_values': [],
        'cos_sim_values': [],
        'saturation_warnings': 0
    }


def print_quantization_summary(quantization_stats, args=None):
    """Print a summary of quantization statistics."""
    if quantization_stats['quantized_tensors'] > 0:
        mse_values = np.array(quantization_stats['mse_values'])
        snr_values = np.array(quantization_stats['snr_values'])
        cos_sim_values = np.array(quantization_stats['cos_sim_values'])

        print("\nQuantization Summary:")
        print(f"MSE - Mean: {np.mean(mse_values):.2e}, Max: {np.max(mse_values):.2e}, Median: {np.median(mse_values):.2e}, Min: {np.min(mse_values):.2e}")
        print(f"SNR - Mean: {np.mean(snr_values):.1f}dB, Max: {np.max(snr_values):.1f}dB, Median: {np.median(snr_values):.1f}dB, Min: {np.min(snr_values):.1f}dB")
        print(f"CosSim - Mean: {np.mean(cos_sim_values):.6f}, Max: {np.mean(cos_sim_values):.6f}, Median: {np.median(cos_sim_values):.6f}, Min: {np.min(cos_sim_values):.6f}")
        fp16_tensors = quantization_stats['total_tensors'] - quantization_stats['quantized_tensors']
        low_snr_fallbacks = quantization_stats.get('low_snr_fallbacks', 0)
        snr_threshold = getattr(args, 'snr_threshold', 30.0) if args else 30.0
        print(f"Processed {quantization_stats['quantized_tensors']} INT8 tensors, {fp16_tensors} FP16 tensors ({low_snr_fallbacks} SNR<{snr_threshold}dB fallbacks)")
