import numpy as np
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List
from .weight_patterns import EMBED_NAMES

try:
    import torch
except ImportError:
    torch = None


GROUP_SIZE = 32
INTERLEAVE_BLOCK = 4

CACTUS_MAGIC = b'CACT'
CACTUS_ALIGNMENT = 32

FLAG_HAS_SCALES = 1 << 0
FLAG_PAGE_ALIGNED = 1 << 1
FLAG_TRANSPOSED = 1 << 2
FLAG_INTERLEAVED = 1 << 3 


def align_offset(offset: int, alignment: int) -> int:
    """Round up offset to next alignment boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def compute_padding(current_offset: int, alignment: int) -> bytes:
    """Compute padding bytes needed to reach alignment boundary."""
    aligned = align_offset(current_offset, alignment)
    padding_size = aligned - current_offset
    return b'\x00' * padding_size


def interleave_weights(data: np.ndarray, block_size: int = INTERLEAVE_BLOCK) -> tuple[np.ndarray, int]:
    """Interleave rows for SIMD-friendly GEMM access using vdotq_laneq_s32.

    Input:  data[N, K] - row-major weights
    Output: data_interleaved[N_padded/block_size, K/4, block_size, 4] flattened
            original_N - the original N before padding

    Memory layout after interleaving (GGML-style):
    For each block of 4 rows and each group of 4 K positions:
        [row0_k0..k3, row1_k0..k3, row2_k0..k3, row3_k0..k3, row0_k4..k7, ...]

    This enables vdotq_laneq_s32 to broadcast activation lanes efficiently:
    - Load 16 bytes of activation: a[k:k+16] (4 groups of 4)
    - Load 16 bytes of weights: 4 columns x 4 K values
    - Use lane broadcast to compute 4 output columns simultaneously
    """
    N, K = data.shape
    original_N = N

    if N % block_size != 0:
        pad_n = block_size - (N % block_size)
        data = np.pad(data, ((0, pad_n), (0, 0)), mode='constant', constant_values=0)
        N = data.shape[0]

    if K % 4 != 0:
        pad_k = 4 - (K % 4)
        data = np.pad(data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
        K = data.shape[1]

    data = data.reshape(N // block_size, block_size, K // 4, 4)
    data = data.transpose(0, 2, 1, 3)
    return data.reshape(-1), original_N


def interleave_scales(scales: np.ndarray, block_size: int = INTERLEAVE_BLOCK) -> tuple[np.ndarray, int]:
    """Interleave scales to match interleaved weight layout.

    Input:  scales[N, num_groups]
    Output: scales_interleaved[N_padded/block_size, num_groups, block_size] flattened
            original_N
    """
    N, num_groups = scales.shape
    original_N = N

    if N % block_size != 0:
        pad_n = block_size - (N % block_size)
        scales = np.pad(scales, ((0, pad_n), (0, 0)), mode='constant', constant_values=1e-10)
        N = scales.shape[0]

    scales = scales.reshape(N // block_size, block_size, num_groups)
    scales = scales.transpose(0, 2, 1)
    return scales.reshape(-1), original_N


def pack_int4_pairs(data: np.ndarray) -> np.ndarray:
    """Pack INT4 values (stored as int8) into bytes using planar layout.

    Input: array of int8 values in range [-8, 7], length must be a multiple of 32
    Output: array of uint8 with half the length

    Packing format (planar, groups of 32):
      For each group of 32 values, the first 16 are stored in the low nibbles
      and the next 16 in the high nibbles of 16 consecutive bytes.
      Nibbles are stored in two's complement form (value & 0x0F).
    """
    assert len(data) % 32 == 0, "Data length must be a multiple of 32 for INT4 planar packing"

    groups = data.reshape(-1, 32)
    low = (groups[:, :16].astype(np.int8).view(np.uint8) & 0x0F).astype(np.uint8)
    high = ((groups[:, 16:].astype(np.int8).view(np.uint8) & 0x0F).astype(np.uint8)) << 4

    return (low | high).astype(np.uint8).reshape(-1)


def save_tensor_with_header(tensor, output_path, precision='INT8', transpose=False, stats_tracker=None, args=None, model_type=None):
    """Save a tensor to binary format with header metadata and group-wise quantization.

    For 2D tensors with INT8/INT4 precision:
    - INT4 is packed (2 values per byte) for 50% storage savings, unpacked to INT8 at load time
    - Weights are interleaved in blocks of 4 rows for SIMD efficiency
    - Layout: [N/4, K/4, 4, 4] enables vdotq_laneq_s32 for efficient GEMV/GEMM

    Args:
        tensor: The tensor to save (PyTorch or NumPy)
        output_path: Path to save the tensor
        precision: Quantization precision ('INT4', 'INT8', 'FP16')
        transpose: Whether to transpose 2D tensors
        stats_tracker: Optional dict to track quantization statistics
        args: Optional args object with additional settings
        model_type: Model type string (e.g., 'gemma', 'llama')
    """
    if torch is not None and isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        data = t.numpy()
    else:
        data = np.array(tensor)

    original_data = data.copy()

    if model_type == 'gemma' and 'norm' in str(output_path):
        data = data + 1.0
        original_data = data.copy()

    if precision in ('INT8', 'INT4'):
        filename = output_path.name
        if any(x in filename for x in ['norm', 'bias', 'vision', 'position_embeddings', 'embed_positions']):
            precision = 'FP16'
        elif precision == 'INT4' and any(x in filename for x in EMBED_NAMES):
            precision = 'INT8'

    shape = list(data.shape)
    if transpose and len(shape) == 2:
        data = data.T
        original_data = original_data.T
        shape = [shape[1], shape[0]]

    if precision == 'INT8' and len(shape) == 2:
        N, K = shape
        original_N = N

        if K % GROUP_SIZE != 0:
            pad_k = GROUP_SIZE - (K % GROUP_SIZE)
            data = np.pad(data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
            original_data = np.pad(original_data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
            K = data.shape[1]

        num_groups = K // GROUP_SIZE

        data_grouped = data.reshape(N, num_groups, GROUP_SIZE)
        group_abs_max = np.max(np.abs(data_grouped), axis=2)
        scales = (group_abs_max / 127.0).astype(np.float32)
        scales = np.maximum(scales, 1e-10)

        quantized = np.clip(
            np.round(data_grouped / scales[:, :, np.newaxis]),
            -128, 127
        ).astype(np.int8)
        quantized_2d = quantized.reshape(N, K)

        dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(N, K)
        mse_error = np.mean((original_data[:original_N, :] - dequantized[:original_N, :]) ** 2)
        snr_db = 10 * np.log10(np.var(original_data[:original_N, :]) / mse_error) if mse_error > 0 else float('inf')
        original_flat = original_data[:original_N, :].flatten()
        dequant_flat = dequantized[:original_N, :].flatten()
        cos_sim = np.dot(original_flat, dequant_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(dequant_flat) + 1e-10)

        quantized_interleaved, _ = interleave_weights(quantized_2d, INTERLEAVE_BLOCK)
        scales_interleaved, _ = interleave_scales(scales, INTERLEAVE_BLOCK)
        scales_fp16 = scales_interleaved.astype(np.float16)

        N_padded = ((N + INTERLEAVE_BLOCK - 1) // INTERLEAVE_BLOCK) * INTERLEAVE_BLOCK

        if stats_tracker:
            stats_tracker['int8_tensors'] += 1
            stats_tracker['quantized_parameters'] += original_N * K
            stats_tracker['mse_values'].append(mse_error)
            stats_tracker['snr_values'].append(snr_db)
            stats_tracker['cos_sim_values'].append(cos_sim)
            stats_tracker['total_tensors'] += 1
            stats_tracker['total_parameters'] += original_N * K

        with open(output_path, 'wb') as f:
            ndim = 2
            data_bytes = quantized_interleaved.size
            scales_bytes = scales_fp16.size * 2
            flags = FLAG_HAS_SCALES | FLAG_INTERLEAVED
            if transpose:
                flags |= FLAG_TRANSPOSED

            f.write(CACTUS_MAGIC)                          # 4 bytes
            f.write(struct.pack('<I', flags))              # 4 bytes
            f.write(struct.pack('<I', CACTUS_ALIGNMENT))   # 4 bytes
            f.write(struct.pack('<I', ndim))               # 4 bytes

            f.write(struct.pack('<Q', N_padded))           # 8 bytes - dim 0 (padded)
            f.write(struct.pack('<Q', K))                  # 8 bytes - dim 1
            f.write(struct.pack('<Q', 0))                  # 8 bytes - dim 2 (unused)
            f.write(struct.pack('<Q', 0))                  # 8 bytes - dim 3 (unused)

            f.write(struct.pack('<I', 0))                  # precision: 0 = INT8 (4 bytes)
            f.write(struct.pack('<Q', data_bytes))         # 8 bytes
            f.write(struct.pack('<Q', scales_bytes))       # 8 bytes
            f.write(struct.pack('<I', GROUP_SIZE))         # 4 bytes
            f.write(struct.pack('<I', num_groups))         # 4 bytes
            f.write(struct.pack('<Q', original_N))         # 8 bytes - original N before padding
            # Header total: 84 bytes

            header_size = 84
            f.write(compute_padding(header_size, CACTUS_ALIGNMENT))

            f.write(scales_fp16.tobytes())
            scales_end = align_offset(header_size, CACTUS_ALIGNMENT) + scales_bytes
            f.write(compute_padding(scales_end, CACTUS_ALIGNMENT))

            f.write(quantized_interleaved.tobytes())

        return

    if precision == 'INT4' and len(shape) == 2:
        N, K = shape
        original_N = N

        if K % GROUP_SIZE != 0:
            pad_k = GROUP_SIZE - (K % GROUP_SIZE)
            data = np.pad(data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
            original_data = np.pad(original_data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
            K = data.shape[1]

        num_groups = K // GROUP_SIZE
        data_grouped = data.reshape(N, num_groups, GROUP_SIZE)

        group_abs_max = np.max(np.abs(data_grouped), axis=2)
        scales = (group_abs_max / 7.0).astype(np.float32)
        scales = np.maximum(scales, 1e-10)

        quantized = np.clip(
            np.round(data_grouped / scales[:, :, np.newaxis]),
            -8, 7
        ).astype(np.int8)
        quantized_2d = quantized.reshape(N, K)

        dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(N, K)
        mse_error = np.mean((original_data[:original_N, :] - dequantized[:original_N, :]) ** 2)
        snr_db = 10 * np.log10(np.var(original_data[:original_N, :]) / mse_error) if mse_error > 0 else float('inf')
        original_flat = original_data[:original_N, :].flatten()
        dequant_flat = dequantized[:original_N, :].flatten()
        cos_sim = np.dot(original_flat, dequant_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(dequant_flat) + 1e-10)

        quantized_interleaved, _ = interleave_weights(quantized_2d, INTERLEAVE_BLOCK)
        scales_interleaved, _ = interleave_scales(scales, INTERLEAVE_BLOCK)
        scales_fp16 = scales_interleaved.astype(np.float16)

        packed_data = pack_int4_pairs(quantized_interleaved)

        N_padded = ((N + INTERLEAVE_BLOCK - 1) // INTERLEAVE_BLOCK) * INTERLEAVE_BLOCK

        if stats_tracker:
            stats_tracker['int4_tensors'] += 1
            stats_tracker['quantized_parameters'] += original_N * K
            stats_tracker['mse_values'].append(mse_error)
            stats_tracker['snr_values'].append(snr_db)
            stats_tracker['cos_sim_values'].append(cos_sim)
            stats_tracker['total_tensors'] += 1
            stats_tracker['total_parameters'] += original_N * K

        with open(output_path, 'wb') as f:
            ndim = 2
            data_bytes = packed_data.size  
            scales_bytes = scales_fp16.size * 2
            flags = FLAG_HAS_SCALES | FLAG_INTERLEAVED
            if transpose:
                flags |= FLAG_TRANSPOSED

            f.write(CACTUS_MAGIC)                          # 4 bytes
            f.write(struct.pack('<I', flags))              # 4 bytes
            f.write(struct.pack('<I', CACTUS_ALIGNMENT))   # 4 bytes
            f.write(struct.pack('<I', ndim))               # 4 bytes

            f.write(struct.pack('<Q', N_padded))           # 8 bytes
            f.write(struct.pack('<Q', K))                  # 8 bytes
            f.write(struct.pack('<Q', 0))                  # 8 bytes
            f.write(struct.pack('<Q', 0))                  # 8 bytes

            f.write(struct.pack('<I', 3))                  # precision: 3 = INT4 packed (4 bytes)
            f.write(struct.pack('<Q', data_bytes))         # 8 bytes
            f.write(struct.pack('<Q', scales_bytes))       # 8 bytes
            f.write(struct.pack('<I', GROUP_SIZE))         # 4 bytes
            f.write(struct.pack('<I', num_groups))         # 4 bytes
            f.write(struct.pack('<Q', original_N))         # 8 bytes

            header_size = 84
            f.write(compute_padding(header_size, CACTUS_ALIGNMENT))

            f.write(scales_fp16.tobytes())
            scales_end = align_offset(header_size, CACTUS_ALIGNMENT) + scales_bytes
            f.write(compute_padding(scales_end, CACTUS_ALIGNMENT))

            f.write(packed_data.tobytes())

        return

    if precision == 'INT8' and len(shape) == 1:
        K = shape[0]

        if K % GROUP_SIZE != 0:
            pad_k = GROUP_SIZE - (K % GROUP_SIZE)
            data = np.pad(data, (0, pad_k), mode='constant', constant_values=0)
            original_data = np.pad(original_data, (0, pad_k), mode='constant', constant_values=0)
            K = data.shape[0]
            shape = [K]

        num_groups = K // GROUP_SIZE
        N = 1

        data_grouped = data.reshape(1, num_groups, GROUP_SIZE)

        group_abs_max = np.max(np.abs(data_grouped), axis=2)
        scales = (group_abs_max / 127.0).astype(np.float32)
        scales = np.maximum(scales, 1e-10)

        quantized = np.clip(
            np.round(data_grouped / scales[:, :, np.newaxis]),
            -128, 127
        ).astype(np.int8)
        quantized_flat = quantized.reshape(K)

        dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(K)
        mse_error = np.mean((original_data - dequantized) ** 2)
        snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')
        cos_sim = np.dot(original_data, dequantized) / (np.linalg.norm(original_data) * np.linalg.norm(dequantized) + 1e-10)

        scales_fp16 = scales.flatten().astype(np.float16)

        if stats_tracker:
            stats_tracker['int8_tensors'] += 1
            stats_tracker['quantized_parameters'] += original_data.size
            stats_tracker['mse_values'].append(mse_error)
            stats_tracker['snr_values'].append(snr_db)
            stats_tracker['cos_sim_values'].append(cos_sim)
            stats_tracker['total_tensors'] += 1
            stats_tracker['total_parameters'] += original_data.size

        with open(output_path, 'wb') as f:
            ndim = len(shape)
            data_bytes = quantized_flat.size
            scales_bytes = scales_fp16.size * 2
            flags = FLAG_HAS_SCALES 
            if transpose:
                flags |= FLAG_TRANSPOSED

            f.write(CACTUS_MAGIC)
            f.write(struct.pack('<I', flags))
            f.write(struct.pack('<I', CACTUS_ALIGNMENT))
            f.write(struct.pack('<I', ndim))

            for i in range(4):
                if i < ndim:
                    f.write(struct.pack('<Q', shape[i]))
                else:
                    f.write(struct.pack('<Q', 0))

            f.write(struct.pack('<I', 0))  # precision: INT8
            f.write(struct.pack('<Q', data_bytes))
            f.write(struct.pack('<Q', scales_bytes))
            f.write(struct.pack('<I', GROUP_SIZE))
            f.write(struct.pack('<I', num_groups))
            f.write(struct.pack('<Q', K)) 

            header_size = 84
            f.write(compute_padding(header_size, CACTUS_ALIGNMENT))

            f.write(scales_fp16.tobytes())
            scales_end = align_offset(header_size, CACTUS_ALIGNMENT) + scales_bytes
            f.write(compute_padding(scales_end, CACTUS_ALIGNMENT))

            f.write(quantized_flat.tobytes())

        return

    data = data.astype(np.float16)

    if stats_tracker:
        stats_tracker['fp16_tensors'] += 1
        stats_tracker['total_tensors'] += 1
        stats_tracker['total_parameters'] += original_data.size

    data_flat = data.flatten()

    with open(output_path, 'wb') as f:
        ndim = len(shape)
        data_bytes = data_flat.size * 2  # FP16 = 2 bytes
        flags = 0
        if transpose:
            flags |= FLAG_TRANSPOSED

        f.write(CACTUS_MAGIC)
        f.write(struct.pack('<I', flags))
        f.write(struct.pack('<I', CACTUS_ALIGNMENT))
        f.write(struct.pack('<I', ndim))

        for i in range(4):
            if i < ndim:
                f.write(struct.pack('<Q', shape[i]))
            else:
                f.write(struct.pack('<Q', 0))

        f.write(struct.pack('<I', 1)) 
        f.write(struct.pack('<Q', data_bytes))
        f.write(struct.pack('<Q', 0))  
        f.write(struct.pack('<I', 0))  
        f.write(struct.pack('<I', 0))  
        f.write(struct.pack('<Q', shape[0] if ndim >= 1 else 0)) 

        header_size = 84
        f.write(compute_padding(header_size, CACTUS_ALIGNMENT))

        f.write(data_flat.tobytes())


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
        'int8_tensors': 0,
        'int4_tensors': 0,
        'fp16_tensors': 0,
        'total_parameters': 0,
        'quantized_parameters': 0,
        'mse_values': [],
        'snr_values': [],
        'cos_sim_values': [],
        'saturation_warnings': 0
    }


def print_quantization_summary(quantization_stats, args=None):
    """Print a summary of quantization statistics."""
    int8_count = quantization_stats.get('int8_tensors', 0)
    int4_count = quantization_stats.get('int4_tensors', 0)
    fp16_count = quantization_stats.get('fp16_tensors', 0)
    quantized_count = int8_count + int4_count

    if quantized_count > 0:
        mse_values = np.array(quantization_stats['mse_values'])
        snr_values = np.array(quantization_stats['snr_values'])
        cos_sim_values = np.array(quantization_stats['cos_sim_values'])

        print("\nQuantization Summary:")
        print(f"MSE - Mean: {np.mean(mse_values):.2e}, Max: {np.max(mse_values):.2e}, Median: {np.median(mse_values):.2e}, Min: {np.min(mse_values):.2e}")
        print(f"SNR - Mean: {np.mean(snr_values):.1f}dB, Max: {np.max(snr_values):.1f}dB, Median: {np.median(snr_values):.1f}dB, Min: {np.min(snr_values):.1f}dB")
        print(f"CosSim - Mean: {np.mean(cos_sim_values):.6f}, Max: {np.max(cos_sim_values):.6f}, Median: {np.median(cos_sim_values):.6f}, Min: {np.min(cos_sim_values):.6f}")
        print(f"Processed {int8_count} INT8 tensors, {int4_count} INT4 tensors, {fp16_count} FP16 tensors")
