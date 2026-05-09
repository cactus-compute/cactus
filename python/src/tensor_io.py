import numpy as np
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import torch
except ImportError:
    torch = None


CACTUS_MAGIC = b'CACT'
CACTUS_ALIGNMENT = 32

FLAG_PAGE_ALIGNED = 1 << 1
FLAG_TRANSPOSED = 1 << 2


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



def fold_bn_into_conv(conv_w, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    gamma = bn_weight.float().numpy()
    beta = bn_bias.float().numpy()
    mean = bn_mean.float().numpy()
    var = bn_var.float().numpy()
    w = conv_w.float().numpy()

    inv_std = gamma / np.sqrt(var + eps)
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    new_w = w * inv_std.reshape(shape)
    new_b = beta - mean * inv_std

    return new_w, new_b


def save_tensor_with_header(tensor, output_path, precision='FP16', transpose=False, stats_tracker=None, args=None, model_type=None):
    """Save a tensor to binary format with header metadata.

    Args:
        tensor: The tensor to save (PyTorch or NumPy)
        output_path: Path to save the tensor
        precision: Storage precision ('FP16')
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

    if model_type == 'gemma' and 'norm' in str(output_path):
        data = data + 1.0

    if model_type == 'gemma4':
        GEMMA4_WEIGHT_SCALE = 16.0
        filename = output_path.name
        is_audio_weight = filename.startswith('audio_')
        if any(x in filename for x in ['input_norm', 'post_attn_norm', 'pre_ffn_norm', 'post_ffn_norm',
                                       'post_per_layer_norm', 'post_proj_norm']):
            data = data / GEMMA4_WEIGHT_SCALE
        elif any(x in filename for x in ['ffn_gate', 'ffn_up', 'per_layer_gate', 'moe_gate_proj', 'moe_up_proj']):
            data = data * GEMMA4_WEIGHT_SCALE
        elif 'router_scale' in filename:
            data = data / GEMMA4_WEIGHT_SCALE
        elif filename in ('token_embeddings.weights', 'output_weight.weights',
                          'embed_vision_proj.weights', 'embed_vision_embedding.weights'):
            data = data / GEMMA4_WEIGHT_SCALE
        elif filename == 'output_norm.weights':
            data = data * GEMMA4_WEIGHT_SCALE

    shape = list(data.shape)
    if transpose and len(shape) == 2:
        data = data.T
        shape = [shape[1], shape[0]]

    if (
        model_type == 'gemma4'
        and len(shape) == 3
        and 'vision' not in output_path.name
        and not output_path.name.startswith('audio_')
    ):
        data = data.transpose(0, 2, 1)
        shape = [shape[0], shape[2], shape[1]]

    data = data.astype(np.float16)

    if stats_tracker:
        stats_tracker['fp16_tensors'] += 1
        stats_tracker['total_tensors'] += 1
        stats_tracker['total_parameters'] += data.size

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
    """Create an empty stats tracker dictionary."""
    return {
        'total_tensors': 0,
        'fp16_tensors': 0,
        'total_parameters': 0,
    }


def print_quantization_summary(quantization_stats, args=None):
    """Print a summary of conversion statistics."""
    fp16_count = quantization_stats.get('fp16_tensors', 0)
    total = quantization_stats.get('total_tensors', 0)
    params = quantization_stats.get('total_parameters', 0)
    if total > 0:
        print(f"\nConversion Summary: {fp16_count} FP16 tensors, {params} total parameters")
