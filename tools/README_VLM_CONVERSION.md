# VLM Weight Conversion Guide

This guide explains how to convert Vision-Language Model (VLM) weights from HuggingFace format to Cactus format with INT8 quantization support.

## Supported VLM Architectures

The conversion tool now supports multiple VLM architectures:

### 1. SmolVLM
- **Architecture**: Compact vision-language model
- **Vision Encoder**: Standard ViT-based encoder
- **Projection**: Single linear projection layer
- **Example Models**: HuggingFaceTB/SmolVLM-Instruct

### 2. Qwen-VL
- **Architecture**: Qwen vision-language series
- **Vision Encoder**: CLIP-style encoder with attention pooling
- **Projection**: Attention-based pooling + projection
- **Example Models**: Qwen/Qwen-VL, Qwen/Qwen-VL-Chat
- **Special Features**:
  - Combined QKV attention weights (automatically split during conversion)
  - Attention pooling layers for vision feature aggregation

### 3. LLaVA
- **Architecture**: Large Language and Vision Assistant
- **Vision Encoder**: CLIP ViT encoder
- **Projection**: Multi-layer MLP projector
- **Example Models**: liuhaotian/llava-v1.5-7b, llava-hf/llava-1.5-7b-hf
- **Special Features**:
  - Two-layer MLP projection (linear_1 + linear_2)
  - Flexible vision tower configurations

## Usage

### Basic Conversion

```bash
python tools/convert_hf.py <model_name> <output_dir> --precision INT8
```

### Examples

#### Convert SmolVLM
```bash
python tools/convert_hf.py HuggingFaceTB/SmolVLM-Instruct ./models/smolvlm --precision INT8
```

#### Convert Qwen-VL
```bash
python tools/convert_hf.py Qwen/Qwen-VL-Chat ./models/qwen-vl --precision INT8
```

#### Convert LLaVA
```bash
python tools/convert_hf.py liuhaotian/llava-v1.5-7b ./models/llava-1.5-7b --precision INT8
```

### Advanced Options

#### Custom Quantization Parameters
```bash
python tools/convert_hf.py <model_name> <output_dir> \
    --precision INT8 \
    --snr-threshold 20.0 \
    --saturation-threshold 0.01 \
    --outlier-percentile 0.01 \
    --sigma-multiplier 3.5
```

#### Using Cache Directory
```bash
python tools/convert_hf.py <model_name> <output_dir> \
    --precision INT8 \
    --cache-dir ~/.cache/huggingface
```

## Output Structure

The conversion tool creates the following files in the output directory:

### Text Model Weights
- `token_embeddings.weights` - Token embedding matrix
- `output_weight.weights` - Output projection (if not tied)
- `output_norm.weights` - Final layer normalization
- `layer_N_*.weights` - Transformer layer weights

### Vision Encoder Weights
- `vision_patch_embedding.weights` - Patch embedding convolution
- `vision_class_embedding.weights` - Class token embedding (if present)
- `vision_position_embedding.weights` - Position embeddings
- `vision_pre_layernorm.weights` - Pre-encoder normalization (if present)
- `vision_post_layernorm.weights` - Post-encoder normalization
- `vision_layer_N_*.weights` - Vision transformer layer weights

### Projection Weights
- **SmolVLM**: `connector_proj.weights`
- **Qwen-VL**: 
  - `connector_proj.weights`
  - `vision_attn_pool_*.weights` (attention pooling layers)
- **LLaVA**:
  - `connector_proj_1.weights` + `connector_proj_1.bias.weights`
  - `connector_proj_2.weights` + `connector_proj_2.bias.weights`

### Configuration Files
- `config.txt` - Model configuration
- `vocab.txt` - Vocabulary
- `merges.txt` - BPE merges (if applicable)
- `tokenizer_config.txt` - Tokenizer configuration
- `special_tokens.json` - Special token mappings
- `preprocessor_config.json` - Image preprocessing configuration

## INT8 Quantization

### Vision Encoder Quantization

The tool applies INT8 quantization to vision encoder weights with the following strategy:

1. **Quantized Layers**:
   - Attention projection weights (Q, K, V, Output)
   - MLP/FFN weights (fc1, fc2)
   - Vision-to-language projection weights

2. **FP16 Preserved Layers**:
   - Layer normalization weights and biases
   - Embedding layers (patch, class, position)
   - Layers with high saturation or low SNR

3. **Quantization Method**:
   - Symmetric quantization with scale factor
   - Outlier clipping for better range utilization
   - Per-tensor quantization (not per-channel)

### Quality Metrics

The tool reports quantization quality metrics:
- **MSE**: Mean Squared Error between original and quantized weights
- **SNR**: Signal-to-Noise Ratio in dB
- **CosSim**: Cosine similarity between original and quantized weights

Example output:
```
Quantization Summary:
MSE - Mean: 1.23e-05, Max: 4.56e-04, Median: 8.90e-06, Min: 1.23e-07
SNR - Mean: 45.2dB, Max: 78.9dB, Median: 43.1dB, Min: 32.5dB
CosSim - Mean: 0.999876, Max: 0.999998, Median: 0.999891, Min: 0.998234
Processed 156 INT8 tensors, 24 FP16 tensors (2 SNR<20.0dB fallbacks)
```

## Architecture Detection

The tool automatically detects the VLM architecture based on:
1. Model type string in configuration
2. Presence of architecture-specific weight keys
3. Configuration structure (text_config, vision_config)

Detection logic:
- Contains "smolvlm" or "smol" → SmolVLM
- Contains "qwen" + "vl" → Qwen-VL
- Contains "llava" → LLaVA
- Default → SmolVLM (with warning)

## Troubleshooting

### Missing Dependencies

If you see errors about missing packages:
```bash
pip install torch transformers Pillow num2words torchvision
```

### Architecture Not Detected

If the tool doesn't recognize your VLM architecture:
1. Check the model's `config.json` for the `model_type` field
2. Add a new architecture pattern to `tools/vlm_architectures.py`
3. Or use the default SmolVLM patterns (may require manual weight mapping)

### Missing Weights

If some weights are not exported:
1. Check `missing_weights.txt` in the output directory
2. Verify the weight keys in the model's state_dict
3. Add missing patterns to the architecture definition

### Quantization Issues

If quantization quality is poor:
1. Adjust `--snr-threshold` (lower = more aggressive quantization)
2. Increase `--saturation-threshold` to enable outlier clipping
3. Use `--precision FP16` for critical layers
4. Check the quantization summary for problematic tensors

## Adding New VLM Architectures

To add support for a new VLM architecture:

1. **Update `tools/vlm_architectures.py`**:
   - Add detection logic in `detect_vlm_architecture()`
   - Define weight patterns in `get_vision_encoder_patterns()`
   - Define projection patterns in `get_projection_patterns()`

2. **Test the conversion**:
   ```bash
   python tools/convert_hf.py <new_model> ./test_output --precision INT8
   ```

3. **Verify output**:
   - Check all expected weight files are created
   - Verify quantization quality metrics
   - Test inference with converted weights

## Performance Considerations

### Memory Usage
- Vision encoders can be large (1-2GB for ViT-L)
- INT8 quantization reduces size by ~4x
- Use `--cache-dir` to avoid re-downloading models

### Conversion Time
- Typical conversion: 2-5 minutes
- Depends on model size and quantization settings
- GPU not required (CPU-only conversion)

### Mobile Optimization
- INT8 quantization optimized for mobile inference
- Vision encoder weights are the largest component
- Consider using smaller vision encoders for mobile deployment

## References

- [Cactus Documentation](../docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Vision-Language Models Overview](../docs/vision_weights.md)
