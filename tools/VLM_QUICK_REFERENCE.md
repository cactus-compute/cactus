# VLM Weight Conversion - Quick Reference

## Quick Start

```bash
# Convert any supported VLM model
python tools/convert_hf.py <model_name> <output_dir> --precision INT8
```

## Supported Models

| Architecture | Example Models | Special Features |
|-------------|----------------|------------------|
| **SmolVLM** | HuggingFaceTB/SmolVLM-Instruct | Standard ViT encoder |
| **Qwen-VL** | Qwen/Qwen-VL, Qwen/Qwen-VL-Chat | Attention pooling, Combined QKV |
| **LLaVA** | liuhaotian/llava-v1.5-7b | Multi-layer MLP projector |

## Common Commands

```bash
# Basic conversion with INT8 quantization
python tools/convert_hf.py HuggingFaceTB/SmolVLM-Instruct ./models/smolvlm --precision INT8

# Use custom cache directory
python tools/convert_hf.py Qwen/Qwen-VL-Chat ./models/qwen-vl --cache-dir ~/.cache/hf

# Adjust quantization quality
python tools/convert_hf.py <model> <output> --precision INT8 --snr-threshold 25.0

# Use FP16 instead of INT8
python tools/convert_hf.py <model> <output> --precision FP16
```

## Output Files

### Essential Files
- `config.txt` - Model configuration
- `token_embeddings.weights` - Text embeddings
- `vision_patch_embedding.weights` - Vision patch embeddings
- `vision_layer_N_*.weights` - Vision transformer layers
- `connector_proj*.weights` - Vision-to-language projection

### Configuration Files
- `vocab.txt` - Vocabulary
- `tokenizer_config.txt` - Tokenizer settings
- `preprocessor_config.json` - Image preprocessing config

## Architecture-Specific Outputs

### SmolVLM
```
connector_proj.weights
```

### Qwen-VL
```
connector_proj.weights
vision_attn_pool_q.weights
vision_attn_pool_k.weights
vision_attn_pool_v.weights
vision_attn_pool_out.weights
```

### LLaVA
```
connector_proj_1.weights
connector_proj_1.bias.weights
connector_proj_2.weights
connector_proj_2.bias.weights
```

## Quantization Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--precision` | INT8 | INT8, FP16, or FP32 |
| `--snr-threshold` | 20.0 | Minimum SNR for INT8 (dB) |
| `--saturation-threshold` | 0.01 | Outlier clipping threshold |
| `--outlier-percentile` | 0.01 | Percentile for outliers |
| `--sigma-multiplier` | 3.5 | Std dev multiplier for clipping |

## Troubleshooting

### Missing Dependencies
```bash
pip install torch transformers Pillow num2words torchvision
```

### Check Architecture Detection
```bash
python tools/test_vlm_conversion.py
```

### Verify Conversion
```bash
# Check output directory
ls -lh <output_dir>/*.weights

# Check config
cat <output_dir>/config.txt
```

## Testing

```bash
# Run test suite
python tools/test_vlm_conversion.py

# Expected output:
# ✓ All tests passed!
```

## Adding New Architectures

1. Edit `tools/vlm_architectures.py`
2. Add detection logic to `detect_vlm_architecture()`
3. Add patterns to `get_vision_encoder_patterns()`
4. Add patterns to `get_projection_patterns()`
5. Run tests: `python tools/test_vlm_conversion.py`

## Performance Tips

- **Memory**: Use `--cache-dir` to avoid re-downloading
- **Speed**: Conversion takes 2-5 minutes typically
- **Size**: INT8 reduces model size by ~4x
- **Quality**: Check quantization summary for SNR/MSE metrics

## Example Workflow

```bash
# 1. Convert model
python tools/convert_hf.py HuggingFaceTB/SmolVLM-Instruct ./models/smolvlm --precision INT8

# 2. Check output
ls -lh ./models/smolvlm/

# 3. Verify config
cat ./models/smolvlm/config.txt

# 4. Check quantization quality
# (Look for "Quantization Summary" in conversion output)

# 5. Test inference
# (Use converted weights with Cactus engine)
```

## Documentation

- Full guide: `tools/README_VLM_CONVERSION.md`
- Test suite: `tools/test_vlm_conversion.py`
- Architecture module: `tools/vlm_architectures.py`
- Main converter: `tools/convert_hf.py`
