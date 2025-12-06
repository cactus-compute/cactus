#!/usr/bin/env python3
import numpy as np
import struct
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoConfig
    import torch
    # Note: for SmolVLM, we also need Pillow + num2words + torchvision
except ImportError as e:
    print(f"Please install required packages: pip install torch transformers\nError: {e}")
    sys.exit(1)

try:
    from transformers import Lfm2VlForConditionalGeneration
except ImportError:
    Lfm2VlForConditionalGeneration = None
try:
    from huggingface_hub import hf_hub_download  # type: ignore
except ImportError:
    hf_hub_download = None  # type: ignore


def save_tensor_with_header(tensor, output_path, precision='FP32', transpose=False, stats_tracker=None, args=None, model_type=None):
    if isinstance(tensor, torch.Tensor):
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
            # print(f"Skipping INT8 quantization for {filename}, using FP16 instead.")
            precision = 'FP16'
    
    if precision == 'INT8':
        qmin, qmax = -128, 127
        standard_scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
        
        standard_zero_point = qmax - max_val / standard_scale
        standard_zero_point_clipped = np.clip(np.round(standard_zero_point), qmin, qmax)
        test_quantized = np.clip(np.round(original_data / standard_scale + standard_zero_point_clipped), qmin, qmax)
        test_saturation = np.sum(np.abs(test_quantized) >= 127) / original_data.size
        
        saturation_threshold = args.saturation_threshold if args else 0.01
        if test_saturation > saturation_threshold:
            outlier_percentile = args.outlier_percentile if args else 0.01
            lower_percentile = np.percentile(original_data, outlier_percentile)
            upper_percentile = np.percentile(original_data, 100 - outlier_percentile)
            
            mean_val = np.mean(original_data)
            std_val = np.std(original_data)
            sigma_multiplier = args.sigma_multiplier if args else 3.5
            three_sigma_min = mean_val - sigma_multiplier * std_val
            three_sigma_max = mean_val + sigma_multiplier * std_val
            
            clipped_min = max(min_val, min(lower_percentile, three_sigma_min))
            clipped_max = min(max_val, max(upper_percentile, three_sigma_max))
            
            range_threshold = args.range_threshold if args else 0.5
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
            saturation_warning_threshold = args.saturation_warning_threshold if args else 0.1
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
    
    #print(f"Saving {output_path.name}: {precision} {shape}")
    
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
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)

def convert_hf_model_weights(model, output_dir, precision='INT8', args=None):
    
    quantization_stats = {
        'total_tensors': 0,
        'quantized_tensors': 0,
        'total_parameters': 0,
        'quantized_parameters': 0,
        'mse_values': [],
        'snr_values': [],
        'cos_sim_values': [],
        'saturation_warnings': 0
    }

    state_dict = model.state_dict()
    config = model.config
    saved_tensor_full_names = set()
    
    # Helper function to safely get config attributes
    def _cfg_get(c, key, default=None):
        if c is None:
            return default
        try:
            if isinstance(c, dict):
                return c.get(key, default)
        except Exception:
            pass
        try:
            return getattr(c, key, default)
        except Exception:
            return default
    
    # Check if this is a VLM model by looking for text_config/vision_config
    text_cfg = _cfg_get(config, 'text_config', None)
    vision_cfg = _cfg_get(config, 'vision_config', None)
    is_vlm = text_cfg is not None or vision_cfg is not None
    
    # Use text_cfg if available, otherwise use main config
    cfg = text_cfg if text_cfg is not None else config
    
    tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
    model_type_str = _cfg_get(cfg, 'model_type', _cfg_get(config, 'model_type', '')).lower()

    if 'gemma' in model_type_str:
        detected_model_type = 'gemma'
    elif 'lfm2' in model_type_str:
        detected_model_type = 'lfm2'
    elif 'qwen' in model_type_str:
        detected_model_type = 'qwen'
    elif 'llama' in model_type_str:
        if('smol' in str(output_dir)):
            detected_model_type = 'smol'
        else:
            detected_model_type = 'llama'
    elif 'bert' in model_type_str:
        detected_model_type = 'bert'
    elif 'whisper' in model_type_str:
        detected_model_type = 'whisper'
    else:
        detected_model_type = 'qwen'
        if model_type_str:
            print(f"  Warning: Unknown model type '{model_type_str}', defaulting to 'qwen'")

    # Build model config - use VLM structure if available
    model_config = {
        'vocab_size': _cfg_get(cfg, 'vocab_size', _cfg_get(config, 'vocab_size', 0)),
        'hidden_dim': _cfg_get(cfg, 'hidden_size', _cfg_get(cfg, 'hidden_dim', 0)),
        'num_layers': int(_cfg_get(cfg, 'num_hidden_layers', _cfg_get(cfg, 'num_layers', 0) or 0)),
        'attention_heads': _cfg_get(cfg, 'num_attention_heads', 0),
        'attention_kv_heads': _cfg_get(cfg, 'num_key_value_heads', _cfg_get(cfg, 'num_attention_heads', 0)),
        'ffn_intermediate_dim': _cfg_get(cfg, 'intermediate_size', _cfg_get(cfg, 'n_inner', 0)),
        'context_length': _cfg_get(cfg, 'max_position_embeddings', _cfg_get(cfg, 'max_sequence_length', 0)),
        'rope_theta': _cfg_get(cfg, 'rope_theta', _cfg_get(config, 'rope_theta', 10000.0)),
        'attention_head_dim': int(_cfg_get(cfg, 'head_dim', int(_cfg_get(cfg, 'hidden_size', _cfg_get(cfg, 'hidden_dim', 0)) // max(1, _cfg_get(cfg, 'num_attention_heads', 1))))),
        'layer_norm_eps': _cfg_get(cfg, 'layer_norm_eps', _cfg_get(cfg, 'layer_norm_epsilon', _cfg_get(cfg, 'rms_norm_eps', 1e-6))),
        'num_experts': _cfg_get(cfg, 'num_experts', 0),
        'num_shared_experts': _cfg_get(cfg, 'num_shared_experts', 0),
        'num_top_experts': _cfg_get(cfg, 'moe_top_k', _cfg_get(cfg, 'num_top_experts', 0)),
        'moe_every_n_layers': _cfg_get(cfg, 'moe_every_n_layers', 0),
        'tie_word_embeddings': tie_word_embeddings,
        'model_type': detected_model_type,
    }

    # Add VLM-specific config if this is a VLM model
    if is_vlm and vision_cfg is not None:
        vision_hidden = int(_cfg_get(vision_cfg, 'hidden_size', 0))
        vision_image_size = _cfg_get(vision_cfg, 'image_size', _cfg_get(vision_cfg, 'size', {}).get('longest_edge', 0) if isinstance(_cfg_get(vision_cfg, 'size', {}), dict) else _cfg_get(vision_cfg, 'image_size', 0))
        vision_patch = int(_cfg_get(vision_cfg, 'patch_size', 0))
        vision_heads = int(_cfg_get(vision_cfg, 'num_attention_heads', 0))
        vision_num_layers = int(_cfg_get(vision_cfg, 'num_hidden_layers', _cfg_get(vision_cfg, 'num_layers', 0) or 0))
        num_channels = int(_cfg_get(vision_cfg, 'num_channels', 3))
        visual_tokens_per_img = 0
        try:
            if vision_patch > 0 and vision_image_size > 0:
                per_side = vision_image_size // vision_patch
                visual_tokens_per_img = per_side * per_side
        except Exception:
            visual_tokens_per_img = 0

        pixel_shuffle_factor = int(_cfg_get(config, 'scale_factor', _cfg_get(vision_cfg, 'scale_factor', 1) or 1))
        downsample_factor = int(_cfg_get(config, 'downsample_factor', 2))
        
        model_config.update({
            'vision_hidden_size': int(vision_hidden),
            'vision_num_layers': int(vision_num_layers),
            'vision_image_size': int(vision_image_size),
            'vision_patch_size': int(vision_patch),
            'vision_attention_heads': int(vision_heads),
            'vision_embed_dim': int(vision_hidden),
            'num_channels': int(num_channels),
            'visual_tokens_per_img': int(visual_tokens_per_img),
            'use_pixel_shuffle': bool(pixel_shuffle_factor > 1),
            'pixel_shuffle_factor': int(pixel_shuffle_factor),
            'use_image_tokens': bool(_cfg_get(config, 'image_token_id', None) is not None),
            'use_layout_tags': False,
            'downsample_factor': int(downsample_factor),
        })

    if detected_model_type == 'lfm2':
        layer_types = getattr(cfg, 'layer_types', [])
        model_config['layer_types'] = layer_types
        model_config['conv_L_cache'] = getattr(cfg, 'conv_L_cache', 0)
    
    # Token embeddings - support both regular and VLM prefixes
    embed_names = [
        'model.language_model.embed_tokens.weight', 'model.text_model.embed_tokens.weight',
        'model.embed_tokens.weight', 'embed_tokens.weight', 'embeddings.weight', 'transformer.wte.weight'
    ]
    embedding_found = False
    for name in embed_names:
        if name in state_dict:
            embedding_tensor = state_dict[name]
            save_tensor_with_header(embedding_tensor, output_dir / "token_embeddings.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
            saved_tensor_full_names.add(name)
            embedding_found = True
            break
    if model_type_str == 'nomic_bert':
        if 'embeddings.word_embeddings.weight' in state_dict:
            fused_embedding_tensor = state_dict['embeddings.word_embeddings.weight'] + state_dict.get('embeddings.token_type_embeddings.weight', torch.zeros([1]))
            save_tensor_with_header(fused_embedding_tensor, output_dir / "token_embeddings.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
            saved_tensor_full_names.add('embeddings.word_embeddings.weight')
            if 'embeddings.token_type_embeddings.weight' in state_dict:
                saved_tensor_full_names.add('embeddings.token_type_embeddings.weight')
            embedding_found = True


    elif model_type_str == 'whisper':
        weights = ['decoder.embed_tokens.weight', 'decoder.embed_positions.weight', 'decoder.layer_norm.weight', 'decoder.layer_norm.bias', 'proj_out.weight', 'encoder.embed_positions.weight', 'encoder.conv1.bias', 'encoder.conv1.weight', 'encoder.conv2.bias', 'encoder.conv2.weight', 'encoder.layer_norm.bias', 'encoder.layer_norm.weight']
        save_names = ['decoder_token_embeddings.weights', 'decoder_position_embeddings.weights', 'decoder_norm.weights', 'decoder_norm.bias', 'output_layer.weights', 'encoder_position_embeddings.weights', 'encoder_conv1_bias.bias', 'encoder_conv1_weight.weights', 'encoder_conv2_bias.bias', 'encoder_conv2_weight.weights', 'encoder_norm_bias.bias', 'encoder_norm_weight.weights']
        for name, save_name in zip(weights, save_names):
            if name in state_dict:
                save_tensor_with_header(state_dict[name], output_dir / save_name, precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)
        embedding_found = True
    
    if embedding_found:
        embedding_norm_names = {'emb_ln.weight': 'embedding_layernorm.weight', 'emb_ln.bias': 'embedding_layernorm.bias'}
        for name, file_name in embedding_norm_names.items():
            if name in state_dict:
                save_tensor_with_header(state_dict[name], output_dir / file_name, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)
    
    if not tie_word_embeddings or is_vlm:
        output_names = ['lm_head.weight', 'output.weight', 'transformer.lm_head.weight', 'model.text_model.lm_head.weight']
        for name in output_names:
            if name in state_dict:
                tensor = state_dict[name]
                save_tensor_with_header(tensor, output_dir / "output_weight.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)
                break
                
    # Output norm (final norm before output head) - supports both regular and VLM prefixes
    output_norm_names = ['model.norm.weight', 'norm.weight', 'final_layernorm.weight', 'transformer.ln_f.weight', 
                          'model.embedding_norm.weight', 'model.language_model.embedding_norm.weight', 'model.text_model.norm.weight']
    for name in output_norm_names:
        if name in state_dict:
            tensor = state_dict[name]
            save_tensor_with_header(tensor, output_dir / "output_norm.weights", precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
            saved_tensor_full_names.add(name)
            break

    # Vision tower weights (VLM models only - will be skipped for regular models)
    if is_vlm:
        vision_items = [
            ('model.vision_tower.vision_model.embeddings.patch_embedding.weight', 'vision_patch_embedding.weights'),
            ('model.vision_model.embeddings.patch_embedding.weight', 'vision_patch_embedding.weights'),
            ('model.vision_tower.vision_model.embeddings.patch_embedding.bias', 'vision_patch_embedding.bias.weights'),
            ('model.vision_model.embeddings.patch_embedding.bias', 'vision_patch_embedding.bias.weights'),
            ('model.vision_tower.vision_model.embeddings.position_embedding.weight', 'vision_position_embedding.weights'),
            ('model.vision_model.embeddings.position_embedding.weight', 'vision_position_embedding.weights'),
            ('model.vision_tower.vision_model.post_layernorm.weight', 'vision_post_layernorm.weights'),
            ('model.vision_model.post_layernorm.weight', 'vision_post_layernorm.weights'),
            ('model.vision_tower.vision_model.post_layernorm.bias', 'vision_post_layernorm.bias.weights'),
            ('model.vision_model.post_layernorm.bias', 'vision_post_layernorm.bias.weights')
        ]
        for key, outname in vision_items:
            if key in state_dict:
                save_tensor_with_header(state_dict[key], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(key)

        # Multi-modal projector weights
        projector_weights = [
            ('model.multi_modal_projector.linear_1.weight', 'projector_linear1.weights'),
            ('model.multi_modal_projector.linear_1.bias', 'projector_linear1.bias.weights'),
            ('model.multi_modal_projector.linear_2.weight', 'projector_linear2.weights'),
            ('model.multi_modal_projector.linear_2.bias', 'projector_linear2.bias.weights'),
            ('model.multi_modal_projector.layer_norm.weight', 'projector_layer_norm.weights'),
            ('model.multi_modal_projector.layer_norm.bias', 'projector_layer_norm.bias.weights'),
        ]
        for key, outname in projector_weights:
            if key in state_dict:
                save_tensor_with_header(state_dict[key], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(key)

        # Vision encoder layers
        max_v_idx = -1
        vision_prefix = None
        for k in state_dict.keys():
            m = re.search(r'model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.', k)
            if m:
                vision_prefix = 'model.vision_tower.vision_model.encoder.layers.'
                try:
                    idx = int(m.group(1))
                    if idx > max_v_idx:
                        max_v_idx = idx
                except Exception:
                    pass
            if not vision_prefix:
                m = re.search(r'model\.vision_model\.encoder\.layers\.(\d+)\.', k)
                if m:
                    vision_prefix = 'model.vision_model.encoder.layers.'
                    try:
                        idx = int(m.group(1))
                        if idx > max_v_idx:
                            max_v_idx = idx
                    except Exception:
                        pass

        if not vision_prefix:
            vision_prefix = 'model.vision_model.encoder.layers.'

        vision_layers = max_v_idx + 1 if max_v_idx >= 0 else 0

        for i_v in range(vision_layers):
            vpref = f'{vision_prefix}{i_v}.'
            vision_layer_weights = [
                (vpref + 'layer_norm1.weight', f'vision_layer_{i_v}_layer_norm1.weights'),
                (vpref + 'layer_norm1.bias', f'vision_layer_{i_v}_layer_norm1.bias.weights'),
                (vpref + 'layer_norm2.weight', f'vision_layer_{i_v}_layer_norm2.weights'),
                (vpref + 'layer_norm2.bias', f'vision_layer_{i_v}_layer_norm2.bias.weights'),
                (vpref + 'mlp.fc1.weight', f'vision_layer_{i_v}_ffn_fc1.weights'),
                (vpref + 'mlp.fc1.bias', f'vision_layer_{i_v}_ffn_fc1.bias.weights'),
                (vpref + 'mlp.fc2.weight', f'vision_layer_{i_v}_ffn_fc2.weights'),
                (vpref + 'mlp.fc2.bias', f'vision_layer_{i_v}_ffn_fc2.bias.weights'),
                (vpref + 'self_attn.q_proj.weight', f'vision_layer_{i_v}_self_attn_q.weights'),
                (vpref + 'self_attn.k_proj.weight', f'vision_layer_{i_v}_self_attn_k.weights'),
                (vpref + 'self_attn.v_proj.weight', f'vision_layer_{i_v}_self_attn_v.weights'),
                (vpref + 'self_attn.out_proj.weight', f'vision_layer_{i_v}_self_attn_out.weights'),
                (vpref + 'self_attn.q_proj.bias', f'vision_layer_{i_v}_self_attn_q.bias.weights'),
                (vpref + 'self_attn.k_proj.bias', f'vision_layer_{i_v}_self_attn_k.bias.weights'),
                (vpref + 'self_attn.v_proj.bias', f'vision_layer_{i_v}_self_attn_v.bias.weights'),
                (vpref + 'self_attn.out_proj.bias', f'vision_layer_{i_v}_self_attn_out.bias.weights'),
            ]
            for fname, out in vision_layer_weights:
                if fname in state_dict:
                    save_tensor_with_header(state_dict[fname], output_dir / out, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    saved_tensor_full_names.add(fname)

    num_layers = model_config['num_layers']
    missing_tensors = []
    for i in range(num_layers):
        
        layer_prefixes = [f'model.language_model.layers.{i}.', f'model.text_model.layers.{i}.', f'model.layers.{i}.', f'layers.{i}.', f'transformer.h.{i}.', f'encoder.layers.{i}.', f'decoder.layers.{i}.', f'model.encoder.layers.{i}.', f'model.decoder.layers.{i}.']
        
        existing_prefixes = set() #Culprit
        for prefix in layer_prefixes:
            for key in state_dict.keys():
                if(key.startswith(prefix)):
                    existing_prefixes.add(prefix)

        if not existing_prefixes:
            missing_tensors.append((i, "<no-layer-prefix>", ["<no-matching-prefix>"]))
            continue

        weight_patterns = [
            (['self_attn.q_proj.weight', 'attn.q_proj.weight', 'attn.c_attn.weight'], precision, f'layer_{i}_attn_q.weights', False),
            (['self_attn.k_proj.weight', 'attn.k_proj.weight'], precision, f'layer_{i}_attn_k.weights', False),
            (['self_attn.v_proj.weight', 'attn.v_proj.weight'], precision, f'layer_{i}_attn_v.weights', False),
            (['self_attn.o_proj.weight', 'attn.o_proj.weight', 'attn.c_proj.weight', 'self_attn.out_proj.weight'], precision, f'layer_{i}_attn_output.weights', False),
            (['input_layernorm.weight', 'ln_1.weight', 'operator_norm.weight'], precision, f'layer_{i}_input_norm.weights', False),
            (['self_attn.q_norm.weight', 'self_attn.q_layernorm.weight'], precision, f'layer_{i}_attn_q_norm.weights', False),
            (['self_attn.k_norm.weight', 'self_attn.k_layernorm.weight'], precision, f'layer_{i}_attn_k_norm.weights', False),
            (['mlp.gate_proj.weight', 'mlp.c_fc.weight', 'feed_forward.w1.weight'], precision, f'layer_{i}_ffn_gate.weights', False),
            (['mlp.up_proj.weight', 'feed_forward.w3.weight'], precision, f'layer_{i}_ffn_up.weights', False),
            (['mlp.down_proj.weight', 'mlp.c_proj.weight', 'feed_forward.w2.weight'], precision, f'layer_{i}_ffn_down.weights', False),
            (['post_attention_layernorm.weight', 'ln_2.weight', 'ffn_norm.weight'], precision, f'layer_{i}_post_attn_norm.weights', False),
            # Gemma3 specific layer norms 
            (['pre_feedforward_layernorm.weight'], precision, f'layer_{i}_pre_ffn_norm.weights', False),
            (['post_feedforward_layernorm.weight'], precision, f'layer_{i}_post_ffn_norm.weights', False),
            (['conv.in_proj.weight'], precision, f'layer_{i}_conv_in_proj.weights', False),
            (['conv.out_proj.weight'], precision, f'layer_{i}_conv_out_proj.weights', False),
            (['conv.conv.weight'], precision, f'layer_{i}_conv_depthwise.weights', False),
            # Nomic BERT specific parameters
            (['attn.Wqkv.bias'], precision, f'layer_{i}_attn_{{channel}}.bias', False),
            (['attn.Wqkv.weight'], precision, f'layer_{i}_attn_{{channel}}.weights', False),
            (['attn.out_proj.bias'], precision, f'layer_{i}_attn_output.bias', False),
            (['attn.out_proj.weight'], precision, f'layer_{i}_attn_output.weights', False),
            (['mlp.fc1.bias'], precision, f'layer_{i}_mlp_fc1.bias', False),
            (['mlp.fc1.weight'], precision, f'layer_{i}_mlp_fc1.weights', False),
            (['mlp.fc2.bias'], precision, f'layer_{i}_mlp_fc2.bias', False),
            (['mlp.fc2.weight'], precision, f'layer_{i}_mlp_fc2.weights', False),
            (['norm1.bias'], precision, f'layer_{i}_norm1.bias', False),
            (['norm1.weight'], precision, f'layer_{i}_norm1.weights', False),
            (['norm2.bias'], precision, f'layer_{i}_norm2.bias', False),
            (['norm2.weight'], precision, f'layer_{i}_norm2.weights', False),
            (['mlp.experts.bias'], precision, f'layer_{i}_mlp_experts.bias', False),
            (['mlp.experts.mlp.w1'], precision, f'layer_{i}_mlp_expert_{{channel}}.mlp1.weights', False),
            (['mlp.experts.mlp.w2'], precision, f'layer_{i}_mlp_expert_{{channel}}.mlp2.weights', True),
            (['mlp.router.layer.weight'], precision, f'layer_{i}_mlp_router.layer.weights', False),
            # Whisper specific parameters
            (['encoder_attn.q_proj.weight'], precision, f'layer_{i}_encoder_attn_q.weights', False),
            (['encoder_attn.k_proj.weight'], precision, f'layer_{i}_encoder_attn_k.weights', False),
            (['encoder_attn.v_proj.weight'], precision, f'layer_{i}_encoder_attn_v.weights', False),
            (['encoder_attn.out_proj.weight'], precision, f'layer_{i}_encoder_attn_output.weights', False),
            (['encoder_attn.q_proj.bias'], precision, f'layer_{i}_encoder_attn_q.bias', False),
            (['encoder_attn.v_proj.bias'], precision, f'layer_{i}_encoder_attn_v.bias', False),
            (['encoder_attn.out_proj.bias'], precision, f'layer_{i}_encoder_attn_output.bias', False),
            (['encoder_attn_layer_norm.weight'], precision, f'layer_{i}_encoder_attn_norm.weights', False),
            (['encoder_attn_layer_norm.bias'], precision, f'layer_{i}_encoder_attn_norm.bias', False),
            (['fc1.weight'], precision, f'layer_{i}_mlp_fc1.weights', False),
            (['fc1.bias'], precision, f'layer_{i}_mlp_fc1.bias', False),
            (['fc2.weight'], precision, f'layer_{i}_mlp_fc2.weights', False),
            (['fc2.bias'], precision, f'layer_{i}_mlp_fc2.bias', False),
            (['final_layer_norm.weight'], precision, f'layer_{i}_final_norm.weights', False),
            (['final_layer_norm.bias'], precision, f'layer_{i}_final_norm.bias', False),
            (['self_attn.q_proj.weight'], precision, f'layer_{i}_self_attn_q.weights', False),
            (['self_attn.k_proj.weight'], precision, f'layer_{i}_self_attn_k.weights', False),
            (['self_attn.v_proj.weight'], precision, f'layer_{i}_self_attn_v.weights', False),
            (['self_attn.q_proj.bias'], precision, f'layer_{i}_self_attn_q.bias', False),
            (['self_attn.v_proj.bias'], precision, f'layer_{i}_self_attn_v.bias', False),
            (['self_attn.out_proj.weight'], precision, f'layer_{i}_self_attn_output.weights', False),
            (['self_attn.out_proj.bias'], precision, f'layer_{i}_self_attn_output.bias', False),
            (['self_attn_layer_norm.weight'], precision, f'layer_{i}_self_attn_norm.weights', False),
            (['self_attn_layer_norm.bias'], precision, f'layer_{i}_self_attn_norm.bias', False),
        ]
        for layer_prefix in existing_prefixes:
            #Going to add a line here which will differentiate between encoder and decoder fc
            for name_patterns, tensor_precision, output_name, should_transpose in weight_patterns:
                found = False
                for pattern in name_patterns:
                    full_name = layer_prefix + pattern
                    if full_name in state_dict:
                        tensor = state_dict[full_name]
                        if pattern.startswith('attn.Wqkv.') and model_type_str == 'nomic_bert':
                            if tensor.ndim == 1:
                                tensor = tensor.reshape(3, -1)
                            elif tensor.ndim == 2:
                                tensor = tensor.reshape(3, -1, tensor.size(-1))
                            else:
                                raise ValueError(f"Invalid tensor shape: {tensor.shape}")
                            for j, ch in enumerate(['q', 'k', 'v']):
                                channel_output_name = output_name.replace('{channel}', ch)
                                save_tensor_with_header(tensor[j], output_dir / channel_output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                                saved_tensor_full_names.add(full_name)
                            found = True
                            break
                        elif model_type_str == 'nomic_bert' and pattern.startswith('mlp.experts.') and 'bias' not in pattern:
                            num_experts = model_config['num_experts']
                            if tensor.ndim != 2:
                                raise ValueError(f"Invalid tensor shape: {tensor.shape}")
                            tensor = tensor.reshape(num_experts, -1, tensor.size(-1))
                            for expert_idx in range(num_experts):
                                expert_tensor = tensor[expert_idx]
                                expert_output_name = output_name.replace('{channel}', str(expert_idx))
                                save_tensor_with_header(expert_tensor, output_dir / expert_output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                                saved_tensor_full_names.add(full_name)
                            found = True
                            break
                        if model_type_str == 'whisper':
                            temp = layer_prefix[:layer_prefix.find('.')] + "." + output_name
                            save_tensor_with_header(tensor, output_dir / temp, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                        else:
                            save_tensor_with_header(tensor, output_dir / output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                        saved_tensor_full_names.add(full_name)
                        found = True
                        break

                if not found and 'c_attn.weight' in name_patterns[0]:
                    attn_name = layer_prefix + 'attn.c_attn.weight'
                    if attn_name in state_dict:
                        combined_weight = state_dict[attn_name]
                        hidden_size = combined_weight.shape[0]
                        q_weight = combined_weight[:, :hidden_size]
                        k_weight = combined_weight[:, hidden_size:2*hidden_size]
                        v_weight = combined_weight[:, 2*hidden_size:]

                        save_tensor_with_header(q_weight, output_dir / f'layer_{i}_attn_q.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                        save_tensor_with_header(k_weight, output_dir / f'layer_{i}_attn_k.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                        save_tensor_with_header(v_weight, output_dir / f'layer_{i}_attn_v.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                        saved_tensor_full_names.add(attn_name)
                        found = True
    
    if saved_tensor_full_names != set(state_dict.keys()):
        print(f"Warning: Unsaved tensors: {set(state_dict.keys()) - saved_tensor_full_names}")
        
        if not found:
            missing_tensors.append((i, output_name, name_patterns))
    
    if missing_tensors:
        missing_report = output_dir / "missing_weights.txt"
        with open(missing_report, 'w') as fh:
            fh.write("# Missing tensors during conversion\n")
            for layer_idx, output_name, patterns in missing_tensors:
                pattern_list = ', '.join(patterns)
                fh.write(f"layer={layer_idx}, output={output_name}, patterns=[{pattern_list}]\n")
        print(f"Warning: {len(missing_tensors)} tensors were not exported. See {missing_report.name} for details.")

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
        snr_threshold = args.snr_threshold if args else 30.0
        print(f"Processed {quantization_stats['quantized_tensors']} INT8 tensors, {fp16_tensors} FP16 tensors ({low_snr_fallbacks} SNR<{snr_threshold}dB fallbacks)")

    return model_config


def convert_hf_model_weights_vlm(model, output_dir, precision='INT8', args=None):
    quantization_stats = {
        'total_tensors': 0,
        'quantized_tensors': 0,
        'total_parameters': 0,
        'quantized_parameters': 0,
        'mse_values': [],
        'snr_values': [],
        'cos_sim_values': [],
        'saturation_warnings': 0
    }

    state_dict = model.state_dict()
    config = model.config

    tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)

    def _cfg_get(c, key, default=None):
        if c is None:
            return default
        try:
            if isinstance(c, dict):
                return c.get(key, default)
        except Exception:
            pass
        try:
            return getattr(c, key, default)
        except Exception:
            return default

    text_cfg = _cfg_get(config, 'text_config', None)
    vision_cfg = _cfg_get(config, 'vision_config', None)

    text_vocab = _cfg_get(text_cfg, 'vocab_size', _cfg_get(config, 'vocab_size', 0))
    text_hidden = _cfg_get(text_cfg, 'hidden_size', _cfg_get(text_cfg, 'hidden_dim', 0))
    text_num_layers = int(_cfg_get(text_cfg, 'num_hidden_layers', _cfg_get(text_cfg, 'num_layers', 0) or 0))
    text_attention_heads = int(_cfg_get(text_cfg, 'num_attention_heads', 0))
    text_attention_kv_heads = int(_cfg_get(text_cfg, 'num_key_value_heads', _cfg_get(text_cfg, 'num_attention_heads', 0)))
    text_ffn = int(_cfg_get(text_cfg, 'intermediate_size', 0))
    text_context = int(_cfg_get(text_cfg, 'max_position_embeddings', _cfg_get(text_cfg, 'max_sequence_length', 0)))
    text_rope = _cfg_get(text_cfg, 'rope_theta', _cfg_get(config, 'rope_theta', 10000.0))
    text_head_dim = int(_cfg_get(text_cfg, 'head_dim', int(text_hidden // max(1, text_attention_heads))))

    vision_hidden = int(_cfg_get(vision_cfg, 'hidden_size', 0))
    vision_image_size = _cfg_get(vision_cfg, 'image_size', _cfg_get(vision_cfg, 'size', {}).get('longest_edge', 0) if isinstance(_cfg_get(vision_cfg, 'size', {}), dict) else _cfg_get(vision_cfg, 'image_size', 0))
    vision_patch = int(_cfg_get(vision_cfg, 'patch_size', 0))
    vision_heads = int(_cfg_get(vision_cfg, 'num_attention_heads', 0))
    vision_num_layers = int(_cfg_get(vision_cfg, 'num_hidden_layers', _cfg_get(vision_cfg, 'num_layers', 0) or 0))
    num_channels = int(_cfg_get(vision_cfg, 'num_channels', _cfg_get(vision_cfg, 'num_channels', 3)))
    vision_embed_dim = int(vision_hidden)
    visual_tokens_per_img = 0
    try:
        if vision_patch > 0 and vision_image_size > 0:
            per_side = vision_image_size // vision_patch
            visual_tokens_per_img = per_side * per_side
    except Exception:
        visual_tokens_per_img = 0

    pixel_shuffle_factor = int(_cfg_get(config, 'scale_factor', _cfg_get(vision_cfg, 'scale_factor', 1) or 1))
    use_pixel_shuffle = bool(pixel_shuffle_factor > 1)
    use_image_tokens = bool(_cfg_get(config, 'image_token_id', None) is not None)
    use_layout_tags = False

    model_type_str = _cfg_get(text_cfg, 'model_type', None) or _cfg_get(config, 'model_type', '')
    if 'smolvlm' in model_type_str:
        detected_model_type = 'smolvlm'
    else:
        detected_model_type = 'smolvlm'
        print(f"  Warning: Unknown VLM model type '{model_type_str}', defaulting to 'smolvlm'")


    model_config = {
        'vocab_size': int(text_vocab),
        'model_type': detected_model_type,
        'hidden_dim': int(text_hidden),
        'num_layers': int(text_num_layers),
        'attention_heads': int(text_attention_heads),
        'attention_kv_heads': int(text_attention_kv_heads),
        'ffn_intermediate_dim': int(text_ffn),
        'context_length': int(text_context),
        'rope_theta': float(text_rope),
        'attention_head_dim': int(text_head_dim),
        'vision_hidden_size': int(vision_hidden),
        'vision_num_layers': int(vision_num_layers),
        'vision_image_size': int(vision_image_size),
        'vision_patch_size': int(vision_patch),
        'vision_attention_heads': int(vision_heads),
        'vision_embed_dim': int(vision_embed_dim),
        'num_channels': int(num_channels),
        'visual_tokens_per_img': int(visual_tokens_per_img),
        'use_pixel_shuffle': bool(use_pixel_shuffle),
        'pixel_shuffle_factor': int(pixel_shuffle_factor),
        'use_image_tokens': bool(use_image_tokens),
        'use_layout_tags': bool(use_layout_tags),
        'tie_word_embeddings': tie_word_embeddings
    }

    embed_names = ['model.embed_tokens.weight', 'embed_tokens.weight', 'embeddings.weight', 'transformer.wte.weight', 'model.text_model.embed_tokens.weight']
    for name in embed_names:
        if name in state_dict:
            save_tensor_with_header(state_dict[name], output_dir / "token_embeddings.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
            break

    if not tie_word_embeddings:
        output_names = ['lm_head.weight', 'output.weight', 'transformer.lm_head.weight', 'model.text_model.lm_head.weight']
        for name in output_names:
            if name in state_dict:
                tensor = state_dict[name]
                save_tensor_with_header(tensor, output_dir / "output_weight.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                break

    output_norm_names = ['model.norm.weight', 'norm.weight', 'final_layernorm.weight', 'transformer.ln_f.weight', 'model.text_model.norm.weight']
    for name in output_norm_names:
        if name in state_dict:
            tensor = state_dict[name]
            save_tensor_with_header(tensor, output_dir / "output_norm.weights", precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
            break

    vision_items = [
        ('model.vision_model.embeddings.patch_embedding.weight', 'vision_patch_embedding.weights'),
        ('model.vision_model.embeddings.patch_embedding.bias', 'vision_patch_embedding.bias.weights'),
        ('model.vision_model.embeddings.position_embedding.weight', 'vision_position_embedding.weights'),
        ('model.vision_model.post_layernorm.weight', 'vision_post_layernorm.weights'),
        ('model.vision_model.post_layernorm.bias', 'vision_post_layernorm.bias.weights')
    ]
    for key, outname in vision_items:
        if key in state_dict:
            save_tensor_with_header(state_dict[key], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)

        # Detect number of vision encoder layers
        import re
        max_v_idx = -1
        vision_prefix = None
        for k in state_dict.keys():
            # Check vision_tower prefix first (LFM2-VL)
            m = re.search(r'model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.', k)
            if m:
                vision_prefix = 'model.vision_tower.vision_model.encoder.layers.'
                try:
                    idx = int(m.group(1))
                    if idx > max_v_idx:
                        max_v_idx = idx
                except Exception:
                    pass
            # Check model.vision_model prefix
            if not vision_prefix:
                m = re.search(r'model\.vision_model\.encoder\.layers\.(\d+)\.', k)
                if m:
                    vision_prefix = 'model.vision_model.encoder.layers.'
                    try:
                        idx = int(m.group(1))
                        if idx > max_v_idx:
                            max_v_idx = idx
                    except Exception:
                        pass
        
        if not vision_prefix:
            vision_prefix = 'model.vision_model.encoder.layers.'
        
        vision_layers = max_v_idx + 1 if max_v_idx >= 0 else 0

        for i_v in range(vision_layers):
            vpref = f'{vision_prefix}{i_v}.'
            for fname, out in [
                (vpref + 'layer_norm1.weight', f'vision_layer_{i_v}_layer_norm1.weights'),
                (vpref + 'layer_norm1.bias', f'vision_layer_{i_v}_layer_norm1.bias.weights'),
                (vpref + 'layer_norm2.weight', f'vision_layer_{i_v}_layer_norm2.weights'),
                (vpref + 'layer_norm2.bias', f'vision_layer_{i_v}_layer_norm2.bias.weights')
            ]:
                if fname in state_dict:
                    save_tensor_with_header(state_dict[fname], output_dir / out, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    saved_tensor_full_names.add(fname)

            for fname, out in [
                (vpref + 'mlp.fc1.weight', f'vision_layer_{i_v}_ffn_fc1.weights'),
                (vpref + 'mlp.fc1.bias', f'vision_layer_{i_v}_ffn_fc1.bias.weights'),
                (vpref + 'mlp.fc2.weight', f'vision_layer_{i_v}_ffn_fc2.weights'),
                (vpref + 'mlp.fc2.bias', f'vision_layer_{i_v}_ffn_fc2.bias.weights')
            ]:
                if fname in state_dict:
                    save_tensor_with_header(state_dict[fname], output_dir / out, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    saved_tensor_full_names.add(fname)

            for fname, out in [
                (vpref + 'self_attn.q_proj.weight', f'vision_layer_{i_v}_self_attn_q.weights'),
                (vpref + 'self_attn.k_proj.weight', f'vision_layer_{i_v}_self_attn_k.weights'),
                (vpref + 'self_attn.v_proj.weight', f'vision_layer_{i_v}_self_attn_v.weights'),
                (vpref + 'self_attn.out_proj.weight', f'vision_layer_{i_v}_self_attn_out.weights'),
                (vpref + 'self_attn.q_proj.bias', f'vision_layer_{i_v}_self_attn_q.bias.weights'),
                (vpref + 'self_attn.k_proj.bias', f'vision_layer_{i_v}_self_attn_k.bias.weights'),
                (vpref + 'self_attn.v_proj.bias', f'vision_layer_{i_v}_self_attn_v.bias.weights'),
                (vpref + 'self_attn.out_proj.bias', f'vision_layer_{i_v}_self_attn_out.bias.weights')
            ]:
                if fname in state_dict:
                    save_tensor_with_header(state_dict[fname], output_dir / out, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    saved_tensor_full_names.add(fname)

        # Multi-modal projector (LFM2-VL)
        projector_weights = [
            ('model.multi_modal_projector.linear_1.weight', 'projector_linear1.weights'),
            ('model.multi_modal_projector.linear_1.bias', 'projector_linear1.bias.weights'),
            ('model.multi_modal_projector.linear_2.weight', 'projector_linear2.weights'),
            ('model.multi_modal_projector.linear_2.bias', 'projector_linear2.bias.weights'),
            ('model.multi_modal_projector.layer_norm.weight', 'projector_layer_norm.weights'),
            ('model.multi_modal_projector.layer_norm.bias', 'projector_layer_norm.bias.weights'),
        ]
        for key, outname in projector_weights:
            if key in state_dict:
                save_tensor_with_header(state_dict[key], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(key)
        
        # Connector weights
        connector_keys = [
            'model.connector.modality_projection.proj.weight',
            'connector.modality_projection.proj.weight',
            'model.connector.proj.weight',
            'connector.proj.weight'
        ]
        for ck in connector_keys:
            if ck in state_dict:
                save_tensor_with_header(state_dict[ck], output_dir / 'connector_proj.weights', precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(ck)
                break

    num_layers = model_config['num_layers']
    missing_tensors = []
    for i in range(num_layers):
        
        # Support both regular and VLM layer prefixes
        layer_prefixes = [f'model.language_model.layers.{i}.', f'model.text_model.layers.{i}.', 
                          f'model.layers.{i}.', f'layers.{i}.', f'transformer.h.{i}.', f'encoder.layers.{i}.']
        
        layer_prefix = None
        for prefix in layer_prefixes:
            if any(key.startswith(prefix) for key in state_dict.keys()):
                layer_prefix = prefix
                break

        if not layer_prefix:
            continue

        # Conv layers for LFM2 (will be skipped if not present)
        conv_patterns = [
            ('conv.conv.weight', f'layer_{i}_conv_depthwise.weights'),
            ('conv.in_proj.weight', f'layer_{i}_conv_in_proj.weights'),
            ('conv.out_proj.weight', f'layer_{i}_conv_out_proj.weights'),
        ]
        for suffix, outname in conv_patterns:
            fname = layer_prefix + suffix
            if fname in state_dict:
                save_tensor_with_header(state_dict[fname], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(fname)

        weight_patterns = [
            (['self_attn.q_proj.weight', 'attn.q_proj.weight', 'attn.c_attn.weight'], precision, f'layer_{i}_attn_q.weights', False),
            (['self_attn.k_proj.weight', 'attn.k_proj.weight'], precision, f'layer_{i}_attn_k.weights', False),
            (['self_attn.v_proj.weight', 'attn.v_proj.weight'], precision, f'layer_{i}_attn_v.weights', False),
            (['self_attn.out_proj.weight', 'self_attn.o_proj.weight', 'attn.o_proj.weight', 'attn.c_proj.weight'], precision, f'layer_{i}_attn_output.weights', False),
            (['operator_norm.weight', 'input_layernorm.weight', 'ln_1.weight'], precision, f'layer_{i}_input_norm.weights', False),
            (['self_attn.q_norm.weight', 'self_attn.q_layernorm.weight'], precision, f'layer_{i}_attn_q_norm.weights', False),
            (['self_attn.k_norm.weight', 'self_attn.k_layernorm.weight'], precision, f'layer_{i}_attn_k_norm.weights', False),
            (['feed_forward.w1.weight', 'mlp.gate_proj.weight', 'mlp.c_fc.weight'], precision, f'layer_{i}_ffn_gate.weights', False),
            (['feed_forward.w3.weight', 'mlp.up_proj.weight'], precision, f'layer_{i}_ffn_up.weights', False),
            (['feed_forward.w2.weight', 'mlp.down_proj.weight', 'mlp.c_proj.weight'], precision, f'layer_{i}_ffn_down.weights', False),
            (['ffn_norm.weight', 'post_attention_layernorm.weight', 'ln_2.weight'], precision, f'layer_{i}_post_attn_norm.weights', False),
            # Gemma3 specific layer norms 
            (['pre_feedforward_layernorm.weight'], precision, f'layer_{i}_pre_ffn_norm.weights', False),
            (['post_feedforward_layernorm.weight'], precision, f'layer_{i}_post_ffn_norm.weights', False),
            # Nomic BERT specific parameters
            (['attn.Wqkv.bias'], precision, f'layer_{i}_attn_{{channel}}.bias', False),
            (['attn.Wqkv.weight'], precision, f'layer_{i}_attn_{{channel}}.weights', False),
            (['attn.out_proj.bias'], precision, f'layer_{i}_attn_output.bias', False),
            (['attn.out_proj.weight'], precision, f'layer_{i}_attn_output.weights', False),
            (['mlp.fc1.bias'], precision, f'layer_{i}_mlp_fc1.bias', False),
            (['mlp.fc1.weight'], precision, f'layer_{i}_mlp_fc1.weights', False),
            (['mlp.fc2.bias'], precision, f'layer_{i}_mlp_fc2.bias', False),
            (['mlp.fc2.weight'], precision, f'layer_{i}_mlp_fc2.weights', False),
            (['norm1.bias'], precision, f'layer_{i}_norm1.bias', False),
            (['norm1.weight'], precision, f'layer_{i}_norm1.weights', False),
            (['norm2.bias'], precision, f'layer_{i}_norm2.bias', False),
            (['norm2.weight'], precision, f'layer_{i}_norm2.weights', False),
            (['mlp.experts.bias'], precision, f'layer_{i}_mlp_experts.bias', False),
            (['mlp.experts.mlp.w1'], precision, f'layer_{i}_mlp_expert_{{channel}}.mlp1.weights', False),
            (['mlp.experts.mlp.w2'], precision, f'layer_{i}_mlp_expert_{{channel}}.mlp2.weights', True),
            (['mlp.router.layer.weight'], precision, f'layer_{i}_mlp_router.layer.weights', False),
        ]

        for name_patterns, tensor_precision, output_name, should_transpose in weight_patterns:
            found = False
            for pattern in name_patterns:
                full_name = layer_prefix + pattern
                if full_name in state_dict:
                    tensor = state_dict[full_name]
                    if pattern.startswith('attn.Wqkv.') and model_type_str == 'nomic_bert':
                        if tensor.ndim == 1:
                            tensor = tensor.reshape(3, -1)
                        elif tensor.ndim == 2:
                            tensor = tensor.reshape(3, -1, tensor.size(-1))
                        else:
                            raise ValueError(f"Invalid tensor shape: {tensor.shape}")
                        for j, ch in enumerate(['q', 'k', 'v']):
                            channel_output_name = output_name.replace('{channel}', ch)
                            save_tensor_with_header(tensor[j], output_dir / channel_output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                            saved_tensor_full_names.add(full_name)
                        found = True
                        break
                    elif model_type_str == 'nomic_bert' and pattern.startswith('mlp.experts.') and 'bias' not in pattern:
                        num_experts = model_config['num_experts']
                        if tensor.ndim != 2:
                            raise ValueError(f"Invalid tensor shape: {tensor.shape}")
                        tensor = tensor.reshape(num_experts, -1, tensor.size(-1))
                        for expert_idx in range(num_experts):
                            expert_tensor = tensor[expert_idx]
                            expert_output_name = output_name.replace('{channel}', str(expert_idx))
                            save_tensor_with_header(expert_tensor, output_dir / expert_output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                            saved_tensor_full_names.add(full_name)
                        found = True
                        break
                    save_tensor_with_header(tensor, output_dir / output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    saved_tensor_full_names.add(full_name)
                    found = True
                    break

            if not found and 'c_attn.weight' in name_patterns[0]:
                attn_name = layer_prefix + 'attn.c_attn.weight'
                if attn_name in state_dict:
                    combined_weight = state_dict[attn_name]
                    hidden_size = combined_weight.shape[0]
                    q_weight = combined_weight[:, :hidden_size]
                    k_weight = combined_weight[:, hidden_size:2*hidden_size]
                    v_weight = combined_weight[:, 2*hidden_size:]

                    save_tensor_with_header(q_weight, output_dir / f'layer_{i}_attn_q.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    save_tensor_with_header(k_weight, output_dir / f'layer_{i}_attn_k.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    save_tensor_with_header(v_weight, output_dir / f'layer_{i}_attn_v.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    saved_tensor_full_names.add(attn_name)
                    found = True
    
    if saved_tensor_full_names != set(state_dict.keys()):
        print(f"Warning: Unsaved tensors: {set(state_dict.keys()) - saved_tensor_full_names}")
        
        if not found:
            missing_tensors.append((i, output_name, name_patterns))
    
    if missing_tensors:
        missing_report = output_dir / "missing_weights.txt"
        with open(missing_report, 'w') as fh:
            fh.write("# Missing tensors during conversion\n")
            for layer_idx, output_name, patterns in missing_tensors:
                pattern_list = ', '.join(patterns)
                fh.write(f"layer={layer_idx}, output={output_name}, patterns=[{pattern_list}]\n")
        print(f"Warning: {len(missing_tensors)} tensors were not exported. See {missing_report.name} for details.")

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
        snr_threshold = args.snr_threshold if args else 30.0
        print(f"Processed {quantization_stats['quantized_tensors']} INT8 tensors, {fp16_tensors} FP16 tensors ({low_snr_fallbacks} SNR<{snr_threshold}dB fallbacks)")

    return model_config

def convert_hf_tokenizer(tokenizer, output_dir, token=None):
    is_sentencepiece = False
    tokenizer_model_path = None

    if hasattr(tokenizer, 'vocab_file'):
        vocab_file = tokenizer.vocab_file
        if vocab_file and vocab_file.endswith('.model'):
            is_sentencepiece = True
            tokenizer_model_path = vocab_file

    if not is_sentencepiece and hasattr(tokenizer, 'sp_model'):
        is_sentencepiece = True
        if hf_hub_download:
            try:
                tokenizer_model_path = hf_hub_download(
                    repo_id=tokenizer.name_or_path,
                    filename="tokenizer.model",
                    token=token,
                )
            except Exception:
                pass

    if is_sentencepiece and tokenizer_model_path:
        import shutil
        dest_path = output_dir / "tokenizer.model"
        try:
            shutil.copy2(tokenizer_model_path, dest_path)
            print(f"  Copied SentencePiece model to {dest_path.name}")
        except Exception as e:
            print(f"  Warning: Could not copy tokenizer.model: {e}")

    tokenizer_json_data = {}
    tokenizer_json_path = output_dir / "tokenizer.json"
    try:
        tokenizer.save_pretrained(output_dir)
        if tokenizer_json_path.exists():
            with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                tokenizer_json_data = json.load(f)
    except Exception as e:
        print(f"  Warning: Could not save tokenizer JSON: {e}")

    vocab = tokenizer.get_vocab()

    id_to_token = [""] * len(vocab)
    for token, token_id in vocab.items():
        if token_id < len(id_to_token):
            id_to_token[token_id] = token

    vocab_output = output_dir / "vocab.txt"

    if is_sentencepiece:
        with open(vocab_output, 'w', encoding='utf-8') as f:
            for token_id, token in enumerate(id_to_token):
                if token:
                    f.write(f"{token_id}\t{token}\n")
        print(f"  Saved SentencePiece vocabulary (ID\\ttoken format)")
    else:
        with open(vocab_output, 'w', encoding='utf-8') as f:
            for token in id_to_token:
                f.write(token + '\n')
        print(f"  Saved BPE vocabulary (line-by-line format)")
    
    
    merges_output = output_dir / "merges.txt"

    def write_merges_file(merges_list):
        with open(merges_output, 'w', encoding='utf-8', newline='') as f:
            f.write("#version: 0.2\n")
            for merge in merges_list:
                f.write(f"{' '.join(merge)}\n")

    merges_written = False

    if not is_sentencepiece and tokenizer_json_data:
        merges_from_json = tokenizer_json_data.get("model", {}).get("merges", []) or []
        write_merges_file(merges_from_json)
        merges_written = True

    if not merges_written and hf_hub_download:
        try:
            import shutil
            merges_file = hf_hub_download(repo_id=tokenizer.name_or_path, filename="merges.txt", token=token)
            shutil.copy2(merges_file, merges_output)
            merges_written = True
        except Exception:
            pass

    if not merges_written and hasattr(tokenizer, 'backend_tokenizer') and tokenizer.backend_tokenizer:
        backend = tokenizer.backend_tokenizer
        merges = []

        if hasattr(backend, 'model'):
            model = backend.model
            if hasattr(model, 'merges'):
                merges = model.merges

        write_merges_file(merges)
        merges_written = True

    if not merges_written:
        write_merges_file([])
    
    
    special_tokens = {}
    special_token_ids = {}
    
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_token_ids['eos_token_id'] = tokenizer.eos_token_id
        special_tokens[tokenizer.eos_token_id] = tokenizer.eos_token or "<|endoftext|>"
    
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        special_token_ids['pad_token_id'] = tokenizer.pad_token_id
        special_tokens[tokenizer.pad_token_id] = tokenizer.pad_token or "<|endoftext|>"
    
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_token_ids['bos_token_id'] = tokenizer.bos_token_id
        special_tokens[tokenizer.bos_token_id] = tokenizer.bos_token or "<|startoftext|>"
    
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        special_token_ids['unk_token_id'] = tokenizer.unk_token_id
        special_tokens[tokenizer.unk_token_id] = tokenizer.unk_token or "<|unknown|>"
    
    additional_special_tokens = []
    if hasattr(tokenizer, 'additional_special_tokens'):
        for token in tokenizer.additional_special_tokens or []:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                special_tokens[token_id] = token
                additional_special_tokens.append({"token": token, "id": token_id})

    model_type = getattr(tokenizer, 'name_or_path', '').lower()
    if 'gemma' in model_type:
        gemma_special_tokens = {
            '<start_of_turn>': None,
            '<end_of_turn>': None,
            '<start_of_image>': None,
            '<end_of_image>': None
        }

        vocab = tokenizer.get_vocab()
        for token_str in gemma_special_tokens.keys():
            if token_str in vocab:
                token_id = vocab[token_str]
                gemma_special_tokens[token_str] = token_id
                special_tokens[token_id] = token_str
                print(f"    Found Gemma special token: {token_str} (ID: {token_id})")

        missing_tokens = [k for k, v in gemma_special_tokens.items() if v is None]
        if missing_tokens and is_sentencepiece and tokenizer_model_path:
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
                for token_str in missing_tokens:
                    token_id = sp.piece_to_id(token_str)
                    if token_id != sp.unk_id():
                        gemma_special_tokens[token_str] = token_id
                        special_tokens[token_id] = token_str
                        print(f"    Found Gemma special token via SentencePiece: {token_str} (ID: {token_id})")
            except Exception as e:
                print(f"    Warning: Could not check SentencePiece for Gemma tokens: {e}")

        if gemma_special_tokens['<start_of_turn>'] is None:
            hardcoded_ids = {
                '<start_of_turn>': 105,
                '<end_of_turn>': 106
            }
            for token_str, token_id in hardcoded_ids.items():
                if token_str in gemma_special_tokens and gemma_special_tokens[token_str] is None:
                    if token_id not in special_tokens:
                        gemma_special_tokens[token_str] = token_id
                        special_tokens[token_id] = token_str
                        print(f"    Using hardcoded Gemma special token: {token_str} (ID: {token_id})")
    
    chat_template_data = {}
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        chat_template_output = output_dir / "chat_template.jinja2"
        with open(chat_template_output, 'w', encoding='utf-8') as f:
            f.write(tokenizer.chat_template)
        chat_template_data["chat_template"] = tokenizer.chat_template
    
    tokenizer_full_config = {}
    added_tokens_decoder = {}
    tool_tokens = {}
    
    try:
        config_path = None
        if hasattr(tokenizer, 'name_or_path') and hf_hub_download:
            try:
                config_path = hf_hub_download(repo_id=tokenizer.name_or_path, filename="tokenizer_config.json", token=token)
                with open(config_path, 'r') as f:
                    tokenizer_full_config = json.load(f)
                    
                    if 'chat_template' in tokenizer_full_config and not chat_template_data:
                        chat_template_output = output_dir / "chat_template.jinja2"
                        with open(chat_template_output, 'w', encoding='utf-8') as f:
                            f.write(tokenizer_full_config['chat_template'])
                        chat_template_data["chat_template"] = tokenizer_full_config['chat_template']
                    
                    if 'added_tokens_decoder' in tokenizer_full_config:
                        added_tokens_decoder = tokenizer_full_config['added_tokens_decoder']
                        
                        print("  Extracting special tokens from tokenizer_config.json...")
                        for token_id_str, token_info in added_tokens_decoder.items():
                            content = token_info.get('content', '')
                            token_id = int(token_id_str)
                            
                            tool_related = ['<tool_call>', '</tool_call>', 
                                          '<tool_response>', '</tool_response>',
                                          '<tools>', '</tools>',
                                          '<think>', '</think>']
                            
                            if any(x == content for x in tool_related):
                                tool_tokens[token_id] = token_info
                                print(f"    Found tool token: {content} (ID: {token_id})")
                                special_tokens[token_id] = content
                                
            except Exception as e:
                print(f"  Note: Could not load full tokenizer config: {e}")
                pass
    except Exception:
        pass
    
    
    special_tokens_output = output_dir / "special_tokens.json"
    with open(special_tokens_output, 'w', encoding='utf-8') as f:
        json.dump({
            **special_token_ids,
            "vocab_size": len(vocab),
            "model_max_length": getattr(tokenizer, 'model_max_length', 131072),
            "special_tokens": special_tokens,
            "additional_special_tokens": additional_special_tokens,
            **chat_template_data
        }, f, indent=2, ensure_ascii=False)
    
    
    tokenizer_config_output = output_dir / "tokenizer_config.txt"
    with open(tokenizer_config_output, 'w') as f:
        f.write(f"vocab_size={len(vocab)}\n")
        for key, value in special_token_ids.items():
            f.write(f"{key}={value}\n")
        f.write(f"model_max_length={getattr(tokenizer, 'model_max_length', 131072)}\n")

        if is_sentencepiece:
            f.write("tokenizer_type=sentencepiece\n")
        else:
            f.write("tokenizer_type=bpe\n")

        if chat_template_data:
            f.write("has_chat_template=true\n")
        else:
            f.write("has_chat_template=false\n")
        if len(tool_tokens) > 0:
            f.write(f"has_tool_support=true\n")
            f.write(f"tool_token_count={len(tool_tokens)}\n")

def convert_processors(processor, model_name, output_dir, token=None):
    if processor is None:
        return

    try:
        if hasattr(processor, 'save_pretrained'):
            processor.save_pretrained(str(output_dir))
            print("  Saved processor with save_pretrained()")
            return
    except Exception as e:
        print(f"  Warning: processor.save_pretrained failed: {e}")

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        print("  Note: huggingface_hub not available, skipping processor file download")
        return

    candidate_files = [
        'preprocessor_config.json',
        'processor_config.json'
    ]

    for fname in candidate_files:
        try:
            path = hf_hub_download(repo_id=model_name, filename=fname, token=token)
            import shutil
            shutil.copy2(path, output_dir / fname)
            print(f"  Downloaded and saved {fname}")
        except Exception:
            pass
    
def convert_hf_to_cactus_vlm(model_name, output_dir, precision='INT8', cache_dir=None, args=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = getattr(args, 'token', None) if args else None

    print(f"Converting VLM {model_name} to {precision}...")

    try:
        missing_deps = []
        try:
            from PIL import Image  # Pillow
        except Exception:
            missing_deps.append('Pillow')
        try:
            import num2words
        except Exception:
            missing_deps.append('num2words')
        try:
            import torchvision
        except Exception:
            missing_deps.append('torchvision')

        if missing_deps:
            print(f"Error: Missing packages required for VLM models: {', '.join(missing_deps)}")
            print(f"Install with: pip install {' '.join(missing_deps)}")
            sys.exit(1)

        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            token=token,
        )

        tokenizer = None
        try:
            tokenizer = processor.tokenizer
        except Exception:
            tokenizer = None

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                token=token,
            )
        try:
            convert_processors(processor, model_name, output_dir, token=token)
        except Exception as e:
            print(f"  Warning: convert_processors failed: {e}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    config = convert_hf_model_weights_vlm(model, output_dir, precision, args)

    if precision == 'INT8':
        config['precision'] = "FP16"
    else:
        config['precision'] = precision

    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        for key, value in config.items():
            if isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)
            f.write(f"{key}={value_str}\n")

    convert_hf_tokenizer(tokenizer, output_dir, token=token)
    print(f"\nConversion complete: {output_dir}")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _pick_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # fall back to fp16 on GPU
        return torch.float16
    return torch.float32

def _is_lfm2_vl(model_name: str, cfg) -> bool:
    if getattr(cfg, "model_type", None) == "lfm2-vl":
        return True
    # also catch common repo names
    name = (model_name or "").lower()
    return "lfm2-vl" in name

def _vision_weight_sanity(model):
    ok = True
    vt = getattr(model, "vision_tower", None)
    try:
        emb = vt.vision_model.embeddings
        w_mean = emb.patch_embedding.weight.detach().abs().mean().item()
        p_mean = emb.position_embedding.weight.detach().abs().mean().item()
        print(f"[sanity] |patch W| mean={w_mean:.5f} |pos W| mean={p_mean:.5f}")
        # heuristics: randomly init’d weights often have small similar scales;
        # pretrained pos tables usually have noticeably different stats
        if w_mean < 1e-3 or p_mean < 1e-3:
            ok = False
    except Exception:
        pass
    return ok

def convert_hf_to_cactus_vlm(model_name, output_dir, precision='INT8', cache_dir=None, args=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = getattr(args, 'token', None) if args else None

    print(f"Converting VLM {model_name} to {precision}...")

    try:
        missing_deps = []
        try:
            from PIL import Image  # Pillow
        except Exception:
            missing_deps.append('Pillow')
        try:
            import num2words
        except Exception:
            missing_deps.append('num2words')
        try:
            import torchvision
        except Exception:
            missing_deps.append('torchvision')

        if missing_deps:
            print(f"Error: Missing packages required for VLM models: {', '.join(missing_deps)}")
            print(f"Install with: pip install {' '.join(missing_deps)}")
            sys.exit(1)

        # Load processor (this is correct for LFM2-VL)
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )

        # Decide the correct model class
        cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, token=token)
        dtype = _pick_dtype()
        print(f"[info] selected torch_dtype={dtype}")

        if _is_lfm2_vl(model_name, cfg) and Lfm2VlForConditionalGeneration is not None:
            print("[info] Detected LFM2-VL checkpoint; loading Lfm2VlForConditionalGeneration")
            model = Lfm2VlForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                token=token,
            )
        else:
            from transformers import AutoModelForImageTextToText
            print("[warn] Non-LFM2-VL model; using AutoModelForImageTextToText (may re-init heads)")
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                token=token,
            )

        # Optional tokenizer extraction (AutoProcessor usually provides it)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                token=token,
            )

        if _is_lfm2_vl(model_name, cfg):
            ok = _vision_weight_sanity(model)
            if not ok:
                print("[error] Vision embeddings look randomly initialized. "
                      "Double-check model class/ckpt or set trust_remote_code=True.")
                sys.exit(1)

        # Convert processor assets
        try:
            convert_processors(processor, model_name, output_dir, token=token)
        except Exception as e:
            print(f"  Warning: convert_processors failed: {e}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Proceed to weight export / quantization
    config = convert_hf_model_weights(model, output_dir, precision, args)

    # Preserve your original precision field behavior
    if precision == 'INT8':
        config['precision'] = "FP16"
    else:
        config['precision'] = precision

    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        for key, value in config.items():
            if isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)
            f.write(f"{key}={value_str}\n")

    convert_hf_tokenizer(tokenizer, output_dir, token=token)
    print(f"\nConversion complete: {output_dir}")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def convert_hf_to_cactus(model_name, output_dir, precision='INT8', cache_dir=None, args=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = getattr(args, 'token', None) if args else None

    if 'vl' in str(model_name).lower() or 'vlm' in str(model_name).lower():
        return convert_hf_to_cactus_vlm(model_name, output_dir, precision, cache_dir, args)

    print(f"Converting {model_name} to {precision}...")

    if 'whisper' in str(model_name).lower():
        tokenizer = tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )

        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
        )
    
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                trust_remote_code=True,
                token=token,
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    token=token,
                )
            except ValueError:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    token=token,
                )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    config = convert_hf_model_weights(model, output_dir, precision, args)
    model_name_l = str(model_name).lower()
    if 'extract' in model_name_l:
        config['model_variant'] = 'extract'
    elif 'vlm' in model_name_l:
        config['model_variant'] = 'vlm'
    elif 'rag' in model_name_l:
        config['model_variant'] = 'rag'
    else:
        config.setdefault('model_variant', 'default')

    if precision == 'INT8':
        config['precision'] = "FP16"
    else:
        config['precision'] = precision
    
    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}={format_config_value(value)}\n")
    
    convert_hf_tokenizer(tokenizer, output_dir, token=token)
    print(f"\nConversion complete: {output_dir}")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_parser():
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace models to Cactus format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('model_name', help='HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")')
    parser.add_argument('output_dir', help='Directory to write converted files')
    parser.add_argument('--precision', choices=['INT8', 'FP16', 'FP32'], default='INT8',
                       help='Quantization precision')
    parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    parser.add_argument('--token', type=str, help='HuggingFace API token for gated models (or set HF_TOKEN env var)')

    quant_group = parser.add_argument_group('Quantization Parameters')
    quant_group.add_argument('--snr-threshold', type=float, default=20.0,
                            help='Minimum SNR (dB) for INT8 quantization, fallback to FP32 below this')
    quant_group.add_argument('--saturation-threshold', type=float, default=0.01,
                            help='Saturation threshold for outlier clipping (0.0-1.0)')
    quant_group.add_argument('--outlier-percentile', type=float, default=0.01,
                            help='Percentile for outlier detection (0.0-50.0)')
    quant_group.add_argument('--sigma-multiplier', type=float, default=3.5,
                            help='Standard deviation multiplier for range clipping')
    quant_group.add_argument('--range-threshold', type=float, default=0.5,
                            help='Minimum range preservation ratio (0.0-1.0)')
    quant_group.add_argument('--saturation-warning-threshold', type=float, default=0.1,
                            help='Saturation percentage threshold for warnings')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    if args.precision not in ['INT8', 'FP16', 'FP32']:
        print(f"Error: Invalid precision '{args.precision}'. Must be INT8, FP16, or FP32")
        sys.exit(1)
    
    convert_hf_to_cactus(args.model_name, args.output_dir, args.precision, args.cache_dir, args)
