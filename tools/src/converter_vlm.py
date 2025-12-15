import re
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from .tensor_io import save_tensor_with_header, create_quantization_stats, print_quantization_summary
from .config_utils import cfg_get, is_lfm2_vl, pick_torch_dtype, vision_weight_sanity_check
from .weight_patterns import (
    EMBED_NAMES, OUTPUT_NAMES, OUTPUT_NORM_NAMES,
    VISION_ITEMS, PROJECTOR_WEIGHTS, CONNECTOR_KEYS,
    get_layer_weight_patterns, get_vision_layer_weights
)


def convert_hf_model_weights_vlm(model, output_dir, precision='INT8', args=None):
    """Convert VLM (Vision-Language Model) weights to Cactus binary format."""
    quantization_stats = create_quantization_stats()

    state_dict = model.state_dict()
    config = model.config
    saved_tensor_full_names = set()

    tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)

    text_cfg = cfg_get(config, 'text_config', None)
    vision_cfg = cfg_get(config, 'vision_config', None)

    text_vocab = cfg_get(text_cfg, 'vocab_size', cfg_get(config, 'vocab_size', 0))
    text_hidden = cfg_get(text_cfg, 'hidden_size', cfg_get(text_cfg, 'hidden_dim', 0))
    text_num_layers = int(cfg_get(text_cfg, 'num_hidden_layers', cfg_get(text_cfg, 'num_layers', 0) or 0))
    text_attention_heads = int(cfg_get(text_cfg, 'num_attention_heads', 0))
    text_attention_kv_heads = int(cfg_get(text_cfg, 'num_key_value_heads', cfg_get(text_cfg, 'num_attention_heads', 0)))
    text_ffn = int(cfg_get(text_cfg, 'intermediate_size', 0))
    text_context = int(cfg_get(text_cfg, 'max_position_embeddings', cfg_get(text_cfg, 'max_sequence_length', 0)))
    text_rope = cfg_get(text_cfg, 'rope_theta', cfg_get(config, 'rope_theta', 10000.0))
    text_head_dim = int(cfg_get(text_cfg, 'head_dim', int(text_hidden // max(1, text_attention_heads))))

    vision_hidden = int(cfg_get(vision_cfg, 'hidden_size', 0))
    vision_image_size = cfg_get(vision_cfg, 'image_size', cfg_get(vision_cfg, 'size', {}).get('longest_edge', 0) if isinstance(cfg_get(vision_cfg, 'size', {}), dict) else cfg_get(vision_cfg, 'image_size', 0))
    vision_patch = int(cfg_get(vision_cfg, 'patch_size', 0))
    vision_heads = int(cfg_get(vision_cfg, 'num_attention_heads', 0))
    vision_num_layers = int(cfg_get(vision_cfg, 'num_hidden_layers', cfg_get(vision_cfg, 'num_layers', 0) or 0))
    num_channels = int(cfg_get(vision_cfg, 'num_channels', cfg_get(vision_cfg, 'num_channels', 3)))
    vision_embed_dim = int(vision_hidden)
    visual_tokens_per_img = 0
    try:
        if vision_patch > 0 and vision_image_size > 0:
            per_side = vision_image_size // vision_patch
            visual_tokens_per_img = per_side * per_side
    except Exception:
        visual_tokens_per_img = 0

    pixel_shuffle_factor = int(cfg_get(config, 'scale_factor', cfg_get(vision_cfg, 'scale_factor', 1) or 1))
    use_pixel_shuffle = bool(pixel_shuffle_factor > 1)
    use_image_tokens = bool(cfg_get(config, 'image_token_id', None) is not None)
    use_layout_tags = False

    model_type_str = cfg_get(text_cfg, 'model_type', None) or cfg_get(config, 'model_type', '')
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

        for key, outname in PROJECTOR_WEIGHTS:
            if key in state_dict:
                save_tensor_with_header(state_dict[key], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(key)

        for ck in CONNECTOR_KEYS:
            if ck in state_dict:
                save_tensor_with_header(state_dict[ck], output_dir / 'connector_proj.weights', precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(ck)
                break

    num_layers = model_config['num_layers']
    missing_tensors = []
    for i in range(num_layers):

        layer_prefixes = [f'model.language_model.layers.{i}.', f'model.text_model.layers.{i}.',
                          f'model.layers.{i}.', f'layers.{i}.', f'transformer.h.{i}.', f'encoder.layers.{i}.']

        layer_prefix = None
        for prefix in layer_prefixes:
            if any(key.startswith(prefix) for key in state_dict.keys()):
                layer_prefix = prefix
                break

        if not layer_prefix:
            continue

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
            (['pre_feedforward_layernorm.weight'], precision, f'layer_{i}_pre_ffn_norm.weights', False),
            (['post_feedforward_layernorm.weight'], precision, f'layer_{i}_post_ffn_norm.weights', False),
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

    print_quantization_summary(quantization_stats, args)

    return model_config


def convert_processors(processor, model_name, output_dir, token=None):
    """Save VLM processor config files to the output directory."""
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
