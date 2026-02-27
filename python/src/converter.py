import os
import re
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from .tensor_io import save_tensor_with_header, create_quantization_stats, print_quantization_summary
from .config_utils import cfg_get, detect_model_type, extract_base_config, extract_vision_config, extract_lfm2_config, is_vlm_model, extract_moonshine_config
from .weight_patterns import (
    EMBED_NAMES, OUTPUT_NAMES, OUTPUT_NORM_NAMES, LAYER_PREFIXES,
    VISION_ITEMS, PROJECTOR_WEIGHTS, WHISPER_GLOBAL_WEIGHTS, MOONSHINE_GLOBAL_WEIGHTS,
    get_layer_weight_patterns, get_vision_layer_weights
)


def _find_first_key(state_dict, candidates):
    for key in candidates:
        if key in state_dict:
            return key
    return None


def convert_hf_model_weights(model, output_dir, precision='INT8', args=None):
    """Convert HuggingFace model weights to Cactus binary format."""
    quantization_stats = create_quantization_stats()

    state_dict = model.state_dict()
    root_config = model.config
    saved_tensor_full_names = set()

    text_config = cfg_get(root_config, 'text_config', None)
    vision_config = cfg_get(root_config, 'vision_config', None)
    is_vlm = text_config is not None or vision_config is not None

    config = text_config if text_config is not None else root_config

    model_type_str = cfg_get(config, 'model_type', cfg_get(root_config, 'model_type', '')).lower()
    tie_word_embeddings = cfg_get(config, 'tie_word_embeddings', cfg_get(root_config, 'tie_word_embeddings', None))
    if tie_word_embeddings is None:
        # HF snapshots for lfm2_moe may omit this field; runtime expects tied embeddings by default.
        tie_word_embeddings = (model_type_str == 'lfm2_moe')
    else:
        tie_word_embeddings = bool(tie_word_embeddings)

    detected_model_type = detect_model_type(config, root_config, output_dir)

    model_config = extract_base_config(config, root_config)
    model_config['tie_word_embeddings'] = tie_word_embeddings
    model_config['model_type'] = detected_model_type

    if is_vlm and vision_config is not None:
        model_config.update(extract_vision_config(root_config, vision_config))

    if detected_model_type == 'lfm2':
        model_config.update(extract_lfm2_config(config))
    elif detected_model_type == 'moonshine':
        model_config.update(extract_moonshine_config(config))
    elif detected_model_type == 'parakeet':
        encoder_cfg = cfg_get(config, 'encoder_config', None)
        if encoder_cfg is None:
            raise ValueError("Parakeet conversion requires encoder_config in model config")

        hidden_dim = int(cfg_get(encoder_cfg, 'hidden_size', 0))
        num_layers = int(cfg_get(encoder_cfg, 'num_hidden_layers', 0))
        attention_heads = int(cfg_get(encoder_cfg, 'num_attention_heads', 0))
        attention_kv_heads = int(cfg_get(encoder_cfg, 'num_key_value_heads', attention_heads))
        head_dim = int(hidden_dim // max(1, attention_heads))
        layer_norm_eps = cfg_get(encoder_cfg, 'layer_norm_eps', cfg_get(encoder_cfg, 'norm_eps', 1e-5))
        if layer_norm_eps is None:
            layer_norm_eps = 1e-5
        rope_theta = cfg_get(encoder_cfg, 'rope_theta', 0.0)
        if rope_theta is None:
            rope_theta = 0.0

        model_config.update({
            'vocab_size': int(cfg_get(config, 'vocab_size', cfg_get(root_config, 'vocab_size', 0))),
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'attention_heads': attention_heads,
            'attention_kv_heads': attention_kv_heads,
            'attention_head_dim': head_dim,
            'ffn_intermediate_dim': int(cfg_get(encoder_cfg, 'intermediate_size', 0)),
            'context_length': int(cfg_get(encoder_cfg, 'max_position_embeddings', 0)),
            'layer_norm_eps': float(layer_norm_eps),
            'rope_theta': float(rope_theta),
            'conv_kernel_size': int(cfg_get(encoder_cfg, 'conv_kernel_size', 9)),
            'subsampling_conv_kernel_size': int(cfg_get(encoder_cfg, 'subsampling_conv_kernel_size', 3)),
            'subsampling_conv_stride': int(cfg_get(encoder_cfg, 'subsampling_conv_stride', 2)),
            'subsampling_conv_channels': int(cfg_get(encoder_cfg, 'subsampling_conv_channels', 256)),
            'subsampling_factor': int(cfg_get(encoder_cfg, 'subsampling_factor', 8)),
            'num_mel_bins': int(cfg_get(encoder_cfg, 'num_mel_bins', 80)),
            'pad_token_id': int(cfg_get(config, 'pad_token_id', cfg_get(root_config, 'pad_token_id', 0))),
            'encoder_hidden_act': cfg_get(encoder_cfg, 'hidden_act', 'silu'),
        })

    num_layers = model_config['num_layers']

    embedding_found = False
    for name in EMBED_NAMES:
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
        for name, save_name in WHISPER_GLOBAL_WEIGHTS:
            if name in state_dict:
                save_tensor_with_header(state_dict[name], output_dir / save_name, precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)
        embedding_found = True

    elif model_type_str == 'moonshine':
        for name, save_name in MOONSHINE_GLOBAL_WEIGHTS:
            if name in state_dict:
                tensor = state_dict[name]
                if name == 'model.encoder.conv2.weight':
                    tensor = tensor.permute(1, 2, 0).contiguous()
                
                save_tensor_with_header(tensor, output_dir / save_name, precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)
        embedding_found = True
        model_config['dec_hidden_act'] = config.decoder_hidden_act
        model_config['enc_hidden_act'] = config.encoder_hidden_act
        model_config['num_encoder_layers'] = config.encoder_num_hidden_layers
        model_config['num_decoder_layers'] = config.decoder_num_hidden_layers

    if embedding_found:
        embedding_norm_names = {'emb_ln.weight': 'embedding_layernorm.weight', 'emb_ln.bias': 'embedding_layernorm.bias'}
        for name, file_name in embedding_norm_names.items():
            if name in state_dict:
                save_tensor_with_header(state_dict[name], output_dir / file_name, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)

    if tie_word_embeddings:
        src = output_dir / "token_embeddings.weights"
        dst = output_dir / "output_weight.weights"
        if src.exists():
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        else:
            print(f"Warning: tie_word_embeddings is True but {src} not found")
        if embedding_found:
            for name in OUTPUT_NAMES:
                if name in state_dict:
                    saved_tensor_full_names.add(name)
    else:
        for name in OUTPUT_NAMES:
            if name in state_dict:
                tensor = state_dict[name]
                save_tensor_with_header(tensor, output_dir / "output_weight.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)
                break

    for name in OUTPUT_NORM_NAMES:
        if name in state_dict:
            tensor = state_dict[name]
            save_tensor_with_header(tensor, output_dir / "output_norm.weights", precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
            saved_tensor_full_names.add(name)
            break

    if is_vlm:
        for key, outname in VISION_ITEMS:
            if key in state_dict:
                save_tensor_with_header(state_dict[key], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(key)

        for key, outname in PROJECTOR_WEIGHTS:
            if key in state_dict:
                save_tensor_with_header(state_dict[key], output_dir / outname, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(key)

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
            vision_layer_weights = get_vision_layer_weights(i_v, vpref)
            for fname, out in vision_layer_weights:
                if fname in state_dict:
                    save_tensor_with_header(state_dict[fname], output_dir / out, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                    saved_tensor_full_names.add(fname)
    missing_tensors = []
    if detected_model_type == 'parakeet':
        global_mappings = [
            (['encoder.subsampling.layers.0.weight'], 'subsampling_conv0_weight.weights'),
            (['encoder.subsampling.layers.0.bias'], 'subsampling_conv0_bias.bias'),
            (['encoder.subsampling.layers.2.weight'], 'subsampling_depthwise1_weight.weights'),
            (['encoder.subsampling.layers.2.bias'], 'subsampling_depthwise1_bias.bias'),
            (['encoder.subsampling.layers.3.weight'], 'subsampling_pointwise1_weight.weights'),
            (['encoder.subsampling.layers.3.bias'], 'subsampling_pointwise1_bias.bias'),
            (['encoder.subsampling.layers.5.weight'], 'subsampling_depthwise2_weight.weights'),
            (['encoder.subsampling.layers.5.bias'], 'subsampling_depthwise2_bias.bias'),
            (['encoder.subsampling.layers.6.weight'], 'subsampling_pointwise2_weight.weights'),
            (['encoder.subsampling.layers.6.bias'], 'subsampling_pointwise2_bias.bias'),
            (['encoder.subsampling.linear.weight'], 'subsampling_linear_weight.weights'),
            (['encoder.subsampling.linear.bias'], 'subsampling_linear_bias.bias'),
            (['ctc_head.weight'], 'ctc_head_weight.weights'),
            (['ctc_head.bias'], 'ctc_head_bias.bias'),
        ]

        for candidate_keys, out_name in global_mappings:
            key = _find_first_key(state_dict, candidate_keys)
            if key is None:
                missing_tensors.append((-1, out_name, candidate_keys))
                continue
            save_tensor_with_header(
                state_dict[key], output_dir / out_name, precision, transpose=False,
                stats_tracker=quantization_stats, args=args, model_type=detected_model_type
            )
            saved_tensor_full_names.add(key)

        layer_mappings = [
            ('feed_forward1.linear1.weight', 'ff1_linear1.weights'),
            ('feed_forward1.linear1.bias', 'ff1_linear1.bias'),
            ('feed_forward1.linear2.weight', 'ff1_linear2.weights'),
            ('feed_forward1.linear2.bias', 'ff1_linear2.bias'),
            ('feed_forward2.linear1.weight', 'ff2_linear1.weights'),
            ('feed_forward2.linear1.bias', 'ff2_linear1.bias'),
            ('feed_forward2.linear2.weight', 'ff2_linear2.weights'),
            ('feed_forward2.linear2.bias', 'ff2_linear2.bias'),
            ('self_attn.q_proj.weight', 'self_attn_q.weights'),
            ('self_attn.q_proj.bias', 'self_attn_q.bias'),
            ('self_attn.k_proj.weight', 'self_attn_k.weights'),
            ('self_attn.k_proj.bias', 'self_attn_k.bias'),
            ('self_attn.v_proj.weight', 'self_attn_v.weights'),
            ('self_attn.v_proj.bias', 'self_attn_v.bias'),
            ('self_attn.o_proj.weight', 'self_attn_output.weights'),
            ('self_attn.o_proj.bias', 'self_attn_output.bias'),
            ('self_attn.relative_k_proj.weight', 'self_attn_relative_k.weights'),
            ('self_attn.bias_u', 'self_attn_bias_u.weights'),
            ('self_attn.bias_v', 'self_attn_bias_v.weights'),
            ('conv.pointwise_conv1.weight', 'conv_pointwise1.weights'),
            ('conv.pointwise_conv1.bias', 'conv_pointwise1.bias'),
            ('conv.depthwise_conv.weight', 'conv_depthwise.weights'),
            ('conv.depthwise_conv.bias', 'conv_depthwise.bias'),
            ('conv.pointwise_conv2.weight', 'conv_pointwise2.weights'),
            ('conv.pointwise_conv2.bias', 'conv_pointwise2.bias'),
            ('conv.norm.weight', 'conv_batchnorm_weight.weights'),
            ('conv.norm.bias', 'conv_batchnorm_bias.bias'),
            ('conv.norm.running_mean', 'conv_batchnorm_running_mean.weights'),
            ('conv.norm.running_var', 'conv_batchnorm_running_var.weights'),
            ('norm_feed_forward1.weight', 'norm_ff1.weights'),
            ('norm_feed_forward1.bias', 'norm_ff1.bias'),
            ('norm_self_att.weight', 'norm_self_attn.weights'),
            ('norm_self_att.bias', 'norm_self_attn.bias'),
            ('norm_conv.weight', 'norm_conv.weights'),
            ('norm_conv.bias', 'norm_conv.bias'),
            ('norm_feed_forward2.weight', 'norm_ff2.weights'),
            ('norm_feed_forward2.bias', 'norm_ff2.bias'),
            ('norm_out.weight', 'norm_out.weights'),
            ('norm_out.bias', 'norm_out.bias'),
        ]

        for i in range(num_layers):
            layer_prefix = f'encoder.layers.{i}.'
            for suffix, out_suffix in layer_mappings:
                key = layer_prefix + suffix
                out_name = f'layer_{i}_{out_suffix}'
                if key not in state_dict:
                    if suffix.endswith('num_batches_tracked'):
                        continue
                    missing_tensors.append((i, out_name, [key]))
                    continue
                save_tensor_with_header(
                    state_dict[key], output_dir / out_name, precision, transpose=False,
                    stats_tracker=quantization_stats, args=args, model_type=detected_model_type
                )
                saved_tensor_full_names.add(key)
            tracked_key = layer_prefix + 'conv.norm.num_batches_tracked'
            if tracked_key in state_dict:
                saved_tensor_full_names.add(tracked_key)
    else:
        for i in range(num_layers):
            layer_prefixes = [p.format(i=i) for p in LAYER_PREFIXES]

            existing_prefixes = set()
            for prefix in layer_prefixes:
                for key in state_dict.keys():
                    if key.startswith(prefix):
                        existing_prefixes.add(prefix)

            if not existing_prefixes:
                missing_tensors.append((i, "<no-layer-prefix>", ["<no-matching-prefix>"]))
                continue

            weight_patterns = get_layer_weight_patterns(i, precision, model_type_str)

            for layer_prefix in existing_prefixes:
                for name_patterns, tensor_precision, output_name, should_transpose in weight_patterns:
                    found = False
                    for pattern in name_patterns:
                        if model_type_str == 'lfm2_moe' and pattern.startswith('feed_forward.experts.{channel}.'):
                            num_channels = int(model_config.get('num_experts', 0))
                            if num_channels <= 0:
                                continue

                            matched_any_channel = False
                            for channel_idx in range(num_channels):
                                full_name = layer_prefix + pattern.replace('{channel}', str(channel_idx))
                                if full_name not in state_dict:
                                    continue

                                channel_output_name = output_name.replace('{channel}', str(channel_idx))
                                tensor = state_dict[full_name]
                                save_tensor_with_header(tensor, output_dir / channel_output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                                saved_tensor_full_names.add(full_name)
                                matched_any_channel = True

                            if matched_any_channel:
                                found = True
                                break

                        full_name = layer_prefix + pattern
                        if full_name in state_dict:
                            tensor = state_dict[full_name]

                            if 'mlp.fc1.weight' in pattern and model_type_str == 'moonshine':
                                activation = model_config.get('enc_hidden_act', 'gelu') if 'encoder' in layer_prefix else model_config.get('dec_hidden_act', 'gelu')
                                if activation == 'silu':
                                    w = tensor
                                    b_name = full_name.replace('weight', 'bias')
                                    b = state_dict.get(b_name)

                                    inter_size = model_config.get('intermediate_size', 0)
                                    if inter_size == 0:
                                        inter_size = model_config.get('ffn_intermediate_dim', 0)

                                    half_dim = w.shape[0] // 2
                                    w_up = w[:half_dim, :]
                                    w_gate = w[half_dim:, :]

                                    save_name_prefix = output_name.replace('mlp_fc1.weights', '')
                                    if 'encoder' in layer_prefix:
                                        save_name_prefix = "encoder_" + save_name_prefix

                                    save_tensor_with_header(w_gate, output_dir / (save_name_prefix + "ffn_gate.weights"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                                    save_tensor_with_header(w_up, output_dir / (save_name_prefix + "ffn_up.weights"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)

                                    if b is not None:
                                        b_up = b[:half_dim]
                                        b_gate = b[half_dim:]
                                        save_tensor_with_header(b_gate, output_dir / (save_name_prefix + "ffn_gate.bias"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                                        save_tensor_with_header(b_up, output_dir / (save_name_prefix + "ffn_up.bias"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)

                                    saved_tensor_full_names.add(full_name)
                                    if b is not None:
                                        saved_tensor_full_names.add(b_name)
                                    found = True
                                    break

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
                            elif model_type_str == 'moonshine' and 'encoder' in layer_prefix:
                                enc_output_name = "encoder_" + output_name
                                save_tensor_with_header(tensor, output_dir / enc_output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                            else:
                                save_tensor_with_header(tensor, output_dir / output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                            saved_tensor_full_names.add(full_name)
                            found = True
                            break

                    if not found and 'mlp.fc1.weight' in output_name and model_type_str == 'moonshine':
                        activation = model_config.get('enc_hidden_act', 'gelu') if 'encoder' in layer_prefix else model_config.get('dec_hidden_act', 'gelu')

                        if activation == 'silu':
                            full_name = layer_prefix + name_patterns[0][0]
                            w_name = layer_prefix + 'mlp.fc1.weight'
                            b_name = layer_prefix + 'mlp.fc1.bias'

                            if w_name in state_dict:
                                w = state_dict[w_name]
                                if b_name in state_dict:
                                    b = state_dict[b_name]
                                else:
                                    b = None

                                inter_size = model_config.get('intermediate_size', 0)
                                if inter_size == 0:
                                    inter_size = model_config.get('ffn_intermediate_dim', 0)

                                half_dim = w.shape[0] // 2

                                w_up = w[:half_dim, :]
                                w_gate = w[half_dim:, :]

                                save_name_prefix = output_name.replace('mlp_fc1.weights', '')
                                if 'encoder' in layer_prefix:
                                    save_name_prefix = "encoder_" + save_name_prefix

                                save_tensor_with_header(w_gate, output_dir / (save_name_prefix + "ffn_gate.weights"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                                save_tensor_with_header(w_up, output_dir / (save_name_prefix + "ffn_up.weights"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)

                                if b is not None:
                                    b_up = b[:half_dim]
                                    b_gate = b[half_dim:]
                                    save_tensor_with_header(b_gate, output_dir / (save_name_prefix + "ffn_gate.bias"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                                    save_tensor_with_header(b_up, output_dir / (save_name_prefix + "ffn_up.bias"), precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)

                                saved_tensor_full_names.add(w_name)
                                if b_name in state_dict:
                                    saved_tensor_full_names.add(b_name)
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

    if missing_tensors:
        missing_report = output_dir / "missing_weights.txt"
        with open(missing_report, 'w') as fh:
            fh.write("# Missing tensors during conversion\n")
            for layer_idx, output_name, patterns in missing_tensors:
                pattern_list = ', '.join(patterns)
                fh.write(f"layer={layer_idx}, output={output_name}, patterns=[{pattern_list}]\n")
        print(f"Warning: {len(missing_tensors)} tensors were not exported. See {missing_report.name} for details.")

    print_quantization_summary(quantization_stats, args)

    if detected_model_type in ['whisper', 'moonshine', 'parakeet']:
        if torch is None:
            print("Warning: torch not available, skipping VAD bundling")
        else:
            print(f"Bundling Silero VAD weights for {detected_model_type} model...")
            try:
                import urllib.request
                import tempfile

                # Download silero VAD JIT model directly to avoid torchaudio import issues
                vad_jit_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit"
                with tempfile.NamedTemporaryFile(suffix='.jit', delete=False) as f:
                    jit_path = f.name
                urllib.request.urlretrieve(vad_jit_url, jit_path)
                vad_model = torch.jit.load(jit_path, map_location='cpu')
                os.unlink(jit_path)

                vad_output_dir = str(Path(output_dir) / "vad")
                convert_silero_vad_weights(vad_model, vad_output_dir, precision, args)
                del vad_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("VAD weights bundled successfully")
            except Exception as e:
                print(f"Warning: Failed to bundle VAD weights: {e}")

    return model_config


def convert_silero_vad_weights(model, output_dir, precision="FP16", args=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    stft_basis = state_dict["_model.stft.forward_basis_buffer"]
    n_fft_bins, _, window_size = stft_basis.shape

    lstm_weight_ih = state_dict["_model.decoder.rnn.weight_ih"]
    lstm_hidden_size = lstm_weight_ih.shape[1]

    encoder_channels = []
    for i in range(4):
        key = f"_model.encoder.{i}.reparam_conv.weight"
        if key in state_dict:
            weight = state_dict[key]
            out_ch, in_ch, kernel = weight.shape
            encoder_channels.append((in_ch, out_ch, kernel))

    config = {
        "model_type": "silero_vad",
        "sampling_rate": 16000,
        "window_size": int(window_size),
        "n_fft_bins": int(n_fft_bins),
        "num_encoder_blocks": len(encoder_channels),
        "lstm_hidden_size": int(lstm_hidden_size),
        "model_variant": "default",
        "precision": precision,
    }

    save_tensor_with_header(
        stft_basis, output_dir / "stft_basis.weights", precision=precision
    )

    for i in range(config["num_encoder_blocks"]):
        save_tensor_with_header(
            state_dict[f"_model.encoder.{i}.reparam_conv.weight"],
            output_dir / f"encoder_block_{i}_conv_weight.weights",
            precision=precision,
        )
        save_tensor_with_header(
            state_dict[f"_model.encoder.{i}.reparam_conv.bias"],
            output_dir / f"encoder_block_{i}_conv_bias.weights",
            precision=precision,
        )

    lstm_weights = [
        ("_model.decoder.rnn.weight_ih", "lstm_weight_ih.weights"),
        ("_model.decoder.rnn.weight_hh", "lstm_weight_hh.weights"),
        ("_model.decoder.rnn.bias_ih", "lstm_bias_ih.weights"),
        ("_model.decoder.rnn.bias_hh", "lstm_bias_hh.weights"),
    ]
    for key, filename in lstm_weights:
        save_tensor_with_header(
            state_dict[key], output_dir / filename, precision="FP16"
        )

    save_tensor_with_header(
        state_dict["_model.decoder.decoder.2.weight"],
        output_dir / "output_conv_weight.weights",
        precision=precision,
    )
    save_tensor_with_header(
        state_dict["_model.decoder.decoder.2.bias"],
        output_dir / "output_conv_bias.weights",
        precision=precision,
    )

    config_path = output_dir / "config.txt"
    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    return config
