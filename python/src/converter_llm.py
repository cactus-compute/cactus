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


def convert_hf_model_weights(model, output_dir, precision='INT8', args=None):
    """Convert HuggingFace model weights to Cactus binary format."""
    quantization_stats = create_quantization_stats()

    state_dict = model.state_dict()
    config = model.config
    saved_tensor_full_names = set()

    text_cfg = cfg_get(config, 'text_config', None)
    vision_cfg = cfg_get(config, 'vision_config', None)
    is_vlm = text_cfg is not None or vision_cfg is not None

    cfg = text_cfg if text_cfg is not None else config

    tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
    model_type_str = cfg_get(cfg, 'model_type', cfg_get(config, 'model_type', '')).lower()

    detected_model_type = detect_model_type(cfg, config, output_dir)

    model_config = extract_base_config(cfg, config)
    model_config['tie_word_embeddings'] = tie_word_embeddings
    model_config['model_type'] = detected_model_type

    if is_vlm and vision_cfg is not None:
        model_config.update(extract_vision_config(config, vision_cfg))

    if detected_model_type == 'lfm2':
        model_config.update(extract_lfm2_config(cfg))
    elif detected_model_type == 'moonshine':
        model_config.update(extract_moonshine_config(cfg))

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
        model_config['dec_hidden_act'] = cfg.decoder_hidden_act
        model_config['enc_hidden_act'] = cfg.encoder_hidden_act
        model_config['num_encoder_layers'] = cfg.encoder_num_hidden_layers
        model_config['num_decoder_layers'] = cfg.decoder_num_hidden_layers

    if embedding_found:
        embedding_norm_names = {'emb_ln.weight': 'embedding_layernorm.weight', 'emb_ln.bias': 'embedding_layernorm.bias'}
        for name, file_name in embedding_norm_names.items():
            if name in state_dict:
                save_tensor_with_header(state_dict[name], output_dir / file_name, precision, stats_tracker=quantization_stats, args=args, model_type=detected_model_type)
                saved_tensor_full_names.add(name)

    if not tie_word_embeddings or is_vlm:
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

    if detected_model_type in ['whisper', 'moonshine']:
        if torch is None:
            print("Warning: torch not available, skipping VAD bundling")
        else:
            from .converter_silero_vad import convert_silero_vad_weights

            print(f"Bundling Silero VAD weights for {detected_model_type} model...")
            try:
                vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)
                vad_output_dir = str(Path(output_dir) / "vad")
                convert_silero_vad_weights(vad_model, vad_output_dir, precision, args)
                del vad_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("VAD weights bundled successfully")
            except Exception as e:
                print(f"Warning: Failed to bundle VAD weights: {e}")

    return model_config
