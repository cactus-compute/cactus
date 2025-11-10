#!/usr/bin/env python3
"""
VLM Architecture Support Module
Provides architecture-specific weight extraction patterns for different VLM models
"""

def detect_vlm_architecture(config, model_type_str):
    """
    Detect the VLM architecture type from model configuration
    
    Args:
        config: Model configuration object
        model_type_str: Model type string from config
        
    Returns:
        str: Detected architecture type ('smolvlm', 'qwen_vl', 'llava', etc.)
    """
    model_name_lower = str(model_type_str).lower()
    
    if 'smolvlm' in model_name_lower or 'smol' in model_name_lower:
        return 'smolvlm'
    elif 'qwen' in model_name_lower and 'vl' in model_name_lower:
        return 'qwen_vl'
    elif 'llava' in model_name_lower:
        return 'llava'
    else:
        print(f"  Warning: Unknown VLM model type '{model_type_str}', defaulting to 'smolvlm'")
        return 'smolvlm'


def get_vision_encoder_patterns(architecture):
    """
    Get vision encoder weight patterns for a specific architecture
    
    Args:
        architecture: VLM architecture type
        
    Returns:
        dict: Dictionary containing weight patterns for the architecture
    """
    patterns = {
        'smolvlm': {
            'embeddings': [
                ('model.vision_model.embeddings.patch_embedding.weight', 'vision_patch_embedding.weights'),
                ('model.vision_model.embeddings.patch_embedding.bias', 'vision_patch_embedding.bias.weights'),
                ('model.vision_model.embeddings.position_embedding.weight', 'vision_position_embedding.weights'),
            ],
            'post_norm': [
                ('model.vision_model.post_layernorm.weight', 'vision_post_layernorm.weights'),
                ('model.vision_model.post_layernorm.bias', 'vision_post_layernorm.bias.weights'),
            ],
            'layer_prefix_pattern': r'model\.vision_model\.encoder\.layers\.(\d+)\.',
            'layer_prefix_template': 'model.vision_model.encoder.layers.{}.', 
            'layer_weights': {
                'norm1': [
                    ('layer_norm1.weight', 'vision_layer_{}_layer_norm1.weights'),
                    ('layer_norm1.bias', 'vision_layer_{}_layer_norm1.bias.weights'),
                ],
                'norm2': [
                    ('layer_norm2.weight', 'vision_layer_{}_layer_norm2.weights'),
                    ('layer_norm2.bias', 'vision_layer_{}_layer_norm2.bias.weights'),
                ],
                'mlp': [
                    ('mlp.fc1.weight', 'vision_layer_{}_ffn_fc1.weights'),
                    ('mlp.fc1.bias', 'vision_layer_{}_ffn_fc1.bias.weights'),
                    ('mlp.fc2.weight', 'vision_layer_{}_ffn_fc2.weights'),
                    ('mlp.fc2.bias', 'vision_layer_{}_ffn_fc2.bias.weights'),
                ],
                'attn': [
                    ('self_attn.q_proj.weight', 'vision_layer_{}_self_attn_q.weights'),
                    ('self_attn.k_proj.weight', 'vision_layer_{}_self_attn_k.weights'),
                    ('self_attn.v_proj.weight', 'vision_layer_{}_self_attn_v.weights'),
                    ('self_attn.out_proj.weight', 'vision_layer_{}_self_attn_out.weights'),
                    ('self_attn.q_proj.bias', 'vision_layer_{}_self_attn_q.bias.weights'),
                    ('self_attn.k_proj.bias', 'vision_layer_{}_self_attn_k.bias.weights'),
                    ('self_attn.v_proj.bias', 'vision_layer_{}_self_attn_v.bias.weights'),
                    ('self_attn.out_proj.bias', 'vision_layer_{}_self_attn_out.bias.weights'),
                ],
            },
        },
        'qwen_vl': {
            'embeddings': [
                ('visual.conv1.weight', 'vision_patch_embedding.weights'),
                ('visual.class_embedding', 'vision_class_embedding.weights'),
                ('visual.positional_embedding', 'vision_position_embedding.weights'),
            ],
            'pre_norm': [
                ('visual.ln_pre.weight', 'vision_pre_layernorm.weights'),
                ('visual.ln_pre.bias', 'vision_pre_layernorm.bias.weights'),
            ],
            'post_norm': [
                ('visual.ln_post.weight', 'vision_post_layernorm.weights'),
                ('visual.ln_post.bias', 'vision_post_layernorm.bias.weights'),
            ],
            'layer_prefix_pattern': r'visual\.transformer\.resblocks\.(\d+)\.',
            'layer_prefix_template': 'visual.transformer.resblocks.{}.',
            'layer_weights': {
                'norm1': [
                    ('ln_1.weight', 'vision_layer_{}_layer_norm1.weights'),
                    ('ln_1.bias', 'vision_layer_{}_layer_norm1.bias.weights'),
                ],
                'norm2': [
                    ('ln_2.weight', 'vision_layer_{}_layer_norm2.weights'),
                    ('ln_2.bias', 'vision_layer_{}_layer_norm2.bias.weights'),
                ],
                'mlp': [
                    ('mlp.c_fc.weight', 'vision_layer_{}_ffn_fc1.weights'),
                    ('mlp.c_fc.bias', 'vision_layer_{}_ffn_fc1.bias.weights'),
                    ('mlp.c_proj.weight', 'vision_layer_{}_ffn_fc2.weights'),
                    ('mlp.c_proj.bias', 'vision_layer_{}_ffn_fc2.bias.weights'),
                ],
                'attn': [
                    ('attn.in_proj_weight', None),  # Combined QKV - needs splitting
                    ('attn.in_proj_bias', None),
                    ('attn.out_proj.weight', 'vision_layer_{}_self_attn_out.weights'),
                    ('attn.out_proj.bias', 'vision_layer_{}_self_attn_out.bias.weights'),
                ],
            },
        },
        'llava': {
            'embeddings': [
                ('model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight', 'vision_patch_embedding.weights'),
                ('model.vision_tower.vision_tower.vision_model.embeddings.class_embedding', 'vision_class_embedding.weights'),
                ('model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight', 'vision_position_embedding.weights'),
                ('vision_tower.vision_model.embeddings.patch_embedding.weight', 'vision_patch_embedding.weights'),
                ('vision_tower.vision_model.embeddings.class_embedding', 'vision_class_embedding.weights'),
                ('vision_tower.vision_model.embeddings.position_embedding.weight', 'vision_position_embedding.weights'),
            ],
            'pre_norm': [
                ('model.vision_tower.vision_tower.vision_model.pre_layrnorm.weight', 'vision_pre_layernorm.weights'),
                ('model.vision_tower.vision_tower.vision_model.pre_layrnorm.bias', 'vision_pre_layernorm.bias.weights'),
                ('vision_tower.vision_model.pre_layrnorm.weight', 'vision_pre_layernorm.weights'),
                ('vision_tower.vision_model.pre_layrnorm.bias', 'vision_pre_layernorm.bias.weights'),
            ],
            'post_norm': [
                ('model.vision_tower.vision_tower.vision_model.post_layernorm.weight', 'vision_post_layernorm.weights'),
                ('model.vision_tower.vision_tower.vision_model.post_layernorm.bias', 'vision_post_layernorm.bias.weights'),
                ('vision_tower.vision_model.post_layernorm.weight', 'vision_post_layernorm.weights'),
                ('vision_tower.vision_model.post_layernorm.bias', 'vision_post_layernorm.bias.weights'),
            ],
            'layer_prefix_pattern': r'(?:model\.)?vision_tower(?:\.vision_tower)?\.vision_model\.encoder\.layers\.(\d+)\.',
            'layer_prefix_templates': [
                'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.',
                'vision_tower.vision_model.encoder.layers.{}.',
            ],
            'layer_weights': {
                'norm1': [
                    ('layer_norm1.weight', 'vision_layer_{}_layer_norm1.weights'),
                    ('layer_norm1.bias', 'vision_layer_{}_layer_norm1.bias.weights'),
                ],
                'norm2': [
                    ('layer_norm2.weight', 'vision_layer_{}_layer_norm2.weights'),
                    ('layer_norm2.bias', 'vision_layer_{}_layer_norm2.bias.weights'),
                ],
                'mlp': [
                    ('mlp.fc1.weight', 'vision_layer_{}_ffn_fc1.weights'),
                    ('mlp.fc1.bias', 'vision_layer_{}_ffn_fc1.bias.weights'),
                    ('mlp.fc2.weight', 'vision_layer_{}_ffn_fc2.weights'),
                    ('mlp.fc2.bias', 'vision_layer_{}_ffn_fc2.bias.weights'),
                ],
                'attn': [
                    ('self_attn.q_proj.weight', 'vision_layer_{}_self_attn_q.weights'),
                    ('self_attn.k_proj.weight', 'vision_layer_{}_self_attn_k.weights'),
                    ('self_attn.v_proj.weight', 'vision_layer_{}_self_attn_v.weights'),
                    ('self_attn.out_proj.weight', 'vision_layer_{}_self_attn_out.weights'),
                    ('self_attn.q_proj.bias', 'vision_layer_{}_self_attn_q.bias.weights'),
                    ('self_attn.k_proj.bias', 'vision_layer_{}_self_attn_k.bias.weights'),
                    ('self_attn.v_proj.bias', 'vision_layer_{}_self_attn_v.bias.weights'),
                    ('self_attn.out_proj.bias', 'vision_layer_{}_self_attn_out.bias.weights'),
                ],
            },
        },
    }
    
    return patterns.get(architecture, patterns['smolvlm'])


def get_projection_patterns(architecture):
    """
    Get vision-to-language projection layer patterns for a specific architecture
    
    Args:
        architecture: VLM architecture type
        
    Returns:
        list: List of (weight_key, output_name) tuples
    """
    patterns = {
        'smolvlm': [
            ('model.connector.modality_projection.proj.weight', 'connector_proj.weights'),
            ('connector.modality_projection.proj.weight', 'connector_proj.weights'),
            ('model.connector.proj.weight', 'connector_proj.weights'),
            ('connector.proj.weight', 'connector_proj.weights'),
        ],
        'qwen_vl': [
            ('visual.proj', 'connector_proj.weights'),
            ('visual.attn_pool.q_proj.weight', 'vision_attn_pool_q.weights'),
            ('visual.attn_pool.k_proj.weight', 'vision_attn_pool_k.weights'),
            ('visual.attn_pool.v_proj.weight', 'vision_attn_pool_v.weights'),
            ('visual.attn_pool.c_proj.weight', 'vision_attn_pool_out.weights'),
        ],
        'llava': [
            ('model.multi_modal_projector.linear_1.weight', 'connector_proj_1.weights'),
            ('model.multi_modal_projector.linear_1.bias', 'connector_proj_1.bias.weights'),
            ('model.multi_modal_projector.linear_2.weight', 'connector_proj_2.weights'),
            ('model.multi_modal_projector.linear_2.bias', 'connector_proj_2.bias.weights'),
            ('multi_modal_projector.linear_1.weight', 'connector_proj_1.weights'),
            ('multi_modal_projector.linear_1.bias', 'connector_proj_1.bias.weights'),
            ('multi_modal_projector.linear_2.weight', 'connector_proj_2.weights'),
            ('multi_modal_projector.linear_2.bias', 'connector_proj_2.bias.weights'),
            ('multi_modal_projector.weight', 'connector_proj.weights'),
            ('multi_modal_projector.bias', 'connector_proj.bias.weights'),
        ],
    }
    
    return patterns.get(architecture, patterns['smolvlm'])
