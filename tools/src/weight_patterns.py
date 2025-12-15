EMBED_NAMES = [
    'model.language_model.embed_tokens.weight',
    'model.text_model.embed_tokens.weight',
    'model.embed_tokens.weight',
    'embed_tokens.weight',
    'embeddings.weight',
    'transformer.wte.weight'
]

OUTPUT_NAMES = [
    'lm_head.weight',
    'output.weight',
    'transformer.lm_head.weight',
    'model.text_model.lm_head.weight'
]

OUTPUT_NORM_NAMES = [
    'model.norm.weight',
    'norm.weight',
    'final_layernorm.weight',
    'transformer.ln_f.weight',
    'model.embedding_norm.weight',
    'model.language_model.embedding_norm.weight',
    'model.text_model.norm.weight'
]

LAYER_PREFIXES = [
    'model.language_model.layers.{i}.',
    'model.text_model.layers.{i}.',
    'model.layers.{i}.',
    'layers.{i}.',
    'transformer.h.{i}.',
    'encoder.layers.{i}.',
    'decoder.layers.{i}.',
    'model.encoder.layers.{i}.',
    'model.decoder.layers.{i}.'
]

VISION_ITEMS = [
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

PROJECTOR_WEIGHTS = [
    ('model.multi_modal_projector.linear_1.weight', 'projector_linear1.weights'),
    ('model.multi_modal_projector.linear_1.bias', 'projector_linear1.bias.weights'),
    ('model.multi_modal_projector.linear_2.weight', 'projector_linear2.weights'),
    ('model.multi_modal_projector.linear_2.bias', 'projector_linear2.bias.weights'),
    ('model.multi_modal_projector.layer_norm.weight', 'projector_layer_norm.weights'),
    ('model.multi_modal_projector.layer_norm.bias', 'projector_layer_norm.bias.weights'),
]

CONNECTOR_KEYS = [
    'model.connector.modality_projection.proj.weight',
    'connector.modality_projection.proj.weight',
    'model.connector.proj.weight',
    'connector.proj.weight'
]

WHISPER_GLOBAL_WEIGHTS = [
    ('decoder.embed_tokens.weight', 'decoder_token_embeddings.weights'),
    ('decoder.embed_positions.weight', 'decoder_position_embeddings.weights'),
    ('decoder.layer_norm.weight', 'decoder_norm.weights'),
    ('decoder.layer_norm.bias', 'decoder_norm.bias'),
    ('proj_out.weight', 'output_layer.weights'),
    ('encoder.embed_positions.weight', 'encoder_position_embeddings.weights'),
    ('encoder.conv1.bias', 'encoder_conv1_bias.bias'),
    ('encoder.conv1.weight', 'encoder_conv1_weight.weights'),
    ('encoder.conv2.bias', 'encoder_conv2_bias.bias'),
    ('encoder.conv2.weight', 'encoder_conv2_weight.weights'),
    ('encoder.layer_norm.bias', 'encoder_norm_bias.bias'),
    ('encoder.layer_norm.weight', 'encoder_norm_weight.weights')
]


def get_layer_weight_patterns(i, precision):
    return [
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
        (['pre_feedforward_layernorm.weight'], precision, f'layer_{i}_pre_ffn_norm.weights', False),
        (['post_feedforward_layernorm.weight'], precision, f'layer_{i}_post_ffn_norm.weights', False),
        (['conv.in_proj.weight'], precision, f'layer_{i}_conv_in_proj.weights', False),
        (['conv.out_proj.weight'], precision, f'layer_{i}_conv_out_proj.weights', False),
        (['conv.conv.weight'], precision, f'layer_{i}_conv_depthwise.weights', False),
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


def get_vision_layer_weights(i_v, vpref):
    return [
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
