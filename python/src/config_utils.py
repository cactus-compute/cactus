import json
import os
from pathlib import Path
from typing import Optional, Any, Dict


class CactusConfig:
    def __init__(self):
        self.config_dir = Path.home() / ".cactus"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        self.telemetry_cache_dir = Path.home() / "Library" / "Caches" / "cactus" / "telemetry"
        self.cloud_api_key_cache_file = self.telemetry_cache_dir / "cloud_api_key"

    def load_config(self):
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        return {}

    def save_config(self, config):
        self.config_file.write_text(json.dumps(config, indent=2))

    def get_api_key(self):
        env_key = os.getenv("CACTUS_CLOUD_KEY")
        if not env_key:
            env_key = os.getenv("CACTUS_CLOUD_API_KEY")
        if env_key:
            return env_key
        config = self.load_config()
        config_key = config.get("api_key", "")
        if config_key:
            return config_key
        return self.load_cached_api_key()

    def cache_api_key(self, key):
        if not key:
            return
        self.telemetry_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cloud_api_key_cache_file.write_text(key)

    def load_cached_api_key(self):
        if not self.cloud_api_key_cache_file.exists():
            return ""
        return self.cloud_api_key_cache_file.read_text().strip()

    def clear_cached_api_key(self):
        if self.cloud_api_key_cache_file.exists():
            self.cloud_api_key_cache_file.unlink()

    def set_api_key(self, key):
        config = self.load_config()
        config["api_key"] = key
        self.save_config(config)
        self.cache_api_key(key)

    def clear_api_key(self):
        config = self.load_config()
        config.pop("api_key", None)
        self.save_config(config)
        self.clear_cached_api_key()


def cfg_get(c, key, default=None):
    """Get a config value from a dict or object, with fallback default."""
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


def detect_model_type(cfg, config, output_dir=None):
    """Detect the model architecture type from config."""
    model_type_str = str(cfg_get(cfg, 'model_type', cfg_get(config, 'model_type', '')) or '').lower().strip()
    decoding_cfg = cfg_get(cfg, 'decoding', cfg_get(config, 'decoding', None))
    decoding_model_type = str(cfg_get(decoding_cfg, 'model_type', '')).lower()
    loss_cfg = cfg_get(cfg, 'loss', cfg_get(config, 'loss', None))
    loss_name = str(cfg_get(loss_cfg, 'loss_name', '')).lower()

    # NeMo Parakeet-TDT configs often do not expose HF-style model_type names.
    if decoding_model_type == 'tdt' or loss_name == 'tdt':
        return 'parakeet_tdt'

    if 'gemma3n' in model_type_str:
        return 'gemma3n'
    elif 'gemma' in model_type_str:
        return 'gemma'
    elif 'lfm2' in model_type_str:
        return 'lfm2'
    elif model_type_str.startswith('qwen3_5'):
        return 'qwen3_5'
    elif 'qwen' in model_type_str:
        return 'qwen'
    elif 'moonshine' in model_type_str:
        return 'moonshine'
    elif 'llama' in model_type_str:
        if output_dir and 'smol' in str(output_dir):
            return 'smol'
        else:
            return 'llama'
    elif 'bert' in model_type_str:
        return 'bert'
    elif 'whisper' in model_type_str:
        return 'whisper'
    elif 'parakeet' in model_type_str:
        return 'parakeet'
    else:
        if model_type_str:
            print(f"  Warning: Unknown model type '{model_type_str}', defaulting to 'qwen'")
        return 'qwen'


def extract_base_config(cfg, config):
    """Extract base model configuration parameters."""
    rope_parameters = cfg_get(cfg, 'rope_parameters', {})
    rope_theta = cfg_get(cfg, 'rope_theta', None)
    partial_rotary_factor = cfg_get(cfg, 'partial_rotary_factor', None)
    if rope_theta is None and isinstance(rope_parameters, dict):
        rope_theta = rope_parameters.get('rope_theta', None)
    if partial_rotary_factor is None and isinstance(rope_parameters, dict):
        partial_rotary_factor = rope_parameters.get('partial_rotary_factor', None)
    if rope_theta is None:
        rope_theta = cfg_get(config, 'rope_theta', 10000.0)

    num_experts_per_tok = cfg_get(cfg, 'num_experts_per_tok', cfg_get(cfg, 'moe_top_k', cfg_get(cfg, 'num_top_experts', 0)))

    base = {
        'vocab_size': cfg_get(cfg, 'vocab_size', cfg_get(config, 'vocab_size', 0)),
        'hidden_dim': cfg_get(cfg, 'hidden_size', cfg_get(cfg, 'hidden_dim', 0)),
        'num_layers': int(cfg_get(cfg, 'num_hidden_layers', cfg_get(cfg, 'num_layers', 0) or 0)),
        'attention_heads': cfg_get(cfg, 'num_attention_heads', 0),
        'attention_kv_heads': cfg_get(cfg, 'num_key_value_heads', cfg_get(cfg, 'num_attention_heads', 0)),
        'ffn_intermediate_dim': cfg_get(cfg, 'intermediate_size', cfg_get(cfg, 'n_inner', 0)),
        'context_length': cfg_get(cfg, 'max_position_embeddings', cfg_get(cfg, 'max_sequence_length', 0)),
        'rope_theta': rope_theta,
        'attention_head_dim': int(cfg_get(cfg, 'head_dim', int(cfg_get(cfg, 'hidden_size', cfg_get(cfg, 'hidden_dim', 0)) // max(1, cfg_get(cfg, 'num_attention_heads', 1))))),
        'layer_norm_eps': cfg_get(cfg, 'layer_norm_eps', cfg_get(cfg, 'layer_norm_epsilon', cfg_get(cfg, 'rms_norm_eps', cfg_get(cfg, 'norm_eps', 1e-6)))),
        'num_experts': cfg_get(cfg, 'num_experts', 0) or 0,
        'num_shared_experts': cfg_get(cfg, 'num_shared_experts', 0),
        'num_top_experts': num_experts_per_tok,
        'num_experts_per_tok': num_experts_per_tok,
        'moe_every_n_layers': cfg_get(cfg, 'moe_every_n_layers', 0),
    }
    if partial_rotary_factor is not None:
        base['partial_rotary_factor'] = float(partial_rotary_factor)

    layer_types = cfg_get(cfg, 'layer_types', None)
    if isinstance(layer_types, (list, tuple)) and layer_types:
        base['layer_types'] = list(layer_types)

    conv_l_cache = cfg_get(cfg, 'conv_L_cache', None)
    if conv_l_cache is not None:
        base['conv_L_cache'] = int(conv_l_cache)

    linear_num_key_heads = cfg_get(cfg, 'linear_num_key_heads', None)
    linear_key_head_dim = cfg_get(cfg, 'linear_key_head_dim', None)
    linear_num_value_heads = cfg_get(cfg, 'linear_num_value_heads', None)
    linear_value_head_dim = cfg_get(cfg, 'linear_value_head_dim', None)
    if linear_num_key_heads is not None:
        base['linear_num_key_heads'] = int(linear_num_key_heads)
    if linear_key_head_dim is not None:
        base['linear_key_head_dim'] = int(linear_key_head_dim)
    if linear_num_value_heads is not None:
        base['linear_num_value_heads'] = int(linear_num_value_heads)
    if linear_value_head_dim is not None:
        base['linear_value_head_dim'] = int(linear_value_head_dim)
    if linear_num_key_heads is not None and linear_key_head_dim is not None:
        base['linear_q_proj_dim'] = int(linear_num_key_heads) * int(linear_key_head_dim)
        base['linear_k_proj_dim'] = int(linear_num_key_heads) * int(linear_key_head_dim)
    if linear_num_value_heads is not None and linear_value_head_dim is not None:
        base['linear_v_proj_dim'] = int(linear_num_value_heads) * int(linear_value_head_dim)

    return base


def extract_vision_config(config, vision_cfg):
    """Extract vision encoder configuration for VLM models."""
    vision_hidden = int(cfg_get(vision_cfg, 'hidden_size', 0))
    vision_image_size = cfg_get(vision_cfg, 'image_size', cfg_get(vision_cfg, 'size', {}).get('longest_edge', 0) if isinstance(cfg_get(vision_cfg, 'size', {}), dict) else cfg_get(vision_cfg, 'image_size', 0))
    vision_patch = int(cfg_get(vision_cfg, 'patch_size', 0))
    vision_heads = int(cfg_get(vision_cfg, 'num_attention_heads', 0))
    vision_num_layers = int(cfg_get(vision_cfg, 'num_hidden_layers', cfg_get(vision_cfg, 'num_layers', 0) or 0))
    num_channels = int(cfg_get(vision_cfg, 'num_channels', 3))
    visual_tokens_per_img = 0
    try:
        if vision_patch > 0 and vision_image_size > 0:
            per_side = vision_image_size // vision_patch
            visual_tokens_per_img = per_side * per_side
    except Exception:
        visual_tokens_per_img = 0

    pixel_shuffle_factor = int(cfg_get(config, 'scale_factor', cfg_get(vision_cfg, 'scale_factor', 1) or 1))
    downsample_factor = int(cfg_get(config, 'downsample_factor', 2))

    vision_head_dim = int(cfg_get(vision_cfg, 'head_dim', 64))
    vision_kv_heads = int(cfg_get(vision_cfg, 'num_key_value_heads', vision_heads))
    vision_intermediate_size = int(cfg_get(vision_cfg, 'intermediate_size', 4 * vision_hidden))
    vision_position_embedding_size = int(cfg_get(vision_cfg, 'position_embedding_size', 10240))
    vision_pooling_kernel_size = int(cfg_get(vision_cfg, 'pooling_kernel_size', 3))
    vision_default_output_length = cfg_get(vision_cfg, 'default_output_length', 280)
    if isinstance(vision_default_output_length, (list, tuple)):
        vision_default_output_length = vision_default_output_length[0]
    vision_default_output_length = int(vision_default_output_length)

    vision_rope_params = cfg_get(vision_cfg, 'rope_parameters', {})
    vision_rope_theta = 100.0
    if isinstance(vision_rope_params, dict):
        for v in vision_rope_params.values():
            if isinstance(v, dict) and 'rope_theta' in v:
                vision_rope_theta = float(v['rope_theta'])
                break

    return {
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
        'use_image_tokens': bool(cfg_get(config, 'image_token_id', None) is not None),
        'image_token_id': int(cfg_get(config, 'image_token_id', 0)),
        'use_layout_tags': False,
        'downsample_factor': int(downsample_factor),
        'vision_head_dim': vision_head_dim,
        'vision_kv_heads': vision_kv_heads,
        'vision_intermediate_size': vision_intermediate_size,
        'vision_position_embedding_size': vision_position_embedding_size,
        'vision_pooling_kernel_size': vision_pooling_kernel_size,
        'vision_default_output_length': vision_default_output_length,
        'vision_rope_theta': vision_rope_theta,
    }


def extract_lfm2_config(cfg):
    """Extract LFM2-specific configuration parameters."""
    layer_types = cfg_get(cfg, 'layer_types', [])
    conv_L_cache = cfg_get(cfg, 'conv_L_cache', 0)
    moe_intermediate_size = cfg_get(cfg, 'moe_intermediate_size', 0)
    num_dense_layers = cfg_get(cfg, 'num_dense_layers', 0)
    num_experts_per_tok = cfg_get(cfg, 'num_experts_per_tok', 0)
    norm_topk_prob = bool(cfg_get(cfg, 'norm_topk_prob', False))
    use_expert_bias = bool(cfg_get(cfg, 'use_expert_bias', False))
    routed_scaling_factor = float(cfg_get(cfg, 'routed_scaling_factor', 1.0))
    return {
        'layer_types': layer_types,
        'conv_L_cache': conv_L_cache,
        'moe_intermediate_size': int(moe_intermediate_size),
        'num_dense_layers': int(num_dense_layers),
        'num_experts_per_tok': int(num_experts_per_tok),
        'norm_topk_prob': norm_topk_prob,
        'use_expert_bias': use_expert_bias,
        'routed_scaling_factor': routed_scaling_factor,
    }

def extract_moonshine_config(cfg):
    """Extract Moonshine-specific configuration parameters."""
    rot_factor = getattr(cfg, "partial_rotary_factor", 0.9)
    return {
        'partial_rotary_factor': rot_factor,
    }


def extract_complex_gemma_config(cfg, root_config):
    """Extract configuration parameters for complex Gemma-family models."""
    altup_num_inputs = int(cfg_get(cfg, 'altup_num_inputs', cfg_get(root_config, 'altup_num_inputs', 4)))
    laurel_rank = int(cfg_get(cfg, 'laurel_rank', cfg_get(root_config, 'laurel_rank', 64)))
    hidden_size_per_layer_input = int(cfg_get(cfg, 'hidden_size_per_layer_input',
        cfg_get(root_config, 'hidden_size_per_layer_input', 256)))
    rope_local_base_freq = float(cfg_get(cfg, 'rope_local_base_freq',
        cfg_get(root_config, 'rope_local_base_freq', 10000.0)))

    num_kv_shared_layers = int(cfg_get(cfg, 'num_kv_shared_layers',
        cfg_get(root_config, 'num_kv_shared_layers', 0)))
    sliding_window = int(cfg_get(cfg, 'sliding_window',
        cfg_get(root_config, 'sliding_window', 512)))

    rope_theta = cfg_get(root_config, 'rope_theta', None)
    if rope_theta is None:
        rope_theta = cfg_get(cfg, 'rope_theta', 1000000.0)

    attention_types = cfg_get(cfg, 'attention_type_pattern', cfg_get(root_config, 'attention_type_pattern', None))
    if attention_types is None:
        attention_types = cfg_get(cfg, 'attention_types', cfg_get(root_config, 'attention_types', None))
    if attention_types is None:
        attention_types = cfg_get(cfg, 'layer_types', cfg_get(root_config, 'layer_types', None))
    layer_types = []
    if attention_types:
        num_layers = int(cfg_get(cfg, 'num_hidden_layers', cfg_get(cfg, 'num_layers', 30)))
        if isinstance(attention_types, (list, tuple)):
            pattern = list(attention_types)
        else:
            pattern = [str(attention_types)]
        pattern_len = len(pattern)
        for i in range(num_layers):
            at = str(pattern[i % pattern_len]).lower()
            if 'global' in at or 'full' in at:
                layer_types.append('global')
            else:
                layer_types.append('sliding')

    final_logit_softcapping = float(cfg_get(cfg, 'final_logit_softcapping',
        cfg_get(root_config, 'final_logit_softcapping', 30.0)))

    query_pre_attn_scalar = int(cfg_get(cfg, 'query_pre_attn_scalar',
        cfg_get(root_config, 'query_pre_attn_scalar', 0)))

    rope_params = cfg_get(cfg, 'rope_parameters', cfg_get(root_config, 'rope_parameters', {}))
    global_rope = rope_params.get('full_attention', {}) if isinstance(rope_params, dict) else {}
    global_partial_rotary_factor = float(cfg_get(cfg, 'global_partial_rotary_factor',
        global_rope.get('partial_rotary_factor', 1.0) if isinstance(global_rope, dict) else 1.0))

    if rope_theta is None or rope_theta == 1000000.0:
        global_rope_theta = global_rope.get('rope_theta', None) if isinstance(global_rope, dict) else None
        if global_rope_theta is not None:
            rope_theta = float(global_rope_theta)

    sliding_rope = rope_params.get('sliding_attention', {}) if isinstance(rope_params, dict) else {}
    if isinstance(sliding_rope, dict) and 'rope_theta' in sliding_rope:
        rope_local_base_freq = float(sliding_rope['rope_theta'])

    activation_sparsity = cfg_get(cfg, 'activation_sparsity_pattern',
        cfg_get(root_config, 'activation_sparsity_pattern', None))

    activation_sparsity_ppf = None
    if activation_sparsity:
        import torch, math
        activation_sparsity_ppf = []
        for s in activation_sparsity:
            if s > 0:
                activation_sparsity_ppf.append(
                    round(math.sqrt(2) * torch.erfinv(torch.tensor(2.0 * s - 1.0)).item(), 7))
            else:
                activation_sparsity_ppf.append(0.0)

    result = {
        'altup_num_inputs': altup_num_inputs,
        'laurel_rank': laurel_rank,
        'hidden_size_per_layer_input': hidden_size_per_layer_input,
        'num_kv_shared_layers': num_kv_shared_layers,
        'sliding_window': sliding_window,
        'rope_local_base_freq': rope_local_base_freq,
        'rope_theta': float(rope_theta),
        'final_logit_softcapping': final_logit_softcapping,
        'query_pre_attn_scalar': query_pre_attn_scalar,
        'global_partial_rotary_factor': global_partial_rotary_factor,
    }
    if layer_types:
        result['layer_types'] = layer_types
    if activation_sparsity_ppf:
        result['activation_sparsity_ppf'] = activation_sparsity_ppf
    return result

def extract_audio_config(config, audio_cfg):
    """Extract audio encoder configuration for multimodal models."""
    hidden = int(cfg_get(audio_cfg, 'hidden_size', 1024))
    num_heads = int(cfg_get(audio_cfg, 'conf_num_attention_heads', 8))
    sscp_channels = cfg_get(audio_cfg, 'sscp_conv_channel_size', [128, 32])
    if not isinstance(sscp_channels, (list, tuple)):
        sscp_channels = [128, 32]
    output_proj = cfg_get(audio_cfg, 'output_proj_dims', None)
    if output_proj is None:
        output_proj = 0
    return {
        'audio_hidden_dim': hidden,
        'audio_num_layers': int(cfg_get(audio_cfg, 'conf_num_hidden_layers', 12)),
        'audio_num_heads': num_heads,
        'audio_head_dim': hidden // max(1, num_heads),
        'audio_input_feat_size': int(cfg_get(audio_cfg, 'input_feat_size', 128)),
        'audio_conf_conv_kernel_size': int(cfg_get(audio_cfg, 'conf_conv_kernel_size', 5)),
        'audio_chunk_size': int(cfg_get(audio_cfg, 'conf_attention_chunk_size', 12)),
        'audio_context_left': int(cfg_get(audio_cfg, 'conf_attention_context_left', 13)),
        'audio_context_right': int(cfg_get(audio_cfg, 'conf_attention_context_right', 0)),
        'audio_logit_cap': float(cfg_get(audio_cfg, 'conf_attention_logit_cap', 50.0)),
        'audio_residual_weight': float(cfg_get(audio_cfg, 'conf_residual_weight', 0.5)),
        'audio_output_proj_dims': int(output_proj),
        'audio_vocab_size': int(cfg_get(audio_cfg, 'vocab_size', 128)),
        'audio_vocab_offset': int(cfg_get(audio_cfg, 'vocab_offset', 0)),
        'audio_soft_tokens': int(cfg_get(audio_cfg, 'audio_soft_tokens_per_image', 188)),
        'audio_sscp_conv0_channels': int(sscp_channels[0]),
        'audio_sscp_conv1_channels': int(sscp_channels[1]),
        'audio_rms_norm_eps': float(cfg_get(audio_cfg, 'rms_norm_eps', 1e-6)),
        'audio_token_id': int(cfg_get(config, 'audio_token_id', 0)),
    }


def is_vlm_model(config):
    """Check if a model config indicates a vision-language model."""
    text_cfg = cfg_get(config, 'text_config', None)
    vision_cfg = cfg_get(config, 'vision_config', None)
    return text_cfg is not None or vision_cfg is not None


def is_lfm2_vl(model_name, cfg):
    """Check if the model is an LFM2 vision-language model."""
    if getattr(cfg, "model_type", None) == "lfm2-vl":
        return True
    name = (model_name or "").lower()
    return "lfm2-vl" in name


def pick_dtype():
    """Select torch dtype for model loading — bf16 preferred, float16 fallback."""
    import torch
    try:
        torch.zeros(1, dtype=torch.bfloat16)
        return torch.bfloat16
    except Exception:
        return torch.float16


def vision_weight_sanity_check(model):
    """Verify vision tower weights are properly initialized."""
    ok = True
    vt = getattr(model, "vision_tower", None)
    try:
        emb = vt.vision_model.embeddings
        w_mean = emb.patch_embedding.weight.detach().abs().mean().item()
        p_mean = emb.position_embedding.weight.detach().abs().mean().item()
        print(f"[sanity] |patch W| mean={w_mean:.5f} |pos W| mean={p_mean:.5f}")
        if w_mean < 1e-3 or p_mean < 1e-3:
            ok = False
    except Exception:
        pass
    return ok
