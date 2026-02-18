import json
import os
from pathlib import Path
from typing import Optional, Any, Dict


class CactusConfig:
    def __init__(self):
        self.config_dir = Path.home() / ".cactus"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)

    def load_config(self):
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        return {}

    def save_config(self, config):
        self.config_file.write_text(json.dumps(config, indent=2))

    def get_api_key(self):
        env_key = os.getenv("CACTUS_CLOUD_API_KEY")
        if env_key:
            return env_key
        config = self.load_config()
        return config.get("api_key", "")

    def set_api_key(self, key):
        config = self.load_config()
        config["api_key"] = key
        self.save_config(config)

    def clear_api_key(self):
        config = self.load_config()
        config.pop("api_key", None)
        self.save_config(config)


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
    model_type_str = cfg_get(cfg, 'model_type', cfg_get(config, 'model_type', '')).lower()

    if 'gemma' in model_type_str:
        return 'gemma'
    elif 'lfm2' in model_type_str:
        return 'lfm2'
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
    else:
        if model_type_str:
            print(f"  Warning: Unknown model type '{model_type_str}', defaulting to 'qwen'")
        return 'qwen'


def extract_base_config(cfg, config):
    """Extract base model configuration parameters."""
    return {
        'vocab_size': cfg_get(cfg, 'vocab_size', cfg_get(config, 'vocab_size', 0)),
        'hidden_dim': cfg_get(cfg, 'hidden_size', cfg_get(cfg, 'hidden_dim', 0)),
        'num_layers': int(cfg_get(cfg, 'num_hidden_layers', cfg_get(cfg, 'num_layers', 0) or 0)),
        'attention_heads': cfg_get(cfg, 'num_attention_heads', 0),
        'attention_kv_heads': cfg_get(cfg, 'num_key_value_heads', cfg_get(cfg, 'num_attention_heads', 0)),
        'ffn_intermediate_dim': cfg_get(cfg, 'intermediate_size', cfg_get(cfg, 'n_inner', 0)),
        'context_length': cfg_get(cfg, 'max_position_embeddings', cfg_get(cfg, 'max_sequence_length', 0)),
        'rope_theta': cfg_get(cfg, 'rope_theta', cfg_get(config, 'rope_theta', 10000.0)),
        'attention_head_dim': int(cfg_get(cfg, 'head_dim', int(cfg_get(cfg, 'hidden_size', cfg_get(cfg, 'hidden_dim', 0)) // max(1, cfg_get(cfg, 'num_attention_heads', 1))))),
        'layer_norm_eps': cfg_get(cfg, 'layer_norm_eps', cfg_get(cfg, 'layer_norm_epsilon', cfg_get(cfg, 'rms_norm_eps', 1e-6))),
        'num_experts': cfg_get(cfg, 'num_experts', 0),
        'num_shared_experts': cfg_get(cfg, 'num_shared_experts', 0),
        'num_top_experts': cfg_get(cfg, 'moe_top_k', cfg_get(cfg, 'num_top_experts', 0)),
        'moe_every_n_layers': cfg_get(cfg, 'moe_every_n_layers', 0),
    }


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
        'use_layout_tags': False,
        'downsample_factor': int(downsample_factor),
    }


def extract_lfm2_config(cfg):
    """Extract LFM2-specific configuration parameters."""
    layer_types = getattr(cfg, 'layer_types', [])
    conv_L_cache = getattr(cfg, 'conv_L_cache', 0)
    return {
        'layer_types': layer_types,
        'conv_L_cache': conv_L_cache,
    }

def extract_moonshine_config(cfg):
    """Extract Moonshine-specific configuration parameters."""
    rot_factor = getattr(cfg, "partial_rotary_factor", 0.9)
    return {
        'partial_rotary_factor': rot_factor,
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
    """Select the best torch dtype based on hardware capabilities."""
    import torch
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


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
