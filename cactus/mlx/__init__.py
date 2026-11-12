"""Reusable MLX inference for Cactus quantized models."""
from ._environment import require_runtime


def load(model, *, config_dir=None, progress=None):
    """Load a supported checkpoint from its Hugging Face ID or local path."""
    require_runtime()
    from .runtime import load_runtime
    return load_runtime(model, config_dir=config_dir, progress=progress)


def generate(model, tokenizer, prompt, **kwargs):
    require_runtime()
    from .generation import generate as run
    return run(model, tokenizer, prompt, **kwargs)


def stream_generate(model, tokenizer, prompt, **kwargs):
    require_runtime()
    from .generation import stream_generate as run
    return run(model, tokenizer, prompt, **kwargs)
