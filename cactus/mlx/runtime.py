"""Resolve a local checkpoint or a Hugging Face repository and load CQ inference."""
import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download, snapshot_download
from mlx_lm.utils import load_tokenizer

from .models import loader_for

MODEL_FILES = ['model.safetensors', 'config.json', 'generation_config.json',
               'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
               'preprocessor_config.json', 'LICENSE', 'LICENSE-OpenMDW', 'NOTICE']


def resolve_model(source, config_dir=None):
    path = Path(source).expanduser()
    if path.is_file():
        weights = path
        directory = Path(config_dir).expanduser() if config_dir else path.parent
    else:
        if not path.is_dir():
            config_file = hf_hub_download(source, 'config.json')
            loader_for(json.loads(Path(config_file).read_text()))
            path = Path(snapshot_download(source, allow_patterns=MODEL_FILES))
        weights = path / 'model.safetensors'
        directory = Path(config_dir).expanduser() if config_dir else path
    for required in (weights, directory / 'config.json', directory / 'tokenizer.json',
                     directory / 'tokenizer_config.json', directory / 'chat_template.jinja'):
        if not required.is_file():
            raise FileNotFoundError(required)
    return weights, directory


def load_runtime(source, config_dir=None, progress=None):
    mx.set_cache_limit(256 * 2**20)
    mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
    weights, directory = resolve_model(source, config_dir)
    config = json.loads((directory / 'config.json').read_text())
    load = loader_for(config)
    model = load(weights, directory / 'config.json', progress=progress)
    from .vision import ImageProcessor
    processor_config = directory / 'preprocessor_config.json'
    model.image_processor = ImageProcessor(json.loads(processor_config.read_text()) if processor_config.exists() else None)
    tokenizer = load_tokenizer(directory, {'trust_remote_code': False},
                               eos_token_ids=config.get('eos_token_id'))
    return model, tokenizer
