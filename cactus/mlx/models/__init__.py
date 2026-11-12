"""Select architecture adapters from checkpoint configuration, not repository names."""
from importlib import import_module

_ADAPTERS = {'nemotron_h': '.nemotron'}


def loader_for(config):
    quant_method = config.get('quantization_config', {}).get('quant_method')
    if quant_method != 'cq':
        raise ValueError('This Cactus MLX release requires a CQ checkpoint')
    model_type = config.get('model_type')
    if model_type not in _ADAPTERS:
        raise ValueError(f'Unsupported CQ architecture {model_type!r}; supported: {", ".join(_ADAPTERS)}')
    return import_module(_ADAPTERS[model_type], __package__).load
