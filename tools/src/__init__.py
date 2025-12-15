__version__ = "0.1.0"


def __getattr__(name):
    if name == "main":
        from .cli import main
        return main
    if name == "convert_hf_to_cactus":
        from .cli import convert_hf_to_cactus
        return convert_hf_to_cactus
    if name == "get_model_dir_name":
        from .cli import get_model_dir_name
        return get_model_dir_name
    if name == "save_tensor_with_header":
        from .tensor_io import save_tensor_with_header
        return save_tensor_with_header
    if name == "convert_hf_tokenizer":
        from .tokenizer import convert_hf_tokenizer
        return convert_hf_tokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
