from dataclasses import dataclass


@dataclass(slots=True)
class Components:
    name:str
    patterns:tuple[str, ...]

@dataclass(slots=True)
class Combinations:
    input:tuple[str, ...]
    output:str

@dataclass(slots=True)
class InferencePattern:
    name:str
    route:tuple[Combinations, ...]

@dataclass(slots=True)
class ModelProfile:
    model_profiles:str #Model family for which this profile is valid (Theoretically can be shared across model families that have similar architectures)
    components: dict[str, Components] #Individual components (vision tower, audio tower, token merge, etc.) this model will be split into
    inference_type: dict[str, InferencePattern] #Specific operations mapped to their tuple of components ordered by execution 
    cache_type: tuple[str, ...] #Which layers/operations to expect KV cache for
    cache_policy: tuple[str, ...] #How to handle the cache during different phases of inference
    files: tuple[str, ...] #Config and other necessary files 
    fusion_fields: tuple[str, ...] #Specific fusion groups to consider during fusion process
    supported_modalties: tuple[str, ...] #Input modalities supported by model
    input_strategy: str #Specifies what functions/procedures to use when generating sample input for torch.export
    export_patches: tuple[str, ...] #Specifies what patches/masks to apply to sample inputs
    load_strategy: str #Specifies which Hugging Face AutoModel class/loading path to prefer
    
