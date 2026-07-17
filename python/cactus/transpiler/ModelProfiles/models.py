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
    model_profiles:str
    components: dict[str, Components]
    inference_type: dict[str, InferencePattern]
    cache_type: tuple[str, ...]
    files: tuple[str, ...]
    fusion_fields: tuple[str, ...] 
    features: tuple[str, ...]
