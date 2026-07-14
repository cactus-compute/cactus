from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class Components:
    name:str
    patterns:tuple[str, ...]

@dataclass(slots=True)
class Combinations:
    input:tuple[str, ...]
    output:str

@dataclass(slots=True)
class Route:
    route:tuple[Combinations, ...]

@dataclass(slots=True)
class InferencePattern:
    name:str
    route:Route

@dataclass(slots=True)
class Cache:
    types:tuple[str, ...]

@dataclass(slots=True)
class Files:
    required: tuple[str, ...]
    optional: tuple[str, ...]

@dataclass(slots=True)
class ModelProfile:
    model_profiles:str
    components: dict[str, Components]
    inference_type: dict[str, InferencePattern]
    cache_type: Cache
    files: Files
    fusion_fields: tuple[str, ...] 
    features: tuple[str, ...]
