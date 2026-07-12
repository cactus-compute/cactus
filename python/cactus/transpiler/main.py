from dataclasses import dataclass

@dataclass(slots=True)
class ModelProfile:
    model_family:str
    component_graphs:tuple[str]
    export_modes:ep