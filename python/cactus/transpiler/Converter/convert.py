from pathlib import Path

from . import models
from ..ModelProfiles import models as MP_Models
from ..ModelProfiles import profiles as MP_Profiles

#Runs the full converter pipeline and writes the exported LayerMap JSON.
def convert(model_id: str, input_modalities: tuple[str, ...], output_path: str, custom_profile: MP_Models.ModelProfile | None = None, inference_mode: str = "prefill_no_cache") -> models.LayerMap: 
    profile = custom_profile or MP_Profiles.MODEL_ID_MAP[model_id]
    model = models.create_model(mp=profile, input_modalities=input_modalities, model_id=model_id, inference_mode=inference_mode)
    layer_map = model.export(model.input)
    Path(output_path).write_text(layer_map.model_dump_json(indent=4), encoding="utf-8")
    return layer_map


if __name__ == "__main__":
    convert(model_id="google/gemma-4-E2B", input_modalities=("text", "vision", "audio"))
