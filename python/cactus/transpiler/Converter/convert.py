from pathlib import Path

from . import models
from ..ModelProfiles import models as MP_Models
from ..ModelProfiles import profiles as MP_Profiles
from ..IR import simplify_ir

def export_layer_map(
    model_id: str,
    input_modalities: tuple[str, ...],
    custom_profile: MP_Models.ModelProfile | None = None,
    inference_mode: str = "prefill_no_cache",
) -> models.LayerMap:
    profile = custom_profile or MP_Profiles.MODEL_ID_MAP[model_id]
    model = models.create_model(mp=profile, input_modalities=input_modalities, model_id=model_id, inference_mode=inference_mode)
    return model.export(model.input)

def convert(
    model_id: str,
    input_modalities: tuple[str, ...],
    output_path: str,
    custom_profile: MP_Models.ModelProfile | None = None,
    inference_mode: str = "prefill_no_cache",
    simplified_output_path: str | None = None,
) -> models.LayerMap:
    profile = custom_profile or MP_Profiles.MODEL_ID_MAP[model_id]
    layer_map = export_layer_map(model_id=model_id, input_modalities=input_modalities, custom_profile=profile, inference_mode=inference_mode)
    raw_json = layer_map.model_dump_json(indent=4)
    Path(output_path).write_text(raw_json, encoding="utf-8")
    if simplified_output_path is not None:
        simplify_ir.write_simplified_json(
            models.LayerMap.model_validate_json(raw_json),
            simplified_output_path,
            input_modalities=input_modalities,
            fusion_fields=profile.fusion_fields,
            disabled_fusion_fields=profile.disabled_fusion_fields,
            disabled_fusions=profile.disabled_fusions,
        )
    return layer_map

if __name__ == "__main__":
    convert(
        model_id="google/gemma-4-E2B",
        input_modalities=("text", "vision", "audio"),
        output_path="jsons/output.json",
    )
