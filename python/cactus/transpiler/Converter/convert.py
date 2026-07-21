from pathlib import Path

from . import models
from ..ModelProfiles import models as MP_Models
from ..ModelProfiles import profiles as MP


#Selects the best known model profile for a Hugging Face model id.
def _infer_profile(model_id: str) -> MP_Models.ModelProfile:
    model_id_lower = model_id.lower()

    for mapped_model_id, profile in MP.MODEL_ID_MAP.items():
        if model_id_lower == mapped_model_id.lower():
            return profile

    for required_terms, profile in MP.MODEL_ID_HINTS:
        if all(term in model_id_lower for term in required_terms):
            return profile

    raise ValueError(f"Could not infer model profile for {model_id}. Pass custom_profile instead.")


#Resolves where the exported LayerMap JSON should be written.
def _output_json_path(output_path: str | Path | None = None) -> Path:
    if output_path is None:
        path = Path(__file__).resolve().parent / "jsons" / "output.json"
    else:
        path = Path(output_path)
        if path.suffix != ".json":
            path = path / "output.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


#Runs the full converter pipeline and writes the exported LayerMap JSON.
def convert(model_id: str, input_modalities: tuple[str, ...], custom_profile: MP_Models.ModelProfile | None = None, inference_mode: str = "prefill_no_cache", output_path: str | Path | None = None) -> models.LayerMap: 
    profile = custom_profile or _infer_profile(model_id)
    model = models.create_model(mp=profile, input_modalities=input_modalities, model_id=model_id, inference_mode=inference_mode)
    layer_map = model.export(model.input)
    output_file = _output_json_path(output_path)
    output_file.write_text(layer_map.model_dump_json(indent=4), encoding="utf-8")
    return layer_map


if __name__ == "__main__":
    convert(
        model_id="google/gemma-4-E2B",
        input_modalities=("text", "vision", "audio"),
    )
