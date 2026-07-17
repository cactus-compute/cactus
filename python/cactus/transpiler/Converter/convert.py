from . import models
from ..ModelProfiles import profiles as MP
from ..ModelProfiles import models as MP_Models


def convert(model_id:str, input_modalities:tuple[str,...], custom_profile:MP_Models.ModelProfile = None) -> models.LayerMap | None:
    if custom_profile is not None:
        return None
    
    