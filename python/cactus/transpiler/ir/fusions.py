import models, nodes


RMS_NORM = models.FusionGraph(
    target="rms_norm",
    root="weighted_output",
    edges=(
        models.Edge("weighted_output", 0, "normalized_x"),
        models.Edge("normalized_x", 1, "inverse_rms"),
        models.Edge("inverse_rms", 0, "variance_with_eps"),
        models.Edge("variance_with_eps", 0, "variance"),
        models.Edge("variance", 0, "squared_x"),
    ),
)


ALL_FUSIONS = (
    RMS_NORM,
)


ROOT_TARGET_MAP = {
    "aten.mul.Tensor": (
        RMS_NORM,
    ),
}
