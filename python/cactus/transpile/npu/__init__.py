"""NPU transpiler pipeline.

Emits CoreML `.mlpackage`s for Apple Neural Engine **audio + vision encoders**
alongside the cactus graph transpiler's output. Text-decoder prefill is
intentionally not on NPU — CPU prefill is the supported path.

Entry point: `run_encoder_pipeline`, invoked from
`hf_model._run_component_pipeline_transpile` when `--npu` is passed.
"""

from .pipeline import run_encoder_pipeline

__all__ = ["run_encoder_pipeline"]
