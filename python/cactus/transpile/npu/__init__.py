"""NPU transpiler pipeline.

Parallel to the graph transpiler: takes a captured HF causal LM and emits a
CoreML `.mlpackage` for Apple Neural Engine prefill, alongside the
`.cactus` graphs the graph transpiler produces. Entry point is
`run_prefill_pipeline`, invoked from `hf_model._run_component_pipeline_transpile`
and the legacy single-graph path.
"""

from .pipeline import run_encoder_pipeline, run_prefill_pipeline

__all__ = ["run_prefill_pipeline", "run_encoder_pipeline"]
