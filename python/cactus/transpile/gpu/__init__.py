"""GPU transpiler pipeline.

Emits per-model GPU bundles (weights pre-packed into Metal-ready layouts +
a `gpu_plan.json` listing per-layer dispatches) alongside the cactus
graph transpiler's output. The runtime side
(``cactus-engine/src/gpu/``) loads the bundle, builds Metal compute
pipelines via ``cactus-kernels-gpu``, and runs the entire decoder hot
loop on GPU with the KV cache GPU-resident.

Entry point: :func:`run_gpu_pipeline`, invoked from
``hf_model._run_component_pipeline_transpile`` when ``--gpu`` is passed.

See ``docs/gpu/DESIGN.md`` for the full architecture.
"""
from .pipeline import run_gpu_pipeline

__all__ = ["run_gpu_pipeline"]
