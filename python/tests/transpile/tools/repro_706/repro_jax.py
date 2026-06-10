"""Reachability repro for PR #706 (JAX capture path).

    pip install "jax[cpu]"
    python python/tests/transpile/tools/repro_706/repro_jax.py

Shows the stride-aware slice-aliasing fix in capture_jax.py: a strided slice
(x[::2]) must not be treated as a no-op alias.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp

from cactus.transpile.capture_jax import capture_jax_function


def describe(tag, fn, args):
    ir = capture_jax_function(fn, args)
    ops = [ir.nodes[n].op for n in ir.order]
    out_id, in_id = ir.outputs[0], ir.inputs[0]
    out_shape = getattr(ir.values.get(out_id), "shape", None)
    print(f"[{tag}] ops={ops} out_shape={out_shape} output_is_input_alias={out_id == in_id}")


# Strided slice over a length-6 vector -> expected length 3.
# origin: stride ignored in changed_axes -> start=0,limit=full -> aliased no-op (WRONG).
# this branch: stride!=1 -> a real slice node, length 3.
x = jnp.arange(6.0)
print("eager jax x[::2] =", np.asarray(jax.jit(lambda v: v[::2])(x)))
describe("strided-slice x[::2]", lambda v: v[::2], (x,))
