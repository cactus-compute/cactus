"""Runtime patches for coremltools 9.0 to support Gemma 4 + torch.export.

Each patch is idempotent and guarded so reimporting the module is safe.
Call `apply_all_coremltools_patches()` once before invoking `ct.convert`.

Patches:
- register_func: allow dunder-suffixed op names (e.g. ``__and__``) past the
  inplace-op rejection check.
- new_ones: register a translation that maps to ``mb.fill(shape=..., value=1.0)``.
- SDPA scalar-cast: ``mb.sub/mul/add`` auto-cast python scalars to the dtype
  of the tensor input (avoids fp16/fp32 mismatch in the SDPA mask path).
- Pipeline: remove ``common::fuse_prelu`` (KeyError on float64 in 9.0).
"""
from __future__ import annotations

_APPLIED = False


def apply_all_coremltools_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return
    try:
        _patch_register_func_allow_dunder()
        _register_new_ones_op()
        _register_logical_and_op()
        _override_layer_norm_translator()
        _patch_pipeline_remove_fuse_prelu()
        _patch_mb_binops_scalar_cast()
        _patch_fp16_cast_skip_layer_norm()
        _APPLIED = True
        print("npu.coremltools_patches: applied")
    except Exception as exc:
        print(f"npu.coremltools_patches: failed to apply ({type(exc).__name__}: {exc})")


def _patch_register_func_allow_dunder() -> None:
    from coremltools.converters.mil.frontend.torch import torch_op_registry

    cls = torch_op_registry.TorchOpsRegistry
    if getattr(cls.register_func, "_cactus_dunder_patched", False):
        return

    original = cls.register_func

    def register_func(self, func=None, torch_alias=None, override=False):
        f_name = func.__name__
        all_f_names = [f_name]
        if torch_alias is not None:
            all_f_names.extend(torch_alias)
        for name in all_f_names:
            is_dunder = name.startswith("__") and name.endswith("__")
            if name.endswith("_") and not is_dunder:
                raise Exception(
                    f'Attempting to register "{name}" op. Do not register inplace ops.'
                )
            if not override and name in self.name_to_func_mapping:
                raise ValueError(f"Torch op {name} already registered.")
            self.set_func_by_name(func, name)

    register_func._cactus_dunder_patched = True
    cls.register_func = register_func


def _register_new_ones_op() -> None:
    """Translate Tensor.new_ones(size, ...) -> mb.fill(shape=size, value=1).

    Mirrors coremltools' built-in `ones` translator (shift by 1 to skip the
    `self` tensor argument). Direct registration via set_func_by_name avoids
    the decorator's redundant name lookups.
    """
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
    )
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs

    if "new_ones" in _TORCH_OPS_REGISTRY.name_to_func_mapping:
        return

    def _to_int32(v):
        if isinstance(v, list):
            casted = [_to_int32(x) for x in v]
            return mb.concat(values=casted, axis=0)
        dtype = getattr(v, "dtype", None)
        name = getattr(dtype, "__name__", str(dtype) if dtype else "")
        if "int32" in name:
            return v
        return mb.cast(x=v, dtype="int32")

    def new_ones(context, node):
        inputs = _get_inputs(context, node, min_expected=2)
        size = _to_int32(inputs[1])
        res = mb.fill(shape=size, value=1.0, name=node.name)
        context.add(res, node.name)

    _TORCH_OPS_REGISTRY.set_func_by_name(new_ones, "new_ones")


def _register_logical_and_op() -> None:
    """Register __and__ / __or__ translations. Gemma 4's attention mask path emits
    `__and__.Tensor` nodes for combining causal + padding masks; coremltools 9.0
    doesn't ship a translator for those names by default."""
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
        register_torch_op,
    )
    from coremltools.converters.mil import Builder as mb

    def _force_bool(v):
        dtype = getattr(v, "dtype", None)
        name = getattr(dtype, "__name__", str(dtype) if dtype else "")
        if "bool" in name:
            return v
        # cast through fp32 -> bool to handle exotic source dtypes (e.g. fp64)
        if "fp16" in name or "fp32" in name or "fp64" in name or "double" in name or "float" in name:
            non_zero = mb.not_equal(x=v, y=0.0)
            return non_zero
        if "int" in name:
            non_zero = mb.not_equal(x=v, y=0)
            return non_zero
        return mb.cast(x=v, dtype="bool")

    def make_logical(op_kind):
        def _impl(context, node):
            inputs = [context[i] for i in node.inputs]
            x = _force_bool(inputs[0])
            y = _force_bool(inputs[1])
            if op_kind == "and":
                out = mb.logical_and(x=x, y=y, name=node.name)
            elif op_kind == "or":
                out = mb.logical_or(x=x, y=y, name=node.name)
            else:
                out = mb.logical_xor(x=x, y=y, name=node.name)
            context.add(out, node.name)
        return _impl

    # Register all aliases that sanitize_op_kind + unify_inplace_and_functional
    # may resolve fx node `__and__.Tensor` / `__or__.Tensor` to. The lookup
    # chain for `__and__.Tensor` ends at `__and_` (one trailing underscore),
    # for `aten::__and__` ends at `__and_`, and the raw key `__and__` ends at
    # `and`. Cover all three so the bool-casting handler always wins.
    for tag, op_kind in [("__and_", "and"), ("__or_", "or"), ("__xor_", "xor")]:
        if tag not in _TORCH_OPS_REGISTRY.name_to_func_mapping:
            _TORCH_OPS_REGISTRY.set_func_by_name(make_logical(op_kind), tag)
    for tag, op_kind in [("__and__", "and"), ("__or__", "or"), ("__xor__", "xor")]:
        if tag not in _TORCH_OPS_REGISTRY.name_to_func_mapping:
            _TORCH_OPS_REGISTRY.set_func_by_name(make_logical(op_kind), tag)
    # The default `bitwise_and` translator chokes on mixed bool+float inputs
    # that Gemma 4's masking emits; override the lookup keys with the
    # bool-casting variant. `and` and `or` are the sanitize_op_kind results
    # for many aten op kinds, so override those too.
    for tag in ("bitwise_and", "and"):
        _TORCH_OPS_REGISTRY.set_func_by_name(make_logical("and"), tag)
    for tag in ("bitwise_or", "or"):
        _TORCH_OPS_REGISTRY.set_func_by_name(make_logical("or"), tag)


def _override_layer_norm_translator() -> None:
    """Override the torch frontend's ``layer_norm`` translator so gamma/beta/
    epsilon share one dtype before reaching ``mb.layer_norm``.

    coremltools 9.0's iOS17/iOS18 layer_norm op enforces this at validation
    time. FP16-traced models hit it because the python-scalar ``eps`` arg
    survives as fp32 while gamma/beta come through as fp16.
    """
    import numpy as np
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
    )
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil import Builder as mb

    def _dtype_name(v):
        d = getattr(v, "dtype", None)
        return getattr(d, "__name__", str(d) if d else "")

    def _to_typed_const(v, target):
        """Force v to a const Var of dtype `target`. Eps stays a const so
        later MIL passes (add_fp16_cast) don't promote it back to fp32."""
        if v is None:
            return v
        np_dtype = np.float16 if target == "fp16" else np.float32
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return mb.const(val=np_dtype(v))
        if hasattr(v, "val") and v.val is not None and hasattr(v, "dtype"):
            try:
                return mb.const(val=np.asarray(v.val, dtype=np_dtype))
            except Exception:
                pass
        if hasattr(v, "dtype") and _dtype_name(v) != target:
            return mb.cast(x=v, dtype=target)
        return v

    def layer_norm(context, node):
        inputs = _get_inputs(context, node, min_expected=2)
        nargs = len(inputs)
        x = inputs[0]
        normalized_shape = inputs[1]
        weight = inputs[2] if nargs > 2 else None
        bias = inputs[3] if nargs > 3 else None
        eps = inputs[4] if nargs > 4 else None
        if eps is None:
            eps = 1e-5

        ref = weight if weight is not None else x
        ref_name = _dtype_name(ref)
        target = "fp16" if ("fp16" in ref_name or "float16" in ref_name) else (
            "fp32" if ("fp32" in ref_name or "float32" in ref_name) else "fp16"
        )

        # x is a runtime tensor — keep as cast Var; gamma/beta/eps are const-like.
        if x is not None and hasattr(x, "dtype") and _dtype_name(x) != target:
            x = mb.cast(x=x, dtype=target)
        weight = _to_typed_const(weight, target)
        bias = _to_typed_const(bias, target)
        eps = _to_typed_const(eps, target)

        out = mb.layer_norm(
            x=x,
            axes=list(range(-len(normalized_shape.val), 0)),
            gamma=weight,
            beta=bias,
            epsilon=eps,
            name=node.name,
        )

        if node.kind == "native_layer_norm":
            context.add((out, None, None), torch_name=node.name)
        else:
            context.add(out)

    _TORCH_OPS_REGISTRY.set_func_by_name(layer_norm, "layer_norm")
    _TORCH_OPS_REGISTRY.set_func_by_name(layer_norm, "native_layer_norm")


def _patch_pipeline_remove_fuse_prelu() -> None:
    """No-op placeholder. ``PassPipeline.DEFAULT`` is a classproperty that
    returns a fresh instance each time, so mutating it in-place doesn't
    persist. We instead build a trimmed pipeline at convert time via
    :func:`build_cactus_pass_pipeline` and pass it to ``ct.convert``.
    """
    return


_PASSES_TO_DROP = [
    # fuse_prelu crashes on fp64 const tables in coremltools 9.0.
    "common::fuse_prelu",
]


def build_cactus_pass_pipeline():
    """Construct a fresh PassPipeline with our problem passes removed.

    Pass this to ``ct.convert(..., pass_pipeline=pipeline)`` — mutating
    ``PassPipeline.DEFAULT`` doesn't work because that's a classproperty
    returning a new instance every time.
    """
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
    pipeline = PassPipeline.DEFAULT
    for pass_name in _PASSES_TO_DROP:
        try:
            pipeline.remove_passes([pass_name])
        except Exception:
            pass
    return pipeline


def _patch_mb_binops_scalar_cast() -> None:
    """Auto-align dtypes for mb.{sub,mul,add,div} so SDPA mask paths don't
    crash on mixed fp16/fp32 inputs. Casts both scalars and tensors to
    match whichever operand is already in the lower-precision dtype."""
    from coremltools.converters.mil import Builder as mb

    def _dtype_name(v):
        d = getattr(v, "dtype", None)
        return getattr(d, "__name__", str(d) if d else "")

    def _is_fp(name):
        return "fp" in name or "float" in name or "double" in name

    for op_name in ("sub", "mul", "add", "div", "real_div"):
        original = getattr(mb, op_name, None)
        if original is None or getattr(original, "_cactus_scalar_cast_patched", False):
            continue

        def make_wrapper(orig):
            def wrapper(**kwargs):
                x = kwargs.get("x")
                y = kwargs.get("y")
                xn = _dtype_name(x)
                yn = _dtype_name(y)
                # If only one is a tensor, cast scalars to match it.
                if xn and not yn and _is_fp(xn) and isinstance(y, (int, float)) and not isinstance(y, bool):
                    kwargs["y"] = mb.cast(x=y, dtype=xn)
                elif yn and not xn and _is_fp(yn) and isinstance(x, (int, float)) and not isinstance(x, bool):
                    kwargs["x"] = mb.cast(x=x, dtype=yn)
                elif xn and yn and xn != yn:
                    # Both tensors, mismatched. Prefer fp16 to keep memory low.
                    target = "fp16" if (xn == "fp16" or yn == "fp16") else xn
                    if xn != target:
                        kwargs["x"] = mb.cast(x=x, dtype=target)
                    if yn != target:
                        kwargs["y"] = mb.cast(x=y, dtype=target)
                return orig(**kwargs)
            wrapper._cactus_scalar_cast_patched = True
            wrapper.__name__ = orig.__name__
            return wrapper

        setattr(mb, op_name, make_wrapper(original))

    # layer_norm: epsilon (scalar) must match gamma/beta dtype. Parakeet's
    # Conformer norms produce fp16 gamma but coremltools emits fp32 epsilon.
    orig_ln = getattr(mb, "layer_norm", None)
    if orig_ln is not None and not getattr(orig_ln, "_cactus_scalar_cast_patched", False):
        def _ln_wrap(orig):
            def wrapper(**kwargs):
                gamma = kwargs.get("gamma")
                gn = _dtype_name(gamma)
                if _is_fp(gn):
                    eps = kwargs.get("epsilon")
                    if isinstance(eps, (int, float)) and not isinstance(eps, bool):
                        kwargs["epsilon"] = mb.cast(x=eps, dtype=gn)
                    elif hasattr(eps, "dtype") and _dtype_name(eps) != gn:
                        kwargs["epsilon"] = mb.cast(x=eps, dtype=gn)
                return orig(**kwargs)
            wrapper._cactus_scalar_cast_patched = True
            wrapper.__name__ = orig.__name__
            return wrapper
        setattr(mb, "layer_norm", _ln_wrap(orig_ln))

    # batch_norm / instance_norm have similar epsilon shapes
    for op_name in ("batch_norm", "instance_norm", "rms_norm"):
        orig_op = getattr(mb, op_name, None)
        if orig_op is None or getattr(orig_op, "_cactus_scalar_cast_patched", False):
            continue
        def _bn_wrap(orig):
            def wrapper(**kwargs):
                # Reference dtype: gamma if present, else x
                ref = kwargs.get("gamma") or kwargs.get("x")
                rn = _dtype_name(ref)
                if _is_fp(rn):
                    eps = kwargs.get("epsilon")
                    if isinstance(eps, (int, float)) and not isinstance(eps, bool):
                        kwargs["epsilon"] = mb.cast(x=eps, dtype=rn)
                    elif hasattr(eps, "dtype") and _dtype_name(eps) != rn:
                        kwargs["epsilon"] = mb.cast(x=eps, dtype=rn)
                return orig(**kwargs)
            wrapper._cactus_scalar_cast_patched = True
            wrapper.__name__ = orig.__name__
            return wrapper
        setattr(mb, op_name, _bn_wrap(orig_op))


def _patch_fp16_cast_skip_layer_norm() -> None:
    """Tell the ``common::add_fp16_cast`` MIL pass to skip ``layer_norm`` and
    ``batch_norm``-family ops. The pass casts gamma to fp16 but leaves
    epsilon as fp32, which violates the iOS17+ ``layer_norm`` constraint that
    gamma, beta, and epsilon share one dtype.

    By marking them unsupported, we keep them at their original (fp32) dtype.
    """
    try:
        from coremltools.converters.mil.mil.passes.defs.quantization import (
            FP16ComputePrecision,
        )
    except Exception as exc:
        print(f"npu.coremltools_patches: skip_layer_norm patch unavailable ({exc})")
        return
    base = set(FP16ComputePrecision._UNSUPPORTED_FP16_OPS)
    for op in ("layer_norm", "batch_norm", "instance_norm", "rms_norm"):
        base.add(op)
    FP16ComputePrecision._UNSUPPORTED_FP16_OPS = base
