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
        _override_one_hot_translator()
        _register_unfold_op()
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
    crash on mixed fp16/fp32 inputs.

    Critical: only Var operands get a ``mb.cast`` injected — numpy
    arrays/scalars are retyped via ``numpy.astype`` because injecting a
    fresh ``mb.cast`` Var from inside MIL passes lands it in the wrong
    block scope (e.g. ``divide_to_multiply`` calling ``mb.mul`` with
    ``before_op=`` on an op nested in ``block0``). That mis-scoping fails
    the "var visibility" validator and trips a hard ValueError mid-pass.
    Python scalars are materialized as typed ``mb.const`` for the same
    reason."""
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    def _dtype_name(v):
        d = getattr(v, "dtype", None)
        return getattr(d, "__name__", str(d) if d else "")

    def _is_fp(name):
        return "fp" in name or "float" in name or "double" in name

    def _np_for(dtype_name):
        if "fp16" in dtype_name or "float16" in dtype_name:
            return np.float16
        if "fp32" in dtype_name or "float32" in dtype_name:
            return np.float32
        if "fp64" in dtype_name or "float64" in dtype_name or "double" in dtype_name:
            return np.float64
        return np.float32

    def _is_var(v):
        # MIL Vars carry sym_type / op fields; numpy arrays don't.
        return hasattr(v, "op") and hasattr(v, "sym_type")

    def _retype_in_place(v, target_dtype_name):
        """Convert a *non-Var* operand (numpy array/scalar/python scalar) to
        ``target_dtype_name``. Returning a fresh numpy value (not a new Var)
        keeps the operand inline so MIL passes don't see a stray Var in the
        wrong block."""
        np_dtype = _np_for(target_dtype_name)
        if isinstance(v, np.ndarray):
            return v.astype(np_dtype)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return np_dtype(v)
        # Best-effort fallback (lists, etc.)
        try:
            return np.asarray(v, dtype=np_dtype)
        except Exception:
            return v

    def _scalar_const(s, dtype_name):
        return mb.const(val=_np_for(dtype_name)(s))

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
                # If only one is a tensor, materialize scalars as typed consts.
                if xn and not yn and _is_fp(xn) and isinstance(y, (int, float)) and not isinstance(y, bool):
                    kwargs["y"] = _scalar_const(y, xn)
                elif yn and not xn and _is_fp(yn) and isinstance(x, (int, float)) and not isinstance(x, bool):
                    kwargs["x"] = _scalar_const(x, yn)
                elif xn and yn and xn != yn:
                    # Both have a dtype but mismatched. Prefer fp16.
                    target = "fp16" if (xn == "fp16" or yn == "fp16") else xn
                    if xn != target:
                        kwargs["x"] = mb.cast(x=x, dtype=target) if _is_var(x) else _retype_in_place(x, target)
                    if yn != target:
                        kwargs["y"] = mb.cast(x=y, dtype=target) if _is_var(y) else _retype_in_place(y, target)
                return orig(**kwargs)
            wrapper._cactus_scalar_cast_patched = True
            wrapper.__name__ = orig.__name__
            return wrapper

        setattr(mb, op_name, make_wrapper(original))

    # mb.select(cond, a, b) requires a and b to share a dtype. Gemma 4 vision
    # hits this via `torch.where(cond, scalar, tensor)` — the scalar becomes
    # fp32 while the tensor is fp16. Materialize scalars as typed consts.
    select_op = getattr(mb, "select", None)
    if select_op is not None and not getattr(select_op, "_cactus_scalar_cast_patched", False):
        def _select_wrap(orig):
            def wrapper(**kwargs):
                a = kwargs.get("a")
                b = kwargs.get("b")
                an = _dtype_name(a)
                bn = _dtype_name(b)
                if an and not bn and _is_fp(an) and isinstance(b, (int, float)) and not isinstance(b, bool):
                    kwargs["b"] = _scalar_const(b, an)
                elif bn and not an and _is_fp(bn) and isinstance(a, (int, float)) and not isinstance(a, bool):
                    kwargs["a"] = _scalar_const(a, bn)
                elif an and bn and an != bn:
                    target = "fp16" if (an == "fp16" or bn == "fp16") else an
                    if an != target:
                        kwargs["a"] = mb.cast(x=a, dtype=target) if _is_var(a) else _retype_in_place(a, target)
                    if bn != target:
                        kwargs["b"] = mb.cast(x=b, dtype=target) if _is_var(b) else _retype_in_place(b, target)
                return orig(**kwargs)
            wrapper._cactus_scalar_cast_patched = True
            wrapper.__name__ = orig.__name__
            return wrapper
        setattr(mb, "select", _select_wrap(select_op))

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
                        kwargs["epsilon"] = _scalar_const(eps, gn)
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
                        kwargs["epsilon"] = _scalar_const(eps, rn)
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


def _override_one_hot_translator() -> None:
    """Override ``one_hot`` to cast indices to int32 before calling ``mb.one_hot``.

    The MIL ``one_hot`` op requires int32 indices, but Gemma 4 vision feeds
    ``pixel_position_ids`` as fp32 through ``torch.export``. The default
    coremltools translator passes the labels through unchanged, which trips
    the type domain check.
    """
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
    )
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs, _get_kwinputs
    from coremltools.converters.mil import Builder as mb

    def _dtype_name(v):
        d = getattr(v, "dtype", None)
        return getattr(d, "__name__", str(d) if d else "")

    def one_hot(context, node):
        inputs = _get_inputs(context, node, expected=(1, 2))
        labels = inputs[0]
        num_classes = inputs[1] if len(inputs) > 1 else -1
        num_classes = _get_kwinputs(context, node, "num_classes", default=[num_classes])[0]
        if hasattr(num_classes, "val") and num_classes.val is not None:
            num_classes = num_classes.val

        if hasattr(labels, "dtype") and _dtype_name(labels) != "int32":
            labels = mb.cast(x=labels, dtype="int32")

        res = mb.one_hot(indices=labels, one_hot_vector_size=num_classes, name=node.name)
        context.add(res)

    _TORCH_OPS_REGISTRY.set_func_by_name(one_hot, "one_hot")


def _register_unfold_op() -> None:
    """Translate ``Tensor.unfold(dimension, size, step)`` to ``mb.sliding_windows``.

    Layout note: PyTorch's ``unfold`` *appends* the window-size dim at the
    end of the output (rank N+1). MIL's ``mb.sliding_windows`` *inserts*
    the size dim at ``axis+1`` instead. We translate to ``sliding_windows``
    then ``mb.transpose`` the size dim to the trailing position so downstream
    permutes/movedims see the shape PyTorch produced.

    Gemma 4 audio's ``_extract_block_context`` is the canonical caller and
    deeply depends on this layout (size at last axis, then ``movedim(-1, 2)``).
    """
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
    )
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil import Builder as mb

    def unfold(context, node):
        inputs = _get_inputs(context, node, min_expected=4)
        x = inputs[0]
        dimension = inputs[1].val if hasattr(inputs[1], "val") else int(inputs[1])
        size = inputs[2].val if hasattr(inputs[2], "val") else int(inputs[2])
        step = inputs[3].val if hasattr(inputs[3], "val") else int(inputs[3])

        rank = len(x.shape) if hasattr(x, "shape") else x.rank
        dim = int(dimension)
        if dim < 0:
            dim += rank

        windowed = mb.sliding_windows(
            x=x,
            axis=dim,
            size=int(size),
            stride=int(step),
        )

        # sliding_windows inserts the window-size dim at axis+1; PyTorch
        # unfold appends it at the end. Permute (size_axis) to the last axis.
        out_rank = rank + 1
        perm = list(range(out_rank))
        size_axis = dim + 1
        perm.pop(size_axis)
        perm.append(size_axis)
        if perm == list(range(out_rank)):
            out = mb.identity(x=windowed, name=node.name)
        else:
            out = mb.transpose(x=windowed, perm=perm, name=node.name)
        context.add(out)

    _TORCH_OPS_REGISTRY.set_func_by_name(unfold, "unfold")
