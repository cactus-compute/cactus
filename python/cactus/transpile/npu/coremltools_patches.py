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
        _register_noop_assert_ops()
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

    for tag, op_kind in [("__and_", "and"), ("__or_", "or"), ("__xor_", "xor")]:
        if tag not in _TORCH_OPS_REGISTRY.name_to_func_mapping:
            _TORCH_OPS_REGISTRY.set_func_by_name(make_logical(op_kind), tag)
    for tag, op_kind in [("__and__", "and"), ("__or__", "or"), ("__xor__", "xor")]:
        if tag not in _TORCH_OPS_REGISTRY.name_to_func_mapping:
            _TORCH_OPS_REGISTRY.set_func_by_name(make_logical(op_kind), tag)
    for tag in ("bitwise_and", "and"):
        _TORCH_OPS_REGISTRY.set_func_by_name(make_logical("and"), tag)
    for tag in ("bitwise_or", "or"):
        _TORCH_OPS_REGISTRY.set_func_by_name(make_logical("or"), tag)


def _override_layer_norm_translator() -> None:
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
    return


_PASSES_TO_DROP = [
    "common::fuse_prelu",
]


def build_cactus_pass_pipeline():
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
    pipeline = PassPipeline.DEFAULT
    for pass_name in _PASSES_TO_DROP:
        try:
            pipeline.remove_passes([pass_name])
        except Exception:
            pass
    return pipeline


def _patch_mb_binops_scalar_cast() -> None:
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
        return hasattr(v, "op") and hasattr(v, "sym_type")

    def _retype_in_place(v, target_dtype_name):
        np_dtype = _np_for(target_dtype_name)
        if isinstance(v, np.ndarray):
            return v.astype(np_dtype)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return np_dtype(v)
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
                if xn and not yn and _is_fp(xn) and isinstance(y, (int, float)) and not isinstance(y, bool):
                    kwargs["y"] = _scalar_const(y, xn)
                elif yn and not xn and _is_fp(yn) and isinstance(x, (int, float)) and not isinstance(x, bool):
                    kwargs["x"] = _scalar_const(x, yn)
                elif xn and yn and xn != yn:
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

    for op_name in ("batch_norm", "instance_norm", "rms_norm"):
        orig_op = getattr(mb, op_name, None)
        if orig_op is None or getattr(orig_op, "_cactus_scalar_cast_patched", False):
            continue
        def _bn_wrap(orig):
            def wrapper(**kwargs):
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


def _register_noop_assert_ops() -> None:
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
    )

    def _noop(context, node):
        return

    _TORCH_OPS_REGISTRY.set_func_by_name(_noop, "_assert_tensor_metadata")


def _register_unfold_op() -> None:
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
