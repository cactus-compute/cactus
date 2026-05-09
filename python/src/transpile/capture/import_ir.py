from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch.export.graph_signature import ConstantArgument
from torch.export.graph_signature import InputKind

from src.transpile.capture.capture_pytorch import format_target
from src.transpile.capture.capture_pytorch import get_dtype
from src.transpile.capture.capture_pytorch import get_shape
from src.transpile.capture.capture_pytorch import resolve_attr
from src.transpile.capture.graph_ir import IRGraph
from src.transpile.capture.graph_ir import verify_ir
from src.transpile.capture.importers import extract_literals
from src.transpile.capture.importers import ImportContext
from src.transpile.capture.importers import import_call_function
from src.transpile.capture.importers import import_get_attr
from src.transpile.capture.importers import import_output
from src.transpile.capture.importers import import_placeholder
from src.transpile.capture.importers import value_id
from src.transpile.capture.normalize import dtype_to_ir
from src.transpile.capture.import_semantics import apply_import_semantics
from src.transpile.capture.weight_binding import resolve_transpile_weights_dir
from src.transpile.capture.weight_binding import resolve_weight_binding


def import_captured_to_ir(captured: Any, *, strict: bool = True) -> IRGraph:
    transpile_metadata = getattr(captured, "transpile_metadata", {}) or {}
    graph_meta = transpile_metadata.get("graph", {})
    ir = IRGraph(values={}, nodes={}, order=[], inputs=[], outputs=[], constants={}, meta=dict(graph_meta))
    ctx = ImportContext(strict=strict, transpile_metadata=transpile_metadata)
    weights_dir = resolve_transpile_weights_dir(ir.meta)
    if weights_dir:
        ir.meta["weights_dir"] = weights_dir
    placeholder_specs: dict[str, Any] = {}
    inline_counter = 0

    def _try_register_weight_binding(value_id_str: str, target: str, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            return
        binding = resolve_weight_binding(
            weights_dir=weights_dir,
            source_name=target,
        )
        if binding is None:
            return
        bindings = ir.meta.setdefault("weight_bindings", {})
        if isinstance(bindings, dict):
            bindings[value_id_str] = {
                "path": binding.path,
                "kind": binding.kind,
                "source_name": binding.source_name,
            }
        if value_id_str in ir.values:
            ir.values[value_id_str].meta.update(
                {
                    "path": binding.path,
                    "kind": binding.kind,
                    "source_name": binding.source_name,
                }
            )

    def _make_inlined_node(node: Any, *, inline_index: int) -> Any:
        return SimpleNamespace(
            name=f"inl_{inline_index}_{node.name}",
            op=node.op,
            target=node.target,
            args=node.args,
            kwargs=getattr(node, "kwargs", {}),
            meta=getattr(node, "meta", {}),
        )

    def _resolve_target(target: str) -> Any:
        candidates: list[str] = []
        parts = target.split(".")
        for i in range(len(parts)):
            candidate = ".".join(parts[i:])
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        exported_program = getattr(captured, "exported_program", None)
        if exported_program is not None:
            tensor_constants = getattr(exported_program, "tensor_constants", None)
            if tensor_constants is not None:
                for candidate in candidates:
                    if candidate in tensor_constants:
                        return tensor_constants[candidate]
            constants = getattr(exported_program, "constants", None)
            if constants is not None:
                for candidate in candidates:
                    if candidate in constants:
                        return constants[candidate]

        for candidate in candidates:
            try:
                return resolve_attr(captured.graph_module, candidate)
            except AttributeError:
                pass

        for candidate in candidates:
            if candidate in captured.state_dict:
                return captured.state_dict[candidate]

        source_module = getattr(captured, "source_module", None)
        if source_module is not None:
            for candidate in candidates:
                try:
                    return resolve_attr(source_module, candidate)
                except AttributeError:
                    pass

        raise NotImplementedError(f"could not resolve get_attr target: {target}") from None

    def _collect_output_ids(arg: Any, local_alias: dict[str, str]) -> list[str]:
        if hasattr(arg, "name") and hasattr(arg, "op"):
            name = arg.name
            if name in local_alias:
                return [local_alias[name]]
            return [ctx.resolve_value_id(arg)]
        if isinstance(arg, (tuple, list)):
            out: list[str] = []
            for item in arg:
                out.extend(_collect_output_ids(item, local_alias))
            return out
        return []

    def _inline_wrap_with_set_grad_enabled(node: Any, shape: Any, dtype: Any) -> None:
        nonlocal inline_counter
        subgraph_ref = node.args[1]
        if not hasattr(subgraph_ref, "op") or subgraph_ref.op != "get_attr":
            raise NotImplementedError("wrap_with_set_grad_enabled without get_attr submodule is not supported")

        subgraph_module = _resolve_target(subgraph_ref.target)
        if not hasattr(subgraph_module, "graph"):
            raise NotImplementedError("wrap_with_set_grad_enabled target is not a GraphModule")

        inline_counter += 1
        inline_index = inline_counter
        local_alias: dict[str, str] = {}
        subgraph_inputs = list(node.args[2:])
        input_index = 0

        ctx.push_local_aliases(local_alias)
        try:
            for sub_node in subgraph_module.graph.nodes:
                sub_shape = get_shape(sub_node)
                sub_dtype = dtype_to_ir(get_dtype(sub_node))
                sub_desc = (
                    f"inlined:{node.name}:{sub_node.op}:{getattr(sub_node, 'name', '<unnamed>')}:"
                    f"{format_target(sub_node) if sub_node.op == 'call_function' else getattr(sub_node, 'target', '')}"
                )
                ctx.push_node(sub_desc)
                try:
                    if sub_node.op == "placeholder":
                        if input_index >= len(subgraph_inputs):
                            raise NotImplementedError("subgraph placeholder/input arity mismatch during inlining")
                        local_alias[sub_node.name] = ctx.resolve_value_id(subgraph_inputs[input_index])
                        input_index += 1
                        continue

                    if sub_node.op == "get_attr":
                        prefixed = _make_inlined_node(sub_node, inline_index=inline_index)
                        value = _resolve_target(sub_node.target)
                        import_get_attr(ir, prefixed, ctx, value, shape=sub_shape, dtype=sub_dtype, source_name=str(sub_node.target))
                        _try_register_weight_binding(f"v_{prefixed.name}", str(sub_node.target), value)
                        local_alias[sub_node.name] = f"v_{prefixed.name}"
                        continue

                    if sub_node.op == "call_function":
                        prefixed = _make_inlined_node(sub_node, inline_index=inline_index)
                        import_call_function(
                            ir,
                            prefixed,
                            ctx,
                            shape=sub_shape,
                            dtype=sub_dtype,
                            torch_op=format_target(sub_node),
                        )
                        local_alias[sub_node.name] = f"v_{prefixed.name}"
                        continue

                    if sub_node.op == "output":
                        ctx.tuple_aliases[node.name] = _collect_output_ids(sub_node.args[0], local_alias)
                        return

                    raise NotImplementedError(f"unsupported inlined subgraph node op: {sub_node.op}")
                finally:
                    ctx.pop_node()
        finally:
            ctx.pop_local_aliases()

        raise NotImplementedError("inlined subgraph had no output node")

    for spec in captured.exported_program.graph_signature.input_specs:
        arg = getattr(spec, "arg", None)
        name = getattr(arg, "name", None)
        if name is not None:
            placeholder_specs[name] = spec

    for node in captured.graph.nodes:
        shape = get_shape(node)
        dtype = dtype_to_ir(get_dtype(node))
        node_desc = f"top:{node.op}:{getattr(node, 'name', '<unnamed>')}:{getattr(node, 'target', '')}"
        ctx.push_node(node_desc)
        try:
            if node.op == "placeholder":
                spec = placeholder_specs.get(node.name)
                if spec is not None and isinstance(getattr(spec, "arg", None), ConstantArgument):
                    import_get_attr(
                        ir,
                        node,
                        ctx,
                        spec.arg.value,
                        shape=shape,
                        dtype=dtype,
                        source_name=node.name,
                    )
                    continue
                if spec is not None and spec.kind in {InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR}:
                    target = getattr(spec, "target", None)
                    if target is None:
                        raise NotImplementedError(f"placeholder {node.name} has no target for lifted constant input")
                    value = _resolve_target(target)
                    import_get_attr(ir, node, ctx, value, shape=shape, dtype=dtype, source_name=str(target))
                    _try_register_weight_binding(value_id(node, ctx), str(target), value)
                    continue
                import_placeholder(ir, node, ctx, shape=shape, dtype=dtype)
                continue

            if node.op == "get_attr":
                if str(node.target).startswith("submod_"):
                    continue
                value = _resolve_target(node.target)
                import_get_attr(ir, node, ctx, value, shape=shape, dtype=dtype, source_name=str(node.target))
                _try_register_weight_binding(value_id(node, ctx), str(node.target), value)
                continue

            if node.op == "call_function":
                torch_op = format_target(node)
                if "wrap_with_set_grad_enabled" in torch_op:
                    _inline_wrap_with_set_grad_enabled(node, shape, dtype)
                    continue
                if torch_op == "<built-in function getitem>":
                    source = node.args[0]
                    if hasattr(source, "name") and source.name in ctx.tuple_aliases:
                        index_value = extract_literals(node.args[1])
                        if not isinstance(index_value, int):
                            raise NotImplementedError(f"unsupported tuple getitem index: {node.args!r}")
                        ctx.alias_values[node.name] = ctx.tuple_aliases[source.name][index_value]
                        continue
                import_call_function(ir, node, ctx, shape=shape, dtype=dtype, torch_op=torch_op)
                continue

            if node.op == "output":
                import_output(ir, node, ctx)
                continue

            raise NotImplementedError(f"unsupported FX node op: {node.op}")
        finally:
            ctx.pop_node()

    apply_import_semantics(ir)
    verify_ir(ir)
    return ir
