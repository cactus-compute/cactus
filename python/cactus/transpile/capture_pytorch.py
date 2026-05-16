from dataclasses import dataclass
import inspect
from contextlib import contextmanager
from typing import Any
from typing import Iterable

import torch
from torch.export import export
from torch.fx.passes.shape_prop import ShapeProp

from cactus.transpile.aten_ops import canonical_torch_op
from cactus.transpile.graph_ir import IRGraph


class CapturePhaseError(RuntimeError):
    def __init__(self, phase: str, message: str, *, cause: Exception | None = None):
        super().__init__(f"[capture:{phase}] {message}")
        self.phase = phase
        self.cause = cause


@dataclass
class TensorMetadata:
    shape: tuple[Any, ...]
    dtype: Any
    stride: tuple[Any, ...] | None = None
    device: Any | None = None


@dataclass
class CapturedModel:
    exported_program: Any
    graph_module: Any
    graph: Any
    state_dict: dict[str, Any]
    ir_graph: IRGraph
    source_module: Any
    example_args: tuple[Any, ...]
    example_kwargs: dict[str, Any]
    strict: bool
    transpile_metadata: dict[str, Any]

    def named_parameters(self):
        return self.exported_program.named_parameters()

    def named_buffers(self):
        return self.exported_program.named_buffers()

    def parameters_dict(self) -> dict[str, Any]:
        return dict(self.named_parameters())

    def buffers_dict(self) -> dict[str, Any]:
        return dict(self.named_buffers())

    def placeholders(self) -> list[Any]:
        return [node for node in self.graph.nodes if node.op == "placeholder"]

    def outputs(self) -> list[Any]:
        return [node for node in self.graph.nodes if node.op == "output"]


def _normalize_args(args: Any) -> tuple[Any, ...]:
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    if isinstance(args, list):
        return tuple(args)
    return (args,)


def _normalize_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    return {} if kwargs is None else dict(kwargs)


def _clone_example_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_example_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_example_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_example_value(v) for k, v in value.items()}
    return value


def _clone_examples(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (
        tuple(_clone_example_value(arg) for arg in args),
        {k: _clone_example_value(v) for k, v in kwargs.items()},
    )


def _inject_export_safe_kwargs(model: torch.nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return kwargs

    updated = dict(kwargs)
    parameters = signature.parameters

    # HF causal LM forwards often default to returning cache objects and ModelOutput
    # containers, which torch.export cannot serialize cleanly. Keep the capture
    # boundary tensor-only unless the caller explicitly asked otherwise.
    if "use_cache" in parameters and "use_cache" not in updated:
        updated["use_cache"] = False
    if "return_dict" in parameters and "return_dict" not in updated:
        updated["return_dict"] = False
    return updated


def _call_transpile_metadata_provider(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    for attr_name in ("get_transpile_metadata", "transpile_metadata"):
        provider = getattr(module, attr_name, None)
        if not callable(provider):
            continue
        try:
            metadata = provider(args=args, kwargs=kwargs)
        except TypeError:
            metadata = provider()
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise TypeError(f"{type(module).__name__}.{attr_name} must return a dict or None")
        return metadata
    return {}


def _collect_transpile_metadata(model: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    graph_meta: dict[str, Any] = {}
    provider_graph_meta: dict[str, dict[str, Any]] = {}
    seen_modules: set[int] = set()

    for module_path, module in model.named_modules():
        module_id = id(module)
        if module_id in seen_modules:
            continue
        seen_modules.add(module_id)

        metadata = _call_transpile_metadata_provider(module, args, kwargs)
        if not metadata:
            continue

        provider_key = module_path or type(module).__name__
        provider_meta = metadata.get("graph", {})
        if isinstance(provider_meta, dict) and provider_meta:
            provider_graph_meta[provider_key] = dict(provider_meta)
            if not module_path:
                graph_meta.update(provider_meta)

    if provider_graph_meta:
        graph_meta.setdefault("transpile_metadata_providers", provider_graph_meta)

    if not graph_meta:
        return {}
    return {"graph": graph_meta}


def _prepare_import_graph_module(ep: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    graph_module = ep.graph_module
    if kwargs:
        return graph_module

    try:
        graph_module = ep.module()
        ShapeProp(graph_module).propagate(*args)
    except Exception:
        return ep.graph_module
    return graph_module


@contextmanager
def _suppress_transformers_model_output_registration():
    """Avoid tracing a Transformers global-set mutation inside ModelOutput init.

    Recent Transformers versions lazily register ModelOutput dataclasses as
    PyTrees from their __post_init__. That side effect is irrelevant for our
    capture wrapper because the exported boundary returns tensors, but Dynamo
    sees the intermediate set membership check and rejects it.
    """

    try:
        import transformers.utils.generic as hf_generic
    except Exception:
        yield
        return

    original = getattr(hf_generic, "_register_model_output_pytree_node", None)
    if original is None:
        yield
        return

    def _noop_register_model_output_pytree_node(*_args: Any, **_kwargs: Any) -> None:
        return None

    hf_generic._register_model_output_pytree_node = _noop_register_model_output_pytree_node
    try:
        yield
    finally:
        hf_generic._register_model_output_pytree_node = original


def capture_model(model, args, kwargs=None, *, strict=True) -> CapturedModel:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    normalized_args = _normalize_args(args)
    normalized_kwargs = _inject_export_safe_kwargs(model, _normalize_kwargs(kwargs))

    if not normalized_args and not normalized_kwargs:
        raise ValueError("capture_model requires example args or kwargs")

    model = model.eval()
    example_args, example_kwargs = _clone_examples(normalized_args, normalized_kwargs)
    transpile_metadata = _collect_transpile_metadata(model, example_args, example_kwargs)

    try:
        with _suppress_transformers_model_output_registration():
            ep = export(model, args=example_args, kwargs=example_kwargs, strict=strict)
    except Exception as exc:
        raise CapturePhaseError(
            "export",
            f"torch.export failed for model={type(model).__name__} strict={strict}: {exc}",
            cause=exc,
        ) from exc
    from cactus.transpile.import_ir import import_captured_to_ir

    import_graph_module = _prepare_import_graph_module(ep, example_args, example_kwargs)

    raw_captured = CapturedModel(
        exported_program=ep,
        graph_module=import_graph_module,
        graph=import_graph_module.graph,
        state_dict=dict(ep.state_dict),
        ir_graph=IRGraph(values={}, nodes={}, order=[], inputs=[], outputs=[]),
        source_module=model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        strict=strict,
        transpile_metadata=transpile_metadata,
    )
    try:
        ir_graph = import_captured_to_ir(raw_captured, strict=strict)
    except Exception as exc:
        raise CapturePhaseError(
            "import",
            f"failed to import exported graph to IR for model={type(model).__name__} strict={strict}: {exc}",
            cause=exc,
        ) from exc

    return CapturedModel(
        exported_program=ep,
        graph_module=ep.graph_module,
        graph=ep.graph,
        state_dict=dict(ep.state_dict),
        ir_graph=ir_graph,
        source_module=model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        strict=strict,
        transpile_metadata=transpile_metadata,
    )


def capture_model_with_fallback(model, args, kwargs=None) -> CapturedModel:
    try:
        return capture_model(model, args, kwargs=kwargs, strict=True)
    except Exception:
        return capture_model(model, args, kwargs=kwargs, strict=False)


def resolve_attr(root: Any, target: str) -> Any:
    obj = root
    for atom in target.split("."):
        obj = getattr(obj, atom)
    return obj


def format_target(node: Any) -> str:
    return canonical_torch_op(node.target)


def _try_materialize_int_tuple(values: Iterable[Any]) -> tuple[int, ...] | None:
    materialized: list[int] = []
    try:
        for value in values:
            materialized.append(int(value))
    except Exception:
        return None
    return tuple(materialized)


def get_tensor_metadata(node: Any) -> TensorMetadata | None:
    meta = getattr(node, "meta", {}) or {}
    tensor_meta = meta.get("tensor_meta")
    if tensor_meta is not None:
        shape = tuple(getattr(tensor_meta, "shape", ()))
        dtype = getattr(tensor_meta, "dtype", None)
        stride_value = getattr(tensor_meta, "stride", None)
        stride = _try_materialize_int_tuple(stride_value) if isinstance(stride_value, Iterable) else None
        device = getattr(tensor_meta, "device", None)
        if shape or dtype is not None or stride is not None or device is not None:
            return TensorMetadata(
                shape=shape,
                dtype=dtype,
                stride=stride,
                device=device,
            )

    value = meta.get("val")
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        stride = _try_materialize_int_tuple(value.stride()) if value.layout == torch.strided else None
        return TensorMetadata(
            shape=tuple(value.shape),
            dtype=value.dtype,
            stride=stride,
            device=value.device,
        )

    shape = tuple(getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)
    stride_value = getattr(value, "stride", None)
    stride = tuple(stride_value) if isinstance(stride_value, Iterable) else None
    device = getattr(value, "device", None)
    if not shape and dtype is None and stride is None and device is None:
        return None
    return TensorMetadata(shape=shape, dtype=dtype, stride=stride, device=device)


def get_shape(node: Any) -> tuple[Any, ...] | None:
    metadata = get_tensor_metadata(node)
    return None if metadata is None else metadata.shape


def get_dtype(node: Any) -> Any | None:
    metadata = get_tensor_metadata(node)
    return None if metadata is None else metadata.dtype


def dump_graph(captured: CapturedModel, *, include_meta: bool = True) -> str:
    lines: list[str] = []
    for index, node in enumerate(captured.graph.nodes):
        lines.append(f"[{index}] {node.op} {node.name} target={node.target}")
        lines.append(f"  args={node.args}")
        lines.append(f"  kwargs={node.kwargs}")
        if include_meta:
            tensor_meta = get_tensor_metadata(node)
            if tensor_meta is not None:
                lines.append(
                    "  tensor_meta="
                    f"shape={tensor_meta.shape} dtype={tensor_meta.dtype} "
                    f"stride={tensor_meta.stride} device={tensor_meta.device}"
                )
            else:
                lines.append(f"  meta={getattr(node, 'meta', {})}")
    return "\n".join(lines)


def print_graph(captured: CapturedModel, *, include_meta: bool = True) -> None:
    print(dump_graph(captured, include_meta=include_meta))
