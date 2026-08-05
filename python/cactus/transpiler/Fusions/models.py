from dataclasses import dataclass, field
from typing import Any

ConstraintSpec = dict[str, Any]
ConstraintValue = ConstraintSpec | tuple[ConstraintSpec, ...] | list[ConstraintSpec]

class ValueKind:
    """Names the semantic role of a real IR value."""

    UNKNOWN = "unknown"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    USER_INPUT = "user_input"
    LIFTED_CONSTANT = "lifted_constant"
    ACTIVATION = "activation"
    OUTPUT = "output"
    CACHE_INPUT = "cache_input"
    CACHE_OUTPUT = "cache_output"
    CACHE_STATE = "cache_state"

class CacheKind:
    """Names the broad family of cache/state a fusion consumes or produces."""

    KV = "kv"
    CONV = "conv"
    RECURRENT = "recurrent"

class CacheTensorRole:
    """Names the role of one tensor inside a cache family."""

    KEY = "key"
    VALUE = "value"
    STATE = "state"
    POSITION = "position"

@dataclass(slots=True)
class TensorConstraint:
    """Describes tensor properties a matched real value must satisfy."""

    rank: int | None = None
    min_rank: int | None = None
    max_rank: int | None = None
    dtype: str | None = None
    shape: tuple[Any, ...] = ()
    dim_equals: tuple[tuple[int, Any], ...] = ()
    same_dim_as: tuple[tuple[int, str, int], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AttrConstraint:
    """Describes one required attribute condition for a pattern node."""

    name: str
    value: Any = None
    source_node: str | None = None #Which node to pull attr from
    source_attr: str | None = None #Which attr from external node to pull
    comparator: str = "eq"
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AttrCapture:
    """Describes which raw-node attribute becomes a fused-node attribute."""

    name: str
    source_node: str | None = None
    source_attr: str | None = None
    default: Any = None
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class NodeRef:
    """Points to a pattern node and optionally one of its input/output positions."""

    node: str
    parent_index: int | None = None
    output_index: int | None = None

@dataclass(slots=True)
class FusionNode:
    """Describes one synthetic node inside a fusion pattern graph."""

    name: str
    ops: tuple[str, ...] = ()
    attrs: tuple[AttrConstraint, ...] = ()
    repeated: bool = False
    repeated_group: str | None = None
    allowed_value_kinds: tuple[str, ...] = ()
    tensor_constraints: tuple[TensorConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class FusionEdge:
    """Describes one directed producer-to-consumer edge in a fusion pattern."""

    source: str
    dest: str
    dest_input_index: int | None = None
    source_output_index: int | None = None
    repeated: bool = False
    repeated_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class FusionInput:
    """Describes an external input that must remain visible after fusion."""

    role: str
    source: NodeRef
    optional: bool = False
    variadic: bool = False
    min_count: int = 1
    max_count: int | None = None
    end_parent_index: int | None = None
    repeated_group: str | None = None
    allowed_value_kinds: tuple[str, ...] = ()
    tensor_constraints: tuple[TensorConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class FusionOutput:
    """Describes an output produced by the fused op."""

    role: str
    node: str
    output_index: int | None = None
    tensor_constraints: tuple[TensorConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class CacheInput:
    """Describes an old cache/state tensor consumed by a fused op."""

    role: str
    source: NodeRef
    cache_kind: str = CacheKind.KV
    tensor_role: str | None = None
    layer_index: int | None = None
    optional: bool = False
    tensor_constraints: tuple[TensorConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class CacheOutput:
    """Describes a new or updated cache/state tensor produced by a fused op."""

    role: str
    node: str
    output_index: int | None = None
    cache_kind: str = CacheKind.KV
    tensor_role: str | None = None
    layer_index: int | None = None
    optional: bool = False
    tensor_constraints: tuple[TensorConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class CacheMutation:
    """Describes read/write semantics for a cache-aware fusion."""

    name: str
    cache_kind: str = CacheKind.KV
    read_roles: tuple[str, ...] = ()
    write_roles: tuple[str, ...] = ()
    sequence_axis: int | None = -2
    window_attr: str | None = None
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class RepeatedSubgraph:
    """Describes a nested pattern that can repeat inside a larger fusion."""

    name: str
    graph: "FusionGraph"
    min_count: int = 1
    max_count: int | None = None
    anchor_node: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class FusionGraph:
    """Describes the complete synthetic DAG for one candidate fusion."""

    name: str
    root: str
    nodes: dict[str, FusionNode] = field(default_factory=dict)
    edges: tuple[FusionEdge, ...] = ()
    inputs: tuple[FusionInput, ...] = ()
    shared_inputs: tuple[tuple[NodeRef, NodeRef], ...] = ()
    outputs: tuple[FusionOutput, ...] = ()
    attr_captures: tuple[AttrCapture, ...] = ()
    repeated_subgraphs: tuple[RepeatedSubgraph, ...] = ()
    cache_inputs: tuple[CacheInput, ...] = ()
    cache_outputs: tuple[CacheOutput, ...] = ()
    cache_mutations: tuple[CacheMutation, ...] = ()
    constraints: dict[str, ConstraintValue] = field(default_factory=dict)
    variants: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_root_external_children: bool = True

@dataclass(slots=True)
class FusionDefinition:
    """Wraps a FusionGraph with Cactus-facing metadata."""

    name: str
    target: str
    graph: FusionGraph
    fusion_fields: tuple[str, ...] = ()
    supported_inference_modes: tuple[str, ...] = ()
    supported_modalities: tuple[str, ...] = ()
    cactus_op: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
