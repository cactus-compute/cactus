from dataclasses import dataclass, field
from typing import Any


class ValueKind:
    """
    Names the semantic role of a real IR value.

    IR matchers use this to tell the difference between parameters,
    buffers, runtime inputs, activations, constants, outputs, and cache
    tensors without depending on fragile node-name strings.
    """

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
    """
    Names the broad family of cache/state a fusion consumes or produces.

    KV is for transformer key/value cache, CONV is for rolling convolution
    state, and RECURRENT is for state-space/RNN-style hidden state. Runtime
    planning can use this to route cache tensors differently from weights.
    """

    KV = "kv"
    CONV = "conv"
    RECURRENT = "recurrent"


class CacheTensorRole:
    """
    Names the role of one tensor inside a cache family.

    This lets a cache-aware fusion say whether a tensor is a key cache,
    value cache, generic recurrent/conv state, or cache position tensor.
    """

    KEY = "key"
    VALUE = "value"
    STATE = "state"
    POSITION = "position"


@dataclass(slots=True)
class TensorMeta:
    """
    Stores observed tensor metadata from the exported graph.

    The IR graph can attach this to real nodes so fusion matchers can compare
    shapes, dtypes, devices, grad flags, and strides against TensorConstraint
    without rereading the original torch node metadata each time.
    """

    shape: list[Any] = field(default_factory=list)
    dtype: str | None = None
    device: str | None = None
    requires_grad: bool | None = None
    stride: list[Any] | None = None


@dataclass(slots=True)
class TensorConstraint:
    """
    Describes tensor properties a matched real value must satisfy.

    Fusion patterns use this for rank/dtype/shape checks, fixed dimension
    checks, and cross-node dimension relationships such as "query head_dim
    equals key head_dim". The actual comparison logic belongs in IR.
    """

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
    """
    Describes one required attribute condition for a pattern node.

    Examples are pow exponent equals 2, softmax dim equals -1, or conv stride
    equals a target kernel variant. source_node/source_attr allow an IR matcher
    to compare this attr to another matched node's attr when needed.
    """

    name: str
    value: Any = None
    source_node: str | None = None #Which node to pull attr from
    source_attr: str | None = None #Which attr from external node to pull
    comparator: str = "eq"
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttrCapture:
    """
    Describes which raw-node attribute becomes a fused-node attribute.

    A match can use this to copy values like epsilon, axis, stride, padding,
    scale, or top-k from the matched graph into the simplified Cactus-facing
    fused node. default is used when the raw graph does not expose the attr.
    """

    name: str
    source_node: str | None = None
    source_attr: str | None = None
    default: Any = None
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeRef:
    """
    Points to a pattern node and optionally one of its input/output positions.

    FusionInput, shared-input declarations, and cache declarations use this
    small reference instead of embedding matcher behavior in the Fusion schema.
    parent_index refers to the matched node's ordered parents in the IR DAG.
    """

    node: str
    parent_index: int | None = None
    output_index: int | None = None


@dataclass(slots=True)
class FusionNode:
    """
    Describes one synthetic node inside a fusion pattern graph.

    It stores the allowed raw op targets, node-level attr/tensor constraints,
    and structural hints for repeated pattern regions.
    IR matchers bind these descriptions to real exported nodes during fusion.
    """

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
    """
    Describes one directed producer-to-consumer edge in a fusion pattern.

    source/dest identify synthetic nodes, dest_input_index preserves meaningful
    input ordering from FX args, and repeated flags let IR match repeated graph
    regions such as MoE expert branches.
    """

    source: str
    dest: str
    dest_input_index: int | None = None
    source_output_index: int | None = None
    repeated: bool = False
    repeated_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusionInput:
    """
    Describes an external input that must remain visible after fusion.

    The source points to the synthetic node input position where the external
    real node appears. variadic/min_count/max_count support ops like cat or
    AltUp that accept a variable number of parents.
    """

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
    """
    Describes an output produced by the fused op.

    Most fusions expose the root node output, but multi-output/cache-aware
    fusions can declare multiple outputs with roles so the simplified graph
    and runtime plan know how to reconnect downstream consumers.
    """

    role: str
    node: str
    output_index: int | None = None
    tensor_constraints: tuple[TensorConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CacheInput:
    """
    Describes an old cache/state tensor consumed by a fused op.

    This keeps cache semantics separate from ordinary data inputs, allowing IR
    and runtime planning to bind KV/conv/recurrent state tensors correctly for
    prefill-with-cache and decode-with-cache exports.
    """

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
    """
    Describes a new or updated cache/state tensor produced by a fused op.

    IR can use this to reconnect tuple outputs or explicit cache writes after
    fusion, while the generator can use cache_kind/tensor_role to emit the
    right Cactus runtime cache plumbing.
    """

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
    """
    Describes read/write semantics for a cache-aware fusion.

    CacheInput/CacheOutput say which tensors cross the fusion boundary; this
    object says how the operation conceptually mutates or advances that state,
    such as appending KV rows or rolling a convolution window.
    """

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
    """
    Describes a nested pattern that can repeat inside a larger fusion.

    MoE is the main use case: one router/top-k structure fans into repeated
    expert branches with the same shape. IR owns the logic for discovering and
    binding each repeated branch.
    """

    name: str
    graph: "FusionGraph"
    min_count: int = 1
    max_count: int | None = None
    anchor_node: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusionGraph:
    """
    Describes the complete synthetic DAG for one candidate fusion.

    It stores pattern nodes, required edges, exposed inputs/outputs, attrs to
    capture, repeated subgraphs, cache boundaries, and extra textual constraints
    that IR matchers must enforce before replacing the matched real subgraph
    with a fused op.
    """

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
    constraints: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_root_external_children: bool = True


@dataclass(slots=True)
class FusionDefinition:
    """
    Wraps a FusionGraph with Cactus-facing metadata.

    target/cactus_op tell the generator what op this pattern lowers to, while
    fusion_fields, inference modes, modalities, and metadata let model profiles
    select stronger model-specific patterns before falling back to generic ones.
    """

    name: str
    target: str
    graph: FusionGraph
    fusion_fields: tuple[str, ...] = ()
    supported_inference_modes: tuple[str, ...] = ()
    supported_modalities: tuple[str, ...] = ()
    cactus_op: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusionCatalog:
    """
    Stores the registry of all fusion definitions.

    This is the compact object IR can iterate over or index when running fusion
    passes. The heavier lookup maps live in fusions.py, not in this data model.
    """

    fusions: tuple[FusionDefinition, ...] = ()
