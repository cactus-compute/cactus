from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..Fusions import models as FModels
from ..IR import models as IRModels


PREFILL_WITH_CACHE_TASK = "prefill_with_cache"
DECODE_WITH_CACHE_TASK = "decode_with_cache"
GEMMA4_PREFILL_CHUNK_TOKENS = 128
GENERIC_CAUSAL_PREFILL_CHUNK_TOKENS = 4


@dataclass(slots=True, frozen=True)
class PlaceholderSpec:
    name: str
    logical_name: str
    source_node: str | None = None
    tensor_node: str | None = None
    target: str | None = None
    value_kind: str = FModels.ValueKind.USER_INPUT
    force: bool = False


@dataclass(slots=True, frozen=True)
class OutputSpec:
    node: str
    logical_name: str
    row_limit: int | None = None


@dataclass(slots=True, frozen=True)
class ComponentSplitSpec:
    name: str
    graph: IRModels.Graph
    outputs: tuple[OutputSpec, ...]
    placeholders: tuple[PlaceholderSpec, ...] = ()
    ref_aliases: Mapping[str, str] = field(default_factory=dict)
    input_aliases: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
    chunk_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class Gemma4Boundaries:
    vision_features: str
    audio_features: str
    merged_inputs_embeds: str
    prefill_text_inputs_embeds: str
    prefill_token_ids: str
    prefill_per_layer_inputs: str
    decode_text_inputs_embeds: str
    decode_token_ids: str
    decode_per_layer_inputs: str
    decode_position_ids: str
    prefill_logits: str
    decode_logits: str


@dataclass(slots=True, frozen=True)
class WhisperBoundaries:
    encoder_hidden_states: str
    cross_key_values: tuple[tuple[str, str], ...]
    decode_logits: str
