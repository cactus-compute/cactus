from dataclasses import dataclass, field

GENERIC_TASK_CAUSAL_LM = "causal-lm"
GENERIC_TASK_SPEECH_SEQ2SEQ = "speech-seq2seq"
GENERIC_CACHE_DYNAMIC_KV = "dynamic-kv"
GENERIC_CACHE_ENCODER_DECODER_KV = "encoder-decoder-kv"
GENERIC_CACHE_NONE = "none"
@dataclass(slots=True, frozen=True)
class GenericTranspileContract:
    """User-declared execution contract for an unregistered model."""
    task: str = GENERIC_TASK_CAUSAL_LM
    modalities: tuple[str, ...] = ("text",)
    cache_style: str = GENERIC_CACHE_NONE
    fusion_groups: tuple[str, ...] = ()

@dataclass(slots=True)
class Components:
    name:str
    patterns:tuple[str, ...]

@dataclass(slots=True)
class Combinations:
    input:tuple[str, ...]
    output:str

@dataclass(slots=True)
class InferencePattern:
    name:str
    route:tuple[Combinations, ...]

@dataclass(slots=True, frozen=True)
class ComponentSource:
    mode:str #Export mode this component is captured under
    load_strategy:str #Key into Converter LOAD_STRATEGIES/SYNTHETIC_INPUT_BUILDERS for this component
    source:str = "" #Repository subfolder, or a full repository id when the component ships separately
    preserved_ops:tuple[str, ...] = () #Aten ops this component keeps whole through run_decompositions

@dataclass(slots=True, frozen=True)
class PromptContract:
    style: str = "auto"
    template_source: str = "tokenizer_config"
    text_style: str = ""
    media_style: str = ""
    turn_start_token: str = ""
    turn_end_token: str = ""
    suppress_generation_token_ids: tuple[int, ...] = ()
    repetition_penalty_scope: str = "generated"

@dataclass(slots=True, frozen=True)
class MediaContract:
    image_preprocess_strategy: str = ""
    audio_preprocess_strategy: str = ""
    injection_strategy: str = ""
    media_order: tuple[str, ...] = ("image", "audio")
    focus_policy: str = ""
    image_focus_keywords: tuple[str, ...] = ()
    audio_focus_keywords: tuple[str, ...] = ()
    chunk_prefill_modalities: tuple[str, ...] = ()
    prefill_fallback: str = "error"
    min_new_tokens: int = 0
    chunk_output_sources: tuple[tuple[str, str], ...] = ()
    mask_polarity: str = ""
    span_strategy: str = ""
    placeholder_token_id: int = 0
    image_token_id: int = 0
    audio_token_id: int = 0
    image_token: str = ""
    audio_token: str = ""
    image_begin_token: str = ""
    image_end_token: str = ""
    audio_begin_token: str = ""
    audio_end_token: str = ""
    image_prompt_position: str = "before_text"
    audio_prompt_position: str = "before_text"
    audio_rows_per_frames: str = ""
    image_feature_names: tuple[str, ...] = ()
    audio_feature_names: tuple[str, ...] = ()

@dataclass(slots=True, frozen=True)
class CacheContract:
    prefill_decode_compatibility: str = ""
    state_transfer: str = ""
    decode_uses_media_components: bool = False
    fp16_kv_cache_components: tuple[str, ...] = ()
    max_cache_sequence_length: int = 0
    full_retention_kv_layers: tuple[int, ...] = ()

@dataclass(slots=True, frozen=True)
class StateContract:
    name: str
    kind: str
    producer: str = ""
    consumers: tuple[str, ...] = ()
    lifetime: str = "request"
    transfer: str = "move"
    persist_after_component_unload: bool = True
    required: bool = True
    release_after_consumers: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

@dataclass(slots=True, frozen=True)
class AliasContract:
    source_component: str
    source_output: str
    target_component: str
    target_input: str
    policy: str = "alias_if_compatible"
    lifetime: str = "until_target_execute"
    fallback: str = "copy"
    required: bool = False
    metadata: tuple[tuple[str, str], ...] = ()

@dataclass(slots=True, frozen=True)
class RuntimeContract:
    plan_name: str = "generic"
    execution_strategy: str = "component_graph"
    state_owner: str = "session"
    cache_persistence: str = "component_move"
    output_alias_policy: str = "alias_if_compatible"
    cache_transfer_policy: str = "move"
    states: tuple[StateContract, ...] = ()
    aliases: tuple[AliasContract, ...] = ()

@dataclass(slots=True)
class ModelProfile:
    model_profiles:str #Model family for which this profile is valid (Theoretically can be shared across model families that have similar architectures)
    components: dict[str, Components] #Individual components (vision tower, audio tower, token merge, etc.) this model will be split into
    inference_type: dict[str, InferencePattern] #Specific operations mapped to their tuple of components ordered by execution 
    cache_type: tuple[str, ...] #Which layers/operations to expect KV cache for
    cache_policy: tuple[str, ...] #How to handle the cache during different phases of inference
    files: tuple[str, ...] #Config and other necessary files 
    fusion_fields: tuple[str, ...] #Specific fusion groups to consider during fusion process
    supported_modalties: tuple[str, ...] #Input modalities supported by model
    input_strategy: str #Specifies what functions/procedures to use when generating sample input for torch.export
    export_patches: tuple[str, ...] #Specifies what patches/masks to apply to sample inputs
    load_strategy: str #Specifies which Hugging Face AutoModel class/loading path to prefer
    export_modes: tuple[str, ...] = () #Per-component export modes when the family is not prefill/decode shaped; empty means the default modes

    component_sources: tuple[ComponentSource, ...] = () #Components of a pipeline model that load and export separately
    disabled_fusion_fields: tuple[str, ...] = () #Fusion groups to skip for this profile when a family is known unsafe
    disabled_fusions: tuple[str, ...] = () #Individual fusion names or cactus ops to skip for this profile
    prompt_contract: PromptContract = field(default_factory=PromptContract)
    media_contract: MediaContract = field(default_factory=MediaContract)
    cache_contract: CacheContract = field(default_factory=CacheContract)
    runtime_contract: RuntimeContract = field(default_factory=RuntimeContract)
    
