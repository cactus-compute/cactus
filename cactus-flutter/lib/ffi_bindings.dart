// ignore_for_file: non_constant_identifier_names, camel_case_types
import 'dart:ffi';
import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';

final class CactusContextOpaque extends Opaque {}
typedef CactusContextHandle = Pointer<CactusContextOpaque>;

final class CactusInitParamsC extends Struct {
  external Pointer<Utf8> model_path;
  external Pointer<Utf8> chat_template;

  @Int32()
  external int n_ctx;
  @Int32()
  external int n_batch;
  @Int32()
  external int n_ubatch;
  @Int32()
  external int n_gpu_layers;
  @Int32()
  external int n_threads;
  @Bool()
  external bool use_mmap;
  @Bool()
  external bool use_mlock;
  @Bool()
  external bool embedding;
  @Int32()
  external int pooling_type;
  @Int32()
  external int embd_normalize;
  @Bool()
  external bool flash_attn;
  external Pointer<Utf8> cache_type_k;
  external Pointer<Utf8> cache_type_v;

  external Pointer<NativeFunction<Void Function(Float)>> progress_callback;
}

// STRUCT: cactus_completion_params_c_t (matches C++ exactly)
final class CactusCompletionParamsC extends Struct {
  external Pointer<Utf8> prompt;
  @Int32()
  external int n_predict;
  @Int32()
  external int n_threads;
  @Int32()
  external int seed;
  @Double()
  external double temperature;
  @Int32()
  external int top_k;
  @Double()
  external double top_p;
  @Double()
  external double min_p;
  @Double()
  external double typical_p;
  @Int32()
  external int penalty_last_n;
  @Double()
  external double penalty_repeat;
  @Double()
  external double penalty_freq;
  @Double()
  external double penalty_present;
  @Int32()
  external int mirostat;
  @Double()
  external double mirostat_tau;
  @Double()
  external double mirostat_eta;
  @Bool()
  external bool ignore_eos;
  @Int32()
  external int n_probs;
  external Pointer<Pointer<Utf8>> stop_sequences;
  @Int32()
  external int stop_sequence_count;
  external Pointer<Utf8> grammar;

  external Pointer<NativeFunction<Bool Function(Pointer<Utf8>)>> token_callback;
}

// STRUCT: cactus_token_array_c_t (matches C++ exactly)
final class CactusTokenArrayC extends Struct {
  external Pointer<Int32> tokens;
  @Int32()
  external int count;
}

// STRUCT: cactus_float_array_c_t (matches C++ exactly)
final class CactusFloatArrayC extends Struct {
  external Pointer<Float> values;
  @Int32()
  external int count;
}

// STRUCT: cactus_completion_result_c_t (matches C++ exactly)
final class CactusCompletionResultC extends Struct {
  external Pointer<Utf8> text;
  @Int32()
  external int tokens_predicted;
  @Int32()
  external int tokens_evaluated;
  @Bool()
  external bool truncated;
  @Bool()
  external bool stopped_eos;
  @Bool()
  external bool stopped_word;
  @Bool()
  external bool stopped_limit;
  external Pointer<Utf8> stopping_word;
}

// STRUCT: cactus_tokenize_result_c_t (matches C++ exactly)
final class CactusTokenizeResultC extends Struct {
  external CactusTokenArrayC tokens;
  @Bool()
  external bool has_media;
  external Pointer<Pointer<Utf8>> bitmap_hashes;
  @Int32()
  external int bitmap_hash_count;
  external Pointer<Size> chunk_positions;
  @Int32()
  external int chunk_position_count;
  external Pointer<Size> chunk_positions_media;
  @Int32()
  external int chunk_position_media_count;
}

// STRUCT: cactus_lora_adapter_c_t (matches C++ exactly)
final class CactusLoraAdapterC extends Struct {
  external Pointer<Utf8> path;
  @Float()
  external double scale;
}

// STRUCT: cactus_lora_adapters_c_t (matches C++ exactly)
final class CactusLoraAdaptersC extends Struct {
  external Pointer<CactusLoraAdapterC> adapters;
  @Int32()
  external int count;
}

// STRUCT: cactus_bench_result_c_t (matches C++ exactly)
final class CactusBenchResultC extends Struct {
  external Pointer<Utf8> model_name;
  @Int64()
  external int model_size;
  @Int64()
  external int model_params;
  @Double()
  external double pp_avg;
  @Double()
  external double pp_std;
  @Double()
  external double tg_avg;
  @Double()
  external double tg_std;
}

// STRUCT: cactus_chat_result_c_t (matches C++ exactly)
final class CactusChatResultC extends Struct {
  external Pointer<Utf8> prompt;
  external Pointer<Utf8> json_schema;
  external Pointer<Utf8> tools;
  external Pointer<Utf8> tool_choice;
  @Bool()
  external bool parallel_tool_calls;
}

// =============================================================================
// FUNCTION TYPE DEFINITIONS (matches C++ exactly)
// =============================================================================

// Basic context management
typedef InitContextNative = CactusContextHandle Function(Pointer<CactusInitParamsC> params);
typedef InitContextDart = CactusContextHandle Function(Pointer<CactusInitParamsC> params);

typedef FreeContextNative = Void Function(CactusContextHandle handle);
typedef FreeContextDart = void Function(CactusContextHandle handle);

// Completion functions
typedef CompletionNative = Int32 Function(
    CactusContextHandle handle,
    Pointer<CactusCompletionParamsC> params,
    Pointer<CactusCompletionResultC> result);
typedef CompletionDart = int Function(
    CactusContextHandle handle,
    Pointer<CactusCompletionParamsC> params,
    Pointer<CactusCompletionResultC> result);

typedef MultimodalCompletionNative = Int32 Function(
    CactusContextHandle handle,
    Pointer<CactusCompletionParamsC> params,
    Pointer<Pointer<Utf8>> media_paths,
    Int32 media_count,
    Pointer<CactusCompletionResultC> result);
typedef MultimodalCompletionDart = int Function(
    CactusContextHandle handle,
    Pointer<CactusCompletionParamsC> params,
    Pointer<Pointer<Utf8>> media_paths,
    int media_count,
    Pointer<CactusCompletionResultC> result);

typedef StopCompletionNative = Void Function(CactusContextHandle handle);
typedef StopCompletionDart = void Function(CactusContextHandle handle);

// Tokenization functions
typedef TokenizeNative = CactusTokenArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);
typedef TokenizeDart = CactusTokenArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);

typedef DetokenizeNative = Pointer<Utf8> Function(CactusContextHandle handle, Pointer<Int32> tokens, Int32 count);
typedef DetokenizeDart = Pointer<Utf8> Function(CactusContextHandle handle, Pointer<Int32> tokens, int count);

typedef TokenizeWithMediaNative = CactusTokenizeResultC Function(
    CactusContextHandle handle, Pointer<Utf8> text, Pointer<Pointer<Utf8>> media_paths, Int32 media_count);
typedef TokenizeWithMediaDart = CactusTokenizeResultC Function(
    CactusContextHandle handle, Pointer<Utf8> text, Pointer<Pointer<Utf8>> media_paths, int media_count);

// Embedding functions
typedef EmbeddingNative = CactusFloatArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);
typedef EmbeddingDart = CactusFloatArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);

// Guide tokens
typedef SetGuideTokensNative = Void Function(CactusContextHandle handle, Pointer<Int32> tokens, Int32 count);
typedef SetGuideTokensDart = void Function(CactusContextHandle handle, Pointer<Int32> tokens, int count);

// Multimodal functions
typedef InitMultimodalNative = Int32 Function(CactusContextHandle handle, Pointer<Utf8> mmproj_path, Bool use_gpu);
typedef InitMultimodalDart = int Function(CactusContextHandle handle, Pointer<Utf8> mmproj_path, bool use_gpu);

typedef IsMultimodalEnabledNative = Bool Function(CactusContextHandle handle);
typedef IsMultimodalEnabledDart = bool Function(CactusContextHandle handle);

typedef SupportsVisionNative = Bool Function(CactusContextHandle handle);
typedef SupportsVisionDart = bool Function(CactusContextHandle handle);

typedef SupportsAudioNative = Bool Function(CactusContextHandle handle);
typedef SupportsAudioDart = bool Function(CactusContextHandle handle);

typedef ReleaseMultimodalNative = Void Function(CactusContextHandle handle);
typedef ReleaseMultimodalDart = void Function(CactusContextHandle handle);

// TTS/Vocoder functions
typedef InitVocoderNative = Int32 Function(CactusContextHandle handle, Pointer<Utf8> vocoder_model_path);
typedef InitVocoderDart = int Function(CactusContextHandle handle, Pointer<Utf8> vocoder_model_path);

typedef IsVocoderEnabledNative = Bool Function(CactusContextHandle handle);
typedef IsVocoderEnabledDart = bool Function(CactusContextHandle handle);

typedef GetTTSTypeNative = Int32 Function(CactusContextHandle handle);
typedef GetTTSTypeDart = int Function(CactusContextHandle handle);

typedef GetFormattedAudioCompletionNative = Pointer<Utf8> Function(
    CactusContextHandle handle, Pointer<Utf8> speaker_json_str, Pointer<Utf8> text_to_speak);
typedef GetFormattedAudioCompletionDart = Pointer<Utf8> Function(
    CactusContextHandle handle, Pointer<Utf8> speaker_json_str, Pointer<Utf8> text_to_speak);

typedef GetAudioCompletionGuideTokensNative = CactusTokenArrayC Function(
    CactusContextHandle handle, Pointer<Utf8> text_to_speak);
typedef GetAudioCompletionGuideTokensDart = CactusTokenArrayC Function(
    CactusContextHandle handle, Pointer<Utf8> text_to_speak);

typedef DecodeAudioTokensNative = CactusFloatArrayC Function(
    CactusContextHandle handle, Pointer<Int32> tokens, Int32 count);
typedef DecodeAudioTokensDart = CactusFloatArrayC Function(
    CactusContextHandle handle, Pointer<Int32> tokens, int count);

typedef ReleaseVocoderNative = Void Function(CactusContextHandle handle);
typedef ReleaseVocoderDart = void Function(CactusContextHandle handle);

// LoRA functions
typedef BenchNative = CactusBenchResultC Function(CactusContextHandle handle, Int32 pp, Int32 tg, Int32 pl, Int32 nr);
typedef BenchDart = CactusBenchResultC Function(CactusContextHandle handle, int pp, int tg, int pl, int nr);

typedef ApplyLoraAdaptersNative = Int32 Function(CactusContextHandle handle, Pointer<CactusLoraAdaptersC> adapters);
typedef ApplyLoraAdaptersDart = int Function(CactusContextHandle handle, Pointer<CactusLoraAdaptersC> adapters);

typedef RemoveLoraAdaptersNative = Void Function(CactusContextHandle handle);
typedef RemoveLoraAdaptersDart = void Function(CactusContextHandle handle);

typedef GetLoadedLoraAdaptersNative = CactusLoraAdaptersC Function(CactusContextHandle handle);
typedef GetLoadedLoraAdaptersDart = CactusLoraAdaptersC Function(CactusContextHandle handle);

// Chat template functions
typedef ValidateChatTemplateNative = Bool Function(CactusContextHandle handle, Bool use_jinja, Pointer<Utf8> name);
typedef ValidateChatTemplateDart = bool Function(CactusContextHandle handle, bool use_jinja, Pointer<Utf8> name);

typedef GetFormattedChatNative = Pointer<Utf8> Function(
    CactusContextHandle handle, Pointer<Utf8> messages, Pointer<Utf8> chat_template);
typedef GetFormattedChatDart = Pointer<Utf8> Function(
    CactusContextHandle handle, Pointer<Utf8> messages, Pointer<Utf8> chat_template);

typedef GetFormattedChatWithJinjaNative = CactusChatResultC Function(
    CactusContextHandle handle, Pointer<Utf8> messages, Pointer<Utf8> chat_template,
    Pointer<Utf8> json_schema, Pointer<Utf8> tools, Bool parallel_tool_calls, Pointer<Utf8> tool_choice);
typedef GetFormattedChatWithJinjaDart = CactusChatResultC Function(
    CactusContextHandle handle, Pointer<Utf8> messages, Pointer<Utf8> chat_template,
    Pointer<Utf8> json_schema, Pointer<Utf8> tools, bool parallel_tool_calls, Pointer<Utf8> tool_choice);

// Context management functions
typedef RewindNative = Void Function(CactusContextHandle handle);
typedef RewindDart = void Function(CactusContextHandle handle);

typedef InitSamplingNative = Bool Function(CactusContextHandle handle);
typedef InitSamplingDart = bool Function(CactusContextHandle handle);

typedef BeginCompletionNative = Void Function(CactusContextHandle handle);
typedef BeginCompletionDart = void Function(CactusContextHandle handle);

typedef EndCompletionNative = Void Function(CactusContextHandle handle);
typedef EndCompletionDart = void Function(CactusContextHandle handle);

typedef LoadPromptNative = Void Function(CactusContextHandle handle);
typedef LoadPromptDart = void Function(CactusContextHandle handle);

typedef LoadPromptWithMediaNative = Void Function(
    CactusContextHandle handle, Pointer<Pointer<Utf8>> media_paths, Int32 media_count);
typedef LoadPromptWithMediaDart = void Function(
    CactusContextHandle handle, Pointer<Pointer<Utf8>> media_paths, int media_count);

// Token processing functions
typedef DoCompletionStepNative = Int32 Function(CactusContextHandle handle, Pointer<Pointer<Utf8>> token_text);
typedef DoCompletionStepDart = int Function(CactusContextHandle handle, Pointer<Pointer<Utf8>> token_text);

typedef FindStoppingStringsNative = Size Function(
    CactusContextHandle handle, Pointer<Utf8> text, Size last_token_size, Int32 stop_type);
typedef FindStoppingStringsDart = int Function(
    CactusContextHandle handle, Pointer<Utf8> text, int last_token_size, int stop_type);

// Model information functions
typedef GetNCtxNative = Int32 Function(CactusContextHandle handle);
typedef GetNCtxDart = int Function(CactusContextHandle handle);

typedef GetNEmbdNative = Int32 Function(CactusContextHandle handle);
typedef GetNEmbdDart = int Function(CactusContextHandle handle);

typedef GetModelDescNative = Pointer<Utf8> Function(CactusContextHandle handle);
typedef GetModelDescDart = Pointer<Utf8> Function(CactusContextHandle handle);

typedef GetModelSizeNative = Int64 Function(CactusContextHandle handle);
typedef GetModelSizeDart = int Function(CactusContextHandle handle);

typedef GetModelParamsNative = Int64 Function(CactusContextHandle handle);
typedef GetModelParamsDart = int Function(CactusContextHandle handle);

// Memory management functions
typedef FreeStringNative = Void Function(Pointer<Utf8> str);
typedef FreeStringDart = void Function(Pointer<Utf8> str);

typedef FreeTokenArrayNative = Void Function(CactusTokenArrayC arr);
typedef FreeTokenArrayDart = void Function(CactusTokenArrayC arr);

typedef FreeFloatArrayNative = Void Function(CactusFloatArrayC arr);
typedef FreeFloatArrayDart = void Function(CactusFloatArrayC arr);

typedef FreeCompletionResultMembersNative = Void Function(Pointer<CactusCompletionResultC> result);
typedef FreeCompletionResultMembersDart = void Function(Pointer<CactusCompletionResultC> result);

typedef FreeTokenizeResultNative = Void Function(Pointer<CactusTokenizeResultC> result);
typedef FreeTokenizeResultDart = void Function(Pointer<CactusTokenizeResultC> result);

typedef FreeBenchResultMembersNative = Void Function(Pointer<CactusBenchResultC> result);
typedef FreeBenchResultMembersDart = void Function(Pointer<CactusBenchResultC> result);

typedef FreeLoraAdaptersNative = Void Function(Pointer<CactusLoraAdaptersC> adapters);
typedef FreeLoraAdaptersDart = void Function(Pointer<CactusLoraAdaptersC> adapters);

typedef FreeChatResultMembersNative = Void Function(Pointer<CactusChatResultC> result);
typedef FreeChatResultMembersDart = void Function(Pointer<CactusChatResultC> result);

// =============================================================================
// DYNAMIC LIBRARY LOADING
// =============================================================================

String _getLibraryPath() {
  const String libName = 'cactus';
  if (Platform.isIOS || Platform.isMacOS) {
    return '$libName.framework/$libName';
  }
  if (Platform.isAndroid) {
    return 'lib$libName.so';
  }
  throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
}

final DynamicLibrary cactusLib = DynamicLibrary.open(_getLibraryPath());

// =============================================================================
// FUNCTION BINDINGS (exactly matching C++ FFI function names)
// =============================================================================

// Basic context management
final initContext = cactusLib
    .lookup<NativeFunction<InitContextNative>>('cactus_init_context_c')
    .asFunction<InitContextDart>();

final freeContext = cactusLib
    .lookup<NativeFunction<FreeContextNative>>('cactus_free_context_c')
    .asFunction<FreeContextDart>();

// Completion functions
final completion = cactusLib
    .lookup<NativeFunction<CompletionNative>>('cactus_completion_c')
    .asFunction<CompletionDart>();

final multimodalCompletion = cactusLib
    .lookup<NativeFunction<MultimodalCompletionNative>>('cactus_multimodal_completion_c')
    .asFunction<MultimodalCompletionDart>();

final stopCompletion = cactusLib
    .lookup<NativeFunction<StopCompletionNative>>('cactus_stop_completion_c')
    .asFunction<StopCompletionDart>();

// Tokenization functions
final tokenize = cactusLib
    .lookup<NativeFunction<TokenizeNative>>('cactus_tokenize_c')
    .asFunction<TokenizeDart>();

final detokenize = cactusLib
    .lookup<NativeFunction<DetokenizeNative>>('cactus_detokenize_c')
    .asFunction<DetokenizeDart>();

final tokenizeWithMedia = cactusLib
    .lookup<NativeFunction<TokenizeWithMediaNative>>('cactus_tokenize_with_media_c')
    .asFunction<TokenizeWithMediaDart>();

// Embedding functions
final embedding = cactusLib
    .lookup<NativeFunction<EmbeddingNative>>('cactus_embedding_c')
    .asFunction<EmbeddingDart>();

// Guide tokens
final setGuideTokens = cactusLib
    .lookup<NativeFunction<SetGuideTokensNative>>('cactus_set_guide_tokens_c')
    .asFunction<SetGuideTokensDart>();

// Multimodal functions
final initMultimodal = cactusLib
    .lookup<NativeFunction<InitMultimodalNative>>('cactus_init_multimodal_c')
    .asFunction<InitMultimodalDart>();

final isMultimodalEnabled = cactusLib
    .lookup<NativeFunction<IsMultimodalEnabledNative>>('cactus_is_multimodal_enabled_c')
    .asFunction<IsMultimodalEnabledDart>();

final supportsVision = cactusLib
    .lookup<NativeFunction<SupportsVisionNative>>('cactus_supports_vision_c')
    .asFunction<SupportsVisionDart>();

final supportsAudio = cactusLib
    .lookup<NativeFunction<SupportsAudioNative>>('cactus_supports_audio_c')
    .asFunction<SupportsAudioDart>();

final releaseMultimodal = cactusLib
    .lookup<NativeFunction<ReleaseMultimodalNative>>('cactus_release_multimodal_c')
    .asFunction<ReleaseMultimodalDart>();

// TTS/Vocoder functions
final initVocoder = cactusLib
    .lookup<NativeFunction<InitVocoderNative>>('cactus_init_vocoder_c')
    .asFunction<InitVocoderDart>();

final isVocoderEnabled = cactusLib
    .lookup<NativeFunction<IsVocoderEnabledNative>>('cactus_is_vocoder_enabled_c')
    .asFunction<IsVocoderEnabledDart>();

final getTTSType = cactusLib
    .lookup<NativeFunction<GetTTSTypeNative>>('cactus_get_tts_type_c')
    .asFunction<GetTTSTypeDart>();

final getFormattedAudioCompletion = cactusLib
    .lookup<NativeFunction<GetFormattedAudioCompletionNative>>('cactus_get_formatted_audio_completion_c')
    .asFunction<GetFormattedAudioCompletionDart>();

final getAudioCompletionGuideTokens = cactusLib
    .lookup<NativeFunction<GetAudioCompletionGuideTokensNative>>('cactus_get_audio_completion_guide_tokens_c')
    .asFunction<GetAudioCompletionGuideTokensDart>();

final decodeAudioTokens = cactusLib
    .lookup<NativeFunction<DecodeAudioTokensNative>>('cactus_decode_audio_tokens_c')
    .asFunction<DecodeAudioTokensDart>();

final releaseVocoder = cactusLib
    .lookup<NativeFunction<ReleaseVocoderNative>>('cactus_release_vocoder_c')
    .asFunction<ReleaseVocoderDart>();

// Benchmarking and LoRA functions
final bench = cactusLib
    .lookup<NativeFunction<BenchNative>>('cactus_bench_c')
    .asFunction<BenchDart>();

final applyLoraAdapters = cactusLib
    .lookup<NativeFunction<ApplyLoraAdaptersNative>>('cactus_apply_lora_adapters_c')
    .asFunction<ApplyLoraAdaptersDart>();

final removeLoraAdapters = cactusLib
    .lookup<NativeFunction<RemoveLoraAdaptersNative>>('cactus_remove_lora_adapters_c')
    .asFunction<RemoveLoraAdaptersDart>();

final getLoadedLoraAdapters = cactusLib
    .lookup<NativeFunction<GetLoadedLoraAdaptersNative>>('cactus_get_loaded_lora_adapters_c')
    .asFunction<GetLoadedLoraAdaptersDart>();

// Chat template functions
final validateChatTemplate = cactusLib
    .lookup<NativeFunction<ValidateChatTemplateNative>>('cactus_validate_chat_template_c')
    .asFunction<ValidateChatTemplateDart>();

final getFormattedChat = cactusLib
    .lookup<NativeFunction<GetFormattedChatNative>>('cactus_get_formatted_chat_c')
    .asFunction<GetFormattedChatDart>();

final getFormattedChatWithJinja = cactusLib
    .lookup<NativeFunction<GetFormattedChatWithJinjaNative>>('cactus_get_formatted_chat_with_jinja_c')
    .asFunction<GetFormattedChatWithJinjaDart>();

// Context management functions
final rewind = cactusLib
    .lookup<NativeFunction<RewindNative>>('cactus_rewind_c')
    .asFunction<RewindDart>();

final initSampling = cactusLib
    .lookup<NativeFunction<InitSamplingNative>>('cactus_init_sampling_c')
    .asFunction<InitSamplingDart>();

final beginCompletion = cactusLib
    .lookup<NativeFunction<BeginCompletionNative>>('cactus_begin_completion_c')
    .asFunction<BeginCompletionDart>();

final endCompletion = cactusLib
    .lookup<NativeFunction<EndCompletionNative>>('cactus_end_completion_c')
    .asFunction<EndCompletionDart>();

final loadPrompt = cactusLib
    .lookup<NativeFunction<LoadPromptNative>>('cactus_load_prompt_c')
    .asFunction<LoadPromptDart>();

final loadPromptWithMedia = cactusLib
    .lookup<NativeFunction<LoadPromptWithMediaNative>>('cactus_load_prompt_with_media_c')
    .asFunction<LoadPromptWithMediaDart>();

// Token processing functions
final doCompletionStep = cactusLib
    .lookup<NativeFunction<DoCompletionStepNative>>('cactus_do_completion_step_c')
    .asFunction<DoCompletionStepDart>();

final findStoppingStrings = cactusLib
    .lookup<NativeFunction<FindStoppingStringsNative>>('cactus_find_stopping_strings_c')
    .asFunction<FindStoppingStringsDart>();

// Model information functions
final getNCtx = cactusLib
    .lookup<NativeFunction<GetNCtxNative>>('cactus_get_n_ctx_c')
    .asFunction<GetNCtxDart>();

final getNEmbd = cactusLib
    .lookup<NativeFunction<GetNEmbdNative>>('cactus_get_n_embd_c')
    .asFunction<GetNEmbdDart>();

final getModelDesc = cactusLib
    .lookup<NativeFunction<GetModelDescNative>>('cactus_get_model_desc_c')
    .asFunction<GetModelDescDart>();

final getModelSize = cactusLib
    .lookup<NativeFunction<GetModelSizeNative>>('cactus_get_model_size_c')
    .asFunction<GetModelSizeDart>();

final getModelParams = cactusLib
    .lookup<NativeFunction<GetModelParamsNative>>('cactus_get_model_params_c')
    .asFunction<GetModelParamsDart>();

// Memory management functions
final freeString = cactusLib
    .lookup<NativeFunction<FreeStringNative>>('cactus_free_string_c')
    .asFunction<FreeStringDart>();

final freeTokenArray = cactusLib
    .lookup<NativeFunction<FreeTokenArrayNative>>('cactus_free_token_array_c')
    .asFunction<FreeTokenArrayDart>();

final freeFloatArray = cactusLib
    .lookup<NativeFunction<FreeFloatArrayNative>>('cactus_free_float_array_c')
    .asFunction<FreeFloatArrayDart>();

final freeCompletionResultMembers = cactusLib
    .lookup<NativeFunction<FreeCompletionResultMembersNative>>('cactus_free_completion_result_members_c')
    .asFunction<FreeCompletionResultMembersDart>();

final freeTokenizeResult = cactusLib
    .lookup<NativeFunction<FreeTokenizeResultNative>>('cactus_free_tokenize_result_c')
    .asFunction<FreeTokenizeResultDart>();

final freeBenchResultMembers = cactusLib
    .lookup<NativeFunction<FreeBenchResultMembersNative>>('cactus_free_bench_result_members_c')
    .asFunction<FreeBenchResultMembersDart>();

final freeLoraAdapters = cactusLib
    .lookup<NativeFunction<FreeLoraAdaptersNative>>('cactus_free_lora_adapters_c')
    .asFunction<FreeLoraAdaptersDart>();

final freeChatResultMembers = cactusLib
    .lookup<NativeFunction<FreeChatResultMembersNative>>('cactus_free_chat_result_members_c')
    .asFunction<FreeChatResultMembersDart>();
