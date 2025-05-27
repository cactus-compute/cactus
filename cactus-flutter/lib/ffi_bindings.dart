// ignore_for_file: non_constant_identifier_names, camel_case_types
import 'dart:ffi';
import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';

final class CactusContextOpaque extends Opaque {}
typedef CactusContextHandle = Pointer<CactusContextOpaque>;

final class CactusInitParamsC extends Struct {
  external Pointer<Utf8> model_path;
  external Pointer<Utf8> mmproj_path;
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
  @Bool()
  external bool warmup;
  @Bool()
  external bool mmproj_use_gpu;
  @Int32()
  external int main_gpu;
}

final class CactusCompletionParamsC extends Struct {
  external Pointer<Utf8> prompt;
  external Pointer<Utf8> image_path;

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

final class CactusTokenArrayC extends Struct {
  external Pointer<Int32> tokens;
  @Int32()
  external int count;
}

final class CactusFloatArrayC extends Struct {
  external Pointer<Float> values; 
  @Int32()
  external int count;
}

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

final class CactusVocoderModelParamsC extends Struct {
  external Pointer<Utf8> path;
}

final class CactusVocoderLoadParamsC extends Struct {
  external CactusVocoderModelParamsC model_params;

  external Pointer<Utf8> speaker_file;
  @Bool()
  external bool use_guide_tokens;
}

final class CactusSynthesizeSpeechParamsC extends Struct {
  external Pointer<Utf8> text_input;
  external Pointer<Utf8> output_wav_path;
  external Pointer<Utf8> speaker_id;
}

typedef InitContextNative = CactusContextHandle Function(Pointer<CactusInitParamsC> params);
typedef InitContextDart = CactusContextHandle Function(Pointer<CactusInitParamsC> params);

typedef FreeContextNative = Void Function(CactusContextHandle handle);
typedef FreeContextDart = void Function(CactusContextHandle handle);

typedef CompletionNative = Int32 Function(
    CactusContextHandle handle,
    Pointer<CactusCompletionParamsC> params,
    Pointer<CactusCompletionResultC> result);
typedef CompletionDart = int Function(
    CactusContextHandle handle,
    Pointer<CactusCompletionParamsC> params,
    Pointer<CactusCompletionResultC> result);

typedef StopCompletionNative = Void Function(CactusContextHandle handle);
typedef StopCompletionDart = void Function(CactusContextHandle handle);

typedef TokenizeNative = CactusTokenArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);
typedef TokenizeDart = CactusTokenArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);

typedef DetokenizeNative = Pointer<Utf8> Function(CactusContextHandle handle, Pointer<Int32> tokens, Int32 count);
typedef DetokenizeDart = Pointer<Utf8> Function(CactusContextHandle handle, Pointer<Int32> tokens, int count);

typedef EmbeddingNative = CactusFloatArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);
typedef EmbeddingDart = CactusFloatArrayC Function(CactusContextHandle handle, Pointer<Utf8> text);

typedef _GetFormattedChatNative = Pointer<Utf8> Function(
    CactusContextHandle handle,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> overrideChatTemplate,
    Pointer<Utf8> imagePath);
typedef GetFormattedChatDart = Pointer<Utf8> Function(
    CactusContextHandle handle,
    Pointer<Utf8> messagesJson,
    Pointer<Utf8> overrideChatTemplate,
    Pointer<Utf8> imagePath);

typedef LoadVocoderNative = Int32 Function(
    CactusContextHandle handle,
    Pointer<CactusVocoderLoadParamsC> params
);
typedef LoadVocoderDart = int Function(
    CactusContextHandle handle,
    Pointer<CactusVocoderLoadParamsC> params
);

typedef SynthesizeSpeechNative = Int32 Function(
    CactusContextHandle handle,
    Pointer<CactusSynthesizeSpeechParamsC> params
);
typedef SynthesizeSpeechDart = int Function(
    CactusContextHandle handle,
    Pointer<CactusSynthesizeSpeechParamsC> params
);

typedef FreeStringNative = Void Function(Pointer<Utf8> str);
typedef FreeStringDart = void Function(Pointer<Utf8> str);

typedef FreeTokenArrayNative = Void Function(CactusTokenArrayC arr);
typedef FreeTokenArrayDart = void Function(CactusTokenArrayC arr);

typedef FreeFloatArrayNative = Void Function(CactusFloatArrayC arr);
typedef FreeFloatArrayDart = void Function(CactusFloatArrayC arr);

typedef FreeCompletionResultMembersNative = Void Function(Pointer<CactusCompletionResultC> result);
typedef FreeCompletionResultMembersDart = void Function(Pointer<CactusCompletionResultC> result);

String _getLibraryPath() {
  const String libName = 'cactus'; 
  if (Platform.isIOS) {
    return '$libName.framework/$libName'; 
  }
  if (Platform.isMacOS) {
    return '$libName.framework/$libName'; 
  }
  if (Platform.isAndroid) {
    return 'lib$libName.so';
  }
  throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
}

final DynamicLibrary cactusLib = DynamicLibrary.open(_getLibraryPath());

final initContext = cactusLib
    .lookup<NativeFunction<InitContextNative>>('cactus_init_context_c')
    .asFunction<InitContextDart>();

final freeContext = cactusLib
    .lookup<NativeFunction<FreeContextNative>>('cactus_free_context_c')
    .asFunction<FreeContextDart>();

final completion = cactusLib
    .lookup<NativeFunction<CompletionNative>>('cactus_completion_c')
    .asFunction<CompletionDart>();

final stopCompletion = cactusLib
    .lookup<NativeFunction<StopCompletionNative>>('cactus_stop_completion_c')
    .asFunction<StopCompletionDart>();

final tokenize = cactusLib
    .lookup<NativeFunction<TokenizeNative>>('cactus_tokenize_c')
    .asFunction<TokenizeDart>();

final detokenize = cactusLib
    .lookup<NativeFunction<DetokenizeNative>>('cactus_detokenize_c')
    .asFunction<DetokenizeDart>();

final embedding = cactusLib
    .lookup<NativeFunction<EmbeddingNative>>('cactus_embedding_c')
    .asFunction<EmbeddingDart>();

final getFormattedChat = cactusLib
    .lookup<NativeFunction<_GetFormattedChatNative>>('cactus_get_formatted_chat_c')
    .asFunction<GetFormattedChatDart>();

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

final loadVocoder = cactusLib
    .lookup<NativeFunction<LoadVocoderNative>>('cactus_load_vocoder_c')
    .asFunction<LoadVocoderDart>();

final synthesizeSpeech = cactusLib
    .lookup<NativeFunction<SynthesizeSpeechNative>>('cactus_synthesize_speech_c')
    .asFunction<SynthesizeSpeechDart>();

// +++ LoRA Adapter Management FFI Definitions +++
final class CactusLoraAdapterInfoC extends Struct {
  external Pointer<Utf8> path;
  @Float()
  external double scale; // Dart double for C float
}

typedef ApplyLoraAdaptersNative = Int32 Function(
    CactusContextHandle handle,
    Pointer<CactusLoraAdapterInfoC> adapters,
    Int32 count);
typedef ApplyLoraAdaptersDart = int Function(
    CactusContextHandle handle,
    Pointer<CactusLoraAdapterInfoC> adapters,
    int count);

typedef RemoveLoraAdaptersNative = Void Function(CactusContextHandle handle);
typedef RemoveLoraAdaptersDart = void Function(CactusContextHandle handle);

typedef GetLoadedLoraAdaptersNative = Pointer<CactusLoraAdapterInfoC> Function(
    CactusContextHandle handle,
    Pointer<Int32> count);
typedef GetLoadedLoraAdaptersDart = Pointer<CactusLoraAdapterInfoC> Function(
    CactusContextHandle handle,
    Pointer<Int32> count);

typedef FreeLoraAdaptersArrayNative = Void Function(
    Pointer<CactusLoraAdapterInfoC> adapters,
    Int32 count);
typedef FreeLoraAdaptersArrayDart = void Function(
    Pointer<CactusLoraAdapterInfoC> adapters,
    int count);
// --- End LoRA FFI Definitions ---


// +++ Benchmarking FFI Definitions +++
typedef BenchNative = Pointer<Utf8> Function(
    CactusContextHandle handle,
    Int32 pp,
    Int32 tg,
    Int32 pl,
    Int32 nr);
typedef BenchDart = Pointer<Utf8> Function(
    CactusContextHandle handle,
    int pp,
    int tg,
    int pl,
    int nr);
// --- End Benchmarking FFI Definitions ---


// +++ Advanced Chat Formatting FFI Definitions +++
final class CactusFormattedChatResultC extends Struct {
  external Pointer<Utf8> prompt;
  external Pointer<Utf8> grammar;
  // Add other fields here if they were added to the C struct
}

typedef GetFormattedChatJinjaNative = Int32 Function(
    CactusContextHandle handle,
    Pointer<Utf8> messages_json,
    Pointer<Utf8> override_chat_template,
    Pointer<Utf8> json_schema,
    Pointer<Utf8> tools_json,
    Bool parallel_tool_calls,
    Pointer<Utf8> tool_choice,
    Pointer<CactusFormattedChatResultC> result);
typedef GetFormattedChatJinjaDart = int Function(
    CactusContextHandle handle,
    Pointer<Utf8> messages_json,
    Pointer<Utf8> override_chat_template,
    Pointer<Utf8> json_schema,
    Pointer<Utf8> tools_json,
    bool parallel_tool_calls,
    Pointer<Utf8> tool_choice,
    Pointer<CactusFormattedChatResultC> result);

typedef FreeFormattedChatResultMembersNative = Void Function(
    Pointer<CactusFormattedChatResultC> result);
typedef FreeFormattedChatResultMembersDart = void Function(
    Pointer<CactusFormattedChatResultC> result);
// --- End Advanced Chat Formatting FFI Definitions ---


// +++ Model Information and Validation FFI Definitions +++
typedef ValidateModelChatTemplateNative = Bool Function(
    CactusContextHandle handle,
    Bool use_jinja,
    Pointer<Utf8> name);
typedef ValidateModelChatTemplateDart = bool Function(
    CactusContextHandle handle,
    bool use_jinja,
    Pointer<Utf8> name);

final class CactusModelMetaC extends Struct {
  external Pointer<Utf8> desc;
  @Int64()
  external int size;
  @Int64()
  external int n_params;
}

typedef GetModelMetaNative = Void Function(
    CactusContextHandle handle,
    Pointer<CactusModelMetaC> meta);
typedef GetModelMetaDart = void Function(
    CactusContextHandle handle,
    Pointer<CactusModelMetaC> meta);

typedef FreeModelMetaMembersNative = Void Function(
    Pointer<CactusModelMetaC> meta);
typedef FreeModelMetaMembersDart = void Function(
    Pointer<CactusModelMetaC> meta);
// --- End Model Information and Validation FFI Definitions ---

// +++ LoRA Lookups +++
final applyLoraAdapters = cactusLib
    .lookup<NativeFunction<ApplyLoraAdaptersNative>>('cactus_apply_lora_adapters_c')
    .asFunction<ApplyLoraAdaptersDart>();

final removeLoraAdapters = cactusLib
    .lookup<NativeFunction<RemoveLoraAdaptersNative>>('cactus_remove_lora_adapters_c')
    .asFunction<RemoveLoraAdaptersDart>();

final getLoadedLoraAdapters = cactusLib
    .lookup<NativeFunction<GetLoadedLoraAdaptersNative>>('cactus_get_loaded_lora_adapters_c')
    .asFunction<GetLoadedLoraAdaptersDart>();

final freeLoraAdaptersArray = cactusLib
    .lookup<NativeFunction<FreeLoraAdaptersArrayNative>>('cactus_free_lora_adapters_array_c')
    .asFunction<FreeLoraAdaptersArrayDart>();

// +++ Benchmarking Lookup +++
final bench = cactusLib
    .lookup<NativeFunction<BenchNative>>('cactus_bench_c')
    .asFunction<BenchDart>();

// +++ Advanced Chat Formatting Lookups +++
final getFormattedChatJinja = cactusLib
    .lookup<NativeFunction<GetFormattedChatJinjaNative>>('cactus_get_formatted_chat_jinja_c')
    .asFunction<GetFormattedChatJinjaDart>();

final freeFormattedChatResultMembers = cactusLib
    .lookup<NativeFunction<FreeFormattedChatResultMembersNative>>('cactus_free_formatted_chat_result_members_c')
    .asFunction<FreeFormattedChatResultMembersDart>();

// +++ Model Info/Validation Lookups +++
final validateModelChatTemplate = cactusLib
    .lookup<NativeFunction<ValidateModelChatTemplateNative>>('cactus_validate_model_chat_template_c')
    .asFunction<ValidateModelChatTemplateDart>();

final getModelMeta = cactusLib
    .lookup<NativeFunction<GetModelMetaNative>>('cactus_get_model_meta_c')
    .asFunction<GetModelMetaDart>();

final freeModelMetaMembers = cactusLib
    .lookup<NativeFunction<FreeModelMetaMembersNative>>('cactus_free_model_meta_members_c')
    .asFunction<FreeModelMetaMembersDart>();
