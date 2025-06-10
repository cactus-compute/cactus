package com.cactus;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import android.util.Log;
import android.os.Build;
import android.content.res.AssetManager;

import java.lang.StringBuilder;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;

public class LlamaContext {
  public static final String NAME = "CactusContext";

  private static String loadedLibrary = "";

  private static class NativeLogCallback {
    DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

    public NativeLogCallback(ReactApplicationContext reactContext) {
      this.eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    }

    void emitNativeLog(String level, String text) {
      WritableMap event = Arguments.createMap();
      event.putString("level", level);
      event.putString("text", text);
      eventEmitter.emit("@Cactus_onNativeLog", event);
    }
  }

  static void toggleNativeLog(ReactApplicationContext reactContext, boolean enabled) {
    if (LlamaContext.isArchNotSupported()) {
      throw new IllegalStateException("Only 64-bit architectures are supported");
    }
    if (enabled) {
      setupLog(new NativeLogCallback(reactContext));
    } else {
      unsetLog();
    }
  }

  private int id;
  private ReactApplicationContext reactContext;
  private long context;
  private WritableMap modelDetails;
  private int jobId = -1;
  private DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

  public LlamaContext(int id, ReactApplicationContext reactContext, ReadableMap params) {
    if (LlamaContext.isArchNotSupported()) {
      throw new IllegalStateException("Only 64-bit architectures are supported");
    }
    if (!params.hasKey("model")) {
      throw new IllegalArgumentException("Missing required parameter: model");
    }
    eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    this.id = id;
    this.context = initContext(
      // String model,
      params.getString("model"),
      // String chat_template,
      params.hasKey("chat_template") ? params.getString("chat_template") : "",
      // String reasoning_format,
      params.hasKey("reasoning_format") ? params.getString("reasoning_format") : "none",
      // boolean embedding,
      params.hasKey("embedding") ? params.getBoolean("embedding") : false,
      // int embd_normalize,
      params.hasKey("embd_normalize") ? params.getInt("embd_normalize") : -1,
      // int n_ctx,
      params.hasKey("n_ctx") ? params.getInt("n_ctx") : 512,
      // int n_batch,
      params.hasKey("n_batch") ? params.getInt("n_batch") : 512,
      // int n_ubatch,
      params.hasKey("n_ubatch") ? params.getInt("n_ubatch") : 512,
      // int n_threads,
      params.hasKey("n_threads") ? params.getInt("n_threads") : 0,
      // int n_gpu_layers, // TODO: Support this
      params.hasKey("n_gpu_layers") ? params.getInt("n_gpu_layers") : 0,
      // boolean flash_attn,
      params.hasKey("flash_attn") ? params.getBoolean("flash_attn") : false,
      // String cache_type_k,
      params.hasKey("cache_type_k") ? params.getString("cache_type_k") : "f16",
      // String cache_type_v,
      params.hasKey("cache_type_v") ? params.getString("cache_type_v") : "f16",
      // boolean use_mlock,
      params.hasKey("use_mlock") ? params.getBoolean("use_mlock") : true,
      // boolean use_mmap,
      params.hasKey("use_mmap") ? params.getBoolean("use_mmap") : true,
      //boolean vocab_only,
      params.hasKey("vocab_only") ? params.getBoolean("vocab_only") : false,
      // String lora,
      params.hasKey("lora") ? params.getString("lora") : "",
      // float lora_scaled,
      params.hasKey("lora_scaled") ? (float) params.getDouble("lora_scaled") : 1.0f,
      // ReadableArray lora_adapters,
      params.hasKey("lora_list") ? params.getArray("lora_list") : null,
      // float rope_freq_base,
      params.hasKey("rope_freq_base") ? (float) params.getDouble("rope_freq_base") : 0.0f,
      // float rope_freq_scale
      params.hasKey("rope_freq_scale") ? (float) params.getDouble("rope_freq_scale") : 0.0f,
      // int pooling_type,
      params.hasKey("pooling_type") ? params.getInt("pooling_type") : -1,
      // LoadProgressCallback load_progress_callback
      params.hasKey("use_progress_callback") ? new LoadProgressCallback(this) : null
    );
    if (this.context == -1) {
      throw new IllegalStateException("Failed to initialize context");
    }
    this.modelDetails = loadModelDetails(this.context);
    this.reactContext = reactContext;
  }

  public void interruptLoad() {
    interruptLoad(this.context);
  }

  public long getContext() {
    return context;
  }

  public WritableMap getModelDetails() {
    return modelDetails;
  }

  public String getLoadedLibrary() {
    return loadedLibrary;
  }

  public WritableMap getFormattedChatWithJinja(String messages, String chatTemplate, ReadableMap params) {
    String jsonSchema = params.hasKey("json_schema") ? params.getString("json_schema") : "";
    String tools = params.hasKey("tools") ? params.getString("tools") : "";
    Boolean parallelToolCalls = params.hasKey("parallel_tool_calls") ? params.getBoolean("parallel_tool_calls") : false;
    String toolChoice = params.hasKey("tool_choice") ? params.getString("tool_choice") : "";
    return getFormattedChatWithJinja(
      this.context,
      messages,
      chatTemplate == null ? "" : chatTemplate,
      jsonSchema,
      tools,
      parallelToolCalls,
      toolChoice
    );
  }

  public String getFormattedChat(String messages, String chatTemplate) {
    return getFormattedChat(this.context, messages, chatTemplate == null ? "" : chatTemplate);
  }

  private void emitLoadProgress(int progress) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", LlamaContext.this.id);
    event.putInt("progress", progress);
    eventEmitter.emit("@Cactus_onInitContextProgress", event);
  }

  private static class LoadProgressCallback {
    LlamaContext context;

    public LoadProgressCallback(LlamaContext context) {
      this.context = context;
    }

    void onLoadProgress(int progress) {
      context.emitLoadProgress(progress);
    }
  }

  private void emitPartialCompletion(WritableMap tokenResult) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", LlamaContext.this.id);
    event.putMap("tokenResult", tokenResult);
    eventEmitter.emit("@Cactus_onToken", event);
  }

  private static class PartialCompletionCallback {
    LlamaContext context;
    boolean emitNeeded;

    public PartialCompletionCallback(LlamaContext context, boolean emitNeeded) {
      this.context = context;
      this.emitNeeded = emitNeeded;
    }

    void onPartialCompletion(WritableMap tokenResult) {
      if (!emitNeeded) return;
      context.emitPartialCompletion(tokenResult);
    }
  }

  public WritableMap loadSession(String path) {
    if (path == null || path.isEmpty()) {
      throw new IllegalArgumentException("File path is empty");
    }
    File file = new File(path);
    if (!file.exists()) {
      throw new IllegalArgumentException("File does not exist: " + path);
    }
    WritableMap result = loadSession(this.context, path);
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public int saveSession(String path, int size) {
    if (path == null || path.isEmpty()) {
      throw new IllegalArgumentException("File path is empty");
    }
    return saveSession(this.context, path, size);
  }

  public WritableMap completion(ReadableMap params) {
    Log.d(NAME, "🔵 ANDROID: completion() called");
    if (!params.hasKey("prompt")) {
      throw new IllegalArgumentException("Missing required parameter: prompt");
    }

    Log.d(NAME, "📝 ANDROID: prompt length = " + params.getString("prompt").length());
    Log.d(NAME, "🧮 ANDROID: context ptr = " + this.context);

    double[][] logit_bias = new double[0][0];
    if (params.hasKey("logit_bias")) {
      ReadableArray logit_bias_array = params.getArray("logit_bias");
      logit_bias = new double[logit_bias_array.size()][];
      for (int i = 0; i < logit_bias_array.size(); i++) {
        ReadableArray logit_bias_row = logit_bias_array.getArray(i);
        logit_bias[i] = new double[logit_bias_row.size()];
        for (int j = 0; j < logit_bias_row.size(); j++) {
          logit_bias[i][j] = logit_bias_row.getDouble(j);
        }
      }
    }

    Log.d(NAME, "🚀 ANDROID: About to call doCompletion native method...");
    WritableMap result = doCompletion(
      this.context,
      // String prompt,
      params.getString("prompt"),
      // int chat_format,
      params.hasKey("chat_format") ? params.getInt("chat_format") : 0,
      // String grammar,
      params.hasKey("grammar") ? params.getString("grammar") : "",
      // String json_schema,
      params.hasKey("json_schema") ? params.getString("json_schema") : "",
      // boolean grammar_lazy,
      params.hasKey("grammar_lazy") ? params.getBoolean("grammar_lazy") : false,
      // ReadableArray grammar_triggers,
      params.hasKey("grammar_triggers") ? params.getArray("grammar_triggers") : null,
      // ReadableArray preserved_tokens,
      params.hasKey("preserved_tokens") ? params.getArray("preserved_tokens") : null,
      // float temperature,
      params.hasKey("temperature") ? (float) params.getDouble("temperature") : 0.7f,
      // int n_threads,
      params.hasKey("n_threads") ? params.getInt("n_threads") : 0,
      // int n_predict,
      params.hasKey("n_predict") ? params.getInt("n_predict") : -1,
      // int n_probs,
      params.hasKey("n_probs") ? params.getInt("n_probs") : 0,
      // int penalty_last_n,
      params.hasKey("penalty_last_n") ? params.getInt("penalty_last_n") : 64,
      // float penalty_repeat,
      params.hasKey("penalty_repeat") ? (float) params.getDouble("penalty_repeat") : 1.00f,
      // float penalty_freq,
      params.hasKey("penalty_freq") ? (float) params.getDouble("penalty_freq") : 0.00f,
      // float penalty_present,
      params.hasKey("penalty_present") ? (float) params.getDouble("penalty_present") : 0.00f,
      // float mirostat,
      params.hasKey("mirostat") ? (float) params.getDouble("mirostat") : 0.00f,
      // float mirostat_tau,
      params.hasKey("mirostat_tau") ? (float) params.getDouble("mirostat_tau") : 5.00f,
      // float mirostat_eta,
      params.hasKey("mirostat_eta") ? (float) params.getDouble("mirostat_eta") : 0.10f,
      // int top_k,
      params.hasKey("top_k") ? params.getInt("top_k") : 40,
      // float top_p,
      params.hasKey("top_p") ? (float) params.getDouble("top_p") : 0.95f,
      // float min_p,
      params.hasKey("min_p") ? (float) params.getDouble("min_p") : 0.05f,
      // float xtc_threshold,
      params.hasKey("xtc_threshold") ? (float) params.getDouble("xtc_threshold") : 0.00f,
      // float xtc_probability,
      params.hasKey("xtc_probability") ? (float) params.getDouble("xtc_probability") : 0.00f,
      // float typical_p,
      params.hasKey("typical_p") ? (float) params.getDouble("typical_p") : 1.00f,
      // int seed,
      params.hasKey("seed") ? params.getInt("seed") : -1,
      // String[] stop,
      params.hasKey("stop") ? params.getArray("stop").toArrayList().toArray(new String[0]) : new String[0],
      // boolean ignore_eos,
      params.hasKey("ignore_eos") ? params.getBoolean("ignore_eos") : false,
      // double[][] logit_bias,
      logit_bias,
      // float dry_multiplier,
      params.hasKey("dry_multiplier") ? (float) params.getDouble("dry_multiplier") : 0.00f,
      // float dry_base,
      params.hasKey("dry_base") ? (float) params.getDouble("dry_base") : 1.75f,
      // int dry_allowed_length,
      params.hasKey("dry_allowed_length") ? params.getInt("dry_allowed_length") : 2,
      // int dry_penalty_last_n,
      params.hasKey("dry_penalty_last_n") ? params.getInt("dry_penalty_last_n") : -1,
      // float top_n_sigma,
      params.hasKey("top_n_sigma") ? (float) params.getDouble("top_n_sigma") : -1.0f,
      // String[] dry_sequence_breakers, when undef, we use the default definition from common.h
      params.hasKey("dry_sequence_breakers") ? params.getArray("dry_sequence_breakers").toArrayList().toArray(new String[0]) : new String[]{"\n", ":", "\"", "*"},
      // PartialCompletionCallback partial_completion_callback
      new PartialCompletionCallback(
        this,
        params.hasKey("emit_partial_completion") ? params.getBoolean("emit_partial_completion") : false
      )
    );
    Log.d(NAME, "✅ ANDROID: doCompletion returned successfully");
    if (result.hasKey("error")) {
      Log.e(NAME, "❌ ANDROID: doCompletion returned error: " + result.getString("error"));
      throw new IllegalStateException(result.getString("error"));
    }
    Log.d(NAME, "📤 ANDROID: completion() returning result");
    return result;
  }

  public void stopCompletion() {
    stopCompletion(this.context);
  }

  public boolean isPredicting() {
    return isPredicting(this.context);
  }

  public WritableMap tokenize(String text) {
    WritableMap result = Arguments.createMap();
    result.putArray("tokens", tokenize(this.context, text));
    return result;
  }

  public String detokenize(ReadableArray tokens) {
    int[] toks = new int[tokens.size()];
    for (int i = 0; i < tokens.size(); i++) {
      toks[i] = (int) tokens.getDouble(i);
    }
    return detokenize(this.context, toks);
  }

  public WritableMap getEmbedding(String text, ReadableMap params) {
    if (isEmbeddingEnabled(this.context) == false) {
      throw new IllegalStateException("Embedding is not enabled");
    }
    WritableMap result = embedding(
      this.context,
      text,
      // int embd_normalize,
      params.hasKey("embd_normalize") ? params.getInt("embd_normalize") : -1
    );
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public String bench(int pp, int tg, int pl, int nr) {
    return bench(this.context, pp, tg, pl, nr);
  }

  public int applyLoraAdapters(ReadableArray loraAdapters) {
    int result = applyLoraAdapters(this.context, loraAdapters);
    if (result != 0) {
      throw new IllegalStateException("Failed to apply lora adapters");
    }
    return result;
  }

  public void removeLoraAdapters() {
    removeLoraAdapters(this.context);
  }

  public WritableArray getLoadedLoraAdapters() {
    return getLoadedLoraAdapters(this.context);
  }

  public void release() {
    freeContext(context);
  }

  // Multimodal methods
  public boolean initMultimodal(String mmprojPath, boolean useGpu) {
    return initMultimodal(this.context, mmprojPath, useGpu);
  }

  public boolean isMultimodalEnabled() {
    return isMultimodalEnabled(this.context);
  }

  public boolean isMultimodalSupportVision() {
    return isMultimodalSupportVision(this.context);
  }

  public boolean isMultimodalSupportAudio() {
    return isMultimodalSupportAudio(this.context);
  }

  public void releaseMultimodal() {
    releaseMultimodal(this.context);
  }

  public WritableMap multimodalCompletion(String prompt, ReadableArray mediaPaths, ReadableMap params) {
    if (!params.hasKey("prompt")) {
      params = Arguments.createMap();
      ((WritableMap) params).putString("prompt", prompt);
    }

    String[] mediaPathsArray = new String[mediaPaths.size()];
    for (int i = 0; i < mediaPaths.size(); i++) {
      mediaPathsArray[i] = mediaPaths.getString(i);
    }

    double[][] logit_bias = new double[0][0];
    if (params.hasKey("logit_bias")) {
      ReadableArray logit_bias_array = params.getArray("logit_bias");
      logit_bias = new double[logit_bias_array.size()][];
      for (int i = 0; i < logit_bias_array.size(); i++) {
        ReadableArray logit_bias_row = logit_bias_array.getArray(i);
        logit_bias[i] = new double[logit_bias_row.size()];
        for (int j = 0; j < logit_bias_row.size(); j++) {
          logit_bias[i][j] = logit_bias_row.getDouble(j);
        }
      }
    }

    WritableMap result = doMultimodalCompletion(
      this.context,
      prompt,
      mediaPathsArray,
      params.hasKey("chat_format") ? params.getInt("chat_format") : 0,
      params.hasKey("grammar") ? params.getString("grammar") : "",
      params.hasKey("json_schema") ? params.getString("json_schema") : "",
      params.hasKey("grammar_lazy") ? params.getBoolean("grammar_lazy") : false,
      params.hasKey("grammar_triggers") ? params.getArray("grammar_triggers") : null,
      params.hasKey("preserved_tokens") ? params.getArray("preserved_tokens") : null,
      params.hasKey("temperature") ? (float) params.getDouble("temperature") : 0.7f,
      params.hasKey("n_threads") ? params.getInt("n_threads") : 0,
      params.hasKey("n_predict") ? params.getInt("n_predict") : -1,
      params.hasKey("n_probs") ? params.getInt("n_probs") : 0,
      params.hasKey("penalty_last_n") ? params.getInt("penalty_last_n") : 64,
      params.hasKey("penalty_repeat") ? (float) params.getDouble("penalty_repeat") : 1.00f,
      params.hasKey("penalty_freq") ? (float) params.getDouble("penalty_freq") : 0.00f,
      params.hasKey("penalty_present") ? (float) params.getDouble("penalty_present") : 0.00f,
      params.hasKey("mirostat") ? (float) params.getDouble("mirostat") : 0.00f,
      params.hasKey("mirostat_tau") ? (float) params.getDouble("mirostat_tau") : 5.00f,
      params.hasKey("mirostat_eta") ? (float) params.getDouble("mirostat_eta") : 0.10f,
      params.hasKey("top_k") ? params.getInt("top_k") : 40,
      params.hasKey("top_p") ? (float) params.getDouble("top_p") : 0.95f,
      params.hasKey("min_p") ? (float) params.getDouble("min_p") : 0.05f,
      params.hasKey("xtc_threshold") ? (float) params.getDouble("xtc_threshold") : 0.00f,
      params.hasKey("xtc_probability") ? (float) params.getDouble("xtc_probability") : 0.00f,
      params.hasKey("typical_p") ? (float) params.getDouble("typical_p") : 1.00f,
      params.hasKey("seed") ? params.getInt("seed") : -1,
      params.hasKey("stop") ? params.getArray("stop").toArrayList().toArray(new String[0]) : new String[0],
      params.hasKey("ignore_eos") ? params.getBoolean("ignore_eos") : false,
      logit_bias,
      params.hasKey("dry_multiplier") ? (float) params.getDouble("dry_multiplier") : 0.00f,
      params.hasKey("dry_base") ? (float) params.getDouble("dry_base") : 1.75f,
      params.hasKey("dry_allowed_length") ? params.getInt("dry_allowed_length") : 2,
      params.hasKey("dry_penalty_last_n") ? params.getInt("dry_penalty_last_n") : -1,
      params.hasKey("top_n_sigma") ? (float) params.getDouble("top_n_sigma") : -1.0f,
      params.hasKey("dry_sequence_breakers") ? params.getArray("dry_sequence_breakers").toArrayList().toArray(new String[0]) : new String[]{"\n", ":", "\"", "*"},
      new PartialCompletionCallback(
        this,
        params.hasKey("emit_partial_completion") ? params.getBoolean("emit_partial_completion") : false
      )
    );
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public WritableMap tokenize(String text, ReadableArray mediaPaths) {
    String[] mediaPathsArray = new String[mediaPaths.size()];
    for (int i = 0; i < mediaPaths.size(); i++) {
      mediaPathsArray[i] = mediaPaths.getString(i);
    }
    WritableMap result = tokenize(this.context, text, mediaPathsArray);
    return result;
  }

  // TTS/Vocoder methods
  public boolean initVocoder(String vocoderModelPath) {
    return initVocoder(this.context, vocoderModelPath);
  }

  public boolean isVocoderEnabled() {
    return isVocoderEnabled(this.context);
  }

  public int getTTSType() {
    return getTTSType(this.context);
  }

  public String getFormattedAudioCompletion(String speakerJsonStr, String textToSpeak) {
    return getFormattedAudioCompletion(this.context, speakerJsonStr, textToSpeak);
  }

  public WritableArray getAudioCompletionGuideTokens(String textToSpeak) {
    return getAudioCompletionGuideTokens(this.context, textToSpeak);
  }

  public WritableArray decodeAudioTokens(ReadableArray tokens) {
    int[] toks = new int[tokens.size()];
    for (int i = 0; i < tokens.size(); i++) {
      toks[i] = (int) tokens.getDouble(i);
    }
    return decodeAudioTokens(this.context, toks);
  }

  public void releaseVocoder() {
    releaseVocoder(this.context);
  }

  static {
    Log.d(NAME, "Primary ABI: " + Build.SUPPORTED_ABIS[0]);

    String cpuFeatures = LlamaContext.getCpuFeatures();
    Log.d(NAME, "CPU features: " + cpuFeatures);
    boolean hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");
    boolean hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp");
    boolean hasSve = cpuFeatures.contains("sve");
    boolean hasI8mm = cpuFeatures.contains("i8mm");
    boolean isAtLeastArmV82 = cpuFeatures.contains("asimd") && cpuFeatures.contains("crc32") && cpuFeatures.contains("aes");
    boolean isAtLeastArmV84 = cpuFeatures.contains("dcpop") && cpuFeatures.contains("uscat");
    Log.d(NAME, "- hasFp16: " + hasFp16);
    Log.d(NAME, "- hasDotProd: " + hasDotProd);
    Log.d(NAME, "- hasSve: " + hasSve);
    Log.d(NAME, "- hasI8mm: " + hasI8mm);
    Log.d(NAME, "- isAtLeastArmV82: " + isAtLeastArmV82);
    Log.d(NAME, "- isAtLeastArmV84: " + isAtLeastArmV84);

    // TODO: Add runtime check for cpu features
    if (LlamaContext.isArm64V8a()) {
      if (hasDotProd && hasI8mm) {
        Log.d(NAME, "Loading libcactus_v8_2_dotprod_i8mm.so");
        System.loadLibrary("cactus_v8_2_dotprod_i8mm");
        loadedLibrary = "cactus_v8_2_dotprod_i8mm";
      } else if (hasDotProd) {
        Log.d(NAME, "Loading libcactus_v8_2_dotprod.so");
        System.loadLibrary("cactus_v8_2_dotprod");
        loadedLibrary = "cactus_v8_2_dotprod";
      } else if (hasI8mm) {
        Log.d(NAME, "Loading libcactus_v8_2_i8mm.so");
        System.loadLibrary("cactus_v8_2_i8mm");
        loadedLibrary = "cactus_v8_2_i8mm";
      } else if (hasFp16) {
        Log.d(NAME, "Loading libcactus_v8_2.so");
        System.loadLibrary("cactus_v8_2");
        loadedLibrary = "cactus_v8_2";
      } else {
        Log.d(NAME, "Loading default libcactus_v8.so");
        System.loadLibrary("cactus_v8");
        loadedLibrary = "cactus_v8";
      }
      //  Log.d(NAME, "Loading libcactus_v8_7.so with runtime feature detection");
      //  System.loadLibrary("cactus_v8_7");
    } else if (LlamaContext.isX86_64()) {
      Log.d(NAME, "Loading libcactus_x86_64.so");
      System.loadLibrary("cactus_x86_64");
      loadedLibrary = "cactus_x86_64";
    } else {
      Log.d(NAME, "ARM32 is not supported, skipping loading library");
    }
  }

  private static boolean isArm64V8a() {
    return Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  private static boolean isX86_64() {
    return Build.SUPPORTED_ABIS[0].equals("x86_64");
  }

  private static boolean isArchNotSupported() {
    return isArm64V8a() == false && isX86_64() == false;
  }

  private static String getCpuFeatures() {
    File file = new File("/proc/cpuinfo");
    StringBuilder stringBuilder = new StringBuilder();
    try {
      BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
      String line;
      while ((line = bufferedReader.readLine()) != null) {
        if (line.startsWith("Features")) {
          stringBuilder.append(line);
          break;
        }
      }
      bufferedReader.close();
      return stringBuilder.toString();
    } catch (IOException e) {
      Log.w(NAME, "Couldn't read /proc/cpuinfo", e);
      return "";
    }
  }

  protected static native WritableMap modelInfo(
    String model,
    String[] skip
  );
  protected static native long initContext(
    String model,
    String chat_template,
    String reasoning_format,
    boolean embedding,
    int embd_normalize,
    int n_ctx,
    int n_batch,
    int n_ubatch,
    int n_threads,
    int n_gpu_layers, // TODO: Support this
    boolean flash_attn,
    String cache_type_k,
    String cache_type_v,
    boolean use_mlock,
    boolean use_mmap,
    boolean vocab_only,
    String lora,
    float lora_scaled,
    ReadableArray lora_list,
    float rope_freq_base,
    float rope_freq_scale,
    int pooling_type,
    LoadProgressCallback load_progress_callback
  );
  protected static native void interruptLoad(long contextPtr);
  protected static native WritableMap loadModelDetails(
    long contextPtr
  );
  protected static native WritableMap getFormattedChatWithJinja(
    long contextPtr,
    String messages,
    String chatTemplate,
    String jsonSchema,
    String tools,
    boolean parallelToolCalls,
    String toolChoice
  );
  protected static native String getFormattedChat(
    long contextPtr,
    String messages,
    String chatTemplate
  );
  protected static native WritableMap loadSession(
    long contextPtr,
    String path
  );
  protected static native int saveSession(
    long contextPtr,
    String path,
    int size
  );
  protected static native WritableMap doCompletion(
    long context_ptr,
    String prompt,
    int chat_format,
    String grammar,
    String json_schema,
    boolean grammar_lazy,
    ReadableArray grammar_triggers,
    ReadableArray preserved_tokens,
    float temperature,
    int n_threads,
    int n_predict,
    int n_probs,
    int penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present,
    float mirostat,
    float mirostat_tau,
    float mirostat_eta,
    int top_k,
    float top_p,
    float min_p,
    float xtc_threshold,
    float xtc_probability,
    float typical_p,
    int seed,
    String[] stop,
    boolean ignore_eos,
    double[][] logit_bias,
    float   dry_multiplier,
    float   dry_base,
    int dry_allowed_length,
    int dry_penalty_last_n,
    float top_n_sigma,
    String[] dry_sequence_breakers,
    PartialCompletionCallback partial_completion_callback
  );
  protected static native void stopCompletion(long contextPtr);
  protected static native boolean isPredicting(long contextPtr);
  protected static native WritableArray tokenize(long contextPtr, String text);
  protected static native String detokenize(long contextPtr, int[] tokens);
  protected static native boolean isEmbeddingEnabled(long contextPtr);
  protected static native WritableMap embedding(
    long contextPtr,
    String text,
    int embd_normalize
  );
  protected static native String bench(long contextPtr, int pp, int tg, int pl, int nr);
  protected static native int applyLoraAdapters(long contextPtr, ReadableArray loraAdapters);
  protected static native void removeLoraAdapters(long contextPtr);
  protected static native WritableArray getLoadedLoraAdapters(long contextPtr);
  protected static native void freeContext(long contextPtr);
  protected static native void setupLog(NativeLogCallback logCallback);
  protected static native void unsetLog();

  // Multimodal native methods
  protected static native boolean initMultimodal(long contextPtr, String mmprojPath, boolean useGpu);
  protected static native boolean isMultimodalEnabled(long contextPtr);
  protected static native boolean isMultimodalSupportVision(long contextPtr);
  protected static native boolean isMultimodalSupportAudio(long contextPtr);
  protected static native void releaseMultimodal(long contextPtr);
  protected static native WritableMap doMultimodalCompletion(
    long contextPtr,
    String prompt,
    String[] mediaPaths,
    int chat_format,
    String grammar,
    String json_schema,
    boolean grammar_lazy,
    ReadableArray grammar_triggers,
    ReadableArray preserved_tokens,
    float temperature,
    int n_threads,
    int n_predict,
    int n_probs,
    int penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present,
    float mirostat,
    float mirostat_tau,
    float mirostat_eta,
    int top_k,
    float top_p,
    float min_p,
    float xtc_threshold,
    float xtc_probability,
    float typical_p,
    int seed,
    String[] stop,
    boolean ignore_eos,
    double[][] logit_bias,
    float dry_multiplier,
    float dry_base,
    int dry_allowed_length,
    int dry_penalty_last_n,
    float top_n_sigma,
    String[] dry_sequence_breakers,
    PartialCompletionCallback partial_completion_callback
  );
  protected static native WritableMap tokenize(long contextPtr, String text, String[] mediaPaths);

  // TTS/Vocoder native methods
  protected static native boolean initVocoder(long contextPtr, String vocoderModelPath);
  protected static native boolean isVocoderEnabled(long contextPtr);
  protected static native int getTTSType(long contextPtr);
  protected static native String getFormattedAudioCompletion(long contextPtr, String speakerJsonStr, String textToSpeak);
  protected static native WritableArray getAudioCompletionGuideTokens(long contextPtr, String textToSpeak);
  protected static native WritableArray decodeAudioTokens(long contextPtr, int[] tokens);
  protected static native void releaseVocoder(long contextPtr);
}
