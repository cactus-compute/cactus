#include <jni.h>
// #include <android/asset_manager.h>
// #include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstdlib>
#include <ctime>
#include <sys/sysinfo.h>
#include <string>
#include <thread>
#include <unordered_map>
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "llama-impl.h"
#include "ggml.h"
#include "cactus.h"
#include "jni-utils.h"
#define UNUSED(x) (void)(x)
#define TAG "CACTUS_ANDROID_JNI"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    TAG, __VA_ARGS__)
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

static void cactus_log_callback_default(lm_ggml_log_level level, const char * fmt, void * data) {
    if (level == LM_GGML_LOG_LEVEL_ERROR)     __android_log_print(ANDROID_LOG_ERROR, TAG, fmt, data);
    else if (level == LM_GGML_LOG_LEVEL_INFO) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, data);
    else if (level == LM_GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, fmt, data);
    else __android_log_print(ANDROID_LOG_DEFAULT, TAG, fmt, data);
}

extern "C" {

// Method to create WritableMap
static inline jobject createWriteableMap(JNIEnv *env) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/Arguments");
    jmethodID init = env->GetStaticMethodID(mapClass, "createMap", "()Lcom/facebook/react/bridge/WritableMap;");
    jobject map = env->CallStaticObjectMethod(mapClass, init);
    return map;
}

// Method to put string into WritableMap
static inline void putString(JNIEnv *env, jobject map, const char *key, const char *value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putStringMethod = env->GetMethodID(mapClass, "putString", "(Ljava/lang/String;Ljava/lang/String;)V");

    jstring jKey = env->NewStringUTF(key);
    jstring jValue = env->NewStringUTF(value);

    env->CallVoidMethod(map, putStringMethod, jKey, jValue);
}

// Method to put int into WritableMap
static inline void putInt(JNIEnv *env, jobject map, const char *key, int value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putIntMethod = env->GetMethodID(mapClass, "putInt", "(Ljava/lang/String;I)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putIntMethod, jKey, value);
}

// Method to put double into WritableMap
static inline void putDouble(JNIEnv *env, jobject map, const char *key, double value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putDoubleMethod = env->GetMethodID(mapClass, "putDouble", "(Ljava/lang/String;D)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putDoubleMethod, jKey, value);
}

// Method to put boolean into WritableMap
static inline void putBoolean(JNIEnv *env, jobject map, const char *key, bool value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putBooleanMethod = env->GetMethodID(mapClass, "putBoolean", "(Ljava/lang/String;Z)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putBooleanMethod, jKey, value);
}

// Method to put WriteableMap into WritableMap
static inline void putMap(JNIEnv *env, jobject map, const char *key, jobject value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putMapMethod = env->GetMethodID(mapClass, "putMap", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableMap;)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putMapMethod, jKey, value);
}

// Method to create WritableArray
static inline jobject createWritableArray(JNIEnv *env) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/Arguments");
    jmethodID init = env->GetStaticMethodID(mapClass, "createArray", "()Lcom/facebook/react/bridge/WritableArray;");
    jobject map = env->CallStaticObjectMethod(mapClass, init);
    return map;
}

// Method to push int into WritableArray
static inline void pushInt(JNIEnv *env, jobject arr, int value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    jmethodID pushIntMethod = env->GetMethodID(mapClass, "pushInt", "(I)V");

    env->CallVoidMethod(arr, pushIntMethod, value);
}

// Method to push double into WritableArray
static inline void pushDouble(JNIEnv *env, jobject arr, double value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    jmethodID pushDoubleMethod = env->GetMethodID(mapClass, "pushDouble", "(D)V");

    env->CallVoidMethod(arr, pushDoubleMethod, value);
}

// Method to push string into WritableArray
static inline void pushString(JNIEnv *env, jobject arr, const char *value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    jmethodID pushStringMethod = env->GetMethodID(mapClass, "pushString", "(Ljava/lang/String;)V");

    jstring jValue = env->NewStringUTF(value);
    env->CallVoidMethod(arr, pushStringMethod, jValue);
}

// Method to push WritableMap into WritableArray
static inline void pushMap(JNIEnv *env, jobject arr, jobject value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    jmethodID pushMapMethod = env->GetMethodID(mapClass, "pushMap", "(Lcom/facebook/react/bridge/ReadableMap;)V");

    env->CallVoidMethod(arr, pushMapMethod, value);
}

// Method to put WritableArray into WritableMap
static inline void putArray(JNIEnv *env, jobject map, const char *key, jobject value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putArrayMethod = env->GetMethodID(mapClass, "putArray", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableArray;)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putArrayMethod, jKey, value);
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_modelInfo(
    JNIEnv *env,
    jobject thiz,
    jstring model_path_str,
    jobjectArray skip
) {
    UNUSED(thiz);

    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);

    std::vector<std::string> skip_vec;
    int skip_len = env->GetArrayLength(skip);
    for (int i = 0; i < skip_len; i++) {
        jstring skip_str = (jstring) env->GetObjectArrayElement(skip, i);
        const char *skip_chars = env->GetStringUTFChars(skip_str, nullptr);
        skip_vec.push_back(skip_chars);
        env->ReleaseStringUTFChars(skip_str, skip_chars);
    }

    struct lm_gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };
    struct lm_gguf_context * ctx = lm_gguf_init_from_file(model_path_chars, params);

    if (!ctx) {
        LOGI("%s: failed to load '%s'\n", __func__, model_path_chars);
        return nullptr;
    }

    auto info = createWriteableMap(env);
    putInt(env, info, "version", lm_gguf_get_version(ctx));
    putInt(env, info, "alignment", lm_gguf_get_alignment(ctx));
    putInt(env, info, "data_offset", lm_gguf_get_data_offset(ctx));
    {
        const int n_kv = lm_gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = lm_gguf_get_key(ctx, i);

            bool skipped = false;
            if (skip_len > 0) {
                for (int j = 0; j < skip_len; j++) {
                    if (skip_vec[j] == key) {
                        skipped = true;
                        break;
                    }
                }
            }

            if (skipped) {
                continue;
            }

            const std::string value = lm_gguf_kv_to_str(ctx, i);
            putString(env, info, key, value.c_str());
        }
    }

    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    lm_gguf_free(ctx);

    return reinterpret_cast<jobject>(info);
}

struct callback_context {
    JNIEnv *env;
    cactus::cactus_context *llama;
    jobject callback;
};

std::unordered_map<long, cactus::cactus_context *> context_map;

JNIEXPORT jlong JNICALL
Java_com_cactus_LlamaContext_initContext(
    JNIEnv *env,
    jobject thiz,
    jstring model_path_str,
    jstring chat_template,
    jstring reasoning_format,
    jboolean embedding,
    jint embd_normalize,
    jint n_ctx,
    jint n_batch,
    jint n_ubatch,
    jint n_threads,
    jint n_gpu_layers, // TODO: Support this
    jboolean flash_attn,
    jstring cache_type_k,
    jstring cache_type_v,
    jboolean use_mlock,
    jboolean use_mmap,
    jboolean vocab_only,
    jstring lora_str,
    jfloat lora_scaled,
    jobject lora_list,
    jfloat rope_freq_base,
    jfloat rope_freq_scale,
    jint pooling_type,
    jobject load_progress_callback
) {
    UNUSED(thiz);

    common_params defaultParams;

    defaultParams.vocab_only = vocab_only;
    if(vocab_only) {
        defaultParams.warmup = false;
    }

    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);
    defaultParams.model.path = model_path_chars;

    const char *chat_template_chars = env->GetStringUTFChars(chat_template, nullptr);
    defaultParams.chat_template = chat_template_chars;

    const char *reasoning_format_chars = env->GetStringUTFChars(reasoning_format, nullptr);
    if (strcmp(reasoning_format_chars, "deepseek") == 0) {
        defaultParams.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    } else {
        defaultParams.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    }

    defaultParams.n_ctx = n_ctx;
    defaultParams.n_batch = n_batch;
    defaultParams.n_ubatch = n_ubatch;

    if (pooling_type != -1) {
        defaultParams.pooling_type = static_cast<enum llama_pooling_type>(pooling_type);
    }

    defaultParams.embedding = embedding;
    if (embd_normalize != -1) {
        defaultParams.embd_normalize = embd_normalize;
    }
    if (embedding) {
        // For non-causal models, batch size must be equal to ubatch size
        defaultParams.n_ubatch = defaultParams.n_batch;
    }

    int max_threads = std::thread::hardware_concurrency();
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    int default_n_threads = max_threads == 4 ? 2 : min(4, max_threads);
    defaultParams.cpuparams.n_threads = n_threads > 0 ? n_threads : default_n_threads;

    defaultParams.n_gpu_layers = n_gpu_layers;
    defaultParams.flash_attn = flash_attn;

    const char *cache_type_k_chars = env->GetStringUTFChars(cache_type_k, nullptr);
    const char *cache_type_v_chars = env->GetStringUTFChars(cache_type_v, nullptr);
    defaultParams.cache_type_k = cactus::kv_cache_type_from_str(cache_type_k_chars);
    defaultParams.cache_type_v = cactus::kv_cache_type_from_str(cache_type_v_chars);

    defaultParams.use_mlock = use_mlock;
    defaultParams.use_mmap = use_mmap;

    defaultParams.rope_freq_base = rope_freq_base;
    defaultParams.rope_freq_scale = rope_freq_scale;

    auto llama = new cactus::cactus_context();
    llama->is_load_interrupted = false;
    llama->loading_progress = 0;

    if (load_progress_callback != nullptr) {
        defaultParams.progress_callback = [](float progress, void * user_data) {
            callback_context *cb_ctx = (callback_context *)user_data;
            JNIEnv *env = cb_ctx->env;
            auto llama = cb_ctx->llama;
            jobject callback = cb_ctx->callback;
            int percentage = (int) (100 * progress);
            if (percentage > llama->loading_progress) {
                llama->loading_progress = percentage;
                jclass callback_class = env->GetObjectClass(callback);
                jmethodID onLoadProgress = env->GetMethodID(callback_class, "onLoadProgress", "(I)V");
                env->CallVoidMethod(callback, onLoadProgress, percentage);
            }
            return !llama->is_load_interrupted;
        };

        callback_context *cb_ctx = new callback_context;
        cb_ctx->env = env;
        cb_ctx->llama = llama;
        cb_ctx->callback = env->NewGlobalRef(load_progress_callback);
        defaultParams.progress_callback_user_data = cb_ctx;
    }

    bool is_model_loaded = llama->loadModel(defaultParams);

    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    env->ReleaseStringUTFChars(chat_template, chat_template_chars);
    env->ReleaseStringUTFChars(reasoning_format, reasoning_format_chars);
    env->ReleaseStringUTFChars(cache_type_k, cache_type_k_chars);
    env->ReleaseStringUTFChars(cache_type_v, cache_type_v_chars);

    LOGI("[CACTUS] is_model_loaded %s", (is_model_loaded ? "true" : "false"));
    if (is_model_loaded) {
        if (embedding && llama_model_has_encoder(llama->model) && llama_model_has_decoder(llama->model)) {
            LOGI("[CACTUS] computing embeddings in encoder-decoder models is not supported");
            llama_free(llama->ctx);
            return -1;
        }
        context_map[(long) llama->ctx] = llama;
    } else {
        llama_free(llama->ctx);
    }

    std::vector<common_adapter_lora_info> lora;
    const char *lora_chars = env->GetStringUTFChars(lora_str, nullptr);
    if (lora_chars != nullptr && lora_chars[0] != '\0') {
        common_adapter_lora_info la;
        la.path = lora_chars;
        la.scale = lora_scaled;
        lora.push_back(la);
    }

    if (lora_list != nullptr) {
        // lora_adapters: ReadableArray<ReadableMap>
        int lora_list_size = readablearray::size(env, lora_list);
        for (int i = 0; i < lora_list_size; i++) {
            jobject lora_adapter = readablearray::getMap(env, lora_list, i);
            jstring path = readablemap::getString(env, lora_adapter, "path", nullptr);
            if (path != nullptr) {
                const char *path_chars = env->GetStringUTFChars(path, nullptr);
                common_adapter_lora_info la;
                la.path = path_chars;
                la.scale = readablemap::getFloat(env, lora_adapter, "scaled", 1.0f);
                lora.push_back(la);
                env->ReleaseStringUTFChars(path, path_chars);
            }
        }
    }
    env->ReleaseStringUTFChars(lora_str, lora_chars);
    int result = llama->applyLoraAdapters(lora);
    if (result != 0) {
      LOGI("[Cactus] Failed to apply lora adapters");
      llama_free(llama->ctx);
      return -1;
    }

    return reinterpret_cast<jlong>(llama->ctx);
}


JNIEXPORT void JNICALL
Java_com_cactus_LlamaContext_interruptLoad(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (llama) {
        llama->is_load_interrupted = true;
    }
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_loadModelDetails(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    int count = llama_model_meta_count(llama->model);
    auto meta = createWriteableMap(env);
    for (int i = 0; i < count; i++) {
        char key[256];
        llama_model_meta_key_by_index(llama->model, i, key, sizeof(key));
        char val[4096];
        llama_model_meta_val_str_by_index(llama->model, i, val, sizeof(val));

        putString(env, meta, key, val);
    }

    auto result = createWriteableMap(env);

    char desc[1024];
    llama_model_desc(llama->model, desc, sizeof(desc));

    putString(env, result, "desc", desc);
    putDouble(env, result, "size", llama_model_size(llama->model));
    putDouble(env, result, "nEmbd", llama_model_n_embd(llama->model));
    putDouble(env, result, "nParams", llama_model_n_params(llama->model));
    auto chat_templates = createWriteableMap(env);
    putBoolean(env, chat_templates, "llamaChat", llama->validateModelChatTemplate(false, nullptr));

    auto minja = createWriteableMap(env);
    putBoolean(env, minja, "default", llama->validateModelChatTemplate(true, nullptr));

    auto default_caps = createWriteableMap(env);

    auto default_tmpl = llama->templates.get()->template_default.get();
    auto default_tmpl_caps = default_tmpl->original_caps();
    putBoolean(env, default_caps, "tools", default_tmpl_caps.supports_tools);
    putBoolean(env, default_caps, "toolCalls", default_tmpl_caps.supports_tool_calls);
    putBoolean(env, default_caps, "parallelToolCalls", default_tmpl_caps.supports_parallel_tool_calls);
    putBoolean(env, default_caps, "toolResponses", default_tmpl_caps.supports_tool_responses);
    putBoolean(env, default_caps, "systemRole", default_tmpl_caps.supports_system_role);
    putBoolean(env, default_caps, "toolCallId", default_tmpl_caps.supports_tool_call_id);
    putMap(env, minja, "defaultCaps", default_caps);

    putBoolean(env, minja, "toolUse", llama->validateModelChatTemplate(true, "tool_use"));
    auto tool_use_tmpl = llama->templates.get()->template_tool_use.get();
    if (tool_use_tmpl != nullptr) {
      auto tool_use_caps = createWriteableMap(env);
      auto tool_use_tmpl_caps = tool_use_tmpl->original_caps();
      putBoolean(env, tool_use_caps, "tools", tool_use_tmpl_caps.supports_tools);
      putBoolean(env, tool_use_caps, "toolCalls", tool_use_tmpl_caps.supports_tool_calls);
      putBoolean(env, tool_use_caps, "parallelToolCalls", tool_use_tmpl_caps.supports_parallel_tool_calls);
      putBoolean(env, tool_use_caps, "systemRole", tool_use_tmpl_caps.supports_system_role);
      putBoolean(env, tool_use_caps, "toolResponses", tool_use_tmpl_caps.supports_tool_responses);
      putBoolean(env, tool_use_caps, "toolCallId", tool_use_tmpl_caps.supports_tool_call_id);
      putMap(env, minja, "toolUseCaps", tool_use_caps);
    }

    putMap(env, chat_templates, "minja", minja);
    putMap(env, result, "metadata", meta);
    putMap(env, result, "chatTemplates", chat_templates);

    // deprecated
    putBoolean(env, result, "isChatTemplateSupported", llama->validateModelChatTemplate(false, nullptr));

    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_getFormattedChatWithJinja(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring messages,
    jstring chat_template,
    jstring json_schema,
    jstring tools,
    jboolean parallel_tool_calls,
    jstring tool_choice
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *messages_chars = env->GetStringUTFChars(messages, nullptr);
    const char *tmpl_chars = env->GetStringUTFChars(chat_template, nullptr);
    const char *json_schema_chars = env->GetStringUTFChars(json_schema, nullptr);
    const char *tools_chars = env->GetStringUTFChars(tools, nullptr);
    const char *tool_choice_chars = env->GetStringUTFChars(tool_choice, nullptr);

    auto result = createWriteableMap(env);
    try {
        auto formatted = llama->getFormattedChatWithJinja(
            messages_chars,
            tmpl_chars,
            json_schema_chars,
            tools_chars,
            parallel_tool_calls,
            tool_choice_chars
        );
        putString(env, result, "prompt", formatted.prompt.c_str());
        putInt(env, result, "chat_format", static_cast<int>(formatted.format));
        putString(env, result, "grammar", formatted.grammar.c_str());
        putBoolean(env, result, "grammar_lazy", formatted.grammar_lazy);
        auto grammar_triggers = createWritableArray(env);
        for (const auto &trigger : formatted.grammar_triggers) {
            auto trigger_map = createWriteableMap(env);
            putInt(env, trigger_map, "type", trigger.type);
            putString(env, trigger_map, "value", trigger.value.c_str());
            putInt(env, trigger_map, "token", trigger.token);
            pushMap(env, grammar_triggers, trigger_map);
        }
        putArray(env, result, "grammar_triggers", grammar_triggers);
        auto preserved_tokens = createWritableArray(env);
        for (const auto &token : formatted.preserved_tokens) {
            pushString(env, preserved_tokens, token.c_str());
        }
        putArray(env, result, "preserved_tokens", preserved_tokens);
        auto additional_stops = createWritableArray(env);
        for (const auto &stop : formatted.additional_stops) {
            pushString(env, additional_stops, stop.c_str());
        }
        putArray(env, result, "additional_stops", additional_stops);
    } catch (const std::runtime_error &e) {
        LOGI("[Cactus] Error: %s", e.what());
        putString(env, result, "_error", e.what());
    }
    env->ReleaseStringUTFChars(tools, tools_chars);
    env->ReleaseStringUTFChars(messages, messages_chars);
    env->ReleaseStringUTFChars(chat_template, tmpl_chars);
    env->ReleaseStringUTFChars(json_schema, json_schema_chars);
    env->ReleaseStringUTFChars(tool_choice, tool_choice_chars);
    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_getFormattedChat(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring messages,
    jstring chat_template
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *messages_chars = env->GetStringUTFChars(messages, nullptr);
    const char *tmpl_chars = env->GetStringUTFChars(chat_template, nullptr);

    std::string formatted_chat = llama->getFormattedChat(messages_chars, tmpl_chars);

    env->ReleaseStringUTFChars(messages, messages_chars);
    env->ReleaseStringUTFChars(chat_template, tmpl_chars);

    return env->NewStringUTF(formatted_chat.c_str());
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_loadSession(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring path
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    const char *path_chars = env->GetStringUTFChars(path, nullptr);

    auto result = createWriteableMap(env);
    size_t n_token_count_out = 0;
    llama->embd.resize(llama->params.n_ctx);
    if (!llama_state_load_file(llama->ctx, path_chars, llama->embd.data(), llama->embd.capacity(), &n_token_count_out)) {
      env->ReleaseStringUTFChars(path, path_chars);

      putString(env, result, "error", "Failed to load session");
      return reinterpret_cast<jobject>(result);
    }
    llama->embd.resize(n_token_count_out);
    env->ReleaseStringUTFChars(path, path_chars);

    const std::string text = cactus::tokens_to_str(llama->ctx, llama->embd.cbegin(), llama->embd.cend());
    putInt(env, result, "tokens_loaded", n_token_count_out);
    putString(env, result, "prompt", text.c_str());
    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jint JNICALL
Java_com_cactus_LlamaContext_saveSession(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring path,
    jint size
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *path_chars = env->GetStringUTFChars(path, nullptr);

    std::vector<llama_token> session_tokens = llama->embd;
    int default_size = session_tokens.size();
    int save_size = size > 0 && size <= default_size ? size : default_size;
    if (!llama_state_save_file(llama->ctx, path_chars, session_tokens.data(), save_size)) {
      env->ReleaseStringUTFChars(path, path_chars);
      return -1;
    }

    env->ReleaseStringUTFChars(path, path_chars);
    return session_tokens.size();
}

static inline jobject tokenProbsToMap(
  JNIEnv *env,
  cactus::cactus_context *llama,
  std::vector<cactus::completion_token_output> probs
) {
    auto result = createWritableArray(env);
    for (const auto &prob : probs) {
        auto probsForToken = createWritableArray(env);
        for (const auto &p : prob.probs) {
            std::string tokStr = cactus::tokens_to_output_formatted_string(llama->ctx, p.tok);
            auto probResult = createWriteableMap(env);
            putString(env, probResult, "tok_str", tokStr.c_str());
            putDouble(env, probResult, "prob", p.prob);
            pushMap(env, probsForToken, probResult);
        }
        std::string tokStr = cactus::tokens_to_output_formatted_string(llama->ctx, prob.tok);
        auto tokenResult = createWriteableMap(env);
        putString(env, tokenResult, "content", tokStr.c_str());
        putArray(env, tokenResult, "probs", probsForToken);
        pushMap(env, result, tokenResult);
    }
    return result;
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_doCompletion(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring prompt,
    jint chat_format,
    jstring grammar,
    jstring json_schema,
    jboolean grammar_lazy,
    jobject grammar_triggers,
    jobject preserved_tokens,
    jfloat temperature,
    jint n_threads,
    jint n_predict,
    jint n_probs,
    jint penalty_last_n,
    jfloat penalty_repeat,
    jfloat penalty_freq,
    jfloat penalty_present,
    jfloat mirostat,
    jfloat mirostat_tau,
    jfloat mirostat_eta,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat xtc_threshold,
    jfloat xtc_probability,
    jfloat typical_p,
    jint seed,
    jobjectArray stop,
    jboolean ignore_eos,
    jobjectArray logit_bias,
    jfloat   dry_multiplier,
    jfloat   dry_base,
    jint dry_allowed_length,
    jint dry_penalty_last_n,
    jfloat top_n_sigma,
    jobjectArray dry_sequence_breakers,
    jobject partial_completion_callback
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    llama->rewind();

    //llama_reset_timings(llama->ctx);

    auto prompt_chars = env->GetStringUTFChars(prompt, nullptr);
    llama->params.prompt = prompt_chars;
    llama->params.sampling.seed = (seed == -1) ? time(NULL) : seed;

    int max_threads = std::thread::hardware_concurrency();
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    int default_n_threads = max_threads == 4 ? 2 : min(4, max_threads);
    llama->params.cpuparams.n_threads = n_threads > 0 ? n_threads : default_n_threads;

    llama->params.n_predict = n_predict;
    llama->params.sampling.ignore_eos = ignore_eos;

    auto & sparams = llama->params.sampling;
    sparams.temp = temperature;
    sparams.penalty_last_n = penalty_last_n;
    sparams.penalty_repeat = penalty_repeat;
    sparams.penalty_freq = penalty_freq;
    sparams.penalty_present = penalty_present;
    sparams.mirostat = mirostat;
    sparams.mirostat_tau = mirostat_tau;
    sparams.mirostat_eta = mirostat_eta;
    sparams.top_k = top_k;
    sparams.top_p = top_p;
    sparams.min_p = min_p;
    sparams.typ_p = typical_p;
    sparams.n_probs = n_probs;
    sparams.xtc_threshold = xtc_threshold;
    sparams.xtc_probability = xtc_probability;
    sparams.dry_multiplier = dry_multiplier;
    sparams.dry_base = dry_base;
    sparams.dry_allowed_length = dry_allowed_length;
    sparams.dry_penalty_last_n = dry_penalty_last_n;
    sparams.top_n_sigma = top_n_sigma;

    // grammar
    auto grammar_chars = env->GetStringUTFChars(grammar, nullptr);
    if (grammar_chars && grammar_chars[0] != '\0') {
      sparams.grammar = grammar_chars;
    }
    sparams.grammar_lazy = grammar_lazy;

    if (preserved_tokens != nullptr) {
        int preserved_tokens_size = readablearray::size(env, preserved_tokens);
        for (int i = 0; i < preserved_tokens_size; i++) {
            jstring preserved_token = readablearray::getString(env, preserved_tokens, i);
            auto ids = common_tokenize(llama->ctx, env->GetStringUTFChars(preserved_token, nullptr), /* add_special= */ false, /* parse_special= */ true);
            if (ids.size() == 1) {
                sparams.preserved_tokens.insert(ids[0]);
            } else {
                LOGI("[Cactus] Not preserved because more than 1 token (wrong chat template override?): %s", env->GetStringUTFChars(preserved_token, nullptr));
            }
        }
    }

    if (grammar_triggers != nullptr) {
        int grammar_triggers_size = readablearray::size(env, grammar_triggers);
        for (int i = 0; i < grammar_triggers_size; i++) {
            auto trigger_map = readablearray::getMap(env, grammar_triggers, i);
            const auto type = static_cast<common_grammar_trigger_type>(readablemap::getInt(env, trigger_map, "type", 0));
            jstring trigger_word = readablemap::getString(env, trigger_map, "value", nullptr);
            auto word = env->GetStringUTFChars(trigger_word, nullptr);

            if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                auto ids = common_tokenize(llama->ctx, word, /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    auto token = ids[0];
                    if (std::find(sparams.preserved_tokens.begin(), sparams.preserved_tokens.end(), (llama_token) token) == sparams.preserved_tokens.end()) {
                        throw std::runtime_error("Grammar trigger word should be marked as preserved token");
                    }
                    common_grammar_trigger trigger;
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    trigger.value = word;
                    trigger.token = token;
                    sparams.grammar_triggers.push_back(std::move(trigger));
                } else {
                    sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                }
            } else {
                common_grammar_trigger trigger;
                trigger.type = type;
                trigger.value = word;
                if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                    const auto token = (llama_token) readablemap::getInt(env, trigger_map, "token", 0);
                    trigger.token = token;
                }
                sparams.grammar_triggers.push_back(std::move(trigger));
            }
        }
    }

    auto json_schema_chars = env->GetStringUTFChars(json_schema, nullptr);
    if ((!grammar_chars || grammar_chars[0] == '\0') && json_schema_chars && json_schema_chars[0] != '\0') {
        auto schema = json::parse(json_schema_chars);
        sparams.grammar = json_schema_to_grammar(schema);
    }
    env->ReleaseStringUTFChars(json_schema, json_schema_chars);


    const llama_model * model = llama_get_model(llama->ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    sparams.logit_bias.clear();
    if (ignore_eos) {
        sparams.logit_bias[llama_vocab_eos(vocab)].bias = -INFINITY;
    }

    // dry break seq

    jint size = env->GetArrayLength(dry_sequence_breakers);
    std::vector<std::string> dry_sequence_breakers_vector;

    for (jint i = 0; i < size; i++) {
        jstring javaString = (jstring)env->GetObjectArrayElement(dry_sequence_breakers, i);
        const char *nativeString = env->GetStringUTFChars(javaString, 0);
        dry_sequence_breakers_vector.push_back(std::string(nativeString));
        env->ReleaseStringUTFChars(javaString, nativeString);
        env->DeleteLocalRef(javaString);
    }

    sparams.dry_sequence_breakers = dry_sequence_breakers_vector;

    // logit bias
    const int n_vocab = llama_vocab_n_tokens(vocab);
    jsize logit_bias_len = env->GetArrayLength(logit_bias);

    for (jsize i = 0; i < logit_bias_len; i++) {
        jdoubleArray el = (jdoubleArray) env->GetObjectArrayElement(logit_bias, i);
        if (el && env->GetArrayLength(el) == 2) {
            jdouble* doubleArray = env->GetDoubleArrayElements(el, 0);

            llama_token tok = static_cast<llama_token>(doubleArray[0]);
            if (tok >= 0 && tok < n_vocab) {
                if (doubleArray[1] != 0) {  // If the second element is not false (0)
                    sparams.logit_bias[tok].bias = doubleArray[1];
                } else {
                    sparams.logit_bias[tok].bias = -INFINITY;
                }
            }

            env->ReleaseDoubleArrayElements(el, doubleArray, 0);
        }
        env->DeleteLocalRef(el);
    }

    llama->params.antiprompt.clear();
    int stop_len = env->GetArrayLength(stop);
    for (int i = 0; i < stop_len; i++) {
        jstring stop_str = (jstring) env->GetObjectArrayElement(stop, i);
        const char *stop_chars = env->GetStringUTFChars(stop_str, nullptr);
        llama->params.antiprompt.push_back(stop_chars);
        env->ReleaseStringUTFChars(stop_str, stop_chars);
    }

    if (!llama->initSampling()) {
        auto result = createWriteableMap(env);
        putString(env, result, "error", "Failed to initialize sampling");
        return reinterpret_cast<jobject>(result);
    }
    llama->beginCompletion();
    llama->loadPrompt();

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    while (llama->has_next_token && !llama->is_interrupted) {
        const cactus::completion_token_output token_with_probs = llama->doCompletion();
        if (token_with_probs.tok == -1 || llama->incomplete) {
            continue;
        }
        const std::string token_text = common_token_to_piece(llama->ctx, token_with_probs.tok);

        size_t pos = std::min(sent_count, llama->generated_text.size());

        const std::string str_test = llama->generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos =
            llama->findStoppingStrings(str_test, token_text.size(), cactus::STOP_FULL);
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            llama->generated_text.erase(
                llama->generated_text.begin() + pos + stop_pos,
                llama->generated_text.end());
            pos = std::min(sent_count, llama->generated_text.size());
        } else {
            is_stop_full = false;
            stop_pos = llama->findStoppingStrings(str_test, token_text.size(),
                cactus::STOP_PARTIAL);
        }

        if (
            stop_pos == std::string::npos ||
            // Send rest of the text if we are at the end of the generation
            (!llama->has_next_token && !is_stop_full && stop_pos > 0)
        ) {
            const std::string to_send = llama->generated_text.substr(pos, std::string::npos);

            sent_count += to_send.size();

            std::vector<cactus::completion_token_output> probs_output = {};

            auto tokenResult = createWriteableMap(env);
            putString(env, tokenResult, "token", to_send.c_str());

            if (llama->params.sampling.n_probs > 0) {
              const std::vector<llama_token> to_send_toks = common_tokenize(llama->ctx, to_send, false);
              size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
              size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
              if (probs_pos < probs_stop_pos) {
                  probs_output = std::vector<cactus::completion_token_output>(llama->generated_token_probs.begin() + probs_pos, llama->generated_token_probs.begin() + probs_stop_pos);
              }
              sent_token_probs_index = probs_stop_pos;

              putArray(env, tokenResult, "completion_probabilities", tokenProbsToMap(env, llama, probs_output));
            }

            jclass cb_class = env->GetObjectClass(partial_completion_callback);
            jmethodID onPartialCompletion = env->GetMethodID(cb_class, "onPartialCompletion", "(Lcom/facebook/react/bridge/WritableMap;)V");
            env->CallVoidMethod(partial_completion_callback, onPartialCompletion, tokenResult);
        }
    }

    env->ReleaseStringUTFChars(grammar, grammar_chars);
    env->ReleaseStringUTFChars(prompt, prompt_chars);
    llama_perf_context_print(llama->ctx);
    llama->is_predicting = false;

    auto toolCalls = createWritableArray(env);
    std::string reasoningContent = "";
    std::string content;
    auto toolCallsSize = 0;
    if (!llama->is_interrupted) {
        try {
            common_chat_msg message = common_chat_parse(llama->generated_text, static_cast<common_chat_format>(chat_format));
            if (!message.reasoning_content.empty()) {
                reasoningContent = message.reasoning_content;
            }
            content = message.content;
            for (const auto &tc : message.tool_calls) {
                auto toolCall = createWriteableMap(env);
                putString(env, toolCall, "type", "function");
                auto functionMap = createWriteableMap(env);
                putString(env, functionMap, "name", tc.name.c_str());
                putString(env, functionMap, "arguments", tc.arguments.c_str());
                putMap(env, toolCall, "function", functionMap);
                if (!tc.id.empty()) {
                    putString(env, toolCall, "id", tc.id.c_str());
                }
                pushMap(env, toolCalls, toolCall);
                toolCallsSize++;
            }
        } catch (const std::exception &e) {
            // LOGI("Error parsing tool calls: %s", e.what());
        }
    }

    auto result = createWriteableMap(env);
    putString(env, result, "text", llama->generated_text.c_str());
    if (!content.empty()) {
        putString(env, result, "content", content.c_str());
    }
    if (!reasoningContent.empty()) {
        putString(env, result, "reasoning_content", reasoningContent.c_str());
    }
    if (toolCallsSize > 0) {
        putArray(env, result, "tool_calls", toolCalls);
    }
    putArray(env, result, "completion_probabilities", tokenProbsToMap(env, llama, llama->generated_token_probs));
    putInt(env, result, "tokens_predicted", llama->num_tokens_predicted);
    putInt(env, result, "tokens_evaluated", llama->num_prompt_tokens);
    putInt(env, result, "truncated", llama->truncated);
    putInt(env, result, "stopped_eos", llama->stopped_eos);
    putInt(env, result, "stopped_word", llama->stopped_word);
    putInt(env, result, "stopped_limit", llama->stopped_limit);
    putString(env, result, "stopping_word", llama->stopping_word.c_str());
    putInt(env, result, "tokens_cached", llama->n_past);

    const auto timings_token = llama_perf_context(llama -> ctx);

    auto timingsResult = createWriteableMap(env);
    putInt(env, timingsResult, "prompt_n", timings_token.n_p_eval);
    putInt(env, timingsResult, "prompt_ms", timings_token.t_p_eval_ms);
    putInt(env, timingsResult, "prompt_per_token_ms", timings_token.t_p_eval_ms / timings_token.n_p_eval);
    putDouble(env, timingsResult, "prompt_per_second", 1e3 / timings_token.t_p_eval_ms * timings_token.n_p_eval);
    putInt(env, timingsResult, "predicted_n", timings_token.n_eval);
    putInt(env, timingsResult, "predicted_ms", timings_token.t_eval_ms);
    putInt(env, timingsResult, "predicted_per_token_ms", timings_token.t_eval_ms / timings_token.n_eval);
    putDouble(env, timingsResult, "predicted_per_second", 1e3 / timings_token.t_eval_ms * timings_token.n_eval);

    putMap(env, result, "timings", timingsResult);

    return reinterpret_cast<jobject>(result);
}

JNIEXPORT void JNICALL
Java_com_cactus_LlamaContext_stopCompletion(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    llama->is_interrupted = true;
}

JNIEXPORT jboolean JNICALL
Java_com_cactus_LlamaContext_isPredicting(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    return llama->is_predicting;
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_tokenize(
        JNIEnv *env, jobject thiz, jlong context_ptr, jstring text) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *text_chars = env->GetStringUTFChars(text, nullptr);

    const std::vector<llama_token> toks = common_tokenize(
        llama->ctx,
        text_chars,
        false
    );

    jobject result = createWritableArray(env);
    for (const auto &tok : toks) {
      pushInt(env, result, tok);
    }

    env->ReleaseStringUTFChars(text, text_chars);
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_LlamaContext_detokenize(
        JNIEnv *env, jobject thiz, jlong context_ptr, jintArray tokens) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    jsize tokens_len = env->GetArrayLength(tokens);
    jint *tokens_ptr = env->GetIntArrayElements(tokens, 0);
    std::vector<llama_token> toks;
    for (int i = 0; i < tokens_len; i++) {
        toks.push_back(tokens_ptr[i]);
    }

    auto text = cactus::tokens_to_str(llama->ctx, toks.cbegin(), toks.cend());

    env->ReleaseIntArrayElements(tokens, tokens_ptr, 0);

    return env->NewStringUTF(text.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_cactus_LlamaContext_isEmbeddingEnabled(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    return llama->params.embedding;
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_embedding(
        JNIEnv *env, jobject thiz,
        jlong context_ptr,
        jstring text,
        jint embd_normalize
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    common_params embdParams;
    embdParams.embedding = true;
    embdParams.embd_normalize = llama->params.embd_normalize;
    if (embd_normalize != -1) {
      embdParams.embd_normalize = embd_normalize;
    }

    const char *text_chars = env->GetStringUTFChars(text, nullptr);

    llama->rewind();

    llama_perf_context_reset(llama->ctx);

    llama->params.prompt = text_chars;

    llama->params.n_predict = 0;

    auto result = createWriteableMap(env);
    if (!llama->initSampling()) {
        putString(env, result, "error", "Failed to initialize sampling");
        return reinterpret_cast<jobject>(result);
    }

    llama->beginCompletion();
    llama->loadPrompt();
    llama->doCompletion();

    std::vector<float> embedding = llama->getEmbedding(embdParams);

    auto embeddings = createWritableArray(env);
    for (const auto &val : embedding) {
      pushDouble(env, embeddings, (double) val);
    }
    putArray(env, result, "embedding", embeddings);

    auto promptTokens = createWritableArray(env);
    for (const auto &tok : llama->embd) {
      pushString(env, promptTokens, common_token_to_piece(llama->ctx, tok).c_str());
    }
    putArray(env, result, "prompt_tokens", promptTokens);

    env->ReleaseStringUTFChars(text, text_chars);
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_LlamaContext_bench(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jint pp,
    jint tg,
    jint pl,
    jint nr
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    std::string result = llama->bench(pp, tg, pl, nr);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT jint JNICALL
Java_com_cactus_LlamaContext_applyLoraAdapters(
    JNIEnv *env, jobject thiz, jlong context_ptr, jobjectArray loraAdapters) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    // lora_adapters: ReadableArray<ReadableMap>
    std::vector<common_adapter_lora_info> lora_adapters;
    int lora_adapters_size = readablearray::size(env, loraAdapters);
    for (int i = 0; i < lora_adapters_size; i++) {
        jobject lora_adapter = readablearray::getMap(env, loraAdapters, i);
        jstring path = readablemap::getString(env, lora_adapter, "path", nullptr);
        if (path != nullptr) {
          const char *path_chars = env->GetStringUTFChars(path, nullptr);
          env->ReleaseStringUTFChars(path, path_chars);
          float scaled = readablemap::getFloat(env, lora_adapter, "scaled", 1.0f);
          common_adapter_lora_info la;
          la.path = path_chars;
          la.scale = scaled;
          lora_adapters.push_back(la);
        }
    }
    return llama->applyLoraAdapters(lora_adapters);
}

JNIEXPORT void JNICALL
Java_com_cactus_LlamaContext_removeLoraAdapters(
    JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    llama->removeLoraAdapters();
}

JNIEXPORT jobject JNICALL
Java_com_cactus_LlamaContext_getLoadedLoraAdapters(
    JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    auto loaded_lora_adapters = llama->getLoadedLoraAdapters();
    auto result = createWritableArray(env);
    for (common_adapter_lora_info &la : loaded_lora_adapters) {
        auto map = createWriteableMap(env);
        putString(env, map, "path", la.path.c_str());
        putDouble(env, map, "scaled", la.scale);
        pushMap(env, result, map);
    }
    return result;
}

JNIEXPORT void JNICALL
Java_com_cactus_LlamaContext_freeContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    context_map.erase((long) llama->ctx);
    delete llama;
}

struct log_callback_context {
    JavaVM *jvm;
    jobject callback;
};

static void cactus_log_callback_to_j(lm_ggml_log_level level, const char * text, void * data) {
    auto level_c = "";
    if (level == LM_GGML_LOG_LEVEL_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, text, nullptr);
        level_c = "error";
    } else if (level == LM_GGML_LOG_LEVEL_INFO) {
        __android_log_print(ANDROID_LOG_INFO, TAG, text, nullptr);
        level_c = "info";
    } else if (level == LM_GGML_LOG_LEVEL_WARN) {
        __android_log_print(ANDROID_LOG_WARN, TAG, text, nullptr);
        level_c = "warn";
    } else {
        __android_log_print(ANDROID_LOG_DEFAULT, TAG, text, nullptr);
    }

    log_callback_context *cb_ctx = (log_callback_context *) data;

    JNIEnv *env;
    bool need_detach = false;
    int getEnvResult = cb_ctx->jvm->GetEnv((void**)&env, JNI_VERSION_1_6);

    if (getEnvResult == JNI_EDETACHED) {
        if (cb_ctx->jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            need_detach = true;
        } else {
            return;
        }
    } else if (getEnvResult != JNI_OK) {
        return;
    }

    jobject callback = cb_ctx->callback;
    jclass cb_class = env->GetObjectClass(callback);
    jmethodID emitNativeLog = env->GetMethodID(cb_class, "emitNativeLog", "(Ljava/lang/String;Ljava/lang/String;)V");

    jstring level_str = env->NewStringUTF(level_c);
    jstring text_str = env->NewStringUTF(text);
    env->CallVoidMethod(callback, emitNativeLog, level_str, text_str);
    env->DeleteLocalRef(level_str);
    env->DeleteLocalRef(text_str);

    if (need_detach) {
        cb_ctx->jvm->DetachCurrentThread();
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_LlamaContext_setupLog(JNIEnv *env, jobject thiz, jobject logCallback) {
    UNUSED(thiz);

    log_callback_context *cb_ctx = new log_callback_context;

    JavaVM *jvm;
    env->GetJavaVM(&jvm);
    cb_ctx->jvm = jvm;
    cb_ctx->callback = env->NewGlobalRef(logCallback);

    llama_log_set(cactus_log_callback_to_j, cb_ctx);
}

JNIEXPORT void JNICALL
Java_com_cactus_LlamaContext_unsetLog(JNIEnv *env, jobject thiz) {
    UNUSED(env);
    UNUSED(thiz);
    llama_log_set(cactus_log_callback_default, NULL);
}

// STT JNI Methods

// Placeholder for STT context or similar structure if needed
// static cactus_stt_context* stt_ctx = nullptr; // Example

JNIEXPORT void JNICALL
Java_com_cactus_CactusModule_nativeInitSTT(
    JNIEnv *env,
    jobject thiz, /* this is a reference to the CactusModule instance */
    jstring model_path_str,
    jobject promise) {
    UNUSED(thiz);

    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);
    LOGI("nativeInitSTT called with modelPath: %s", model_path_chars);

    // TODO: Implement actual STT initialization using cactus C++ library
    // Example:
    // if (stt_ctx != nullptr) {
    //   // Clean up existing context if any
    //   // cactus_stt_free(stt_ctx);
    //   stt_ctx = nullptr;
    // }
    // stt_ctx = cactus_stt_init(model_path_chars);
    // if (stt_ctx == nullptr) {
    //   jniutils::rejectPromise(env, promise, "E_STT_INIT", "Failed to initialize STT model");
    //   env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    //   return;
    // }

    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    jniutils::resolvePromise(env, promise, NULL); // Resolve with null or some status map
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusModule_nativeProcessAudioFile(
    JNIEnv *env,
    jobject thiz,
    jstring file_path_str,
    jobject promise) {
    UNUSED(thiz);

    const char *file_path_chars = env->GetStringUTFChars(file_path_str, nullptr);
    LOGI("nativeProcessAudioFile called with filePath: %s", file_path_chars);

    // TODO: Implement actual audio file processing and STT
    // Example:
    // if (stt_ctx == nullptr) {
    //   jniutils::rejectPromise(env, promise, "E_STT_NO_CTX", "STT model not initialized");
    //   env->ReleaseStringUTFChars(file_path_str, file_path_chars);
    //   return;
    // }
    // const char* transcription = cactus_stt_process_file(stt_ctx, file_path_chars);
    // if (transcription == nullptr) {
    //   jniutils::rejectPromise(env, promise, "E_STT_PROCESS", "Failed to process audio file");
    //   env->ReleaseStringUTFChars(file_path_str, file_path_chars);
    //   return;
    // }
    // jobject result = createWriteableMap(env);
    // putString(env, result, "transcription", transcription);
    // jniutils::resolvePromise(env, promise, result);

    // Placeholder:
    jobject result = createWriteableMap(env);
    putString(env, result, "transcription", "Placeholder transcription");
    jniutils::resolvePromise(env, promise, result);


    env->ReleaseStringUTFChars(file_path_str, file_path_chars);
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusModule_nativeReleaseSTT(
    JNIEnv *env,
    jobject thiz,
    jobject promise) {
    UNUSED(thiz);

    LOGI("nativeReleaseSTT called");

    // TODO: Implement actual STT context release
    // Example:
    // if (stt_ctx != nullptr) {
    //   cactus_stt_free(stt_ctx);
    //   stt_ctx = nullptr;
    // }

    jniutils::resolvePromise(env, promise, NULL); // Resolve with null or status
}

} // extern "C"
