#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "cactus_ffi.h"

static constexpr size_t DEFAULT_BUFFER_SIZE = 65536;

static const char* jstring_to_cstr(JNIEnv* env, jstring str) {
    if (str == nullptr) return nullptr;
    return env->GetStringUTFChars(str, nullptr);
}

static void release_jstring(JNIEnv* env, jstring str, const char* cstr) {
    if (str != nullptr && cstr != nullptr) {
        env->ReleaseStringUTFChars(str, cstr);
    }
}

struct TokenCallbackContext {
    JNIEnv* env;
    jobject callback;
    jmethodID method;
};

static void token_callback_bridge(const char* token, uint32_t token_id, void* user_data) {
    if (!user_data) return;
    auto* ctx = static_cast<TokenCallbackContext*>(user_data);
    jstring jtoken = ctx->env->NewStringUTF(token);
    ctx->env->CallVoidMethod(ctx->callback, ctx->method, jtoken, static_cast<jint>(token_id));
    ctx->env->DeleteLocalRef(jtoken);
}

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeSetFramework(JNIEnv*, jobject) {
    cactus_set_telemetry_environment("kotlin", nullptr, nullptr);
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeSetCacheDir(JNIEnv* env, jobject, jstring cacheDir) {
    const char* dir = jstring_to_cstr(env, cacheDir);
    cactus_set_telemetry_environment(nullptr, dir, nullptr);
    release_jstring(env, cacheDir, dir);
}

JNIEXPORT jlong JNICALL
Java_com_cactus_Cactus_nativeInit(JNIEnv* env, jobject, jstring modelPath, jstring corpusDir) {
    const char* path = jstring_to_cstr(env, modelPath);
    const char* corpus = jstring_to_cstr(env, corpusDir);
    jlong handle = reinterpret_cast<jlong>(cactus_init(path, corpus, false));
    release_jstring(env, modelPath, path);
    release_jstring(env, corpusDir, corpus);
    return handle;
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) {
        cactus_destroy(reinterpret_cast<cactus_model_t>(handle));
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeReset(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) {
        cactus_reset(reinterpret_cast<cactus_model_t>(handle));
    }
}

JNIEXPORT void JNICALL
Java_com_cactus_Cactus_nativeStop(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) {
        cactus_stop(reinterpret_cast<cactus_model_t>(handle));
    }
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeComplete(JNIEnv* env, jobject, jlong handle,
                                       jstring messagesJson, jstring optionsJson,
                                       jstring toolsJson, jobject callback) {
    if (handle == 0) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    const char* messages = jstring_to_cstr(env, messagesJson);
    const char* options = jstring_to_cstr(env, optionsJson);
    const char* tools = jstring_to_cstr(env, toolsJson);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    TokenCallbackContext* ctx = nullptr;
    cactus_token_callback cb = nullptr;

    if (callback != nullptr) {
        jclass callbackClass = env->GetObjectClass(callback);
        jmethodID method = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;I)V");
        if (method != nullptr) {
            ctx = new TokenCallbackContext{env, callback, method};
            cb = token_callback_bridge;
        }
    }

    int result = cactus_complete(
        reinterpret_cast<cactus_model_t>(handle),
        messages,
        buffer.data(),
        buffer.size(),
        options,
        tools,
        cb,
        ctx
    );

    delete ctx;

    release_jstring(env, messagesJson, messages);
    release_jstring(env, optionsJson, options);
    release_jstring(env, toolsJson, tools);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeTranscribe(JNIEnv* env, jobject, jlong handle,
                                         jstring audioPath, jstring prompt,
                                         jstring optionsJson, jbyteArray pcmData) {
    if (handle == 0) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    const char* path = jstring_to_cstr(env, audioPath);
    const char* promptStr = jstring_to_cstr(env, prompt);
    const char* options = jstring_to_cstr(env, optionsJson);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    const uint8_t* pcmBuffer = nullptr;
    size_t pcmSize = 0;
    jbyte* pcmBytes = nullptr;

    if (pcmData != nullptr) {
        pcmSize = env->GetArrayLength(pcmData);
        pcmBytes = env->GetByteArrayElements(pcmData, nullptr);
        pcmBuffer = reinterpret_cast<const uint8_t*>(pcmBytes);
    }

    int result = cactus_transcribe(
        reinterpret_cast<cactus_model_t>(handle),
        path,
        promptStr,
        buffer.data(),
        buffer.size(),
        options,
        nullptr,
        nullptr,
        pcmBuffer,
        pcmSize
    );

    if (pcmBytes != nullptr) {
        env->ReleaseByteArrayElements(pcmData, pcmBytes, JNI_ABORT);
    }

    release_jstring(env, audioPath, path);
    release_jstring(env, prompt, promptStr);
    release_jstring(env, optionsJson, options);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jfloatArray JNICALL
Java_com_cactus_Cactus_nativeEmbed(JNIEnv* env, jobject, jlong handle,
                                    jstring text, jboolean normalize) {
    if (handle == 0) {
        return nullptr;
    }

    const char* textStr = jstring_to_cstr(env, text);

    std::vector<float> buffer(4096);
    size_t embeddingDim = 0;

    int result = cactus_embed(
        reinterpret_cast<cactus_model_t>(handle),
        textStr,
        buffer.data(),
        buffer.size(),
        &embeddingDim,
        normalize == JNI_TRUE
    );

    release_jstring(env, text, textStr);

    if (result < 0 || embeddingDim == 0) {
        return nullptr;
    }

    jfloatArray jarray = env->NewFloatArray(static_cast<jsize>(embeddingDim));
    env->SetFloatArrayRegion(jarray, 0, static_cast<jsize>(embeddingDim), buffer.data());

    return jarray;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeRagQuery(JNIEnv* env, jobject, jlong handle,
                                       jstring query, jint topK) {
    if (handle == 0) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    const char* queryStr = jstring_to_cstr(env, query);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    int result = cactus_rag_query(
        reinterpret_cast<cactus_model_t>(handle),
        queryStr,
        buffer.data(),
        buffer.size(),
        static_cast<size_t>(topK)
    );

    release_jstring(env, query, queryStr);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeGetLastError(JNIEnv* env, jobject) {
    const char* error = cactus_get_last_error();
    return env->NewStringUTF(error ? error : "");
}

JNIEXPORT jintArray JNICALL
Java_com_cactus_Cactus_nativeTokenize(JNIEnv* env, jobject, jlong handle, jstring text) {
    if (handle == 0) return nullptr;

    const char* textStr = jstring_to_cstr(env, text);
    std::vector<uint32_t> buffer(8192);
    size_t tokenLen = 0;

    int result = cactus_tokenize(
        reinterpret_cast<cactus_model_t>(handle),
        textStr,
        buffer.data(),
        buffer.size(),
        &tokenLen
    );

    release_jstring(env, text, textStr);

    if (result < 0 || tokenLen == 0) return nullptr;

    jintArray jarray = env->NewIntArray(static_cast<jsize>(tokenLen));
    env->SetIntArrayRegion(jarray, 0, static_cast<jsize>(tokenLen), reinterpret_cast<jint*>(buffer.data()));
    return jarray;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeScoreWindow(JNIEnv* env, jobject, jlong handle,
                                          jintArray tokens, jint start, jint end, jint context) {
    if (handle == 0) return env->NewStringUTF("{\"error\":\"Model not initialized\"}");

    jsize tokenLen = env->GetArrayLength(tokens);
    jint* tokenData = env->GetIntArrayElements(tokens, nullptr);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    int result = cactus_score_window(
        reinterpret_cast<cactus_model_t>(handle),
        reinterpret_cast<uint32_t*>(tokenData),
        static_cast<size_t>(tokenLen),
        static_cast<size_t>(start),
        static_cast<size_t>(end),
        static_cast<size_t>(context),
        buffer.data(),
        buffer.size()
    );

    env->ReleaseIntArrayElements(tokens, tokenData, JNI_ABORT);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jlong JNICALL
Java_com_cactus_Cactus_nativeStreamTranscribeInit(JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return 0;
    return reinterpret_cast<jlong>(cactus_stream_transcribe_start(reinterpret_cast<cactus_model_t>(handle), nullptr));
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeStreamTranscribeProcess(JNIEnv* env, jobject, jlong streamHandle, jbyteArray pcmData) {
    if (streamHandle == 0) return env->NewStringUTF("{\"error\":\"Stream not initialized\"}");

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);
    int result;

    if (pcmData != nullptr) {
        jsize pcmSize = env->GetArrayLength(pcmData);
        jbyte* pcmBytes = env->GetByteArrayElements(pcmData, nullptr);

        result = cactus_stream_transcribe_process(
            reinterpret_cast<cactus_stream_transcribe_t>(streamHandle),
            reinterpret_cast<const uint8_t*>(pcmBytes),
            static_cast<size_t>(pcmSize),
            buffer.data(),
            buffer.size()
        );

        env->ReleaseByteArrayElements(pcmData, pcmBytes, JNI_ABORT);
    } else {
        result = cactus_stream_transcribe_process(
            reinterpret_cast<cactus_stream_transcribe_t>(streamHandle),
            nullptr,
            0,
            buffer.data(),
            buffer.size()
        );
    }

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeStreamTranscribeStop(JNIEnv* env, jobject, jlong streamHandle) {
    if (streamHandle == 0) return env->NewStringUTF("{\"error\":\"Stream not initialized\"}");

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    int result = cactus_stream_transcribe_stop(
        reinterpret_cast<cactus_stream_transcribe_t>(streamHandle),
        buffer.data(),
        buffer.size()
    );

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jfloatArray JNICALL
Java_com_cactus_Cactus_nativeImageEmbed(JNIEnv* env, jobject, jlong handle, jstring imagePath) {
    if (handle == 0) return nullptr;

    const char* path = jstring_to_cstr(env, imagePath);
    std::vector<float> buffer(4096);
    size_t embeddingDim = 0;

    int result = cactus_image_embed(
        reinterpret_cast<cactus_model_t>(handle),
        path,
        buffer.data(),
        buffer.size(),
        &embeddingDim
    );

    release_jstring(env, imagePath, path);

    if (result < 0 || embeddingDim == 0) return nullptr;

    jfloatArray jarray = env->NewFloatArray(static_cast<jsize>(embeddingDim));
    env->SetFloatArrayRegion(jarray, 0, static_cast<jsize>(embeddingDim), buffer.data());
    return jarray;
}

JNIEXPORT jfloatArray JNICALL
Java_com_cactus_Cactus_nativeAudioEmbed(JNIEnv* env, jobject, jlong handle, jstring audioPath) {
    if (handle == 0) return nullptr;

    const char* path = jstring_to_cstr(env, audioPath);
    std::vector<float> buffer(4096);
    size_t embeddingDim = 0;

    int result = cactus_audio_embed(
        reinterpret_cast<cactus_model_t>(handle),
        path,
        buffer.data(),
        buffer.size(),
        &embeddingDim
    );

    release_jstring(env, audioPath, path);

    if (result < 0 || embeddingDim == 0) return nullptr;

    jfloatArray jarray = env->NewFloatArray(static_cast<jsize>(embeddingDim));
    env->SetFloatArrayRegion(jarray, 0, static_cast<jsize>(embeddingDim), buffer.data());
    return jarray;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_Cactus_nativeVad(JNIEnv* env, jobject, jlong handle,
                                  jstring audioPath, jstring optionsJson, jbyteArray pcmData) {
    if (handle == 0) {
        return env->NewStringUTF("{\"error\":\"Model not initialized\"}");
    }

    const char* path = jstring_to_cstr(env, audioPath);
    const char* options = jstring_to_cstr(env, optionsJson);

    std::vector<char> buffer(DEFAULT_BUFFER_SIZE);

    const uint8_t* pcmBuffer = nullptr;
    size_t pcmSize = 0;
    jbyte* pcmBytes = nullptr;

    if (pcmData != nullptr) {
        pcmSize = env->GetArrayLength(pcmData);
        pcmBytes = env->GetByteArrayElements(pcmData, nullptr);
        pcmBuffer = reinterpret_cast<const uint8_t*>(pcmBytes);
    }

    int result = cactus_vad(
        reinterpret_cast<cactus_model_t>(handle),
        path,
        buffer.data(),
        buffer.size(),
        options,
        pcmBuffer,
        pcmSize
    );

    if (pcmBytes != nullptr) {
        env->ReleaseByteArrayElements(pcmData, pcmBytes, JNI_ABORT);
    }

    release_jstring(env, audioPath, path);
    release_jstring(env, optionsJson, options);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    return env->NewStringUTF(buffer.data());
}

JNIEXPORT jlong JNICALL
Java_com_cactus_CactusIndex_nativeIndexInit(JNIEnv* env, jobject, jstring indexDir, jint embeddingDim) {
    const char* dir = jstring_to_cstr(env, indexDir);
    jlong handle = reinterpret_cast<jlong>(cactus_index_init(dir, static_cast<size_t>(embeddingDim)));
    release_jstring(env, indexDir, dir);
    return handle;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusIndex_nativeIndexAdd(JNIEnv* env, jobject, jlong handle,
                                            jintArray ids, jobjectArray documents,
                                            jobjectArray metadatas, jobjectArray embeddings,
                                            jint embeddingDim) {
    if (handle == 0) return -1;

    jsize count = env->GetArrayLength(ids);
    jint* idData = env->GetIntArrayElements(ids, nullptr);

    std::vector<const char*> docPtrs(count);
    std::vector<const char*> metaPtrs(count);
    std::vector<const float*> embPtrs(count);
    std::vector<jstring> docStrings(count);
    std::vector<jstring> metaStrings(count);
    std::vector<jfloatArray> embArrays(count);

    for (jsize i = 0; i < count; i++) {
        docStrings[i] = static_cast<jstring>(env->GetObjectArrayElement(documents, i));
        docPtrs[i] = jstring_to_cstr(env, docStrings[i]);

        if (metadatas != nullptr) {
            metaStrings[i] = static_cast<jstring>(env->GetObjectArrayElement(metadatas, i));
            metaPtrs[i] = jstring_to_cstr(env, metaStrings[i]);
        } else {
            metaPtrs[i] = nullptr;
        }

        embArrays[i] = static_cast<jfloatArray>(env->GetObjectArrayElement(embeddings, i));
        embPtrs[i] = env->GetFloatArrayElements(embArrays[i], nullptr);
    }

    int result = cactus_index_add(
        reinterpret_cast<cactus_index_t>(handle),
        reinterpret_cast<const int*>(idData),
        docPtrs.data(),
        metadatas != nullptr ? metaPtrs.data() : nullptr,
        embPtrs.data(),
        static_cast<size_t>(count),
        static_cast<size_t>(embeddingDim)
    );

    for (jsize i = 0; i < count; i++) {
        release_jstring(env, docStrings[i], docPtrs[i]);
        if (metadatas != nullptr) {
            release_jstring(env, metaStrings[i], metaPtrs[i]);
        }
        env->ReleaseFloatArrayElements(embArrays[i], const_cast<jfloat*>(embPtrs[i]), JNI_ABORT);
    }

    env->ReleaseIntArrayElements(ids, idData, JNI_ABORT);
    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusIndex_nativeIndexDelete(JNIEnv* env, jobject, jlong handle, jintArray ids) {
    if (handle == 0) return -1;

    jsize count = env->GetArrayLength(ids);
    jint* idData = env->GetIntArrayElements(ids, nullptr);

    int result = cactus_index_delete(
        reinterpret_cast<cactus_index_t>(handle),
        reinterpret_cast<const int*>(idData),
        static_cast<size_t>(count)
    );

    env->ReleaseIntArrayElements(ids, idData, JNI_ABORT);
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_cactus_CactusIndex_nativeIndexQuery(JNIEnv* env, jobject, jlong handle,
                                              jfloatArray embedding, jint topK, jstring optionsJson) {
    if (handle == 0) return env->NewStringUTF("{\"error\":\"Index not initialized\"}");

    jsize embDim = env->GetArrayLength(embedding);
    jfloat* embData = env->GetFloatArrayElements(embedding, nullptr);
    const char* options = jstring_to_cstr(env, optionsJson);

    std::vector<int> idBuffer(topK);
    std::vector<float> scoreBuffer(topK);
    size_t idBufferSize = topK;
    size_t scoreBufferSize = topK;

    const float* embPtr = embData;
    int* idPtr = idBuffer.data();
    float* scorePtr = scoreBuffer.data();

    int result = cactus_index_query(
        reinterpret_cast<cactus_index_t>(handle),
        &embPtr,
        1,
        static_cast<size_t>(embDim),
        options,
        &idPtr,
        &idBufferSize,
        &scorePtr,
        &scoreBufferSize
    );

    env->ReleaseFloatArrayElements(embedding, embData, JNI_ABORT);
    release_jstring(env, optionsJson, options);

    if (result < 0) {
        const char* error = cactus_get_last_error();
        std::string errorJson = "{\"error\":\"" + std::string(error ? error : "Unknown error") + "\"}";
        return env->NewStringUTF(errorJson.c_str());
    }

    std::string json = "{\"results\":[";
    for (size_t i = 0; i < idBufferSize; i++) {
        if (i > 0) json += ",";
        json += "{\"id\":" + std::to_string(idBuffer[i]) + ",\"score\":" + std::to_string(scoreBuffer[i]) + "}";
    }
    json += "]}";

    return env->NewStringUTF(json.c_str());
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusIndex_nativeIndexCompact(JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return -1;
    return cactus_index_compact(reinterpret_cast<cactus_index_t>(handle));
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusIndex_nativeIndexDestroy(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) {
        cactus_index_destroy(reinterpret_cast<cactus_index_t>(handle));
    }
}

}
