#include <jni.h>
#include "cactus.h"

struct TokenCallbackContext {
    JavaVM* jvm;
    jobject callback;
    jmethodID method;
};

struct LogCallbackContext {
    JavaVM* jvm;
    jobject callback;
    jmethodID method;
};

static LogCallbackContext* g_log_callback_ctx = nullptr;

extern "C" {

static void token_callback_bridge(const char* token, uint32_t token_id, void* user_data) {
    auto* ctx = static_cast<TokenCallbackContext*>(user_data);
    JNIEnv* env = nullptr;
    bool attached = false;
    jint status = ctx->jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        ctx->jvm->AttachCurrentThread(&env, nullptr);
        attached = true;
    }
    jstring jtoken = env->NewStringUTF(token);
    env->CallVoidMethod(ctx->callback, ctx->method, jtoken, static_cast<jint>(token_id));
    env->DeleteLocalRef(jtoken);
    if (attached) ctx->jvm->DetachCurrentThread();
}

static void log_callback_bridge(int level, const char* component, const char* message, void* user_data) {
    auto* ctx = static_cast<LogCallbackContext*>(user_data);
    JNIEnv* env = nullptr;
    bool attached = false;
    jint status = ctx->jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        ctx->jvm->AttachCurrentThread(&env, nullptr);
        attached = true;
    }
    jstring jcomponent = env->NewStringUTF(component);
    jstring jmessage = env->NewStringUTF(message);
    env->CallVoidMethod(ctx->callback, ctx->method, static_cast<jint>(level), jcomponent, jmessage);
    env->DeleteLocalRef(jcomponent);
    env->DeleteLocalRef(jmessage);
    if (attached) ctx->jvm->DetachCurrentThread();
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL
Java_com_cactus_CactusJNI_nativeInit(JNIEnv* env, jobject, jstring modelPath, jstring corpusDir, jboolean cacheIndex) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    const char* corpus = corpusDir ? env->GetStringUTFChars(corpusDir, nullptr) : nullptr;
    jlong handle = reinterpret_cast<jlong>(cactus_init(path, corpus, cacheIndex == JNI_TRUE));
    env->ReleaseStringUTFChars(modelPath, path);
    if (corpus) env->ReleaseStringUTFChars(corpusDir, corpus);
    return handle;
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    cactus_destroy(reinterpret_cast<cactus_model_t>(handle));
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeReset(JNIEnv*, jobject, jlong handle) {
    cactus_reset(reinterpret_cast<cactus_model_t>(handle));
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeStop(JNIEnv*, jobject, jlong handle) {
    cactus_stop(reinterpret_cast<cactus_model_t>(handle));
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeComplete(JNIEnv* env, jobject, jlong handle,
                                          jstring messagesJson, jbyteArray responseBuffer,
                                          jstring optionsJson, jstring toolsJson,
                                          jobject callback, jbyteArray pcmData) {
    const char* messages = env->GetStringUTFChars(messagesJson, nullptr);
    const char* options = optionsJson ? env->GetStringUTFChars(optionsJson, nullptr) : nullptr;
    const char* tools = toolsJson ? env->GetStringUTFChars(toolsJson, nullptr) : nullptr;

    jsize bufSize = env->GetArrayLength(responseBuffer);
    jbyte* buf = env->GetByteArrayElements(responseBuffer, nullptr);

    TokenCallbackContext* ctx = nullptr;
    cactus_token_callback cb = nullptr;
    if (callback) {
        JavaVM* jvm = nullptr;
        env->GetJavaVM(&jvm);
        jclass cls = env->GetObjectClass(callback);
        jmethodID method = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;I)V");
        ctx = new TokenCallbackContext{jvm, env->NewGlobalRef(callback), method};
        cb = token_callback_bridge;
    }

    jbyte* pcmBytes = nullptr;
    size_t pcmSize = 0;
    if (pcmData) {
        pcmSize = env->GetArrayLength(pcmData);
        pcmBytes = env->GetByteArrayElements(pcmData, nullptr);
    }

    int result = cactus_complete(
        reinterpret_cast<cactus_model_t>(handle),
        messages, reinterpret_cast<char*>(buf), static_cast<size_t>(bufSize),
        options, tools, cb, ctx,
        reinterpret_cast<const uint8_t*>(pcmBytes), pcmSize
    );

    if (ctx) { env->DeleteGlobalRef(ctx->callback); delete ctx; }
    env->ReleaseByteArrayElements(responseBuffer, buf, 0);
    if (pcmBytes) env->ReleaseByteArrayElements(pcmData, pcmBytes, JNI_ABORT);
    env->ReleaseStringUTFChars(messagesJson, messages);
    if (options) env->ReleaseStringUTFChars(optionsJson, options);
    if (tools) env->ReleaseStringUTFChars(toolsJson, tools);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativePrefill(JNIEnv* env, jobject, jlong handle,
                                         jstring messagesJson, jbyteArray responseBuffer,
                                         jstring optionsJson, jstring toolsJson,
                                         jbyteArray pcmData) {
    const char* messages = env->GetStringUTFChars(messagesJson, nullptr);
    const char* options = optionsJson ? env->GetStringUTFChars(optionsJson, nullptr) : nullptr;
    const char* tools = toolsJson ? env->GetStringUTFChars(toolsJson, nullptr) : nullptr;

    jsize bufSize = env->GetArrayLength(responseBuffer);
    jbyte* buf = env->GetByteArrayElements(responseBuffer, nullptr);

    jbyte* pcmBytes = nullptr;
    size_t pcmSize = 0;
    if (pcmData) {
        pcmSize = env->GetArrayLength(pcmData);
        pcmBytes = env->GetByteArrayElements(pcmData, nullptr);
    }

    int result = cactus_prefill(
        reinterpret_cast<cactus_model_t>(handle),
        messages, reinterpret_cast<char*>(buf), static_cast<size_t>(bufSize),
        options, tools,
        reinterpret_cast<const uint8_t*>(pcmBytes), pcmSize
    );

    env->ReleaseByteArrayElements(responseBuffer, buf, 0);
    if (pcmBytes) env->ReleaseByteArrayElements(pcmData, pcmBytes, JNI_ABORT);
    env->ReleaseStringUTFChars(messagesJson, messages);
    if (options) env->ReleaseStringUTFChars(optionsJson, options);
    if (tools) env->ReleaseStringUTFChars(toolsJson, tools);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeTokenize(JNIEnv* env, jobject, jlong handle,
                                          jstring text, jintArray tokenBuffer, jlongArray outTokenLen) {
    const char* textStr = env->GetStringUTFChars(text, nullptr);
    jsize bufLen = env->GetArrayLength(tokenBuffer);
    jint* tokens = env->GetIntArrayElements(tokenBuffer, nullptr);
    size_t tokenLen = 0;

    int result = cactus_tokenize(
        reinterpret_cast<cactus_model_t>(handle),
        textStr,
        reinterpret_cast<uint32_t*>(tokens),
        static_cast<size_t>(bufLen),
        &tokenLen
    );

    env->ReleaseIntArrayElements(tokenBuffer, tokens, 0);
    env->ReleaseStringUTFChars(text, textStr);

    jlong len = static_cast<jlong>(tokenLen);
    env->SetLongArrayRegion(outTokenLen, 0, 1, &len);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeScoreWindow(JNIEnv* env, jobject, jlong handle,
                                             jintArray tokens, jlong start, jlong end,
                                             jlong context, jbyteArray responseBuffer) {
    jsize tokenLen = env->GetArrayLength(tokens);
    jint* tokenData = env->GetIntArrayElements(tokens, nullptr);
    jsize bufSize = env->GetArrayLength(responseBuffer);
    jbyte* buf = env->GetByteArrayElements(responseBuffer, nullptr);

    int result = cactus_score_window(
        reinterpret_cast<cactus_model_t>(handle),
        reinterpret_cast<uint32_t*>(tokenData),
        static_cast<size_t>(tokenLen),
        static_cast<size_t>(start),
        static_cast<size_t>(end),
        static_cast<size_t>(context),
        reinterpret_cast<char*>(buf),
        static_cast<size_t>(bufSize)
    );

    env->ReleaseIntArrayElements(tokens, tokenData, JNI_ABORT);
    env->ReleaseByteArrayElements(responseBuffer, buf, 0);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeTranscribe(JNIEnv* env, jobject, jlong handle,
                                            jstring audioPath, jstring prompt,
                                            jbyteArray responseBuffer,
                                            jstring optionsJson, jobject callback,
                                            jbyteArray pcmData) {
    const char* path = audioPath ? env->GetStringUTFChars(audioPath, nullptr) : nullptr;
    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    const char* options = optionsJson ? env->GetStringUTFChars(optionsJson, nullptr) : nullptr;

    jsize bufSize = env->GetArrayLength(responseBuffer);
    jbyte* buf = env->GetByteArrayElements(responseBuffer, nullptr);

    TokenCallbackContext* ctx = nullptr;
    cactus_token_callback cb = nullptr;
    if (callback) {
        JavaVM* jvm = nullptr;
        env->GetJavaVM(&jvm);
        jclass cls = env->GetObjectClass(callback);
        jmethodID method = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;I)V");
        ctx = new TokenCallbackContext{jvm, env->NewGlobalRef(callback), method};
        cb = token_callback_bridge;
    }

    jbyte* pcmBytes = nullptr;
    size_t pcmSize = 0;
    if (pcmData) {
        pcmSize = env->GetArrayLength(pcmData);
        pcmBytes = env->GetByteArrayElements(pcmData, nullptr);
    }

    int result = cactus_transcribe(
        reinterpret_cast<cactus_model_t>(handle),
        path, promptStr,
        reinterpret_cast<char*>(buf), static_cast<size_t>(bufSize),
        options, cb, ctx,
        reinterpret_cast<const uint8_t*>(pcmBytes), pcmSize
    );

    if (ctx) { env->DeleteGlobalRef(ctx->callback); delete ctx; }
    env->ReleaseByteArrayElements(responseBuffer, buf, 0);
    if (pcmBytes) env->ReleaseByteArrayElements(pcmData, pcmBytes, JNI_ABORT);
    if (path) env->ReleaseStringUTFChars(audioPath, path);
    env->ReleaseStringUTFChars(prompt, promptStr);
    if (options) env->ReleaseStringUTFChars(optionsJson, options);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeEmbed(JNIEnv* env, jobject, jlong handle,
                                       jstring text, jfloatArray embeddingsBuffer,
                                       jlongArray outEmbeddingDim, jboolean normalize) {
    const char* textStr = env->GetStringUTFChars(text, nullptr);
    jsize bufSize = env->GetArrayLength(embeddingsBuffer);
    jfloat* buf = env->GetFloatArrayElements(embeddingsBuffer, nullptr);
    size_t embeddingDim = 0;

    int result = cactus_embed(
        reinterpret_cast<cactus_model_t>(handle),
        textStr, buf, static_cast<size_t>(bufSize),
        &embeddingDim, normalize == JNI_TRUE
    );

    env->ReleaseFloatArrayElements(embeddingsBuffer, buf, 0);
    env->ReleaseStringUTFChars(text, textStr);

    jlong dim = static_cast<jlong>(embeddingDim);
    env->SetLongArrayRegion(outEmbeddingDim, 0, 1, &dim);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeImageEmbed(JNIEnv* env, jobject, jlong handle,
                                            jstring imagePath, jfloatArray embeddingsBuffer,
                                            jlongArray outEmbeddingDim) {
    const char* path = env->GetStringUTFChars(imagePath, nullptr);
    jsize bufSize = env->GetArrayLength(embeddingsBuffer);
    jfloat* buf = env->GetFloatArrayElements(embeddingsBuffer, nullptr);
    size_t embeddingDim = 0;

    int result = cactus_image_embed(
        reinterpret_cast<cactus_model_t>(handle),
        path, buf, static_cast<size_t>(bufSize), &embeddingDim
    );

    env->ReleaseFloatArrayElements(embeddingsBuffer, buf, 0);
    env->ReleaseStringUTFChars(imagePath, path);

    jlong dim = static_cast<jlong>(embeddingDim);
    env->SetLongArrayRegion(outEmbeddingDim, 0, 1, &dim);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeAudioEmbed(JNIEnv* env, jobject, jlong handle,
                                            jstring audioPath, jfloatArray embeddingsBuffer,
                                            jlongArray outEmbeddingDim) {
    const char* path = env->GetStringUTFChars(audioPath, nullptr);
    jsize bufSize = env->GetArrayLength(embeddingsBuffer);
    jfloat* buf = env->GetFloatArrayElements(embeddingsBuffer, nullptr);
    size_t embeddingDim = 0;

    int result = cactus_audio_embed(
        reinterpret_cast<cactus_model_t>(handle),
        path, buf, static_cast<size_t>(bufSize), &embeddingDim
    );

    env->ReleaseFloatArrayElements(embeddingsBuffer, buf, 0);
    env->ReleaseStringUTFChars(audioPath, path);

    jlong dim = static_cast<jlong>(embeddingDim);
    env->SetLongArrayRegion(outEmbeddingDim, 0, 1, &dim);

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeRagQuery(JNIEnv* env, jobject, jlong handle,
                                          jstring query, jbyteArray responseBuffer, jlong topK) {
    const char* queryStr = env->GetStringUTFChars(query, nullptr);
    jsize bufSize = env->GetArrayLength(responseBuffer);
    jbyte* buf = env->GetByteArrayElements(responseBuffer, nullptr);

    int result = cactus_rag_query(
        reinterpret_cast<cactus_model_t>(handle),
        queryStr, reinterpret_cast<char*>(buf), static_cast<size_t>(bufSize),
        static_cast<size_t>(topK)
    );

    env->ReleaseByteArrayElements(responseBuffer, buf, 0);
    env->ReleaseStringUTFChars(query, queryStr);

    return result;
}

JNIEXPORT jlong JNICALL
Java_com_cactus_CactusJNI_nativeIndexInit(JNIEnv* env, jobject, jstring indexDir, jlong embeddingDim) {
    const char* dir = env->GetStringUTFChars(indexDir, nullptr);
    jlong handle = reinterpret_cast<jlong>(cactus_index_init(dir, static_cast<size_t>(embeddingDim)));
    env->ReleaseStringUTFChars(indexDir, dir);
    return handle;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeIndexAdd(JNIEnv* env, jobject, jlong handle,
                                          jintArray ids, jobjectArray documents,
                                          jobjectArray metadatas, jobjectArray embeddings,
                                          jlong embeddingDim) {
    jsize count = env->GetArrayLength(ids);
    jint* idData = env->GetIntArrayElements(ids, nullptr);

    const char** docPtrs = new const char*[count];
    const char** metaPtrs = new const char*[count];
    const float** embPtrs = new const float*[count];
    jstring* docStrings = new jstring[count];
    jstring* metaStrings = new jstring[count];
    jfloatArray* embArrays = new jfloatArray[count];

    for (jsize i = 0; i < count; i++) {
        docStrings[i] = static_cast<jstring>(env->GetObjectArrayElement(documents, i));
        docPtrs[i] = env->GetStringUTFChars(docStrings[i], nullptr);

        if (metadatas) {
            metaStrings[i] = static_cast<jstring>(env->GetObjectArrayElement(metadatas, i));
            metaPtrs[i] = env->GetStringUTFChars(metaStrings[i], nullptr);
        } else {
            metaStrings[i] = nullptr;
            metaPtrs[i] = nullptr;
        }

        embArrays[i] = static_cast<jfloatArray>(env->GetObjectArrayElement(embeddings, i));
        embPtrs[i] = env->GetFloatArrayElements(embArrays[i], nullptr);
    }

    int result = cactus_index_add(
        reinterpret_cast<cactus_index_t>(handle),
        reinterpret_cast<const int*>(idData),
        docPtrs, metadatas ? metaPtrs : nullptr, embPtrs,
        static_cast<size_t>(count), static_cast<size_t>(embeddingDim)
    );

    for (jsize i = 0; i < count; i++) {
        env->ReleaseStringUTFChars(docStrings[i], docPtrs[i]);
        env->DeleteLocalRef(docStrings[i]);
        if (metaStrings[i]) { env->ReleaseStringUTFChars(metaStrings[i], metaPtrs[i]); env->DeleteLocalRef(metaStrings[i]); }
        env->ReleaseFloatArrayElements(embArrays[i], const_cast<jfloat*>(embPtrs[i]), JNI_ABORT);
        env->DeleteLocalRef(embArrays[i]);
    }

    env->ReleaseIntArrayElements(ids, idData, JNI_ABORT);
    delete[] docPtrs;
    delete[] metaPtrs;
    delete[] embPtrs;
    delete[] docStrings;
    delete[] metaStrings;
    delete[] embArrays;

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeIndexDelete(JNIEnv* env, jobject, jlong handle, jintArray ids) {
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

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeIndexGet(JNIEnv* env, jobject, jlong handle,
                                          jintArray ids,
                                          jobjectArray documentBuffers, jlongArray documentBufferSizes,
                                          jobjectArray metadataBuffers, jlongArray metadataBufferSizes,
                                          jobjectArray embeddingBuffers, jlongArray embeddingBufferSizes) {
    jsize count = env->GetArrayLength(ids);
    jint* idData = env->GetIntArrayElements(ids, nullptr);

    char** docBufs = new char*[count];
    size_t* docSizes = new size_t[count];
    char** metaBufs = new char*[count];
    size_t* metaSizes = new size_t[count];
    float** embBufs = new float*[count];
    size_t* embSizes = new size_t[count];

    jbyteArray* docArrays = new jbyteArray[count];
    jbyteArray* metaArrays = new jbyteArray[count];
    jfloatArray* embArrays = new jfloatArray[count];

    jlong* docSizesIn = env->GetLongArrayElements(documentBufferSizes, nullptr);
    jlong* metaSizesIn = env->GetLongArrayElements(metadataBufferSizes, nullptr);
    jlong* embSizesIn = env->GetLongArrayElements(embeddingBufferSizes, nullptr);

    for (jsize i = 0; i < count; i++) {
        docArrays[i] = static_cast<jbyteArray>(env->GetObjectArrayElement(documentBuffers, i));
        docBufs[i] = reinterpret_cast<char*>(env->GetByteArrayElements(docArrays[i], nullptr));
        docSizes[i] = static_cast<size_t>(docSizesIn[i]);

        metaArrays[i] = static_cast<jbyteArray>(env->GetObjectArrayElement(metadataBuffers, i));
        metaBufs[i] = reinterpret_cast<char*>(env->GetByteArrayElements(metaArrays[i], nullptr));
        metaSizes[i] = static_cast<size_t>(metaSizesIn[i]);

        embArrays[i] = static_cast<jfloatArray>(env->GetObjectArrayElement(embeddingBuffers, i));
        embBufs[i] = env->GetFloatArrayElements(embArrays[i], nullptr);
        embSizes[i] = static_cast<size_t>(embSizesIn[i]);
    }

    int result = cactus_index_get(
        reinterpret_cast<cactus_index_t>(handle),
        reinterpret_cast<const int*>(idData),
        static_cast<size_t>(count),
        docBufs, docSizes, metaBufs, metaSizes, embBufs, embSizes
    );

    for (jsize i = 0; i < count; i++) {
        env->ReleaseByteArrayElements(docArrays[i], reinterpret_cast<jbyte*>(docBufs[i]), 0);
        env->DeleteLocalRef(docArrays[i]);
        env->ReleaseByteArrayElements(metaArrays[i], reinterpret_cast<jbyte*>(metaBufs[i]), 0);
        env->DeleteLocalRef(metaArrays[i]);
        env->ReleaseFloatArrayElements(embArrays[i], embBufs[i], 0);
        env->DeleteLocalRef(embArrays[i]);
        docSizesIn[i] = static_cast<jlong>(docSizes[i]);
        metaSizesIn[i] = static_cast<jlong>(metaSizes[i]);
        embSizesIn[i] = static_cast<jlong>(embSizes[i]);
    }

    env->ReleaseLongArrayElements(documentBufferSizes, docSizesIn, 0);
    env->ReleaseLongArrayElements(metadataBufferSizes, metaSizesIn, 0);
    env->ReleaseLongArrayElements(embeddingBufferSizes, embSizesIn, 0);
    env->ReleaseIntArrayElements(ids, idData, JNI_ABORT);

    delete[] docBufs;
    delete[] docSizes;
    delete[] metaBufs;
    delete[] metaSizes;
    delete[] embBufs;
    delete[] embSizes;
    delete[] docArrays;
    delete[] metaArrays;
    delete[] embArrays;

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeIndexQuery(JNIEnv* env, jobject, jlong handle,
                                            jobjectArray embeddings, jlong embeddingDim,
                                            jstring optionsJson,
                                            jobjectArray idBuffers, jlongArray idBufferSizes,
                                            jobjectArray scoreBuffers, jlongArray scoreBufferSizes) {
    jsize embCount = env->GetArrayLength(embeddings);
    const char* options = optionsJson ? env->GetStringUTFChars(optionsJson, nullptr) : nullptr;

    const float** embPtrs = new const float*[embCount];
    jfloatArray* embArrays = new jfloatArray[embCount];

    for (jsize i = 0; i < embCount; i++) {
        embArrays[i] = static_cast<jfloatArray>(env->GetObjectArrayElement(embeddings, i));
        embPtrs[i] = env->GetFloatArrayElements(embArrays[i], nullptr);
    }

    int** idPtrs = new int*[embCount];
    size_t* idSizes = new size_t[embCount];
    float** scorePtrs = new float*[embCount];
    size_t* scoreSizes = new size_t[embCount];

    jintArray* idJArrays = new jintArray[embCount];
    jfloatArray* scoreJArrays = new jfloatArray[embCount];

    jlong* idSizesIn = env->GetLongArrayElements(idBufferSizes, nullptr);
    jlong* scoreSizesIn = env->GetLongArrayElements(scoreBufferSizes, nullptr);

    for (jsize i = 0; i < embCount; i++) {
        idJArrays[i] = static_cast<jintArray>(env->GetObjectArrayElement(idBuffers, i));
        idPtrs[i] = env->GetIntArrayElements(idJArrays[i], nullptr);
        idSizes[i] = static_cast<size_t>(idSizesIn[i]);

        scoreJArrays[i] = static_cast<jfloatArray>(env->GetObjectArrayElement(scoreBuffers, i));
        scorePtrs[i] = env->GetFloatArrayElements(scoreJArrays[i], nullptr);
        scoreSizes[i] = static_cast<size_t>(scoreSizesIn[i]);
    }

    int result = cactus_index_query(
        reinterpret_cast<cactus_index_t>(handle),
        embPtrs, static_cast<size_t>(embCount), static_cast<size_t>(embeddingDim),
        options, idPtrs, idSizes, scorePtrs, scoreSizes
    );

    for (jsize i = 0; i < embCount; i++) {
        env->ReleaseFloatArrayElements(embArrays[i], const_cast<jfloat*>(embPtrs[i]), JNI_ABORT);
        env->DeleteLocalRef(embArrays[i]);
        env->ReleaseIntArrayElements(idJArrays[i], idPtrs[i], 0);
        env->DeleteLocalRef(idJArrays[i]);
        env->ReleaseFloatArrayElements(scoreJArrays[i], scorePtrs[i], 0);
        env->DeleteLocalRef(scoreJArrays[i]);
        idSizesIn[i] = static_cast<jlong>(idSizes[i]);
        scoreSizesIn[i] = static_cast<jlong>(scoreSizes[i]);
    }

    env->ReleaseLongArrayElements(idBufferSizes, idSizesIn, 0);
    env->ReleaseLongArrayElements(scoreBufferSizes, scoreSizesIn, 0);
    if (options) env->ReleaseStringUTFChars(optionsJson, options);

    delete[] embPtrs;
    delete[] embArrays;
    delete[] idPtrs;
    delete[] idSizes;
    delete[] scorePtrs;
    delete[] scoreSizes;
    delete[] idJArrays;
    delete[] scoreJArrays;

    return result;
}

JNIEXPORT jint JNICALL
Java_com_cactus_CactusJNI_nativeIndexCompact(JNIEnv*, jobject, jlong handle) {
    return cactus_index_compact(reinterpret_cast<cactus_index_t>(handle));
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeIndexDestroy(JNIEnv*, jobject, jlong handle) {
    cactus_index_destroy(reinterpret_cast<cactus_index_t>(handle));
}

JNIEXPORT jstring JNICALL
Java_com_cactus_CactusJNI_nativeGetLastError(JNIEnv* env, jobject) {
    return env->NewStringUTF(cactus_get_last_error());
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeLogSetLevel(JNIEnv*, jobject, jint level) {
    cactus_log_set_level(static_cast<int>(level));
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeLogSetCallback(JNIEnv* env, jobject, jobject callback) {
    if (g_log_callback_ctx) {
        env->DeleteGlobalRef(g_log_callback_ctx->callback);
        delete g_log_callback_ctx;
        g_log_callback_ctx = nullptr;
    }
    if (!callback) {
        cactus_log_set_callback(nullptr, nullptr);
        return;
    }
    JavaVM* jvm = nullptr;
    env->GetJavaVM(&jvm);
    jclass cls = env->GetObjectClass(callback);
    jmethodID method = env->GetMethodID(cls, "onLog", "(ILjava/lang/String;Ljava/lang/String;)V");
    g_log_callback_ctx = new LogCallbackContext{jvm, env->NewGlobalRef(callback), method};
    cactus_log_set_callback(log_callback_bridge, g_log_callback_ctx);
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeSetTelemetryEnvironment(JNIEnv* env, jobject,
                                                          jstring framework, jstring cacheLocation, jstring version) {
    const char* fw = framework ? env->GetStringUTFChars(framework, nullptr) : nullptr;
    const char* cache = cacheLocation ? env->GetStringUTFChars(cacheLocation, nullptr) : nullptr;
    const char* ver = version ? env->GetStringUTFChars(version, nullptr) : nullptr;
    cactus_set_telemetry_environment(fw, cache, ver);
    if (fw) env->ReleaseStringUTFChars(framework, fw);
    if (cache) env->ReleaseStringUTFChars(cacheLocation, cache);
    if (ver) env->ReleaseStringUTFChars(version, ver);
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeSetAppId(JNIEnv* env, jobject, jstring appId) {
    const char* id = env->GetStringUTFChars(appId, nullptr);
    cactus_set_app_id(id);
    env->ReleaseStringUTFChars(appId, id);
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeTelemetryFlush(JNIEnv*, jobject) {
    cactus_telemetry_flush();
}

JNIEXPORT void JNICALL
Java_com_cactus_CactusJNI_nativeTelemetryShutdown(JNIEnv*, jobject) {
    cactus_telemetry_shutdown();
}

}
