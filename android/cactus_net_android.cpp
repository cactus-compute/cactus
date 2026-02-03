#ifdef __ANDROID__

#include "cactus_net.h"
#include <jni.h>
#include <android/log.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>

#define LOG_TAG "CactusNet"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define JNI_VISIBLE __attribute__((visibility("default")))

namespace cactus {
namespace net {

static JavaVM* g_jvm = nullptr;
static std::mutex g_mutex;
static jclass g_http_class = nullptr;
static jmethodID g_post_method = nullptr;

struct HttpRequestContext {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> completed{false};
    HttpResponse response;
};

static std::mutex g_requests_mutex;
static std::atomic<int64_t> g_next_request_id{1};
static std::unordered_map<int64_t, HttpRequestContext*> g_pending_requests;

class JNIThreadGuard {
public:
    JNIThreadGuard() : env_(nullptr), needs_detach_(false) {
        if (g_jvm == nullptr) return;
        int status = g_jvm->GetEnv(reinterpret_cast<void**>(&env_), JNI_VERSION_1_6);
        if (status == JNI_EDETACHED) {
            if (g_jvm->AttachCurrentThread(&env_, nullptr) == JNI_OK) {
                needs_detach_ = true;
            } else {
                env_ = nullptr;
            }
        } else if (status != JNI_OK) {
            env_ = nullptr;
        }
    }

    ~JNIThreadGuard() {
        if (needs_detach_ && g_jvm != nullptr) {
            g_jvm->DetachCurrentThread();
        }
    }

    JNIEnv* env() const { return env_; }
    bool valid() const { return env_ != nullptr; }
    JNIThreadGuard(const JNIThreadGuard&) = delete;
    JNIThreadGuard& operator=(const JNIThreadGuard&) = delete;

private:
    JNIEnv* env_;
    bool needs_detach_;
};

static bool check_exception(JNIEnv* env, const char* context) {
    if (env->ExceptionCheck()) {
        LOGE("JNI exception in %s", context);
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

extern "C" JNI_VISIBLE JNIEXPORT void JNICALL
Java_com_cactus_CactusHttp_nativeOnResponse(
    JNIEnv* env, jclass, jlong requestId, jint statusCode, jstring body, jstring error
) {
    std::lock_guard<std::mutex> lock(g_requests_mutex);
    auto it = g_pending_requests.find(requestId);
    if (it == g_pending_requests.end()) {
        LOGE("Unknown request ID: %lld", static_cast<long long>(requestId));
        return;
    }

    HttpRequestContext* ctx = it->second;
    ctx->response.status_code = statusCode;
    ctx->response.success = (statusCode >= 200 && statusCode < 300);

    if (body != nullptr) {
        const char* bodyStr = env->GetStringUTFChars(body, nullptr);
        if (bodyStr) {
            ctx->response.body = bodyStr;
            env->ReleaseStringUTFChars(body, bodyStr);
        }
    }

    if (error != nullptr) {
        const char* errorStr = env->GetStringUTFChars(error, nullptr);
        if (errorStr) {
            ctx->response.error = errorStr;
            env->ReleaseStringUTFChars(error, errorStr);
        }
    }

    {
        std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
        ctx->completed = true;
    }
    ctx->cv.notify_one();
}

extern "C" JNI_VISIBLE JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void*) {
    g_jvm = vm;
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        LOGE("Failed to get JNI environment in JNI_OnLoad");
        return JNI_VERSION_1_6;
    }

    jclass localClass = env->FindClass("com/cactus/CactusHttp");
    if (localClass == nullptr) {
        env->ExceptionClear();
        LOGE("CactusHttp class not found during JNI_OnLoad");
    } else {
        g_http_class = static_cast<jclass>(env->NewGlobalRef(localClass));
        env->DeleteLocalRef(localClass);
        g_post_method = env->GetStaticMethodID(
            g_http_class, "post",
            "(JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I)V"
        );
        if (g_post_method == nullptr) {
            env->ExceptionClear();
            LOGE("CactusHttp.post method not found during JNI_OnLoad");
        }
    }

    LOGI("CactusNet JNI initialized");
    return JNI_VERSION_1_6;
}

extern "C" JNI_VISIBLE JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
        if (g_http_class != nullptr) {
            env->DeleteGlobalRef(g_http_class);
            g_http_class = nullptr;
        }
    }
    g_post_method = nullptr;
    g_jvm = nullptr;
    LOGI("CactusNet JNI unloaded");
}

static bool ensure_jni_initialized(JNIEnv* env) {
    if (g_http_class != nullptr && g_post_method != nullptr) return true;

    if (g_http_class == nullptr) {
        jclass localClass = env->FindClass("com/cactus/CactusHttp");
        if (localClass == nullptr) {
            check_exception(env, "FindClass(CactusHttp)");
            return false;
        }
        g_http_class = static_cast<jclass>(env->NewGlobalRef(localClass));
        env->DeleteLocalRef(localClass);
    }

    if (g_post_method == nullptr) {
        g_post_method = env->GetStaticMethodID(
            g_http_class, "post",
            "(JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I)V"
        );
        if (g_post_method == nullptr) {
            check_exception(env, "GetStaticMethodID(post)");
            return false;
        }
    }
    return true;
}

class JNILocalRef {
public:
    JNILocalRef(JNIEnv* env, jstring str) : env_(env), str_(str) {}
    ~JNILocalRef() { if (str_) env_->DeleteLocalRef(str_); }
    jstring get() const { return str_; }
    JNILocalRef(const JNILocalRef&) = delete;
    JNILocalRef& operator=(const JNILocalRef&) = delete;
private:
    JNIEnv* env_;
    jstring str_;
};

HttpResponse http_post(const HttpRequest& request) {
    HttpResponse response;
    response.status_code = -1;
    response.success = false;

    JNIThreadGuard guard;
    if (!guard.valid()) {
        response.error = "JNI environment not available";
        return response;
    }
    JNIEnv* env = guard.env();

    if (!ensure_jni_initialized(env)) {
        response.error = "CactusHttp class not found";
        return response;
    }

    int64_t requestId = g_next_request_id.fetch_add(1);
    HttpRequestContext ctx;

    {
        std::lock_guard<std::mutex> lock(g_requests_mutex);
        g_pending_requests[requestId] = &ctx;
    }

    JNILocalRef jUrl(env, env->NewStringUTF(request.url.c_str()));
    if (check_exception(env, "NewStringUTF(url)") || !jUrl.get()) {
        std::lock_guard<std::mutex> lock(g_requests_mutex);
        g_pending_requests.erase(requestId);
        response.error = "Failed to create URL string";
        return response;
    }

    JNILocalRef jBody(env, env->NewStringUTF(request.body.c_str()));
    JNILocalRef jContentType(env, env->NewStringUTF(request.content_type.c_str()));
    JNILocalRef jAuth(env, request.authorization.empty() ? nullptr : env->NewStringUTF(request.authorization.c_str()));

    env->CallStaticVoidMethod(
        g_http_class, g_post_method,
        static_cast<jlong>(requestId), jUrl.get(), jBody.get(),
        jContentType.get(), jAuth.get(), static_cast<jint>(request.timeout_ms)
    );

    if (check_exception(env, "CallStaticVoidMethod(post)")) {
        std::lock_guard<std::mutex> lock(g_requests_mutex);
        g_pending_requests.erase(requestId);
        response.error = "Java exception during HTTP request";
        return response;
    }

    {
        std::unique_lock<std::mutex> lock(ctx.mutex);
        bool completed = ctx.cv.wait_for(
            lock, std::chrono::milliseconds(request.timeout_ms + 5000),
            [&ctx] { return ctx.completed.load(); }
        );
        if (!completed) {
            response.error = "HTTP request timed out";
            response.status_code = -1;
        } else {
            response = ctx.response;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_requests_mutex);
        g_pending_requests.erase(requestId);
    }

    return response;
}

bool is_network_available() {
    return g_jvm != nullptr;
}

} // namespace net
} // namespace cactus

#endif // __ANDROID__
