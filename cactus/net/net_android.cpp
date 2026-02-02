// Android HTTP implementation via JNI
// Calls into Java's HttpURLConnection for actual HTTP requests

#ifdef __ANDROID__

#include "net.h"
#include <jni.h>
#include <android/log.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <thread>

#define LOG_TAG "CactusNet"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Ensure JNI functions are visible despite -fvisibility=hidden
#define JNI_VISIBLE __attribute__((visibility("default")))

namespace cactus {
namespace net {

// Global state
static JavaVM* g_jvm = nullptr;
static CloudConfig g_cloud_config;
static std::mutex g_mutex;

// Cached JNI references (initialized in JNI_OnLoad)
static jclass g_http_class = nullptr;        // GlobalRef to CactusHttp class
static jmethodID g_post_method = nullptr;    // Cached method ID for post()

// Request context for synchronous HTTP calls
struct HttpRequestContext {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> completed{false};
    HttpResponse response;
};

// Store pending request contexts (keyed by request ID)
static std::mutex g_requests_mutex;
static std::atomic<int64_t> g_next_request_id{1};
static std::unordered_map<int64_t, HttpRequestContext*> g_pending_requests;

// RAII guard for JNI thread attachment
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

    // Non-copyable
    JNIThreadGuard(const JNIThreadGuard&) = delete;
    JNIThreadGuard& operator=(const JNIThreadGuard&) = delete;

private:
    JNIEnv* env_;
    bool needs_detach_;
};

// Helper to check and clear JNI exceptions
static bool check_exception(JNIEnv* env, const char* context) {
    if (env->ExceptionCheck()) {
        LOGE("JNI exception in %s", context);
        env->ExceptionDescribe();
        env->ExceptionClear();
        return true;
    }
    return false;
}

// Called by Java with HTTP response
extern "C" JNI_VISIBLE JNIEXPORT void JNICALL
Java_com_cactus_CactusHttp_nativeOnResponse(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong requestId,
    jint statusCode,
    jstring body,
    jstring error
) {
    std::lock_guard<std::mutex> lock(g_requests_mutex);

    auto it = g_pending_requests.find(requestId);
    if (it == g_pending_requests.end()) {
        LOGE("Unknown request ID: %lld", static_cast<long long>(requestId));
        return;
    }

    HttpRequestContext* ctx = it->second;

    // Fill response
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

    // Signal completion
    {
        std::lock_guard<std::mutex> ctx_lock(ctx->mutex);
        ctx->completed = true;
    }
    ctx->cv.notify_one();
}

// Initialize JNI - cache class and method IDs
extern "C" JNI_VISIBLE JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_jvm = vm;

    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        LOGE("Failed to get JNI environment in JNI_OnLoad");
        return JNI_VERSION_1_6;
    }

    // Find and cache CactusHttp class as GlobalRef
    jclass localClass = env->FindClass("com/cactus/CactusHttp");
    if (localClass == nullptr) {
        env->ExceptionClear();
        LOGE("CactusHttp class not found during JNI_OnLoad - will retry on first use");
    } else {
        g_http_class = static_cast<jclass>(env->NewGlobalRef(localClass));
        env->DeleteLocalRef(localClass);

        // Cache method ID
        g_post_method = env->GetStaticMethodID(
            g_http_class,
            "post",
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

// Cleanup JNI resources
extern "C" JNI_VISIBLE JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM* vm, void* /*reserved*/) {
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

// Lazy initialization of JNI references (fallback if JNI_OnLoad didn't find the class)
static bool ensure_jni_initialized(JNIEnv* env) {
    if (g_http_class != nullptr && g_post_method != nullptr) {
        return true;
    }

    // Try to find the class now
    if (g_http_class == nullptr) {
        jclass localClass = env->FindClass("com/cactus/CactusHttp");
        if (localClass == nullptr) {
            check_exception(env, "FindClass(CactusHttp)");
            return false;
        }
        g_http_class = static_cast<jclass>(env->NewGlobalRef(localClass));
        env->DeleteLocalRef(localClass);
    }

    // Cache method ID
    if (g_post_method == nullptr) {
        g_post_method = env->GetStaticMethodID(
            g_http_class,
            "post",
            "(JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I)V"
        );
        if (g_post_method == nullptr) {
            check_exception(env, "GetStaticMethodID(post)");
            return false;
        }
    }

    return true;
}

// RAII wrapper for JNI local references to prevent leaks
class JNILocalRef {
public:
    JNILocalRef(JNIEnv* env, jstring str) : env_(env), str_(str) {}
    ~JNILocalRef() { if (str_) env_->DeleteLocalRef(str_); }
    jstring get() const { return str_; }
    operator jstring() const { return str_; }
    // Non-copyable
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

    // Use RAII guard for thread attachment
    JNIThreadGuard guard;
    if (!guard.valid()) {
        response.error = "JNI environment not available";
        return response;
    }
    JNIEnv* env = guard.env();

    // Ensure class and method are cached
    if (!ensure_jni_initialized(env)) {
        response.error = "CactusHttp class not found. Ensure CactusHttp.java is included in your app.";
        return response;
    }

    // Create request context
    int64_t requestId = g_next_request_id.fetch_add(1);
    HttpRequestContext ctx;

    {
        std::lock_guard<std::mutex> lock(g_requests_mutex);
        g_pending_requests[requestId] = &ctx;
    }

    // Prepare Java strings with RAII wrappers (auto-cleanup on any exit path)
    // NOTE: NewStringUTF uses Modified UTF-8, which cannot represent some Unicode
    // characters correctly (e.g., emoji with code points > U+FFFF, encoded as surrogate
    // pairs in true UTF-8). For cloud API requests, this is generally fine since URLs
    // and API keys are ASCII, and request bodies are typically ASCII-safe JSON.
    // If emoji support in prompts becomes critical, consider using NewString with
    // explicit UTF-16 conversion instead.
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

    // Call Java method (this triggers async HTTP, response comes via callback)
    env->CallStaticVoidMethod(
        g_http_class,
        g_post_method,
        static_cast<jlong>(requestId),
        jUrl.get(),
        jBody.get(),
        jContentType.get(),
        jAuth.get(),
        static_cast<jint>(request.timeout_ms)
    );

    // Check for exceptions (local refs cleaned up by RAII destructors)
    if (check_exception(env, "CallStaticVoidMethod(post)")) {
        std::lock_guard<std::mutex> lock(g_requests_mutex);
        g_pending_requests.erase(requestId);
        response.error = "Java exception during HTTP request";
        return response;
    }

    // Wait for response with timeout
    {
        std::unique_lock<std::mutex> lock(ctx.mutex);
        bool completed = ctx.cv.wait_for(
            lock,
            std::chrono::milliseconds(request.timeout_ms + 5000), // Extra buffer for Java overhead
            [&ctx] { return ctx.completed.load(); }
        );

        if (!completed) {
            response.error = "HTTP request timed out";
            response.status_code = -1;
        } else {
            response = ctx.response;
        }
    }

    // Remove from pending requests
    {
        std::lock_guard<std::mutex> lock(g_requests_mutex);
        g_pending_requests.erase(requestId);
    }

    return response;
}

bool is_network_available() {
    // On Android, network is generally available if JNI is set up
    return g_jvm != nullptr;
}

void set_cloud_config(const CloudConfig& config) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_cloud_config = config;
}

CloudConfig get_cloud_config() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_cloud_config;  // Return copy, not reference - safe across threads
}

bool is_cloud_configured() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_cloud_config.is_configured();
}

// Format messages for OpenAI-compatible API
static std::string format_openai_request(const std::string& messages_json,
                                          const std::string& model,
                                          const std::string& /*options_json*/) {
    // Construct OpenAI-compatible request
    // The messages_json is already in a compatible format from cactus
    std::string request = "{\"model\":\"" + model + "\",\"messages\":" + messages_json + "}";
    return request;
}

// Format messages for Anthropic API
static std::string format_anthropic_request(const std::string& messages_json,
                                             const std::string& model,
                                             const std::string& /*options_json*/) {
    // Anthropic uses a different format with "content" blocks
    // For simplicity, we'll use their messages API which is similar to OpenAI
    std::string request = "{\"model\":\"" + model + "\",\"max_tokens\":1024,\"messages\":" + messages_json + "}";
    return request;
}

// HTTP POST with retry logic for transient errors
static HttpResponse http_post_with_retry(const HttpRequest& request, int max_retries = 3) {
    HttpResponse response;
    int retry_delay_ms = 1000; // Start with 1 second

    for (int attempt = 0; attempt <= max_retries; attempt++) {
        response = http_post(request);

        // Success or client error (4xx) - don't retry
        if (response.success || (response.status_code >= 400 && response.status_code < 500)) {
            return response;
        }

        // Server error (5xx) or network error (-1) - retry with backoff
        if (attempt < max_retries) {
            LOGI("HTTP request failed (attempt %d/%d), retrying in %dms: %s",
                 attempt + 1, max_retries + 1, retry_delay_ms, response.error.c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
            retry_delay_ms *= 2; // Exponential backoff
        }
    }

    return response;
}

HttpResponse cloud_complete(const std::string& messages_json,
                            const std::string& options_json) {
    HttpResponse response;

    CloudConfig config;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        config = g_cloud_config;
    }

    if (!config.is_configured()) {
        response.status_code = -1;
        response.success = false;
        response.error = "Cloud not configured. Call set_cloud_config() first.";
        return response;
    }

    HttpRequest request;
    request.timeout_ms = 60000; // 60 second timeout for LLM requests

    // Configure based on provider
    if (config.provider == "openai") {
        request.url = config.endpoint_url.empty()
            ? "https://api.openai.com/v1/chat/completions"
            : config.endpoint_url;
        request.authorization = "Bearer " + config.api_key;
        request.body = format_openai_request(
            messages_json,
            config.model.empty() ? "gpt-4" : config.model,
            options_json
        );
    } else if (config.provider == "anthropic") {
        request.url = config.endpoint_url.empty()
            ? "https://api.anthropic.com/v1/messages"
            : config.endpoint_url;
        request.authorization = config.api_key; // Anthropic uses x-api-key header (handled in Java)
        request.body = format_anthropic_request(
            messages_json,
            config.model.empty() ? "claude-3-opus-20240229" : config.model,
            options_json
        );
    } else if (config.provider == "gcp") {
        // Google Cloud Vertex AI
        request.url = config.endpoint_url; // Must be provided for GCP
        request.authorization = "Bearer " + config.api_key;
        request.body = format_openai_request(
            messages_json,
            config.model.empty() ? "gemini-pro" : config.model,
            options_json
        );
    } else if (config.provider == "openrouter") {
        // OpenRouter - aggregates multiple providers
        request.url = config.endpoint_url.empty()
            ? "https://openrouter.ai/api/v1/chat/completions"
            : config.endpoint_url;
        request.authorization = "Bearer " + config.api_key;
        request.body = format_openai_request(
            messages_json,
            config.model.empty() ? "openai/gpt-3.5-turbo" : config.model,
            options_json
        );
    } else if (config.provider == "custom") {
        // Custom endpoint - user provides everything
        if (config.endpoint_url.empty()) {
            response.status_code = -1;
            response.success = false;
            response.error = "Custom provider requires endpoint_url";
            return response;
        }
        request.url = config.endpoint_url;
        request.authorization = "Bearer " + config.api_key;
        // Assume OpenAI-compatible format
        request.body = format_openai_request(
            messages_json,
            config.model,
            options_json
        );
    } else {
        response.status_code = -1;
        response.success = false;
        response.error = "Unknown cloud provider: " + config.provider;
        return response;
    }

    // Security: Enforce HTTPS for API requests with credentials
    if (!request.authorization.empty() &&
        request.url.substr(0, 8) != "https://") {
        response.status_code = -1;
        response.success = false;
        response.error = "Security error: API requests with credentials require HTTPS";
        LOGE("Refusing to send API key over non-HTTPS connection: %s",
             request.url.substr(0, 30).c_str());
        return response;
    }

    // Use retry logic for transient failures
    return http_post_with_retry(request);
}

} // namespace net
} // namespace cactus

#endif // __ANDROID__
