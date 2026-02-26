#ifndef CACTUS_CLOUD_H
#define CACTUS_CLOUD_H

#include "cactus_utils.h"
#include <string>
#include <vector>

namespace cactus {
namespace ffi {

struct CloudResponse {
    std::string transcript;
    std::string api_key_hash;
    bool used_cloud = false;
    std::string error;
};

struct CloudCompletionRequest {
    std::vector<cactus::engine::ChatMessage> messages;
    std::vector<ToolFunction> tools;
    std::string local_output;
    std::vector<std::string> local_function_calls;
    bool has_images = false;
    std::string cloud_key;
};

struct CloudCompletionResult {
    bool ok = false;
    bool used_cloud = false;
    std::string response;
    std::vector<std::string> function_calls;
    std::string error;
};

std::string cloud_base64_encode(const uint8_t* data, size_t len);
std::vector<uint8_t> cloud_build_wav(const uint8_t* pcm, size_t pcm_bytes);
std::string resolve_cloud_api_key(const char* cloud_key_param);
CloudResponse cloud_transcribe_request(const std::string& audio_b64,
                                       const std::string& fallback_text,
                                       long timeout_seconds = 15L,
                                       const char* cloud_key = nullptr);
CloudCompletionResult cloud_complete_request(const CloudCompletionRequest& request,
                                             long timeout_ms);

} // namespace ffi
} // namespace cactus

#endif // CACTUS_CLOUD_H
