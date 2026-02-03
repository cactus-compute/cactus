#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "cactus_net.h"
#include <cstring>

using namespace cactus::net;

extern "C" {

bool cactus_is_network_available(void) {
    return is_network_available();
}

int cactus_test_http(const char* url, char* response_buffer, size_t buffer_size) {
    if (!url || !response_buffer || buffer_size == 0) {
        return -1;
    }

    HttpRequest request;
    request.url = url;
    request.timeout_ms = 10000;

    HttpResponse response = http_post(request);

    std::string result;
    if (response.success) {
        result = response.body.empty() ? "{\"status\":\"ok\"}" : response.body;
    } else {
        result = "{\"error\":\"" + cactus::ffi::escape_json_string(response.error) + "\",\"status_code\":" +
                 std::to_string(response.status_code) + "}";
    }

    if (result.length() >= buffer_size) {
        result = result.substr(0, buffer_size - 1);
    }
    std::strcpy(response_buffer, result.c_str());

    return response.status_code;
}

} // extern "C"
