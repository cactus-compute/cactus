#ifndef CACTUS_FFI_NET_H
#define CACTUS_FFI_NET_H

#include <string>

namespace cactus {
namespace net {

struct HttpResponse {
    int status_code;
    std::string body;
    std::string error;
    bool success;
};

struct HttpRequest {
    std::string url;
    std::string body;
    std::string content_type;
    std::string authorization;
    int timeout_ms;

    HttpRequest() : content_type("application/json"), timeout_ms(30000) {}
};

HttpResponse http_post(const HttpRequest& request);
bool is_network_available();

} // namespace net
} // namespace cactus

#endif // CACTUS_FFI_NET_H
