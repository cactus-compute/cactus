#ifndef __ANDROID__

#include "cactus_net.h"

namespace cactus {
namespace net {

HttpResponse http_post(const HttpRequest&) {
    HttpResponse response;
    response.status_code = -1;
    response.success = false;
    response.error = "HTTP not available on this platform";
    return response;
}

bool is_network_available() {
    return false;
}

} // namespace net
} // namespace cactus

#endif
